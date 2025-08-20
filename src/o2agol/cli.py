import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

# Note: We use Optional[T] instead of T | None (PEP 604) throughout this file
# because Typer 0.12.3 doesn't support the newer union syntax in Annotated types.
# The UP045 ruff rule is disabled in pyproject.toml to prevent automatic conversion.
import yaml
from dotenv import load_dotenv

from .cleanup import full_cleanup_check, register_cleanup_handlers
from .config import Config
from .config.settings import PipelineLogger, LogContext
from .config.template_config import TemplateConfigParser
from .duck import fetch_gdf
from .dump_manager import DumpManager, DumpConfig
from .publish import publish_or_update
from .transform import normalize_schema

load_dotenv()

app = typer.Typer(help="Overture to AGOL pipeline")


def setup_logging(verbose: bool, target_name: Optional[str] = None, mode: Optional[str] = None, enable_file_logging: bool = False):
    """
    Configure logging with optional timestamped file output for production operations.
    
    Args:
        verbose: Enable debug-level logging if True
        target_name: Target data type for log file naming
        mode: Operation mode for log file naming  
        enable_file_logging: Create timestamped log files when True
    """
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if enable_file_logging and target_name and mode:
        # Create logs directory for timestamped files
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Generate timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{target_name}_{mode}_{timestamp}.log"
        
        # Add file handler for persistent logging
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
        
        # Log file location for reference
        print(f"Logging to: {log_file}")
    
    logging.basicConfig(
        level=level, 
        format="%(asctime)s [%(levelname)s] %(message)s", 
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True
    )


def is_template_config(config_path: str) -> bool:
    """
    Detect if config file uses the template format with dynamic variables.
    
    Template configs have a 'templates' section and either a 'country' section 
    or are designed to work with dynamic country injection.
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Template config if it has templates section
        # Country section is optional (for global configs with dynamic injection)
        return 'templates' in config
    except Exception:
        return False


def load_pipeline_config(config_path: str, country_override: str = None) -> dict[str, Any]:
    """
    Load pipeline configuration from YAML file and combine with secure credentials.
    Supports both legacy and enhanced config formats with templating, plus global configs with country override.
    
    Args:
        config_path: Path to YAML configuration file
        country_override: Optional country code/name for global config mode
        
    Returns:
        Dict containing complete pipeline configuration
        
    Raises:
        FileNotFoundError: If configuration file does not exist
        ConfigurationError: If secure configuration is invalid
        ValueError: If required YAML fields are missing or country not found
    """
    # Load secure configuration (credentials and DuckDB settings only)
    secure_config = Config()
    gis_connection = secure_config.create_gis_connection()
    
    # Check config format and load appropriately
    if is_template_config(config_path):
        # Template config with dynamic variables (and optional country override)
        template_parser = TemplateConfigParser(config_path, country_override)
        
        # Validate template config
        validation_issues = template_parser.validate_config()
        if validation_issues:
            logging.warning(f"Template config validation issues: {validation_issues}")
        
        # Get overture config from template parser
        overture_config = template_parser.get_overture_config()
        overture_release = overture_config.get('release')
        s3_region = overture_config.get('s3_region', 'us-west-2')
        
        # Build base URL from template config
        base_url = overture_config.get('base_url', f's3://overturemaps-{s3_region}/release')
        if not base_url.endswith(overture_release):
            base_url = f"{base_url}/{overture_release}"
        
        pipeline_config = {
            'secure': secure_config,
            'gis': gis_connection,
            'duckdb_settings': secure_config.get_duckdb_settings(),
            'overture': {
                'release': overture_release,
                's3_region': s3_region,
                'base_url': base_url
            },
            'yaml': template_parser.raw_config,  # For backward compatibility
            'template': template_parser,  # Template parser for metadata
            'environment': secure_config.environment,
            'config_format': 'template'
        }
        
        logging.info(f"Loaded template config: {template_parser.country.name} ({template_parser.country.iso3})")
        if template_parser.is_using_global_config():
            logging.info(f"Global config mode: Country data from {template_parser.get_country_source()}")
        
    else:
        # Legacy config format
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file) as f:
            yaml_config = yaml.safe_load(f)
        
        # Extract required Overture settings from YAML
        overture_release = yaml_config.get('overture_release')
        if not overture_release:
            raise ValueError("overture_release is required in YAML configuration")
        
        overture_s3 = yaml_config.get('overture_s3', {})
        s3_region = overture_s3.get('region', 'us-west-2')
        s3_bucket = overture_s3.get('bucket', f'overturemaps-{s3_region}')
        
        pipeline_config = {
            'secure': secure_config,
            'gis': gis_connection,
            'duckdb_settings': secure_config.get_duckdb_settings(),
            'overture': {
                'release': overture_release,        # From YAML
                's3_region': s3_region,            # From YAML  
                'bucket': s3_bucket,               # From YAML
                'base_url': f"s3://{s3_bucket}/release/{overture_release}"
            },
            'yaml': yaml_config,
            'environment': secure_config.environment,
            'config_format': 'legacy'
        }
        
        logging.info(f"Loaded legacy config: {overture_release}")
    
    return pipeline_config


def get_available_queries(config_path: str) -> list[str]:
    """
    Discover all available queries from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        List of available query names
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        targets = config.get('targets', {})
        if not targets:
            raise ValueError(f"No targets found in configuration file: {config_path}")
        
        return list(targets.keys())
    
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file {config_path}: {e}")


def validate_query_exists(config_path: str, query_name: str) -> bool:
    """
    Validate that a query exists in the configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        query_name: Name of query to validate
        
    Returns:
        True if query exists, False otherwise
    """
    try:
        available_queries = get_available_queries(config_path)
        return query_name in available_queries
    except (FileNotFoundError, ValueError):
        return False


def get_query_info(config_path: str, query_name: str) -> dict[str, Any]:
    """
    Get detailed information about a specific query.
    
    Args:
        config_path: Path to YAML configuration file
        query_name: Name of query to get info for
        
    Returns:
        Dictionary with query information
        
    Raises:
        KeyError: If query not found
        ValueError: If config is invalid
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    targets = config.get('targets', {})
    if query_name not in targets:
        available = ', '.join(targets.keys())
        raise KeyError(f"Query '{query_name}' not found. Available queries: {available}")
    
    return targets[query_name]


def export_to_geojson(data, output_path: str, target_name: str) -> None:
    """Export transformed features to GeoJSON file.
    
    Args:
        data: GeoDataFrame or dict of GeoDataFrames (for multi-layer)
        output_path: Path where GeoJSON file will be saved
        target_name: Target data type for metadata
        
    Raises:
        IOError: If file cannot be written
        ValueError: If features are not valid
    """
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Handle both single GeoDataFrame and multi-layer dict
    if isinstance(data, dict):
        # Multi-layer: combine all features
        all_features = []
        layer_counts = {}
        
        for layer_name, gdf in data.items():
            layer_features = gdf.to_json()
            layer_data = json.loads(layer_features)
            
            # Add layer identifier to each feature
            for feature in layer_data.get("features", []):
                feature["properties"]["layer"] = layer_name
            
            all_features.extend(layer_data.get("features", []))
            layer_counts[layer_name] = len(layer_data.get("features", []))
        
        # Create combined GeoJSON
        geojson_data = {
            "type": "FeatureCollection",
            "features": all_features,
            "metadata": {
                "generated": datetime.utcnow().isoformat(),
                "source": "overture-agol-pipeline",
                "target": target_name,
                "layers": layer_counts,
                "total_count": len(all_features)
            }
        }
    else:
        # Single GeoDataFrame
        features_json = data.to_json()
        features_data = json.loads(features_json)
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features_data.get("features", []),
            "metadata": {
                "generated": datetime.utcnow().isoformat(),
                "source": "overture-agol-pipeline", 
                "target": target_name,
                "count": len(features_data.get("features", []))
            }
        }
    
    # Write with proper encoding and formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f, indent=2, ensure_ascii=False)
    
    # Validate written file
    if not validate_geojson_file(output_path):
        raise ValueError(f"Generated GeoJSON file is invalid: {output_path}")
    
    logging.info(f"GeoJSON export completed: {len(geojson_data['features']):,} features written to {output_path}")


def validate_geojson_file(filepath: str) -> bool:
    """Validate that exported GeoJSON file is properly formatted."""
    try:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic GeoJSON structure validation
        if data.get("type") != "FeatureCollection":
            logging.error("Invalid GeoJSON: missing FeatureCollection type")
            return False
        
        if "features" not in data:
            logging.error("Invalid GeoJSON: missing features array")
            return False
        
        if not isinstance(data["features"], list):
            logging.error("Invalid GeoJSON: features must be an array")
            return False
        
        return True
    except Exception as e:
        logging.error(f"GeoJSON validation failed: {e}")
        return False


def get_selector_config(cfg: dict[str, Any], iso2: Optional[str] = None) -> dict[str, Any]:
    """
    Extract or create selector configuration for spatial filtering.
    
    Args:
        cfg: Pipeline configuration dictionary
        iso2: Optional ISO2 country code override
        
    Returns:
        Selector configuration for spatial filtering
    """
    if cfg.get('config_format') == 'template':
        # Template config with resolved variables
        selector = cfg['template'].get_selector_config()
    else:
        # Legacy config
        selector = cfg['yaml'].get('selector', {})
    
    # Apply ISO2 override if specified
    if iso2:
        selector['iso2'] = iso2.upper()
        selector['by'] = selector.get('by', 'division_country')
    
    return selector


def get_target_config(cfg: dict[str, Any], target_name: str) -> dict[str, Any]:
    """
    Extract target configuration for specified data type.
    
    Args:
        cfg: Pipeline configuration dictionary
        target_name: Target data type (roads, buildings, places)
        
    Returns:
        Target configuration dictionary with all required fields
        
    Raises:
        KeyError: If target not found in configuration
        ValueError: If required target fields are missing
    """
    if cfg.get('config_format') == 'template':
        # Template config - get filter config from template parser
        template = cfg['template']
        filter_config = template.get_target_filter_config(target_name)
        
        # Combine with raw target config for backward compatibility
        targets = cfg['yaml'].get('targets', {})
        if target_name not in targets:
            available = list(targets.keys())
            raise KeyError(f"Target '{target_name}' not found. Available: {available}")
        
        target_config = targets[target_name].copy()
        target_config.update(filter_config)  # Overlay enhanced filter config
        
    else:
        # Legacy config
        targets = cfg['yaml'].get('targets', {})
        if target_name not in targets:
            raise KeyError(f"Target '{target_name}' not found in configuration. Available: {list(targets.keys())}")
        
        target_config = targets[target_name].copy()
    
    # Ensure required fields are present
    if 'theme' not in target_config:
        raise ValueError(f"Target '{target_name}' missing required 'theme' field")
    if 'type' not in target_config:
        raise ValueError(f"Target '{target_name}' missing required 'type' field")
    
    logging.debug(f"Target config for {target_name}: theme={target_config['theme']}, type={target_config['type']}")
    
    return target_config


def process_target(
    target_name: str,
    config: str,
    output_mode: str,
    mode: Optional[str] = None,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    iso2: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
    use_divisions: bool = True,
    log_to_file: bool = False,
    country: Optional[str] = None,
    skip_cleanup: bool = False,
):
    """
    Execute data processing pipeline for specified target with comprehensive logging.
    
    Supports both bounding box and Overture Divisions-based spatial filtering
    for precise country boundary adherence in enterprise deployments.
    
    Args:
        target_name: Data target identifier (roads, buildings, etc.)
        config: Path to YAML configuration file
        output_mode: Output mode - "agol" for ArcGIS Online, "geojson" for file export
        mode: Processing mode (initial, overwrite, append) - only used for "agol" mode
        output_path: GeoJSON file path - only used for "geojson" mode
        limit: Optional feature limit for testing and development
        iso2: ISO2 country code override (legacy)
        dry_run: Execute validation without data publication - only used for "agol" mode
        verbose: Enable detailed logging output
        use_divisions: Use Overture Divisions for precise boundaries
        log_to_file: Create timestamped log files
        country: Country code/name for global config mode
        skip_cleanup: Skip temp file cleanup for debugging
    """
    # Validate parameter combinations based on output mode
    if output_mode == "geojson":
        if dry_run:
            logging.error("--dry-run is not supported with GeoJSON export")
            raise typer.Exit(1)
        if mode and mode != "auto":
            logging.error("--mode is not relevant for GeoJSON export")
            raise typer.Exit(1)
    elif output_mode == "agol":
        if output_path:
            logging.error("output_path should only be provided for GeoJSON export")
            raise typer.Exit(1)
    else:
        logging.error(f"Invalid output_mode: {output_mode}. Must be 'agol' or 'geojson'")
        raise typer.Exit(1)
    
    # Initialize logging with optional file output
    setup_logging(verbose, target_name, mode, log_to_file)
    
    # Register cleanup handlers for graceful interruption handling
    register_cleanup_handlers()
    
    # Perform temp directory cleanup and validation
    if not full_cleanup_check(skip_cleanup=skip_cleanup):
        logging.error("Temp directory size limit exceeded - aborting operation")
        raise typer.Exit(1)
    
    # Log execution context for audit trails
    start_time = datetime.now()
    logging.info(f"Initiating {mode} operation for {target_name} target")
    logging.info(f"Execution timestamp: {start_time}")
    
    # Log the original Python command for reproducibility
    if output_mode == "agol":
        cmd_parts = ["python", "-m", "o2agol.cli", "arcgis-upload", target_name, "-c", config]
        if mode and mode != "auto":
            cmd_parts.extend(["--mode", mode])
        if dry_run:
            cmd_parts.append("--dry-run")
    else:  # geojson mode
        cmd_parts = ["python", "-m", "o2agol.cli", "geojson-download", target_name]
        if output_path:
            cmd_parts.append(output_path)
        cmd_parts.extend(["-c", config])
    
    # Add common parameters
    if limit:
        cmd_parts.extend(["--limit", str(limit)])
    if iso2:
        cmd_parts.extend(["--iso2", iso2])
    if country:
        cmd_parts.extend(["--country", country])
    if not use_divisions:
        cmd_parts.append("--use-bbox")
    if verbose:
        cmd_parts.append("--verbose")
    if log_to_file:
        cmd_parts.append("--log-to-file")
    
    command_str = " ".join(cmd_parts)
    logging.info(f"Original command: {command_str}")
    
    logging.info(f"Configuration file: {config}")
    logging.info(f"Feature limit: {limit or 'No limit (full dataset)'}")
    logging.info(f"Spatial filtering method: {'Overture Divisions' if use_divisions else 'Bounding Box'}")
    logging.info(f"Dry run mode: {dry_run}")
    
    # Validate parameter combinations
    if country and iso2:
        logging.warning(f"Both --country ({country}) and --iso2 ({iso2}) provided. Using --country for global config, --iso2 for selector override.")
    
    # Load unified configuration with optional country override
    cfg = load_pipeline_config(config, country)
    
    # Log configuration status
    logging.info(f"Connected to ArcGIS as: {cfg['gis'].users.me.username}")
    logging.info(f"Environment: {cfg['environment']}")
    logging.info(f"Overture release: {cfg['overture']['release']} (from YAML)")

    # Get selector configuration with optional ISO2 override
    selector = get_selector_config(cfg, iso2)
    logging.debug(f"Selector configuration: {selector}")
    
    if iso2:
        logging.info(f"Country code override applied: {iso2.upper()}")
    else:
        # Log the country being used from global config
        selector_iso2 = selector.get('iso2')
        if selector_iso2:
            logging.info(f"Using country from global config: {selector_iso2}")

    # Get target configuration
    try:
        target_config = get_target_config(cfg, target_name)
        logging.info(f"Target theme: {target_config['theme']}, type: {target_config['type']}")
    except (KeyError, ValueError) as e:
        logging.error(str(e))
        raise typer.Exit(1) from e

    # Configure spatial filtering methodology
    bbox_only = not use_divisions
    
    try:
        logging.info("=" * 50)
        logging.info("DATA ACQUISITION PHASE")
        logging.info("=" * 50)
        
        if use_divisions:
            logging.info("Implementing Overture Divisions-based country boundary filtering")
        else:
            logging.info(f"Applying bounding box spatial filter (bbox_only={bbox_only})")
        
        # Create configuration object for fetch_gdf
        class ConfigAdapter:
            def __init__(self, cfg_dict, selector_dict, target_dict, target_name):
                self.yaml_config = cfg_dict['yaml']
                self.secure_config = cfg_dict['secure']
                self.overture_config = cfg_dict['overture']  # YAML-based config
                # Store selector as a simple object with the dict as __dict__ 
                self.selector = type('SelectorConfig', (), {})()
                if selector_dict:
                    self.selector.__dict__.update(selector_dict)
                
                # Create target config with instance attributes (not class attributes)
                target_config_instance = type('TargetConfig', (), {})()
                for key, value in target_dict.items():
                    setattr(target_config_instance, key, value)
                self.targets = {target_name: target_config_instance}
                
        config_adapter = ConfigAdapter(cfg, selector, target_config, target_name)
        
        # Execute data retrieval with specified parameters
        data_result = fetch_gdf(config_adapter, target_name, limit=limit, bbox_only=bbox_only, use_divisions=use_divisions)
        
        # Handle dual query results (dict) vs single query (GeoDataFrame)
        is_dual_query = isinstance(data_result, dict)
        if is_dual_query:
            total_features = sum(len(gdf) for gdf in data_result.values())
            logging.info(f"Data acquisition completed: {total_features:,} features retrieved (dual query)")
        else:
            gdf = data_result
            logging.info(f"Data acquisition completed: {len(gdf):,} features retrieved")

        # Execute dry run validation if requested
        if dry_run:
            logging.info("=" * 50)
            logging.info("DRY RUN VALIDATION MODE")
            logging.info("=" * 50)
            if is_dual_query:
                logging.info(f"Target features for processing: {total_features:,} (places + buildings)")
                for layer_name, layer_gdf in data_result.items():
                    logging.info(f"  - {layer_name}: {len(layer_gdf):,} features")
            else:
                logging.info(f"Target features for processing: {len(gdf):,}")
            
            # Get appropriate title for logging
            if cfg.get('config_format') == 'template':
                template_metadata = cfg['template'].get_target_metadata(target_name)
                title_display = template_metadata.item_title
            else:
                title_display = target_config.get('agol', {}).get('item_title', 'Not specified')
            
            logging.info(f"Target configuration: {title_display}")
            logging.info(f"Processing mode: {mode}")
            
            item_id = target_config.get('agol', {}).get('item_id')
            if item_id:
                logging.info(f"Target item identifier: {item_id}")
            
            logging.info("Dry run validation completed - no data published")
            logging.info("=" * 50)
            return

        logging.info("=" * 50)
        logging.info("SCHEMA NORMALIZATION PHASE")
        logging.info("=" * 50)
        
        if is_dual_query:
            # Normalize each layer separately
            normalized_data = {}
            for layer_name, layer_gdf in data_result.items():
                normalized_data[layer_name] = normalize_schema(layer_gdf, target_name)
                logging.info(f"Schema normalization completed for {layer_name}: {len(normalized_data[layer_name]):,} features")
        else:
            gdf = normalize_schema(gdf, target_name)
            logging.info(f"Schema normalization completed: {len(gdf):,} features processed")

        # Handle output based on mode
        if output_mode == "geojson":
            logging.info("=" * 50)
            logging.info("GEOJSON EXPORT PHASE")
            logging.info("=" * 50)
            
            if is_dual_query:
                export_to_geojson(normalized_data, output_path, target_name)
            else:
                export_to_geojson(gdf, output_path, target_name)
            
            logging.info("GeoJSON export completed successfully")
            return

        logging.info("=" * 50)
        logging.info("DATA PUBLICATION PHASE")
        logging.info("=" * 50)
        
        # Create target config adapter for publish function
        if cfg.get('config_format') == 'template':
            # Use template metadata with dynamic variables
            template_metadata = cfg['template'].get_target_metadata(target_name)
            
            # Create AGOL config with essential metadata
            agol_config_dict = {
                'item_title': template_metadata.item_title,
                'snippet': template_metadata.snippet,
                'description': template_metadata.description,
                'service_name': template_metadata.service_name,
                'tags': template_metadata.tags,
                'access_information': template_metadata.access_information,
                'license_info': template_metadata.license_info,
                'upsert_key': template_metadata.upsert_key,
                'item_id': template_metadata.item_id
            }
            
            target_adapter = type('TargetConfig', (), {
                'agol': type('AGOLConfig', (), agol_config_dict)()
            })()
            
            logging.info(f"Using template metadata: {template_metadata.item_title}")
            
        else:
            # Legacy config adapter
            target_adapter = type('TargetConfig', (), {
                'agol': type('AGOLConfig', (), target_config.get('agol', {}))()
            })()
        
        # Choose appropriate publishing method
        if is_dual_query:
            from .publish import publish_multi_layer_service
            result = publish_multi_layer_service(normalized_data, target_adapter, mode)
            total_published = sum(len(gdf) for gdf in normalized_data.values())
        else:
            result = publish_or_update(gdf, target_adapter, mode)
            total_published = len(gdf)
        
        # Generate execution summary for audit and monitoring
        end_time = datetime.now()
        execution_duration = end_time - start_time
        
        logging.info("=" * 50)
        logging.info(f"{mode.upper()} OPERATION COMPLETED SUCCESSFULLY")
        logging.info("=" * 50)
        logging.info(f"Total execution time: {execution_duration}")
        logging.info(f"Features processed: {total_published:,}")
        logging.info(f"Target layer: {result.title}")
        logging.info(f"Item identifier: {result.itemid}")
        logging.info(f"Service URL: {result.homepage}")
        if is_dual_query:
            logging.info(f"Multi-layer service created with {len(normalized_data)} layers")
        logging.info("Operation completed successfully")
        
    except Exception as e:
        end_time = datetime.now()
        execution_duration = end_time - start_time
        
        logging.error("=" * 50)
        logging.error(f"{mode.upper()} OPERATION FAILED")
        logging.error("=" * 50)
        logging.error(f"Execution time: {execution_duration}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error message: {str(e)}")
        
        # Include processing context for troubleshooting
        if 'gdf' in locals():
            logging.error(f"Features processed before failure: {len(gdf):,}")
        
        logging.error("Operation terminated due to error")
        raise


def generate_export_filename(query: str, country: str = None) -> str:
    """
    Generate default export filename using {iso3}_{query}.geojson pattern
    
    Args:
        query: Query name (e.g., 'education', 'roads')
        country: Country identifier (ISO2, ISO3, or name)
        
    Returns:
        Generated filename string
    """
    if country:
        from .config.countries import CountryRegistry
        try:
            country_info = CountryRegistry.get_country(country)
            if country_info:
                iso3 = country_info.iso3
                export_filename = f"{iso3.lower()}_{query}.geojson"
                logging.info(f"Generated default export filename: {export_filename}")
                return export_filename
            else:
                # Fallback if country not found
                export_filename = f"{country}_{query}.geojson"
                logging.warning(f"Country '{country}' not found in registry, using fallback filename: {export_filename}")
                return export_filename
        except (AttributeError, ValueError) as e:
            # Fallback if country lookup fails
            export_filename = f"{country}_{query}.geojson"
            logging.warning(f"Could not resolve ISO3 for '{country}' ({e}), using fallback filename: {export_filename}")
            return export_filename
    else:
        # Use generic filename if no country specified
        export_filename = f"{query}.geojson"
        logging.info(f"No country specified, using generic filename: {export_filename}")
        return export_filename


@app.command("arcgis-upload")
def arcgis_upload(
    query: Annotated[str, typer.Argument(help="Query type to execute. Use 'list-queries' command to see available options.")],
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "configs/global.yml",
    mode: Annotated[str, typer.Option("--mode", "-m", help="Processing mode: auto (smart detection) | initial | overwrite | append")] = "auto",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    iso2: Annotated[Optional[str], typer.Option("--iso2", help="ISO2 country code override (legacy)")] = None,
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name for global config (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without publishing")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    skip_cleanup: Annotated[bool, typer.Option("--skip-cleanup", help="Skip temp file cleanup for debugging")] = False,
):
    """
    Upload processed Overture Maps data to ArcGIS Online.
    
    Query types are defined in the configuration file under 'targets' section.
    Standard queries include: roads, buildings, places, education, health, markets
    
    Uses configs/global.yml by default. Users can add custom queries by defining 
    new targets in their configuration file. Each query must specify at minimum: 
    theme, type, and optional filter parameters.
    
    Examples:
        Standard queries (using default global config):
            o2agol arcgis-upload roads --country af
            o2agol arcgis-upload education --country pak --limit 100
            o2agol arcgis-upload buildings --country ind --dry-run
            
        With development options:
            o2agol arcgis-upload health --country afg --use-bbox --limit 50
            
        Custom config file:
            o2agol arcgis-upload my_custom_query -c configs/custom.yml --dry-run
    """
    # Validate that the query exists in the configuration
    if not validate_query_exists(config, query):
        try:
            available_queries = get_available_queries(config)
            typer.echo(f"ERROR: Query '{query}' not found in configuration file '{config}'", err=True)
            typer.echo(f"\nAvailable queries: {', '.join(sorted(available_queries))}", err=True)
            if config == "configs/global.yml":
                typer.echo("\nTip: Use 'python -m o2agol.cli list-queries' to see detailed information about each query.", err=True)
            else:
                typer.echo(f"\nTip: Use 'python -m o2agol.cli list-queries -c {config}' to see detailed information about each query.", err=True)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"ERROR loading configuration: {e}", err=True)
        raise typer.Exit(1)
    
    # Log which query is being executed
    logging.info(f"Executing Overture query: {query}")
    
    # Execute the query using existing process_target function
    process_target(
        target_name=query,
        config=config,
        output_mode="agol",
        mode=mode,
        output_path=None,
        limit=limit,
        iso2=iso2,
        dry_run=dry_run,
        verbose=verbose,
        use_divisions=use_divisions,
        log_to_file=log_to_file,
        country=country,
        skip_cleanup=skip_cleanup
    )


@app.command("geojson-download")
def geojson_download(
    query: Annotated[str, typer.Argument(help="Query type to execute. Use 'list-queries' command to see available options.")],
    output_path: Annotated[Optional[str], typer.Argument(help="Output GeoJSON file path (optional - will auto-generate if not provided)")] = None,
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "configs/global.yml",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    iso2: Annotated[Optional[str], typer.Option("--iso2", help="ISO2 country code override (legacy)")] = None,
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name for global config (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    skip_cleanup: Annotated[bool, typer.Option("--skip-cleanup", help="Skip temp file cleanup for debugging")] = False,
):
    """
    Download processed Overture Maps data as GeoJSON file.
    
    Query types are defined in the configuration file under 'targets' section.
    Standard queries include: roads, buildings, places, education, health, markets
    
    If no output path is specified, a filename will be auto-generated using the 
    pattern: {iso3}_{query}.geojson (e.g., afg_roads.geojson)
    
    Examples:
        Auto-generated filename:
            o2agol geojson-download roads --country af
            o2agol geojson-download education --country pak --limit 100
            
        Specify output file:
            o2agol geojson-download buildings afghanistan_buildings.geojson --country af
            
        With development options:
            o2agol geojson-download health --country afg --use-bbox --limit 50
            
        Custom config file:
            o2agol geojson-download my_custom_query output.geojson -c configs/custom.yml
    """
    # Validate that the query exists in the configuration
    if not validate_query_exists(config, query):
        try:
            available_queries = get_available_queries(config)
            typer.echo(f"ERROR: Query '{query}' not found in configuration file '{config}'", err=True)
            typer.echo(f"\nAvailable queries: {', '.join(sorted(available_queries))}", err=True)
            if config == "configs/global.yml":
                typer.echo("\nTip: Use 'python -m o2agol.cli list-queries' to see detailed information about each query.", err=True)
            else:
                typer.echo(f"\nTip: Use 'python -m o2agol.cli list-queries -c {config}' to see detailed information about each query.", err=True)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"ERROR loading configuration: {e}", err=True)
        raise typer.Exit(1)
    
    # Auto-generate filename if not provided
    if not output_path:
        output_path = generate_export_filename(query, country)
        
    # Log which query is being executed
    logging.info(f"Executing Overture query: {query}")
    logging.info(f"Output file: {output_path}")
    
    # Execute the query using existing process_target function
    process_target(
        target_name=query,
        config=config,
        output_mode="geojson",
        mode=None,
        output_path=output_path,
        limit=limit,
        iso2=iso2,
        dry_run=False,
        verbose=verbose,
        use_divisions=use_divisions,
        log_to_file=log_to_file,
        country=country,
        skip_cleanup=skip_cleanup
    )


@app.command("list-queries")
def list_queries(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "configs/global.yml",
):
    """
    List all available Overture queries in the configuration file.
    
    Shows detailed information about each query including theme, type, filter,
    and description to help users understand what data each query retrieves.
    
    Examples:
        python -m o2agol.cli list-queries -c configs/global.yml
        python -m o2agol.cli list-queries -c configs/custom.yml
    """
    try:
        queries = get_available_queries(config)
        
        if not queries:
            typer.echo(f"WARNING: No queries found in configuration file: {config}")
            raise typer.Exit(1)
        
        # Load full config for detailed information
        with open(config) as f:
            config_data = yaml.safe_load(f)
        
        typer.echo("Available Overture Queries")
        typer.echo("=" * 50)
        
        for query_name in sorted(queries):
            target = config_data['targets'][query_name]
            theme = target.get('theme', 'N/A')
            type_ = target.get('type', 'N/A')
            filter_ = target.get('filter', None)
            building_filter = target.get('building_filter', None)
            description = target.get('sector_description', target.get('description', None))
            
            # Query name header
            typer.echo(f"\n* {query_name}")
            typer.echo(f"   Theme: {theme}")
            typer.echo(f"   Type: {type_}")
            
            if filter_:
                typer.echo(f"   Filter: {filter_}")
            
            if building_filter:
                typer.echo(f"   Building Filter: {building_filter}")
            
            if description:
                typer.echo(f"   Description: {description}")
        
        typer.echo(f"\nFound {len(queries)} available queries")
        if config == "configs/global.yml":
            typer.echo("\nUsage:")
            typer.echo("  Upload to ArcGIS Online: o2agol arcgis-upload <query_name> [options]")
            typer.echo("  Export to GeoJSON:      o2agol geojson-download <query_name> [options]")
        else:
            typer.echo("\nUsage:")
            typer.echo(f"  Upload to ArcGIS Online: o2agol arcgis-upload <query_name> -c {config} [options]")
            typer.echo(f"  Export to GeoJSON:      o2agol geojson-download <query_name> -c {config} [options]")
        
    except FileNotFoundError:
        typer.echo(f"ERROR: Configuration file not found: {config}", err=True)
        raise typer.Exit(1)
    except yaml.YAMLError as e:
        typer.echo(f"ERROR: Invalid YAML in configuration file: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)




@app.command("overture-dump")
def overture_dump(
    query: Annotated[str, typer.Argument(help="Query type to execute. Use 'list-queries' command to see available options.")],
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "configs/global.yml",
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    mode: Annotated[str, typer.Option("--mode", "-m", help="Processing mode: auto (smart detection) | initial | overwrite | append")] = "auto",
    download_only: Annotated[bool, typer.Option("--download-only", help="Only download dump without processing")] = False,
    use_local: Annotated[bool, typer.Option("--use-local/--use-s3", help="Use local dump if available")] = True,
    force_download: Annotated[bool, typer.Option("--force-download", help="Force re-download even if dump exists")] = False,
    release: Annotated[str, typer.Option("--release", help="Overture release version")] = "2025-07-23.0",
    output_format: Annotated[str, typer.Option("--format", help="Output format: agol or geojson")] = "agol",
    output_path: Annotated[Optional[str], typer.Option("--output", help="Output GeoJSON file path (for geojson format)")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without processing/publishing")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    # Cache optimization flags
    use_world_bank: Annotated[bool, typer.Option("--world-bank/--overture-divisions", help="Use World Bank boundaries (default: enabled)")] = True,
):
    """
    Process Overture data using efficient country-specific caching.
    
    This command caches data at the country level rather than downloading complete 
    themes, providing the same performance benefits while avoiding memory issues 
    and aligning with Overture Maps best practices.
    
    The cache system stores country-specific extracts in ./cache/ using proven 
    DuckDB streaming extraction from Overture's S3 buckets.
    
    Examples:
        Cache and process data for a country:
            o2agol overture-dump roads --country afg
            o2agol overture-dump roads --country pak  # Each country cached separately
            
        Specify publishing mode (similar to arcgis-upload):
            o2agol overture-dump buildings --country afg --mode overwrite
            o2agol overture-dump places --country pak --mode initial
            
        Force refresh cache:
            o2agol overture-dump buildings --country afg --force-download
            
        Export to GeoJSON instead of AGOL:
            o2agol overture-dump places --country afg --format geojson
            
        Cache only without processing:
            o2agol overture-dump roads --country afg --download-only
    """
    # Validate query exists
    if not validate_query_exists(config, query):
        try:
            available_queries = get_available_queries(config)
            typer.echo(f"ERROR: Query '{query}' not found in configuration file '{config}'", err=True)
            typer.echo(f"\nAvailable queries: {', '.join(sorted(available_queries))}", err=True)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"ERROR loading configuration: {e}", err=True)
        raise typer.Exit(1)
    
    # Validate mode parameter
    valid_modes = ["auto", "initial", "overwrite", "append"]
    if mode not in valid_modes:
        typer.echo(f"ERROR: --mode must be one of: {', '.join(valid_modes)}", err=True)
        raise typer.Exit(1)
    
    # Validate output format
    if output_format not in ["agol", "geojson"]:
        typer.echo("ERROR: --format must be 'agol' or 'geojson'", err=True)
        raise typer.Exit(1)
    
    # Validate parameter combinations
    if output_format == "geojson" and dry_run:
        typer.echo("ERROR: --dry-run is not supported with GeoJSON export", err=True)
        raise typer.Exit(1)
    
    if output_format == "agol" and output_path:
        typer.echo("ERROR: --output is only supported with --format geojson", err=True)
        raise typer.Exit(1)
    
    if not country:
        typer.echo("ERROR: --country is required for dump processing", err=True)
        raise typer.Exit(1)
    
    # Setup logging
    log_mode = "dump"
    setup_logging(verbose, query, log_mode, log_to_file)
    
    # Initialize comprehensive timing
    import time
    from datetime import datetime, timedelta
    
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
    
    # Note: Phase logging now handled by PipelineLogger in the main flow
    
    # Initialize pipeline logger
    base_logger = logging.getLogger(__name__)
    pipeline_logger = PipelineLogger(base_logger)
    
    # Capture original command for reference
    import sys
    import os
    
    # Extract just the command name and arguments, not the full file path
    if sys.argv and sys.argv[0]:
        command_name = os.path.basename(sys.argv[0])
        # Remove .py extension if present
        if command_name.endswith('.py'):
            command_name = command_name[:-3]
        # Handle Windows .exe or just the script name
        if command_name.endswith('.exe'):
            command_name = command_name[:-4]
        command_args = sys.argv[1:] if len(sys.argv) > 1 else []
        original_command = f"{command_name} {' '.join(command_args)}" if command_args else command_name
    else:
        original_command = "o2agol overture-dump"  # Fallback
    
    # Start overall timing
    operation_start_time = time.time()
    pipeline_logger.info(f"Command: {original_command}")
    pipeline_logger.info(f"============ OVERTURE DUMP: {query.upper()} ============")
    
    # Phase timings
    phase_times = {}
    
    # Load configuration to get theme information
    cfg = load_pipeline_config(config, country)
    target_config = get_target_config(cfg, query)
    theme = target_config['theme']
    
    # Get country info for display
    from .config.countries import CountryRegistry
    country_info = CountryRegistry.get_country(country)
    
    # Detect dual-theme queries (places + buildings)
    building_filter = target_config.get('building_filter')
    themes_to_download = [theme]
    if building_filter:
        themes_to_download.append('buildings')
    
    pipeline_logger.info(f"Configuration: {country_info.name} ({country_info.iso3}) | Query: {query} | Release: {release}")
    
    # Create cache configuration from CLI parameters
    from .dump_manager import DumpConfig
    
    # Create DumpConfig with cache optimization settings
    dump_config = DumpConfig()
    
    # Preserve any existing base_path from old config if available
    if hasattr(cfg['secure'], 'dump') and hasattr(cfg['secure'].dump, 'base_path'):
        dump_config.base_path = cfg['secure'].dump.base_path
    
    # Apply CLI optimization parameters
    dump_config.use_world_bank_boundaries = use_world_bank
    dump_config.enable_boundary_cache = use_world_bank  # Enable cache if using World Bank boundaries
    
    # Initialize dump manager with cache system
    dump_manager = DumpManager(config=dump_config)
    
    # Configuration phase complete
    logging.info(f"Cache storage location: {dump_manager.base_path}/country_cache")
    logging.info(f"Using country-specific caching approach")
    logging.info(f"Boundary optimization: world_bank={use_world_bank}")
    
    if dry_run:
        logging.info("Dry run mode - validating configuration...")
        
        # Validate country and configuration
        try:
            from .config.countries import CountryRegistry
            
            country_info = CountryRegistry.get_country(country)
            if not country_info:
                raise ValueError(f"Unknown country: {country}")
            
            # Check cache status for each theme
            cached_themes = []
            missing_themes = []
            
            for theme_to_check in themes_to_download:
                type_name = target_config['type'] if theme_to_check == theme else 'building'
                
                # Check if cache exists for this country/theme combination
                cache_entries = dump_manager.list_cached_data(release)
                theme_cached = any(
                    entry.country == country_info.iso2 and 
                    entry.theme == theme_to_check and 
                    entry.type_name == type_name
                    for entry in cache_entries
                )
                
                if theme_cached:
                    if force_download:
                        logging.info(f"Found cached {theme_to_check} data for {country_info.name} - would be refreshed")
                        missing_themes.append(theme_to_check)
                    else:
                        logging.info(f"Found cached {theme_to_check} data for {country_info.name}")
                        cached_themes.append(theme_to_check)
                else:
                    logging.info(f"No cached {theme_to_check} data for {country_info.name}")
                    missing_themes.append(theme_to_check)
            
            if missing_themes:
                if force_download:
                    logging.info(f"Themes to be refreshed: {missing_themes}")
                else:
                    logging.info(f"Themes to be cached: {missing_themes}")
                logging.info("Would extract and cache this data in non-dry-run mode")
            
            logging.info("Dry run complete - configuration validated.")
            return
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            raise typer.Exit(1)
    
    # Handle download_only mode by caching data for the country  
    if download_only:
        cache_start = time.time()
        pipeline_logger.info("")
        pipeline_logger.phase("PHASE 1: CACHE DOWNLOAD")
        
        try:
            # Cache data for each required theme
            total_features = 0
            for theme_to_cache in themes_to_download:
                type_name = target_config['type'] if theme_to_cache == theme else 'building'
                
                theme_desc = "transportation" if theme_to_cache == "roads" else f"{theme_to_cache} facilities"
                pipeline_logger.info(f"Caching {theme_desc} for {country_info.name}")
                
                result_gdf = dump_manager.cache_country_data(
                    country=country_info.iso2,
                    theme=theme_to_cache,
                    type_name=type_name,
                    config_obj=cfg,
                    release=release,
                    overwrite=force_download
                )
                
                if result_gdf is not None and not result_gdf.empty:
                    total_features += len(result_gdf)
            
            # Phase completion timing
            cache_duration = time.time() - cache_start
            pipeline_logger.info(f"Acquired {total_features:,} features")
            pipeline_logger.info(f"Phase completed in {format_duration(cache_duration)}")
            
            # Final summary for download-only
            total_duration = time.time() - operation_start_time
            pipeline_logger.info("")
            pipeline_logger.info("============ OPERATION COMPLETE ============")
            pipeline_logger.info(f"Total execution time: {format_duration(total_duration)}")
            pipeline_logger.info("Performance breakdown:")
            pipeline_logger.info(f"  Configuration: {format_duration(cache_start - operation_start_time)} ({((cache_start - operation_start_time)/total_duration*100):.1f}%)")
            pipeline_logger.info(f"  Cache Download: {format_duration(cache_duration)} ({(cache_duration/total_duration*100):.1f}%)")
            
            return
            
        except Exception as e:
            pipeline_logger.info(f"Failed to cache data: {e}")
            raise typer.Exit(1)
    
    # Process query using local dump
    try:
        from .dump_manager import DumpQuery
        from .config.countries import CountryRegistry
        
        # Resolve country
        country_info = CountryRegistry.get_country(country)
        if not country_info:
            raise ValueError(f"Unknown country: {country}")
        
        # Build query
        query_obj = DumpQuery(
            theme=theme,
            type=target_config['type'],
            country=country_info.iso2,
            bbox=CountryRegistry.get_bounding_boxes().get(country_info.iso2) if not use_divisions else None,
            filters=target_config.get('filter'),
            limit=limit
        )
        
        # === PHASE 1: DATA ACQUISITION ===
        phase2_start = time.time()
        pipeline_logger.info("")
        pipeline_logger.phase("PHASE 1: DATA ACQUISITION")
        
        # Handle dual-theme queries directly
        if building_filter:
            pipeline_logger.info("Retrieving education facilities (places + buildings)" if query == "education" 
                               else "Retrieving health facilities (places + buildings)" if query == "health"
                               else "Retrieving retail facilities (places + buildings)" if query == "markets"
                               else f"Retrieving {query} facilities (places + buildings)")
            query_result = dump_manager.query_dual_theme(
                query_obj, 
                building_filter, 
                release, 
                cfg, 
                force_download=force_download
            )
        else:
            pipeline_logger.info(f"Retrieving {query} data")
            query_result = dump_manager.query_local_dump(query_obj, release, cfg, force_download=force_download)
        
        # Handle both single GeoDataFrame and dual-query dict results
        if isinstance(query_result, dict):
            total_features = sum(len(gdf) for gdf in query_result.values())
            if total_features == 0:
                pipeline_logger.info("No data found for query")
                return
            
            # Create feature summary
            feature_summary = ", ".join([f"{len(gdf):,} {layer_name}" for layer_name, gdf in query_result.items()])
            pipeline_logger.info(f"Acquired {total_features:,} features ({feature_summary})")
            
            # Add timing for data acquisition
            phase_times['data_acquisition'] = time.time() - phase2_start
            pipeline_logger.info(f"Phase completed in {format_duration(phase_times['data_acquisition'])}")
            
            # === PHASE 2: TRANSFORMATION ===
            phase3_start = time.time()
            pipeline_logger.info("")
            pipeline_logger.phase("PHASE 2: TRANSFORMATION")
            
            # Transform each layer
            transformed_data = {}
            for layer_name, layer_gdf in query_result.items():
                if not layer_gdf.empty:
                    transformed_data[layer_name] = normalize_schema(layer_gdf, query)
            
            if not transformed_data:
                pipeline_logger.info("No valid data after transformation")
                return
            
            pipeline_logger.info(f"Transformed {total_features:,} features")
            phase_times['transformation'] = time.time() - phase3_start
            pipeline_logger.info(f"Phase completed in {format_duration(phase_times['transformation'])}")
            
            # === PHASE 3: OUTPUT ===
            phase4_start = time.time()
            pipeline_logger.info("")
            pipeline_logger.phase("PHASE 3: AGOL PUBLISHING" if output_format == "agol" else "PHASE 3: GEOJSON EXPORT")
            
            # Output based on format
            if output_format == "geojson":
                if not output_path:
                    output_path = generate_export_filename(query, country)
                
                pipeline_logger.info(f"Exporting to {output_path}")
                # Use existing export_to_geojson which handles dict input
                export_to_geojson(transformed_data, output_path, query)
                pipeline_logger.info(f"Export complete: {output_path}")
                
            else:  # agol format
                # Create target configuration adapter (similar to single-theme approach)
                target_config = cfg.get('targets', {}).get(query, {})
                if cfg.get('config_format') == 'template':
                    # Use template metadata with dynamic variables
                    template_metadata = cfg['template'].get_target_metadata(query)
                    
                    # Create AGOL config with essential metadata
                    agol_config_dict = {
                        'item_title': template_metadata.item_title,
                        'snippet': template_metadata.snippet,
                        'description': template_metadata.description,
                        'service_name': template_metadata.service_name,
                        'tags': template_metadata.tags,
                        'access_information': template_metadata.access_information,
                        'license_info': template_metadata.license_info,
                        'upsert_key': template_metadata.upsert_key,
                        'item_id': template_metadata.item_id
                    }
                    
                    target_adapter = type('TargetConfig', (), {
                        'agol': type('AGOLConfig', (), agol_config_dict)()
                    })()
                    
                    pipeline_logger.info(f"Found existing service: {template_metadata.item_title}")
                else:
                    # Legacy config adapter
                    target_adapter = type('TargetConfig', (), {
                        'agol': type('AGOLConfig', (), target_config.get('agol', {}))()
                    })()
                
                # Use multi-layer publishing service (same as arcgis-upload command)
                from .publish import publish_multi_layer_service
                result = publish_multi_layer_service(transformed_data, target_adapter, mode)
                
                # publish_multi_layer_service returns the published item, not a dict
                if result:
                    # Extract layer info for display
                    layers_info = []
                    for layer_name, layer_gdf in transformed_data.items():
                        layer_type = "points" if "places" in layer_name else "polygons"
                        layers_info.append(f"Updated layer {country_info.iso3.lower()}_{query}_{layer_type} ({len(layer_gdf):,} features)")
                    
                    for info in layers_info:
                        pipeline_logger.info(info)
                    
                    pipeline_logger.info(f"Published: {result.homepage}")
                else:
                    pipeline_logger.info("Failed to publish service to ArcGIS Online")
                    raise typer.Exit(1)
            
            # End Phase 3 for dual-theme
            phase_times['output'] = time.time() - phase4_start
            pipeline_logger.info(f"Phase completed in {format_duration(phase_times['output'])}")
        
        else:
            # Single GeoDataFrame result
            gdf = query_result
            
            if gdf.empty:
                pipeline_logger.info("No data found for query")
                return
            
            # Log single-theme acquisition
            pipeline_logger.info(f"Acquired {len(gdf):,} features")
            phase_times['data_acquisition'] = time.time() - phase2_start
            pipeline_logger.info(f"Phase completed in {format_duration(phase_times['data_acquisition'])}")
            
            # === PHASE 2: TRANSFORMATION ===
            phase3_start = time.time()
            pipeline_logger.info("")
            pipeline_logger.phase("PHASE 2: TRANSFORMATION")
            
            # Transform data
            gdf_transformed = normalize_schema(gdf, query)
            pipeline_logger.info(f"Transformed {len(gdf):,} features")
            
            # End Phase 2
            phase_times['transformation'] = time.time() - phase3_start
            pipeline_logger.info(f"Phase completed in {format_duration(phase_times['transformation'])}")
            
            # === PHASE 3: OUTPUT ===
            phase4_start = time.time()
            pipeline_logger.info("")
            pipeline_logger.phase("PHASE 3: AGOL PUBLISHING" if output_format == "agol" else "PHASE 3: GEOJSON EXPORT")
            
            # Output based on format
            if output_format == "geojson":
                if not output_path:
                    output_path = generate_export_filename(query, country)
                
                pipeline_logger.info(f"Exporting to {output_path}")
                export_to_geojson(gdf_transformed, output_path, query)
                pipeline_logger.info(f"Export complete: {output_path}")
                
            else:  # agol format
                
                # Create target configuration adapter (similar to arcgis-upload command)
                target_config = cfg.get('targets', {}).get(query, {})
                if cfg.get('config_format') == 'template':
                    # Use template metadata with dynamic variables
                    template_metadata = cfg['template'].get_target_metadata(query)
                    
                    # Create AGOL config with essential metadata
                    agol_config_dict = {
                        'item_title': template_metadata.item_title,
                        'snippet': template_metadata.snippet,
                        'description': template_metadata.description,
                        'service_name': template_metadata.service_name,
                        'tags': template_metadata.tags,
                        'access_information': template_metadata.access_information,
                        'license_info': template_metadata.license_info,
                        'upsert_key': template_metadata.upsert_key,
                        'item_id': template_metadata.item_id
                    }
                    
                    target_adapter = type('TargetConfig', (), {
                        'agol': type('AGOLConfig', (), agol_config_dict)()
                    })()
                    
                    pipeline_logger.info(f"Found existing service: {template_metadata.item_title}")
                else:
                    # Legacy config adapter
                    target_adapter = type('TargetConfig', (), {
                        'agol': type('AGOLConfig', (), target_config.get('agol', {}))()
                    })()
                
                result = publish_or_update(gdf_transformed, target_adapter, mode)
                
                # publish_or_update returns the published item, not a dict
                if result:
                    pipeline_logger.info(f"Updated layer with {len(gdf_transformed):,} features")
                    pipeline_logger.info(f"Published: {result.homepage}")
                else:
                    pipeline_logger.info("Failed to publish to ArcGIS Online")
                    raise typer.Exit(1)
            
            # End Phase 3 for single-theme
            phase_times['output'] = time.time() - phase4_start
            pipeline_logger.info(f"Phase completed in {format_duration(phase_times['output'])}")
        
        # === OPERATION SUMMARY ===
        total_duration = time.time() - operation_start_time
        pipeline_logger.info("")
        pipeline_logger.info("============ OPERATION COMPLETE ============")
        pipeline_logger.info(f"Total execution time: {format_duration(total_duration)}")
        pipeline_logger.info("Performance breakdown:")
        
        # Calculate percentage for each phase and show both seconds and percentage
        for phase_name, phase_duration in phase_times.items():
            percentage = (phase_duration / total_duration) * 100 if total_duration > 0 else 0
            formatted_phase = phase_name.replace('_', ' ').title().replace('Data ', '').replace('Acquisition', 'Acquisition')
            if formatted_phase == 'Output':
                formatted_phase = 'AGOL Publishing' if output_format == 'agol' else 'GeoJSON Export'
            pipeline_logger.info(f"  {formatted_phase}: {format_duration(phase_duration)} ({percentage:.1f}%)")
    
    except Exception as e:
        # Calculate timing even for errors
        error_duration = time.time() - operation_start_time
        logging.error("=" * 60)
        logging.error("OPERATION FAILED")
        logging.error("=" * 60)
        logging.error(f"Processing failed after {format_duration(error_duration)}: {e}")
        if phase_times:
            logging.error("Completed phases:")
            for phase_name, phase_duration in phase_times.items():
                logging.error(f"  {phase_name.replace('_', ' ').title()}: {format_duration(phase_duration)}")
        raise typer.Exit(1)
    
    finally:
        dump_manager.close()


@app.command("list-dumps")
def list_dumps():
    """
    List all available local Overture dumps with metadata.
    
    Shows information about downloaded dumps including release version,
    themes, download date, size, and validation status.
    """
    # Initialize dump manager with default configuration
    from .dump_manager import DumpConfig
    dump_config = DumpConfig()
    dump_manager = DumpManager(config=dump_config)
    
    try:
        dumps = dump_manager.get_available_dumps()
        
        if not dumps:
            typer.echo("No local dumps found.")
            typer.echo("Use 'o2agol overture-dump <query> --download-only' to cache data.")
            return
        
        typer.echo("Available Local Overture Dumps")
        typer.echo("=" * 50)
        
        for dump in dumps:
            typer.echo(f"\nRelease: {dump.release}")
            typer.echo(f"  Themes: {', '.join(dump.themes)}")
            typer.echo(f"  Downloaded: {dump.download_date}")
            typer.echo(f"  Size: {dump.size_gb:.2f} GB")
            typer.echo(f"  Complete: {'✓' if dump.is_complete else '✗'}")
            typer.echo(f"  Spatial Index: {'✓' if dump.spatial_index_built else '✗'}")
        
        total_size = sum(dump.size_gb for dump in dumps)
        typer.echo(f"\nTotal: {len(dumps)} dumps, {total_size:.2f} GB")
        
    except Exception as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)
    
    finally:
        dump_manager.close()


@app.command("validate-dump")
def validate_dump(
    release: Annotated[str, typer.Argument(help="Overture release version to validate")],
    theme: Annotated[str, typer.Argument(help="Theme to validate")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
):
    """
    Validate the integrity of a local Overture dump.
    
    Checks dump structure, metadata, and runs validation queries to ensure
    the dump is complete and usable.
    
    Examples:
        o2agol validate-dump 2025-07-23.0 buildings
        o2agol validate-dump 2025-07-23.0 transportation --verbose
    """
    setup_logging(verbose)
    
    # Initialize dump manager with default configuration
    from .dump_manager import DumpConfig
    dump_config = DumpConfig()
    dump_manager = DumpManager(config=dump_config)
    
    try:
        logging.info(f"Validating dump: {release}/{theme}")
        
        # Check if dump exists
        if not dump_manager.check_dump_exists(release, theme):
            typer.echo(f"ERROR: Dump not found: {release}/{theme}", err=True)
            raise typer.Exit(1)
        
        # Run validation
        is_valid = dump_manager.validate_dump(release, theme)
        
        if is_valid:
            typer.echo(f"✓ Dump validation successful: {release}/{theme}")
            logging.info("All validation checks passed")
        else:
            typer.echo(f"✗ Dump validation failed: {release}/{theme}", err=True)
            logging.error("One or more validation checks failed")
            raise typer.Exit(1)
    
    except Exception as e:
        typer.echo(f"ERROR during validation: {e}", err=True)
        raise typer.Exit(1)
    
    finally:
        dump_manager.close()


@app.command("list-cache")
def list_cache(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "configs/global.yml",
    release: Annotated[str, typer.Option("--release", help="Overture release version")] = "latest",
):
    """
    List cached country data entries.
    
    Shows all cached data with metadata including country, theme, feature counts, and file sizes.
    """
    try:
        # Load minimal config to get dump manager
        from .dump_manager import DumpManager, DumpConfig
        import os
        
        dump_config = DumpConfig()
        dump_config.base_path = os.path.abspath('./overturedump')
        dump_manager = DumpManager(config=dump_config)
        
        # Get cache entries
        cache_entries = dump_manager.list_cached_data(release)
        
        if not cache_entries:
            typer.echo(f"No cached data found for release: {release}")
            return
        
        # Group by country for better display
        countries = {}
        for entry in cache_entries:
            if entry.country not in countries:
                countries[entry.country] = []
            countries[entry.country].append(entry)
        
        typer.echo(f"Cached data for release: {release}")
        typer.echo("=" * 60)
        
        total_size_mb = 0
        total_features = 0
        
        for country_code, entries in sorted(countries.items()):
            typer.echo(f"\nCountry: {country_code}")
            for entry in entries:
                size_str = f"{entry.size_mb:.1f} MB"
                features_str = f"{entry.feature_count:,} features"
                typer.echo(f"  {entry.theme}/{entry.type_name}: {features_str}, {size_str}")
                total_size_mb += entry.size_mb
                total_features += entry.feature_count
        
        typer.echo("\n" + "=" * 60)
        typer.echo(f"Total: {len(cache_entries)} cache entries, {total_features:,} features, {total_size_mb:.1f} MB")
        
        # Show cache statistics
        stats = dump_manager.get_cache_stats()
        typer.echo(f"Cache location: {stats['cache_path']}")
        
    except Exception as e:
        typer.echo(f"ERROR: Failed to list cache: {e}", err=True)
        raise typer.Exit(1)


@app.command("clear-cache")
def clear_cache(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "configs/global.yml",
    country: Annotated[Optional[str], typer.Option("--country", help="Specific country to clear (optional)")] = None,
    release: Annotated[Optional[str], typer.Option("--release", help="Specific release to clear (optional)")] = None,
    confirm: Annotated[bool, typer.Option("--confirm", help="Confirm deletion without prompt")] = False,
):
    """
    Clear cached data entries.
    
    Can clear all cache, specific release, or specific country data.
    Use with caution as this will delete cached data files.
    """
    try:
        # Load minimal config to get dump manager
        from .dump_manager import DumpManager, DumpConfig
        import os
        
        dump_config = DumpConfig()
        dump_config.base_path = os.path.abspath('./overturedump')
        dump_manager = DumpManager(config=dump_config)
        
        # Determine what will be cleared
        if country and release:
            target = f"cached data for {country} in release {release}"
        elif country:
            target = f"all cached data for {country}"
        elif release:
            target = f"all cached data for release {release}"
        else:
            target = "ALL cached data"
        
        # Confirmation prompt
        if not confirm:
            if not typer.confirm(f"Are you sure you want to clear {target}?"):
                typer.echo("Operation cancelled.")
                return
        
        # Perform clearing
        dump_manager.clear_cache(country=country, release=release)
        typer.echo(f"Successfully cleared {target}")
        
    except Exception as e:
        typer.echo(f"ERROR: Failed to clear cache: {e}", err=True)
        raise typer.Exit(1)


@app.command("cache-stats")
def cache_stats(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "configs/global.yml",
):
    """
    Show cache statistics and storage usage.
    """
    try:
        # Load minimal config to get dump manager
        from .dump_manager import DumpManager, DumpConfig
        import os
        
        dump_config = DumpConfig()
        dump_config.base_path = os.path.abspath('./overturedump')
        dump_manager = DumpManager(config=dump_config)
        
        # Get cache statistics
        stats = dump_manager.get_cache_stats()
        
        typer.echo("Cache Statistics")
        typer.echo("=" * 40)
        typer.echo(f"Location: {stats['cache_path']}")
        typer.echo(f"Total files: {stats['total_files']}")
        typer.echo(f"Countries: {stats['countries']}")
        typer.echo(f"Releases: {stats['releases']}")
        typer.echo(f"Total size: {stats['total_size_mb']:.1f} MB ({stats['total_size_mb']/1024:.2f} GB)")
        
        if stats['total_files'] > 0:
            avg_size = stats['total_size_mb'] / stats['total_files']
            typer.echo(f"Average file size: {avg_size:.1f} MB")
        
    except Exception as e:
        typer.echo(f"ERROR: Failed to get cache statistics: {e}", err=True)
        raise typer.Exit(1)


@app.command("version")
def version():
    """Display version information and system details."""
    try:
        from . import __version__
        typer.echo(f"o2agol version: {__version__}")
    except ImportError:
        typer.echo("o2agol (development version)")


if __name__ == "__main__":
    app()