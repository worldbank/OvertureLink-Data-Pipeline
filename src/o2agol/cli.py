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
):
    """
    Process Overture data using local dumps for improved performance.
    
    This command downloads complete Overture themes once and reuses them for multiple 
    countries/queries, providing 10-100x performance improvement for batch operations.
    
    The dump system downloads by Overture's six themes (addresses, base, buildings, 
    divisions, places, transportation) and stores them locally in ./overturedump/.
    
    Examples:
        Download dump for theme (one-time operation):
            o2agol overture-dump roads --country afg --download-only
            
        Process from local dump (fast, repeatable):
            o2agol overture-dump roads --country afg
            o2agol overture-dump roads --country pak  # Uses same dump
            
        Force fresh download:
            o2agol overture-dump buildings --country afg --force-download
            
        Export to GeoJSON instead of AGOL:
            o2agol overture-dump places --country afg --format geojson
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
    mode = "dump"
    setup_logging(verbose, query, mode, log_to_file)
    
    # Load configuration to get theme information
    cfg = load_pipeline_config(config, country)
    target_config = get_target_config(cfg, query)
    theme = target_config['theme']
    
    # Detect dual-theme queries (places + buildings)
    building_filter = target_config.get('building_filter')
    themes_to_download = [theme]
    if building_filter:
        themes_to_download.append('buildings')
        logging.info(f"Dual-theme query detected: {query} requires themes {themes_to_download}")
    
    logging.info(f"Processing dump-based query: {query} (themes: {themes_to_download})")
    logging.info(f"Release: {release}")
    logging.info(f"Country: {country}")
    logging.info(f"Output format: {output_format}")
    
    # Initialize dump manager with configuration
    dump_manager = DumpManager(config=cfg.secure_config.dump)
    logging.info(f"Dump storage location: {dump_manager.base_path}")
    
    if dry_run:
        logging.info("Dry run mode - validating configuration and dump access...")
        # In dry run, check if dumps exist but don't download
        missing_themes = []
        for theme_to_download in themes_to_download:
            if not dump_manager.check_dump_exists(release, theme_to_download):
                missing_themes.append(theme_to_download)
            else:
                logging.info(f"Found existing {theme_to_download} dump")
        
        if missing_themes:
            logging.info(f"Missing themes for dry run: {missing_themes}")
            logging.info("Would download these themes in non-dry-run mode")
            logging.info("Dry run complete - configuration valid but dumps need downloading.")
            return
        
        # Test query with existing dumps
        try:
            from .dump_manager import DumpQuery
            from .config.countries import CountryRegistry
            
            country_info = CountryRegistry.get_country(country)
            if not country_info:
                raise ValueError(f"Unknown country: {country}")
            
            test_query = DumpQuery(
                theme=theme,
                type=target_config['type'],
                country=country_info.iso2,
                limit=1
            )
            
            result = dump_manager.query_local_dump(test_query, release)
            logging.info(f"Dump validation successful. Would process {len(result)} features (test with limit=1)")
            logging.info("Dry run complete.")
            return
            
        except Exception as e:
            logging.error(f"Dump validation failed: {e}")
            raise typer.Exit(1)
    
    # Check/download all required themes (non-dry-run only)
    for theme_to_download in themes_to_download:
        if force_download or not dump_manager.check_dump_exists(release, theme_to_download):
            if not force_download:
                logging.info(f"Dump not found for {theme_to_download}, downloading...")
            else:
                logging.info(f"Force downloading {theme_to_download} dump...")
            
            try:
                dump_manager.download_theme(release, theme_to_download)
                logging.info(f"Successfully downloaded {theme_to_download} dump")
            except Exception as e:
                logging.error(f"Failed to download {theme_to_download} dump: {e}")
                raise typer.Exit(1)
        else:
            logging.info(f"Using existing {theme_to_download} dump")
    
    if download_only:
        logging.info("Download complete. Exiting.")
        return
    
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
            filters=target_config.get('filters'),
            limit=limit
        )
        
        # Query local dump
        logging.info("Querying local dump...")
        query_result = dump_manager.query_local_dump(query_obj, release)
        
        # Handle both single GeoDataFrame and dual-query dict results
        if isinstance(query_result, dict):
            # Dual-query result (places + buildings)
            logging.info(f"Dual-query result: {list(query_result.keys())}")
            
            total_features = sum(len(gdf) for gdf in query_result.values())
            if total_features == 0:
                logging.warning("No data found for dual query")
                return
            
            logging.info(f"Found {total_features} total features across layers")
            
            # Transform each layer
            transformed_data = {}
            for layer_name, layer_gdf in query_result.items():
                if not layer_gdf.empty:
                    logging.info(f"Transforming {len(layer_gdf)} {layer_name} features...")
                    transformed_data[layer_name] = normalize_schema(layer_gdf, query)
            
            if not transformed_data:
                logging.warning("No valid data after transformation")
                return
            
            # Output based on format
            if output_format == "geojson":
                if not output_path:
                    output_path = generate_export_filename(query, country)
                
                logging.info(f"Exporting multi-layer data to {output_path}...")
                # Use existing export_to_geojson which handles dict input
                export_to_geojson(transformed_data, output_path, query)
                logging.info(f"Export complete: {output_path}")
                
            else:  # agol format
                logging.info("Publishing multi-layer service to ArcGIS Online...")
                result_dict = publish_or_update(
                    data=transformed_data,
                    pipeline_config=cfg,
                    target_name=query,
                    mode="auto"
                )
                
                if result_dict.get('success'):
                    logging.info("Successfully published multi-layer service to ArcGIS Online")
                    if result_dict.get('item_url'):
                        logging.info(f"Item URL: {result_dict['item_url']}")
                else:
                    logging.error("Failed to publish multi-layer service to ArcGIS Online")
                    raise typer.Exit(1)
        
        else:
            # Single GeoDataFrame result
            gdf = query_result
            
            if gdf.empty:
                logging.warning("No data found for query")
                return
            
            # Transform data
            logging.info(f"Transforming {len(gdf)} features...")
            gdf_transformed = normalize_schema(gdf, query)
            
            # Output based on format
            if output_format == "geojson":
                if not output_path:
                    output_path = generate_export_filename(query, country)
                
                logging.info(f"Exporting to {output_path}...")
                export_to_geojson(gdf_transformed, output_path, query)
                logging.info(f"Export complete: {output_path}")
                
            else:  # agol format
                logging.info("Publishing to ArcGIS Online...")
                result_dict = publish_or_update(
                    data=gdf_transformed,
                    pipeline_config=cfg,
                    target_name=query,
                    mode="auto"
                )
                
                if result_dict.get('success'):
                    logging.info("Successfully published to ArcGIS Online")
                    if result_dict.get('item_url'):
                        logging.info(f"Item URL: {result_dict['item_url']}")
                else:
                    logging.error("Failed to publish to ArcGIS Online")
                    raise typer.Exit(1)
    
    except Exception as e:
        logging.error(f"Processing failed: {e}")
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
    # Initialize dump manager with environment-based configuration
    from .config.settings import Config
    temp_config = Config()  # This will load environment variables
    dump_manager = DumpManager(config=temp_config.dump)
    
    try:
        dumps = dump_manager.get_available_dumps()
        
        if not dumps:
            typer.echo("No local dumps found.")
            typer.echo("Use 'o2agol overture-dump <query> --download-only' to download dumps.")
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
    
    # Initialize dump manager with environment-based configuration
    from .config.settings import Config
    temp_config = Config()  # This will load environment variables
    dump_manager = DumpManager(config=temp_config.dump)
    
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