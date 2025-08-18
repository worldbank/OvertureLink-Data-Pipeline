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
    mode: str,
    limit: Optional[int],
    iso2: Optional[str],
    dry_run: bool,
    verbose: bool,
    use_divisions: bool = True,
    log_to_file: bool = False,
    country: Optional[str] = None,
    skip_cleanup: bool = False,
    export_geojson: Optional[str] = None,
):
    """
    Execute data processing pipeline for specified target with comprehensive logging.
    
    Supports both bounding box and Overture Divisions-based spatial filtering
    for precise country boundary adherence in enterprise deployments.
    
    Args:
        target_name: Data target identifier (roads, buildings, etc.)
        config: Path to YAML configuration file
        mode: Processing mode (initial, overwrite, append)
        limit: Optional feature limit for testing and development
        iso2: ISO2 country code override (legacy)
        dry_run: Execute validation without data publication
        verbose: Enable detailed logging output
        use_divisions: Use Overture Divisions for precise boundaries
        log_to_file: Create timestamped log files
        country: Country code/name for global config mode
        skip_cleanup: Skip temp file cleanup for debugging
        export_geojson: Export results as GeoJSON file instead of publishing to AGOL
    """
    # Validate parameter combinations
    if dry_run and export_geojson:
        logging.error("Cannot use both --dry-run and --export-geojson flags together")
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
    cmd_parts = ["python", "-m", "o2agol.cli", target_name, "-c", config]
    if mode != "auto":
        cmd_parts.extend(["--mode", mode])
    if limit:
        cmd_parts.extend(["--limit", str(limit)])
    if iso2:
        cmd_parts.extend(["--iso2", iso2])
    if country:
        cmd_parts.extend(["--country", country])
    if not use_divisions:
        cmd_parts.append("--use-bbox")
    if dry_run:
        cmd_parts.append("--dry-run")
    if verbose:
        cmd_parts.append("--verbose")
    if log_to_file:
        cmd_parts.append("--log-to-file")
    if export_geojson:
        cmd_parts.extend(["--export-geojson", export_geojson])
    
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

        # Handle GeoJSON export if requested
        if export_geojson:
            logging.info("=" * 50)
            logging.info("GEOJSON EXPORT PHASE")
            logging.info("=" * 50)
            
            if is_dual_query:
                export_to_geojson(normalized_data, export_geojson, target_name)
            else:
                export_to_geojson(gdf, export_geojson, target_name)
            
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


@app.command("roads")
def roads(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Processing mode: auto (smart detection) | initial | overwrite | append")] = "auto",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    iso2: Annotated[Optional[str], typer.Option("--iso2", help="ISO2 country code override (legacy)")] = None,
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name for global config (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without publishing")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    skip_cleanup: Annotated[bool, typer.Option("--skip-cleanup", help="Skip temp file cleanup for debugging")] = False,
    export_geojson: Annotated[Optional[str], typer.Option("--export-geojson", help="Export results as GeoJSON file instead of publishing to AGOL. Cannot be used with --dry-run.")] = None,
):
    """
    Process road network data from Overture Maps for publication to ArcGIS Online.

    Supports both country-specific configs and global config with --country parameter.
    Also supports both bounding box and Overture Divisions-based spatial filtering
    for precise country boundary adherence.
    
    Examples:
      Global config (recommended):
        python -m o2agol.cli roads -c configs/global.yml --country af --limit 1000
        python -m o2agol.cli roads -c configs/global.yml --country pak --dry-run
      
      Country-specific config (legacy):
        python -m o2agol.cli roads -c configs/afg.yml --limit 1000
        python -m o2agol.cli roads -c configs/afg.yml --dry-run
      
      Production deployment:
        python -m o2agol.cli roads -c configs/global.yml --country ind --use-divisions --log-to-file
    """
    process_target("roads", config, mode, limit, iso2, dry_run, verbose, use_divisions, log_to_file, country, skip_cleanup, export_geojson)


@app.command("buildings")
def buildings(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Processing mode: auto (smart detection) | initial | overwrite | append")] = "auto",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    iso2: Annotated[Optional[str], typer.Option("--iso2", help="ISO2 country code override (legacy)")] = None,
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name for global config (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without publishing")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    skip_cleanup: Annotated[bool, typer.Option("--skip-cleanup", help="Skip temp file cleanup for debugging")] = False,
    export_geojson: Annotated[Optional[str], typer.Option("--export-geojson", help="Export results as GeoJSON file instead of publishing to AGOL. Cannot be used with --dry-run.")] = None,
):
    """
    Process building footprint data from Overture Maps for publication to ArcGIS Online.

    Supports global config with --country parameter and bounding box/Divisions spatial filtering.
    Optimized for large-scale building datasets with enterprise-grade error handling.
    
    Examples:
      python -m o2agol.cli buildings -c configs/global.yml --country bd --limit 1000
    """
    process_target("buildings", config, mode, limit, iso2, dry_run, verbose, use_divisions, log_to_file, country, skip_cleanup, export_geojson)


@app.command("places")
def places(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Processing mode: auto (smart detection) | initial | overwrite | append")] = "auto",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    iso2: Annotated[Optional[str], typer.Option("--iso2", help="ISO2 country code override (legacy)")] = None,
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name for global config (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without publishing")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    skip_cleanup: Annotated[bool, typer.Option("--skip-cleanup", help="Skip temp file cleanup for debugging")] = False,
    export_geojson: Annotated[Optional[str], typer.Option("--export-geojson", help="Export results as GeoJSON file instead of publishing to AGOL. Cannot be used with --dry-run.")] = None,
):
    """
    Process points of interest data from Overture Maps for publication to ArcGIS Online.

    Supports global config with --country parameter for processing place categories.
    
    Examples:
      python -m o2agol.cli places -c configs/global.yml --country in --limit 500
    """
    process_target("places", config, mode, limit, iso2, dry_run, verbose, use_divisions, log_to_file, country, skip_cleanup, export_geojson)


@app.command("education")
def education(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Processing mode: auto (smart detection) | initial | overwrite | append")] = "auto",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    iso2: Annotated[Optional[str], typer.Option("--iso2", help="ISO2 country code override (legacy)")] = None,
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name for global config (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without publishing")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    skip_cleanup: Annotated[bool, typer.Option("--skip-cleanup", help="Skip temp file cleanup for debugging")] = False,
    export_geojson: Annotated[Optional[str], typer.Option("--export-geojson", help="Export results as GeoJSON file instead of publishing to AGOL. Cannot be used with --dry-run.")] = None,
):
    """
    Process education facilities from Overture Maps for publication to ArcGIS Online.

    Supports global config with --country parameter for educational institution mapping.
    
    Examples:
      python -m o2agol.cli education -c configs/global.yml --country ng --limit 100
    """
    process_target("education", config, mode, limit, iso2, dry_run, verbose, use_divisions, log_to_file, country, skip_cleanup, export_geojson)


@app.command("health")
def health(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Processing mode: auto (smart detection) | initial | overwrite | append")] = "auto",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    iso2: Annotated[Optional[str], typer.Option("--iso2", help="ISO2 country code override (legacy)")] = None,
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name for global config (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without publishing")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    skip_cleanup: Annotated[bool, typer.Option("--skip-cleanup", help="Skip temp file cleanup for debugging")] = False,
    export_geojson: Annotated[Optional[str], typer.Option("--export-geojson", help="Export results as GeoJSON file instead of publishing to AGOL. Cannot be used with --dry-run.")] = None,
):
    """
    Process health and medical facilities from Overture Maps for publication to ArcGIS Online.

    Supports global config with --country parameter for healthcare facility mapping.
    
    Examples:
      python -m o2agol.cli health -c configs/global.yml --country et --limit 50
    """
    process_target("health", config, mode, limit, iso2, dry_run, verbose, use_divisions, log_to_file, country, skip_cleanup, export_geojson)


@app.command("markets")
def markets(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="Processing mode: auto (smart detection) | initial | overwrite | append")] = "auto",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    iso2: Annotated[Optional[str], typer.Option("--iso2", help="ISO2 country code override (legacy)")] = None,
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name for global config (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without publishing")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    skip_cleanup: Annotated[bool, typer.Option("--skip-cleanup", help="Skip temp file cleanup for debugging")] = False,
    export_geojson: Annotated[Optional[str], typer.Option("--export-geojson", help="Export results as GeoJSON file instead of publishing to AGOL. Cannot be used with --dry-run.")] = None,
):
    """
    Process markets and retail establishments from Overture Maps for publication to ArcGIS Online.

    Supports global config with --country parameter for commercial facility mapping.
    
    Examples:
      python -m o2agol.cli markets -c configs/global.yml --country br --limit 200
    """
    process_target("markets", config, mode, limit, iso2, dry_run, verbose, use_divisions, log_to_file, country, skip_cleanup, export_geojson)


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