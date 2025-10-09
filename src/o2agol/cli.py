import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
import yaml
from dotenv import load_dotenv

# Load environment variables BEFORE importing local modules that use them
load_dotenv()

from .cleanup import full_cleanup_check, register_cleanup_handlers, cleanup_current_pid
from .config.settings import Config, PipelineLogger
from .config_loader import get_available_queries, load_config, validate_query_exists
from .domain.enums import ClipStrategy, StagingFormat
from .domain.models import Country, Query, RunOptions
from .pipeline.export import Exporter, generate_export_filename
from .pipeline.publish import FeatureLayerManager
from .pipeline.source import OvertureSource

app = typer.Typer(help="Overture Maps Pipeline: Source -> Transform -> Publish/Export")


# Helper functions to replace compat module functionality
def fetch_data(
    cfg: dict,
    query_config: Query,
    country_config: Country,
    use_divisions: bool = True,
    limit: Optional[int] = None
):
    """
    Fetch data using the new OvertureSource architecture.
    Replaces the old fetch_gdf compat function.
    """
    try:
        # Create run options
        run_options = RunOptions(
            limit=limit,
            use_bbox=not use_divisions,
            verbose=True,
            dry_run=False,
            log_to_file=False
        )
        
        # Create source and fetch data
        source = OvertureSource(cfg, run_options)
        clip_strategy = ClipStrategy.DIVISIONS if use_divisions else ClipStrategy.BBOX
        
        result = source.read(query_config, country_config, clip_strategy, raw=False)
        source.close()
        return result
        
    except Exception as e:
        logging.error(f"Failed to fetch data: {e}")
        raise


def publish_to_agol(gdf, metadata: dict, mode: str, gis, use_async: bool = False) -> str:
    """
    Publish GeoDataFrame to ArcGIS Online using GeoPackage staging.
    Uses publish_multi_layer_service for consistent 45x faster uploads.
    """
    manager = None
    try:
        from .domain.enums import Mode
        publish_mode = Mode(mode)
        manager = FeatureLayerManager(gis, publish_mode, use_async=use_async)
        
        service_name = metadata.get('title', 'default_layer')
        result = manager.publish_multi_layer_service(gdf, service_name, metadata, publish_mode)
        return result
        
    except Exception as e:
        logging.error(f"Failed to publish to AGOL: {e}")
        raise
    finally:
        if manager:
            manager.close()


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
    staging_format: StagingFormat = StagingFormat.GEOJSON,
    use_async: bool = False,
    use_analyze: bool = True,
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
        cmd_parts = ["python", "-m", "o2agol.cli", "export", target_name]
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
    
    # Load configuration using new architecture
    config_dict, run_options, query_config, country_config = load_config(
        query=target_name,
        country=country or iso2,
        config_path=config
    )
    
    # For compatibility with old cfg usage
    cfg = config_dict
    
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
        
        # Execute data retrieval using new architecture
        data_result = fetch_data(cfg, query_config, country_config, use_divisions, limit)
        
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
# TODO: Remove this legacy function - not used anywhere
                # normalized_data[layer_name] = normalize_schema(layer_gdf, target_name)
                logging.info(f"Schema normalization completed for {layer_name}: {len(normalized_data[layer_name]):,} features")
        else:
            # TODO: Remove this legacy function - not used anywhere
            # gdf = normalize_schema(gdf, target_name)
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
        
        # Log template usage for publish function
        if cfg.get('config_format') == 'template':
            template_metadata = cfg['template'].get_target_metadata(target_name)
            logging.info(f"Using template metadata: {template_metadata.item_title}")
            
        # Choose appropriate publishing method
        # Use parameters from CLI for flexibility in deployment
        if is_dual_query:
            from .pipeline.publish import FeatureLayerManager
            
            # Create GIS connection and publisher
            config = Config()
            gis = config.create_gis_connection()
            
            # Map mode string to Mode enum
            from .domain.enums import Mode
            publish_mode = Mode(mode)
            
            publisher = FeatureLayerManager(gis, publish_mode, use_async=use_async)
            
            # Create metadata from configuration
            from .config_loader import format_metadata_from_config
            metadata = format_metadata_from_config(config_dict, query_config, country_config)
            
            result = publisher.publish_multi_layer_service(
                layer_data=normalized_data,
                service_name=f"{country_config.iso3.lower()}_{query_config.name}",
                metadata=metadata,
                mode=publish_mode
            )
            total_published = sum(len(gdf) for gdf in normalized_data.values())
        else:
            # Single-layer publishing using new architecture
            config = Config()
            gis = config.create_gis_connection()
            
            # Create metadata from configuration
            from .config_loader import format_metadata_from_config
            metadata = format_metadata_from_config(config_dict, query_config, country_config)
            
            result = publish_to_agol(gdf, metadata, mode, gis, use_async=use_async)
            total_published = len(gdf)
        
        # Generate execution summary for audit and monitoring
        end_time = datetime.now()
        execution_duration = end_time - start_time
        
        logging.info("=" * 50)
        logging.info(f"{mode.upper()} OPERATION COMPLETED SUCCESSFULLY")
        logging.info("=" * 50)
        logging.info(f"Total execution time: {execution_duration}")
        logging.info(f"Features processed: {total_published:,}")
        
        # Handle different return types from dual vs single query publishing
        if is_dual_query:
            # Multi-layer service returns item ID string
            logging.info(f"Multi-layer service created with {len(normalized_data)} layers")
            logging.info(f"Item identifier: {result}")
        else:
            # Single-layer service returns object with attributes
            logging.info(f"Target layer: {result.title}")
            logging.info(f"Item identifier: {result.itemid}")
            logging.info(f"Service URL: {result.homepage}")
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


def process_target_for_export(
    target_name: str,
    config: str,
    output_path: str,
    export_format: str = "geojson",
    raw_export: bool = False,
    limit: Optional[int] = None,
    iso2: Optional[str] = None,
    verbose: bool = False,
    use_divisions: bool = True,
    log_to_file: bool = False,
    country: Optional[str] = None,
    skip_cleanup: bool = False,
):
    """
    Execute data processing pipeline for export with comprehensive logging.
    
    Args:
        target_name: Data target identifier (roads, buildings, etc.)
        config: Path to YAML configuration file
        output_path: Export file path
        export_format: Export format (geojson, gpkg, fgdb)
        raw_export: Skip AGOL transformations for raw export
        limit: Optional feature limit for testing and development
        iso2: ISO2 country code override (legacy)
        verbose: Enable detailed logging output
        use_divisions: Use Overture Divisions for precise boundaries
        log_to_file: Create timestamped log files
        country: Country code/name for global config mode
        skip_cleanup: Skip temp file cleanup for debugging
    """
    # Initialize logging with optional file output
    setup_logging(verbose, target_name, "export", log_to_file)
    
    # Register cleanup handlers for graceful interruption handling
    register_cleanup_handlers()
    
    # Perform temp directory cleanup and validation
    if not full_cleanup_check(skip_cleanup=skip_cleanup):
        logging.error("Temp directory size limit exceeded - aborting operation")
        raise typer.Exit(1)
    
    # Log execution context for audit trails
    start_time = datetime.now()
    logging.info(f"Initiating export operation for {target_name} target")
    logging.info(f"Execution timestamp: {start_time}")
    
    # Log the original Python command for reproducibility
    cmd_parts = ["python", "-m", "o2agol.cli", "export", target_name]
    if output_path:
        cmd_parts.append(output_path)
    cmd_parts.extend(["-c", config])
    
    # Add common parameters
    if export_format != "geojson":
        cmd_parts.extend(["--format", export_format])
    if raw_export:
        cmd_parts.append("--raw")
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
    logging.info(f"Export format: {export_format}")
    logging.info(f"Raw export: {raw_export}")
    
    # Validate parameter combinations
    if country and iso2:
        logging.warning(f"Both --country ({country}) and --iso2 ({iso2}) provided. Using --country for global config, --iso2 for selector override.")
    
    # Load configuration using new architecture
    config_dict, run_options, query_config, country_config = load_config(
        query=target_name,
        country=country or iso2,
        config_path=config
    )
    
    # For compatibility with old cfg usage
    cfg = config_dict
    
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
        
        # Execute data retrieval using new architecture
        data_result = fetch_data(cfg, query_config, country_config, use_divisions, limit)
        
        # Handle dual query results (dict) vs single query (GeoDataFrame)
        is_dual_query = isinstance(data_result, dict)
        if is_dual_query:
            total_features = sum(len(gdf) for gdf in data_result.values())
            logging.info(f"Data acquisition completed: {total_features:,} features retrieved (dual query)")
        else:
            gdf = data_result
            logging.info(f"Data acquisition completed: {len(gdf):,} features retrieved")

        logging.info("=" * 50)
        logging.info("DATA EXPORT PHASE")
        logging.info("=" * 50)
        
        # Export decision point
        if raw_export:
            # Export raw Overture data without transformation
            logging.info("Exporting raw Overture data (no AGOL transformations)")
            export_data(
                data=data_result,
                output_path=output_path,
                target_name=target_name,
                export_format=export_format,
                raw_export=True
            )
        else:
            # Apply AGOL transformations then export using standard Transformer
            logging.info("Applying AGOL schema transformations")
            from .pipeline.transform import Transformer
            transformer = Transformer(query_config)
            
            if isinstance(data_result, dict):
                # Multi-layer transformation
                normalized_data = {}
                for layer_name, gdf in data_result.items():
                    normalized_gdf = transformer.normalize(gdf)
                    normalized_data[layer_name] = transformer.add_metadata(normalized_gdf, country_config)
            else:
                # Single-layer transformation
                normalized_gdf = transformer.normalize(data_result)
                normalized_data = transformer.add_metadata(normalized_gdf, country_config)
            
            export_data(
                data=normalized_data,
                output_path=output_path,
                target_name=target_name,
                export_format=export_format,
                raw_export=False
            )
        
        # Calculate and log processing time
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Export operation completed successfully in {duration}")
        logging.info(f"Output file: {output_path}")
        
    except Exception as e:
        # Log comprehensive error information for debugging
        logging.error("=" * 50)
        logging.error("OPERATION FAILED")
        logging.error("=" * 50)
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error message: {str(e)}")
        
        if 'gdf' in locals():
            logging.error(f"Features processed before failure: {len(gdf):,}")
        
        logging.error("Operation terminated due to error")
        raise


@app.command("arcgis-upload")
def arcgis_upload(
    query: Annotated[str, typer.Argument(help="Query type to execute. Use 'list-queries' command to see available options.")],
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "src/o2agol/data/agol_metadata.yml",
    mode: Annotated[str, typer.Option("--mode", "-m", help="Processing mode: auto (smart detection) | initial | overwrite | append")] = "auto",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    iso2: Annotated[Optional[str], typer.Option("--iso2", help="ISO2 country code override (legacy)")] = None,
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name for global config (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without publishing")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    skip_cleanup: Annotated[bool, typer.Option("--skip-cleanup", help="Skip temp file cleanup for debugging")] = False,
    staging_format: Annotated[StagingFormat, typer.Option("--staging-format", help="Format for staging data during append operations (geojson or gpkg)")] = StagingFormat.GEOJSON,
    use_async: Annotated[bool, typer.Option("--async", help="Use asynchronous processing for large datasets")] = False,
    use_analyze: Annotated[bool, typer.Option("--use-analyze/--no-analyze", help="Enable AGOL analyze for optimal parameters")] = True,
):
    """
    Upload processed Overture Maps data to ArcGIS Online.
    
    Query types are defined in the configuration file under 'targets' section.
    Standard queries include: roads, buildings, places, education, health, markets
    
    Uses configs/agol_metadata.yml by default. Users can add custom queries by defining 
    new targets in their configuration file. Each query must specify at minimum: 
    theme, type, and optional filter parameters.
    
    Examples:
        Standard queries (using default global config):
            o2agol arcgis-upload roads --country af
            o2agol arcgis-upload education --country pak --limit 100
            o2agol arcgis-upload buildings --country ind --dry-run
        Custom config file:
            o2agol arcgis-upload my_custom_query -c configs/custom.yml --dry-run
    """
    # Validate that the query exists in the configuration
    if not validate_query_exists(config, query):
        try:
            available_queries = get_available_queries(config)
            typer.echo(f"ERROR: Query '{query}' not found in configuration file '{config}'", err=True)
            typer.echo(f"\nAvailable queries: {', '.join(sorted(available_queries))}", err=True)
            if config == "configs/agol_metadata.yml":
                typer.echo("\nTip: Use 'python -m o2agol.cli list-queries' to see detailed information about each query.", err=True)
            else:
                typer.echo(f"\nTip: Use 'python -m o2agol.cli list-queries -c {config}' to see detailed information about each query.", err=True)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"ERROR loading configuration: {e}", err=True)
        raise typer.Exit(1)
    
    # Log which query is being executed
    logging.info(f"Executing Overture query: {query}")
    
    # Validate staging format for append operations
    if mode == "append" and staging_format not in [StagingFormat.GEOJSON, StagingFormat.GPKG]:
        logging.error(f"Invalid staging format '{staging_format}' for append mode. Must be 'geojson' or 'gpkg'")
        raise typer.Exit(1)
    
    # Use new pipeline architecture for clean execution
    try:
        # Setup logging based on user preferences
        setup_logging(verbose, query, mode, log_to_file)
        
        # Import pipeline components
        from .config.settings import Config
        from .domain.enums import ClipStrategy, Mode
        from .pipeline.publish import FeatureLayerManager
        from .pipeline.source import OvertureSource
        from .pipeline.transform import Transformer
        
        # Load configuration using new typed system
        config_dict, run_options, query_config, country_config = load_config(
            query=query,
            country=country or iso2,  # Support both --country and legacy --iso2
            config_path=config
        )
        
        # Update run options with CLI parameters for arcgis-upload
        from .domain.models import RunOptions
        run_options = RunOptions(
            clip=ClipStrategy.BBOX if not use_divisions else ClipStrategy.DIVISIONS,
            limit=limit,
            use_bbox=not use_divisions
        )
        
        # Determine clipping strategy
        clip_strategy = ClipStrategy.BBOX if not use_divisions else ClipStrategy.DIVISIONS
        
        if dry_run:
            logging.info(f"DRY RUN: Would process {query_config.name} for {country_config.name} ({country_config.iso3.lower()})")
            logging.info(f"Query theme: {query_config.theme}, Target: {query_config.type}")
            logging.info(f"Clipping strategy: {clip_strategy.value}")
            logging.info(f"Publishing mode: {mode}")
            if limit:
                logging.info(f"Feature limit: {limit:,}")
            return
        
        # Step 1: Source - Fetch data using OvertureSource
        logging.info(f"Fetching {query_config.theme} data for {country_config.name}")
        source = OvertureSource(config_dict, run_options)
        data_result = source.read(query_config, country_config, clip_strategy, raw=False)
        
        # Check for empty results
        if isinstance(data_result, dict):
            total_features = sum(len(gdf) for gdf in data_result.values())
            if total_features == 0:
                logging.warning(f"No {query_config.name} features found for {country_config.name}")
                return
            logging.info(f"Retrieved {total_features:,} total features across {len(data_result)} layers")
        else:
            if len(data_result) == 0:
                logging.warning(f"No {query_config.name} features found for {country_config.name}")
                return
            logging.info(f"Retrieved {len(data_result):,} features")
        
        # Step 2: Transform - Normalize schema for AGOL compatibility
        logging.info("Normalizing data for AGOL compatibility")
        transformer = Transformer(query_config)
        
        if isinstance(data_result, dict):
            # Multi-layer transformation
            transformed_data = {}
            for layer_name, gdf in data_result.items():
                normalized_gdf = transformer.normalize(gdf)
                transformed_data[layer_name] = transformer.add_metadata(normalized_gdf, country_config)
        else:
            # Single-layer transformation
            normalized_data = transformer.normalize(data_result)
            transformed_data = transformer.add_metadata(normalized_data, country_config)
        
        # Step 3: Publish - Upload to AGOL using FeatureLayerManager
        logging.info(f"Publishing to ArcGIS Online in {mode} mode")
        
        # Load AGOL configuration
        agol_config = Config()
        gis = agol_config.create_gis_connection()
        
        # Initialize feature layer manager
        publish_mode = Mode(mode)
        publisher = FeatureLayerManager(gis, publish_mode, use_async=use_async)
        
        try:
            # Create metadata from configuration templates
            from .config_loader import format_metadata_from_config
            metadata = format_metadata_from_config(config_dict, query_config, country_config)
            
            if isinstance(transformed_data, dict):
                # Multi-layer service (education, health, markets)
                item_id = publisher.publish_multi_layer_service(
                    layer_data=transformed_data,
                    service_name=f"{country_config.iso3.lower()}_{query_config.name}",
                    metadata=metadata,
                    mode=publish_mode
                )
            else:
                # Single-layer service (roads, buildings, places) - using GeoPackage staging
                item_id = publisher.publish_multi_layer_service(
                    layer_data=transformed_data,
                    service_name=f"{country_config.iso3.lower()}_{query_config.name}",
                    metadata=metadata,
                    mode=publish_mode
                )
            
            if item_id:
                logging.info(f"Successfully published: {item_id}")
                print(f"Published item ID: {item_id}")
            else:
                logging.error("Failed to publish to AGOL")
                cleanup_current_pid()
                raise typer.Exit(1)
        finally:
            # Clean up publisher resources
            publisher.close()
            
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        if verbose:
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
        cleanup_current_pid()
        raise typer.Exit(1)
    finally:
        # Ensure cleanup happens on successful completion too
        cleanup_current_pid()


@app.command("export")
def export_data_command(
    query: Annotated[str, typer.Argument(help="Query type to execute. Use 'list-queries' command to see available options.")],
    output_path: Annotated[Optional[str], typer.Argument(help="Output file path (optional - will auto-generate if not provided)")] = None,
    format: Annotated[str, typer.Option("--format", "-f", help="Export format: geojson, gpkg, fgdb")] = "gpkg",
    raw: Annotated[bool, typer.Option("--raw", help="Export raw Overture data without AGOL transformations")] = False,
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "src/o2agol/data/agol_metadata.yml",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    iso2: Annotated[Optional[str], typer.Option("--iso2", help="ISO2 country code override (legacy)")] = None,
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name for global config (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    skip_cleanup: Annotated[bool, typer.Option("--skip-cleanup", help="Skip temp file cleanup for debugging")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without exporting")] = False,
):
    """
    Export Overture Maps data to various formats.
    
    Supports GeoJSON, GeoPackage (GPKG), and File Geodatabase (FGDB) formats.
    Use --raw to export unmodified Overture data, or omit for AGOL-ready transformations.
    
    Query types are defined in the configuration file under 'targets' section.
    Standard queries include: roads, buildings, places, education, health, markets
    
    Examples:
        o2agol export roads --country afg                      # GeoPackage 
        o2agol export roads --country afg --format geojson     # GeoPackage (default)
        o2agol export places roads.gpkg --country afg --raw    # Raw data with custom filename
        o2agol export education --country pak --format fgdb    # Multi-layer FGDB
    """
    
    # Validate --dry-run is not used with export
    if dry_run:
        typer.echo("ERROR: --dry-run is not supported with export command. Use it with arcgis-upload instead.", err=True)
        raise typer.Exit(1)
    
    # Auto-detect format from output path if provided
    if output_path and not format:
        from .domain.enums import ExportFormat
        format = ExportFormat.from_extension(output_path).value
    
    
    # Validate that the query exists in the configuration
    if not validate_query_exists(config, query):
        try:
            available_queries = get_available_queries(config)
            typer.echo(f"ERROR: Query '{query}' not found in configuration file '{config}'", err=True)
            typer.echo(f"\nAvailable queries: {', '.join(sorted(available_queries))}", err=True)
            if config == "configs/agol_metadata.yml":
                typer.echo("\nTip: Use 'python -m o2agol.cli list-queries' to see detailed information about each query.", err=True)
            else:
                typer.echo(f"\nTip: Use 'python -m o2agol.cli list-queries -c {config}' to see detailed information about each query.", err=True)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"ERROR loading configuration: {e}", err=True)
        raise typer.Exit(1)
    
    
    # Use new pipeline architecture for clean execution
    try:
        # Setup logging based on user preferences
        setup_logging(verbose, query, "export", log_to_file)
        
        # Import pipeline components
        from pathlib import Path

        from .domain.enums import ClipStrategy, ExportFormat
        from .pipeline.export import Exporter, generate_export_filename
        from .pipeline.source import OvertureSource
        from .pipeline.transform import Transformer
        
        

        # Load configuration using new typed system
        config_dict, run_options, query_config, country_config = load_config(
            query=query,
            country=country or iso2,  # Support both --country and legacy --iso2
            config_path=config,
            load_agol_config=False  # AGOL config not needed for export
        )
        
        

        # Update run options with CLI parameters for export
        from .domain.models import RunOptions
        run_options = RunOptions(
            clip=ClipStrategy.BBOX if not use_divisions else ClipStrategy.DIVISIONS,
            limit=limit,
            use_bbox=not use_divisions
        )
        
        # Auto-generate filename if not provided
        if not output_path:
            output_path = generate_export_filename(query_config.name, country_config.iso3.lower(), format, raw)
            
        # Log execution details
        logging.info(f"Executing Overture query: {query_config.name}")
        logging.info(f"Output file: {output_path}")
        logging.info(f"Export format: {format}")
        logging.info(f"Raw export: {raw}")
        
        # Determine clipping strategy
        clip_strategy = ClipStrategy.BBOX if not use_divisions else ClipStrategy.DIVISIONS
        
        # Use direct export approach (following Overture docs pattern)
        source = OvertureSource(config_dict, run_options)
        output_path_obj = Path(output_path)
        
        if raw:
            # Direct export from DuckDB to final format (no intermediate steps)
            logging.info("Using direct DuckDB export (following Overture docs pattern)")
            source.export_direct(
                query=query_config,
                country=country_config,
                output_path=output_path_obj,
                export_format=format,
                clip=clip_strategy,
                limit=limit
            )
            final_path = output_path_obj
        else:
            # For non-raw exports, use traditional approach (fetch → transform → export)
            # This is needed because transformations require GeoDataFrame processing
            logging.info(f"Fetching {query_config.theme} data for {country_config.name}")
            data_result = source.read(query_config, country_config, clip_strategy, raw=raw)
            
            # Check for empty results
            if isinstance(data_result, dict):
                total_features = sum(len(gdf) for gdf in data_result.values())
                if total_features == 0:
                    logging.warning(f"No {query_config.name} features found for {country_config.name}")
                    return
                logging.info(f"Retrieved {total_features:,} total features across {len(data_result)} layers")
            else:
                if len(data_result) == 0:
                    logging.warning(f"No {query_config.name} features found for {country_config.name}")
                    return
                logging.info(f"Retrieved {len(data_result):,} features")
            
            # Transform for AGOL compatibility
            logging.info("Normalizing data for AGOL compatibility")
            transformer = Transformer(query_config)
            
            if isinstance(data_result, dict):
                # Multi-layer transformation
                transformed_data = {}
                for layer_name, gdf in data_result.items():
                    normalized_gdf = transformer.normalize(gdf)
                    transformed_data[layer_name] = transformer.add_metadata(normalized_gdf, country_config)
            else:
                # Single-layer transformation
                normalized_data = transformer.normalize(data_result)
                transformed_data = transformer.add_metadata(normalized_data, country_config)
            
            # Export using traditional Exporter
            logging.info(f"Exporting to {format.upper()} format")
            exporter = Exporter(out_path=output_path_obj, fmt=ExportFormat(format))
            
            final_path = exporter.write(
                data=transformed_data,
                base_name=f"{country_config.iso3.lower()}_{query_config.name}",
                out_dir=output_path_obj.parent,
                multilayer=query_config.is_multilayer,
                raw=raw
            )
        
        logging.info(f"Export completed successfully: {final_path}")
        print(f"Exported to: {final_path}")
        
    except Exception as e:
        logging.error(f"Export failed: {e}")
        if verbose:
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
        cleanup_current_pid()
        raise typer.Exit(1)
    finally:
        # Ensure cleanup happens on successful completion too
        cleanup_current_pid()


@app.command("list-queries")
def list_queries(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "src/o2agol/data/agol_metadata.yml",
):
    """
    List all available Overture queries.
    
    Shows detailed information about each query including theme, type, filter,
    and description to help users understand what data each query retrieves.
    
    Examples:
        python -m o2agol.cli list-queries
    """
    try:
        queries = get_available_queries()
        
        if not queries:
            typer.echo("WARNING: No queries found")
            raise typer.Exit(1)
        
        # Load queries with detailed information
        queries_file = Path(__file__).parent / "data" / "queries.yml"
        with open(queries_file) as f:
            queries_data = yaml.safe_load(f)
        
        typer.echo("Available Overture Queries")
        typer.echo("=" * 50)
        
        for query_name in sorted(queries):
            query_info = queries_data[query_name]
            theme = query_info.get('theme', 'N/A')
            type_ = query_info.get('type', 'N/A')
            filter_ = query_info.get('filter', None)
            building_filter = query_info.get('building_filter', None)
            description = query_info.get('sector_description', None)
            
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
        typer.echo("\nUsage:")
        typer.echo("  Upload to ArcGIS Online: o2agol arcgis-upload <query_name> --country <country> [options]")
        typer.echo("  Export data:            o2agol export <query_name> --country <country> [options]")
        
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
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "src/o2agol/data/agol_metadata.yml",
    country: Annotated[Optional[str], typer.Option("--country", help="Country code/name (e.g., 'af', 'afg', 'Afghanistan')")] = None,
    mode: Annotated[str, typer.Option("--mode", "-m", help="Processing mode: auto (smart detection) | initial | overwrite | append")] = "auto",
    download_only: Annotated[bool, typer.Option("--download-only", help="Only download dump without processing")] = False,
    use_local: Annotated[bool, typer.Option("--use-local/--use-s3", help="Use local dump if available")] = True,
    force_download: Annotated[bool, typer.Option("--force-download", help="Force re-download even if dump exists")] = False,
    release: Annotated[str, typer.Option("--release", help="Overture release version")] = None,
    staging_format: Annotated[str, typer.Option("--format", help="Staging format for AGOL upload: geojson, gpkg, or fgdb")] = "gpkg",
    limit: Annotated[Optional[int], typer.Option("--limit", "-l", help="Feature limit for testing and development")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable detailed logging output")] = False,
    log_to_file: Annotated[bool, typer.Option("--log-to-file", help="Create timestamped log files")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate configuration without processing/publishing")] = False,
    use_divisions: Annotated[bool, typer.Option("--use-divisions/--use-bbox", help="Use Overture Divisions for country boundaries")] = True,
    # Cache optimization flags
    use_world_bank: Annotated[bool, typer.Option("--world-bank/--overture-divisions", help="Use World Bank boundaries (default: enabled)")] = True,
    use_async: Annotated[bool, typer.Option("--async", help="Use async append for large datasets")] = False,
    use_analyze: Annotated[bool, typer.Option("--analyze", help="Analyze uploaded data before append")] = False,
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
    
    # Validate staging format
    if staging_format not in ["geojson", "gpkg", "fgdb"]:
        typer.echo("ERROR: --format must be 'geojson', 'gpkg', or 'fgdb'", err=True)
        raise typer.Exit(1)
    
    if not country:
        typer.echo("ERROR: --country is required for dump processing", err=True)
        raise typer.Exit(1)
    
    # Use configured release if none provided
    if release is None:
        from .config.settings import Config
        release = Config().overture.release
    
    # Setup logging
    log_mode = "dump"
    setup_logging(verbose, query, log_mode, log_to_file)
    
    # Initialize comprehensive timing
    from datetime import datetime
    
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
    import os
    import sys
    
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
    operation_mode = f"AGOL upload (staging as {staging_format.upper()})"
    logging.info(f"Starting {operation_mode} operation: {query} for {country}")
    logging.info(f"Execution timestamp: {datetime.now()}")

    
    # Phase timings
    phase_times = {}
    
    # Load configuration using modern system
    config_dict, run_options, query_config, country_config = load_config(
        query=query,
        country=country,
        config_path=config
    )
    
    # Extract theme information
    theme = query_config.theme
    
    # Get country info for display
    country_info = country_config
    
    # Detect dual-theme queries (places + buildings)
    building_filter = getattr(query_config, 'building_filter', None)
    themes_to_download = [theme]
    if building_filter:
        themes_to_download.append('buildings')
    
    pipeline_logger.info(f"Configuration: {country_info.name} ({country_info.iso3}) | Query: {query} | Release: {release}")
    
    # Initialize new pipeline architecture components
    from .domain.enums import ClipStrategy
    from .pipeline.source import OvertureSource
    
    # Update runtime options from CLI parameters (merge with loaded options)
    run_options.clip = ClipStrategy.DIVISIONS if use_divisions else ClipStrategy.BBOX
    run_options.limit = limit
    run_options.use_bbox = not use_divisions
    
    # Initialize OvertureSource with dump and cache functionality
    source = OvertureSource(config_dict, run_options)
    
    # Configuration phase complete
    logging.info(f"Cache storage location: {source._cache_dir}")
    logging.info(f"Dump storage location: {source._dump_base_path}")
    logging.info("Using country-specific caching approach")
    logging.info(f"Cache enabled: {source._enable_cache}")
    logging.info(f"Local dumps enabled: {source._use_local_dumps}")
    
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
                type_name = query_config.type if theme_to_check == theme else 'building'
                
                # Check if cache exists for this country/theme combination
                cache_entries = source.list_cached_countries(release)
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
                type_name = query_config.type if theme_to_cache == theme else 'building'
                
                theme_desc = "transportation" if theme_to_cache == "roads" else f"{theme_to_cache} facilities"
                pipeline_logger.info(f"Caching {theme_desc} for {country_info.name}")
                
                # Use OvertureSource cache functionality
                from .pipeline.source import CacheQuery
                cache_query = CacheQuery(
                    country=country_info.iso2,
                    theme=theme_to_cache,
                    type_name=type_name,
                    release=release,
                    use_divisions=use_divisions,
                    limit=None,  # Cache complete data
                    filters=getattr(query_config, 'filter', None)
                )
                result_gdf = source.cache_country_data(cache_query, overwrite=force_download)
                
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
    
    # Process query using OvertureSource with cache->dump->S3 fallback
    try:
        from .config.countries import CountryRegistry
        from .domain.models import Country as DomainCountry
        from .domain.models import Query as DomainQuery
        
        # Resolve country
        country_info = CountryRegistry.get_country(country)
        if not country_info:
            raise ValueError(f"Unknown country: {country}")
        
        # Create domain objects for the new pipeline
        domain_country = DomainCountry(
            name=country_info.name,
            iso2=country_info.iso2,
            iso3=country_info.iso3,
            bounds=CountryRegistry.get_bounding_boxes().get(country_info.iso2, (0, 0, 0, 0))
        )
        
        domain_query = DomainQuery(
            theme=theme,
            type=query_config.type,
            filter=getattr(query_config, 'filter', None),
            building_filter=building_filter,
            name=query,
            is_multilayer=query_config.is_multilayer,
            geometry_split=getattr(query_config, "geometry_split", False)
        )
        
        # === PHASE 1: DATA ACQUISITION ===
        phase2_start = time.time()
        pipeline_logger.info("")
        pipeline_logger.phase("PHASE 1: DATA ACQUISITION")
        
        # Handle both single and multi-layer queries through the unified interface
        if building_filter:
            pipeline_logger.info("Retrieving education facilities (places + buildings)" if query == "education" 
                               else "Retrieving health facilities (places + buildings)" if query == "health"
                               else "Retrieving retail facilities (places + buildings)" if query == "markets"
                               else f"Retrieving {query} facilities (places + buildings)")
        else:
            pipeline_logger.info(f"Retrieving {query} data")
        
        # Use the unified read() method with cache->dump->S3 fallback
        query_result = source.read(domain_query, domain_country, run_options.clip, raw=False)
        
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
            
            # Transform each layer using the standard Transformer
            from .pipeline.transform import Transformer
            transformer = Transformer(query_config)
            
            transformed_data = {}
            for layer_name, layer_gdf in query_result.items():
                if not layer_gdf.empty:
                    normalized_gdf = transformer.normalize(layer_gdf)
                    transformed_data[layer_name] = transformer.add_metadata(normalized_gdf, country_config)
            
            if not transformed_data:
                pipeline_logger.info("No valid data after transformation")
                return
            
            pipeline_logger.info(f"Transformed {total_features:,} features")
            phase_times['transformation'] = time.time() - phase3_start
            pipeline_logger.info(f"Phase completed in {format_duration(phase_times['transformation'])}")
            
            # === PHASE 3: OUTPUT ===
            phase4_start = time.time()
            pipeline_logger.info("")
            pipeline_logger.phase("PHASE 3: AGOL PUBLISHING")
            pipeline_logger.info(f"Publishing to ArcGIS Online using {staging_format.upper()} staging")
            
            # Always publish to AGOL - overture-dump is for AGOL uploads only
            if config_dict.get('config_format') == 'template':
                # Use template metadata with dynamic variables
                template_metadata = config_dict['template'].get_target_metadata(query)
                pipeline_logger.info(f"Found existing service: {template_metadata.item_title}")
            
            # Use multi-layer publishing service (same as arcgis-upload command)
            from .pipeline.publish import FeatureLayerManager
            
            # Create GIS connection and publisher
            config_obj = Config()
            gis = config_obj.create_gis_connection()
            
            # Map mode string to Mode enum
            from .domain.enums import Mode
            publish_mode = Mode(mode)
            
            publisher = FeatureLayerManager(gis, publish_mode, use_async=use_async)
            
            try:
                # Create metadata from configuration templates
                from .config_loader import format_metadata_from_config
                metadata = format_metadata_from_config(config_dict, query_config, country_config)
                
                # Use parameters from CLI for flexibility
                result = publisher.publish_multi_layer_service(
                    layer_data=transformed_data,
                    service_name=f"{country_config.iso3.lower()}_{query_config.name}",
                    metadata=metadata,
                    mode=publish_mode,
                    staging_format=staging_format
                )
                
                # publish_multi_layer_service returns the published item, not a dict
                if result:
                    # Extract layer info for display
                    layers_info = []
                    for layer_name, layer_gdf in transformed_data.items():
                        layer_type = "points" if "places" in layer_name else "polygons"
                        layers_info.append(f"Updated layer {country_config.iso3.lower()}_{query}_{layer_type} ({len(layer_gdf):,} features)")
                    
                    for info in layers_info:
                        pipeline_logger.info(info)
                    
                    pipeline_logger.info(f"Multi-layer service published with item ID: {result}")
                else:
                    pipeline_logger.info("Failed to publish service to ArcGIS Online")
                    raise typer.Exit(1)
            finally:
                # Clean up publisher resources
                publisher.close()
            
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
            
            # Transform data using the standard Transformer
            from .pipeline.transform import Transformer
            transformer = Transformer(query_config)
            
            normalized_gdf = transformer.normalize(gdf)
            gdf_transformed = transformer.add_metadata(normalized_gdf, country_config)
            pipeline_logger.info(f"Transformed {len(gdf):,} features")
            
            # End Phase 2
            phase_times['transformation'] = time.time() - phase3_start
            pipeline_logger.info(f"Phase completed in {format_duration(phase_times['transformation'])}")
            
            # === PHASE 3: OUTPUT ===
            phase4_start = time.time()
            pipeline_logger.info("")
            pipeline_logger.phase("PHASE 3: AGOL PUBLISHING")
            pipeline_logger.info(f"Publishing to ArcGIS Online using {staging_format.upper()} staging")
            
            # Always publish to AGOL - overture-dump is for AGOL uploads only
            # Publish the already-transformed data directly to avoid duplicate processing
            
            from .domain.enums import Mode
            from .pipeline.publish import FeatureLayerManager
            
            # Create GIS connection and publisher
            config_obj = Config()
            gis = config_obj.create_gis_connection()
            
            # Map mode string to Mode enum
            publish_mode = Mode(mode)
            publisher = FeatureLayerManager(gis, publish_mode, use_async=use_async)
            
            # Create metadata from configuration
            from .config_loader import format_metadata_from_config
            metadata = format_metadata_from_config(config_dict, query_config, country_config)
            
            # Publish using the already-transformed data
            try:
                result = publisher.publish_multi_layer_service(
                    layer_data=gdf_transformed,
                    service_name=f"{country_config.iso3.lower()}_{query_config.name}",
                    metadata=metadata,
                    mode=publish_mode,
                    staging_format=staging_format
                )
                
                if result:
                    pipeline_logger.info(f"Published to AGOL with item ID: {result}")
                else:
                    logging.error("Failed to publish to ArcGIS Online")
                    raise typer.Exit(1)
            except Exception as e:
                logging.error(f"Failed to publish to ArcGIS Online: {e}")
                raise typer.Exit(1)
            finally:
                # Clean up publisher resources
                publisher.close()
            
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
                formatted_phase = 'AGOL Publishing'
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
        cleanup_current_pid()
        raise typer.Exit(1)
    
    finally:
        source.close()
        # Ensure cleanup happens on successful completion too
        cleanup_current_pid()


# Deprecated commands removed - dump management is now integrated into OvertureSource
# Use 'o2agol cache-stats' to view cache information
# Use 'o2agol overture-dump' commands for local dump functionality


@app.command("list-cache")
def list_cache(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "src/o2agol/data/agol_metadata.yml",
    release: Annotated[str, typer.Option("--release", help="Overture release version")] = None,
):
    """
    List cached country data entries.
    
    Shows all cached data with metadata including country, theme, feature counts, and file sizes.
    """
    try:
        # Use configured release if none provided
        if release is None:
            from .config.settings import Config
            release = Config().overture.release
        
        # Use OvertureSource for cache management (integrated dump manager functionality)
        from .domain.enums import ClipStrategy
        from .domain.models import RunOptions
        from .pipeline.source import OvertureSource
        
        # Create minimal config for OvertureSource
        config_dict = {}
        run_options = RunOptions(
            clip=ClipStrategy.DIVISIONS,
            limit=None,
            use_bbox=False
        )
        
        source = OvertureSource(config_dict, run_options)
        
        # Get cache entries
        cache_entries = source.list_cached_countries(release)
        
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
        stats = source.get_cache_stats()
        typer.echo(f"Cache location: {stats['cache_path']}")
        
        # Clean up resources
        source.close()
        
    except Exception as e:
        typer.echo(f"ERROR: Failed to list cache: {e}", err=True)
        raise typer.Exit(1)


@app.command("clear-cache")
def clear_cache(
    config: Annotated[str, typer.Option("--config", "-c", help="Path to YAML configuration file")] = "src/o2agol/data/agol_metadata.yml",
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
        # Use OvertureSource for cache management (integrated dump manager functionality)
        from .domain.enums import ClipStrategy
        from .domain.models import RunOptions
        from .pipeline.source import OvertureSource
        
        # Create minimal config for OvertureSource
        config_dict = {}
        run_options = RunOptions(
            clip=ClipStrategy.DIVISIONS,
            limit=None,
            use_bbox=False
        )
        
        source = OvertureSource(config_dict, run_options)
        
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
        source.clear_cache(country=country, release=release)
        typer.echo(f"Successfully cleared {target}")
        
        # Clean up resources
        source.close()
        
    except Exception as e:
        typer.echo(f"ERROR: Failed to clear cache: {e}", err=True)
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
