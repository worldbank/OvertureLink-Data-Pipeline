from __future__ import annotations

import logging
import os
from pathlib import Path
import time
import tempfile
import platform

import duckdb
import geopandas as gpd
import pandas as pd
import shapely.wkb as swkb

from .config import Config

# Country bounding boxes for efficient spatial filtering
# Eliminates expensive polygon-based country boundary lookups
COUNTRY_BBOXES = {
    'AF': (60.5, 29.4, 74.9, 38.5),   # Afghanistan - Corrected coordinates
    'PK': (60.9, 23.6, 77.8, 37.1),   # Pakistan
    'BD': (88.0, 20.7, 92.7, 26.6),   # Bangladesh  
    'IN': (68.1, 6.7, 97.4, 37.1),    # India
}


def _parquet_url(secure_config: Config, theme: str, type_: str) -> str:
    """Return proper parquet URL for S3 storage."""
    # Use the new config system's Overture settings
    return f"{secure_config.overture.base_url}/theme={theme}/type={type_}/*.parquet"


def _parquet_url_from_config(overture_config: dict, theme: str, type_: str) -> str:
    """Return proper parquet URL using unified Overture configuration."""
    return f"{overture_config['base_url']}/theme={theme}/type={type_}/*.parquet"


def _to_gdf(tbl) -> gpd.GeoDataFrame:
    """Convert Arrow table to GeoDataFrame with optimized geometry handling."""
    df = tbl.to_pandas(types_mapper=pd.ArrowDtype)
    if "geom_wkb" not in df.columns:
        raise ValueError("Expected column 'geom_wkb' missing from result")
    df["geometry"] = df["geom_wkb"].apply(
        lambda b: swkb.loads(bytes(b)) if b is not None else None
    )
    df = df.drop(columns=["geom_wkb"])
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def setup_duckdb_optimized(secure_config: Config) -> duckdb.DuckDBPyConnection:
    """Configure DuckDB with performance optimizations for large-scale geospatial processing."""
    con = duckdb.connect()
    
    # Install required extensions
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # S3 configuration using new config system
    con.execute(f"SET s3_region='{secure_config.overture.s3_region}';")
    
    # Get DuckDB settings from new config system
    duckdb_settings = secure_config.get_duckdb_settings()
    con.execute(f"SET memory_limit='{duckdb_settings['memory_limit']}';")
    con.execute(f"SET threads={duckdb_settings['threads']};")
    
    # Extended timeout for long-distance network connections
    con.execute("SET http_timeout=1800000;")  # 30 minutes
    
    # Enable progress reporting
    os.environ['DUCKDB_PROGRESS_BAR'] = '1'
    
    logging.info(f"DuckDB configured: {duckdb_settings['threads']} threads, {duckdb_settings['memory_limit']} memory, extended timeouts")
    return con


def fetch_gdf(config_obj, target_name: str, **kwargs) -> gpd.GeoDataFrame:
    """
    Fetch geospatial data with automatic retry logic for network resilience.
    
    Args:
        config_obj: Configuration object (adapted from CLI integration)
        target_name: Target data type (roads, buildings, places)
        **kwargs: Additional parameters including limit, use_divisions, etc.
        
    Returns:
        GeoDataFrame containing requested geospatial data
    """
    max_retries = kwargs.get('max_retries', 3)
    
    for attempt in range(max_retries):
        try:
            return _fetch_gdf_attempt(config_obj, target_name, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"All {max_retries} attempts failed")
                raise
            
            # If divisions query fails, fall back to bbox filtering
            if "divisions" in str(e).lower() or "country" in str(e).lower():
                logging.warning(f"Divisions query failed: {e}")
                logging.warning("Falling back to bbox filtering for retry...")
                kwargs['use_divisions'] = False
            
            wait_time = 300 * (attempt + 1)  # Exponential backoff: 5, 10, 15 minutes
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            logging.warning(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


def _extract_config_data(config_obj) -> tuple:
    """
    Extract configuration data from config object (handles CLI adapter format).
    
    Returns:
        Tuple of (secure_config, selector_dict, target_dict, overture_config)
    """
    if hasattr(config_obj, 'yaml_config'):
        # This is the CLI adapter format with unified Overture config
        secure_config = config_obj.secure_config
        overture_config = config_obj.overture_config
        selector = config_obj.selector.__dict__ if config_obj.selector else {}
        
        # Simple extraction - same as before
        targets = {name: getattr(target, '__dict__', target) for name, target in config_obj.targets.items()}
        
        # Quick debug to see what we actually get
        print(f"SIMPLE DEBUG: targets = {targets}")
        print(f"SIMPLE DEBUG: raw target = {dict(config_obj.targets['roads'].__dict__)}, vars = {vars(config_obj.targets['roads'])}")
            
        return secure_config, selector, targets, overture_config
    else:
        # Fallback for other formats
        raise ValueError("Unsupported config object format")


def _fetch_gdf_attempt(config_obj, target_name: str, **kwargs) -> gpd.GeoDataFrame:
    """Execute single data fetch attempt with intelligent spatial filtering selection."""
    logging.info("Starting optimized data fetch...")
    
    # Extract configuration data with unified Overture config
    secure_config, selector, targets, overture_config = _extract_config_data(config_obj)
    con = setup_duckdb_optimized(secure_config)
    
    try:
        target_config = targets[target_name]
        limit = kwargs.get('limit')
        use_divisions = kwargs.get('use_divisions', True)
        
        # Log the Overture config being used
        logging.info(f"Using Overture release: {overture_config['release']}")
        logging.info(f"Target theme: {target_config.get('theme')}, type: {target_config.get('type')}")
        
        if use_divisions:
            # Use precise Overture Divisions for country boundaries
            logging.info("Using Overture Divisions for precise country boundaries")
            sql = _build_divisions_query(overture_config, target_name, target_config, selector, limit)
        else:
            # Fall back to bbox filtering
            iso2 = selector.get('iso2', 'AF').upper()
            if iso2 not in COUNTRY_BBOXES:
                raise ValueError(f"No bounding box defined for country {iso2}. Add to COUNTRY_BBOXES in duck.py")
            
            xmin, ymin, xmax, ymax = COUNTRY_BBOXES[iso2]
            logging.info(f"Using bbox for {iso2}: ({xmin}, {ymin}, {xmax}, {ymax})")
            
            parquet_url = _parquet_url_from_config(overture_config, target_config.get('theme'), target_config.get('type'))
            sql = _build_optimized_query(parquet_url, xmin, ymin, xmax, ymax, target_config, limit)
        
        logging.info("Executing optimized query...")
        logging.info(f"SQL preview: {sql[:200]}...")
        start_time = time.time()
        
        # Always use temp file strategy for consistent, predictable behavior
        gdf = _fetch_via_temp_file(con, sql)
        
        elapsed = time.time() - start_time
        logging.info(f"Fetched {len(gdf):,} features in {elapsed:.1f} seconds")
        return gdf
        
    finally:
        con.close()


def _fetch_via_temp_file(con: duckdb.DuckDBPyConnection, sql: str) -> gpd.GeoDataFrame:
    """Memory-efficient data processing using temporary file strategy."""
    
    # Create temporary file with proper cleanup handling
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        # Export query results to temporary Parquet file using DuckDB's native Parquet support
        export_sql = f"""
        COPY (
            SELECT *, ST_AsWKB(geom) AS geometry
            FROM ({sql})
        ) TO '{temp_path}' 
        (FORMAT PARQUET);
        """
        
        logging.info(f"Exporting to temporary file: {temp_path}")
        con.execute(export_sql)
        
        # Platform-specific file system synchronization
        if platform.system() == "Windows":
            time.sleep(2)  # Allow file system to release locks
        
        # Load processed data from temporary file
        logging.info("Loading from temporary file...")
        df = pd.read_parquet(temp_path)
        
        # Convert WKB geometry column to shapely geometries
        if "geometry" in df.columns:
            df["geometry"] = df["geometry"].apply(
                lambda b: swkb.loads(bytes(b)) if b is not None else None
            )
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        else:
            raise ValueError("No geometry column found in temp file")
        
        logging.info(f"Loaded {len(gdf):,} features from temp file")
        return gdf
        
    finally:
        # Robust cleanup with retry logic
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    break
            except (PermissionError, OSError) as e:
                if attempt == max_attempts - 1:
                    logging.warning(f"Could not delete temp file {temp_path}: {e}")
                else:
                    time.sleep(1)  # Wait before retry


def _build_divisions_query(overture_config: dict, target_name: str, target_config: dict, selector: dict, limit: int = None) -> str:
    """
    Build query using Overture Divisions for precise country boundaries.
    Uses unified Overture configuration from YAML.
    """
    target_type = target_config.get('type')
    target_theme = target_config.get('theme')
    iso2 = selector.get('iso2', 'AF').upper()
    
    if not target_type or not target_theme:
        raise ValueError(f"Missing theme ({target_theme}) or type ({target_type}) in target configuration")
    
    # Select essential columns based on data type
    if target_type == "segment":  # Transportation
        columns = ['id', 'class', 'subtype', 'names', 'sources', 'version', 'routes']
    elif target_type == "building":  # Buildings
        columns = ['id', 'names', 'sources', 'version', 'height', 'num_floors', 'class', 'subtype']
    elif target_type == "place":  # Places
        columns = ['id', 'names', 'sources', 'version', 'class', 'subtype', 'categories']
    else:
        columns = ['id', 'names', 'sources', 'version', 'class', 'subtype']
    
    cols_sql = ", ".join(columns)
    
    # Get Overture data URLs using unified config
    data_url = f"{overture_config['base_url']}/theme={target_theme}/type={target_type}/*.parquet"
    divisions_url = f"{overture_config['base_url']}/theme=divisions/type=division_area/*.parquet"
    
    logging.info(f"Data URL: {data_url}")
    logging.info(f"Divisions URL: {divisions_url}")
    
    # Simplified query - we know country='AF' works
    sql = f"""
    WITH country_boundary AS (
        SELECT ST_GEOMFROMWKB(geometry) as country_geom
        FROM read_parquet('{divisions_url}', filename=true, hive_partitioning=1)
        WHERE subtype = 'country' AND country = '{iso2}'
        LIMIT 1
    ),
    filtered_data AS (
        SELECT {cols_sql}, ST_GEOMFROMWKB(geometry) AS geom
        FROM read_parquet('{data_url}', filename=true, hive_partitioning=1), country_boundary
        WHERE ST_INTERSECTS(ST_GEOMFROMWKB(geometry), country_geom)
    )
    SELECT {cols_sql}, geom
    FROM filtered_data
    """
    
    # Apply data-specific filters
    target_filter = target_config.get('filter')
    if target_filter:
        sql += f" WHERE {target_filter}"
    
    # Apply record limit for testing
    if limit:
        sql += f" LIMIT {limit}"
    
    return sql


def _build_optimized_query(parquet_url: str, xmin: float, ymin: float, xmax: float, ymax: float, 
                          target_config: dict, limit: int = None) -> str:
    """
    Construct optimized SQL query with bounding box spatial filtering.
    """
    target_type = target_config.get('type')
    
    # Select essential columns based on data type
    if target_type == "segment":  # Transportation
        columns = ['id', 'class', 'subtype', 'names', 'sources', 'version', 'routes']
    elif target_type == "building":  # Buildings
        columns = ['id', 'names', 'sources', 'version', 'height', 'num_floors', 'class', 'subtype']
    else:
        columns = ['id', 'names', 'sources', 'version', 'class', 'subtype']
    
    cols_sql = ", ".join(columns)
    
    # Optimized query with early geometry type conversion
    sql = f"""
    SELECT {cols_sql}, ST_GEOMFROMWKB(geometry) AS geom
    FROM read_parquet('{parquet_url}', filename=true, hive_partitioning=1)
    WHERE bbox.xmin >= {xmin}
      AND bbox.xmax <= {xmax}
      AND bbox.ymin >= {ymin}
      AND bbox.ymax <= {ymax}
    """
    
    # Apply data-specific filters
    target_filter = target_config.get('filter')
    if target_filter and "class = 'primary'" not in target_filter:
        sql += f" AND {target_filter}"
    
    # Apply record limit for testing and development workflows
    if limit:
        sql += f" LIMIT {limit}"
    
    return sql


def debug_divisions_structure(secure_config: Config) -> None:
    """
    Debug function to inspect the actual structure of divisions data.
    """
    con = setup_duckdb_optimized(secure_config)
    divisions_url = f"{secure_config.overture.base_url}/theme=divisions/type=division_area/*.parquet"
    
    try:
        # First, let's see what columns exist
        schema_sql = f"""
        SELECT * FROM read_parquet('{divisions_url}', filename=true, hive_partitioning=1)
        WHERE subtype = 'country'
        LIMIT 1
        """
        
        result = con.execute(schema_sql).fetchdf()
        print("Available columns in divisions data:")
        print(result.columns.tolist())
        print("\nSample country record:")
        print(result.to_dict('records')[0] if not result.empty else "No records found")
        
    finally:
        con.close()


def debug_divisions_afghanistan(secure_config: Config) -> str:
    """
    Diagnostic query to examine Afghanistan entries in Overture divisions dataset.
    """
    divisions_url = f"{secure_config.overture.base_url}/theme=divisions/type=division_area/*.parquet"
    
    return f"""
    SELECT 
        country,
        names.primary as primary_name,
        names.common as common_name,
        subtype,
        ST_AREA(ST_GEOMFROMWKB(geometry)) as area_sq_degrees
    FROM read_parquet('{divisions_url}', filename=true, hive_partitioning=1)
    WHERE subtype = 'country' 
    AND (
        country ILIKE '%AF%' OR 
        country ILIKE '%Afghanistan%' OR
        names.primary ILIKE '%Afghanistan%' OR 
        names.common ILIKE '%Afghanistan%'
    )
    ORDER BY area_sq_degrees DESC
    LIMIT 10
    """


def validate_data_quality(gdf: gpd.GeoDataFrame, target_name: str, min_features: int = 100) -> bool:
    """
    Comprehensive data quality validation before publication.
    """
    errors = []
    
    # Validate feature count meets minimum requirements
    if len(gdf) < min_features:
        errors.append(f"Too few features: {len(gdf)} < {min_features}")
    
    # Validate required schema columns
    required_cols = ['id', 'geometry']
    missing_cols = [col for col in required_cols if col not in gdf.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Validate and standardize coordinate reference system
    if gdf.crs is None:
        errors.append("No CRS defined")
    elif gdf.crs.to_string() != "EPSG:4326":
        logging.warning(f"CRS is {gdf.crs}, reprojecting to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
    
    # Assess geometry validity within acceptable thresholds
    invalid_geoms = gdf.geometry.isna().sum()
    invalid_pct = (invalid_geoms / len(gdf)) * 100
    if invalid_pct > 5:  # 5% tolerance for invalid geometries
        errors.append(f"Too many invalid geometries: {invalid_geoms} ({invalid_pct:.1f}%)")
    
    # Check for data integrity issues
    duplicate_ids = gdf['id'].duplicated().sum()
    if duplicate_ids > 0:
        errors.append(f"Duplicate IDs found: {duplicate_ids}")
    
    if errors:
        logging.error(f"Data validation failed for {target_name}:")
        for error in errors:
            logging.error(f"  - {error}")
        return False
    
    logging.info(f"Data validation passed: {len(gdf):,} features, {invalid_pct:.1f}% invalid geometries")
    return True


def _load_aoi_wkt(aoi_path: str) -> str:
    """Load area of interest geometry and convert to WKT format for spatial operations."""
    gdf = gpd.read_file(Path(aoi_path)).to_crs(4326)
    if gdf.empty:
        raise ValueError(f"AOI is empty: {aoi_path}")
    return gdf.unary_union.wkt