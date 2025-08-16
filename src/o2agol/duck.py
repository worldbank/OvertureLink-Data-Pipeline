from __future__ import annotations

import logging
import os
import platform
import tempfile
import time

import duckdb
import geopandas as gpd
import pandas as pd
import shapely.wkb as swkb

from .config import Config

# Country bounding boxes for efficient spatial filtering
# Eliminates expensive polygon-based country boundary lookups
COUNTRY_BBOXES = {
    'AF': (60.5, 29.4, 74.9, 38.5),   # Afghanistan
    'PK': (60.9, 23.6, 77.8, 37.1),   # Pakistan
    'BD': (88.0, 20.7, 92.7, 26.6),   # Bangladesh  
    'IN': (68.1, 6.7, 97.4, 37.1),    # India
}


def _parquet_url_from_config(overture_config: dict, theme: str, type_: str) -> str:
    """Return proper parquet URL using unified Overture configuration."""
    return f"{overture_config['base_url']}/theme={theme}/type={type_}/*.parquet"


def setup_duckdb_optimized(secure_config: Config) -> duckdb.DuckDBPyConnection:
    """Configure DuckDB with optimized settings for spatial queries."""
    con = duckdb.connect()
    
    # Install required extensions
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # Get project root directory for temp files
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    temp_dir = os.path.join(project_root, 'temp', 'duckdb')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Advanced DuckDB configuration based on official performance guidelines
    duckdb_settings = secure_config.get_duckdb_settings()
    threads = duckdb_settings['threads']
    
    # Memory optimization: 1-4 GB per thread (DuckDB recommendation)
    total_memory_gb = int(os.environ.get('DUCKDB_MEMORY_LIMIT', duckdb_settings['memory_limit']).replace('GB', ''))
    memory_per_thread = max(1, min(4, total_memory_gb // threads))
    optimized_memory = f"{memory_per_thread * threads}GB"
    
    # Core memory and threading configuration
    con.execute(f"SET memory_limit='{optimized_memory}';")
    con.execute(f"SET threads={threads};")
    
    # Temp directory configuration (use fast local storage)
    con.execute(f"SET temp_directory='{temp_dir.replace(os.sep, '/')}';")
    con.execute("SET max_temp_directory_size='50GB';")  # Reasonable limit for temp files
    
    # S3 and remote file optimizations (only use settings that exist)
    con.execute(f"SET s3_region='{secure_config.overture.s3_region}';")
    
    # HTTP connection optimizations
    con.execute("SET http_timeout=1800000;")  # 30 minutes for large transfers
    con.execute("SET http_retries=3;")
    con.execute("SET http_keep_alive=true;")
    
    # Core performance optimizations (verified to exist)
    con.execute("SET preserve_insertion_order=false;")  # Reduces memory usage for large datasets
    
    # Query execution optimizations  
    con.execute("SET enable_progress_bar=true;")
    
    # Only set these if they exist in this DuckDB version
    try:
        con.execute("SET enable_http_metadata_cache=true;")
    except Exception:
        logging.debug("enable_http_metadata_cache not available in this DuckDB version")
        
    try:
        con.execute("SET s3_url_compatibility_mode=false;")
    except Exception:
        logging.debug("s3_url_compatibility_mode not available in this DuckDB version")
    
    # Set environment for progress reporting
    os.environ['DUCKDB_PROGRESS_BAR'] = '1'
    
    logging.info(f"DuckDB configured (advanced): {threads} threads @ {memory_per_thread}GB each, temp: {temp_dir}")
    return con


def fetch_gdf(config_obj, target_name: str, **kwargs) -> gpd.GeoDataFrame:
    """
    Fetch geospatial data.
    
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
        
        # Configuration extracted successfully
        logging.debug(f"Loaded {len(targets)} target(s): {list(targets.keys())}")
            
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
        
        # Check if this is a dual query (places + buildings)
        building_filter = target_config.get('building_filter')
        if building_filter:
            logging.info("Dual query detected: fetching both places and buildings")
            return _fetch_dual_query(con, overture_config, target_name, target_config, selector, limit, use_divisions)
        
        # Single query logic (existing behavior)
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
            sql = _build_bbox_query(parquet_url, xmin, ymin, xmax, ymax, target_config, limit)
        
        logging.info("Executing optimized query with two-step processing...")
        logging.info(f"SQL preview: {sql[:200]}...")
        start_time = time.time()
        
        # Use optimized two-step processing (recommended by Overture docs)
        gdf = _fetch_via_temp_file(con, sql)
        
        elapsed = time.time() - start_time
        logging.info(f"Fetched {len(gdf):,} features in {elapsed:.1f} seconds")
        return gdf
        
    finally:
        con.close()


def _fetch_dual_query(con: duckdb.DuckDBPyConnection, overture_config: dict, target_name: str, 
                     target_config: dict, selector: dict, limit: int = None, 
                     use_divisions: bool = True) -> dict:
    """
    Execute dual query for places and buildings, returning separate GeoDataFrames for multi-layer service.
    
    Returns:
        Dict with 'places' and 'buildings' keys containing respective GeoDataFrames
    """
    start_time = time.time()
    result = {}
    
    # Query 1: Places (points)
    logging.info("Fetching places data (points)...")
    places_config = target_config.copy()
    if use_divisions:
        places_sql = _build_divisions_query(overture_config, target_name, places_config, selector, limit)
    else:
        iso2 = selector.get('iso2', 'AF').upper()
        xmin, ymin, xmax, ymax = COUNTRY_BBOXES[iso2]
        parquet_url = _parquet_url_from_config(overture_config, places_config.get('theme'), places_config.get('type'))
        places_sql = _build_bbox_query(parquet_url, xmin, ymin, xmax, ymax, places_config, limit)
    
    places_gdf = _fetch_via_temp_file(con, places_sql)
    if len(places_gdf) > 0:
        places_gdf['source_type'] = 'place'
        result['places'] = places_gdf
        logging.info(f"Fetched {len(places_gdf):,} places")
    
    # Query 2: Buildings (polygons)  
    logging.info("Fetching buildings data (polygons)...")
    buildings_config = target_config.copy()
    buildings_config['theme'] = 'buildings'
    buildings_config['type'] = 'building'
    buildings_config['filter'] = target_config.get('building_filter')
    
    if use_divisions:
        buildings_sql = _build_divisions_query(overture_config, target_name, buildings_config, selector, limit)
    else:
        iso2 = selector.get('iso2', 'AF').upper()
        xmin, ymin, xmax, ymax = COUNTRY_BBOXES[iso2]
        parquet_url = _parquet_url_from_config(overture_config, buildings_config.get('theme'), buildings_config.get('type'))
        buildings_sql = _build_bbox_query(parquet_url, xmin, ymin, xmax, ymax, buildings_config, limit)
    
    buildings_gdf = _fetch_via_temp_file(con, buildings_sql)
    if len(buildings_gdf) > 0:
        buildings_gdf['source_type'] = 'building'
        result['buildings'] = buildings_gdf
        
        # Debug: Check building geometries
        building_geom_types = buildings_gdf.geometry.geom_type.value_counts().to_dict()
        logging.info(f"Fetched {len(buildings_gdf):,} buildings with geometry types: {building_geom_types}")
        
        # Debug: Sample of building data
        if len(buildings_gdf) > 0:
            sample = buildings_gdf.iloc[0]
            logging.info(f"Sample building: id={sample.get('id', 'N/A')}, geometry_type={sample.geometry.geom_type}")
    else:
        logging.warning("Buildings query returned 0 features!")
    
    if not result:
        raise ValueError("No data found for either places or buildings")
    
    elapsed = time.time() - start_time
    total_features = sum(len(gdf) for gdf in result.values())
    logging.info(f"Dual query completed: {total_features:,} total features in {elapsed:.1f} seconds")
    
    # Debug: Final result structure
    for layer_name, layer_gdf in result.items():
        geom_types = layer_gdf.geometry.geom_type.value_counts().to_dict()
        logging.info(f"Returning {layer_name}: {len(layer_gdf):,} features, geometries: {geom_types}")
    
    return result


def _fetch_via_temp_file(con: duckdb.DuckDBPyConnection, sql: str) -> gpd.GeoDataFrame:
    """Optimized two-step processing following Overture documentation recommendations."""
    
    # Step 1: Execute query and save to temporary parquet file (Overture recommended approach)
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        start_time = time.time()
        
        # First step: Save query result directly to parquet (more efficient than inline processing)
        logging.info(f"Step 1: Executing query and saving to parquet: {temp_path}")
        
        # Execute the main query with any SET variable commands
        if "SET variable" in sql:
            # Find the end of the variable setting (look for "); followed by SELECT)
            # This is more robust than splitting on first SELECT which might be inside the variable
            pattern_end = ");"
            pattern_start = "SELECT"
            
            # Find all positions where "); occurs
            pos = 0
            variable_end = -1
            while pos < len(sql):
                pos = sql.find(pattern_end, pos)
                if pos == -1:
                    break
                # Check if SELECT follows after some whitespace
                after_pattern = sql[pos + len(pattern_end):].strip()
                if after_pattern.startswith(pattern_start):
                    variable_end = pos + len(pattern_end)
                    break
                pos += 1
            
            if variable_end > 0:
                variable_sql = sql[:variable_end].strip()
                main_query = sql[variable_end:].strip()
                
                # Execute variable settings first
                logging.info("Executing variable settings...")
                con.execute(variable_sql)
                
                # Then execute main query with COPY
                copy_sql = f"""
                COPY (
                    {main_query}
                ) TO '{temp_path}' (FORMAT PARQUET);
                """
            else:
                # Fallback: couldn't split properly
                logging.warning("Could not split variable SQL properly, executing as single statement")
                copy_sql = f"""
                COPY (
                    {sql}
                ) TO '{temp_path}' (FORMAT PARQUET);
                """
        else:
            # Single query approach for non-variable queries
            copy_sql = f"""
            COPY (
                {sql}
            ) TO '{temp_path}' (FORMAT PARQUET);
            """
        
        con.execute(copy_sql)
        
        query_time = time.time() - start_time
        logging.info(f"Step 1 completed: Query executed and saved in {query_time:.1f}s")
        
        # Platform-specific file system synchronization
        if platform.system() == "Windows":
            time.sleep(1)  # Allow file system to release locks
        
        # Step 2: Load parquet and convert geometries (separate step for efficiency)
        logging.info("Step 2: Loading parquet and converting geometries...")
        load_start = time.time()
        
        df = pd.read_parquet(temp_path)
        
        # Convert geometry column to shapely geometries
        if "geom" in df.columns:
            # Handle both native geometry and WKB bytes
            def convert_geometry(g):
                if g is None:
                    return None
                elif isinstance(g, bytes):
                    # Convert WKB bytes to shapely geometry
                    return swkb.loads(g)
                else:
                    # Already a geometry object
                    return g
            
            df["geometry"] = df["geom"].apply(convert_geometry)
            df = df.drop(columns=["geom"])
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        else:
            raise ValueError("No geometry column found in temp file")
        
        load_time = time.time() - load_start
        total_time = time.time() - start_time
        
        logging.info(f"Step 2 completed: Loaded {len(gdf):,} features in {load_time:.1f}s")
        logging.info(f"Two-step processing completed: {len(gdf):,} features in {total_time:.1f}s total")
        
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




def _get_columns_for_target_type(target_type: str) -> list[str]:
    """
    Get essential columns for a specific Overture data type.
    
    Includes all fields needed for AGOL filtering and analysis.
    
    Args:
        target_type: Overture data type (segment, building, place, etc.)
        
    Returns:
        List of column names for the target type
    """
    if target_type == "segment":  # Transportation
        return ['id', 'class', 'subtype', 'names', 'sources', 'version', 'routes']
    elif target_type == "building":  # Buildings  
        # Include class and subtype for type_category creation and filtering
        return ['id', 'names', 'sources', 'version', 'height', 'num_floors', 'class', 'subtype']
    elif target_type == "place":  # Places
        # Include categories for type_category creation and confidence for quality filtering
        return ['id', 'names', 'sources', 'version', 'categories', 'confidence', 'addresses', 'websites']
    else:
        # Default - ensure class/subtype available for filtering
        return ['id', 'names', 'sources', 'version', 'class', 'subtype']


def _build_divisions_query(overture_config: dict, target_name: str, target_config: dict, selector: dict, limit: int = None) -> str:
    """
    Build query using Overture Divisions.
    Uses Overture configuration from YAML.
    """
    target_type = target_config.get('type')
    target_theme = target_config.get('theme')
    iso2 = selector.get('iso2', 'AF').upper()
    
    if not target_type or not target_theme:
        raise ValueError(f"Missing theme ({target_theme}) or type ({target_type}) in target configuration")
    
    # Use centralized column selection logic (optimized for projection pushdown)
    columns = _get_columns_for_target_type(target_type)
    
    # Get Overture data URLs using unified config
    data_url = f"{overture_config['base_url']}/theme={target_theme}/type={target_type}/*.parquet"
    divisions_url = f"{overture_config['base_url']}/theme=divisions/type=division_area/*.parquet"
    
    logging.info(f"Data URL: {data_url}")
    logging.info(f"Divisions URL: {divisions_url}")
    
    # CRITICAL OPTIMIZATION: Always use bbox pre-filtering for spatial queries
    bbox_filter = ""
    if iso2 in COUNTRY_BBOXES:
        xmin, ymin, xmax, ymax = COUNTRY_BBOXES[iso2]
        bbox_buffer = 0.1  # Small buffer for edge cases
        bbox_filter = f"""
        AND d.bbox.xmin > {xmin - bbox_buffer}
        AND d.bbox.xmax < {xmax + bbox_buffer}
        AND d.bbox.ymin > {ymin - bbox_buffer}
        AND d.bbox.ymax < {ymax + bbox_buffer}
        """
        logging.info(f"Using bbox pre-filter for {iso2}: ({xmin}, {ymin}, {xmax}, {ymax}) + {bbox_buffer} buffer")
    else:
        logging.warning(f"No bbox defined for {iso2} - spatial query may be slow")
    
    # Optimized query using DuckDB variables (recommended by Overture docs)
    # DuckDB 1.1.3+ with native GeoParquet support - geometry is already GEOMETRY type
    sql = f"""
    SET variable country_geom = (
        SELECT geometry
        FROM read_parquet('{divisions_url}', filename=true, hive_partitioning=1)
        WHERE subtype = 'country' AND country = '{iso2}'
        LIMIT 1
    );
    
    SELECT {', '.join([f'd.{col}' for col in columns])}, d.geometry AS geom
    FROM read_parquet('{data_url}', filename=true, hive_partitioning=1) d
    WHERE 1=1
    {bbox_filter}
    AND ST_INTERSECTS(d.geometry, getvariable('country_geom'))
    """
    
    # Apply data-specific filters
    target_filter = target_config.get('filter')
    if target_filter:
        sql += f" AND {target_filter}"
    
    # Apply record limit for testing
    if limit:
        sql += f" LIMIT {limit}"
    
    return sql


def _build_bbox_query(parquet_url: str, xmin: float, ymin: float, xmax: float, ymax: float, 
                      target_config: dict, limit: int = None) -> str:
    """
    Construct SQL query with bounding box spatial filtering and projection pushdown.
    """
    target_type = target_config.get('type')
    
    # Use centralized column selection logic (enables projection pushdown)
    columns = _get_columns_for_target_type(target_type)
    
    # Optimized query with explicit column selection and table alias
    # DuckDB 1.1.3+ with native GeoParquet support - geometry is already GEOMETRY type
    sql = f"""
    SELECT {', '.join([f'd.{col}' for col in columns])}, d.geometry AS geom
    FROM read_parquet('{parquet_url}', filename=true, hive_partitioning=1) d
    WHERE d.bbox.xmin > {xmin}
      AND d.bbox.xmax < {xmax}
      AND d.bbox.ymin > {ymin}
      AND d.bbox.ymax < {ymax}
    """
    
    # Apply data-specific filters with table alias
    target_filter = target_config.get('filter')
    if target_filter:
        # Add table alias to filter if not present
        if 'd.' not in target_filter and not any(op in target_filter for op in ['=', '<', '>', 'IN', 'LIKE']):
            # Simple column references - add alias
            filter_with_alias = target_filter.replace('class', 'd.class').replace('subtype', 'd.subtype')
            sql += f" AND {filter_with_alias}"
        else:
            sql += f" AND {target_filter}"
    
    # Apply record limit for testing and development workflows
    if limit:
        sql += f" LIMIT {limit}"
    
    return sql




