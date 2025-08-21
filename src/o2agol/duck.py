from __future__ import annotations

import logging
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

import duckdb
import geopandas as gpd
import pandas as pd
import shapely.wkb as swkb

from .cleanup import get_pid_temp_dir
from .config import Config
from .config.countries import CountryRegistry


# Get country bounding boxes from centralized registry
# Maintains backward compatibility with existing code patterns
def get_country_bboxes() -> dict[str, tuple[float, float, float, float]]:
    """Get country bounding boxes from registry for spatial filtering"""
    bboxes = CountryRegistry.get_bounding_boxes()
    # Convert list format to tuple format for backward compatibility
    return {iso2: tuple(bbox) for iso2, bbox in bboxes.items()}


def _parquet_url_from_config(overture_config: dict, theme: str, type_: str) -> str:
    """Return proper parquet URL using unified Overture configuration."""
    return f"{overture_config['base_url']}/theme={theme}/type={type_}/*.parquet"


def setup_duckdb_optimized(secure_config: Config) -> duckdb.DuckDBPyConnection:
    """Configure DuckDB with optimized settings for spatial queries."""
    con = duckdb.connect()
    
    # Install required extensions
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # Get PID-isolated temp directory for DuckDB
    temp_dir = get_pid_temp_dir() / 'duckdb'
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_dir_str = str(temp_dir).replace('\\', '/')
    
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
    
    # Temp directory configuration (use PID-isolated directory)
    con.execute(f"SET temp_directory='{temp_dir_str}';")
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
    logging.debug(f"PID-isolated temp directory: {temp_dir}")
    return con


def set_duckdb_session(con: duckdb.DuckDBPyConnection) -> None:
    """
    Enable DuckDB remote/object caches and progress for faster S3/HTTP reads.
    Safe to call multiple times.
    """
    con.execute("SET enable_http_metadata_cache=true;")
    con.execute("SET enable_object_cache=true;")
    con.execute("SET enable_progress_bar=true;")


def describe_parquet_schema(con: duckdb.DuckDBPyConnection, url: str) -> pd.DataFrame:
    """
    Create a temp table from the parquet schema (LIMIT 0) and return DESCRIBE.
    Use filename=true + hive_partitioning=1 if you rely on partitioned paths.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE _o2_temp_schema AS
        SELECT * FROM read_parquet('{url}', filename=true, hive_partitioning=1) LIMIT 0
    """)
    return con.execute("DESCRIBE _o2_temp_schema").df()


def _fmt_num(v: float) -> str:
    # Similar to invariant culture "G" with enough precision
    return format(float(v), ".15g")


def aoi_where_clause(bbox: Optional[Tuple[float, float, float, float]]) -> str:
    """
    Build a WHERE clause you can splice into a SELECT.
    You must have min/max coordinates available; if you store bbox columns, adapt accordingly.
    """
    if not bbox:
        return ""
    xmin, ymin, xmax, ymax = bbox
    return (
        "WHERE bbox_xmin >= {xmin} AND bbox_ymin >= {ymin} "
        "AND bbox_xmax <= {xmax} AND bbox_ymax <= {ymax}"
    ).format(xmin=_fmt_num(xmin), ymin=_fmt_num(ymin), xmax=_fmt_num(xmax), ymax=_fmt_num(ymax))


def materialize_current_table(
    con: duckdb.DuckDBPyConnection,
    url: str,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> int:
    """
    Builds a working table `_o2_current` from the parquet source, optionally filtered by `bbox`.
    Returns row count. Early-exit on empty.
    """
    where = aoi_where_clause(bbox)
    con.execute(f"""
        CREATE OR REPLACE TABLE _o2_current AS
        SELECT * FROM read_parquet('{url}', filename=true, hive_partitioning=1)
        {where}
    """)
    cnt = con.execute("SELECT COUNT(*) FROM _o2_current").fetchone()[0]
    return int(cnt)


_GEOM_FAMILIES = {
    "points": ("POINT", "MULTIPOINT"),
    "lines": ("LINESTRING", "MULTILINESTRING"),
    "polygons": ("POLYGON", "MULTIPOLYGON"),
}


def select_geometry_family(con: duckdb.DuckDBPyConnection, family: str) -> pd.DataFrame:
    """
    Returns a DataFrame for the given geometry family from `_o2_current`.
    Ensures geometry is the last column: SELECT * EXCLUDE geometry, geometry
    """
    allowed = _GEOM_FAMILIES[family]
    geom_in = ",".join(f"'{g}'" for g in allowed)
    q = f"""
        SELECT * EXCLUDE geometry, geometry
        FROM _o2_current
        WHERE ST_GeometryType(geometry) IN ({geom_in})
    """
    return con.execute(q).df()


def export_parquet_by_family(
    con: duckdb.DuckDBPyConnection,
    out_dir: Path,
    family: str,
    filename: Optional[str] = None,
    row_group_size: int = 100_000,
    compression: str = "ZSTD",
) -> Path:
    """
    COPY the selected family to a parquet file with row groups & compression.
    Geometry column is last (per SELECT). Returns the output path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = filename or f"{family.lower()}"
    out = out_dir / f"{base}.parquet"
    allowed = _GEOM_FAMILIES[family]
    geom_in = ",".join(f"'{g}'" for g in allowed)
    q = f"SELECT * EXCLUDE geometry, geometry FROM _o2_current WHERE ST_GeometryType(geometry) IN ({geom_in})"
    con.execute(f"""
        COPY ({q}) TO '{out.as_posix()}'
        (FORMAT 'PARQUET', ROW_GROUP_SIZE {row_group_size}, COMPRESSION '{compression}')
    """)
    return out


def fetch_family_df(
    s3_url: str,
    family: str,   # "points" | "lines" | "polygons"
    bbox: Optional[Tuple[float, float, float, float]] = None,
    use_cache: bool = False,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    con = duckdb.connect()
    set_duckdb_session(con)

    # (Optional) preview schema — log/validate if you want
    _schema = describe_parquet_schema(con, s3_url)
    logging.debug("Schema preview:\n%s", _schema)

    # materialize current table with optional AOI
    cnt = materialize_current_table(con, s3_url, bbox=bbox)
    if cnt == 0:
        logging.info("No rows after AOI filter; skipping.")
        return pd.DataFrame()

    # optional local parquet cache (fast retries/resume)
    if use_cache and cache_dir:
        export_parquet_by_family(con, cache_dir, family)

    # return in-memory frame with geometry column last
    return select_geometry_family(con, family)


def fetch_gdf(config_obj, target_name: str, use_local_dump: bool = False, 
              dump_manager=None, **kwargs) -> gpd.GeoDataFrame:
    """
    Fetch geospatial data with optional local dump support.
    
    Args:
        config_obj: Configuration object (adapted from CLI integration)
        target_name: Target data type (roads, buildings, places)
        use_local_dump: Whether to use local dump if available
        dump_manager: DumpManager instance for local queries
        **kwargs: Additional parameters including limit, use_divisions, etc.
        
    Returns:
        GeoDataFrame containing requested geospatial data
    """
    # Check for local dump availability first
    if use_local_dump and dump_manager:
        try:
            return _fetch_from_local_dump(config_obj, target_name, dump_manager, **kwargs)
        except Exception as e:
            logging.warning(f"Local dump query failed: {e}")
            logging.info("Falling back to S3 query...")
    
    # Fall back to existing S3 implementation
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
            iso2 = selector.get('iso2')
            if not iso2:
                raise ValueError("ISO2 country code missing from selector configuration. Check country selection.")
            iso2 = iso2.upper()
            country_bboxes = get_country_bboxes()
            if iso2 not in country_bboxes:
                available_countries = list(country_bboxes.keys())[:10]
                raise ValueError(f"No bounding box defined for country {iso2}. Available countries: {available_countries}")
            
            xmin, ymin, xmax, ymax = country_bboxes[iso2]
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
        iso2 = selector.get('iso2')
        if not iso2:
            raise ValueError("ISO2 country code missing from selector configuration. Check country selection.")
        iso2 = iso2.upper()
        country_bboxes = get_country_bboxes()
        xmin, ymin, xmax, ymax = country_bboxes[iso2]
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
        iso2 = selector.get('iso2')
        if not iso2:
            raise ValueError("ISO2 country code missing from selector configuration. Check country selection.")
        iso2 = iso2.upper()
        country_bboxes = get_country_bboxes()
        xmin, ymin, xmax, ymax = country_bboxes[iso2]
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
                elif isinstance(g, bytearray):
                    # Convert bytearray to bytes then to shapely geometry
                    return swkb.loads(bytes(g))
                else:
                    # Already a geometry object or handle unexpected types
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
    
    # Get ISO2 from selector - no default fallback to ensure proper country selection
    iso2 = selector.get('iso2')
    if not iso2:
        raise ValueError("ISO2 country code missing from selector configuration. Check country selection.")
    iso2 = iso2.upper()
    
    logging.info(f"Building Divisions query for country: {iso2}")
    
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
    country_bboxes = get_country_bboxes()
    if iso2 in country_bboxes:
        xmin, ymin, xmax, ymax = country_bboxes[iso2]
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


def _fetch_from_local_dump(config_obj, target_name: str, dump_manager, **kwargs) -> gpd.GeoDataFrame:
    """
    Query local dump with spatial filtering for improved performance.
    
    Args:
        config_obj: Configuration object
        target_name: Target data type
        dump_manager: DumpManager instance
        **kwargs: Query parameters including limit, use_divisions, country, etc.
        
    Returns:
        GeoDataFrame with query results
    """
    from .dump_manager import DumpQuery
    from .config.countries import CountryRegistry
    
    logging.info("Fetching data from local dump...")
    
    # Extract configuration
    secure_config, selector, targets, overture_config = _extract_config_data(config_obj)
    target_config = targets[target_name]
    
    # Get parameters
    limit = kwargs.get('limit')
    use_divisions = kwargs.get('use_divisions', True)
    country_override = kwargs.get('country')
    
    # Determine country
    if country_override:
        # Use country override from kwargs
        country_info = CountryRegistry.get_country(country_override)
        if not country_info:
            raise ValueError(f"Unknown country: {country_override}")
        iso2 = country_info.iso2
    else:
        # Use selector configuration
        iso2 = selector.get('iso2')
        if not iso2:
            raise ValueError("No country specified in selector or kwargs")
        iso2 = iso2.upper()
    
    logging.info(f"Querying local dump for country: {iso2}")
    
    # Build dump query
    theme = target_config.get('theme')
    type_ = target_config.get('type')
    
    if not theme or not type_:
        raise ValueError(f"Missing theme ({theme}) or type ({type_}) in target configuration")
    
    # Get bounding box if using bbox mode
    bbox = None
    if not use_divisions:
        country_bboxes = get_country_bboxes()
        if iso2 in country_bboxes:
            bbox = country_bboxes[iso2]
            logging.info(f"Using bbox filtering: {bbox}")
        else:
            logging.warning(f"No bbox found for {iso2}, using divisions")
            use_divisions = True
    
    # Create query object
    query = DumpQuery(
        theme=theme,
        type=type_,
        country=iso2,
        bbox=bbox,
        filters=target_config.get('filters'),
        limit=limit
    )
    
    # Get release version
    release = overture_config.get('release', '2025-07-23.0')
    
    # Execute query using optimized method if available
    if hasattr(dump_manager, 'query_local_dump_pyarrow') and dump_manager.config.use_pyarrow_queries:
        logging.debug("Using PyArrow-optimized query method")
        gdf = dump_manager.query_local_dump_pyarrow(query, release)
    else:
        logging.debug("Using cache-based query method")
        gdf = dump_manager.query_local_dump(query, release, secure_config)
    
    if gdf.empty:
        logging.warning("Local dump query returned no results")
        return gdf
    
    # Apply additional target-specific filtering if needed
    target_filter = target_config.get('filter')
    if target_filter:
        # Apply filter using pandas operations
        original_count = len(gdf)
        
        try:
            # Convert SQL-like filter to pandas query
            # This is a simplified filter application - more complex filters may need custom handling
            import re
            
            if 'class IN' in target_filter:
                # Handle class IN (...) filters
                match = re.search(r"class IN \('([^']+)'\)", target_filter)
                if match and 'class' in gdf.columns:
                    allowed_classes = match.group(1).split("', '")
                    gdf = gdf[gdf['class'].isin(allowed_classes)]
            elif 'class =' in target_filter:
                # Handle class = 'value' filters
                match = re.search(r"class = '([^']+)'", target_filter)
                if match and 'class' in gdf.columns:
                    allowed_class = match.group(1)
                    gdf = gdf[gdf['class'] == allowed_class]
            elif 'categories.primary =' in target_filter:
                # Handle categories.primary = 'value' filters
                match = re.search(r"categories\.primary = '([^']+)'", target_filter)
                if match and 'category_primary' in gdf.columns:
                    allowed_category = match.group(1)
                    gdf = gdf[gdf['category_primary'] == allowed_category]
            elif 'categories.primary IN' in target_filter:
                # Handle categories.primary IN (...) filters
                match = re.search(r"categories\.primary IN \('([^']+)'\)", target_filter)
                if match and 'category_primary' in gdf.columns:
                    allowed_categories = match.group(1).split("', '")
                    gdf = gdf[gdf['category_primary'].isin(allowed_categories)]
            
            filtered_count = len(gdf)
            if filtered_count != original_count:
                logging.info(f"Applied target filter: {original_count} -> {filtered_count} features")
                
        except Exception as e:
            logging.warning(f"Could not apply target filter '{target_filter}': {e}")
    
    # Handle dual queries (places + buildings) if needed
    building_filter = target_config.get('building_filter')
    if building_filter:
        logging.info("Dual query detected - fetching buildings separately")
        
        # Check if buildings dump exists
        if not dump_manager.check_dump_exists(release, 'buildings'):
            logging.warning("Buildings dump not found for dual query - only returning places data")
            logging.warning("Use: o2agol overture-dump <query> --download-only to download all required themes")
            return gdf
        
        # Create buildings query
        buildings_query = DumpQuery(
            theme='buildings',
            type='building', 
            country=iso2,
            bbox=bbox,
            filters=None,  # Building filter applied later
            limit=limit
        )
        
        # Use same optimized query method for buildings
        if hasattr(dump_manager, 'query_local_dump_pyarrow') and dump_manager.config.use_pyarrow_queries:
            buildings_gdf = dump_manager.query_local_dump_pyarrow(buildings_query, release)
        else:
            buildings_gdf = dump_manager.query_local_dump(buildings_query, release, secure_config)
        
        if not buildings_gdf.empty:
            # Apply building filter
            try:
                if 'categories' in building_filter and 'categories' in buildings_gdf.columns:
                    # Filter buildings by categories (simplified)
                    buildings_gdf = buildings_gdf[buildings_gdf['categories'].notna()]
                
                # Add source type identifier
                gdf['source_type'] = 'place'
                buildings_gdf['source_type'] = 'building'
                
                # Return combined result as dict for multi-layer processing
                return {
                    'places': gdf,
                    'buildings': buildings_gdf
                }
                
            except Exception as e:
                logging.warning(f"Could not apply building filter: {e}")
        else:
            logging.warning("No buildings found for dual query")
    
    logging.info(f"Local dump query completed: {len(gdf)} features")
    return gdf

