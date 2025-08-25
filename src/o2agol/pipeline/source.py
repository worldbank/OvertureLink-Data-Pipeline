"""
OvertureSource - Unified Data Source Management

Consolidates DuckDB remote queries and local dump management into a single interface.
This module replaces duck.py and dump_manager.py functionality.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import geopandas as gpd
import pandas as pd

from ..cleanup import get_pid_temp_dir
from ..domain.enums import ClipStrategy
from ..domain.models import Country, Query, RunOptions

# Column definitions for each Overture schema type
# Users can easily add/remove columns here to customize data export
# Following Overture docs pattern: explicit column selection with nested field handling
OVERTURE_COLUMNS = {
    'segment': [  # Roads/Transportation
        'id',
        'names.primary as name',
        'class',
        'subtype', 
        'version',
        # 'routes',  # Complex struct
        'geometry'
    ],
    'building': [  # Buildings
        'id',
        'names.primary as name',
        'height',
        'num_floors',
        'class',
        'subtype',
        'version',
        'geometry'
    ],
    'place': [  # Places
        'id',
        'names.primary as name',
        'categories',  # Full categories struct for filtering
        'categories.primary as category',  # Simplified category for export
        'confidence',
        # 'addresses',  # Complex struct array
        # 'websites',   # Array of strings
        # 'socials',    # Array of strings
        'version',
        'geometry'
    ]
}
from ..config.countries import CountryRegistry


def apply_sql_filter(gdf: gpd.GeoDataFrame, sql_filter: str) -> gpd.GeoDataFrame:
    """
    Apply SQL-style filter string to a GeoDataFrame.
    
    Supports basic SQL conditions like:
    - subtype = 'medical'
    - categories.primary = 'health_and_medical'
    
    Args:
        gdf: GeoDataFrame to filter
        sql_filter: SQL-style filter string
        
    Returns:
        Filtered GeoDataFrame
    """
    if not sql_filter or sql_filter.strip() == "":
        return gdf
    
    try:
        # Handle simple equality filters: column = 'value'
        if " = " in sql_filter and " IN " not in sql_filter:
            parts = sql_filter.split(" = ", 1)
            if len(parts) == 2:
                column = parts[0].strip()
                value = parts[1].strip().strip("'\"")
                
                # Handle nested column access like 'categories.primary'
                if "." in column:
                    base_col, nested_key = column.split(".", 1)
                    if base_col in gdf.columns:
                        logging.debug(f"Applying nested filter: {column} = {value}")
                        # Filter based on nested dictionary access
                        mask = gdf[base_col].apply(lambda x: isinstance(x, dict) and x.get(nested_key) == value if x is not None else False)
                        return gdf[mask]
                    else:
                        logging.warning(f"Base column '{base_col}' not found in data, returning empty result")
                        return gdf.iloc[0:0]
                elif column in gdf.columns:
                    return gdf[gdf[column] == value]
                else:
                    logging.warning(f"Column '{column}' not found in data, returning empty result")
                    return gdf.iloc[0:0]  # Return empty GeoDataFrame with same structure
        
        # Handle IN filters: column IN ('value1', 'value2')
        elif " IN " in sql_filter:
            parts = sql_filter.split(" IN ", 1)
            if len(parts) == 2:
                column = parts[0].strip()
                values_part = parts[1].strip()
                
                # Extract values from parentheses
                if values_part.startswith("(") and values_part.endswith(")"):
                    values_str = values_part[1:-1]  # Remove parentheses
                    # Split by comma and clean up quotes
                    values = [v.strip().strip("'\"") for v in values_str.split(",")]
                    
                    # Handle nested column access like 'categories.primary'
                    if "." in column:
                        base_col, nested_key = column.split(".", 1)
                        if base_col in gdf.columns:
                            logging.debug(f"Applying nested IN filter: {column} IN {values}")
                            # Filter based on nested dictionary access
                            mask = gdf[base_col].apply(lambda x: isinstance(x, dict) and x.get(nested_key) in values if x is not None else False)
                            return gdf[mask]
                        else:
                            logging.warning(f"Base column '{base_col}' not found in data, returning empty result")
                            return gdf.iloc[0:0]
                    elif column in gdf.columns:
                        return gdf[gdf[column].isin(values)]
                    else:
                        logging.warning(f"Column '{column}' not found in data, returning empty result")
                        return gdf.iloc[0:0]  # Return empty GeoDataFrame with same structure
        
        logging.warning(f"Unsupported filter format: {sql_filter}, returning unfiltered data")
        return gdf
        
    except Exception as e:
        logging.error(f"Failed to apply SQL filter '{sql_filter}': {e}, returning unfiltered data")
        return gdf


@dataclass
class CacheMetadata:
    """Metadata for a country-specific cache entry."""
    country: str
    release: str
    theme: str
    type_name: str
    cached_date: str
    feature_count: int
    size_mb: float
    bbox: tuple[float, float, float, float]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'country': self.country,
            'release': self.release,
            'theme': self.theme,
            'type': self.type_name,
            'cached_date': self.cached_date,
            'feature_count': self.feature_count,
            'size_mb': self.size_mb,
            'bbox': list(self.bbox)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CacheMetadata':
        """Create from dictionary."""
        return cls(
            country=data['country'],
            release=data['release'],
            theme=data['theme'],
            type_name=data['type'],
            cached_date=data['cached_date'],
            feature_count=data['feature_count'],
            size_mb=data['size_mb'],
            bbox=tuple(data['bbox'])
        )


@dataclass
class CacheQuery:
    """Query configuration for cache operations."""
    country: str
    theme: str
    type_name: str
    release: str = "latest"
    use_divisions: bool = True
    limit: Optional[int] = None
    filters: Optional[dict | str] = None


class OvertureSource:
    """
    Unified data source for Overture Maps data.
    
    Handles both remote S3 queries via DuckDB and local dump management
    with automatic fallback between data sources.
    """
    
    # Overture's six official themes (moved from DumpManager)
    OVERTURE_THEMES = [
        "addresses", "base", "buildings", 
        "divisions", "places", "transportation"
    ]
    
    def __init__(self, cfg: dict, run: RunOptions):
        """
        Initialize data source with configuration and runtime options.
        
        Args:
            cfg: Pipeline configuration dictionary
            run: Runtime options including clip strategy, limits, etc.
        """
        self.cfg = cfg
        self.run = run
        self._connection = None
        
        # Dump management properties (integrated from DumpManager)
        dump_config = cfg.get('dump', {})
        self._dump_base_path = Path(dump_config.get('base_path', './overturedump'))
        self._cache_dir = self._dump_base_path / "country_cache"
        self._use_local_dumps = dump_config.get('enabled', True)
        self._max_memory_gb = dump_config.get('max_memory_gb', 16)
        self._compression = dump_config.get('compression', 'zstd')
        self._enable_cache = dump_config.get('use_cache', True)
        
        # Cache for dump metadata to avoid repeated filesystem checks
        self._dump_metadata_cache = {}
        
        # Initialize dump directories if local dumps are enabled
        if self._use_local_dumps:
            try:
                self._dump_base_path.mkdir(parents=True, exist_ok=True)
                if self._enable_cache:
                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                logging.debug(f"Dump directories initialized: {self._dump_base_path}")
            except Exception as e:
                logging.warning(f"Failed to initialize dump directories: {e}")
                self._use_local_dumps = False
        
    def _setup_duckdb_optimized(self) -> duckdb.DuckDBPyConnection:
        """Configure DuckDB with optimized settings for spatial queries."""
        con = duckdb.connect()
        
        # Install required extensions
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL spatial; LOAD spatial;")
        
        # Get PID-isolated temp directory for DuckDB
        temp_dir = get_pid_temp_dir() / 'duckdb'
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_str = str(temp_dir).replace('\\', '/')
        
        # Memory and threading configuration
        threads = int(os.environ.get('DUCKDB_THREADS', '4'))
        memory_limit = os.environ.get('DUCKDB_MEMORY_LIMIT', '8GB')
        
        # Memory optimization: 1-4 GB per thread (DuckDB recommendation)
        total_memory_gb = int(memory_limit.replace('GB', ''))
        memory_per_thread = max(1, min(4, total_memory_gb // threads))
        optimized_memory = f"{memory_per_thread * threads}GB"
        
        # Core memory and threading configuration
        con.execute(f"SET memory_limit='{optimized_memory}';")
        con.execute(f"SET threads={threads};")
        
        # Temp directory configuration (use PID-isolated directory)
        con.execute(f"SET temp_directory='{temp_dir_str}';")
        con.execute("SET max_temp_directory_size='50GB';")
        
        # S3 and remote file optimizations
        s3_region = self.cfg.get('overture', {}).get('s3_region', 'us-west-2')
        con.execute(f"SET s3_region='{s3_region}';")
        
        # HTTP connection optimizations
        con.execute("SET http_timeout=1800000;")  # 30 minutes for large transfers
        con.execute("SET http_retries=3;")
        con.execute("SET http_keep_alive=true;")
        
        # Core performance optimizations
        con.execute("SET preserve_insertion_order=false;")
        con.execute("SET enable_progress_bar=true;")
        
        # Optional settings (try/catch for version compatibility)
        try:
            con.execute("SET enable_http_metadata_cache=true;")
            con.execute("SET enable_object_cache=true;")
        except Exception:
            logging.debug("Some DuckDB cache settings not available in this version")
        
        # Set environment for progress reporting
        os.environ['DUCKDB_PROGRESS_BAR'] = '1'
        
        logging.info(f"DuckDB configured: {threads} threads @ {memory_per_thread}GB each, temp: {temp_dir}")
        return con
        
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create optimized DuckDB connection."""
        if self._connection is None:
            self._connection = self._setup_duckdb_optimized()
        return self._connection
        
    def _parquet_url_from_config(self, theme: str, type_: str) -> str:
        """Return proper parquet URL using Overture configuration."""
        overture_config = self.cfg.get('overture', {})
        base_url = overture_config.get('base_url', 's3://overturemaps-us-west-2/release')
        release = overture_config.get('release')
        if not release:
            raise ValueError("No Overture release configured. Set OVERTURE_RELEASE environment variable.")
        
        # Ensure base_url includes the release
        if not base_url.endswith(release):
            base_url = f"{base_url}/{release}"
            
        return f"{base_url}/theme={theme}/type={type_}/*.parquet"
        
    def _get_country_bboxes(self) -> dict[str, tuple[float, float, float, float]]:
        """Get country bounding boxes from registry for spatial filtering."""
        bboxes = CountryRegistry.get_bounding_boxes()
        # Convert list format to tuple format for backward compatibility
        return {iso2: tuple(bbox) for iso2, bbox in bboxes.items()}
        
    def _build_spatial_query(self, query: Query, country: Country, clip_strategy: ClipStrategy, limit: Optional[int] = None) -> str:
        """
        Unified spatial query builder with pluggable clipping strategies.
        
        This method eliminates code duplication between divisions and bbox approaches
        while maintaining the performance characteristics of each strategy.
        
        Args:
            query: Query configuration with theme and type
            country: Country information with bounds and ISO codes
            clip_strategy: ClipStrategy.DIVISIONS or ClipStrategy.BBOX
            limit: Optional feature limit for testing/development
            
        Returns:
            SQL query string optimized for the selected clipping strategy
        """
        # Common validation
        if not query.theme or not query.type:
            raise ValueError(f"Missing theme ({query.theme}) or type ({query.type}) in query configuration")
        
        # Get explicit columns for this schema type (following Overture docs pattern)
        columns = OVERTURE_COLUMNS.get(query.type, ['id', 'geometry'])  # fallback for unknown types
        column_list = ', '.join([f'd.{col}' for col in columns])
        
        # Unified URL construction using centralized method (fixes release bug)
        data_url = self._parquet_url_from_config(query.theme, query.type)
        
        # Build strategy-specific spatial filtering
        if clip_strategy == ClipStrategy.DIVISIONS:
            spatial_sql = self._build_divisions_spatial_filter(data_url, column_list, country.iso2.upper(), country.bounds)
            logging.info(f"Building Divisions query for country: {country.iso2.upper()}")
        else:  # ClipStrategy.BBOX
            if not country.bounds:
                raise ValueError(f"No bounding box available for country {country.name}")
            spatial_sql = self._build_bbox_spatial_filter(data_url, column_list, country.bounds)
            logging.info(f"Using bbox for {country.iso2}: {country.bounds}")
        
        # Apply common post-processing (eliminates duplication)
        if query.category_filter:
            spatial_sql += f" AND {query.category_filter}"
            
        if limit:
            spatial_sql += f" LIMIT {limit}"
            
        return spatial_sql
    
    def _build_divisions_spatial_filter(self, data_url: str, column_list: str, iso2: str, bounds: Optional[tuple[float, float, float, float]] = None) -> str:
        """Build spatial filter using Overture Divisions with DuckDB variables and bbox pre-filter for optimal performance."""
        divisions_url = self._parquet_url_from_config('divisions', 'division_area')
        
        # Critical optimization: bbox pre-filter before spatial intersection (from deprecated duck.py)
        bbox_filter = ""
        if bounds:
            xmin, ymin, xmax, ymax = bounds
            bbox_buffer = 0.1  # Small buffer for edge cases
            bbox_filter = f"""
            AND d.bbox.xmin > {xmin - bbox_buffer}
            AND d.bbox.xmax < {xmax + bbox_buffer}
            AND d.bbox.ymin > {ymin - bbox_buffer}
            AND d.bbox.ymax < {ymax + bbox_buffer}"""
            logging.info(f"Using bbox pre-filter for {iso2}: ({xmin}, {ymin}, {xmax}, {ymax}) + {bbox_buffer} buffer")
        else:
            logging.warning(f"No bbox defined for {iso2} - spatial query may be slow")
        
        return f"""
        SET variable country_geom = (
            SELECT geometry
            FROM read_parquet('{divisions_url}', filename=true, hive_partitioning=1)
            WHERE subtype = 'country' AND country = '{iso2}'
            LIMIT 1
        );
        
        SELECT {column_list}
        FROM read_parquet('{data_url}', filename=true, hive_partitioning=1) d
        WHERE 1=1{bbox_filter}
        AND ST_INTERSECTS(d.geometry, getvariable('country_geom'))
        """
    
    def _build_bbox_spatial_filter(self, data_url: str, column_list: str, bounds: tuple[float, float, float, float]) -> str:
        """Build spatial filter using bounding box for fast development workflows."""
        xmin, ymin, xmax, ymax = bounds
        
        return f"""
        SELECT {column_list}
        FROM read_parquet('{data_url}', filename=true, hive_partitioning=1) d
        WHERE d.bbox.xmin > {xmin}
          AND d.bbox.xmax < {xmax}
          AND d.bbox.ymin > {ymin}
          AND d.bbox.ymax < {ymax}
        """
        
    def _fetch_dual_query(self, query: Query, country: Country, clip: ClipStrategy, limit: Optional[int] = None) -> dict[str, gpd.GeoDataFrame]:
        """
        Execute dual query for places and buildings, returning separate GeoDataFrames for multi-layer service.
        
        This is used for education, health, and markets queries that need both places and buildings data.
        
        Args:
            query: Query configuration with multi-layer flag
            country: Country information with bounds
            clip: Clipping strategy (divisions or bbox)
            limit: Optional feature limit
            
        Returns:
            Dict with 'places' and 'buildings' keys containing respective GeoDataFrames
        """
        con = self._get_connection()
        start_time = time.time()
        result = {}
        
        # Query 1: Places (points)
        logging.info("Fetching places data (points)...")
        places_query = Query(
            name=f"{query.name}_places",
            theme="places",
            target_type="place",
            is_multilayer=False,
            overture_type="place",
            category_filter=query.category_filter,
            field_mappings=query.field_mappings
        )
        
        places_gdf = self._execute_single_query(con, places_query, country, clip, limit)
        if len(places_gdf) > 0:
            places_gdf['source_type'] = 'place'
            result['places'] = places_gdf
            logging.info(f"Fetched {len(places_gdf):,} places")
        
        # Query 2: Buildings (polygons) - check if building_filter exists in original query data
        buildings_filter = None
        if hasattr(query, '_original_config'):
            buildings_filter = query.original_config.get('building_filter')
        
        if buildings_filter:
            logging.info("Fetching buildings data (polygons)...")
            buildings_query = Query(
                name=f"{query.name}_buildings",
                theme="buildings",
                target_type="building",
                is_multilayer=False,
                overture_type="building",
                filter=buildings_filter,
                field_mappings=query.field_mappings
            )
            
            buildings_gdf = self._execute_single_query(con, buildings_query, country, clip, limit)
            if len(buildings_gdf) > 0:
                buildings_gdf['source_type'] = 'building'
                result['buildings'] = buildings_gdf
                
                # Debug: Check building geometries
                building_geom_types = buildings_gdf.geometry.geom_type.value_counts().to_dict()
                logging.info(f"Fetched {len(buildings_gdf):,} buildings with geometry types: {building_geom_types}")
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
        
    def _execute_single_query(self, con: duckdb.DuckDBPyConnection, query: Query, country: Country, clip: ClipStrategy, limit: Optional[int] = None) -> gpd.GeoDataFrame:
        """Execute a single query and return GeoDataFrame."""
        # Use unified spatial query builder
        sql = self._build_spatial_query(query, country, clip, limit)
        
        return self._fetch_via_temp_file(con, sql)
        
    def export_direct(self, query: Query, country: Country, output_path: Path, export_format: str, clip: ClipStrategy = ClipStrategy.DIVISIONS, limit: Optional[int] = None) -> None:
        """Export data directly from DuckDB to file using GDAL drivers (following Overture docs pattern)."""
        logging.info(f"Direct export to {export_format.upper()}: {output_path}")
        
        con = self._setup_duckdb_optimized()
        try:
            # Build the same spatial query we use for fetching
            sql = self._build_spatial_query(query, country, clip, limit)
            
            logging.info("Executing direct export with optimized spatial query...")
            logging.info(f"SQL preview: {sql[:200]}...")
            start_time = time.time()
            
            # Export directly to final format (no intermediate steps)
            self._export_direct_to_file(con, sql, output_path, export_format)
            
            elapsed = time.time() - start_time
            logging.info(f"Direct export completed in {elapsed:.1f} seconds")
            
        finally:
            con.close()
    
    def _export_direct_to_file(self, con: duckdb.DuckDBPyConnection, sql: str, output_path: Path, export_format: str) -> None:
        """Direct export from DuckDB to final format using GDAL drivers (following Overture docs pattern)."""
        
        # Format to GDAL driver mapping
        format_drivers = {
            'geojson': 'GeoJSON',
            'geojsonseq': 'GeoJSONSeq', 
            'gpkg': 'GPKG',
            'fgdb': 'OpenFileGDB',
            'shp': 'ESRI Shapefile',
            'fgb': 'FlatGeobuf'
        }
        
        driver = format_drivers.get(export_format.lower())
        if not driver:
            raise ValueError(f"Unsupported export format: {export_format}. Supported: {list(format_drivers.keys())}")
        
        try:
            # Handle DuckDB variables separately from main query (if present)
            if "SET variable" in sql:
                # Split variable declarations from main SELECT query
                pattern_end = ");"
                pattern_start = "SELECT"
                
                pos = 0
                variable_end = -1
                while pos < len(sql):
                    pos = sql.find(pattern_end, pos)
                    if pos == -1:
                        break
                    after_pattern = sql[pos + len(pattern_end):].strip()
                    if after_pattern.startswith(pattern_start):
                        variable_end = pos + len(pattern_end)
                        break
                    pos += len(pattern_end)
                
                if variable_end != -1:
                    variable_part = sql[:variable_end].strip()
                    select_part = sql[variable_end:].strip()
                    
                    # Execute variable declarations first
                    logging.debug("Executing DuckDB variable declarations...")
                    con.execute(variable_part)
                    
                    # Use only the SELECT part for COPY
                    sql = select_part
            
            # Direct export using GDAL (just like Overture docs)
            export_sql = f"""
            COPY (
                {sql}
            ) TO '{output_path}' WITH (FORMAT GDAL, DRIVER '{driver}')
            """
            
            logging.debug(f"Exporting directly to {export_format.upper()} using GDAL driver: {driver}")
            con.execute(export_sql)
            
        except Exception as e:
            logging.error(f"Direct export failed: {e}")
            raise

    def _fetch_via_temp_file(self, con: duckdb.DuckDBPyConnection, sql: str) -> gpd.GeoDataFrame:
        """Optimized two-step processing following Overture documentation recommendations."""
        # Get PID-isolated temp directory
        temp_dir = get_pid_temp_dir()
        temp_file = temp_dir / f"overture_temp_{os.getpid()}_{int(time.time())}.parquet"
        
        try:
            # Handle DuckDB variables separately from main query (following deprecated duck.py pattern)
            if "SET variable" in sql:
                # Split variable declarations from main SELECT query
                pattern_end = ");"
                pattern_start = "SELECT"
                
                pos = 0
                variable_end = -1
                while pos < len(sql):
                    pos = sql.find(pattern_end, pos)
                    if pos == -1:
                        break
                    after_pattern = sql[pos + len(pattern_end):].strip()
                    if after_pattern.startswith(pattern_start):
                        variable_end = pos + len(pattern_end)
                        break
                    pos += len(pattern_end)
                
                if variable_end != -1:
                    variable_part = sql[:variable_end].strip()
                    select_part = sql[variable_end:].strip()
                    
                    # Execute variable declarations first
                    logging.debug("Executing DuckDB variable declarations...")
                    con.execute(variable_part)
                    
                    # Use only the SELECT part for COPY
                    sql = select_part
            
            # Step 1: Export filtered data to temporary parquet file
            export_sql = f"""
            COPY ({sql}) TO '{temp_file}' 
            (FORMAT 'PARQUET', ROW_GROUP_SIZE 100000, COMPRESSION 'ZSTD')
            """
            
            logging.debug("Step 1: Exporting filtered data to temporary file...")
            con.execute(export_sql)
            
            # Step 2: Load parquet and convert geometries (following deprecated duck.py pattern)
            logging.debug("Step 2: Loading parquet and converting geometries...")
            
            # Use pandas.read_parquet directly (like Overture docs and working deprecated code)
            df = pd.read_parquet(temp_file)
            
            if len(df) == 0:
                logging.warning("Query returned 0 features")
                return gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
            
            # Convert geometry column to shapely geometries (consistent with Overture schema)
            if "geometry" in df.columns:
                # Handle both native geometry and WKB bytes
                import shapely.wkb as swkb
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
                
                df["geometry"] = df["geometry"].apply(convert_geometry)
                gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            else:
                raise ValueError("No geometry column found in temp file")
            return gdf
            
        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
                
    def read(
        self, 
        query: Query,
        country: Country, 
        clip: ClipStrategy, 
        raw: bool = False
    ) -> gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame]:
        """
        Read Overture data for specified query and country.
        
        Implements cache->dump->S3 fallback logic with robust error handling:
        1. First try cache (if enabled)
        2. Then try local dumps (if available) 
        3. Finally fall back to direct S3 query
        
        Args:
            query: Query configuration with theme and filters
            country: Country information with bounds
            clip: Clipping strategy (divisions or bbox)
            raw: If True, skip transform-level enrichments downstream
            
        Returns:
            GeoDataFrame for single-layer queries, or Dict of GeoDataFrames for multi-layer queries
        """
        # Implement cache->dump->S3 fallback logic
        
        # Check if this is a multi-layer query (places + buildings)
        if query.is_multilayer:
            logging.info("Multi-layer query detected: fetching both places and buildings")
            return self._read_multilayer_with_fallback(query, country, clip)
        
        # Single-layer query with fallback logic
        return self._read_single_layer_with_fallback(query, country, clip)
    
    def _read_single_layer_with_fallback(self, query: Query, country: Country, clip: ClipStrategy) -> gpd.GeoDataFrame:
        """
        Read single-layer data with cache->dump->S3 fallback logic.
        """
        # Step 1: Try cache first (if enabled)
        if self._enable_cache:
            try:
                logging.debug(f"Trying cache for {country.iso2}/{query.theme}/{query.type}")
                configured_release = self.cfg.get('overture', {}).get('release')
                if not configured_release:
                    raise ValueError("No Overture release configured. Set OVERTURE_RELEASE environment variable.")
                cache_query = CacheQuery(
                    country=country.iso2,
                    theme=query.theme,
                    type_name=query.type,
                    release=configured_release,
                    use_divisions=clip == ClipStrategy.DIVISIONS,
                    limit=self.run.limit,
                    filters=query.filter
                )
                cached_data = self.get_cached_data(cache_query)
                if cached_data is not None:
                    logging.info(f"Cache hit: using cached data for {country.iso2}/{query.theme}/{query.type}")
                    return cached_data
                    
                logging.debug(f"Cache miss for {country.iso2}/{query.theme}/{query.type}")
            except Exception as e:
                logging.warning(f"Cache read failed: {e}, falling back to next method")
        
        # Step 2: Try local dumps (if enabled and available)
        if self._use_local_dumps:
            try:
                logging.debug(f"Trying local dumps for {country.iso2}/{query.theme}/{query.type}")
                
                # Check if dump exists for the theme
                dump_release = self.cfg.get('overture', {}).get('release')
                if not dump_release:
                    raise ValueError("No Overture release configured. Set OVERTURE_RELEASE environment variable.")
                if self._check_dump_exists(dump_release, query.theme):
                    logging.info(f"Local dump available for {query.theme}, using dump-based query")
                    gdf = self._read_from_dump(
                        theme=query.theme,
                        type_name=query.type,
                        country=country.iso2,
                        release=dump_release,
                        limit=self.run.limit,
                        filters=query.filter
                    )
                    return gdf
                else:
                    logging.debug(f"No local dump available for {query.theme}")
            except Exception as e:
                logging.warning(f"Local dump read failed: {e}, falling back to S3")
        
        # Step 3: Fall back to direct S3 query (original logic)
        logging.info(f"Using direct S3 query for {country.iso2}/{query.theme}/{query.type}")
        gdf = self._execute_s3_query_with_retry(query, country, clip)
        
        # Cache the result for future use (essential for overture-dump functionality)
        if self._enable_cache and gdf is not None and len(gdf) > 0:
            try:
                logging.info(f"Caching S3 query result for future use: {country.iso2}/{query.theme}/{query.type}")
                cache_query = CacheQuery(
                    country=country.iso2,
                    theme=query.theme,
                    type_name=query.type,
                    release=configured_release,
                    use_divisions=clip == ClipStrategy.DIVISIONS,
                    limit=None,  # Cache full data, not limited
                    filters=None  # Cache full data, not filtered
                )
                # Save to cache (create a copy to avoid modifying the original)
                cache_file = self._get_cache_file_path(cache_query)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                gdf.to_parquet(cache_file, compression='zstd')
                
                # Create metadata
                metadata = {
                    'country': cache_query.country,
                    'theme': cache_query.theme, 
                    'type': cache_query.type_name,
                    'release': cache_query.release,
                    'features': len(gdf),
                    'cached_at': datetime.now().isoformat(),
                    'use_divisions': cache_query.use_divisions
                }
                
                # Write metadata
                metadata_file = cache_file.with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logging.info(f"Cached {len(gdf):,} features to {cache_file}")
                
            except Exception as e:
                logging.warning(f"Failed to cache S3 result: {e}")
                # Continue execution even if caching fails
        
        return gdf
    
    def _read_multilayer_with_fallback(self, query: Query, country: Country, clip: ClipStrategy) -> dict[str, gpd.GeoDataFrame]:
        """
        Read multi-layer data with cache->dump->S3 fallback logic.
        """
        # For multi-layer queries, we need to read both places and buildings
        themes = ['places', 'buildings']
        types = ['place', 'building']
        results = {}
        
        for theme, type_name in zip(themes, types, strict=False):
            # Create a single-layer query for each theme with appropriate filter
            if theme == 'buildings':
                # Use building_filter for buildings (e.g., "subtype = 'education'")
                filter_to_use = query.building_filter
            else:
                # Use main filter for places (e.g., "categories.primary = 'education'")
                filter_to_use = query.filter
                
            single_query = Query(
                theme=theme,
                type=type_name,
                filter=filter_to_use,
                name=f"{query.name}_{theme}",
                is_multilayer=False
            )
            
            # Use single-layer fallback logic for each component
            gdf = self._read_single_layer_with_fallback(single_query, country, clip)
            results[theme] = gdf
        
        return results
    
    def _execute_s3_query_with_retry(self, query: Query, country: Country, clip: ClipStrategy) -> gpd.GeoDataFrame:
        """
        Execute S3 query with retry logic and fallback between divisions and bbox.
        
        This is the original read() method logic for direct S3 access.
        """
        max_retries = 3
        current_clip = clip
        
        for attempt in range(max_retries):
            try:
                return self._execute_single_query_with_retry(query, country, current_clip, attempt)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"All {max_retries} attempts failed")
                    raise
                
                # If divisions query fails, fall back to bbox filtering
                if current_clip == ClipStrategy.DIVISIONS and ("divisions" in str(e).lower() or "country" in str(e).lower()):
                    logging.warning(f"Divisions query failed: {e}")
                    logging.warning("Falling back to bbox filtering for retry...")
                    current_clip = ClipStrategy.BBOX
                
                wait_time = 300 * (attempt + 1)  # Exponential backoff: 5, 10, 15 minutes
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                logging.warning(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
    def _execute_single_query_with_retry(self, query: Query, country: Country, clip: ClipStrategy, attempt: int) -> gpd.GeoDataFrame:
        """Execute a single query with error handling and fallback logic."""
        con = self._get_connection()
        
        try:
            # Determine effective clipping strategy  
            effective_clip = ClipStrategy.BBOX if self.run.use_bbox else clip
            
            if effective_clip == ClipStrategy.DIVISIONS:
                logging.info("Using Overture Divisions for precise country boundaries")
            else:
                logging.info(f"Using bbox filtering for {country.iso2}")
            
            # Use unified spatial query builder
            sql = self._build_spatial_query(query, country, effective_clip, self.run.limit)
            
            logging.info("Executing optimized query with two-step processing...")
            logging.info(f"SQL preview: {sql[:200]}...")
            start_time = time.time()
            
            # Use optimized two-step processing (recommended by Overture docs)
            gdf = self._fetch_via_temp_file(con, sql)
            
            elapsed = time.time() - start_time
            logging.info(f"Fetched {len(gdf):,} features in {elapsed:.1f} seconds")
            return gdf
            
        except Exception as e:
            # Enhanced error context
            error_msg = f"Query failed on attempt {attempt + 1}: {str(e)}"
            if attempt > 0:
                error_msg += f" (using {clip.value} clipping)"
            logging.error(error_msg)
            raise
            
    def _fetch_dual_query_with_retry(self, query: Query, country: Country, clip: ClipStrategy, limit: Optional[int], attempt: int) -> dict[str, gpd.GeoDataFrame]:
        """Execute dual query with error handling and fallback logic."""
        try:
            return self._fetch_dual_query(query, country, clip, limit)
        except Exception as e:
            # Enhanced error context for dual queries
            error_msg = f"Dual query failed on attempt {attempt + 1}: {str(e)}"
            if attempt > 0:
                error_msg += f" (using {clip.value} clipping)"
            logging.error(error_msg)
            raise
            
    def close(self):
        """Close the DuckDB connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            
    # =============================================================================
    # Dump Management Methods
    # =============================================================================
    
    def ensure_dump(self, theme: str, release: str = None, force_download: bool = False) -> Path:
        """
        Download complete theme from Overture S3 bucket if needed.
        
        Args:
            theme: Theme to download (e.g., "transportation", "buildings")
            release: Overture release version
            force_download: Force re-download even if dump exists
            
        Returns:
            Path to downloaded theme directory
            
        Raises:
            ValueError: If theme is invalid
            RuntimeError: If download fails
        """
        if not self._use_local_dumps:
            raise RuntimeError("Local dumps are disabled in configuration")
        
        # Use configured release if none provided
        if release is None:
            release = self.cfg.get('overture', {}).get('release')
            if not release:
                raise ValueError("No Overture release configured. Set OVERTURE_RELEASE environment variable.")
            
        if theme not in self.OVERTURE_THEMES:
            raise ValueError(f"Invalid theme: {theme}. Valid themes: {self.OVERTURE_THEMES}")
        
        dump_path = self._dump_base_path / release / f"theme={theme}"
        dump_path.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if self._check_dump_exists(release, theme):
            if force_download:
                logging.info(f"Force download requested - removing existing dump: {dump_path}")
                self._delete_dump(release, theme)
                # Recreate directory after deletion
                dump_path.mkdir(parents=True, exist_ok=True)
            else:
                logging.info(f"Dump already exists: {dump_path}")
                return dump_path
        
        # Provide theme size estimates to set user expectations
        theme_size_estimates = {
            "transportation": "~50GB",
            "buildings": "~100GB", 
            "places": "~20GB",
            "divisions": "~5GB",
            "addresses": "~30GB",
            "base": "~15GB"
        }
        estimated_size = theme_size_estimates.get(theme, "~25GB")
        
        logging.info(f"Downloading {theme} theme for release {release}")
        logging.info(f"Expected download size: {estimated_size} (this may take 20-60 minutes)")
        logging.info("Download is active - progress updates will appear every few minutes...")
        start_time = time.time()
        
        # Setup DuckDB connection for download
        con = self._setup_download_connection()
        
        try:
            # Get theme types from S3 structure
            theme_types = self._get_theme_types(con, release, theme)
            total_types = len(theme_types)
            
            logging.info(f"Theme contains {total_types} data type(s): {', '.join(theme_types)}")
            total_size_gb = 0.0
            
            for i, type_name in enumerate(theme_types, 1):
                type_start = time.time()
                logging.info(f"[{i}/{total_types}] Starting download: {theme}/{type_name}")
                logging.info("DuckDB is actively downloading from S3 - please wait...")
                
                source_pattern = f"s3://overturemaps-us-west-2/release/{release}/theme={theme}/type={type_name}/*.parquet"
                target_dir = dump_path / f"type={type_name}"
                target_dir.mkdir(exist_ok=True)
                target_file = target_dir / "data.parquet"
                
                # Download using DuckDB COPY for efficiency
                download_sql = f"""
                COPY (
                    SELECT * FROM read_parquet('{source_pattern}')
                ) TO '{target_file}' 
                (FORMAT PARQUET, COMPRESSION {self._compression.upper()}, ROW_GROUP_SIZE 100000)
                """
                
                # Execute download with periodic progress indicators
                logging.info(f"Executing download query for {type_name}...")
                con.execute(download_sql)
                
                # Track completion and size
                if target_file.exists():
                    size_gb = target_file.stat().st_size / (1024**3)
                    total_size_gb += size_gb
                    type_elapsed = time.time() - type_start
                    logging.info(f"✓ Completed {type_name}: {size_gb:.2f} GB in {type_elapsed/60:.1f} minutes")
                else:
                    logging.error(f"✗ Failed to download {type_name}: target file not created")
                
                # Progress update
                elapsed_total = time.time() - start_time
                if i < total_types:
                    logging.info(f"Progress: {i}/{total_types} types completed ({total_size_gb:.1f} GB so far, {elapsed_total/60:.1f} min elapsed)")
                    logging.info("Continuing to next data type...")
            
            # Post-processing steps
            logging.info("Download completed! Starting post-processing...")
            
            # Create metadata file
            metadata = {
                'release': release,
                'theme': theme,
                'download_date': datetime.now().isoformat(),
                'size_gb': total_size_gb,
                'is_complete': True,
                'theme_types': theme_types
            }
            
            metadata_file = dump_path / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2))
            
            # Update cache
            self._dump_metadata_cache[f"{release}:{theme}"] = metadata
            
            elapsed_total = time.time() - start_time
            logging.info("Download completed successfully!")
            logging.info(f"Total size: {total_size_gb:.2f} GB")
            logging.info(f"Total time: {elapsed_total/60:.1f} minutes")
            logging.info(f"Location: {dump_path}")
            
            return dump_path
            
        except Exception as e:
            logging.error(f"Download failed: {e}")
            # Clean up partial download
            self._delete_dump(release, theme)
            raise RuntimeError(f"Failed to download {theme} theme: {e}") from e
        finally:
            con.close()
    
    def _check_dump_exists(self, release: str, theme: str) -> bool:
        """
        Check if dump exists for release and theme.
        
        Args:
            release: Overture release version (e.g., "2025-07-23.0")
            theme: Overture theme name
            
        Returns:
            True if dump exists and is valid
        """
        # Check cache first
        cache_key = f"{release}:{theme}"
        if cache_key in self._dump_metadata_cache:
            return self._dump_metadata_cache[cache_key].get('is_complete', False)
        
        dump_path = self._dump_base_path / release / f"theme={theme}"
        metadata_file = dump_path / "metadata.json"
        
        if not dump_path.exists() or not metadata_file.exists():
            return False
        
        try:
            metadata = json.loads(metadata_file.read_text())
            self._dump_metadata_cache[cache_key] = metadata  # Cache for future checks
            return metadata.get('is_complete', False)
        except Exception as e:
            logging.warning(f"Failed to read dump metadata: {e}")
            return False
    
    def _delete_dump(self, release: str, theme: str) -> None:
        """
        Delete existing dump for release and theme.
        
        Args:
            release: Overture release version (e.g., "2025-07-23.0")
            theme: Overture theme name
        """
        dump_path = self._dump_base_path / release / f"theme={theme}"
        
        if dump_path.exists():
            import shutil
            logging.info(f"Removing existing dump directory: {dump_path}")
            shutil.rmtree(dump_path)
            
            # Remove from cache
            cache_key = f"{release}:{theme}"
            self._dump_metadata_cache.pop(cache_key, None)
        else:
            logging.debug(f"No existing dump to remove: {dump_path}")
    
    def _setup_download_connection(self) -> duckdb.DuckDBPyConnection:
        """Setup DuckDB connection optimized for downloads."""
        con = duckdb.connect()
        
        # Install extensions
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL spatial; LOAD spatial;")
        
        # Memory configuration for downloads
        memory_limit = f"{self._max_memory_gb}GB"
        con.execute(f"SET memory_limit='{memory_limit}';")
        con.execute("SET threads=4;")  # Conservative for downloads
        
        # S3 and HTTP optimizations
        s3_region = self.cfg.get('overture', {}).get('s3_region', 'us-west-2')
        con.execute(f"SET s3_region='{s3_region}';")
        con.execute("SET http_timeout=1800000;")  # 30 minutes for large downloads
        con.execute("SET http_retries=3;")
        con.execute("SET http_keep_alive=true;")
        
        logging.debug(f"Download connection configured: {memory_limit} memory, 4 threads")
        return con
    
    def _get_theme_types(self, con: duckdb.DuckDBPyConnection, release: str, theme: str) -> list[str]:
        """Get available types for a theme using known Overture schema."""
        # Use known Overture schema - more reliable than querying S3 structure
        type_mapping = {
            "transportation": ["segment"],
            "buildings": ["building"],
            "places": ["place"],
            "addresses": ["address"],
            "base": ["infrastructure", "land", "land_use", "water"],
            "divisions": ["division", "division_boundary"]
        }
        
        theme_types = type_mapping.get(theme, [])
        if not theme_types:
            # Fall back to dynamic discovery if needed
            logging.warning(f"Unknown theme types for {theme}, attempting dynamic discovery")
            try:
                # Query S3 for available types
                query = f"""
                SELECT DISTINCT regexp_extract(filename, 'type=([^/]+)', 1) as type_name
                FROM glob('s3://overturemaps-us-west-2/release/{release}/theme={theme}/type=*/*.parquet')
                WHERE type_name IS NOT NULL
                ORDER BY type_name
                """
                result = con.execute(query).fetchall()
                theme_types = [row[0] for row in result]
                logging.info(f"Discovered types for {theme}: {theme_types}")
            except Exception as e:
                logging.error(f"Failed to discover theme types: {e}")
                raise
        
        return theme_types
    
    # =====================================================================
    # Cache Management Methods 
    # =====================================================================
    
    def get_cached_data(self, query: CacheQuery) -> Optional[gpd.GeoDataFrame]:
        """
        Get cached data for a country/theme/type combination.
        
        Args:
            query: Cache query configuration
            
        Returns:
            Cached GeoDataFrame if available, None otherwise
        """
        if not self._enable_cache:
            return None
            
        cache_file = self._get_cache_file_path(query)
        
        if not cache_file.exists():
            return None
        
        try:
            # Read cached parquet file
            gdf = gpd.read_parquet(cache_file)
            
            # Apply additional filters if specified
            if query.filters:
                if isinstance(query.filters, dict):
                    # Dictionary-based column filtering
                    for column, value in query.filters.items():
                        if column in gdf.columns:
                            gdf = gdf[gdf[column] == value]
                else:
                    # String-based SQL filter - apply after loading from cache
                    original_count = len(gdf)
                    gdf = apply_sql_filter(gdf, query.filters)
                    logging.debug(f"Applied filter '{query.filters}': {original_count:,} -> {len(gdf):,} features")
            
            # Apply limit if specified
            if query.limit:
                gdf = gdf.head(query.limit)
            
            logging.debug(f"Cache hit: {cache_file.name} ({len(gdf):,} features)")
            return gdf
            
        except Exception as e:
            logging.warning(f"Failed to read cache file {cache_file}: {e}")
            return None
    
    def cache_country_data(self, query: CacheQuery, overwrite: bool = False) -> gpd.GeoDataFrame:
        """
        Cache data for a specific country using DuckDB streaming extraction.
        
        Note: This method always caches complete country data without limits.
        Limits are applied only when retrieving from cache via get_cached_data().
        
        Args:
            query: Cache query configuration
            overwrite: Whether to overwrite existing cache
            
        Returns:
            GeoDataFrame with extracted and cached data
        """
        if not self._enable_cache:
            logging.warning("Cache is disabled, extracting data without caching")
            return self._extract_country_data_for_cache(query)
            
        cache_file = self._get_cache_file_path(query)
        
        # Check if cache exists and is valid
        if not overwrite and cache_file.exists():
            logging.info(f"Cache exists for {query.country}/{query.theme}/{query.type_name}")
            cached_data = self.get_cached_data(query)
            if cached_data is not None:
                return cached_data
        
        logging.info(f"Extracting and caching data for {query.country}/{query.theme}/{query.type_name}")
        start_time = time.time()
        
        # Extract data using proven DuckDB streaming approach (always complete data)
        # Create extraction query without limits for caching
        extraction_query = CacheQuery(
            country=query.country,
            theme=query.theme,
            type_name=query.type_name,
            release=query.release,
            use_divisions=query.use_divisions,
            limit=None,  # Never apply limits when caching
            filters=None  # Never apply filters when caching - cache complete data
        )
        gdf = self._extract_country_data_for_cache(extraction_query)
        
        if gdf.empty:
            logging.warning(f"No data found for {query.country}/{query.theme}/{query.type_name}")
            return gdf
        
        # Cache the data
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(cache_file, compression='zstd')
        
        # Create metadata
        metadata = CacheMetadata(
            country=query.country,
            release=query.release,
            theme=query.theme,
            type_name=query.type_name,
            cached_date=datetime.now().isoformat(),
            feature_count=len(gdf),
            size_mb=cache_file.stat().st_size / (1024 * 1024),
            bbox=tuple(gdf.total_bounds)
        )
        
        # Save metadata
        metadata_file = cache_file.with_suffix('.json')
        metadata_file.write_text(json.dumps(metadata.to_dict(), indent=2))
        
        elapsed = time.time() - start_time
        logging.debug(f"Cached {len(gdf):,} features in {elapsed:.1f}s: {cache_file.name}")
        
        # Apply limit to the result if specified in original query
        if query.limit:
            gdf = gdf.head(query.limit)
            logging.debug(f"Applied limit: returning {len(gdf):,} features")
        
        return gdf
    
    def _extract_country_data_for_cache(self, query: CacheQuery) -> gpd.GeoDataFrame:
        """
        Extract country data using DuckDB streaming (reuses existing patterns).
        """
        con = self._get_connection()
        
        try:
            # Build target configuration for the query
            # Never apply filters during caching - always extract complete country data
            target_config = {
                'theme': query.theme,
                'type': query.type_name,
                'filter': None  # No filters during caching - extract ALL data for the country
            }
            
            # Build selector configuration
            country_data = CountryRegistry.get_country(query.country)
            if not country_data:
                raise ValueError(f"Unknown country: {query.country}")
            country_iso2 = country_data.iso2
            selector = {'iso2': country_iso2}
            
            # Create domain objects for the new _build_spatial_query method
            domain_query = Query(
                theme=query.theme,
                type=query.type_name,
                filter=None,  # No filters during caching - extract ALL data for the country
                name=f"{query.theme}_{query.type_name}",
                is_multilayer=False,
                building_filter=None,
                category_filter=None,
                field_mappings={},
                original_config=target_config
            )
            
            domain_country = Country(
                name=country_data.name,
                iso2=country_data.iso2,
                iso3=country_data.iso3,
                bounds=CountryRegistry.get_bounding_boxes().get(country_data.iso2, (0, 0, 0, 0))
            )
            
            # Use the new unified spatial query builder
            clip_strategy = ClipStrategy.DIVISIONS if query.use_divisions else ClipStrategy.BBOX
            sql = self._build_spatial_query(
                query=domain_query,
                country=domain_country,
                clip_strategy=clip_strategy,
                limit=None  # No limit for caching - extract all country data
            )
            
            # Execute query using the proven two-step approach
            logging.debug(f"Executing cache extraction query: {sql[:200]}...")
            
            # Use the same reliable geometry handling as the working pipeline
            gdf = self._fetch_via_temp_file(con, sql)
            
            return gdf
            
        except Exception as e:
            logging.error(f"Failed to extract country data for cache: {e}")
            raise
    
    def _get_cache_file_path(self, query: CacheQuery) -> Path:
        """Generate cache file path for a query using {iso2}_{sector} naming."""
        country_data = CountryRegistry.get_country(query.country)
        if not country_data:
            raise ValueError(f"Unknown country: {query.country}")
        country_iso2 = country_data.iso2
        
        # Map theme/type to sector name for cleaner file naming
        # This maps from Overture themes to logical sector names
        sector_map = {
            ('transportation', 'segment'): 'roads',
            ('buildings', 'building'): 'buildings', 
            ('places', 'place'): 'places',
            ('education', 'place'): 'education',  # Dual query
            ('health', 'place'): 'health',        # Dual query
            ('markets', 'place'): 'markets'       # Dual query
        }
        
        sector_name = sector_map.get((query.theme, query.type_name), f"{query.theme}_{query.type_name}")
        
        return (self._cache_dir / query.release / country_iso2 / 
                f"{country_iso2}_{sector_name}.parquet")
    
    def list_cached_countries(self, release: str = "latest") -> list[CacheMetadata]:
        """
        List all cached countries for a release.
        
        Args:
            release: Release version or "latest"
            
        Returns:
            List of cache metadata entries
        """
        if not self._enable_cache:
            return []
            
        if release == "latest":
            release = self._get_latest_cache_release()
        
        release_dir = self._cache_dir / release
        if not release_dir.exists():
            return []
        
        cached_entries = []
        
        for country_dir in release_dir.iterdir():
            if not country_dir.is_dir():
                continue
            
            for metadata_file in country_dir.glob("*.json"):
                try:
                    metadata_data = json.loads(metadata_file.read_text())
                    metadata = CacheMetadata.from_dict(metadata_data)
                    cached_entries.append(metadata)
                except Exception as e:
                    logging.warning(f"Failed to read metadata {metadata_file}: {e}")
        
        return cached_entries
    
    def clear_cache(self, country: Optional[str] = None, 
                   release: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            country: Specific country to clear (optional)
            release: Specific release to clear (optional)
        """
        if not self._enable_cache:
            logging.warning("Cache is disabled")
            return
            
        if release:
            release_dir = self._cache_dir / release
            if country:
                country_data = CountryRegistry.get_country(country)
                if not country_data:
                    raise ValueError(f"Unknown country: {country}")
                country_iso2 = country_data.iso2
                country_dir = release_dir / country_iso2
                if country_dir.exists():
                    import shutil
                    shutil.rmtree(country_dir)
                    logging.info(f"Cleared cache for {country} ({country_iso2}) in {release}")
            else:
                if release_dir.exists():
                    import shutil
                    shutil.rmtree(release_dir)
                    logging.info(f"Cleared all cache for release {release}")
        else:
            if self._cache_dir.exists():
                import shutil
                shutil.rmtree(self._cache_dir)
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                logging.info("Cleared entire cache")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if not self._enable_cache:
            return {
                'total_size_mb': 0,
                'total_files': 0,
                'countries': 0,
                'releases': 0,
                'cache_path': str(self._cache_dir),
                'cache_enabled': False
            }
            
        total_size_mb = 0
        total_files = 0
        countries = set()
        releases = set()
        
        for metadata in self.list_cached_countries():
            total_size_mb += metadata.size_mb
            total_files += 1
            countries.add(metadata.country)
            releases.add(metadata.release)
        
        return {
            'total_size_mb': total_size_mb,
            'total_files': total_files,
            'countries': len(countries),
            'releases': len(releases),
            'cache_path': str(self._cache_dir),
            'cache_enabled': True
        }
    
    def _get_latest_cache_release(self) -> str:
        """Get the latest release version from cache directories."""
        if not self._cache_dir.exists():
            configured_release = self.cfg.get('overture', {}).get('release')
            if not configured_release:
                raise ValueError("No Overture release configured. Set OVERTURE_RELEASE environment variable.")
            return configured_release
            
        releases = []
        for item in self._cache_dir.iterdir():
            if item.is_dir() and item.name != "latest":
                releases.append(item.name)
        
        if not releases:
            configured_release = self.cfg.get('overture', {}).get('release')
            if not configured_release:
                raise ValueError("No Overture release configured. Set OVERTURE_RELEASE environment variable.")
            return configured_release
        
        # Sort releases and return the latest
        releases.sort(reverse=True)
        return releases[0]
    
    # =====================================================================
    # Local Dump Query Methods 
    # =====================================================================
    
    def _read_from_dump(self, theme: str, type_name: str, country: str, 
                       release: str = None, limit: Optional[int] = None, 
                       filters: Optional[dict | str] = None, 
                       force_refresh: bool = False) -> gpd.GeoDataFrame:
        """
        Read data from local dumps using country-specific caching.
        
        This method integrates the query_local_dump logic, using cache as the 
        primary storage mechanism for country-level data instead of full theme dumps.
        
        Args:
            theme: Overture theme name  
            type_name: Overture type name
            country: Country identifier (name, ISO2, or ISO3)
            release: Overture release version
            limit: Optional limit on number of features
            filters: Optional filters to apply (dict or SQL string)
            force_refresh: Force refresh of cached data
            
        Returns:
            GeoDataFrame with requested data
        """
        if not self._use_local_dumps:
            raise ValueError("Local dumps are disabled")
            
        # Use configured release if none provided
        if release is None:
            release = self.cfg.get('overture', {}).get('release')
            if not release:
                raise ValueError("No Overture release configured. Set OVERTURE_RELEASE environment variable.")
            
        if not country:
            raise ValueError("Country must be specified for dump-based queries")
        
        # Create cache query (cache stores complete data without limits)
        cache_query = CacheQuery(
            country=country,
            theme=theme,
            type_name=type_name,
            release=release,
            use_divisions=self.run.clip == ClipStrategy.DIVISIONS,
            limit=limit,  # Apply limit only on retrieval, not caching
            filters=filters
        )
        
        # Check for force refresh first
        if force_refresh:
            logging.info(f"Force refresh requested for {country}/{theme}/{type_name}")
            gdf = self.cache_country_data(cache_query, overwrite=True)
            return gdf
        
        # Try to get from cache first
        cached_data = self.get_cached_data(cache_query)
        if cached_data is not None:
            logging.info(f"Using cached data for {country}/{theme}/{type_name}")
            return cached_data
        
        # Cache miss - need to extract and cache
        logging.info(f"Cache miss for {country}/{theme}/{type_name} - extracting data")
        
        # Extract and cache data using proven DuckDB streaming approach
        gdf = self.cache_country_data(cache_query)
        
        return gdf
    
    def _apply_target_filter(self, gdf: gpd.GeoDataFrame, target_filter: str) -> gpd.GeoDataFrame:
        """
        Apply target-specific filter to GeoDataFrame using same logic as duck.py.
        
        Args:
            gdf: GeoDataFrame to filter
            target_filter: Filter string to apply
            
        Returns:
            Filtered GeoDataFrame
        """
        try:
            import re

            import pandas as pd
            
            if 'categories.primary =' in target_filter:
                # Handle categories.primary = 'value' filters
                match = re.search(r"categories\.primary = '([^']+)'", target_filter)
                if match:
                    allowed_category = match.group(1)
                    if 'category_primary' in gdf.columns:
                        # Use normalized column if available
                        gdf = gdf[gdf['category_primary'] == allowed_category]
                    elif 'categories' in gdf.columns:
                        # Use raw Overture categories column (JSON)
                        def has_primary_category(categories_json, target_cat):
                            if pd.isna(categories_json) or categories_json is None:
                                return False
                            if isinstance(categories_json, dict) and 'primary' in categories_json:
                                return categories_json['primary'] == target_cat
                            return False
                        
                        gdf = gdf[gdf['categories'].apply(lambda x: has_primary_category(x, allowed_category))]
                        
            elif 'categories.primary IN' in target_filter:
                # Handle categories.primary IN (...) filters
                match = re.search(r"categories\.primary IN \('([^']+)'\)", target_filter)
                if match:
                    allowed_categories = match.group(1).split("', '")
                    if 'category_primary' in gdf.columns:
                        # Use normalized column if available
                        gdf = gdf[gdf['category_primary'].isin(allowed_categories)]
                    elif 'categories' in gdf.columns:
                        # Use raw Overture categories column (JSON)
                        def has_primary_category_in_list(categories_json, target_cats):
                            if pd.isna(categories_json) or categories_json is None:
                                return False
                            if isinstance(categories_json, dict) and 'primary' in categories_json:
                                return categories_json['primary'] in target_cats
                            return False
                        
                        gdf = gdf[gdf['categories'].apply(lambda x: has_primary_category_in_list(x, allowed_categories))]
                        
            elif 'subtype =' in target_filter:
                # Handle subtype = 'value' filters
                match = re.search(r"subtype = '([^']+)'", target_filter)
                if match:
                    allowed_subtype = match.group(1)
                    if 'subtype' in gdf.columns:
                        gdf = gdf[gdf['subtype'] == allowed_subtype]
                        
            elif 'subtype IN' in target_filter:
                # Handle subtype IN (...) filters
                match = re.search(r"subtype IN \('([^']+)'\)", target_filter)
                if match:
                    allowed_subtypes = match.group(1).split("', '")
                    if 'subtype' in gdf.columns:
                        gdf = gdf[gdf['subtype'].isin(allowed_subtypes)]
            
            logging.debug(f"Applied target filter '{target_filter}': {len(gdf)} features remaining")
            return gdf
            
        except Exception as e:
            logging.warning(f"Failed to apply target filter '{target_filter}': {e}")
            return gdf