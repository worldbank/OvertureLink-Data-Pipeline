"""
Local Overture Maps dump management for improved performance.

Manages downloading, storing, and querying complete Overture theme data locally
to eliminate redundant S3 access for multi-country operations.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Union

import duckdb
import geopandas as gpd

from .config.countries import CountryRegistry
from .cache_manager import CountryCacheManager, CacheQuery

logger = logging.getLogger(__name__)


@dataclass
class DumpMetadata:
    """Metadata for a local Overture dump."""
    release: str
    themes: List[str]
    download_date: str
    size_gb: float
    is_complete: bool
    spatial_index_built: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'release': self.release,
            'themes': self.themes,
            'download_date': self.download_date,
            'size_gb': self.size_gb,
            'is_complete': self.is_complete,
            'spatial_index_built': self.spatial_index_built
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DumpMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DumpQuery:
    """Query configuration for local dump."""
    theme: str
    type: str
    country: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    filters: Optional[Dict] = None
    limit: Optional[int] = None


@dataclass
class DumpConfig:
    """Configuration for local dump operations."""
    base_path: str = "/overturedump"
    max_memory_gb: int = 16  # Reduced from 32GB for optimization
    chunk_size: int = 5  # Countries per chunk
    enable_spatial_index: bool = True
    compression: str = "zstd"
    partitioning: str = "hive"
    # Cache optimization settings
    enable_spatial_hash: bool = True
    use_pyarrow_queries: bool = True
    # Boundary optimization settings
    use_world_bank_boundaries: bool = True
    boundary_simplify_tolerance: float = 0.01
    enable_boundary_cache: bool = True


class DumpManager:
    """
    Handles downloading complete themes from S3, storing them locally as Parquet,
    and providing spatial queries for multiple countries/queries.
    """
    
    # Overture's six official themes
    OVERTURE_THEMES = [
        "addresses", "base", "buildings", 
        "divisions", "places", "transportation"
    ]
    
    def __init__(self, base_path: Optional[Path] = None, config: Optional[DumpConfig] = None):
        """
        Initialize dump manager.
        
        Args:
            base_path: Base directory for dumps (default: /overturedump)
            config: Dump configuration
        """
        self.config = config or DumpConfig()
        self.base_path = Path(base_path or self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        
        # Initialize country cache manager for efficient caching
        cache_dir = self.base_path / "country_cache"
        self._cache_manager = CountryCacheManager(cache_dir)
        
        # Initialize boundary manager if enabled
        self._boundary_manager: Optional = None
        if self.config.enable_boundary_cache:
            try:
                from .boundaries import BoundaryManager
                boundary_cache_dir = self.base_path / "boundaries_cache"
                self._boundary_manager = BoundaryManager(boundary_cache_dir)
                logger.debug("Boundary manager initialized")
            except ImportError as e:
                logger.warning(f"Could not initialize boundary manager: {e}")
        
        logger.info(f"DumpManager initialized: {self.base_path}")
    
    def check_dump_exists(self, release: str, theme: str) -> bool:
        """
        Check if dump exists for release and theme.
        
        Args:
            release: Overture release version (e.g., "2025-07-23.0")
            theme: Overture theme name
            
        Returns:
            True if dump exists and is valid
        """
        dump_path = self.base_path / release / f"theme={theme}"
        metadata_file = dump_path / "metadata.json"
        
        if not dump_path.exists() or not metadata_file.exists():
            return False
        
        try:
            metadata = json.loads(metadata_file.read_text())
            return metadata.get('is_complete', False)
        except Exception as e:
            logger.warning(f"Failed to read dump metadata: {e}")
            return False
    
    def _delete_dump(self, release: str, theme: str) -> None:
        """
        Delete existing dump for release and theme.
        
        Args:
            release: Overture release version (e.g., "2025-07-23.0")
            theme: Overture theme name
        """
        dump_path = self.base_path / release / f"theme={theme}"
        
        if dump_path.exists():
            import shutil
            logger.info(f"Removing existing dump directory: {dump_path}")
            shutil.rmtree(dump_path)
        else:
            logger.debug(f"No existing dump to remove: {dump_path}")
    
    def download_theme(self, release: str, theme: str, validate: bool = True, force_download: bool = False) -> Path:
        """
        Download complete theme from Overture S3 bucket.
        
        Args:
            release: Overture release version
            theme: Theme to download
            validate: Whether to validate after download
            force_download: Force re-download even if dump exists
            
        Returns:
            Path to downloaded theme directory
        """
        if theme not in self.OVERTURE_THEMES:
            raise ValueError(f"Invalid theme: {theme}. Valid themes: {self.OVERTURE_THEMES}")
        
        dump_path = self.base_path / release / f"theme={theme}"
        dump_path.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if self.check_dump_exists(release, theme):
            if force_download:
                logger.info(f"Force download requested - removing existing dump: {dump_path}")
                self._delete_dump(release, theme)
                # Recreate directory after deletion
                dump_path.mkdir(parents=True, exist_ok=True)
            else:
                logger.info(f"Dump already exists: {dump_path}")
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
        
        logger.info(f"Downloading {theme} theme for release {release}")
        logger.info(f"Expected download size: {estimated_size} (this may take 20-60 minutes)")
        logger.info("Download is active - progress updates will appear every few minutes...")
        start_time = time.time()
        
        # Setup DuckDB connection for download
        con = self._setup_download_connection()
        
        try:
            # Get theme types from S3 structure
            theme_types = self._get_theme_types(con, release, theme)
            total_types = len(theme_types)
            
            logger.info(f"Theme contains {total_types} data type(s): {', '.join(theme_types)}")
            total_size_gb = 0.0
            
            for i, type_name in enumerate(theme_types, 1):
                type_start = time.time()
                logger.info(f"[{i}/{total_types}] Starting download: {theme}/{type_name}")
                logger.info("DuckDB is actively downloading from S3 - please wait...")
                
                source_pattern = f"s3://overturemaps-us-west-2/release/{release}/theme={theme}/type={type_name}/*.parquet"
                target_dir = dump_path / f"type={type_name}"
                target_dir.mkdir(exist_ok=True)
                target_file = target_dir / "data.parquet"
                
                # Download using DuckDB COPY for efficiency
                download_sql = f"""
                COPY (
                    SELECT * FROM read_parquet('{source_pattern}')
                ) TO '{target_file}' 
                (FORMAT PARQUET, COMPRESSION {self.config.compression.upper()}, ROW_GROUP_SIZE 100000)
                """
                
                # Execute download with periodic progress indicators
                logger.info(f"Executing download query for {type_name}...")
                con.execute(download_sql)
                
                # Track completion and size
                if target_file.exists():
                    size_gb = target_file.stat().st_size / (1024**3)
                    total_size_gb += size_gb
                    type_elapsed = time.time() - type_start
                    logger.info(f"✓ Completed {type_name}: {size_gb:.2f} GB in {type_elapsed/60:.1f} minutes")
                else:
                    logger.error(f"✗ Failed to download {type_name}: target file not created")
                
                # Progress update
                elapsed_total = time.time() - start_time
                if i < total_types:
                    logger.info(f"Progress: {i}/{total_types} types completed ({total_size_gb:.1f} GB so far, {elapsed_total/60:.1f} min elapsed)")
                    logger.info("Continuing to next data type...")
            
            # Post-processing steps
            logger.info("Download completed! Starting post-processing...")
            
            # Always ensure divisions are available for spatial operations
            if theme != "divisions":
                logger.info("Ensuring divisions theme is available for spatial operations...")
                self._ensure_divisions_available(con, release)
            
            # Build spatial indexes for performance
            if self.config.enable_spatial_index:
                if theme == "divisions":
                    logger.info("Building spatial index for divisions...")
                    self._build_spatial_index(con, release)
                elif theme in ["transportation", "buildings", "places"]:
                    logger.info(f"Building bbox index for {theme}...")
                    self._build_bbox_index(con, release, theme)
            
            # Write metadata
            logger.info("Writing dump metadata...")
            metadata = DumpMetadata(
                release=release,
                themes=[theme],
                download_date=datetime.now().isoformat(),
                size_gb=total_size_gb,
                is_complete=True,
                spatial_index_built=(theme == "divisions" and self.config.enable_spatial_index)
            )
            
            metadata_file = dump_path / "metadata.json"
            metadata_file.write_text(json.dumps(metadata.to_dict(), indent=2))
            
            # Validate if requested
            if validate:
                logger.info("Validating dump integrity...")
                if not self.validate_dump(release, theme):
                    raise RuntimeError(f"Dump validation failed for {theme}")
                logger.info("✓ Dump validation successful")
            
            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"Download complete!")
            logger.info(f"Theme: {theme}")
            logger.info(f"Size: {total_size_gb:.2f} GB")
            logger.info(f"Time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
            logger.info(f"Location: {dump_path}")
            logger.info("You can now process multiple countries rapidly using this dump!")
            logger.info("=" * 60)
            
            return dump_path
            
        finally:
            con.close()
    
    # NOTE: The previous "dump" approach of downloading complete themes (50-100GB)
    # has been replaced with a country-specific caching system that aligns with
    # Overture Maps best practices and avoids memory issues.
    # See: https://docs.overturemaps.org/getting-data/duckdb/
    
    def query_local_dump(self, query: DumpQuery, release: str = "latest", 
                        config_obj=None, force_download: bool = False):
        """
        Query local cache with efficient country-specific caching.
        
        This method uses a country-specific caching system instead of
        downloading complete themes, as I had performance issues with full dumps.
        
        Supports both single-theme queries (returns GeoDataFrame) and dual-theme
        queries for education/health/markets (returns dict with places + buildings).
        
        Args:
            query: Query configuration
            release: Release version or "latest"
            config_obj: Configuration object (required for cache misses)
            force_download: Force refresh of cached data if True
            
        Returns:
            GeoDataFrame for single-theme queries, or dict for dual-theme queries
        """
        if not query.country:
            raise ValueError("Country must be specified for cache-based queries")
        
        # Resolve latest release
        if release == "latest":
            release = self._get_latest_release()
        
        # Single-theme query processing
        # Create cache query (cache stores complete data without limits)
        cache_query = CacheQuery(
            country=query.country,
            theme=query.theme,
            type_name=query.type,
            release=release,
            use_divisions=True,  # Default to divisions for accuracy
            limit=query.limit,  # Apply limit only on retrieval, not caching
            filters=query.filters
        )
        
        # Check for force download first
        if force_download:
            if config_obj is None:
                raise ValueError("config_obj is required for forced data extraction")
            logger.info(f"Force download requested for {query.country}/{query.theme}/{query.type}")
            gdf = self._cache_manager.cache_country_data(cache_query, config_obj, overwrite=True)
            return gdf
        
        # Try to get from cache first
        cached_data = self._cache_manager.get_cached_data(cache_query)
        if cached_data is not None:
            logger.info(f"Using cached data for {query.country}/{query.theme}/{query.type}")
            return cached_data
        
        # Cache miss - need to extract and cache
        if config_obj is None:
            raise ValueError("config_obj is required for cache misses (data extraction)")
        
        logger.info(f"Cache miss for {query.country}/{query.theme}/{query.type} - extracting data")
        
        # Extract and cache data using proven DuckDB streaming approach
        gdf = self._cache_manager.cache_country_data(cache_query, config_obj)
        
        return gdf
    
    def _apply_target_filter(self, gdf: gpd.GeoDataFrame, target_filter: str) -> gpd.GeoDataFrame:
        """
        Apply target-specific filter to GeoDataFrame using same logic as duck.py
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
                        def has_primary_category_in(categories_json, target_cats):
                            if pd.isna(categories_json) or categories_json is None:
                                return False
                            if isinstance(categories_json, dict) and 'primary' in categories_json:
                                return categories_json['primary'] in target_cats
                            return False
                        
                        gdf = gdf[gdf['categories'].apply(lambda x: has_primary_category_in(x, allowed_categories))]
            elif 'class IN' in target_filter:
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
                    
        except Exception as e:
            logger.warning(f"Could not apply target filter '{target_filter}': {e}")
        
        return gdf
    
    def query_local_dump_pyarrow(self, query: DumpQuery, release: str = "latest") -> gpd.GeoDataFrame:
        """
        PyArrow-based query for improved performance on local files.
        Falls back to DuckDB method if PyArrow fails.
        """
        if not self.config.use_pyarrow_queries:
            return self.query_local_dump(query, release)
        
        try:
            return self._query_with_pyarrow(query, release)
        except Exception as e:
            logger.warning(f"PyArrow query failed: {e}")
            logger.info("Falling back to DuckDB query method...")
            return self.query_local_dump(query, release)
    
    def _query_with_pyarrow(self, query: DumpQuery, release: str) -> gpd.GeoDataFrame:
        """Execute query using PyArrow for better local file performance."""
        # Resolve latest release
        if release == "latest":
            release = self._get_latest_release()
        
        # Verify dump exists
        if not self.check_dump_exists(release, query.theme):
            raise FileNotFoundError(f"Dump not found: {release}/{query.theme}")
        
        # Get parquet file path
        dump_path = self.base_path / release / f"theme={query.theme}" / f"type={query.type}"
        parquet_file = dump_path / "data.parquet"
        
        if not parquet_file.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
        
        logger.debug(f"Executing PyArrow query on: {parquet_file}")
        
        # Build PyArrow filters for better performance
        filters = self._build_pyarrow_filters(query, release)
        
        # Read with filters applied at parquet level (predicate pushdown)
        columns = self._get_essential_columns(query.type)
        
        try:
            # Use PyArrow's efficient parquet reading
            table = pq.read_table(
                parquet_file,
                columns=columns,
                filters=filters,
                use_threads=True
            )
            
            # Convert to pandas
            df = table.to_pandas()
            
            if df.empty:
                logger.warning("PyArrow query returned no results")
                return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
            
            # Apply limit after initial filtering
            if query.limit and len(df) > query.limit:
                df = df.head(query.limit)
                logger.debug(f"Limited results to {query.limit} records")
            
            # Convert geometry and create GeoDataFrame
            gdf = self._convert_to_geodataframe_pyarrow(df)
            
            # Apply spatial filtering if needed
            if query.country and not self._filters_include_spatial(filters):
                # Spatial filtering not applied at parquet level, do it now
                gdf = self._apply_spatial_filter(gdf, query.country, release)
            
            logger.info(f"PyArrow query returned {len(gdf)} features")
            return gdf
            
        except Exception as e:
            logger.error(f"PyArrow query execution failed: {e}")
            raise
    
    def _build_pyarrow_filters(self, query: DumpQuery, release: str) -> List:
        """Build PyArrow filters for efficient predicate pushdown."""
        filters = []
        
        # Add custom field filters
        if query.filters:
            for field, condition in query.filters.items():
                if isinstance(condition, str) and condition.startswith("="):
                    value = condition[1:].strip().strip("'\"")
                    filters.append((field, '==', value))
                elif isinstance(condition, str) and " IN " in condition.upper():
                    # Handle IN clauses
                    values_part = condition.upper().split(" IN ")[1].strip("()")
                    values = [v.strip().strip("'\"") for v in values_part.split(",")]
                    filters.append((field, 'in', values))
        
        # Add bbox filter if available (much faster than spatial intersection)
        if query.country and query.bbox:
            xmin, ymin, xmax, ymax = query.bbox
            # Note: PyArrow filters on bbox columns would need to be implemented
            # depending on Overture schema structure
        
        return filters
    
    def _get_essential_columns(self, type_name: str) -> List[str]:
        """Get essential columns for a data type to minimize I/O."""
        # Import from existing duck.py module
        try:
            from .duck import _get_columns_for_target_type
            columns = _get_columns_for_target_type(type_name)
            return columns + ['geometry', 'bbox']  # Always include geometry and bbox
        except Exception:
            # Fallback: return None to read all columns
            return None
    
    def _convert_to_geodataframe_pyarrow(self, df) -> gpd.GeoDataFrame:
        """Convert pandas DataFrame to GeoDataFrame with robust geometry handling."""
        def convert_geometry_robust(geom_data):
            """Convert geometry data handling multiple formats."""
            if geom_data is None:
                return None
            
            try:
                if isinstance(geom_data, (bytes, bytearray)):
                    import shapely.wkb as swkb
                    return swkb.loads(bytes(geom_data))
                elif hasattr(geom_data, 'geom_type'):
                    return geom_data  # Already a shapely geometry
                else:
                    logger.debug(f"Unexpected geometry type: {type(geom_data)}")
                    return None
            except Exception as e:
                logger.debug(f"Failed to convert geometry: {e}")
                return None
        
        # Convert geometries
        if 'geometry' in df.columns:
            geometry_series = df['geometry'].apply(convert_geometry_robust)
            df_clean = df.drop(columns=['geometry'])
            
            gdf = gpd.GeoDataFrame(
                df_clean,
                geometry=geometry_series,
                crs="EPSG:4326"
            )
        else:
            # No geometry column found
            logger.warning("No geometry column found in data")
            gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")
        
        return gdf
    
    def _filters_include_spatial(self, filters: List) -> bool:
        """Check if filters already include spatial filtering."""
        # This would be implemented based on the specific filter structure
        # For now, return False to always apply spatial filtering
        return False
    
    def _apply_spatial_filter(self, gdf: gpd.GeoDataFrame, country: str, release: str) -> gpd.GeoDataFrame:
        """Apply spatial filtering using cached boundaries."""
        try:
            # Use spatial hash if available
            if self.config.enable_spatial_hash:
                hash_index = self._load_spatial_hash(release, gdf.columns[0])  # Theme name approximation
                if hash_index and country in hash_index:
                    relevant_indices = hash_index[country]
                    gdf = gdf.iloc[relevant_indices]
                    return gdf
            
            # Fallback to boundary intersection
            country_iso2 = self._resolve_country_code(country)
            
            # Try to get cached boundary
            boundary_geom = self._get_cached_boundary(country_iso2, release)
            if boundary_geom is not None:
                mask = gdf.geometry.intersects(boundary_geom)
                gdf = gdf[mask]
            
            return gdf
            
        except Exception as e:
            logger.warning(f"Spatial filtering failed: {e}")
            return gdf  # Return unfiltered data
    
    def _get_cached_boundary(self, country_iso2: str, release: str) -> Optional:
        """Get cached country boundary geometry using optimized boundary manager."""
        try:
            # Use boundary manager if available and enabled
            if self._boundary_manager and self.config.use_world_bank_boundaries:
                from .boundaries import BoundarySource
                
                # Try World Bank first, fallback to Natural Earth
                boundary = self._boundary_manager.get_boundary(
                    country_iso2, 
                    source=BoundarySource.WORLD_BANK,
                    simplify_tolerance=self.config.boundary_simplify_tolerance
                )
                
                if boundary:
                    logger.debug(f"Using optimized boundary for {country_iso2}")
                    return boundary
                
                # Fallback to Natural Earth
                boundary = self._boundary_manager.get_boundary(
                    country_iso2,
                    source=BoundarySource.NATURAL_EARTH,
                    simplify_tolerance=self.config.boundary_simplify_tolerance
                )
                
                if boundary:
                    logger.debug(f"Using Natural Earth boundary for {country_iso2}")
                    return boundary
            
            # Fallback to Overture divisions (original method)
            logger.debug(f"Using Overture divisions boundary for {country_iso2}")
            divisions_path = self.base_path / release / "theme=divisions" / "type=division_area" / "data.parquet"
            if not divisions_path.exists():
                return None
            
            # Use PyArrow to efficiently read just the needed country
            table = pq.read_table(
                divisions_path,
                filters=[('country', '==', country_iso2)],
                columns=['geometry']
            )
            
            if len(table) == 0:
                return None
            
            # Convert first geometry
            df = table.to_pandas()
            gdf = self._convert_to_geodataframe_pyarrow(df)
            
            if len(gdf) > 0:
                geometry = gdf.geometry.iloc[0]
                
                # Apply simplification if configured
                if self.config.boundary_simplify_tolerance > 0:
                    geometry = geometry.simplify(
                        self.config.boundary_simplify_tolerance, 
                        preserve_topology=True
                    )
                
                return geometry
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not load cached boundary for {country_iso2}: {e}")
            return None
    
    def _build_spatial_hash(self, release: str, theme: str):
        """Build spatial hash index for fast country lookups."""
        try:
            hash_file = self.base_path / release / f"spatial_hash_{theme}.json"
            
            if hash_file.exists():
                logger.debug(f"Spatial hash already exists: {hash_file}")
                return
            
            logger.info(f"Building spatial hash index for {theme}...")
            
            # Load theme data efficiently
            theme_path = self.base_path / release / f"theme={theme}"
            hash_data = {}
            
            # For each type in theme
            for type_dir in theme_path.iterdir():
                if type_dir.is_dir() and type_dir.name.startswith("type="):
                    parquet_file = type_dir / "data.parquet"
                    if parquet_file.exists():
                        type_name = type_dir.name.replace("type=", "")
                        hash_data[type_name] = self._compute_spatial_hash_for_type(
                            parquet_file, release
                        )
            
            # Save hash index
            with open(hash_file, 'w') as f:
                json.dump(hash_data, f, indent=2)
            
            logger.info(f"✓ Spatial hash index built: {hash_file}")
            
        except Exception as e:
            logger.warning(f"Failed to build spatial hash index: {e}")
    
    def _compute_spatial_hash_for_type(self, parquet_file: Path, release: str) -> Dict:
        """Compute spatial hash mapping countries to feature indices."""
        try:
            # Load divisions for country boundaries
            divisions_path = self.base_path / release / "theme=divisions" / "type=division_area" / "data.parquet"
            if not divisions_path.exists():
                return {}
            
            # Read country boundaries
            divisions_table = pq.read_table(divisions_path, columns=['country', 'geometry'])
            divisions_df = divisions_table.to_pandas()
            divisions_gdf = self._convert_to_geodataframe_pyarrow(divisions_df)
            
            # Build spatial index for divisions
            divisions_sindex = divisions_gdf.sindex
            
            # Read theme data in chunks to manage memory
            hash_mapping = {}
            chunk_size = 100000
            
            parquet_file_obj = pq.ParquetFile(parquet_file)
            
            for batch_idx, batch in enumerate(parquet_file_obj.iter_batches(batch_size=chunk_size)):
                batch_df = batch.to_pandas()
                
                if 'geometry' not in batch_df.columns:
                    continue
                
                batch_gdf = self._convert_to_geodataframe_pyarrow(batch_df)
                batch_sindex = batch_gdf.sindex
                
                # For each country, find intersecting features
                for country_idx, country_row in divisions_gdf.iterrows():
                    country_code = country_row['country']
                    country_geom = country_row['geometry']
                    
                    if country_code not in hash_mapping:
                        hash_mapping[country_code] = []
                    
                    # Use spatial index for fast intersection
                    possible_matches_idx = list(batch_sindex.intersection(country_geom.bounds))
                    
                    if possible_matches_idx:
                        possible_matches = batch_gdf.iloc[possible_matches_idx]
                        precise_matches = possible_matches[possible_matches.geometry.intersects(country_geom)]
                        
                        # Add global indices (accounting for batch offset)
                        global_indices = [batch_idx * chunk_size + idx for idx in precise_matches.index]
                        hash_mapping[country_code].extend(global_indices)
            
            return hash_mapping
            
        except Exception as e:
            logger.warning(f"Failed to compute spatial hash: {e}")
            return {}
    
    def _load_spatial_hash(self, release: str, theme: str) -> Optional[Dict]:
        """Load spatial hash index for fast lookups."""
        try:
            hash_file = self.base_path / release / f"spatial_hash_{theme}.json"
            if hash_file.exists():
                with open(hash_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.debug(f"Could not load spatial hash: {e}")
            return None
    
    def get_connection(self, release: str) -> duckdb.DuckDBPyConnection:
        """Get optimized DuckDB connection for local queries."""
        if self._connection is None:
            self._connection = self._setup_query_connection(release)
        return self._connection
    
    def get_available_dumps(self) -> List[DumpMetadata]:
        """List all available local dumps with metadata."""
        dumps = []
        
        if not self.base_path.exists():
            return dumps
        
        for release_dir in self.base_path.iterdir():
            if not release_dir.is_dir() or release_dir.name == "latest":
                continue
            
            for theme_dir in release_dir.iterdir():
                if not theme_dir.is_dir() or not theme_dir.name.startswith("theme="):
                    continue
                
                metadata_file = theme_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        data = json.loads(metadata_file.read_text())
                        dumps.append(DumpMetadata.from_dict(data))
                    except Exception as e:
                        logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
        
        return sorted(dumps, key=lambda d: d.download_date, reverse=True)
    
    def validate_dump(self, release: str, theme: str) -> bool:
        """
        Validate dump integrity and completeness.
        
        Args:
            release: Release version
            theme: Theme to validate
            
        Returns:
            True if dump is valid
        """
        dump_path = self.base_path / release / f"theme={theme}"
        
        # Check structure
        if not dump_path.exists():
            logger.error(f"Dump path does not exist: {dump_path}")
            return False
        
        # Check metadata
        metadata_file = dump_path / "metadata.json"
        if not metadata_file.exists():
            logger.error("Metadata file missing")
            return False
        
        try:
            metadata = json.loads(metadata_file.read_text())
        except Exception as e:
            logger.error(f"Invalid metadata file: {e}")
            return False
        
        # Check type directories exist
        type_dirs = [d for d in dump_path.iterdir() if d.is_dir() and d.name.startswith("type=")]
        if not type_dirs:
            logger.error("No type directories found")
            return False
        
        # Validate with DuckDB
        con = self.get_connection(release)
        
        try:
            for type_dir in type_dirs:
                parquet_file = type_dir / "data.parquet"
                if not parquet_file.exists():
                    logger.error(f"Missing parquet file: {parquet_file}")
                    return False
                
                # Test query
                test_sql = f"SELECT COUNT(*) as count FROM read_parquet('{parquet_file}')"
                result = con.execute(test_sql).fetchone()
                
                if result[0] == 0:
                    logger.warning(f"Empty parquet file: {parquet_file}")
                else:
                    logger.debug(f"Validated {type_dir.name}: {result[0]} rows")
            
            logger.info(f"Dump validation successful for {theme}")
            return True
            
        except Exception as e:
            logger.error(f"Validation query failed: {e}")
            return False
    
    def cleanup_old_dumps(self, keep_latest: int = 2) -> None:
        """Remove old dumps to manage disk space."""
        if not self.base_path.exists():
            return
        
        # Get all release directories
        release_dirs = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name != "latest":
                try:
                    # Parse release version for sorting
                    release_dirs.append((item.name, item))
                except:
                    continue
        
        # Sort by release version (newest first)
        release_dirs.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old releases
        for i, (release_name, release_path) in enumerate(release_dirs):
            if i >= keep_latest:
                logger.info(f"Removing old dump: {release_name}")
                import shutil
                shutil.rmtree(release_path)
    
    def _setup_download_connection(self) -> duckdb.DuckDBPyConnection:
        """Setup DuckDB connection for downloads."""
        con = duckdb.connect()
        
        # Install extensions
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL spatial; LOAD spatial;")
        
        # Memory optimization
        memory_limit = f"{self.config.max_memory_gb}GB"
        con.execute(f"SET memory_limit='{memory_limit}';")
        con.execute(f"SET threads={min(os.cpu_count() or 8, 16)};")
        
        # S3 optimizations
        con.execute("SET s3_region='us-west-2';")
        con.execute("SET http_timeout=1800000;")  # 30 minutes
        con.execute("SET http_retries=3;")
        con.execute("SET preserve_insertion_order=false;")
        
        logger.debug(f"Download connection configured: {memory_limit}, threads={min(os.cpu_count() or 8, 16)}")
        return con
    
    def _setup_query_connection(self, release: str) -> duckdb.DuckDBPyConnection:
        """Setup DuckDB connection for local queries."""
        con = duckdb.connect()
        
        # Install extensions
        con.execute("INSTALL spatial; LOAD spatial;")
        
        # Configure for local queries
        memory_limit = f"{self.config.max_memory_gb}GB"
        con.execute(f"SET memory_limit='{memory_limit}';")
        con.execute(f"SET threads={min(os.cpu_count() or 8, 16)};")
        
        # Local parquet optimizations
        con.execute("SET preserve_insertion_order=false;")
        con.execute("SET enable_progress_bar=true;")
        # Note: max_memory is automatically managed by memory_limit setting above
        
        # Spatial query optimizations for local files (using valid DuckDB settings)
        
        # Setup spatial index view if divisions available
        divisions_path = self.base_path / release / "theme=divisions" / "type=division_area" / "data.parquet"
        if divisions_path.exists():
            con.execute(f"""
            CREATE VIEW country_boundaries AS
            SELECT 
                country,
                subtype,
                ST_ENVELOPE(geometry) as bbox_geom,
                geometry,
                names
            FROM read_parquet('{divisions_path}')
            WHERE subtype = 'country'
            """)
            logger.debug("Spatial index view created for local queries")
        
        return con
    
    def _get_theme_types(self, con: duckdb.DuckDBPyConnection, release: str, theme: str) -> List[str]:
        """Get available types for a theme using known Overture schema."""
        # Use known Overture schema instead of querying S3 structure
        # This is more reliable and avoids DuckDB version compatibility issues
        type_mapping = {
            "transportation": ["segment"],
            "buildings": ["building"],
            "places": ["place"],
            "divisions": ["division_area"],
            "addresses": ["address"],
            "base": ["infrastructure", "land", "land_use", "water"]
        }
        
        types = type_mapping.get(theme, ["unknown"])
        
        if theme in type_mapping:
            logger.info(f"Using known types for {theme}: {types}")
        else:
            logger.warning(f"Unknown theme {theme}, using fallback: {types}")
        
        return types
    
    def _ensure_divisions_available(self, con: duckdb.DuckDBPyConnection, release: str) -> None:
        """Ensure divisions theme is available for spatial operations."""
        if not self.check_dump_exists(release, "divisions"):
            logger.info("Divisions not found, downloading for spatial operations...")
            self.download_theme(release, "divisions", validate=False)
    
    def _build_spatial_index(self, con: duckdb.DuckDBPyConnection, release: str) -> None:
        """Build spatial index for country boundaries."""
        divisions_path = self.base_path / release / "theme=divisions" / "type=division_area" / "data.parquet"
        if not divisions_path.exists():
            logger.warning("Cannot build spatial index: divisions data not found")
            return
        
        # Create spatial index table
        index_sql = f"""
        CREATE TABLE IF NOT EXISTS spatial_index AS
        SELECT 
            country,
            ST_XMin(ST_ENVELOPE(geometry)) as xmin,
            ST_YMin(ST_ENVELOPE(geometry)) as ymin,
            ST_XMax(ST_ENVELOPE(geometry)) as xmax,
            ST_YMax(ST_ENVELOPE(geometry)) as ymax
        FROM read_parquet('{divisions_path}')
        WHERE subtype = 'country'
        """
        
        try:
            con.execute(index_sql)
            logger.info("Spatial index built successfully")
        except Exception as e:
            logger.warning(f"Failed to build spatial index: {e}")

    def _build_bbox_index(self, con: duckdb.DuckDBPyConnection, release: str, theme: str) -> None:
        """Build bbox-based index for faster spatial filtering on data themes."""
        theme_path = self.base_path / release / f"theme={theme}"
        
        for type_dir in theme_path.iterdir():
            if type_dir.is_dir() and type_dir.name.startswith("type="):
                parquet_file = type_dir / "data.parquet"
                if parquet_file.exists():
                    try:
                        # Create bbox statistics for faster queries
                        index_sql = f"""
                        CREATE TABLE IF NOT EXISTS {theme}_bbox_stats AS
                        SELECT 
                            COUNT(*) as total_records,
                            MIN(bbox.xmin) as min_x,
                            MAX(bbox.xmax) as max_x,
                            MIN(bbox.ymin) as min_y,
                            MAX(bbox.ymax) as max_y
                        FROM read_parquet('{parquet_file}')
                        """
                        con.execute(index_sql)
                        logger.debug(f"Built bbox index for {theme}")
                    except Exception as e:
                        logger.warning(f"Failed to build bbox index for {theme}: {e}")
    
    def _get_latest_release(self) -> str:
        """Get the latest available release."""
        latest_link = self.base_path / "latest"
        if latest_link.exists() and latest_link.is_symlink():
            return latest_link.readlink().name
        
        # Fallback: find newest release directory
        if not self.base_path.exists():
            return "2025-07-23.0"  # Default
        
        releases = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name != "latest":
                releases.append(item.name)
        
        if releases:
            return max(releases)  # Lexicographic sort should work for YYYY-MM-DD.N format
        
        return "2025-07-23.0"  # Default
    
    def _resolve_country_code(self, country_input: str) -> str:
        """Resolve country name/code to ISO2 code."""
        country_data = CountryRegistry.get_country(country_input)
        if not country_data:
            raise ValueError(f"Unknown country: {country_input}")
        return country_data.iso2
    
    def cache_country_data(self, country: str, theme: str, type_name: str,
                          config_obj, release: str = "latest", 
                          overwrite: bool = False) -> gpd.GeoDataFrame:
        """
        Cache data for a specific country/theme/type combination.
        
        Args:
            country: Country code or name
            theme: Overture theme
            type_name: Overture type
            config_obj: Configuration object
            release: Release version
            overwrite: Whether to overwrite existing cache
            
        Returns:
            GeoDataFrame with cached data
        """
        cache_query = CacheQuery(
            country=country,
            theme=theme,
            type_name=type_name,
            release=release,
            use_divisions=True
        )
        
        return self._cache_manager.cache_country_data(cache_query, config_obj, overwrite)
    
    def list_cached_data(self, release: str = "latest") -> List:
        """
        List all cached data entries.
        
        Args:
            release: Release version
            
        Returns:
            List of cache metadata entries
        """
        return self._cache_manager.list_cached_countries(release)
    
    def clear_cache(self, country: Optional[str] = None, 
                   release: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            country: Specific country to clear (optional)
            release: Specific release to clear (optional)
        """
        self._cache_manager.clear_cache(country, release)
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self._cache_manager.get_cache_stats()
    
    def query_dual_theme(self, query: DumpQuery, building_filter: str, release: str = "latest", 
                         config_obj=None, force_download: bool = False, target_name: str = None) -> Dict[str, gpd.GeoDataFrame]:
        """
        Handle dual-theme queries that require both places and buildings data.
        
        Args:
            query: Original query configuration (places theme)
            building_filter: Filter to apply to buildings data
            release: Release version or "latest"
            config_obj: Configuration object (required for cache misses)
            force_download: Force refresh of cached data if True
            
        Returns:
            Dictionary with 'places' and 'buildings' GeoDataFrames
        """
        # Resolve latest release
        if release == "latest":
            release = self._get_latest_release()
        
        logger.debug(f"Processing dual-theme query internally for {query.country}")
        
        result = {}
        
        # Query 1: Places data (original query)
        places_query = CacheQuery(
            country=query.country,
            theme=query.theme,  # 'places'
            type_name=query.type,  # 'place' 
            release=release,
            use_divisions=True,
            limit=None,  # Don't limit cache retrieval - apply limit after filtering
            filters=query.filters
        )
        
        logger.debug(f"Querying places data for {query.country}")
        if force_download:
            places_gdf = self._cache_manager.cache_country_data(places_query, config_obj, overwrite=True)
        else:
            places_gdf = self._cache_manager.get_cached_data(places_query)
            if places_gdf is None:
                logger.debug(f"Places cache miss - extracting data")
                places_gdf = self._cache_manager.cache_country_data(places_query, config_obj)
        
        # Apply target-specific filter to places data if needed
        if not places_gdf.empty and config_obj:
            targets_config = None
            if 'yaml' in config_obj and 'targets' in config_obj['yaml']:
                targets_config = config_obj['yaml']['targets']
            elif 'targets' in config_obj:
                targets_config = config_obj['targets']
            
            if targets_config:
                # Use provided target_name or fall back to first target
                query_target_name = target_name if target_name and target_name in targets_config else list(targets_config.keys())[0]
                target_config = targets_config[query_target_name]
                places_filter = target_config.get('filter')
                logger.info(f"Places filter for {query_target_name}: '{places_filter}'")
                if places_filter:
                    original_count = len(places_gdf)
                    logger.debug(f"Places GDF columns: {list(places_gdf.columns)}")
                    if 'category_primary' in places_gdf.columns:
                        unique_cats = places_gdf['category_primary'].unique()
                        edu_count = len(places_gdf[places_gdf['category_primary'] == 'education'])
                        logger.info(f"Found {edu_count} places with category_primary='education' out of {len(unique_cats)} unique categories")
                    places_gdf = self._apply_target_filter(places_gdf, places_filter)
                    if len(places_gdf) != original_count:
                        logger.info(f"Applied places filter: {original_count:,} -> {len(places_gdf):,} features")
                    else:
                        logger.warning(f"Filter '{places_filter}' did not reduce feature count")
        
            else:
                logger.debug(f"No targets config found in config_obj")
        
        # Apply limit to places data after filtering
        if not places_gdf.empty and query.limit and len(places_gdf) > query.limit:
            places_gdf = places_gdf.head(query.limit)
            logger.info(f"Limited places results to {query.limit} features")
        
        if not places_gdf.empty:
            result['places'] = places_gdf
        
        # Query 2: Buildings data - cache complete data, apply filter during retrieval
        # Create separate queries for caching and filtering
        buildings_cache_query = CacheQuery(
            country=query.country,
            theme='buildings',
            type_name='building',
            release=release,
            use_divisions=True,
            limit=None,  # No limit for caching
            filters=None  # No filter for caching - cache ALL buildings
        )
        
        buildings_filter_query = CacheQuery(
            country=query.country,
            theme='buildings',
            type_name='building',
            release=release,
            use_divisions=True,
            limit=query.limit,
            filters=building_filter if building_filter else None  # Apply filter during retrieval
        )
        
        logger.debug(f"Querying buildings data for {query.country}")
        if force_download:
            # Force recache complete buildings data (no filter)
            buildings_gdf = self._cache_manager.cache_country_data(buildings_cache_query, config_obj, overwrite=True)
            # Apply filter to the complete cached data
            if building_filter:
                original_count = len(buildings_gdf)
                from .cache_manager import apply_sql_filter
                buildings_gdf = apply_sql_filter(buildings_gdf, building_filter)
                logger.debug(f"Applied building filter: {original_count:,} -> {len(buildings_gdf):,} features")
            # Apply limit if specified
            if query.limit:
                buildings_gdf = buildings_gdf.head(query.limit)
        else:
            # Try to get filtered data from cache
            buildings_gdf = self._cache_manager.get_cached_data(buildings_filter_query)
            if buildings_gdf is None:
                logger.debug(f"Buildings cache miss - extracting complete data")
                # Cache complete buildings data (no filter)
                self._cache_manager.cache_country_data(buildings_cache_query, config_obj)
                # Now get the filtered data
                buildings_gdf = self._cache_manager.get_cached_data(buildings_filter_query)
        
        if not buildings_gdf.empty:
            result['buildings'] = buildings_gdf  
        
        total_features = sum(len(gdf) for gdf in result.values())
        logger.info(f"Dual-theme query complete: {total_features:,} features")
        
        return result
    
    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None