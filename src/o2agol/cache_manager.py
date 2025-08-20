"""
Country-specific caching system for Overture Maps data.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import duckdb
import geopandas as gpd

from .duck import setup_duckdb_optimized, _build_divisions_query, _build_bbox_query, _get_columns_for_target_type
from .config.countries import CountryRegistry

logger = logging.getLogger(__name__)


def apply_sql_filter(gdf: gpd.GeoDataFrame, sql_filter: str) -> gpd.GeoDataFrame:
    """
    Apply SQL-style filter string to a GeoDataFrame.
    
    Supports basic SQL conditions like:
    - subtype = 'medical'
    - categories.primary = 'health_and_medical'
    - subtype IN ('service', 'commercial')
    - categories.primary IN ('retail', 'shopping', 'food_and_drink')
    
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
                
                if column in gdf.columns:
                    return gdf[gdf[column] == value]
                else:
                    logger.warning(f"Column '{column}' not found in data, returning empty result")
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
                    
                    if column in gdf.columns:
                        return gdf[gdf[column].isin(values)]
                    else:
                        logger.warning(f"Column '{column}' not found in data, returning empty result")
                        return gdf.iloc[0:0]  # Return empty GeoDataFrame with same structure
        
        logger.warning(f"Unsupported filter format: {sql_filter}, returning unfiltered data")
        return gdf
        
    except Exception as e:
        logger.error(f"Failed to apply SQL filter '{sql_filter}': {e}, returning unfiltered data")
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
    bbox: Tuple[float, float, float, float]
    
    def to_dict(self) -> Dict:
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
    def from_dict(cls, data: Dict) -> 'CacheMetadata':
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
    filters: Optional[Union[Dict, str]] = None


class CountryCacheManager:
    """
    Manages country-specific caches for Overture Maps data.
    
    This approach caches data at the country level rather than downloading
    complete themes, providing the same performance benefits while avoiding
    memory issues and aligning with Overture Maps best practices.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize cache manager.
        
        Args:
            base_path: Base directory for caches (default: ./cache)
        """
        self.base_path = Path(base_path or "./cache")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Country registry for ISO code resolution
        self.country_registry = CountryRegistry()
        
        logger.info(f"CountryCacheManager initialized: {self.base_path}")
    
    def get_cached_data(self, query: CacheQuery) -> Optional[gpd.GeoDataFrame]:
        """
        Get cached data for a country/theme/type combination.
        
        Args:
            query: Cache query configuration
            
        Returns:
            Cached GeoDataFrame if available, None otherwise
        """
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
                    logger.debug(f"Applied filter '{query.filters}': {original_count:,} -> {len(gdf):,} features")
            
            # Apply limit if specified
            if query.limit:
                gdf = gdf.head(query.limit)
            
            logger.debug(f"Cache hit: {cache_file.name} ({len(gdf):,} features)")
            return gdf
            
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            return None
    
    def cache_country_data(self, query: CacheQuery, config_obj, 
                          overwrite: bool = False) -> gpd.GeoDataFrame:
        """
        Cache data for a specific country using DuckDB streaming extraction.
        
        Note: This method always caches complete country data without limits.
        Limits are applied only when retrieving from cache via get_cached_data().
        
        Args:
            query: Cache query configuration
            config_obj: Configuration object with Overture settings
            overwrite: Whether to overwrite existing cache
            
        Returns:
            GeoDataFrame with extracted and cached data
        """
        cache_file = self._get_cache_file_path(query)
        
        # Check if cache exists and is valid
        if not overwrite and cache_file.exists():
            logger.info(f"Cache exists for {query.country}/{query.theme}/{query.type_name}")
            cached_data = self.get_cached_data(query)
            if cached_data is not None:
                return cached_data
        
        logger.info(f"Extracting and caching data for {query.country}/{query.theme}/{query.type_name}")
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
            filters=query.filters
        )
        gdf = self._extract_country_data(extraction_query, config_obj)
        
        if gdf.empty:
            logger.warning(f"No data found for {query.country}/{query.theme}/{query.type_name}")
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
        logger.debug(f"Cached {len(gdf):,} features in {elapsed:.1f}s: {cache_file.name}")
        
        # Apply limit to the result if specified in original query
        if query.limit:
            gdf = gdf.head(query.limit)
            logger.debug(f"Applied limit: returning {len(gdf):,} features")
        
        return gdf
    
    def _extract_country_data(self, query: CacheQuery, config_obj) -> gpd.GeoDataFrame:
        """
        Extract country data using DuckDB streaming (reuses existing patterns).
        """
        # Use the proven DuckDB extraction approach from duck.py
        # Extract secure config from the full config object
        secure_config = config_obj.get('secure') if isinstance(config_obj, dict) else config_obj.secure_config
        con = setup_duckdb_optimized(secure_config)
        
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
            
            # Extract overture config from the full config object
            overture_config = config_obj.get('overture') if isinstance(config_obj, dict) else config_obj.overture_config
            
            # Use the same query building logic as the working pipeline
            # Note: Never pass limit to SQL queries during caching - always extract complete data
            if query.use_divisions:
                sql = _build_divisions_query(
                    overture_config, 
                    f"{query.theme}_{query.type_name}",  # target_name
                    target_config, 
                    selector, 
                    None  # No limit for caching - extract all country data
                )
            else:
                # Use bbox fallback
                country_bboxes = CountryRegistry.get_bounding_boxes()
                if country_iso2 not in country_bboxes:
                    raise ValueError(f"No bounding box defined for country {country_iso2}")
                
                xmin, ymin, xmax, ymax = country_bboxes[country_iso2]
                parquet_url = f"{overture_config['base_url']}/theme={query.theme}/type={query.type_name}/*.parquet"
                sql = _build_bbox_query(parquet_url, xmin, ymin, xmax, ymax, target_config, None)  # No limit for caching
            
            # Execute query using the proven two-step approach from duck.py
            logger.debug(f"Executing cache extraction query: {sql[:200]}...")
            
            # Use the same reliable geometry handling as the working pipeline
            from .duck import _fetch_via_temp_file
            gdf = _fetch_via_temp_file(con, sql)
            
            return gdf
            
        finally:
            con.close()
    
    def _get_cache_file_path(self, query: CacheQuery) -> Path:
        """Generate cache file path for a query."""
        country_data = CountryRegistry.get_country(query.country)
        if not country_data:
            raise ValueError(f"Unknown country: {query.country}")
        country_iso2 = country_data.iso2
        return (self.base_path / query.release / country_iso2 / 
                f"{query.theme}_{query.type_name}.parquet")
    
    def list_cached_countries(self, release: str = "latest") -> List[CacheMetadata]:
        """
        List all cached countries for a release.
        
        Args:
            release: Release version or "latest"
            
        Returns:
            List of cache metadata entries
        """
        if release == "latest":
            release = self._get_latest_release()
        
        release_dir = self.base_path / release
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
                    logger.warning(f"Failed to read metadata {metadata_file}: {e}")
        
        return cached_entries
    
    def clear_cache(self, country: Optional[str] = None, 
                   release: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            country: Specific country to clear (optional)
            release: Specific release to clear (optional)
        """
        if release:
            release_dir = self.base_path / release
            if country:
                country_data = CountryRegistry.get_country(country)
                if not country_data:
                    raise ValueError(f"Unknown country: {country}")
                country_iso2 = country_data.iso2
                country_dir = release_dir / country_iso2
                if country_dir.exists():
                    import shutil
                    shutil.rmtree(country_dir)
                    logger.info(f"Cleared cache for {country} ({country_iso2}) in {release}")
            else:
                if release_dir.exists():
                    import shutil
                    shutil.rmtree(release_dir)
                    logger.info(f"Cleared all cache for release {release}")
        else:
            if self.base_path.exists():
                import shutil
                shutil.rmtree(self.base_path)
                self.base_path.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared entire cache")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
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
            'cache_path': str(self.base_path)
        }
    
    def _get_latest_release(self) -> str:
        """Get the latest release version from cache directories."""
        releases = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name != "latest":
                releases.append(item.name)
        
        if not releases:
            return "2025-07-23.0"  # Default current release
        
        # Sort releases and return the latest
        releases.sort(reverse=True)
        return releases[0]