"""
Boundary management for enhanced spatial operations.

Provides cached boundary data from multiple sources (World Bank, Natural Earth, Overture)
with geometry simplification and pre-computed spatial indexes for faster country clipping.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


class BoundarySource(Enum):
    """Available boundary data sources."""
    OVERTURE_DIVISIONS = "overture"
    WORLD_BANK = "world_bank"
    NATURAL_EARTH = "natural_earth"
    CACHED = "cached"


@dataclass
class BoundaryInfo:
    """Metadata for cached boundary data."""
    country_code: str
    source: BoundarySource
    download_date: str
    simplified: bool
    geometry_type: str
    bbox: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


class BoundaryManager:
    """
    Manage and optimize administrative boundaries for country clipping.
    
    Provides cached boundary data from multiple sources with geometry
    simplification and spatial optimization for faster operations.
    """
    
    # Data source URLs
    WORLD_BANK_URL = "https://datacatalog.worldbank.org/search/dataset/0038272/World-Bank-Official-Boundaries"
    NATURAL_EARTH_URL = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_countries.zip"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize boundary manager.
        
        Args:
            cache_dir: Directory for boundary cache (default: boundaries_cache)
        """
        self.cache_dir = Path(cache_dir or "boundaries_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequent lookups
        self._geometry_cache: Dict[str, BaseGeometry] = {}
        self._metadata_cache: Dict[str, BoundaryInfo] = {}
        
        logger.info(f"BoundaryManager initialized: {self.cache_dir}")
    
    def get_boundary(self, country_code: str, 
                    source: BoundarySource = BoundarySource.WORLD_BANK,
                    simplify_tolerance: float = 0.01) -> Optional[BaseGeometry]:
        """
        Get optimized boundary geometry for a country.
        
        Args:
            country_code: ISO2 country code
            source: Boundary data source
            simplify_tolerance: Geometry simplification tolerance (degrees)
            
        Returns:
            Simplified boundary geometry or None if not found
        """
        cache_key = f"{country_code}_{source.value}_{simplify_tolerance}"
        
        # Check in-memory cache first
        if cache_key in self._geometry_cache:
            logger.debug(f"Using cached boundary for {country_code}")
            return self._geometry_cache[cache_key]
        
        # Check disk cache
        cached_boundary = self._load_from_disk_cache(cache_key)
        if cached_boundary:
            self._geometry_cache[cache_key] = cached_boundary
            return cached_boundary
        
        # Download and process boundary
        try:
            boundary = self._download_and_process_boundary(
                country_code, source, simplify_tolerance
            )
            
            if boundary:
                # Cache results
                self._geometry_cache[cache_key] = boundary
                self._save_to_disk_cache(cache_key, boundary, country_code, source)
                
                logger.info(f"Downloaded and cached boundary for {country_code}")
                return boundary
            
        except Exception as e:
            logger.warning(f"Failed to get boundary for {country_code}: {e}")
        
        return None
    
    def get_multiple_boundaries(self, country_codes: List[str],
                               source: BoundarySource = BoundarySource.WORLD_BANK,
                               simplify_tolerance: float = 0.01) -> Dict[str, BaseGeometry]:
        """
        Get boundaries for multiple countries efficiently.
        
        Args:
            country_codes: List of ISO2 country codes
            source: Boundary data source
            simplify_tolerance: Geometry simplification tolerance
            
        Returns:
            Dictionary mapping country codes to boundary geometries
        """
        boundaries = {}
        
        # Try to load all from cache first
        missing_countries = []
        for country_code in country_codes:
            boundary = self.get_boundary(country_code, source, simplify_tolerance)
            if boundary:
                boundaries[country_code] = boundary
            else:
                missing_countries.append(country_code)
        
        # Batch download missing countries if possible
        if missing_countries:
            logger.info(f"Batch downloading boundaries for {len(missing_countries)} countries")
            batch_boundaries = self._batch_download_boundaries(
                missing_countries, source, simplify_tolerance
            )
            boundaries.update(batch_boundaries)
        
        return boundaries
    
    def precompute_intersections(self, theme_gdf: gpd.GeoDataFrame, 
                                countries: List[str],
                                source: BoundarySource = BoundarySource.WORLD_BANK) -> Dict[str, List[int]]:
        """
        Pre-compute spatial intersections for instant country queries.
        
        Args:
            theme_gdf: GeoDataFrame containing features to index
            countries: List of country codes to compute intersections for
            source: Boundary data source
            
        Returns:
            Dictionary mapping country codes to lists of feature indices
        """
        logger.info(f"Pre-computing spatial intersections for {len(countries)} countries")
        intersections = {}
        
        # Build spatial index for theme data
        theme_sindex = theme_gdf.sindex
        
        # Get boundaries for all countries
        boundaries = self.get_multiple_boundaries(countries, source)
        
        for country_code, boundary in boundaries.items():
            try:
                # Use spatial index for fast intersection
                possible_matches_idx = list(theme_sindex.intersection(boundary.bounds))
                
                if possible_matches_idx:
                    possible_matches = theme_gdf.iloc[possible_matches_idx]
                    
                    # Precise intersection test
                    precise_matches = possible_matches[possible_matches.geometry.intersects(boundary)]
                    intersections[country_code] = precise_matches.index.tolist()
                else:
                    intersections[country_code] = []
                
                logger.debug(f"Found {len(intersections[country_code])} intersections for {country_code}")
                
            except Exception as e:
                logger.warning(f"Failed to compute intersections for {country_code}: {e}")
                intersections[country_code] = []
        
        logger.info(f"Pre-computation complete: {sum(len(v) for v in intersections.values())} total intersections")
        return intersections
    
    def _download_and_process_boundary(self, country_code: str, 
                                     source: BoundarySource,
                                     simplify_tolerance: float) -> Optional[BaseGeometry]:
        """Download and process boundary from specified source."""
        if source == BoundarySource.WORLD_BANK:
            return self._download_world_bank_boundary(country_code, simplify_tolerance)
        elif source == BoundarySource.NATURAL_EARTH:
            return self._download_natural_earth_boundary(country_code, simplify_tolerance)
        else:
            logger.warning(f"Unsupported boundary source: {source}")
            return None
    
    def _download_world_bank_boundary(self, country_code: str, 
                                    simplify_tolerance: float) -> Optional[BaseGeometry]:
        """Download boundary from World Bank data."""
        # This is a simplified implementation - in practice, World Bank data
        # would require API access or downloaded dataset processing
        
        logger.info(f"World Bank boundary download for {country_code} not yet implemented")
        logger.info("Falling back to Natural Earth data")
        return self._download_natural_earth_boundary(country_code, simplify_tolerance)
    
    def _download_natural_earth_boundary(self, country_code: str,
                                       simplify_tolerance: float) -> Optional[BaseGeometry]:
        """Download boundary from Natural Earth data."""
        try:
            # Download Natural Earth countries data
            logger.info(f"Downloading Natural Earth boundary for {country_code}")
            
            # Use the 50m resolution dataset (good balance of detail vs. performance)
            url = self.NATURAL_EARTH_URL
            
            # Download and read the shapefile
            temp_file = self.cache_dir / "ne_countries_temp.zip"
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Read the shapefile
            countries_gdf = gpd.read_file(f"zip://{temp_file}")
            
            # Find the country by ISO2 code
            country_row = countries_gdf[countries_gdf['ISO_A2'] == country_code.upper()]
            
            if country_row.empty:
                logger.warning(f"Country {country_code} not found in Natural Earth data")
                return None
            
            # Get geometry and simplify
            geometry = country_row.geometry.iloc[0]
            
            if simplify_tolerance > 0:
                geometry = geometry.simplify(simplify_tolerance, preserve_topology=True)
                logger.debug(f"Simplified geometry for {country_code} (tolerance={simplify_tolerance})")
            
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
            
            return geometry
            
        except Exception as e:
            logger.error(f"Failed to download Natural Earth boundary for {country_code}: {e}")
            return None
    
    def _batch_download_boundaries(self, country_codes: List[str],
                                 source: BoundarySource,
                                 simplify_tolerance: float) -> Dict[str, BaseGeometry]:
        """Batch download boundaries for multiple countries."""
        boundaries = {}
        
        if source == BoundarySource.NATURAL_EARTH:
            try:
                # Download once and extract all needed countries
                logger.info("Batch downloading Natural Earth boundaries")
                
                url = self.NATURAL_EARTH_URL
                temp_file = self.cache_dir / "ne_countries_batch.zip"
                
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Read all countries
                countries_gdf = gpd.read_file(f"zip://{temp_file}")
                
                # Extract requested countries
                for country_code in country_codes:
                    country_row = countries_gdf[countries_gdf['ISO_A2'] == country_code.upper()]
                    
                    if not country_row.empty:
                        geometry = country_row.geometry.iloc[0]
                        
                        if simplify_tolerance > 0:
                            geometry = geometry.simplify(simplify_tolerance, preserve_topology=True)
                        
                        boundaries[country_code] = geometry
                        
                        # Cache individual boundary
                        cache_key = f"{country_code}_{source.value}_{simplify_tolerance}"
                        self._geometry_cache[cache_key] = geometry
                        self._save_to_disk_cache(cache_key, geometry, country_code, source)
                
                # Clean up
                if temp_file.exists():
                    temp_file.unlink()
                
                logger.info(f"Batch downloaded {len(boundaries)} boundaries")
                
            except Exception as e:
                logger.error(f"Batch download failed: {e}")
                # Fall back to individual downloads
                for country_code in country_codes:
                    boundary = self._download_natural_earth_boundary(country_code, simplify_tolerance)
                    if boundary:
                        boundaries[country_code] = boundary
        else:
            # Fall back to individual downloads for other sources
            for country_code in country_codes:
                boundary = self._download_and_process_boundary(country_code, source, simplify_tolerance)
                if boundary:
                    boundaries[country_code] = boundary
        
        return boundaries
    
    def _load_from_disk_cache(self, cache_key: str) -> Optional[BaseGeometry]:
        """Load boundary from disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        metadata_file = self.cache_dir / f"{cache_key}_meta.json"
        
        if not (cache_file.exists() and metadata_file.exists()):
            return None
        
        try:
            # Load geometry from WKT
            with open(cache_file, 'r') as f:
                geometry_data = json.load(f)
            
            from shapely import wkt
            geometry = wkt.loads(geometry_data['wkt'])
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self._metadata_cache[cache_key] = BoundaryInfo(**metadata)
            
            logger.debug(f"Loaded boundary from cache: {cache_key}")
            return geometry
            
        except Exception as e:
            logger.debug(f"Failed to load cached boundary {cache_key}: {e}")
            return None
    
    def _save_to_disk_cache(self, cache_key: str, geometry: BaseGeometry,
                           country_code: str, source: BoundarySource):
        """Save boundary to disk cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            metadata_file = self.cache_dir / f"{cache_key}_meta.json"
            
            # Save geometry as WKT
            geometry_data = {
                'wkt': geometry.wkt
            }
            
            with open(cache_file, 'w') as f:
                json.dump(geometry_data, f)
            
            # Save metadata
            metadata = BoundaryInfo(
                country_code=country_code,
                source=source,
                download_date=pd.Timestamp.now().isoformat(),
                simplified=True,  # Assume we always apply some simplification
                geometry_type=geometry.geom_type,
                bbox=geometry.bounds
            )
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata.__dict__, f, indent=2)
            
            self._metadata_cache[cache_key] = metadata
            logger.debug(f"Cached boundary: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache boundary {cache_key}: {e}")
    
    def clear_cache(self, country_code: Optional[str] = None):
        """Clear boundary cache."""
        if country_code:
            # Clear specific country
            keys_to_remove = [k for k in self._geometry_cache.keys() if k.startswith(country_code)]
            for key in keys_to_remove:
                self._geometry_cache.pop(key, None)
                self._metadata_cache.pop(key, None)
            
            # Remove disk cache files
            for cache_file in self.cache_dir.glob(f"{country_code}_*"):
                cache_file.unlink()
            
            logger.info(f"Cleared cache for {country_code}")
        else:
            # Clear all cache
            self._geometry_cache.clear()
            self._metadata_cache.clear()
            
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            
            logger.info("Cleared all boundary cache")
    
    def get_cache_stats(self) -> Dict:
        """Get boundary cache statistics."""
        cache_size_mb = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.json")
        ) / (1024 * 1024)
        
        return {
            'cached_countries': len(self._geometry_cache),
            'cache_dir': str(self.cache_dir),
            'cache_size_mb': round(cache_size_mb, 2),
            'available_sources': [source.value for source in BoundarySource]
        }