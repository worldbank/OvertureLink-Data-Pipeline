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
from typing import Dict, List, Optional, Tuple

import duckdb
import geopandas as gpd

from .config.countries import CountryRegistry

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
    max_memory_gb: int = 32
    chunk_size: int = 5  # Countries per chunk
    enable_spatial_index: bool = True
    compression: str = "zstd"
    partitioning: str = "hive"


class DumpManager:
    """
    Manages local Overture data dumps for improved performance.
    
    Handles downloading complete themes from S3, storing them locally as Parquet,
    and providing efficient spatial queries for multiple countries/queries.
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
    
    def download_theme(self, release: str, theme: str, validate: bool = True) -> Path:
        """
        Download complete theme from Overture S3 bucket.
        
        Args:
            release: Overture release version
            theme: Theme to download
            validate: Whether to validate after download
            
        Returns:
            Path to downloaded theme directory
        """
        if theme not in self.OVERTURE_THEMES:
            raise ValueError(f"Invalid theme: {theme}. Valid themes: {self.OVERTURE_THEMES}")
        
        dump_path = self.base_path / release / f"theme={theme}"
        dump_path.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if self.check_dump_exists(release, theme):
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
            
            # Build spatial index if enabled
            if self.config.enable_spatial_index and theme == "divisions":
                logger.info("Building spatial index for divisions...")
                self._build_spatial_index(con, release)
            
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
    
    def query_local_dump(self, query: DumpQuery, release: str = "latest") -> gpd.GeoDataFrame:
        """
        Query local dump with spatial filtering.
        
        Args:
            query: Query configuration
            release: Release version or "latest"
            
        Returns:
            GeoDataFrame with query results
        """
        # Resolve latest release
        if release == "latest":
            release = self._get_latest_release()
        
        # Verify dump exists
        if not self.check_dump_exists(release, query.theme):
            raise FileNotFoundError(f"Dump not found: {release}/{query.theme}")
        
        con = self.get_connection(release)
        
        # Build base query
        dump_path = self.base_path / release / f"theme={query.theme}" / f"type={query.type}"
        parquet_pattern = f"{dump_path}/data.parquet"
        
        sql_parts = [f"SELECT * FROM read_parquet('{parquet_pattern}')"]
        
        # Add spatial filtering
        if query.country:
            if query.bbox:
                # Use bbox filtering (fast)
                xmin, ymin, xmax, ymax = query.bbox
                sql_parts.append(f"""
                WHERE bbox.xmin >= {xmin} AND bbox.xmax <= {xmax}
                  AND bbox.ymin >= {ymin} AND bbox.ymax <= {ymax}
                """)
            else:
                # Use precise division filtering
                country_iso2 = self._resolve_country_code(query.country)
                sql_parts.append(f"""
                WHERE ST_Intersects(
                    geometry,
                    (SELECT geometry FROM country_boundaries WHERE country = '{country_iso2}')
                )
                """)
        
        # Add custom filters
        if query.filters:
            for field, condition in query.filters.items():
                sql_parts.append(f"AND {field} {condition}")
        
        # Add limit
        if query.limit:
            sql_parts.append(f"LIMIT {query.limit}")
        
        sql = " ".join(sql_parts)
        
        # Execute query
        logger.debug(f"Executing local dump query: {sql}")
        result_df = con.execute(sql).df()
        
        if result_df.empty:
            logger.warning("Query returned no results")
            return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            result_df,
            geometry=gpd.GeoSeries.from_wkb(result_df['geometry']),
            crs="EPSG:4326"
        )
        
        logger.info(f"Local dump query returned {len(gdf)} features")
        return gdf
    
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
        """Setup optimized DuckDB connection for downloads."""
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
        """Setup optimized DuckDB connection for local queries."""
        con = duckdb.connect()
        
        # Install extensions
        con.execute("INSTALL spatial; LOAD spatial;")
        
        # Configure for local queries
        memory_limit = f"{self.config.max_memory_gb}GB"
        con.execute(f"SET memory_limit='{memory_limit}';")
        con.execute(f"SET threads={min(os.cpu_count() or 8, 16)};")
        
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
            logger.debug("Spatial index view created")
        
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
        registry = CountryRegistry()
        country_data = registry.get_country_info(country_input)
        if not country_data:
            raise ValueError(f"Unknown country: {country_input}")
        return country_data['iso2']
    
    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None