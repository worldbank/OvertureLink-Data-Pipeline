"""
Exporter - Multi-format Data Export

Enhanced version of the existing export.py with cleaner interface and format detection.
Handles export to GeoJSON, GeoPackage, and File Geodatabase formats.
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import fiona
import geopandas as gpd

from ..domain.enums import ExportFormat

logger = logging.getLogger(__name__)


class Exporter:
    """
    Multi-format data exporter.
    
    Supports GeoJSON, GeoPackage, and File Geodatabase export with
    automatic format detection from file extensions and multi-layer support.
    """
    
    def __init__(self, out_path: Optional[Path] = None, fmt: Optional[ExportFormat] = None):
        """
        Initialize exporter with output path and format.
        
        Args:
            out_path: Output file path (format inferred from extension if not specified)
            fmt: Explicit format override
        """
        self.out_path = out_path
        self.fmt = fmt
        
        # Infer format from file extension if not explicitly provided
        if out_path and not fmt:
            suffix = out_path.suffix.lower()
            if suffix in ['.geojson', '.json']:
                self.fmt = ExportFormat.GEOJSON
            elif suffix == '.gpkg':
                self.fmt = ExportFormat.GPKG
            elif suffix in ['.gdb', '.fgdb']:
                self.fmt = ExportFormat.FGDB
            else:
                self.fmt = ExportFormat.GEOJSON  # default
                
    def write(
        self, 
        data: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame], 
        base_name: str, 
        out_dir: Path, 
        multilayer: bool = False, 
        raw: bool = False
    ) -> Path:
        """
        Write GeoDataFrame(s) to specified format.
        
        Args:
            data: GeoDataFrame or dict of GeoDataFrames (multi-layer)
            base_name: Base filename (without extension)
            out_dir: Output directory
            multilayer: Whether this is a multi-layer export
            raw: Whether this is raw Overture data (affects metadata)
            
        Returns:
            Path to the created file
            
        Supports:
            - GeoJSON: Single file with FeatureCollection
            - GPKG: Single file with layers
            - FGDB: Directory with feature classes
        """
        # Use provided path or generate from base_name and format
        if self.out_path:
            output_path = self.out_path
        else:
            # Generate filename with appropriate extension
            extensions = {
                ExportFormat.GEOJSON: "geojson",
                ExportFormat.GPKG: "gpkg", 
                ExportFormat.FGDB: "gdb"
            }
            extension = extensions.get(self.fmt, "geojson")
            output_path = out_dir / f"{base_name}.{extension}"
        
        # Call the main export function
        self.export_data(
            data=data,
            output_path=str(output_path),
            target_name=base_name.split('_')[-1] if '_' in base_name else base_name,  # Extract target from base_name
            export_format=self.fmt.value,
            raw_export=raw,
            include_metadata=True
        )
        
        return output_path
        
    def export_data(
        self,
        data: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame],
        output_path: str,
        target_name: str,
        export_format: str = "geojson",
        raw_export: bool = False,
        include_metadata: bool = True
    ) -> None:
        """
        Export geospatial data to specified format.
        
        Args:
            data: GeoDataFrame or dict of GeoDataFrames (multi-layer)
            output_path: Output file path
            target_name: Target data type for metadata
            export_format: Export format (geojson, gpkg, fgdb)
            raw_export: If True, indicates raw Overture data
            include_metadata: Whether to include metadata
        """
        format_enum = ExportFormat(export_format.lower())
        
        # Validate output path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Route to format-specific handler
        if format_enum == ExportFormat.GEOJSON:
            self._export_to_geojson(data, output_path, target_name, raw_export, include_metadata)
        elif format_enum == ExportFormat.GPKG:
            self._export_to_gpkg(data, output_path, target_name, raw_export, include_metadata)
        elif format_enum == ExportFormat.FGDB:
            self._export_to_fgdb(data, output_path, target_name, raw_export, include_metadata)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Successfully exported to {output_path} ({export_format})")

    def create_staging_file(
        self, 
        gdf: gpd.GeoDataFrame, 
        layer_name: str = "features",
        staging_format: str = "geojson"
    ) -> tempfile.NamedTemporaryFile:
        """
        Create temporary staging file for AGOL publishing workflow.
        
        Args:
            gdf: GeoDataFrame to stage
            layer_name: Layer name for GPKG staging
            staging_format: Format for staging ("geojson" or "gpkg")
            
        Returns:
            NamedTemporaryFile object with staging data
        """
        from ..utils import get_pid_temp_dir
        
        # Get PID-isolated temp directory
        temp_dir = get_pid_temp_dir()
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        if staging_format.lower() == "geojson":
            return self._gdf_to_geojson_tempfile(gdf, temp_dir)
        elif staging_format.lower() == "gpkg":
            return self._gdf_to_gpkg_tempfile(gdf, layer_name, temp_dir)
        else:
            raise ValueError(f"Unsupported staging format: {staging_format}")
    
    def _gdf_to_geojson_tempfile(self, gdf: gpd.GeoDataFrame, temp_dir: Path) -> tempfile.NamedTemporaryFile:
        """
        Convert GeoDataFrame to temporary GeoJSON file for ArcGIS Online upload.
        
        Uses GeoPandas built-in GeoJSON export for consistent handling of all data types.
        Creates temp files in project /temp directory to avoid system temp clutter.
        
        Args:
            gdf: GeoDataFrame to convert
            temp_dir: Directory for temporary files
            
        Returns:
            NamedTemporaryFile containing GeoJSON data
            
        Note:
            Caller is responsible for cleanup of temporary file
        """
        # Create temp file in PID-isolated directory and close handle to avoid Windows file locking
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False, 
                                         dir=str(temp_dir), encoding="utf-8")
        tmp_name = tmp.name
        tmp.close()  # Close handle before GeoPandas writes to avoid file locking on Windows
        
        # Use GeoPandas built-in GeoJSON export - handles all data types correctly
        gdf.to_file(tmp_name, driver='GeoJSON')
        
        # Return compatible object with name attribute
        class TempFile:
            def __init__(self, name):
                self.name = name
        
        return TempFile(tmp_name)

    def _gdf_to_gpkg_tempfile(self, gdf: gpd.GeoDataFrame, layer_name: str, temp_dir: Path):
        """Write GeoDataFrame to a temporary GeoPackage with a single layer.
        Returns an object with a .name pointing to the .gpkg path."""
        tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".gpkg", delete=False, dir=str(temp_dir))
        tmp_name = tmp.name
        tmp.close()

        # Ensure CRS is EPSG:4326 (if your pipeline guarantees it already, skip)
        if gdf.crs is not None and str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
            gdf = gdf.to_crs(epsg=4326)

        gdf.to_file(tmp_name, driver="GPKG", layer=layer_name)

        class TempFile:  # mimic NamedTemporaryFile interface used by existing code
            def __init__(self, name): self.name = name
        return TempFile(tmp_name)

    def _export_to_geojson(
        self,
        data: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame],
        output_path: Path,
        target_name: str,
        raw_export: bool,
        include_metadata: bool
    ) -> None:
        """Export to GeoJSON format - preserve existing functionality"""
        
        # Handle both single GeoDataFrame and multi-layer dict
        if isinstance(data, dict):
            # Multi-layer: combine all features
            all_features = []
            layer_counts = {}
            
            for layer_name, gdf in data.items():
                # Use standard geopandas export - data should already be clean
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
            }
            
            if include_metadata:
                geojson_data["metadata"] = {
                    "generated": datetime.utcnow().isoformat(),
                    "source": "overture-agol-pipeline",
                    "target": target_name,
                    "data_type": "raw_overture" if raw_export else "agol_transformed",
                    "layers": layer_counts,
                    "total_count": len(all_features)
                }
        else:
            # Single GeoDataFrame - use standard geopandas export
            features_json = data.to_json()
            features_data = json.loads(features_json)
            
            geojson_data = {
                "type": "FeatureCollection",
                "features": features_data.get("features", []),
            }
            
            if include_metadata:
                geojson_data["metadata"] = {
                    "generated": datetime.utcnow().isoformat(),
                    "source": "overture-agol-pipeline", 
                    "target": target_name,
                    "data_type": "raw_overture" if raw_export else "agol_transformed",
                    "count": len(features_data.get("features", []))
                }
        
        # Use GeoPandas to_file() for clean, standard GeoJSON formatting (same as raw export)
        data.to_file(output_path, driver='GeoJSON')
        
        # Validate written file
        if not self._validate_geojson_file(str(output_path)):
            raise ValueError(f"Generated GeoJSON file is invalid: {output_path}")
        
        logger.info(f"GeoJSON export completed: {len(geojson_data['features']):,} features written to {output_path}")

    def _export_to_gpkg(
        self,
        data: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame],
        output_path: Path,
        target_name: str,
        raw_export: bool,
        include_metadata: bool
    ) -> None:
        """Export to GeoPackage format with layer support"""
        
        if isinstance(data, dict):
            # Multi-layer export (e.g., education = places + buildings)
            for i, (layer_name, gdf) in enumerate(data.items()):
                mode = 'w' if i == 0 else 'a'  # Write first, append rest
                layer_table = f"{target_name}_{layer_name}" if not raw_export else layer_name
                
                logger.debug(f"Exporting layer '{layer_table}' with {len(gdf)} features")
                gdf.to_file(output_path, driver='GPKG', layer=layer_table, mode=mode)
        else:
            # Single layer export
            layer_name = target_name if not raw_export else "features"
            logger.debug(f"Exporting single layer '{layer_name}' with {len(data)} features")
            data.to_file(output_path, driver='GPKG', layer=layer_name)
        
        # Add metadata if requested
        if include_metadata:
            self._add_gpkg_metadata(output_path, target_name, raw_export)

    def _export_to_fgdb(
        self,
        data: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame],
        output_path: Path,
        target_name: str,
        raw_export: bool,
        include_metadata: bool
    ) -> None:
        """Export to ESRI File Geodatabase format"""
        
        # Validate FGDB driver availability
        if 'OpenFileGDB' not in fiona.supported_drivers:
            raise RuntimeError("OpenFileGDB driver not available. Please install GDAL with OpenFileGDB support.")
        
        # Create .gdb directory structure (remove if exists)
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
        
        if isinstance(data, dict):
            # Multi-layer export
            for i, (layer_name, gdf) in enumerate(data.items()):
                fc_name = f"{target_name}_{layer_name}" if not raw_export else layer_name
                # Prepare for FGDB compatibility (field name limits)
                gdf = self._prepare_for_fgdb(gdf)
                
                mode = 'w' if i == 0 else 'a'
                logger.debug(f"Exporting feature class '{fc_name}' with {len(gdf)} features")
                gdf.to_file(output_path, driver='OpenFileGDB', layer=fc_name, mode=mode)
        else:
            # Single feature class
            fc_name = target_name if not raw_export else "features"
            data = self._prepare_for_fgdb(data)
            logger.debug(f"Exporting feature class '{fc_name}' with {len(data)} features")
            data.to_file(output_path, driver='OpenFileGDB', layer=fc_name)

    def _prepare_for_fgdb(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Prepare GeoDataFrame for FGDB export (field name limits, etc.)"""
        gdf = gdf.copy()
        
        # Truncate column names to 64 characters (FGDB limit)
        rename_map = {}
        for col in gdf.columns:
            if col != 'geometry' and len(col) > 64:
                new_name = col[:64]
                logger.warning(f"Truncating field name '{col}' to '{new_name}' for FGDB compatibility")
                rename_map[col] = new_name
        
        if rename_map:
            gdf = gdf.rename(columns=rename_map)
        
        return gdf

    def _add_gpkg_metadata(self, output_path: Path, target_name: str, raw_export: bool) -> None:
        """Add metadata table to GeoPackage"""
        import sqlite3
        
        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()
        
        # Create metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Insert metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'target_name': target_name,
            'data_type': 'raw_overture' if raw_export else 'agol_transformed',
            'source': 'Overture Maps Foundation',
            'pipeline': 'Overture-ArcGIS-Pipeline'
        }
        
        for key, value in metadata.items():
            cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
        
        conn.commit()
        conn.close()

    def _validate_geojson_file(self, filepath: str) -> bool:
        """Validate that exported GeoJSON file is properly formatted."""
        try:
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
            
            # Basic GeoJSON structure validation
            if not isinstance(data, dict):
                logger.error("Invalid GeoJSON: root must be an object")
                return False
            
            if data.get("type") != "FeatureCollection":
                logger.error("Invalid GeoJSON: type must be 'FeatureCollection'")
                return False
            
            if "features" not in data:
                logger.error("Invalid GeoJSON: missing 'features' property")
                return False
            
            if not isinstance(data["features"], list):
                logger.error("Invalid GeoJSON: features must be an array")
                return False
            
            return True
        except Exception as e:
            logger.error(f"GeoJSON validation failed: {e}")
            return False


def generate_export_filename(query: str, country: str = None, export_format: str = "geojson", raw_export: bool = False) -> str:
    """
    Generate default export filename based on format
    
    Args:
        query: Query name (e.g., 'education', 'roads')
        country: Country identifier (ISO2, ISO3, or name)
        export_format: Export format (geojson, gpkg, fgdb)
        raw_export: Whether this is a raw export
        
    Returns:
        Generated filename string
    """
    if country:
        from ..config.countries import CountryRegistry
        try:
            country_info = CountryRegistry.get_country(country)
            if country_info:
                iso3 = country_info.iso3.lower()
            else:
                # Fallback if country not found
                iso3 = country.lower()
                logger.warning(f"Country '{country}' not found in registry, using fallback")
        except Exception:
            iso3 = country.lower()
    else:
        iso3 = "unknown"
    
    # Add suffix for raw exports
    suffix = "_raw" if raw_export else ""
    
    # Map formats to extensions
    extensions = {
        "geojson": "geojson",
        "gpkg": "gpkg", 
        "fgdb": "gdb"
    }
    
    extension = extensions.get(export_format, "geojson")
    filename = f"{iso3}_{query}{suffix}.{extension}"
    
    logger.info(f"Generated export filename: {filename}")
    return filename