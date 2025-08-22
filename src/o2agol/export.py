"""
Export functionality for Overture Maps data in multiple formats.

This module provides unified export capabilities to GeoJSON, GeoPackage (GPKG), 
and File Geodatabase (FGDB) formats with optional raw data preservation.
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Optional
from enum import Enum

import geopandas as gpd
import fiona

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats for Overture data"""
    GEOJSON = "geojson"
    GPKG = "gpkg"
    FGDB = "fgdb"
    
    @classmethod
    def from_extension(cls, path: str) -> 'ExportFormat':
        """Infer format from file extension"""
        ext = Path(path).suffix.lower()
        mapping = {
            '.geojson': cls.GEOJSON,
            '.json': cls.GEOJSON,
            '.gpkg': cls.GPKG,
            '.gdb': cls.FGDB
        }
        return mapping.get(ext, cls.GEOJSON)


def export_data(
    data: Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]],
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
        _export_to_geojson(data, output_path, target_name, raw_export, include_metadata)
    elif format_enum == ExportFormat.GPKG:
        _export_to_gpkg(data, output_path, target_name, raw_export, include_metadata)
    elif format_enum == ExportFormat.FGDB:
        _export_to_fgdb(data, output_path, target_name, raw_export, include_metadata)
    else:
        raise ValueError(f"Unsupported export format: {export_format}")
    
    logger.info(f"Successfully exported to {output_path} ({export_format})")


def _export_to_geojson(
    data: Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]],
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
    
    # Write with proper encoding and formatting
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f, indent=2, ensure_ascii=False)
    
    # Validate written file
    if not _validate_geojson_file(str(output_path)):
        raise ValueError(f"Generated GeoJSON file is invalid: {output_path}")
    
    logger.info(f"GeoJSON export completed: {len(geojson_data['features']):,} features written to {output_path}")


def _export_to_gpkg(
    data: Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]],
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
        _add_gpkg_metadata(output_path, target_name, raw_export)


def _export_to_fgdb(
    data: Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]],
    output_path: Path,
    target_name: str,
    raw_export: bool,
    include_metadata: bool
) -> None:
    """Export to ESRI File Geodatabase format"""
    
    # Validate FGDB driver availability
    if 'FileGDB' not in fiona.supported_drivers:
        raise RuntimeError("FileGDB driver not available. Please install GDAL with FileGDB support.")
    
    # Create .gdb directory structure (remove if exists)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    if isinstance(data, dict):
        # Multi-layer export
        for i, (layer_name, gdf) in enumerate(data.items()):
            fc_name = f"{target_name}_{layer_name}" if not raw_export else layer_name
            # Prepare for FGDB compatibility (field name limits)
            gdf = _prepare_for_fgdb(gdf)
            
            mode = 'w' if i == 0 else 'a'
            logger.debug(f"Exporting feature class '{fc_name}' with {len(gdf)} features")
            gdf.to_file(output_path, driver='FileGDB', layer=fc_name, mode=mode)
    else:
        # Single feature class
        fc_name = target_name if not raw_export else "features"
        data = _prepare_for_fgdb(data)
        logger.debug(f"Exporting feature class '{fc_name}' with {len(data)} features")
        data.to_file(output_path, driver='FileGDB', layer=fc_name)


def _prepare_for_fgdb(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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


def _add_gpkg_metadata(output_path: Path, target_name: str, raw_export: bool) -> None:
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


def _validate_geojson_file(filepath: str) -> bool:
    """Validate that exported GeoJSON file is properly formatted."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
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
        from .config.countries import CountryRegistry
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