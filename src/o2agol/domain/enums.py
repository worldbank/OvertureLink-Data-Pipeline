"""
Pipeline Enumerations

Core enums for type safety and clear interface definitions across the pipeline.
"""

from enum import Enum


class Mode(str, Enum):
    """Publishing modes for AGOL feature layers."""
    AUTO = "auto"           # Smart detection: create new or truncate-append existing
    INITIAL = "initial"     # Force creation of new layer
    OVERWRITE = "overwrite" # Force update of existing layer (requires item_id)
    APPEND = "append"       # Add data to existing layer without clearing


class ClipStrategy(str, Enum):
    """Spatial clipping strategies for data extraction."""
    DIVISIONS = "divisions" # Use Overture Divisions for precise boundaries
    BBOX = "bbox"           # Fast bbox-only filtering for development/testing


class TargetType(str, Enum):
    """Target data types available for processing."""
    ROADS = "roads"         # Transportation networks
    BUILDINGS = "buildings" # Building footprints
    PLACES = "places"       # Points of interest
    EDUCATION = "education" # Education facilities (multi-layer)
    HEALTH = "health"       # Health facilities (multi-layer) 
    MARKETS = "markets"     # Retail facilities (multi-layer)


class ExportFormat(str, Enum):
    """Export format options for data output."""
    GEOJSON = "geojson"     # Standards-compliant JSON format
    GPKG = "gpkg"           # SQLite-based format with multi-layer support
    FGDB = "fgdb"           # ESRI File Geodatabase format


class StagingFormat(str, Enum):
    """Staging formats for batch processing."""
    GEOJSON = "geojson"     # GeoJSON staging files
    GPKG = "gpkg"           # GeoPackage staging files
    FGDB = "fgdb"           # File Geodatabase staging files