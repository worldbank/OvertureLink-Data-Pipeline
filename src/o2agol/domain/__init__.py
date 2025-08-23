"""
Domain Models and Types

This module contains the core domain models and enumerations used throughout the pipeline.
These typed models ensure data integrity and provide clear interfaces for the pipeline components.

Models:
- Country: Country information with boundaries and metadata
- Query: Query configuration with field mappings and themes
- RunOptions: Runtime configuration and feature flags

Enums:
- Mode: Publishing modes (auto, initial, overwrite, append)
- ClipStrategy: Spatial clipping strategies (divisions, bbox)  
- TargetType: Target data types (roads, buildings, places, etc.)
- ExportFormat: Export format options (geojson, gpkg, fgdb)
"""

from .enums import ClipStrategy, ExportFormat, Mode, TargetType
from .models import Country, ItemIds, Metadata, Query, RunOptions

__all__ = [
    "Country", "Query", "RunOptions", "ItemIds", "Metadata",
    "Mode", "ClipStrategy", "TargetType", "ExportFormat"
]