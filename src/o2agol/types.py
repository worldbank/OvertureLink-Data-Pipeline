"""
Type definitions for the Overture-ArcGIS-Pipeline staging system.

This module provides type safety and structured configuration for staging operations,
particularly for the hardened append path with multiple staging format options.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class StagingFormat(str, Enum):
    """Enumeration of supported staging formats for AGOL append operations.
    
    GEOJSON: Default format, broadly compatible but less efficient for large datasets
    GPKG: GeoPackage format, better compression and performance for large polygon datasets
    """
    GEOJSON = "geojson"
    GPKG = "gpkg"


@dataclass(frozen=True)
class Part:
    """Metadata for batch processing parts."""
    item_id: str
    name: str
    est_features: int = 0


@dataclass(frozen=True)
class AppendOptions:
    """Immutable configuration object for append operations.
    
    Provides comprehensive configuration for the hardened append workflow
    with sensible defaults for production use.
    """
    timeout_s: int = 5400  # 90 minutes for large datasets
    max_retries: int = 2
    poll_interval_s: float = 2.0
    rollback: bool = True  # Enable rollback on failure
    return_messages: bool = True  # Get detailed job messages
    staging_format: StagingFormat = StagingFormat.GEOJSON
    source_table_name: Optional[str] = None  # Required for container formats like GPKG


@dataclass(frozen=True)  
class AppendJobResult:
    """Result metadata from append operations.
    
    Provides detailed information about the append operation outcome
    for monitoring and debugging purposes.
    """
    added: int = 0
    updated: int = 0
    failed: int = 0
    duration_s: float = 0.0
    staging_format_used: StagingFormat = StagingFormat.GEOJSON
    messages: Optional[list] = None


# Staging-specific exception hierarchy
class StagingError(Exception):
    """Base exception for staging operations."""
    pass


class StagingFormatError(StagingError):
    """Error in staging format conversion or processing."""
    def __init__(self, format_type: StagingFormat, message: str):
        self.format_type = format_type
        super().__init__(f"Staging format {format_type.value} error: {message}")


class StagingCleanupError(StagingError):
    """Error during staging file or resource cleanup."""
    pass


class StagingValidationError(StagingError):
    """Error in staging data validation or schema checking."""
    pass


class StagingUploadError(StagingError):
    """Error during staging file upload to AGOL."""
    def __init__(self, item_id: Optional[str], message: str):
        self.item_id = item_id
        super().__init__(f"Staging upload error (item: {item_id}): {message}")