"""
Error taxonomy for the Overture pipeline.

All pipeline errors inherit from O2AGOLError and carry a context dict with
canonical fields {country, theme, release, stage} plus any caller-specific extras.
The base class is a plain dataclass (not pydantic) to keep error construction cheap
on the error path; validation of context keys is by convention, not by framework.

Extensibility
-------------
The context dict is deliberately untyped so that downstream projects (the
minimum-data-package-hubs climate-risk category, for example) can carry their
own identifiers such as ``grid_cell`` or ``hazard_type`` without subclassing.

Phase 1 status
--------------
Phase 1 of the hardening pass creates this file and its classes, but **no
pipeline code raises these exceptions yet**. The 78 bare ``except Exception``
sites across the pipeline are migrated to named exceptions in Phase 3, once
the Phase 2 test harness exists as a safety net.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class O2AGOLError(Exception):
    """Base class for all pipeline errors.

    Attributes:
        message: Human-readable short description.
        context: Structured fields for logging and debugging. Canonical keys
            are {country, theme, release, stage} but any keys are allowed.
    """

    message: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure Exception machinery has a sane str representation.
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | context={self.context}"
        return self.message


@dataclass
class OvertureError(O2AGOLError):
    """Raised when Overture data ingestion fails (S3, DuckDB, schema drift)."""


@dataclass
class TransformError(O2AGOLError):
    """Raised when geometry validation or schema transformation fails."""


@dataclass
class PublishError(O2AGOLError):
    """Raised when ArcGIS Online publishing fails (auth, schema, append job)."""


@dataclass
class ExportError(O2AGOLError):
    """Raised when file export fails (GeoJSON/GPKG/FGDB driver errors)."""


@dataclass
class ConfigError(O2AGOLError):
    """Raised when configuration loading or credential resolution fails."""


__all__ = [
    "O2AGOLError",
    "OvertureError",
    "TransformError",
    "PublishError",
    "ExportError",
    "ConfigError",
]
