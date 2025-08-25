"""
Pipeline Domain Models

Pydantic models for type safety and validation across the pipeline.
These models ensure data integrity and provide clear interfaces.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .enums import ClipStrategy


class Country(BaseModel):
    """Country information with spatial boundaries and metadata."""
    name: str = Field(..., description="Full country name")
    iso2: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    iso3: str = Field(..., description="ISO 3166-1 alpha-3 country code")
    bounds: tuple[float, float, float, float] = Field(..., description="Bounding box (minx, miny, maxx, maxy)")
    region: Optional[str] = Field(None, description="Geographic region")
    
    class Config:
        """Pydantic configuration."""
        frozen = True  # Make immutable for safety


class Query(BaseModel):
    """Query configuration with theme, filters, and metadata."""
    name: str = Field(..., description="Query identifier (roads, buildings, etc.)")
    theme: str = Field(..., description="Overture theme (transportation, buildings, places)")
    type: str = Field(..., description="Overture data type (segment, building, place)")
    is_multilayer: bool = Field(default=False, description="Whether query produces multiple layers")
    
    # Filter configuration
    filter: Optional[str] = Field(None, description="Primary filter expression")
    building_filter: Optional[str] = Field(None, description="Building filter for dual-theme queries")
    category_filter: Optional[str] = Field(None, description="Category filter for places/buildings")
    
    # Additional pipeline attributes
    field_mappings: Optional[dict] = Field(default_factory=dict, description="Field mappings for transformation")
    original_config: Optional[dict] = Field(None, description="Original config for backward compatibility")
    
    # Metadata fields for AGOL publishing
    sector_title: Optional[str] = Field(None, description="Human-readable sector title")
    sector_description: Optional[str] = Field(None, description="Sector description for metadata")
    sector_tag: Optional[str] = Field(None, description="Short tag for naming")
    data_type: Optional[str] = Field(None, description="Data type description")
    tags: Optional[list[str]] = Field(None, description="Tags for categorization")
    upsert_key: Optional[str] = Field(None, description="Key field for upsert operations")
    
    class Config:
        """Pydantic configuration."""
        frozen = True


class RunOptions(BaseModel):
    """Runtime configuration and feature flags."""
    clip: ClipStrategy = Field(default=ClipStrategy.DIVISIONS, description="Spatial clipping strategy")
    limit: Optional[int] = Field(None, description="Feature limit for testing")
    use_bbox: bool = Field(default=False, description="Use bbox-only clipping")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True  # Allow Path types


class ItemIds(BaseModel):
    """AGOL item ID mappings for existing feature layers."""
    roads: Optional[str] = None
    buildings: Optional[str] = None
    places: Optional[str] = None
    education: Optional[str] = None
    health: Optional[str] = None
    markets: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow additional fields for custom queries


class Metadata(BaseModel):
    """Feature layer metadata for publishing."""
    title: str = Field(..., description="Feature layer title")
    description: str = Field(..., description="Feature layer description")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    snippet: Optional[str] = Field(None, description="Short description")
    credits: Optional[str] = Field(None, description="Data source credits")
    use_limitations: Optional[str] = Field(None, description="Usage limitations")
    
    class Config:
        """Pydantic configuration."""
        frozen = True


class DumpConfig(BaseModel):
    """Configuration for local dump management."""
    dump_path: Path = Field(default=Path("/overturedump"), description="Base path for dumps")
    max_memory: str = Field(default="32GB", description="Maximum memory for dump operations")
    chunk_size: int = Field(default=5, description="Countries per processing chunk")
    
    # Backward compatibility fields for existing tests
    base_path: Optional[str] = Field(None, description="Legacy base path field")
    max_memory_gb: Optional[int] = Field(None, description="Legacy memory limit in GB")
    parallel_downloads: Optional[int] = Field(None, description="Legacy parallel downloads setting")
    use_parallel_http: Optional[bool] = Field(None, description="Legacy parallel HTTP setting")
    enable_spatial_hash: Optional[bool] = Field(None, description="Legacy spatial hash setting")
    use_pyarrow_queries: Optional[bool] = Field(None, description="Legacy PyArrow queries setting")
    use_world_bank_boundaries: Optional[bool] = Field(None, description="Legacy WB boundaries setting")
    boundary_simplify_tolerance: Optional[float] = Field(None, description="Legacy boundary tolerance")
    enable_boundary_cache: Optional[bool] = Field(None, description="Legacy boundary cache setting")
    
    def __post_init__(self):
        """Handle backward compatibility conversions."""
        if self.base_path and not self.dump_path:
            self.dump_path = Path(self.base_path)
        if self.max_memory_gb and not self.max_memory:
            self.max_memory = f"{self.max_memory_gb}GB"
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        extra = "allow"  # Allow additional fields for backward compatibility