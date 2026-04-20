"""
Publish-time output contracts for AGOL-bound data.

These contracts validate the shape of transformed GeoDataFrames and item
metadata before any ArcGIS network call is attempted.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from pydantic import BaseModel, Field, model_validator

_CORE_REQUIRED_COLUMNS = {"id", "geometry"}
_RUNTIME_METADATA_COLUMNS = {"processed_date", "country_iso3", "country_name"}


class PublishMetadataContract(BaseModel):
    """Minimal metadata contract for an AGOL item update/publish payload."""

    title: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tags: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        tags = normalized.get("tags")
        if isinstance(tags, str):
            normalized["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]
        return normalized


class PublishLayerContract(BaseModel):
    """Summary contract for a single transformed AGOL layer payload."""

    layer_name: str = Field(min_length=1)
    feature_count: int = Field(ge=0)
    geometry_column: str = Field(min_length=1)
    crs_epsg: int | None = None
    columns: list[str] = Field(min_length=1)
    required_columns: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_columns(self) -> PublishLayerContract:
        available = set(self.columns)
        missing = [column for column in self.required_columns if column not in available]
        if missing:
            raise ValueError(
                f"missing required AGOL output columns for layer '{self.layer_name}': {missing}"
            )
        if self.geometry_column != "geometry":
            raise ValueError(
                f"layer '{self.layer_name}' must use 'geometry' as active geometry column, "
                f"got '{self.geometry_column}'"
            )
        if self.crs_epsg is not None and self.crs_epsg != 4326:
            raise ValueError(
                f"layer '{self.layer_name}' must be EPSG:4326 before publish, got EPSG:{self.crs_epsg}"
            )
        return self

    @classmethod
    def from_geodataframe(cls, layer_name: str, gdf: gpd.GeoDataFrame) -> PublishLayerContract:
        required = set(_CORE_REQUIRED_COLUMNS)
        if len(gdf) > 0:
            required |= _RUNTIME_METADATA_COLUMNS

        crs_epsg = None
        if gdf.crs is not None:
            try:
                crs_epsg = gdf.crs.to_epsg()
            except Exception:
                crs_epsg = None

        return cls(
            layer_name=layer_name,
            feature_count=len(gdf),
            geometry_column=getattr(gdf.geometry, "name", ""),
            crs_epsg=crs_epsg,
            columns=list(gdf.columns),
            required_columns=sorted(required),
        )


def validate_publish_contracts(
    layer_data: dict[str, gpd.GeoDataFrame],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Validate publish inputs before any AGOL network call is attempted."""

    for layer_name, gdf in layer_data.items():
        PublishLayerContract.from_geodataframe(layer_name=layer_name, gdf=gdf)

    if metadata is not None:
        PublishMetadataContract.model_validate(metadata)
