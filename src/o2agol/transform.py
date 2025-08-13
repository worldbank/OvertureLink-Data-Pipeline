from __future__ import annotations

from typing import Literal

import geopandas as gpd


def normalize_schema(
    gdf: gpd.GeoDataFrame, layer: Literal["roads", "buildings"]
) -> gpd.GeoDataFrame:
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Keep only a few core fields to start. You can expand later.
    keep = [c for c in ["id", "class", "subtype"] if c in gdf.columns]
    cols = keep + ["geometry"]
    gdf = gdf[cols]

    # Ensure id is string (for AGOL upsert)
    if "id" in gdf.columns:
        gdf["id"] = gdf["id"].astype(str)

    return gdf
