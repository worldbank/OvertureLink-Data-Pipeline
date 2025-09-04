"""
Transformer - Data Schema Normalization and Transformation

Handles field mapping, geometry normalization, and metadata enrichment.
Integrates proven transformation logic from the existing pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.wkb as _swkb
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry

from ..domain.models import Country, Query

# Constants for AGOL compatibility
AGOL_STRING_MAX = 255  # safe default for text fields

# Geometry validation constants
MIN_POLYGON_AREA = 1e-12  # Only remove truly degenerate polygons (~0.01 m² = 1 cm²)
MIN_LINE_LENGTH = 1e-10   # Only remove truly degenerate lines (~1 mm globally)

# AGOL reserved keywords and field name improvements
RESERVED_KEYWORDS = {
    'select': 'select_field',
    'from': 'from_field', 
    'where': 'where_field',
    'order': 'order_field',
    'group': 'group_field',
    'by': 'by_field',
    'join': 'join_field',
    'inner': 'inner_field',
    'left': 'left_field',
    'right': 'right_field',
    'on': 'on_field',
    'as': 'as_field',
    'distinct': 'distinct_field',
    'count': 'count_field',
    'sum': 'sum_field',
    'avg': 'avg_field',
    'max': 'max_field',
    'min': 'min_field',
    'table': 'table_field',
    'column': 'column_field',
    'index': 'index_field',
    'key': 'key_field',
    'primary': 'primary_field',
    'foreign': 'foreign_field',
    'unique': 'unique_field',
    'not': 'not_field',
    'null': 'null_field',
    'and': 'and_field',
    'or': 'or_field',
    'in': 'in_field',
    'like': 'like_field',
    'between': 'between_field',
    'exists': 'exists_field',
    'having': 'having_field',
    'union': 'union_field',
    'intersect': 'intersect_field',
    'except': 'except_field'
}

FIELD_IMPROVEMENTS = {
    'bbox_xmin': 'bbox_west',
    'bbox_ymin': 'bbox_south', 
    'bbox_xmax': 'bbox_east',
    'bbox_ymax': 'bbox_north',
    'primary_name': 'name_primary',
    'common_name': 'name_common',
    'alternate_name': 'name_alternate',
    'full_address': 'address_full',
    'locality': 'address_locality',
    'region': 'address_region',
    'postcode': 'address_postcode',
    'country': 'address_country',
    'phone_number': 'phone',
    'email_address': 'email',
    'website_url': 'website',
    'opening_hours': 'hours',
    'primary_category': 'category_primary',
    'alternate_category': 'category_alternate',
    'building_height': 'height_m',
    'floor_count': 'floors',
    'construction_year': 'year_built'
}

# Preferred column order for publishing
PREFERRED_ORDER = [
    "id",
    "name",  
    "road_class", "road_type",
    "building_class", "building_type", "height_m", "floors",
    "feature_type", "name_primary", "name_common",
    "category_primary", "category_alternate",
    "address_full", "address_locality", "address_country",
    "website", "email", "phone",
]


class Transformer:
    """
    Data transformation and schema normalization for AGOL compatibility.
    
    Handles field mapping, geometry validation, stable ID retention,
    and metadata enrichment for downstream publishing or export.
    """
    
    def __init__(self, query: Query):
        """
        Initialize transformer with query configuration.
        
        Args:
            query: Query configuration containing field mappings and metadata
        """
        self.query = query
        
    def normalize(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Apply field mapping, geometry normalization, and stable ID retention.
        
        Args:
            df: Raw GeoDataFrame from data source
            
        Returns:
            Normalized GeoDataFrame ready for publishing/export
        """
        if df.empty:
            logging.warning("Empty GeoDataFrame provided for transformation")
            return df
            
        # Determine layer type from query name for schema normalization
        layer_type = self._get_layer_type()
        
        logging.info(f"Normalizing schema for {layer_type} ({len(df):,} features)")
        
        # Apply the full schema normalization pipeline
        normalized_gdf = normalize_schema(df, layer_type)
        
        logging.info(f"Schema normalization completed: {len(normalized_gdf):,} features processed")
        return normalized_gdf
        
    def add_metadata(self, df: gpd.GeoDataFrame, country: Country) -> gpd.GeoDataFrame:
        """
        Add metadata fields like titles, descriptions, timestamps.
        
        Args:
            df: Normalized GeoDataFrame
            country: Country information for metadata context
            
        Returns:
            GeoDataFrame enriched with metadata
        """
        if df.empty:
            return df
            
        result_df = df.copy()
        
        # Add processing metadata
        result_df['processed_date'] = datetime.now().isoformat()
        result_df['country_iso3'] = country.iso3
        result_df['country_name'] = country.name
        
        # Add query-specific metadata if available
        if hasattr(self.query, 'sector_title') and self.query.sector_title:
            result_df['data_sector'] = self.query.sector_title
            
        logging.debug(f"Added metadata fields to {len(result_df):,} features")
        return result_df
        
    def _get_layer_type(self) -> Literal["roads", "buildings", "education", "health", "markets", "places"]:
        """
        Determine layer type from query configuration for schema normalization.
        
        Returns:
            Layer type for schema normalization
        """
        query_name = self.query.name.lower()
        
        if 'road' in query_name or self.query.theme == 'transportation':
            return 'roads'
        elif 'building' in query_name or self.query.theme == 'buildings':
            return 'buildings'
        elif query_name in ['education', 'health', 'markets']:
            return query_name  # type: ignore[return-value]
        elif 'place' in query_name or self.query.theme == 'places':
            return 'places'
        else:
            # Default to places for unknown types
            return 'places'


# ============================================================================
# Core transformation functions
# ============================================================================

def _force_2d(geom: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    """Force geometry to 2D for AGOL compatibility."""
    if geom is None:
        return None
    try:
        return _swkb.loads(_swkb.dumps(geom, output_dimension=2))
    except Exception:
        return geom


def _make_valid_if_needed(geom: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    """Make invalid geometries valid using GeoPandas make_valid()."""
    if geom is None:
        return None
    try:
        if not geom.is_valid:
            # Use make_valid() instead of buffer(0)
            return geom.make_valid()
        return geom
    except Exception:
        # Fallback to buffer(0) only if make_valid fails
        try:
            return geom if geom.is_valid else geom.buffer(0)
        except Exception:
            return geom



def _preserve_simple_polygon(geom: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    """
    If a geometry is a MultiPolygon with exactly one part, return that single Polygon.
    Otherwise, return the geometry unchanged.
    Keeps valid Polygons as Polygon (avoids everything becoming MultiPolygon).
    """
    if geom is None:
        return None
    try:
        if isinstance(geom, MultiPolygon) and len(geom.geoms) == 1:
            return geom.geoms[0]
        return geom
    except Exception:
        # If anything odd happens, fall back to the original geometry
        return geom


def enforce_geometry_rules(
    gdf: gpd.GeoDataFrame,
    expected: Literal["points", "lines", "polygons"],
    validate_polygons: bool = True
) -> gpd.GeoDataFrame:
    """Enforce geometry rules for AGOL compatibility."""
    # Ensure WGS84 coordinate system
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
        
    # Remove empty or null geometries
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    
    # Force 2D geometries
    gdf.geometry = gdf.geometry.apply(_force_2d)
    
    # Validate polygons if requested
    if expected == "polygons" and validate_polygons:
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(_make_valid_if_needed)
    
    # Remove degenerate geometries based on improved thresholds
    if expected == "polygons":
        # Remove polygons with area below threshold
        gdf = gdf[gdf.geometry.area > MIN_POLYGON_AREA]
    elif expected == "lines":
        # Remove lines with length below threshold  
        gdf = gdf[gdf.geometry.length > MIN_LINE_LENGTH]
    
    # Filter by expected geometry family
    fam = {
        "points": {"Point", "MultiPoint"},
        "lines": {"LineString", "MultiLineString"},
        "polygons": {"Polygon", "MultiPolygon"},
    }[expected]
    
    bad = ~gdf.geometry.geom_type.isin(fam)
    if bad.any():
        logging.warning("Dropping %d features not matching expected geometry family=%s", int(bad.sum()), expected)
        gdf = gdf.loc[~bad].copy()
        
    return gdf


def sanitize_field_names(gdf: gpd.GeoDataFrame, layer_type: str = "generic") -> gpd.GeoDataFrame:
    """
    Sanitize field names for AGOL compatibility while preserving meaningful names.
    Only handles length limits and character cleanup - no renaming.
    """
    rename_mapping = {}
    
    for col in gdf.columns:
        if col == 'geometry':  # Skip geometry column
            continue
            
        new_name = col
        
        # Ensure field name length is under 31 characters (AGOL limit)
        if len(new_name) > 30:
            new_name = new_name[:30]
            
        # Clean up any remaining character issues
        new_name = new_name.replace(' ', '_').replace('-', '_')
        
        if new_name != col:
            rename_mapping[col] = new_name
    
    # Apply the renaming only for length/character issues
    if rename_mapping:
        gdf = gdf.rename(columns=rename_mapping)
    
    return gdf


def _clip_strings(series: pd.Series, max_len: int = AGOL_STRING_MAX) -> pd.Series:
    """Clip string values to maximum length for AGOL compatibility."""
    if series.dtype != "object":
        return series
    return series.apply(lambda v: None if (v is None or pd.isna(v)) else str(v)[:max_len])


def order_columns_for_publish(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Order columns in preferred sequence for publishing."""
    cols = [c for c in PREFERRED_ORDER if c in gdf.columns and c != "geometry"]
    rest = [c for c in gdf.columns if c not in cols and c != "geometry"]
    return gpd.GeoDataFrame(gdf[cols + rest + ["geometry"]], geometry="geometry", crs=gdf.crs)


def normalize_schema(
    gdf: gpd.GeoDataFrame, 
    layer: Literal["roads", "buildings", "education", "health", "markets", "places"]
) -> gpd.GeoDataFrame:
    """
    Normalize Overture data schema to AGOL-compatible flat structure.
    
    Flattens all complex nested fields (names, categories, addresses, etc.) into simple fields
    that AGOL can natively handle. This eliminates serialization issues with numpy arrays
    and nested dictionaries.
    """
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Apply field name sanitization first
    gdf = sanitize_field_names(gdf, layer_type=layer)

    # Create a copy to avoid modifying original
    result_gdf = gdf.copy()
    
    # Clean all complex data types early to prevent numpy array issues
    result_gdf = _clean_complex_data_types(result_gdf)

    if layer == "roads":
        result_gdf = _normalize_roads_schema(result_gdf)
        
    elif layer == "buildings":
        result_gdf = _normalize_buildings_schema(result_gdf)
        
    elif layer in ["education", "health", "markets", "places"]:
        result_gdf = _normalize_places_schema(result_gdf)
    
    else:
        # Default fallback - keep basic fields only
        keep_cols = [c for c in ["id", "type", "subtype"] if c in result_gdf.columns]
        result_gdf = result_gdf[keep_cols + ["geometry"]].copy()

    # Final data type cleanup
    result_gdf = _finalize_data_types(result_gdf)
    
    return result_gdf


def _clean_complex_data_types(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean all complex data types that contain numpy arrays or nested structures.
    This must be done early to prevent serialization issues later.
    """
    cleaned_gdf = gdf.copy()
    
    for col in cleaned_gdf.columns:
        if col == 'geometry':
            continue
            
        if cleaned_gdf[col].dtype == 'object':
            # Clean any numpy arrays or complex nested structures
            cleaned_gdf[col] = cleaned_gdf[col].apply(_clean_value_recursive)
    
    return cleaned_gdf


def _clean_value_recursive(value: Any) -> Any:
    """
    Recursively clean a value by converting numpy arrays to lists.
    This ensures no numpy arrays remain in the data.
    """
    try:
        if value is None or pd.isna(value):
            return None
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: _clean_value_recursive(v) for k, v in value.items()}
        elif isinstance(value, list | tuple):
            return [_clean_value_recursive(item) for item in value]
        else:
            return value
    except (ValueError, TypeError):
        # If cleaning fails, convert to string
        return str(value) if value is not None else None

# =====================================================================
# Schema normalization
# The three functions below normalize the schema for Overture's
# three types: roads, buildings, and places
# =====================================================================

def _normalize_roads_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalize transportation/segment schema to simple AGOL-compatible fields.
    
    Flattens the complex 'names' field into simple string fields.
    """
    result_gdf = gdf[["id", "geometry"]].copy()
    
    if "name" in gdf.columns:
        result_gdf["name"] = gdf["name"].astype(str)
    
    # Add basic classification fields
    if "class" in gdf.columns:
        result_gdf["road_class"] = gdf["class"].astype(str)
    
    if "subtype" in gdf.columns:
        result_gdf["road_type"] = gdf["subtype"].astype(str)
    
    # Flatten names field (preserves both original and flattened fields)
    if "names" in gdf.columns:
        names_data = _flatten_names_field(gdf["names"])
        result_gdf = pd.concat([result_gdf, names_data], axis=1)
    
    # Add other simple fields if available
    simple_fields = ["road_surface", "road_flags", "speed_limits"]
    for field in simple_fields:
        if field in gdf.columns:
            result_gdf[field] = gdf[field].astype(str)
    
    # Apply transform helpers before return
    result_gdf = enforce_geometry_rules(result_gdf, expected="lines", validate_polygons=False)
    # clip long strings
    for col in result_gdf.columns:
        if col != "geometry":
            result_gdf[col] = _clip_strings(result_gdf[col], AGOL_STRING_MAX)
    # stable column order
    result_gdf = order_columns_for_publish(result_gdf)
    
    return result_gdf


def _normalize_buildings_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalize building schema to simple AGOL-compatible fields.
    """
    result_gdf = gdf[["id", "geometry"]].copy()
    
    if "name" in gdf.columns:
        result_gdf["name"] = gdf["name"].astype(str)
    
    # Add building classification
    if "class" in gdf.columns:
        result_gdf["building_class"] = gdf["class"].astype(str)
    
    if "subtype" in gdf.columns:
        result_gdf["building_type"] = gdf["subtype"].astype(str)
    
    # Flatten names field if present (preserves both original and flattened fields)
    if "names" in gdf.columns:
        names_data = _flatten_names_field(gdf["names"])
        result_gdf = pd.concat([result_gdf, names_data], axis=1)
    
    # Add numeric fields with proper type handling
    if "height" in gdf.columns:
        result_gdf["height_m"] = _safe_numeric_convert(gdf["height"], float)
    
    if "num_floors" in gdf.columns:
        result_gdf["floors"] = _safe_numeric_convert(gdf["num_floors"], int)
    elif "floor_count" in gdf.columns:
        result_gdf["floors"] = _safe_numeric_convert(gdf["floor_count"], int)
    
    # Apply geometry validation
    result_gdf = enforce_geometry_rules(result_gdf, expected="polygons", validate_polygons=True)

    # Preserve simple Polygons as Polygon (avoid coercing everything to MultiPolygon)
    result_gdf.geometry = result_gdf.geometry.apply(_preserve_simple_polygon)
    
    # clip long strings
    for col in result_gdf.columns:
        if col != "geometry":
            result_gdf[col] = _clip_strings(result_gdf[col], AGOL_STRING_MAX)
    # stable column order
    result_gdf = order_columns_for_publish(result_gdf)
    
    return result_gdf


def _normalize_places_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normalize places schema to simple AGOL-compatible fields.
    Handles both places and buildings data in education/health/markets queries.
    """
    result_gdf = gdf[["id", "geometry"]].copy()
    
    if "name" in gdf.columns:
        result_gdf["name"] = gdf["name"].astype(str)
    
    # Add source type if available (distinguishes places from buildings)
    if "source_type" in gdf.columns:
        result_gdf["feature_type"] = gdf["source_type"].astype(str)
    
    # Flatten names field (preserves both original and flattened fields)
    if "names" in gdf.columns:
        names_data = _flatten_names_field(gdf["names"])
        result_gdf = pd.concat([result_gdf, names_data], axis=1)
    
    # Flatten categories field (places only)
    if "categories" in gdf.columns:
        categories_data = _flatten_categories_field(gdf["categories"])
        result_gdf = pd.concat([result_gdf, categories_data], axis=1)
    
    # Flatten addresses field (places only)  
    if "addresses" in gdf.columns:
        addresses_data = _flatten_addresses_field(gdf["addresses"])
        result_gdf = pd.concat([result_gdf, addresses_data], axis=1)
    
    # Flatten contact fields
    if "websites" in gdf.columns:
        result_gdf["website"] = gdf["websites"].apply(_extract_first_from_array)
    
    if "emails" in gdf.columns:
        result_gdf["email"] = gdf["emails"].apply(_extract_first_from_array)
    
    if "phones" in gdf.columns:
        result_gdf["phone"] = gdf["phones"].apply(_extract_first_from_array)
    
    # Add confidence field if available (places only)
    if "confidence" in gdf.columns:
        result_gdf["confidence"] = _safe_numeric_convert(gdf["confidence"], float)
    
    # Add building-specific fields for mixed data
    if "class" in gdf.columns:
        result_gdf["building_class"] = gdf["class"].astype(str)
    if "height" in gdf.columns:
        result_gdf["height_m"] = _safe_numeric_convert(gdf["height"], float)
    
    if "floor_count" in gdf.columns:
        result_gdf["floors"] = _safe_numeric_convert(gdf["floor_count"], int)
    
    return result_gdf


def _flatten_names_field(names_series: pd.Series) -> pd.DataFrame:
    """
    Flatten Overture names field into simple string columns.
    
    From: {"primary": "Main St", "common": {"en": "Main Street"}, "rules": [...]}
    To: name_primary, name_common columns
    """
    result_data: dict[str, list[Optional[str]]] = {
        "name_primary": [],
        "name_common": []
    }
    
    for names_value in names_series:
        primary, common = _extract_names_from_value(names_value)
        result_data["name_primary"].append(primary)
        result_data["name_common"].append(common)
    
    return pd.DataFrame(result_data, index=names_series.index)


def _flatten_categories_field(categories_series: pd.Series) -> pd.DataFrame:
    """
    Flatten Overture categories field into simple string columns.
    
    From: {"primary": "restaurant", "alternate": ["fast_food", "cafe"]}
    To: category_primary, category_alternate columns
    """
    result_data: dict[str, list[Optional[str]]] = {
        "category_primary": [],
        "category_alternate": []
    }
    
    for categories_value in categories_series:
        primary, alternate = _extract_categories_from_value(categories_value)
        result_data["category_primary"].append(primary)
        result_data["category_alternate"].append(alternate)
    
    return pd.DataFrame(result_data, index=categories_series.index)


def _flatten_addresses_field(addresses_series: pd.Series) -> pd.DataFrame:
    """
    Flatten Overture addresses field into simple string columns.
    
    From: [{"freeform": "123 Main St", "locality": "City", "country": "US"}]
    To: address_full, address_locality, address_country columns  
    """
    result_data: dict[str, list[Optional[str]]] = {
        "address_full": [],
        "address_locality": [],
        "address_country": []
    }
    
    for addresses_value in addresses_series:
        full, locality, country = _extract_address_from_value(addresses_value)
        result_data["address_full"].append(full)
        result_data["address_locality"].append(locality)
        result_data["address_country"].append(country)
    
    return pd.DataFrame(result_data, index=addresses_series.index)


# ============================================================================
# Helper functions for field extraction and data type conversion
# ============================================================================

def _extract_names_from_value(names_value: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract primary and common names from a names value."""
    try:
        if not names_value:
            return None, None
            
        # Handle string JSON
        if isinstance(names_value, str):
            names_value = json.loads(names_value)
        
        # Handle dict
        if isinstance(names_value, dict):
            primary = names_value.get("primary")
            
            # Extract common name from various formats
            common = None
            common_field = names_value.get("common")
            if isinstance(common_field, dict):
                # Get first language variant
                common = next(iter(common_field.values())) if common_field else None
            elif isinstance(common_field, str):
                common = common_field
            elif isinstance(common_field, list) and common_field:
                common = str(common_field[0])
            
            return _safe_string(primary), _safe_string(common)
            
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logging.debug(f"Failed to parse names value: {e}")
    
    return _safe_string(names_value), None


def _extract_categories_from_value(categories_value: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract primary and alternate categories from a categories value."""
    try:
        if not categories_value:
            return None, None
            
        # Handle string JSON
        if isinstance(categories_value, str):
            categories_value = json.loads(categories_value)
        
        # Handle dict
        if isinstance(categories_value, dict):
            primary = categories_value.get("primary")
            alternate_list = categories_value.get("alternate", [])
            alternate = alternate_list[0] if isinstance(alternate_list, list) and alternate_list else None
            
            return _safe_string(primary), _safe_string(alternate)
            
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logging.debug(f"Failed to parse categories value: {e}")
    
    return _safe_string(categories_value), None


def _extract_address_from_value(addresses_value: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract formatted address components from an addresses value."""
    try:
        if not addresses_value:
            return None, None, None
            
        # Handle string JSON
        if isinstance(addresses_value, str):
            addresses_value = json.loads(addresses_value)
        
        # Handle array - take first address
        if isinstance(addresses_value, list) and addresses_value:
            addr = addresses_value[0]
            if isinstance(addr, dict):
                full = addr.get("freeform")
                locality = addr.get("locality")  
                country = addr.get("country")
                
                return _safe_string(full), _safe_string(locality), _safe_string(country)
            
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logging.debug(f"Failed to parse addresses value: {e}")
    
    return None, None, None


def _extract_first_from_array(array_value: Any) -> Optional[str]:
    """Extract first item from an array field (websites, emails, phones)."""
    try:
        if not array_value:
            return None
            
        # Handle string JSON
        if isinstance(array_value, str):
            array_value = json.loads(array_value)
        
        # Handle array
        if isinstance(array_value, list) and array_value:
            return _safe_string(array_value[0])
            
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logging.debug(f"Failed to parse array value: {e}")
    
    return _safe_string(array_value)


def _safe_string(value: Any, max_length: int = 255) -> Optional[str]:
    """Safely convert a value to a string with length limit."""
    if value is None:
        return None
    
    try:
        str_value = str(value)
        return str_value[:max_length] if len(str_value) > max_length else str_value
    except Exception:
        return None


def _safe_numeric_convert(series: pd.Series, target_type: type) -> pd.Series:
    """Safely convert a series to numeric type, handling nulls and invalid values."""
    def convert_value(x: Any) -> Any:
        if x is None or pd.isna(x):
            return None
        try:
            if target_type is int:
                return int(float(x))  # Convert to float first to handle string numbers
            else:
                return target_type(x)
        except (ValueError, TypeError):
            return None
    
    return series.apply(convert_value)


def _finalize_data_types(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Final cleanup of data types to ensure AGOL compatibility.
    Converts all object columns to strings and handles nulls properly.
    """
    result_gdf = gdf.copy()
    
    # Ensure id is string (for AGOL upsert)
    if "id" in result_gdf.columns:
        result_gdf["id"] = result_gdf["id"].astype(str)
    
    # Convert all object columns to strings (except geometry)
    for col in result_gdf.columns:
        if col == "geometry":
            continue
        
        if result_gdf[col].dtype == 'object':
            # Convert to string, handling nulls
            result_gdf[col] = result_gdf[col].astype(str)
            result_gdf[col] = result_gdf[col].replace('None', None)
            result_gdf[col] = result_gdf[col].replace('nan', None)
    
    return result_gdf