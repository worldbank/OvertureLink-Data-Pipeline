from __future__ import annotations

from typing import Literal

import geopandas as gpd


def normalize_schema(
    gdf: gpd.GeoDataFrame, layer: Literal["roads", "buildings", "education", "health", "markets", "places"]
) -> gpd.GeoDataFrame:
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Handle different layer types with appropriate field selection
    if layer in ["roads"]:
        # Transportation data - use original simple approach that worked. "names", "routes" had issues. To investigate.
        keep = [c for c in ["id", "class", "subtype"] if c in gdf.columns]
        
    elif layer in ["buildings"]:
        # Building data  
        keep = [c for c in ["id", "names", "class", "subtype", "height", "num_floors"] if c in gdf.columns]
        
    elif layer in ["education", "health", "markets", "places"]:
        # Mixed places and buildings data - handle unified schema
        keep = ["id"]
        
        # Extract primary name from names JSON - vectorized for GeoPandas 1.0+
        if "names" in gdf.columns:
            # Use vectorized string operations where possible
            gdf["name"] = gdf["names"].apply(extract_primary_name)
            keep.append("name")
        
        # Handle source_type field (distinguishes places vs buildings)
        if "source_type" in gdf.columns:
            keep.append("source_type")
        
        # Create unified type_category field from places categories or building class
        # Optimized for GeoPandas 1.0+ with vectorized operations
        if "categories" in gdf.columns or "class" in gdf.columns:
            # Initialize with None
            gdf["type_category"] = None
            
            # Vectorized assignment for places
            if "categories" in gdf.columns and "source_type" in gdf.columns:
                place_mask = gdf["source_type"] == "place"
                gdf.loc[place_mask, "type_category"] = gdf.loc[place_mask, "categories"].apply(extract_primary_category)
            
            # Vectorized assignment for buildings
            if "class" in gdf.columns and "source_type" in gdf.columns:
                building_mask = gdf["source_type"] == "building"
                gdf.loc[building_mask, "type_category"] = gdf.loc[building_mask, "class"]
            
            keep.append("type_category")
        
        # Keep confidence if available (places only)
        if "confidence" in gdf.columns:
            keep.append("confidence")
        
        # Extract contact information from complex fields (places only)
        if "websites" in gdf.columns:
            gdf["website"] = gdf["websites"].apply(extract_first_website)
            keep.append("website")
            
        if "addresses" in gdf.columns:
            gdf["address"] = gdf["addresses"].apply(extract_formatted_address)
            keep.append("address")
        
        # Building-specific fields
        if "height" in gdf.columns:
            keep.append("height")
        if "num_floors" in gdf.columns:
            keep.append("num_floors")
        if "subtype" in gdf.columns:
            keep.append("subtype")
    
    else:
        # Default fallback
        keep = [c for c in ["id", "class", "subtype"] if c in gdf.columns]
    
    # Ensure geometry is included
    cols = keep + ["geometry"]
    gdf = gdf[[c for c in cols if c in gdf.columns]]

    # Ensure id is string (for AGOL upsert)
    if "id" in gdf.columns:
        gdf["id"] = gdf["id"].astype(str)

    return gdf


def extract_primary_name(names_json) -> str:
    """Extract primary name from Overture names JSON structure"""
    if not names_json or names_json == '{}':
        return None
    
    try:
        if isinstance(names_json, dict):
            # Direct dict access
            return names_json.get('primary', names_json.get('common', list(names_json.values())[0] if names_json else None))
        elif isinstance(names_json, str):
            import json
            names = json.loads(names_json)
            return names.get('primary', names.get('common', list(names.values())[0] if names else None))
    except:
        pass
    return str(names_json)[:100] if names_json else None


def extract_primary_category(categories_json) -> str:
    """Extract primary category from Overture categories JSON structure"""
    if not categories_json or categories_json == '{}':
        return None
        
    try:
        if isinstance(categories_json, dict):
            return categories_json.get('primary')
        elif isinstance(categories_json, str):
            import json
            categories = json.loads(categories_json)
            return categories.get('primary')
    except:
        pass
    return str(categories_json)[:50] if categories_json else None




def extract_first_website(websites_json) -> str:
    """Extract first website from websites array"""
    if not websites_json:
        return None
        
    try:
        if isinstance(websites_json, list) and websites_json:
            return websites_json[0]
        elif isinstance(websites_json, str):
            import json
            websites = json.loads(websites_json)
            return websites[0] if websites and isinstance(websites, list) else None
    except:
        pass
    return None


def extract_formatted_address(addresses_json) -> str:
    """Extract formatted address string from addresses array"""
    if not addresses_json:
        return None
        
    try:
        if isinstance(addresses_json, list) and addresses_json:
            addr = addresses_json[0]
            if isinstance(addr, dict):
                # Build formatted address from components
                parts = []
                if addr.get('freeform'):
                    parts.append(addr['freeform'])
                if addr.get('locality'):
                    parts.append(addr['locality'])
                if addr.get('region'):
                    parts.append(addr['region'])
                if addr.get('country'):
                    parts.append(addr['country'])
                return ', '.join(parts) if parts else None
        elif isinstance(addresses_json, str):
            import json
            addresses = json.loads(addresses_json)
            if addresses and isinstance(addresses, list):
                addr = addresses[0]
                if isinstance(addr, dict):
                    parts = []
                    if addr.get('freeform'):
                        parts.append(addr['freeform'])
                    if addr.get('locality'):
                        parts.append(addr['locality'])
                    if addr.get('region'):
                        parts.append(addr['region'])
                    if addr.get('country'):
                        parts.append(addr['country'])
                    return ', '.join(parts) if parts else None
    except:
        pass
    return None
