from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Optional

import geopandas as gpd
import pandas as pd  # make sure this import exists at the top
from arcgis.gis import GIS

# Import secure configuration system
from o2agol.config.settings import Config, ConfigurationError


def _login_gis_for_publish() -> GIS:
    """
    Create secure ArcGIS Online connection for publishing operations.
    
    Uses the new secure configuration system with comprehensive validation
    and error handling. Publishing requires named-user credentials.
    
    Returns:
        Authenticated GIS connection with publishing privileges
        
    Raises:
        ConfigurationError: If credentials are missing or invalid
        RuntimeError: If connection fails or user lacks publishing permissions
    """
    try:
        # Use secure configuration system
        config = Config(validate_on_init=False)  # Skip full validation for performance
        gis = config.create_gis_connection()
        
        # Verify publishing permissions
        user = gis.users.me
        if user is None:
            raise RuntimeError("Authentication succeeded but user context is None — check account permissions.")
        
        # Check if user can create content
        if not user.privileges or 'portal:user:createItem' not in user.privileges:
            logging.warning(f"User {user.username} may not have content creation privileges")
        
        logging.debug(f"Connected for publishing as: {user.username} ({user.role})")
        return gis
        
    except ConfigurationError as e:
        # Convert configuration errors to more specific publishing errors
        if "AGOL_USERNAME" in str(e) or "AGOL_PASSWORD" in str(e):
            raise RuntimeError(
                "Publishing requires named user credentials (AGOL_USERNAME / AGOL_PASSWORD). "
                "App-only tokens cannot create items. Set named-user credentials in your .env file "
                "or use mode 'overwrite'/'append' with an existing item_id."
            ) from e
        else:
            raise RuntimeError(f"Configuration error: {e}") from e
    
    except Exception as e:
        raise RuntimeError(f"Failed to establish publishing connection: {e}") from e


def _gdf_to_geojson_tempfile(gdf: gpd.GeoDataFrame) -> tempfile.NamedTemporaryFile:
    """
    Convert GeoDataFrame to temporary GeoJSON file for ArcGIS Online upload.
    
    Args:
        gdf: GeoDataFrame to convert
        
    Returns:
        NamedTemporaryFile containing GeoJSON data
        
    Note:
        Caller is responsible for cleanup of temporary file
    """
    # We'll iterate rows without the geometry column
    attrs = gdf.drop(columns=["geometry"], errors="ignore")

    features = []
    for i, row in attrs.iterrows():
        # row has no geometry; build properties directly
        props = {k: (None if pd.isna(v) else v) for k, v in row.items()}

        # use the geometry from the GeoDataFrame
        geom = gdf.geometry.iat[i]
        geom_json = geom.__geo_interface__ if geom is not None else None

        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": geom_json,
        })

    fc = {"type": "FeatureCollection", "features": features}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False, encoding="utf-8")
    json.dump(fc, tmp)
    tmp.flush()
    return tmp


def publish_or_update(gdf: gpd.GeoDataFrame, tgt, mode: str = "initial"):
    """
    Publish or update geospatial data to ArcGIS Online using secure configuration.
    
    This function uses the World Bank compliant secure configuration system
    for credential management and provides comprehensive error handling.
    
    Args:
        gdf: GeoDataFrame containing features to publish
        tgt: Target configuration including AGOL settings
        mode: Publication mode - 'initial', 'overwrite', or 'append'
        
    Returns:
        Published or updated feature layer item
        
    Raises:
        RuntimeError: If operation fails or configuration is invalid
        ValueError: If mode is not supported
        ConfigurationError: If credentials are missing or invalid
    """
    if gdf is None or gdf.empty:
        raise RuntimeError("No features to publish - GeoDataFrame is empty")

    # Validate configuration against mode
    has_item_id = getattr(tgt.agol, 'item_id', None) is not None and getattr(tgt.agol, 'item_id', '').strip() != ""
    
    if mode == "initial" and has_item_id:
        logging.warning(
            f"Mode is 'initial' but item_id is configured. "
            f"This will create a new layer, not update item {tgt.agol.item_id}"
        )
    
    if mode in ("overwrite", "append") and not has_item_id:
        raise RuntimeError(
            f"Mode '{mode}' requires item_id in configuration. "
            f"Set targets.{tgt.agol.item_title}.agol.item_id or use mode 'initial'"
        )

    # Use secure GIS connection
    gis = _login_gis_for_publish()
    title = tgt.agol.item_title
    tags = tgt.agol.tags or []
    
    logging.info(f"Starting {mode} operation: {len(gdf):,} features")
    logging.info(f"Connected as: {gis.users.me.username} to {gis.properties.name}")

    # Create new hosted feature layer
    if mode == "initial":
        tmp = _gdf_to_geojson_tempfile(gdf)
        
        try:
            # Build comprehensive metadata dict for enterprise publishing
            item_properties = {
                "type": "GeoJson",
                "title": title,
                "tags": ",".join(tags),
            }
            
            # Add enhanced metadata fields if available
            if hasattr(tgt.agol, 'snippet') and tgt.agol.snippet:
                item_properties["snippet"] = tgt.agol.snippet
            else:
                item_properties["snippet"] = f"Generated from Overture Maps pipeline - {len(gdf):,} features"
            
            if hasattr(tgt.agol, 'description') and tgt.agol.description:
                item_properties["description"] = tgt.agol.description
            else:
                item_properties["description"] = f"Overture Maps data with {len(gdf):,} features"
            
            # Add ESRI enterprise metadata fields if available
            if hasattr(tgt.agol, 'access_information') and tgt.agol.access_information:
                item_properties["accessInformation"] = tgt.agol.access_information
            
            if hasattr(tgt.agol, 'license_info') and tgt.agol.license_info:
                item_properties["licenseInfo"] = tgt.agol.license_info
            
            # Note: Removed categories, type_keywords, and custom properties 
            # as they require admin permissions and add unnecessary complexity
            
            logging.info(f"Creating item with metadata: {len(item_properties)} fields")
            
            item = gis.content.add(item_properties, data=tmp.name)
            
            fl_item = item.publish()
            logging.info(f"Created layer '{fl_item.title}' with ID: {fl_item.itemid}")
            logging.info(f"Add to config - item_id: {fl_item.itemid}")
            logging.info(f"Access at: {fl_item.homepage}")
            
            return fl_item
            
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # Update existing hosted feature layer
    if mode in ("overwrite", "append"):
        item_id = getattr(tgt.agol, 'item_id', None)
        fl_item = gis.content.get(item_id)
        
        if not fl_item:
            raise RuntimeError(f"Item not found: {item_id}")
        if fl_item.type != "Feature Service":
            raise RuntimeError(f"Item {item_id} is not a Feature Service (found: {fl_item.type})")
        if not fl_item.layers:
            raise RuntimeError(f"No layers found in service {item_id}")
        
        feature_layer = fl_item.layers[0]
        existing_count = feature_layer.query(return_count_only=True)
        
        logging.info(f"Target layer '{fl_item.title}' has {existing_count:,} existing features")
        
        # Clear existing data for overwrite mode
        if mode == "overwrite":
            logging.info("Clearing existing data...")
            truncate_result = feature_layer.manager.truncate()
            if not truncate_result.get('success', False):
                raise RuntimeError(f"Failed to clear existing data: {truncate_result}")
            logging.info("Existing data cleared")
        
        # Update approach for monthly data replacement while preserving item_id
        # Based on Esri Community best practices for automated updates
        tmp = _gdf_to_geojson_tempfile(gdf)
        
        try:
            # Find the underlying data item for this feature service
            # Feature services are published from data items (CSV, GeoJSON, etc.)
            related_items = fl_item.related_items('Service2Data', 'forward')
            
            if not related_items:
                # Fallback: Try direct feature append/update
                logging.warning("No source data item found - using direct feature update")
                
                # Convert to feature set and append/update
                feature_set = gdf.spatial.to_featureset()
                
                if mode == "append":
                    edit_result = feature_layer.edit_features(adds=feature_set.features)
                else:  # overwrite - data already cleared above
                    edit_result = feature_layer.edit_features(adds=feature_set.features)
                
                if not edit_result.get('addResults', [{}])[0].get('success', False):
                    raise RuntimeError(f"Failed to update features: {edit_result}")
                
                logging.info("Direct feature update completed")
                
            else:
                # Standard data item update approach
                data_item = related_items[0]
                logging.info(f"Found source data item: {data_item.title} ({data_item.type})")
                
                # Update the source data item with new file
                logging.info("Updating source data item...")
                update_result = data_item.update(data=tmp.name)
                
                if not update_result:
                    raise RuntimeError("Failed to update source data item")
                
                # Republish the feature service from updated data item
                logging.info("Republishing feature service from updated data...")
                publish_result = data_item.publish(overwrite=True)
                
                if not publish_result:
                    raise RuntimeError("Failed to republish feature service")
                
                logging.info("Data replacement completed successfully")
            
            logging.info(f"Item ID preserved: {item_id}")
            
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        
        # Get final count
        final_count = feature_layer.query(return_count_only=True)
        logging.info(f"Operation complete: {final_count:,} total features in layer")
        logging.info(f"Updated layer: {fl_item.homepage}")
        
        return fl_item

    raise ValueError(f"Unsupported mode: {mode}. Use 'initial', 'overwrite', or 'append'")


