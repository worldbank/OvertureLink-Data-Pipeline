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
        mode: Publication mode - 'auto' (smart detection), 'initial', 'overwrite', or 'append'
        
    Returns:
        Published or updated feature layer item
        
    Raises:
        RuntimeError: If operation fails or configuration is invalid
        ValueError: If mode is not supported
        ConfigurationError: If credentials are missing or invalid
    """
    if gdf is None or gdf.empty:
        raise RuntimeError("No features to publish - GeoDataFrame is empty")

    # Handle smart auto-detection mode
    if mode == "auto":
        gis = _login_gis_for_publish()
        title = tgt.agol.item_title
        existing_service = _find_existing_service(gis, title)
        if existing_service:
            logging.info(f"Smart detection: Found existing service '{title}' - using overwrite mode")
            mode = "overwrite"
            # Set the item_id for overwrite mode
            tgt.agol.item_id = existing_service.itemid
        else:
            logging.info(f"Smart detection: No existing service '{title}' found - using initial mode")
            mode = "initial"

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
        
        # Update metadata before modifying data
        metadata_updates = _build_metadata_dict(tgt)
        if metadata_updates:
            logging.info(f"Updating service metadata: {list(metadata_updates.keys())}")
            fl_item.update(item_properties=metadata_updates)
            logging.debug("Service metadata updated successfully")
        
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
                # Ensure spatial accessor is properly initialized
                gdf = gdf.spatial.set_geometry('geometry')
                feature_set = gdf.spatial.to_featureset()
                
                if mode == "append":
                    edit_result = feature_layer.edit_features(adds=feature_set.features)
                else:  # overwrite - data already cleared above
                    edit_result = feature_layer.edit_features(adds=feature_set.features)
                
                if not edit_result.get('addResults', [{}])[0].get('success', False):
                    raise RuntimeError(f"Failed to update features: {edit_result}")
                
                logging.info("Direct feature update completed")
                
            else:
                # Standard data item update approach (file-based overwrite)
                data_item = related_items[0]
                logging.info(f"Found source data item: {data_item.title} ({data_item.type})")
                
                # File-based overwrite using feature layer collection manager
                # This approach is more reliable than feature set updates for mixed geometries
                try:
                    from arcgis.features import FeatureLayerCollection
                    feature_layer_collection = FeatureLayerCollection.fromitem(fl_item)
                    logging.info("Using feature layer collection file-based overwrite...")
                    overwrite_result = feature_layer_collection.manager.overwrite(tmp.name)
                    
                    if overwrite_result:
                        logging.info("File-based data replacement completed successfully")
                    else:
                        raise RuntimeError("File-based overwrite returned False")
                        
                except Exception as flc_error:
                    # Fallback to data item update approach
                    logging.warning(f"Feature layer collection overwrite failed: {flc_error}")
                    logging.info("Falling back to data item update approach...")
                    
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


def publish_multi_layer_service(layer_data: Dict[str, gpd.GeoDataFrame], tgt, mode: str = "auto"):
    """
    Publish or update multi-layer feature service for combined Overture data.
    
    Handles both places (points) and buildings (polygons) in a single service,
    with intelligent mode detection and proper geometry type handling.
    
    Args:
        layer_data: Dictionary mapping layer names to GeoDataFrames
        tgt: Target configuration with AGOL settings
        mode: Publishing mode - 'auto', 'initial', or 'overwrite'
        
    Returns:
        Published or updated feature service item
    """
    if not layer_data or all(gdf.empty for gdf in layer_data.values()):
        raise RuntimeError("No features to publish - all GeoDataFrames are empty")
    
    # Prepare each layer's data
    prepared_layers = {}
    for layer_name, gdf in layer_data.items():
        if not gdf.empty:
            prepared_layers[layer_name] = _prepare_gdf_for_agol(gdf, layer_name)
    
    # Get title from configuration
    title = tgt.agol.item_title
    tags = tgt.agol.tags or []
    
    # Count features by type
    total_features = sum(len(gdf) for gdf in prepared_layers.values())
    source_type_counts = {}
    for name, gdf in prepared_layers.items():
        if 'source_type' in gdf.columns:
            source_type_counts[name] = gdf['source_type'].iloc[0] if len(gdf) > 0 else name
        else:
            source_type_counts[name] = name
    
    logging.info(f"Publishing multi-layer service: {total_features:,} total features")
    for name, gdf in prepared_layers.items():
        logging.info(f"  - {name}: {len(gdf):,} features")
    
    # Combine all data for unified processing
    combined_gdfs = list(prepared_layers.values())
    combined_gdf = pd.concat(combined_gdfs, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs='EPSG:4326')
    
    # Prepare combined GeoDataFrame
    combined_gdf = _prepare_gdf_for_agol(combined_gdf, "combined")
    
    # Log geometry type distribution
    geom_types = combined_gdf.geometry.geom_type.value_counts()
    logging.info(f"Combined geometry types: {geom_types.to_dict()}")
    
    # Use secure GIS connection
    gis = _login_gis_for_publish()
    
    # Determine actual mode
    if mode == "auto":
        # Check if service already exists
        existing_service = _find_existing_service(gis, title)
        if existing_service:
            logging.info(f"Found existing service '{title}' (ID: {existing_service.itemid})")
            return _update_existing_service(existing_service, combined_gdf, prepared_layers, gis, tgt)
        else:
            logging.info(f"No existing service found for '{title}' - creating new...")
    
    elif mode == "initial":
        logging.info(f"Creating new multi-layer service: {title}")
    
    elif mode == "overwrite":
        existing_service = _find_existing_service(gis, title)
        if not existing_service:
            raise RuntimeError(f"Cannot overwrite - service '{title}' not found")
        return _update_existing_service(existing_service, combined_gdf, prepared_layers)
    
    # Create new service
    tmp = _gdf_to_geojson_tempfile(combined_gdf)
    
    try:
        # Create feature collection item
        item_properties = {
            "type": "GeoJson",
            "title": title,
            "tags": ",".join(tags),
            "description": f"Overture Maps data with {total_features:,} features",
            "snippet": f"Multi-layer dataset - {', '.join([f'{k}: {len(v)}' for k, v in prepared_layers.items()])}"
        }
        
        item = gis.content.add(item_properties, data=tmp.name)
        
        # Publish as feature service
        publish_params = {
            'name': title.replace(' ', '_').replace('-', '_'),
            'hasStaticData': True,
            'maxRecordCount': 2000,
            'layerInfo': {
                'capabilities': 'Query,Extract'
            }
        }
        
        fl_item = item.publish(publish_parameters=publish_params)
        
        logging.info(f"Created multi-layer service '{fl_item.title}' with ID: {fl_item.itemid}")
        logging.info(f"Service URL: {fl_item.homepage}")
        
        return fl_item
        
    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _find_existing_service(gis: GIS, title: str):
    """
    Search for existing feature service by title and service name patterns.
    
    AGOL converts titles to URL-safe service names, so we need to check both
    the exact title and potential service name variations to avoid conflicts.
    
    Args:
        gis: Authenticated GIS connection
        title: Exact title to search for
        
    Returns:
        Feature service item if found, None otherwise
    """
    try:
        # Generate potential service name (AGOL URL-safe conversion)
        service_name = title.replace(' ', '_').replace('-', '_').replace('.', '_')
        # Remove multiple underscores and clean up
        import re
        service_name = re.sub(r'_+', '_', service_name).strip('_')
        
        # Search 1: Exact title match
        title_query = f'title:"{title}" AND type:"Feature Service"'
        title_results = gis.content.search(
            query=title_query,
            item_type="Feature Service", 
            max_items=10
        )
        
        # Search 2: Service name pattern match (broader search)
        name_query = f'name:*{service_name}* AND type:"Feature Service"'
        name_results = gis.content.search(
            query=name_query,
            item_type="Feature Service",
            max_items=20
        )
        
        # Combine and deduplicate results
        all_results = title_results + [item for item in name_results if item not in title_results]
        
        # Filter for exact title matches first
        exact_title_matches = [item for item in all_results if item.title == title]
        if exact_title_matches:
            service = exact_title_matches[0] 
            logging.info(f"Found exact title match for '{title}': {service.itemid}")
            return service
        
        # Check for service name conflicts (could prevent creation)
        potential_conflicts = []
        for item in all_results:
            # Check if service name would conflict
            item_service_name = getattr(item, 'name', '') or getattr(item, 'url', '').split('/')[-1]
            if service_name.lower() in item_service_name.lower():
                potential_conflicts.append(item)
        
        if potential_conflicts:
            conflict_service = potential_conflicts[0]
            logging.info(f"Found potential service name conflict: '{conflict_service.title}' (ID: {conflict_service.itemid})")
            logging.info(f"Service name pattern '{service_name}' matches existing service")
            return conflict_service
        
        logging.info(f"No existing service found for '{title}' (checked title and service name patterns)")
        return None
            
    except Exception as e:
        logging.warning(f"Error searching for existing service '{title}': {e}")
        return None


def _update_existing_service(existing_service, combined_gdf, layer_data, gis: GIS, tgt=None):
    """
    Update existing multi-layer feature service with new data and metadata.
    
    Handles mixed geometry types (Points and Polygons) by updating
    each layer separately with the appropriate geometry type.
    
    Args:
        existing_service: Existing ArcGIS Online feature service item
        combined_gdf: Combined GeoDataFrame with all features
        layer_data: Dict mapping layer names to their GeoDataFrames
        gis: Authenticated GIS connection
        tgt: Target configuration for metadata updates (optional)
        
    Returns:
        Updated feature service item
    """
    try:
        logging.info(f"Updating existing service: {existing_service.title}")
        
        # Update metadata if configuration is provided
        if tgt:
            metadata_updates = _build_metadata_dict(tgt)
            if metadata_updates:
                logging.info(f"Updating service metadata: {list(metadata_updates.keys())}")
                existing_service.update(item_properties=metadata_updates)
                logging.debug("Service metadata updated successfully")
        
        # Get the feature layers from the service
        if not existing_service.layers:
            raise RuntimeError(f"No layers found in service {existing_service.itemid}")
        
        # For services with multiple geometry types, we need to update each layer separately
        # Group data by geometry type
        point_data = []
        polygon_data = []
        
        for layer_name, gdf in layer_data.items():
            if gdf.empty:
                continue
                
            # Ensure the geometry column is properly set for ArcGIS spatial accessor
            if 'geometry' not in gdf.columns:
                raise ValueError(f"No geometry column in {layer_name} data")
            
            # Explicitly set the geometry column for the spatial accessor
            gdf = gdf.set_geometry('geometry')
            
            # Ensure CRS is set
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')
            elif gdf.crs.to_string() != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            # Separate by geometry type
            geom_types = gdf.geometry.geom_type.unique()
            
            for geom_type in geom_types:
                if geom_type in ['Point', 'MultiPoint']:
                    point_subset = gdf[gdf.geometry.geom_type == geom_type].copy()
                    if not point_subset.empty:
                        point_data.append(point_subset)
                elif geom_type in ['Polygon', 'MultiPolygon']:
                    polygon_subset = gdf[gdf.geometry.geom_type == geom_type].copy()
                    if not polygon_subset.empty:
                        polygon_data.append(polygon_subset)
        
        # Combine data by geometry type
        if point_data:
            combined_points = pd.concat(point_data, ignore_index=True)
            combined_points = gpd.GeoDataFrame(combined_points, geometry='geometry', crs='EPSG:4326')
        else:
            combined_points = None
            
        if polygon_data:
            combined_polygons = pd.concat(polygon_data, ignore_index=True)
            combined_polygons = gpd.GeoDataFrame(combined_polygons, geometry='geometry', crs='EPSG:4326')
        else:
            combined_polygons = None
        
        # Update approach depends on service structure
        # Most Overture services have a single layer, but we'll handle both cases
        if len(existing_service.layers) == 1:
            # Single layer service - update with all data
            feature_layer = existing_service.layers[0]
            
            # Check what geometry type the layer expects
            layer_info = feature_layer.properties
            layer_geom_type = layer_info.get('geometryType', '').lower()
            
            # Clear existing data
            logging.info("Clearing existing data...")
            truncate_result = feature_layer.manager.truncate()
            if not truncate_result.get('success', False):
                raise RuntimeError(f"Failed to clear existing data: {truncate_result}")
            logging.info("Existing data cleared")
            
            # Determine which data to use based on layer geometry type
            if 'point' in layer_geom_type and combined_points is not None:
                data_to_upload = combined_points
            elif 'polygon' in layer_geom_type and combined_polygons is not None:
                data_to_upload = combined_polygons
            else:
                # Mixed or unknown - try to use all data
                data_to_upload = combined_gdf
                # Ensure geometry column is set
                data_to_upload = data_to_upload.set_geometry('geometry')
            
            logging.info(f"Uploading {len(data_to_upload):,} features...")
            
            # Method 1: Try using append with correct ArcGIS API pattern
            try:
                _append_via_item(feature_layer, data_to_upload, gis)
                logging.info(f"Data append completed successfully")
            except Exception as append_error:
                logging.warning(f"Append via item failed: {append_error}")
                
                # Fallback to edit_features method
                logging.info("Falling back to direct feature edit method...")
                data_to_upload = data_to_upload.set_geometry('geometry')
                feature_set = data_to_upload.spatial.to_featureset()
                
                edit_result = feature_layer.edit_features(adds=feature_set.features)
                if not edit_result.get('addResults', [{}])[0].get('success', False):
                    raise RuntimeError(f"Failed to update features: {edit_result}")
                    
                logging.info("Fallback edit_features completed successfully")
                    
        else:
            # Multi-layer service - update each layer separately
            for layer in existing_service.layers:
                layer_info = layer.properties
                layer_name = layer_info.get('name', '')
                layer_geom_type = layer_info.get('geometryType', '').lower()
                
                # Determine which data matches this layer
                if 'point' in layer_geom_type and combined_points is not None:
                    data_to_upload = combined_points
                elif 'polygon' in layer_geom_type and combined_polygons is not None:
                    data_to_upload = combined_polygons
                else:
                    logging.warning(f"No matching data for layer {layer_name} with geometry type {layer_geom_type}")
                    continue
                
                # Clear and update this layer
                logging.info(f"Updating layer {layer_name} with {len(data_to_upload):,} features...")
                
                # Clear existing data before adding new data
                logging.info(f"Clearing existing data from layer {layer_name}...")
                truncate_result = layer.manager.truncate()
                if not truncate_result.get('success', False):
                    logging.warning(f"Failed to clear layer {layer_name}: {truncate_result}")
                else:
                    logging.info(f"Existing data cleared from layer {layer_name}")
                
                # Upload new data using correct ArcGIS API pattern
                try:
                    _append_via_item(layer, data_to_upload, gis)
                    logging.info(f"Layer {layer_name} updated successfully")
                except Exception as append_error:
                    logging.warning(f"Append via item failed for layer {layer_name}: {append_error}")
                    
                    # Fallback to edit_features method
                    logging.info(f"Falling back to direct feature edit for layer {layer_name}...")
                    data_to_upload = data_to_upload.set_geometry('geometry')
                    feature_set = data_to_upload.spatial.to_featureset()
                    
                    edit_result = layer.edit_features(adds=feature_set.features)
                    if not edit_result.get('addResults', [{}])[0].get('success', False):
                        logging.warning(f"Failed to update layer {layer_name}: {edit_result}")
                    else:
                        logging.info(f"Layer {layer_name} fallback update completed successfully")
        
        # Verify final counts
        total_features = 0
        for layer in existing_service.layers:
            count = layer.query(return_count_only=True)
            total_features += count
            logging.info(f"Layer {layer.properties.get('name', 'unnamed')}: {count:,} features")
        
        logging.info(f"Service update completed. Total features: {total_features:,}")
        logging.info(f"Service URL: {existing_service.homepage}")
        
        return existing_service
        
    except Exception as e:
        logging.error(f"Failed to update existing service: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        
        # Log additional debugging information
        if 'data_to_upload' in locals():
            logging.error(f"Data shape: {data_to_upload.shape}")
            logging.error(f"Columns: {list(data_to_upload.columns)}")
            if 'geometry' in data_to_upload.columns:
                logging.error(f"Geometry types: {data_to_upload.geometry.geom_type.value_counts().to_dict()}")
        
        raise RuntimeError(f"Service update failed: {e}")


def _prepare_gdf_for_agol(gdf: gpd.GeoDataFrame, layer_name: str = "data") -> gpd.GeoDataFrame:
    """
    Prepare GeoDataFrame for ArcGIS Online publication.
    
    Ensures geometry column is properly configured, CRS is set,
    and data types are compatible with ArcGIS.
    
    Args:
        gdf: Input GeoDataFrame
        layer_name: Name for logging purposes
        
    Returns:
        Prepared GeoDataFrame ready for ArcGIS operations
    """
    # Make a copy to avoid modifying original
    prepared_gdf = gdf.copy()
    
    # Ensure geometry column exists and is properly set
    if 'geometry' not in prepared_gdf.columns:
        raise ValueError(f"No geometry column in {layer_name}")
    
    # Explicitly set geometry column for spatial accessor
    prepared_gdf = prepared_gdf.set_geometry('geometry')
    
    # Ensure CRS is WGS84
    if prepared_gdf.crs is None:
        logging.debug(f"Setting CRS to EPSG:4326 for {layer_name}")
        prepared_gdf = prepared_gdf.set_crs('EPSG:4326')
    elif prepared_gdf.crs.to_string() != 'EPSG:4326':
        logging.debug(f"Reprojecting {layer_name} from {prepared_gdf.crs} to EPSG:4326")
        prepared_gdf = prepared_gdf.to_crs('EPSG:4326')
    
    # Remove any invalid geometries
    valid_mask = prepared_gdf.geometry.is_valid
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logging.warning(f"Removing {invalid_count} invalid geometries from {layer_name}")
        prepared_gdf = prepared_gdf[valid_mask].copy()
    
    # Remove any null geometries
    null_mask = prepared_gdf.geometry.isna()
    null_count = null_mask.sum()
    if null_count > 0:
        logging.warning(f"Removing {null_count} null geometries from {layer_name}")
        prepared_gdf = prepared_gdf[~null_mask].copy()
    
    # Ensure all object columns are strings (ArcGIS doesn't handle mixed types well)
    for col in prepared_gdf.columns:
        if prepared_gdf[col].dtype == 'object' and col != 'geometry':
            # Convert to string, handling None values
            prepared_gdf[col] = prepared_gdf[col].apply(
                lambda x: str(x) if x is not None else None
            )
    
    # Log geometry type distribution
    geom_types = prepared_gdf.geometry.geom_type.value_counts()
    logging.debug(f"{layer_name} geometry types: {geom_types.to_dict()}")
    
    return prepared_gdf


def _append_via_item(feature_layer, gdf: gpd.GeoDataFrame, gis: GIS) -> bool:
    """
    Append data to feature layer using correct ArcGIS API pattern.
    
    Creates temporary Portal item, analyzes it, appends to target layer,
    then cleans up the temporary item.
    
    Args:
        feature_layer: Target feature layer to append to
        gdf: GeoDataFrame to append
        gis: Authenticated GIS connection
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        RuntimeError: If append operation fails
    """
    import time
    
    # Create temporary GeoJSON file
    tmp = _gdf_to_geojson_tempfile(gdf)
    temp_item = None
    
    try:
        # Create unique temporary item name
        timestamp = int(time.time())
        temp_title = f"temp_append_data_{timestamp}"
        
        # Upload as temporary Portal item
        item_properties = {
            "type": "GeoJson",
            "title": temp_title,
            "tags": "temporary,overture-pipeline"
        }
        
        logging.debug(f"Creating temporary item: {temp_title}")
        temp_item = gis.content.add(item_properties, data=tmp.name)
        
        if not temp_item:
            raise RuntimeError("Failed to create temporary item for append operation")
        
        # Analyze the uploaded item
        logging.debug("Analyzing temporary item for publishing parameters")
        analyzed = gis.content.analyze(item=temp_item.id, file_type='geojson')
        
        # Perform append operation with correct parameters
        append_result = feature_layer.append(
            item_id=temp_item.id,
            upload_format='geojson',
            source_info=analyzed.get('publishParameters', {}),
            upsert=False,
            skip_updates=False,
            use_globalids=False,
            update_geometry=True,
            rollback=True,
            skip_inserts=False,
            return_messages=True
        )
        
        if not append_result:
            raise RuntimeError("Append operation returned no result")
            
        logging.debug("Append operation completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to append data via item: {e}")
        raise RuntimeError(f"Append operation failed: {e}")
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
            
        # Clean up temporary Portal item
        if temp_item:
            try:
                temp_item.delete()
                logging.debug(f"Cleaned up temporary item: {temp_item.title}")
            except Exception as e:
                logging.warning(f"Failed to clean up temporary item {temp_item.title}: {e}")


def _build_metadata_dict(tgt):
    """
    Build metadata dictionary from target configuration for item updates.
    
    Extracts available metadata fields and filters out None/empty values
    to avoid overwriting existing metadata with blanks.
    
    Args:
        tgt: Target configuration object with agol attributes
        
    Returns:
        Dictionary of metadata properties for item.update()
    """
    metadata = {}
    
    # Core metadata fields
    if hasattr(tgt.agol, 'snippet') and tgt.agol.snippet:
        metadata['snippet'] = tgt.agol.snippet
    
    if hasattr(tgt.agol, 'description') and tgt.agol.description:
        metadata['description'] = tgt.agol.description
    
    if hasattr(tgt.agol, 'tags') and tgt.agol.tags:
        # Handle both list and string formats
        if isinstance(tgt.agol.tags, list):
            metadata['tags'] = ','.join(tgt.agol.tags)
        else:
            metadata['tags'] = tgt.agol.tags
    
    # Enterprise metadata fields
    if hasattr(tgt.agol, 'license_info') and tgt.agol.license_info:
        metadata['licenseInfo'] = tgt.agol.license_info
    
    if hasattr(tgt.agol, 'access_information') and tgt.agol.access_information:
        metadata['accessInformation'] = tgt.agol.access_information
    
    return metadata


