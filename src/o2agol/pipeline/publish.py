"""
FeatureLayerManager - ArcGIS Online Publishing

Handles AGOL-specific operations including feature layer creation, updates, and management.
This module focuses solely on AGOL operations, with file I/O moved to export.py.
"""

import logging
import time
from typing import Any, Optional, Dict, Union

import geopandas as gpd
from arcgis.gis import GIS

from ..domain.enums import Mode


class FeatureLayerManager:
    """
    Manages ArcGIS Online feature layer operations.
    
    Handles creation, overwrite, append, and truncate-append operations
    based on the specified mode and existing layer detection.
    """
    
    def __init__(self, gis: GIS, mode: Mode = Mode.AUTO, item_ids: Optional[dict[str, str]] = None):
        """
        Initialize feature layer manager.
        
        Args:
            gis: ArcGIS GIS connection object
            mode: Publishing mode (auto, initial, overwrite, append)
            item_ids: Optional mapping of layer names to existing item IDs
        """
        self.gis = gis
        self.mode = mode
        self.item_ids = item_ids or {}
        
    def login_gis(self) -> GIS:
        """Get or validate GIS connection."""
        if not self.gis:
            # Import here to avoid circular imports
            from ..config.settings import Config
            
            config = Config()
            self.gis = config.create_gis_connection()
            
        return self.gis
        
    def find_existing_service(self, title: str, item_id: Optional[str] = None) -> Optional[Any]:
        """
        Find existing feature service by title or item ID.
        
        Args:
            title: Service title to search for
            item_id: Optional item ID for direct lookup
            
        Returns:
            Existing service item or None if not found
        """
        try:
            if item_id:
                # Direct lookup by item ID
                item = self.gis.content.get(item_id)
                if item and hasattr(item, 'layers'):
                    logging.info(f"Found existing service by item_id: {item_id}")
                    return item
                    
            # Search by title
            search_results = self.gis.content.search(
                query=f'title:"{title}" AND type:"Feature Service"',
                item_type="Feature Service",
                max_items=10
            )
            
            for item in search_results:
                if item.title == title:
                    logging.info(f"Found existing service by title: {title}")
                    return item
                    
            logging.info(f"No existing service found for title: {title}")
            return None
            
        except Exception as e:
            logging.warning(f"Error searching for existing service: {e}")
            return None
            
    def create_feature_service(
        self, 
        df: gpd.GeoDataFrame, 
        service_name: str, 
        metadata: Dict[str, Any],
        staging_data: Any = None
    ) -> str:
        """
        Create a new feature service from GeoDataFrame.
        
        Args:
            df: GeoDataFrame to publish
            service_name: Name for the service
            metadata: Service metadata (title, description, tags, etc.)
            staging_data: Prepared staging data (tempfile or similar)
            
        Returns:
            Item ID of the created service
        """
        try:
            # Use staging data if provided, otherwise create staging data from df
            if staging_data:
                # Publish from staging file
                service_item = staging_data.publish()
            else:
                # Create staging data from GeoDataFrame and publish
                from ..pipeline.export import Exporter
                from ..domain.enums import StagingFormat
                
                # Use GeoJSON as default staging format for fallback
                staging_fmt = StagingFormat.GEOJSON.value
                
                exporter = Exporter(out_path=None, fmt=None)
                temp_file = exporter.create_staging_file(df, service_name, staging_fmt)
                
                # Map staging format to ArcGIS item type
                if staging_fmt == "geojson":
                    item_type = "GeoJson"
                elif staging_fmt == "gpkg":
                    item_type = "GeoPackage" 
                else:
                    item_type = "GeoJson"  # Default fallback
                
                # Upload and publish the staging file
                uploaded_item = self.gis.content.add({
                    'title': f"{service_name}_temp",
                    'type': item_type,
                    'tags': 'temp, staging'
                }, data=temp_file.name)
                
                service_item = uploaded_item.publish()
                
                # Clean up the temporary uploaded item after publishing
                try:
                    uploaded_item.delete()
                    logging.debug(f"Cleaned up temp upload item: {uploaded_item.id}")
                except Exception as e:
                    logging.warning(f"Could not clean up temp upload item {uploaded_item.id}: {e}")
                
            # Update metadata
            if service_item:
                service_item.update(
                    item_properties={
                        'title': metadata.get('title', service_name),
                        'snippet': metadata.get('snippet', ''),
                        'description': metadata.get('description', ''),
                        'tags': metadata.get('tags', []),
                        'accessInformation': metadata.get('access_information', ''),
                        'licenseInfo': metadata.get('license_info', '')
                    }
                )
                
                logging.info(f"Created new feature service: {service_item.id}")
                
                # Clean up any other orphaned items for this service
                try:
                    self.cleanup_orphaned_items(service_name, metadata.get('title'))
                except Exception as e:
                    logging.warning(f"Could not clean up orphaned items: {e}")
                
                return service_item.id
                
            raise Exception("Failed to create service item")
            
        except Exception as e:
            logging.error(f"Failed to create feature service: {e}")
            raise
            
    def update_existing_service(
        self, 
        service_item: Any,
        df: gpd.GeoDataFrame,
        metadata: Dict[str, Any],
        staging_data: Any = None
    ) -> str:
        """
        Update an existing feature service.
        
        Args:
            service_item: Existing service item
            df: New GeoDataFrame data
            metadata: Updated metadata
            staging_data: Prepared staging data
            
        Returns:
            Item ID of the updated service
        """
        try:
            if self.mode == Mode.OVERWRITE:
                # Overwrite the service with new data
                if staging_data:
                    service_item.overwrite(staging_data)
                else:
                    # Direct overwrite from GeoDataFrame
                    service_item.overwrite(df.spatial.to_featureclass())
                    
            elif self.mode == Mode.APPEND:
                # Append data to existing service
                if hasattr(service_item, 'layers') and service_item.layers:
                    layer = service_item.layers[0]
                    if staging_data:
                        append_result = layer.append(staging_data)
                    else:
                        # Convert df to features and append
                        features = df.spatial.to_feature_set()
                        append_result = layer.edit_features(adds=features.features)
                        
                    logging.info(f"Appended {len(df)} features to existing service")
                else:
                    raise Exception("Service has no layers to append to")
                    
            # Update metadata
            service_item.update(
                item_properties={
                    'title': metadata.get('title', service_item.title),
                    'snippet': metadata.get('snippet', service_item.snippet or ''),
                    'description': metadata.get('description', service_item.description or ''),
                    'tags': metadata.get('tags', service_item.tags or []),
                    'accessInformation': metadata.get('access_information', service_item.accessInformation or ''),
                    'licenseInfo': metadata.get('license_info', service_item.licenseInfo or '')
                }
            )
            
            logging.info(f"Updated existing service: {service_item.id}")
            return service_item.id
            
        except Exception as e:
            logging.error(f"Failed to update existing service: {e}")
            raise
            
    def upsert(
        self, 
        df: gpd.GeoDataFrame, 
        layer_name: str, 
        metadata: Dict[str, Any],
        staging_data: Any = None
    ) -> str:
        """
        Create or update feature layer based on mode.
        
        Args:
            df: GeoDataFrame to publish
            layer_name: Name for the feature layer
            metadata: Layer metadata including title, description, tags
            staging_data: Optional prepared staging data
            
        Returns:
            Item ID of the created/updated feature layer
            
        Mode behavior:
            - auto: detect-by-name -> create or truncate-append
            - initial: force creation of new layer
            - overwrite: force update of existing layer
            - append: add data to existing layer
        """
        try:
            # Ensure we have a valid GIS connection
            gis = self.login_gis()
            
            title = metadata.get('title', layer_name)
            item_id = self.item_ids.get(layer_name)
            
            # Find existing service if in auto mode or if we have an item_id
            existing_service = None
            if self.mode == Mode.AUTO or self.mode in [Mode.OVERWRITE, Mode.APPEND]:
                existing_service = self.find_existing_service(title, item_id)
                
            # Decide what to do based on mode and existing service
            if existing_service and self.mode != Mode.INITIAL:
                # Update existing service
                return self.update_existing_service(existing_service, df, metadata, staging_data)
            else:
                # Create new service
                return self.create_feature_service(df, layer_name, metadata, staging_data)
                
        except Exception as e:
            logging.error(f"Failed to upsert feature layer '{layer_name}': {e}")
            raise
            
    def publish_multi_layer_service(
        self,
        layer_data: Dict[str, gpd.GeoDataFrame],
        service_name: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Publish a multi-layer service (e.g., education = places + buildings).
        
        Args:
            layer_data: Dict mapping layer names to GeoDataFrames
            service_name: Base name for the service
            metadata: Service metadata
            
        Returns:
            Item ID of the created/updated multi-layer service
        """
        try:
            # For now, create separate services for each layer
            # TODO: Implement true multi-layer service creation
            created_ids = []
            
            for layer_name, gdf in layer_data.items():
                layer_metadata = metadata.copy()
                layer_metadata['title'] = f"{metadata.get('title', service_name)} - {layer_name.title()}"
                
                item_id = self.upsert(gdf, f"{service_name}_{layer_name}", layer_metadata)
                created_ids.append(item_id)
                
            # Return the first item ID for now
            # TODO: Return a proper multi-layer service ID
            return created_ids[0] if created_ids else ""
            
        except Exception as e:
            logging.error(f"Failed to publish multi-layer service '{service_name}': {e}")
            raise
            
    def cleanup_orphaned_items(self, service_name: str, title: Optional[str] = None) -> int:
        """
        Clean up orphaned items related to a service.
        
        Args:
            service_name: Service name to search for
            title: Optional title filter
            
        Returns:
            Number of items cleaned up
        """
        try:
            # Search for related items (uploads, etc.)
            search_query = f'title:"{service_name}" OR title:"{title}"' if title else f'title:"{service_name}"'
            items = self.gis.content.search(query=search_query, max_items=50)
            
            cleaned_count = 0
            for item in items:
                # Only clean up temporary/upload items, not published services
                if item.type in ['File Geodatabase', 'GeoJSON', 'Shapefile'] and 'temp' in item.title.lower():
                    try:
                        item.delete()
                        cleaned_count += 1
                        logging.debug(f"Cleaned up orphaned item: {item.id}")
                    except Exception as e:
                        logging.warning(f"Failed to clean up item {item.id}: {e}")
                        
            if cleaned_count > 0:
                logging.info(f"Cleaned up {cleaned_count} orphaned items")
                
            return cleaned_count
            
        except Exception as e:
            logging.warning(f"Error during cleanup: {e}")
            return 0