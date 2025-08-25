"""
FeatureLayerManager - ArcGIS Online Publishing

Handles AGOL-specific operations including feature layer creation, updates, and management.
This module focuses solely on AGOL operations, with file I/O moved to export.py.
"""

import contextlib
import logging
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
from arcgis.features import FeatureLayerCollection
from arcgis.gis import GIS

from ..domain.enums import Mode
from ..cleanup import get_pid_temp_dir


def remove_readonly(func, path, exc_info):
    """
    Error handler for shutil.rmtree to handle read-only files on Windows.
    
    This is especially needed for File Geodatabase (.gdb) folders which
    often contain files with read-only attributes that prevent deletion.
    """
    # Check if the error is due to a permissions issue
    if not os.access(path, os.W_OK):
        try:
            # Add write permissions and retry
            os.chmod(path, stat.S_IWUSR | stat.S_IWRITE)
            func(path)
        except Exception:
            # If that doesn't work, try removing all restrictions
            try:
                os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                func(path)
            except Exception:
                # Last resort - ignore if we still can't delete it
                pass
    else:
        # If it's not a permissions issue, re-raise the original exception
        raise exc_info[1]


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
        metadata: dict[str, Any],
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
                from ..domain.enums import StagingFormat
                from ..pipeline.export import Exporter
                
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
        metadata: dict[str, Any],
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
                        layer.append(staging_data)
                    else:
                        # Convert df to features and append
                        features = df.spatial.to_feature_set()
                        layer.edit_features(adds=features.features)
                        
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
        metadata: dict[str, Any],
        staging_data: Any = None
    ) -> str:
        """
        Create or update feature layer based on mode.
        
        DEPRECATED: Use publish_multi_layer_service() for 45x faster GeoPackage staging.
        
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
        import warnings
        warnings.warn(
            "upsert() is deprecated. Use publish_multi_layer_service() for 45x faster GeoPackage staging.",
            DeprecationWarning,
            stacklevel=2
        )
        try:
            # Ensure we have a valid GIS connection
            self.login_gis()
            
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
        layer_data: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame,
        service_name: str,
        metadata: dict[str, Any],
        mode: Mode = Mode.AUTO,
        staging_format: str = "gpkg",  # "gpkg" or "fgdb"
    ) -> str:
        """
        Publish a hosted feature layer with single or multiple sublayers by staging
        one container file (GeoPackage by default).
        layer_data: dict of {sublayer_name: GeoDataFrame} OR single GeoDataFrame
        service_name: base service name (used for container filename and default table prefix)
        metadata: item/service metadata; can include 'title', 'snippet', 'description', 'tags', 'item_id'
        mode: Mode.AUTO | INITIAL | OVERWRITE | APPEND
        staging_format: "gpkg" (recommended) or "fgdb"
        """
        # --- 0) Resolve GIS handle on this manager
        gis = getattr(self, "gis", None)
        if gis is None:
            raise RuntimeError("FeatureLayerManager.gis is not set")

        # --- 0.5) Normalize input to dict format for consistent processing
        if isinstance(layer_data, gpd.GeoDataFrame):
            # Single GeoDataFrame - wrap in dict with service name as key
            layer_data = {service_name: layer_data}

        # --- 1) Create defensive copies to avoid mutating input data
        layer_data_copy = {}
        try:
            for name, gdf in layer_data.items():
                # Create defensive copy and ensure proper CRS
                gdf_copy = gdf.copy()
                if gdf_copy.crs is None:
                    gdf_copy = gdf_copy.set_crs(4326)
                elif getattr(gdf_copy.crs, "to_epsg", lambda: None)() != 4326:
                    try:
                        gdf_copy = gdf_copy.to_crs(4326)
                    except Exception as e:
                        raise RuntimeError(f"Failed to transform CRS for layer '{name}': {e}") from e
                layer_data_copy[name] = gdf_copy
        except Exception as e:
            raise RuntimeError(f"Error preparing layer data: {e}") from e

        # --- 2) Find existing service (by explicit item_id or by title)
        existing = self._find_existing_service(
            title=metadata.get("title") or service_name,
            item_id=metadata.get("item_id"),
        )

        # --- 3) Handle mode logic properly
        if mode == Mode.INITIAL and existing is not None:
            raise RuntimeError(f"Service '{metadata.get('title') or service_name}' already exists. Use Mode.OVERWRITE or Mode.AUTO instead.")

        # --- 4) Stage a single multi-layer container on disk
        src_item = None
        fs_item = None
        tmpdir = None
        try:
            # Create temp directory in project temp space to avoid Windows AppData access issues
            tmpdir = get_pid_temp_dir() / f"agol_staging_{int(time.time())}"
            tmpdir.mkdir(parents=True, exist_ok=True)

            if staging_format.lower() == "gpkg":
                staged_path = tmpdir / f"{service_name}.gpkg"
                for name, gdf_copy in layer_data_copy.items():
                    gdf_copy.to_file(staged_path, layer=name, driver="GPKG")
                item_type = "GeoPackage"

            elif staging_format.lower() == "fgdb":
                gdb_dir = tmpdir / f"{service_name}.gdb"
                gdb_dir.mkdir(parents=True, exist_ok=True)
                for name, gdf_copy in layer_data_copy.items():
                    gdf_copy.to_file(gdb_dir, layer=name, driver="OpenFileGDB")
                # zip the .gdb folder for upload
                staged_path = tmpdir / f"{service_name}.gdb.zip"
                shutil.make_archive(str(staged_path).replace(".zip", ""), "zip", root_dir=tmpdir, base_dir=f"{service_name}.gdb")
                item_type = "File Geodatabase"

            else:
                raise ValueError("staging_format must be 'gpkg' or 'fgdb'")

            # --- 5) Create new service if needed
            if existing is None or mode == Mode.INITIAL:
                add_props = {
                    "title": metadata.get("title", service_name),
                    "snippet": metadata.get("snippet", ""),
                    "description": metadata.get("description", ""),
                    "tags": metadata.get("tags", []),
                    "type": item_type,
                }
                
                try:
                    src_item = gis.content.add(add_props, data=str(staged_path))
                    publish_params = {"name": service_name, "maxRecordCount": 5000}
                    fs_item = src_item.publish(publish_params)
                except Exception as e:
                    # Cleanup on failure
                    if src_item:
                        with contextlib.suppress(Exception):
                            src_item.delete()
                    raise RuntimeError(f"Failed to publish multi-layer service: {e}") from e

                # Apply item metadata
                try:
                    fs_item.update({
                        "title": metadata.get("title", service_name),
                        "snippet": metadata.get("snippet", ""),
                        "description": metadata.get("description", ""),
                        "tags": metadata.get("tags", []),
                        "accessInformation": metadata.get("access_information", ""),
                        "licenseInfo": metadata.get("license_info", "")
                    })
                except Exception as e:
                    logging.warning(f"Failed to update metadata for service {fs_item.id}: {e}")

                # Rename sublayers with better validation
                try:
                    self._rename_sublayers(fs_item, layer_data_copy.keys())
                except Exception as e:
                    logging.warning(f"Sublayer rename failed: {e}")

                # Critical: Clean up source item to prevent orphaned items
                try:
                    if src_item:
                        src_item.delete()
                        logging.debug(f"Cleaned up source item: {src_item.id}")
                except Exception as e:
                    logging.warning(f"Failed to clean up source item: {e}")

                logging.info(f"Created multi-layer feature service: {fs_item.id}")
                return fs_item.id

            # --- 6) Update existing service
            flc = FeatureLayerCollection.fromitem(existing)

            if mode in (Mode.OVERWRITE, Mode.AUTO):
                try:
                    flc.manager.overwrite(str(staged_path))
                    logging.info(f"Overwrote multi-layer feature service: {existing.id}")
                    return existing.id
                except Exception as e:
                    raise RuntimeError(f"Failed to overwrite existing service {existing.id}: {e}") from e

            elif mode == Mode.APPEND:
                # Schema validation and append by matching sublayers by name
                target_layers_by_name = {lyr.properties.name: lyr for lyr in flc.layers}
                
                for name, gdf_copy in layer_data_copy.items():
                    if name not in target_layers_by_name:
                        raise RuntimeError(f"Target sublayer '{name}' not found in existing service. Available layers: {list(target_layers_by_name.keys())}")
                    
                    # Basic schema validation
                    try:
                        target_layer = target_layers_by_name[name]
                        # Validate that we can convert to featureset
                        featureset = gdf_copy.spatial.to_featureset()
                        target_layer.append(featureset, upsert=False)
                    except Exception as e:
                        raise RuntimeError(f"Failed to append to sublayer '{name}': {e}") from e
                
                logging.info(f"Appended to multi-layer feature service: {existing.id}")
                return existing.id

            # Fallback
            return existing.id
                
        except Exception:
            # Cleanup on any failure
            if src_item:
                with contextlib.suppress(Exception):
                    src_item.delete()
                    logging.debug(f"Cleaned up source item after failure: {src_item.id}")
            raise
        finally:
            # NOTE: Temp directory cleanup is handled at CLI level to avoid premature deletion
            # during AGOL upload process. The staged files must remain available until the 
            # entire operation completes successfully.
            pass
    def _find_existing_service(self, title: Optional[str] = None, item_id: Optional[str] = None):
        """Return existing Hosted Feature Layer item by item_id or title (owned by current user)."""
        gis = getattr(self, "gis", None)
        if gis is None:
            return None
        if item_id:
            try:
                return gis.content.get(item_id)
            except Exception:
                return None
        if not title:
            return None
        # Limit to current owner to avoid collisions across the org
        try:
            owner = gis.users.me.username
            # Fixed: Use "Feature Service" for hosted feature layers
            items = gis.content.search(f'title:"{title}" AND owner:{owner} type:"Feature Service"', max_items=5)
            return items[0] if items else None
        except Exception:
            return None

    def _rename_sublayers(self, fs_item, layer_names):
        """Rename sublayers with better validation and error handling."""
        if not fs_item or not fs_item.layers:
            return
            
        layer_mapping = {name: name for name in layer_names}
        renamed_count = 0
        
        for lyr in fs_item.layers:
            try:
                # Get current layer name
                current = getattr(lyr.properties, "name", None) or getattr(lyr.properties, "tableName", None)
                if not current:
                    continue
                    
                # Find matching target name
                target_name = None
                for expected_name in layer_mapping:
                    if current == expected_name or current.endswith(f"_{expected_name}"):
                        target_name = layer_mapping[expected_name]
                        break
                        
                if target_name and target_name != current:
                    lyr.manager.update_definition({"name": target_name})
                    renamed_count += 1
                    logging.debug(f"Renamed layer '{current}' to '{target_name}'")
                    
            except Exception as e:
                logging.warning(f"Failed to rename layer: {e}")
                
        if renamed_count > 0:
            logging.info(f"Renamed {renamed_count} sublayers")

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