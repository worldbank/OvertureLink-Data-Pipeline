from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, Literal, Optional, TypedDict, Union
from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
import numpy as np
from arcgis.gis import GIS
from pathlib import Path

# Import secure configuration system
from o2agol.config.settings import Config, ConfigurationError

# Type definitions for unified publishing
GeoFrame = gpd.GeoDataFrame
Mode = Literal["auto", "initial", "overwrite", "append"]
GeometryType = Literal["Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"]
LayerBundle = Dict[str, GeoFrame]

class PublishParams(TypedDict, total=False):
    name: str
    hasStaticData: bool
    maxRecordCount: int
    layerInfo: Dict[str, Union[str, Dict]]

class AGOLMetadata(TypedDict, total=False):
    item_title: str
    service_name: str
    tags: list
    snippet: str
    description: str
    license_info: str
    access_information: str
    item_id: Optional[str]

@dataclass
class DatasetSize:
    feature_count: int
    file_size_mb: float
    is_large_dataset: bool
    geometry_type: str
    complexity_score: float

from .cleanup import get_pid_temp_dir  # make sure this import exists at the top



def validate_and_clean_geometries(gdf: gpd.GeoDataFrame, layer_name: str = "data") -> gpd.GeoDataFrame:
    """
    Advanced geometry validation and cleaning for all polygon data types.
    
    This function addresses the most common causes of AGOL "Unknown error" failures:
    1. Complex MultiPolygon structures with excessive coordinate counts
    2. Invalid geometries that cause publishing failures
    3. Degenerate or empty geometries
    4. Self-intersecting polygons
    
    Args:
        gdf: Input GeoDataFrame with potentially problematic geometries
        layer_name: Layer name for logging
        
    Returns:
        Cleaned GeoDataFrame optimized for AGOL publishing
    """
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.validation import make_valid
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    if gdf.empty:
        return gdf
    
    logging.debug(f"Validating and cleaning {len(gdf):,} geometries for {layer_name}")
    original_count = len(gdf)
    
    # Step 1: Remove null geometries
    null_mask = gdf.geometry.isna()
    if null_mask.any():
        logging.debug(f"Removing {null_mask.sum()} null geometries")
        gdf = gdf[~null_mask].copy()
    
    # Step 2: Remove empty geometries  
    empty_mask = gdf.geometry.apply(lambda x: x.is_empty if x is not None else True)
    if empty_mask.any():
        logging.debug(f"Removing {empty_mask.sum()} empty geometries")
        gdf = gdf[~empty_mask].copy()
    
    # Step 3: Advanced polygon cleaning for all polygon data
    geom_types = gdf.geometry.geom_type.value_counts()
    has_polygons = any(gt in ['Polygon', 'MultiPolygon'] for gt in geom_types.index)
    
    if has_polygons:
        def clean_polygon_geometry(geom):
            """Clean individual polygon geometry for AGOL compatibility"""
            try:
                if geom is None or geom.is_empty:
                    return None
                
                # Fix invalid geometries using make_valid (more robust than buffer(0))
                if not geom.is_valid:
                    geom = make_valid(geom)
                    if geom is None or geom.is_empty:
                        return None
                
                # Handle complex MultiPolygons that cause AGOL failures
                if isinstance(geom, MultiPolygon):
                    # Count total coordinates across all parts
                    total_coords = sum(len(poly.exterior.coords) + 
                                     sum(len(interior.coords) for interior in poly.interiors)
                                     for poly in geom.geoms)
                    
                    # Simplify if overly complex (empirically determined threshold)
                    if total_coords > 2000:
                        # Progressive simplification
                        for tolerance in [0.00001, 0.0001, 0.001]:
                            simplified = geom.simplify(tolerance, preserve_topology=True)
                            if simplified.is_valid and not simplified.is_empty:
                                geom = simplified
                                break
                    
                    # Remove tiny polygons that cause issues
                    valid_polys = []
                    for poly in geom.geoms:
                        if poly.area > 1e-10:  # Remove extremely small polygons
                            valid_polys.append(poly)
                    
                    if valid_polys:
                        geom = MultiPolygon(valid_polys)
                    else:
                        return None
                
                elif isinstance(geom, Polygon):
                    # Convert to MultiPolygon for consistency
                    coord_count = len(geom.exterior.coords) + sum(len(interior.coords) for interior in geom.interiors)
                    
                    if coord_count > 2000:
                        for tolerance in [0.00001, 0.0001, 0.001]:
                            simplified = geom.simplify(tolerance, preserve_topology=True)
                            if simplified.is_valid and not simplified.is_empty:
                                geom = simplified
                                break
                    
                    # Check minimum area
                    if geom.area > 1e-10:
                        geom = MultiPolygon([geom])
                    else:
                        return None
                
                # Final validation
                if geom and geom.is_valid and not geom.is_empty:
                    return geom
                else:
                    return None
                    
            except Exception:
                # If all cleaning fails, return None to remove the feature
                return None
        
        # Apply cleaning
        logging.debug("Applying advanced geometry cleaning for polygon data...")
        cleaned_geoms = gdf.geometry.apply(clean_polygon_geometry)
        
        # Remove features with None geometries
        valid_mask = cleaned_geoms.notna()
        cleaned_count = valid_mask.sum()
        removed_count = len(gdf) - cleaned_count
        
        if removed_count > 0:
            logging.debug(f"Removed {removed_count} features with uncleanlable geometries")
            gdf = gdf[valid_mask].copy()
            gdf.geometry = cleaned_geoms[valid_mask]
        else:
            gdf.geometry = cleaned_geoms
    
    final_count = len(gdf)
    removed_total = original_count - final_count
    
    if removed_total > 0:
        logging.debug(f"Geometry validation complete: {final_count:,} features ({removed_total} removed)")
    else:
        logging.debug(f"Geometry validation complete: {final_count:,} features")
    
    return gdf

# Configuration constants for large dataset handling
LARGE_DATASET_SEED_SIZE = int(os.environ.get('LARGE_DATASET_SEED_SIZE', '1000'))
LARGE_DATASET_BATCH_SIZE = int(os.environ.get('LARGE_DATASET_BATCH_SIZE', '5000'))
BUILDING_LARGE_THRESHOLD = int(os.environ.get('BUILDING_LARGE_THRESHOLD', '5000000'))


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


def _analyze_dataset_size(gdf: gpd.GeoDataFrame) -> DatasetSize:
    """
    Analyze dataset size and complexity to determine optimal publishing strategy.
    
    Calculates feature count, estimated file size, geometry complexity, and determines
    if dataset should be considered "large" requiring special handling.
    
    Args:
        gdf: GeoDataFrame to analyze
        
    Returns:
        DatasetSize dict with analysis results
    """
    feature_count = len(gdf)
    
    # Estimate file size based on geometry type and feature count
    geom_types = gdf.geometry.geom_type.value_counts()
    primary_geom_type = geom_types.index[0] if len(geom_types) > 0 else "Unknown"
    
    # Estimate bytes per feature based on geometry type
    if primary_geom_type in ['Point', 'MultiPoint']:
        bytes_per_feature = 200  # Points are compact
    elif primary_geom_type in ['LineString', 'MultiLineString']:
        bytes_per_feature = 800  # Lines have moderate complexity
    elif primary_geom_type in ['Polygon', 'MultiPolygon']:
        # Estimate polygon complexity from coordinate count sample
        sample_size = min(1000, len(gdf))
        sample_coords = 0
        for geom in gdf.geometry.head(sample_size):
            if hasattr(geom, 'exterior'):
                sample_coords += len(geom.exterior.coords)
            elif hasattr(geom, 'geoms'):
                for g in geom.geoms:
                    if hasattr(g, 'exterior'):
                        sample_coords += len(g.exterior.coords)
        
        avg_coords = sample_coords / sample_size if sample_size > 0 else 50
        bytes_per_feature = max(1000, avg_coords * 24)  # 24 bytes per coordinate pair + metadata
    else:
        bytes_per_feature = 500  # Default estimate
    
    # Add attribute data size estimate (approximately 200 bytes per feature for typical Overture attributes)
    attribute_bytes = 200
    total_bytes_per_feature = bytes_per_feature + attribute_bytes
    
    estimated_size_mb = (feature_count * total_bytes_per_feature) / (1024 * 1024)
    
    # Calculate complexity score (higher = more complex)
    complexity_score = feature_count / 10000  # Base complexity from feature count
    
    if primary_geom_type in ['Polygon', 'MultiPolygon']:
        complexity_score *= 3  # Polygons are more complex
    elif primary_geom_type in ['LineString', 'MultiLineString']:
        complexity_score *= 1.5  # Lines are moderately complex
    
    # Determine if this is a large dataset requiring special handling
    # Buildings/polygons over 5M features always require seed-and-append due to analyze timeouts
    is_large_dataset = (
        feature_count > 1_000_000 or  # More than 1M features
        (estimated_size_mb > 500 and feature_count > 1000) or   # More than 500MB AND significant feature count (avoid size estimation bugs)
        complexity_score > 250 or    # High complexity score
        (primary_geom_type in ['Polygon', 'MultiPolygon'] and feature_count > BUILDING_LARGE_THRESHOLD)  # Large building datasets
    )
    
    return DatasetSize(
        feature_count=feature_count,
        file_size_mb=estimated_size_mb,
        is_large_dataset=is_large_dataset,
        geometry_type=primary_geom_type,
        complexity_score=complexity_score
    )




def _gdf_to_geojson_tempfile(gdf: gpd.GeoDataFrame) -> tempfile.NamedTemporaryFile:
    """
    Convert GeoDataFrame to temporary GeoJSON file for ArcGIS Online upload.
    
    Uses GeoPandas built-in GeoJSON export for consistent handling of all data types.
    Creates temp files in project /temp directory to avoid system temp clutter.
    
    Args:
        gdf: GeoDataFrame to convert
        
    Returns:
        NamedTemporaryFile containing GeoJSON data
        
    Note:
        Caller is responsible for cleanup of temporary file
    """
    # Get PID-isolated temp directory
    temp_dir = get_pid_temp_dir()
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in PID-isolated directory and close handle to avoid Windows file locking
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False, 
                                     dir=str(temp_dir), encoding="utf-8")
    tmp_name = tmp.name
    tmp.close()  # Close handle before GeoPandas writes to avoid file locking on Windows
    
    # Use GeoPandas built-in GeoJSON export - handles all data types correctly
    gdf.to_file(tmp_name, driver='GeoJSON')
    
    # Return compatible object with name attribute
    class TempFile:
        def __init__(self, name):
            self.name = name
    
    return TempFile(tmp_name)


def _create_feature_service(
    gis: GIS, 
    gdf: GeoFrame,
    service_name: str,
    item_properties: dict,
    analyze: bool = True
) -> object:
    """
    Unified feature service creation with proper AGOL parameters.
    
    This function eliminates duplication between single-layer and multi-layer
    publishing by providing consistent parameter generation and creation logic.
    
    Args:
        gis: Authenticated GIS connection
        gdf: Prepared GeoDataFrame ready for publishing
        service_name: Clean service name for AGOL
        item_properties: Complete item metadata dict
        analyze: Whether to use AGOL analyze for enhanced parameters
        
    Returns:
        Published feature service item
        
    Raises:
        RuntimeError: If service creation fails
    """
    if gdf.empty:
        raise ValueError("Cannot create service from empty GeoDataFrame")
    
    # Create temporary GeoJSON
    tmp = _gdf_to_geojson_tempfile(gdf)
    
    try:
        logging.info(f"Creating AGOL item with {len(item_properties)} metadata fields")
        item = gis.content.add(item_properties, data=tmp.name)
        
        if not item:
            raise RuntimeError("Failed to create GeoJSON item in AGOL")
        
        # Build publish parameters
        publish_params: PublishParams = {'name': service_name}
        
        if analyze:
            try:
                logging.debug("Analyzing GeoJSON for optimal publish parameters")
                analyzed = gis.content.analyze(item=item.id, file_type='geojson')
                analyze_params = analyzed.get('publishParameters', {})
                
                # Filter out parameters that cause "overwrite" errors during initial publish
                # These parameters are meant for update operations, not initial publish
                problematic_params = [
                    'editorTrackingInfo',  # Causes "overwrite unsuccessful" error
                    'useBulkInserts',      # Can conflict with initial service creation
                    'targetSR',            # We handle CRS transformation ourselves
                    'layers',              # Complex layer definition causes "Unknown error"
                    'streamFeatures'       # Can conflict with hasStaticData
                ]
                
                filtered_params = {k: v for k, v in analyze_params.items() 
                                 if k not in problematic_params}
                
                # Merge filtered parameters with our requirements
                publish_params.update(filtered_params)
                logging.debug(f"AGOL analyze provided: {list(analyze_params.keys())}")
                if len(filtered_params) < len(analyze_params):
                    logging.debug(f"Filtered out problematic params: {[k for k in analyze_params.keys() if k in problematic_params]}")
                
            except Exception as analyze_error:
                logging.warning(f"AGOL analyze failed, using basic parameters: {analyze_error}")
        
        # Add/override with our optimized parameters (matching manual publish)
        feature_count = len(gdf)
        publish_params.update({
            'name': service_name,
            'hasStaticData': True,
            'maxRecordCount': 2000,  # Match manual publish exactly
            'layerInfo': {
                'capabilities': 'Query'  # Match manual publish exactly
            },
            'fieldTypesVersion': 'V2'  # Add missing parameter from manual publish
        })
        
        logging.debug(f"Publishing with parameters: {list(publish_params.keys())}")
        logging.debug(f"Full publish parameters: {publish_params}")
        
        # Publish the feature service
        try:
            fl_item = item.publish(publish_parameters=publish_params)
            
            if not fl_item:
                raise RuntimeError("Publish operation returned None")
                
            logging.info(f"Successfully created service '{fl_item.title}' (ID: {fl_item.itemid})")
            logging.info(f"Service URL: {fl_item.homepage}")
            
            # Clean up staged GeoJSON item after successful publish
            cleanup_success = _cleanup_geojson_item(item)
            if cleanup_success:
                logging.debug(f"Cleaned up staged GeoJSON item: {item.itemid}")
            else:
                logging.warning(f"Failed to cleanup staged GeoJSON item: {item.itemid}")
                logging.info("Manual cleanup required: delete the staged GeoJSON item from AGOL interface")
                # Not critical - the service was created successfully
            
            return fl_item
            
        except Exception as publish_error:
            # Enhanced error diagnostics with full API response details
            error_message = str(publish_error).lower()
            
            # Log staged item URL for manual debugging
            if hasattr(item, 'url'):
                logging.error(f"Staged GeoJSON item URL: {item.url}")
                logging.error("You can attempt manual publish via AGOL interface to get more specific error details")
            
            # Log detailed item information for debugging
            logging.error(f"Staged item details:")
            logging.error(f"  - Item ID: {item.itemid}")
            logging.error(f"  - Title: {getattr(item, 'title', 'N/A')}")
            logging.error(f"  - Type: {getattr(item, 'type', 'N/A')}")
            logging.error(f"  - Size: {getattr(item, 'size', 'N/A')} bytes")
            
            # Check for related items and dependencies
            try:
                related_items = item.related_items(relationship_type='Service2Data', direction='reverse')
                if related_items:
                    logging.error(f"  - Found {len(related_items)} related services (reverse lookup)")
                    for rel_item in related_items:
                        logging.error(f"    - {rel_item.title} ({rel_item.itemid})")
                
                dependent_items = item.related_items(relationship_type='Service2Data', direction='forward') 
                if dependent_items:
                    logging.error(f"  - Found {len(dependent_items)} dependent items")
                    for dep_item in dependent_items:
                        logging.error(f"    - {dep_item.title} ({dep_item.itemid})")
            except Exception as rel_error:
                logging.warning(f"Could not check relationships: {rel_error}")
            
            # Capture detailed API response if available
            logging.error(f"Full publish error details: {publish_error}")
            if hasattr(publish_error, 'args') and publish_error.args:
                logging.error(f"Error args: {publish_error.args}")
            
            # Try to extract AGOL API response details
            if hasattr(publish_error, 'response'):
                try:
                    response_text = publish_error.response.text
                    logging.error(f"AGOL API response: {response_text}")
                except:
                    pass
            
            # Provide specific guidance based on error type
            if "unknown error" in error_message:
                logging.error("AGOL 'Unknown error' detected - likely data quality issues:")
                logging.error("  - Complex polygon geometries (>1000 coordinates per feature)")
                logging.error("  - Invalid geometry structures in MultiPolygons")
                logging.error("  - Problematic attribute data (nested JSON, special characters)")
                logging.error("  - Schema incompatibilities with AGOL field types")
                
                # Log diagnostic information about the data
                if not gdf.empty:
                    geom_types = gdf.geometry.geom_type.value_counts()
                    logging.error(f"  - Data contains geometry types: {geom_types.to_dict()}")
                    logging.error(f"  - Feature count: {len(gdf):,}")
                    logging.error(f"  - Columns: {list(gdf.columns)}")
                    
                    # Sample geometric complexity
                    if any(gt in ['Polygon', 'MultiPolygon'] for gt in geom_types.index):
                        sample_geom = next((g for g in gdf.geometry if g.geom_type in ['Polygon', 'MultiPolygon']), None)
                        if sample_geom:
                            coord_count = 0
                            if hasattr(sample_geom, 'exterior'):
                                coord_count = len(sample_geom.exterior.coords)
                            elif hasattr(sample_geom, 'geoms'):
                                coord_count = sum(len(g.exterior.coords) for g in sample_geom.geoms if hasattr(g, 'exterior'))
                            logging.error(f"  - Sample polygon has {coord_count} coordinates")
                
            elif "timeout" in error_message or "analyze" in error_message:
                logging.error("AGOL analyze timeout - try seed-and-append approach for large datasets")
            elif "schema" in error_message or "field" in error_message:
                logging.error("Schema compatibility issue - check field names and data types")
            elif "overwrite unsuccessful" in error_message:
                logging.error("Overwrite error on NEW service - likely staged item deletion timing issue")
                logging.error("Staged item preserved for debugging - check AGOL interface")
            
            # CRITICAL FIX: Do NOT delete staged GeoJSON item on publish failure
            # This was causing cascade failures when AGOL couldn't find the source
            logging.error(f"Service publish failed: {publish_error}")
            logging.info(f"Staged item preserved for debugging: {item.itemid}")
            logging.info("Manual cleanup required: delete the staged GeoJSON item from AGOL interface")
            
            raise RuntimeError(f"Failed to publish service: {publish_error}")
            
    finally:
        # Always cleanup temp file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass





def _initial_with_seed_and_append(
    gis: GIS, 
    gdf: GeoFrame,
    tgt,
    dataset_size: DatasetSize
) -> object:
    """
    Create feature service using seed-and-append strategy for large datasets.
    
    This approach creates the service with a small sample to establish schema quickly,
    then truncates and appends the full dataset in batches. This avoids the AGOL
    analyze timeout that occurs with multi-GB GeoJSON files.
    
    Args:
        gis: Authenticated GIS connection
        gdf: Full GeoDataFrame to publish (large dataset)
        tgt: Target configuration with AGOL settings
        dataset_size: Dataset analysis results
        
    Returns:
        Published feature service item
        
    Raises:
        RuntimeError: If service creation or data append fails
    """
    if gdf.empty:
        raise ValueError("Cannot create service from empty GeoDataFrame")
    
    feature_count = len(gdf)
    service_name = tgt.agol.service_name
    title = tgt.agol.item_title
    
    logging.debug(f"Using seed-and-append strategy for large dataset: {feature_count:,} features")
    
    # PHASE 1: Create seed with small sample to establish schema
    seed_size = min(LARGE_DATASET_SEED_SIZE, feature_count)
    seed_gdf = gdf.head(seed_size)
    
    logging.debug(f"Creating service schema with seed sample: {seed_size:,} features")
    
    # Prepare seed data for AGOL
    prepared_seed = _prepare_gdf_for_agol(seed_gdf, "single-layer")
    
    if prepared_seed.empty:
        raise RuntimeError("No valid features remaining in seed after preparation")
    
    # Build comprehensive metadata dict for enterprise publishing
    item_properties = {
        "type": "GeoJson",
        "title": title,
        "tags": ",".join(tgt.agol.tags or []),
        "snippet": tgt.agol.snippet,
        "description": tgt.agol.description,
        "accessInformation": tgt.agol.access_information,
        "licenseInfo": tgt.agol.license_info
    }
    
    # Create service with seed data (fast - no analyze timeout)
    fl_item = _create_feature_service(
        gis, prepared_seed, service_name, item_properties, analyze=False
    )
    
    logging.info(f"Service created successfully: {fl_item.title} (ID: {fl_item.itemid})")
    
    # PHASE 2: Clear seed data and append full dataset
    try:
        feature_layer = fl_item.layers[0]
        
        # Clear the seed data
        logging.info("Truncating seed data from service...")
        truncate_result = feature_layer.manager.truncate()
        if not truncate_result.get('success', False):
            raise RuntimeError(f"Failed to truncate seed data: {truncate_result}")
        
        logging.info("Seed data cleared - starting batch append of full dataset")
        
        # Prepare full dataset for AGOL
        prepared_gdf = _prepare_gdf_for_agol(gdf, "single-layer")
        
        if prepared_gdf.empty:
            raise RuntimeError("No valid features remaining after preparation")
        
        # Batch append the full dataset
        _append_via_batches(feature_layer, prepared_gdf, LARGE_DATASET_BATCH_SIZE, gis)
        
        logging.info(f"Successfully populated service with {len(prepared_gdf):,} features")
        return fl_item
        
    except Exception as append_error:
        # Clean up failed service
        logging.error(f"Failed to populate service with full data: {append_error}")
        try:
            fl_item.delete()
            logging.info("Cleaned up failed service")
        except Exception as cleanup_error:
            logging.warning(f"Failed to cleanup service: {cleanup_error}")
        
        raise RuntimeError(f"Seed-and-append failed: {append_error}")


def _append_via_batches(
    feature_layer,
    gdf: gpd.GeoDataFrame,
    batch_size: int,
    gis: GIS
) -> None:
    """
    Append data to feature layer in optimized batches.
    
    Uses the existing append-via-item pattern for reliability with
    enhanced batch processing for large polygon datasets.
    
    Args:
        feature_layer: Target feature layer to append to
        gdf: GeoDataFrame to append
        batch_size: Number of features per batch
        gis: Authenticated GIS connection
        
    Raises:
        RuntimeError: If batch append fails
    """
    total_features = len(gdf)
    successful_adds = 0
    
    logging.info(f"Starting batch append: {total_features:,} features in batches of {batch_size:,}")
    
    # Try append-via-item first (more reliable for large datasets)
    try:
        append_success = _append_via_item(feature_layer, gdf, gis)
        if append_success:
            logging.info(f"Batch append completed via item method: {total_features:,} features")
            return
    except Exception as item_error:
        logging.warning(f"Append via item failed: {item_error}")
    
    # Fallback to batched edit_features method
    logging.info("Falling back to batched feature edit method...")
    
    # Ensure proper geometry setup for ArcGIS spatial accessor
    prepared_gdf = gdf.set_geometry('geometry')
    
    for i in range(0, total_features, batch_size):
        batch_gdf = prepared_gdf.iloc[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_features + batch_size - 1) // batch_size
        
        try:
            feature_set = batch_gdf.spatial.to_featureset()
            edit_result = feature_layer.edit_features(adds=feature_set.features)
            
            add_results = edit_result.get('addResults', [])
            batch_successful = sum(1 for r in add_results if r.get('success', False))
            successful_adds += batch_successful
            
            logging.info(f"Batch {batch_num}/{total_batches}: {batch_successful:,}/{len(batch_gdf):,} features added")
            
            if batch_successful < len(batch_gdf):
                failed_count = len(batch_gdf) - batch_successful
                logging.warning(f"Batch {batch_num}: {failed_count} features failed")
                
        except Exception as batch_error:
            logging.error(f"Batch {batch_num} failed: {batch_error}")
    
    if successful_adds != total_features:
        logging.warning(f"Partial upload: {successful_adds:,}/{total_features:,} features uploaded")
        if successful_adds == 0:
            raise RuntimeError("All batches failed - no features were uploaded")
    else:
        logging.info(f"Batch append completed successfully: {successful_adds:,} features uploaded")


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
        
        # TEMPORARY TEST: Force initial mode by modifying title for testing
        if "TEST_INITIAL_MODE" in os.environ:
            title = title + " TEST INITIAL"
            tgt.agol.item_title = title
            logging.info(f"TEST MODE: Modified title to '{title}' to force initial mode testing")
            mode = "initial"
        else:
            existing_service = _find_existing_service(gis, title, getattr(tgt.agol, 'item_id', None), tgt)
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
        # Check for service name conflicts before creating
        service_name = tgt.agol.service_name
        conflicting_service = _find_existing_service(gis, title, None, tgt)
        
        if conflicting_service:
            # Check if this is an orphaned GeoJSON that we can republish
            if hasattr(conflicting_service, '_can_republish') and conflicting_service._can_republish:
                logging.info(f"Found orphaned GeoJSON item with matching title: {conflicting_service.title}")
                logging.info(f"Republishing existing GeoJSON item: {conflicting_service.itemid}")
                
                # Use the existing GeoJSON item to publish our service
                tmp = _gdf_to_geojson_tempfile(gdf)
                
                try:
                    # Update the existing GeoJSON item with new data
                    logging.info("Updating orphaned GeoJSON item with new data...")
                    update_result = conflicting_service.update(data=tmp.name)
                    
                    if not update_result:
                        raise RuntimeError("Failed to update existing GeoJSON item")
                    
                    # Publish the updated GeoJSON as a feature service
                    publish_params = {'name': service_name}
                    fl_item = conflicting_service.publish(publish_parameters=publish_params)
                    
                    logging.info(f"Successfully republished service '{fl_item.title}' with ID: {fl_item.itemid}")
                    logging.info(f"Service URL: {fl_item.homepage}")
                    
                    return fl_item
                    
                finally:
                    # Cleanup temp file
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass
            else:
                # Found a service that would conflict with our service name
                logging.warning(f"Service name '{service_name}' conflicts with existing service: {conflicting_service.title}")
                logging.info(f"Attempting to reuse existing service: {conflicting_service.itemid}")
                
                # Set the item_id and switch to overwrite mode
                tgt.agol.item_id = conflicting_service.itemid
                mode = "overwrite"
                # Fall through to overwrite logic below
        else:
            logging.info(f"Service name '{service_name}' is available - proceeding with creation")
        
        if mode == "initial":  # Only proceed if we didn't switch to overwrite
            
            # STANDARD PATH: Check dataset size and route to appropriate strategy
            dataset_size = _analyze_dataset_size(gdf)
            
            if dataset_size.is_large_dataset:
                logging.info(f"Large dataset detected: {len(gdf):,} features, estimated {dataset_size.file_size_mb:.1f}MB")
                logging.info("Using seed-and-append strategy to avoid analyze timeouts")
                
                # Use seed-and-append strategy for large datasets
                return _initial_with_seed_and_append(gis, gdf, tgt, dataset_size)
            
            # Standard publishing path for smaller datasets
            logging.info(f"Preparing {len(gdf):,} features for AGOL publishing")
            prepared_gdf = _prepare_gdf_for_agol(gdf, "single-layer")
            
            if prepared_gdf.empty:
                raise RuntimeError("No valid features remaining after preparation")
            
            logging.info(f"Publishing {len(prepared_gdf):,} prepared features")
            
            # Build comprehensive metadata dict for enterprise publishing
            item_properties = {
                "type": "GeoJson",
                "title": title,
                "tags": ",".join(tags),
                "snippet": tgt.agol.snippet,
                "description": tgt.agol.description,
                "accessInformation": tgt.agol.access_information,
                "licenseInfo": tgt.agol.license_info
            }
            
            # Clean up any orphaned items before creating new service
            logging.debug(f"Checking for orphaned GeoJSON items for service: {service_name}")
            _cleanup_orphaned_geojson_items(gis, service_name)
            
            # Use unified feature service creation with AGOL analyze
            try:
                fl_item = _create_feature_service(
                    gis=gis,
                    gdf=prepared_gdf,
                    service_name=service_name,
                    item_properties=item_properties,
                    analyze=True  # Re-enable AGOL analyze - parameter filtering handles problematic params
                )
                
                logging.info(f"Add to config - item_id: {fl_item.itemid}")
                return fl_item
                
            except Exception as create_error:
                logging.error(f"Failed to create service: {create_error}")
                raise create_error

    # Update existing hosted feature layer
    if mode in ("overwrite", "append"):
        item_id = getattr(tgt.agol, 'item_id', None)
        fl_item = gis.content.get(item_id)
        
        if not fl_item:
            raise RuntimeError(f"Item not found: {item_id}")
        
        # Special handling for orphaned GeoJSON items
        if fl_item.type == "GeoJson":
            logging.info("Found orphaned GeoJSON item - republishing as Feature Service")
            
            # Update the GeoJSON with new data and republish
            tmp = _gdf_to_geojson_tempfile(gdf)
            
            try:
                # Update the existing GeoJSON item with new data
                logging.info("Updating orphaned GeoJSON item with new data...")
                update_result = fl_item.update(data=tmp.name)
                
                if not update_result:
                    raise RuntimeError("Failed to update existing GeoJSON item")
                
                # Publish the updated GeoJSON as a feature service
                service_name = tgt.agol.service_name
                publish_params = {'name': service_name}
                
                # Try to publish, if it fails due to name conflict, try with unique name
                try:
                    new_fl_item = fl_item.publish(publish_parameters=publish_params)
                except Exception as publish_error:
                    if "already exists" in str(publish_error):
                        # Generate unique service name
                        import time
                        unique_suffix = int(time.time()) % 10000
                        unique_service_name = f"{service_name}_{unique_suffix}"
                        logging.warning(f"Service name '{service_name}' conflicts, trying unique name: {unique_service_name}")
                        
                        publish_params = {'name': unique_service_name}
                        new_fl_item = fl_item.publish(publish_parameters=publish_params)
                    else:
                        raise publish_error
                
                logging.info(f"Successfully republished service '{new_fl_item.title}' with ID: {new_fl_item.itemid}")
                logging.info(f"Service URL: {new_fl_item.homepage}")
                
                return new_fl_item
                
            finally:
                # Cleanup temp file
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
        
        elif fl_item.type != "Feature Service":
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
        
        # For overwrite mode, clear existing data first then update
        if mode == "overwrite":
            logging.info("Clearing existing data...")
            truncate_result = feature_layer.manager.truncate()
            if not truncate_result.get('success', False):
                raise RuntimeError(f"Failed to clear existing data: {truncate_result}")
            logging.info("Existing data cleared")
        
        # Update approach for monthly data replacement while preserving item_id
        # CRITICAL FIX: Always prepare data before any update operations
        logging.info(f"Preparing {len(gdf):,} features for overwrite/append")
        prepared_gdf = _prepare_gdf_for_agol(gdf, "single-layer-update")
        
        if prepared_gdf.empty:
            logging.warning("No features remaining after preparation")
            return fl_item
        
        logging.info(f"Uploading {len(prepared_gdf):,} prepared features")
        
        # Use unified append approach for all overwrite/append operations
        try:
            _append_via_item(feature_layer, prepared_gdf, gis)
            logging.info("Data append completed successfully")
            
        except Exception as append_error:
            logging.warning(f"Append via item failed: {append_error}")
            
            # Fallback to batched edit_features method
            logging.info("Falling back to batched feature edit method...")
            
            # Ensure proper geometry setup for ArcGIS spatial accessor
            prepared_gdf = prepared_gdf.set_geometry('geometry')
            
            # Batch processing for large datasets - optimize batch size by geometry type
            geom_types = prepared_gdf.geometry.geom_type.value_counts()
            primary_geom_type = geom_types.index[0] if len(geom_types) > 0 else "Unknown"
            
            if primary_geom_type in ['Polygon', 'MultiPolygon']:
                batch_size = LARGE_DATASET_BATCH_SIZE  # Larger batches for polygons (5000)
                logging.info(f"Using optimized batch size for {primary_geom_type}: {batch_size:,}")
            else:
                batch_size = 2000  # Keep original size for points/lines
                
            total_features = len(prepared_gdf)
            successful_adds = 0
            
            for i in range(0, total_features, batch_size):
                batch_gdf = prepared_gdf.iloc[i:i+batch_size]
                
                try:
                    feature_set = batch_gdf.spatial.to_featureset()
                    edit_result = feature_layer.edit_features(adds=feature_set.features)
                    
                    add_results = edit_result.get('addResults', [])
                    batch_successful = sum(1 for r in add_results if r.get('success', False))
                    successful_adds += batch_successful
                    
                    logging.info(f"Batch {i//batch_size + 1}: {batch_successful}/{len(batch_gdf)} features")
                    
                except Exception as batch_error:
                    logging.error(f"Batch {i//batch_size + 1} failed: {batch_error}")
            
            if successful_adds != total_features:
                logging.warning(f"Partial upload: {successful_adds}/{total_features} features uploaded")
            else:
                logging.info(f"Batch upload completed: {successful_adds:,} features uploaded")
        
        logging.info(f"Data update operation completed - Item ID preserved: {item_id}")
        
        # Get final count (with brief delay for AGOL indexing)
        import time
        time.sleep(2)
        final_count = feature_layer.query(return_count_only=True)
        logging.info(f"Operation complete: {final_count:,} total features in layer")
        logging.info(f"Updated layer: {fl_item.homepage}")
        
        return fl_item

    raise ValueError(f"Unsupported mode: {mode}. Use 'initial', 'overwrite', or 'append'")



def _generate_service_name(tgt) -> str:
    """Generate service name from target configuration using ISO2 format."""
    if hasattr(tgt, 'agol') and hasattr(tgt.agol, 'service_name'):
        # Check if the existing service name uses the problematic AFG_ pattern
        base_name = tgt.agol.service_name
        
        # Convert AFG_ to AF_ to avoid the blocking issue
        if base_name.startswith('AFG_'):
            base_name = base_name.replace('AFG_', 'AF_')
        
        return base_name
    else:
        # Fallback to dynamic name using ISO2: {country_iso2}_{sector}
        # Extract country and sector from target title/metadata
        title = getattr(tgt.agol, 'item_title', 'Unknown') if hasattr(tgt, 'agol') else 'Unknown'
        
        # Parse title like "Afghanistan Building Footprints" -> "AF_buildings"
        parts = title.split()
        if len(parts) >= 2:
            country = parts[0]  # e.g., "Afghanistan"
            sector = parts[1]   # e.g., "Building" -> "buildings"
            
            # Map country names to ISO2 codes
            country_to_iso2 = {
                'Afghanistan': 'AF',
                'Pakistan': 'PK', 
                'Bangladesh': 'BD',
                'India': 'IN',
                'Nepal': 'NP',
                'Bhutan': 'BT',
                'Sri': 'LK'  # For "Sri Lanka"
            }
            
            iso2 = country_to_iso2.get(country, country[:2].upper())
            sector_clean = sector.lower().rstrip('s') + 's'  # Normalize to plural
            
            return f"{iso2}_{sector_clean}"
        else:
            # Ultimate fallback
            return "Service_Unknown"




def publish_multi_layer_service(layer_data: LayerBundle, tgt, mode: str = "auto"):
    """
    Publish or update multi-layer feature service for combined Overture data.
    
    Handles both places (points) and buildings (polygons) in a single service,
    with intelligent mode detection, duplicate data detection, and proper geometry 
    type handling using the unified publishing architecture.
    
    Args:
        layer_data: Dictionary mapping layer names to GeoDataFrames
        tgt: Target configuration with AGOL settings
        mode: Publishing mode - 'auto', 'initial', or 'overwrite'
        
    Returns:
        Published or updated feature service item
    """
    if not layer_data or all(gdf.empty for gdf in layer_data.values()):
        raise RuntimeError("No features to publish - all GeoDataFrames are empty")
    
    # PHASE 1: Detect and handle duplicate data
    layer_hashes = {}
    unique_layers = {}
    
    for layer_name, gdf in layer_data.items():
        if gdf.empty:
            continue
            
        # Create a simple hash to detect duplicate data
        gdf_hash = hash(tuple(gdf.columns)) + len(gdf)
        
        # Check for exact duplicates by comparing a few key properties
        is_duplicate = False
        for existing_name, existing_gdf in unique_layers.items():
            if (len(gdf) == len(existing_gdf) and 
                list(gdf.columns) == list(existing_gdf.columns)):
                
                # More thorough check: compare first few rows
                if len(gdf) > 0 and len(existing_gdf) > 0:
                    sample_size = min(5, len(gdf))
                    if gdf.iloc[:sample_size].equals(existing_gdf.iloc[:sample_size]):
                        logging.warning(f"Duplicate data detected: '{layer_name}' equals '{existing_name}'")
                        logging.info(f"Skipping duplicate layer '{layer_name}'")
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            unique_layers[layer_name] = gdf
    
    if not unique_layers:
        raise RuntimeError("No unique data to publish after duplicate detection")
    
    logging.info(f"Processing {len(unique_layers)} unique layers after duplicate detection")
    
    # PHASE 2: Prepare each layer's data
    prepared_layers = {}
    for layer_name, gdf in unique_layers.items():
        logging.debug(f"Preparing layer '{layer_name}' with {len(gdf):,} features")
        prepared_layers[layer_name] = _prepare_gdf_for_agol(gdf, layer_name)
    
    # Get title from configuration
    title = tgt.agol.item_title
    tags = tgt.agol.tags or []
    
    # PHASE 3: Safe source_type assignment and feature counting
    total_features = sum(len(gdf) for gdf in prepared_layers.values())
    
    logging.info(f"Publishing multi-layer service: {total_features:,} total features")
    for name, gdf in prepared_layers.items():
        logging.info(f"  - {name}: {len(gdf):,} features")
    
    # PHASE 4: Combine data with safe source_type assignment
    combined_parts = []
    
    for layer_name, gdf in prepared_layers.items():
        if gdf.empty:
            continue
            
        # Create a copy to avoid modifying original
        layer_gdf = gdf.copy()
        
        # Safe source_type assignment using numpy.repeat for exact length matching
        if 'source_type' not in layer_gdf.columns:
            layer_gdf['source_type'] = np.repeat(layer_name, len(layer_gdf))
            logging.debug(f"Added source_type column to {layer_name}: {len(layer_gdf)} values")
        else:
            # Validate existing source_type column
            current_values = layer_gdf['source_type'].unique()
            logging.debug(f"Layer {layer_name} has existing source_type values: {current_values}")
        
        combined_parts.append(layer_gdf)
    
    if not combined_parts:
        raise RuntimeError("No data parts to combine after processing")
    
    # Safely combine all data
    combined_gdf = pd.concat(combined_parts, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs='EPSG:4326')
    
    # Final preparation of combined data (light touch since already prepared)
    combined_gdf = _prepare_gdf_for_agol(combined_gdf, "multi-layer-combined")
    
    # Log final geometry type distribution
    geom_types = combined_gdf.geometry.geom_type.value_counts()
    logging.info(f"Combined geometry types: {geom_types.to_dict()}")
    
    # Validate source_type integrity
    if 'source_type' in combined_gdf.columns:
        source_counts = combined_gdf['source_type'].value_counts()
        logging.info(f"Source type distribution: {source_counts.to_dict()}")
    
    # Use secure GIS connection
    gis = _login_gis_for_publish()
    
    # Determine actual mode
    if mode == "auto":
        # Check if service already exists
        existing_service = _find_existing_service(gis, title, getattr(tgt.agol, 'item_id', None), tgt)
        if existing_service:
            logging.info(f"Found existing service '{title}' (ID: {existing_service.itemid})")
            return _update_existing_service(existing_service, combined_gdf, prepared_layers, gis, tgt)
        else:
            logging.info(f"No existing service found for '{title}' - creating new...")
    
    elif mode == "initial":
        logging.info(f"Creating new multi-layer service: {title}")
    
    elif mode == "overwrite":
        existing_service = _find_existing_service(gis, title, getattr(tgt.agol, 'item_id', None), tgt)
        if not existing_service:
            raise RuntimeError(f"Cannot overwrite - service '{title}' not found")
        return _update_existing_service(existing_service, combined_gdf, prepared_layers)
    
    # PHASE 5: Create new service using unified approach
    service_name = tgt.agol.service_name
    
    # Build comprehensive metadata for multi-layer service
    item_properties = {
        "type": "GeoJson",
        "title": title,
        "tags": ",".join(tags),
        "snippet": tgt.agol.snippet,
        "description": tgt.agol.description,
        "accessInformation": tgt.agol.access_information,
        "licenseInfo": tgt.agol.license_info
    }
    
    # Clean up any orphaned items before creating new service
    logging.debug(f"Checking for orphaned GeoJSON items for multi-layer service: {service_name}")
    _cleanup_orphaned_geojson_items(gis, service_name)
    
    # Use unified feature service creation with AGOL analyze
    try:
        fl_item = _create_feature_service(
            gis=gis,
            gdf=combined_gdf,
            service_name=service_name,
            item_properties=item_properties,
            analyze=True  # Enable AGOL analyze for optimal multi-layer parameters
        )
        
        logging.info(f"Multi-layer service creation completed successfully")
        return fl_item
        
    except Exception as create_error:
        logging.error(f"Failed to create multi-layer service: {create_error}")
        raise create_error


def _find_existing_service(gis: GIS, title: str, item_id: str = None, tgt=None):
    """
    Search for existing feature service by item_id first, then title and service name patterns.
    
    AGOL converts titles to URL-safe service names, so we need robust search logic
    to handle title changes and service name conflicts.
    
    Args:
        gis: Authenticated GIS connection
        title: Current title to search for
        item_id: Specific item ID to look for (optional)
        
    Returns:
        Feature service item if found, None otherwise
    """
    try:
        # Search 1: Direct item_id lookup (most reliable)
        if item_id:
            try:
                service = gis.content.get(item_id)
                if service and service.type == "Feature Service":
                    logging.info(f"Found service by item_id '{item_id}': {service.title}")
                    return service
                else:
                    logging.warning(f"Item {item_id} exists but is not a Feature Service (type: {service.type if service else 'None'})")
            except Exception as e:
                logging.warning(f"Failed to retrieve item by ID '{item_id}': {e}")
        
        # Generate potential service name using same logic as publish functions
        service_name = tgt.agol.service_name if tgt else title.replace(' ', '_').replace('-', '_').replace('&', 'and').lower()
        
        # Search 2: Exact title match for Feature Services
        title_query = f'title:"{title}" AND type:"Feature Service"'
        title_results = gis.content.search(
            query=title_query,
            item_type="Feature Service", 
            max_items=10
        )
        
        # Search 2b: Also check for orphaned GeoJSON items with same title
        geojson_query = f'title:"{title}" AND type:"GeoJson"'
        geojson_results = gis.content.search(
            query=geojson_query,
            item_type="GeoJson",
            max_items=5
        )
        
        # Check if any GeoJSON items can be republished as our service
        for geojson_item in geojson_results:
            try:
                # Check if this GeoJSON has no related feature services
                related_services = geojson_item.related_items('Service2Data', 'reverse')
                if len(related_services) == 0:
                    # This is an orphaned GeoJSON that can be republished
                    logging.info(f"Found orphaned GeoJSON item with matching title: {geojson_item.title} ({geojson_item.itemid})")
                    # Create a mock service object to represent this reusable GeoJSON
                    geojson_item._can_republish = True
                    geojson_item._original_type = geojson_item.type
                    geojson_item.type = "Feature Service"  # Temporarily change type for compatibility
                    title_results.append(geojson_item)
            except Exception as check_error:
                logging.debug(f"Error checking GeoJSON relationships for {geojson_item.title}: {check_error}")
                continue
        
        # Search 3: Service name pattern match (broader search)
        name_query = f'name:*{service_name}* AND type:"Feature Service"'
        name_results = gis.content.search(
            query=name_query,
            item_type="Feature Service",
            max_items=20
        )
        
        # Search 4: Check for exact service name conflicts using REST API
        try:
            # Check if service name exists by searching for URL patterns
            url_query = f'url:*{service_name}* AND type:"Feature Service"'
            url_results = gis.content.search(
                query=url_query,
                item_type="Feature Service",
                max_items=20
            )
            
            # Also check user's services for any containing this service name
            user_services = gis.content.search(
                query=f'owner:{gis.users.me.username} AND type:"Feature Service"',
                max_items=100
            )
            
            # Filter for services that might conflict with our service name
            service_name_conflicts = []
            for service in user_services:
                if hasattr(service, 'url') and service.url:
                    # Extract service name from URL (last segment before /FeatureServer)
                    url_parts = service.url.split('/')
                    if len(url_parts) > 1:
                        actual_service_name = url_parts[-2] if url_parts[-1] == 'FeatureServer' else url_parts[-1]
                        if actual_service_name.lower() == service_name.lower():
                            service_name_conflicts.append(service)
                            logging.warning(f"Found service name conflict: '{service.title}' uses service name '{actual_service_name}'")
            
            url_results.extend(service_name_conflicts)
            
        except Exception as search_error:
            logging.warning(f"URL-based service search failed: {search_error}")
            url_results = []
        
        # Search 5: Fallback - search by owner and service name pattern
        owner_query = f'owner:{gis.users.me.username} AND type:"Feature Service"'
        owner_results = gis.content.search(
            query=owner_query,
            item_type="Feature Service",
            max_items=50
        )
        
        # Combine and deduplicate results
        all_results = title_results + [item for item in name_results if item not in title_results]
        all_results += [item for item in url_results if item not in all_results]
        all_results += [item for item in owner_results if item not in all_results]
        
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
            if service_name.lower() in item_service_name.lower() or item_service_name.lower() in service_name.lower():
                potential_conflicts.append(item)
        
        if potential_conflicts:
            conflict_service = potential_conflicts[0]
            logging.info(f"Found service name match: '{conflict_service.title}' (ID: {conflict_service.itemid})")
            logging.info(f"Service name pattern '{service_name}' matches existing service '{conflict_service.name if hasattr(conflict_service, 'name') else 'unknown'}'")
            return conflict_service
        
        logging.info(f"No existing service found for '{title}' (checked title, service name patterns, and owner services)")
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
                logging.info("Data append completed successfully")
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
    
    Ensures geometry column is properly configured, CRS is set, invalid geometries
    are repaired, and data types are compatible with ArcGIS. Enhanced with 
    polygon-specific normalization and repair for robust publishing.
    
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
    
    # CRITICAL: Apply advanced geometry validation for building data
    # This addresses the primary cause of "Unknown error" in AGOL publishing
    if any(gt in ['Polygon', 'MultiPolygon'] for gt in prepared_gdf.geometry.geom_type.unique()):
        prepared_gdf = validate_and_clean_geometries(prepared_gdf, layer_name)
    
    # Ensure CRS is WGS84
    if prepared_gdf.crs is None:
        logging.debug(f"Setting CRS to EPSG:4326 for {layer_name}")
        prepared_gdf = prepared_gdf.set_crs('EPSG:4326')
    elif prepared_gdf.crs.to_string() != 'EPSG:4326':
        logging.debug(f"Reprojecting {layer_name} from {prepared_gdf.crs} to EPSG:4326")
        prepared_gdf = prepared_gdf.to_crs('EPSG:4326')
    
    # Note: Null geometry removal now handled by validate_and_clean_geometries
    # for polygon data, but keep this for non-polygon data
    if not any(gt in ['Polygon', 'MultiPolygon'] for gt in prepared_gdf.geometry.geom_type.unique()):
        null_mask = prepared_gdf.geometry.isna()
        null_count = null_mask.sum()
        if null_count > 0:
            logging.warning(f"Removing {null_count} null geometries from {layer_name}")
            prepared_gdf = prepared_gdf[~null_mask].copy()
    
    # Note: Invalid geometry repair now handled by validate_and_clean_geometries
    # for polygon data, but keep basic validation for non-polygon data
    if not any(gt in ['Polygon', 'MultiPolygon'] for gt in prepared_gdf.geometry.geom_type.unique()):
        valid_mask = prepared_gdf.geometry.is_valid
        invalid_count = (~valid_mask).sum()
        
        if invalid_count > 0:
            logging.warning(f"Found {invalid_count} invalid geometries in {layer_name} - attempting repair")
            
            # Attempt to repair invalid geometries using buffer(0)
            invalid_geometries = prepared_gdf.loc[~valid_mask, 'geometry']
            repaired_geometries = invalid_geometries.buffer(0)
            
            # Check if repair was successful
            repair_successful = repaired_geometries.is_valid
            successful_repairs = repair_successful.sum()
            
            if successful_repairs > 0:
                prepared_gdf.loc[~valid_mask, 'geometry'] = repaired_geometries
                logging.info(f"Successfully repaired {successful_repairs}/{invalid_count} invalid geometries")
            
            # Remove any geometries that couldn't be repaired
            still_invalid = ~prepared_gdf.geometry.is_valid
            if still_invalid.any():
                remaining_invalid = still_invalid.sum()
                logging.warning(f"Removing {remaining_invalid} geometries that could not be repaired")
                prepared_gdf = prepared_gdf[~still_invalid].copy()
    
    # Enhanced attribute cleaning for AGOL compatibility
    for col in prepared_gdf.columns:
        if prepared_gdf[col].dtype == 'object' and col != 'geometry':
            # Handle problematic data types that cause AGOL publishing failures
            def clean_attribute(x):
                if x is None:
                    return None
                elif isinstance(x, (dict, list)):
                    # Convert complex JSON structures to strings (common in Overture data)
                    return str(x)[:500] if str(x) != '{}' and str(x) != '[]' else None
                elif isinstance(x, str):
                    # Clean problematic characters and limit length
                    cleaned = x.strip()
                    if cleaned in ('{}', '[]', ''):
                        return None
                    return cleaned[:500] if cleaned else None
                else:
                    # Convert other types to string
                    return str(x)[:500] if str(x) else None
            
            prepared_gdf[col] = prepared_gdf[col].apply(clean_attribute)
            
            # Log cleaning statistics
            null_count = prepared_gdf[col].isna().sum()
            total_count = len(prepared_gdf)
            if null_count > 0:
                logging.debug(f"Column '{col}': {null_count}/{total_count} null values after cleaning")
    
    # Final validation with comprehensive diagnostics
    if prepared_gdf.empty:
        logging.warning(f"No valid geometries remaining after preparation for {layer_name}")
    else:
        # Log final geometry type distribution
        final_geom_types = prepared_gdf.geometry.geom_type.value_counts()
        logging.debug(f"{layer_name} final geometry types: {final_geom_types.to_dict()}")
        
        # Additional validation for common AGOL failure points
        null_geom_count = prepared_gdf.geometry.isna().sum()
        if null_geom_count > 0:
            logging.warning(f"Found {null_geom_count} null geometries after preparation")
        
        # Check for extremely small or degenerate geometries
        if any(gt in ['Polygon', 'MultiPolygon'] for gt in prepared_gdf.geometry.geom_type.unique()):
            empty_geom_count = sum(1 for geom in prepared_gdf.geometry if geom.is_empty)
            if empty_geom_count > 0:
                logging.warning(f"Found {empty_geom_count} empty geometries - these may cause publishing issues")
        
        # Validate all required columns exist
        required_cols = ['geometry']
        if 'id' in prepared_gdf.columns:
            required_cols.append('id')
        
        missing_cols = [col for col in required_cols if col not in prepared_gdf.columns]
        if missing_cols:
            logging.error(f"Missing required columns after preparation: {missing_cols}")
            
        logging.info(f"Data preparation completed: {len(prepared_gdf):,} features ready for AGOL")
    
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
            cleanup_success = _cleanup_geojson_item(temp_item)
            if not cleanup_success:
                logging.warning(f"Failed to clean up temporary item: {temp_item.title}")


def _cleanup_geojson_item(item, max_retries: int = 3) -> bool:
    """
    Robust cleanup of temporary GeoJSON items with retry logic.
    
    Args:
        item: GeoJSON item to delete
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if successfully deleted, False otherwise
    """
    import time
    
    if not item:
        return True
        
    for attempt in range(max_retries):
        try:
            # Attempt to delete the item
            result = item.delete()
            
            if result:
                logging.debug(f"Successfully cleaned up GeoJSON item: {item.title} ({item.itemid})")
                
                # Verify item is actually deleted
                try:
                    # Try to access the item - should fail if truly deleted
                    test_item = item.gis.content.get(item.itemid)
                    if test_item is None:
                        return True
                    else:
                        logging.debug(f"Item still exists after delete attempt {attempt + 1}")
                except Exception:
                    # Exception means item not found - good!
                    return True
            
            # If we got here, deletion didn't work
            if attempt < max_retries - 1:
                logging.debug(f"Delete attempt {attempt + 1} failed, retrying in 2 seconds...")
                time.sleep(2)
            
        except Exception as e:
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                # Item already deleted
                logging.debug(f"GeoJSON item already deleted: {item.itemid}")
                return True
            
            if attempt < max_retries - 1:
                logging.debug(f"Delete attempt {attempt + 1} failed with error: {e}, retrying...")
                time.sleep(2)
            else:
                logging.warning(f"Failed to cleanup GeoJSON item after {max_retries} attempts: {e}")
    
    return False


def _cleanup_orphaned_geojson_items(gis, service_name: str) -> int:
    """
    Clean up orphaned GeoJSON items from previous failed attempts.
    
    Args:
        gis: Authenticated GIS connection
        service_name: Service name to search for related orphaned items
        
    Returns:
        Number of items cleaned up
    """
    cleaned_count = 0
    
    try:
        # Search for GeoJSON items that might be orphaned from this service
        # Look for items with the service name or temporary naming patterns
        search_patterns = [
            f'title:*{service_name}* AND type:"GeoJson"',
            f'title:*temp* AND type:"GeoJson" AND tags:"overture-pipeline"',
            f'title:*staging* AND type:"GeoJson" AND owner:{gis.users.me.username}'
        ]
        
        orphaned_items = []
        
        for pattern in search_patterns:
            try:
                results = gis.content.search(
                    query=pattern,
                    max_items=20,
                    outside_org=False
                )
                
                for item in results:
                    # Check if this looks like an orphaned item
                    is_orphaned = False
                    
                    # Check 1: Has temporary naming patterns
                    temp_patterns = ['temp_', 'staging_', '_temp']
                    if any(pattern in item.title.lower() for pattern in temp_patterns):
                        is_orphaned = True
                    
                    # Check 2: Tagged as temporary
                    if hasattr(item, 'tags') and item.tags:
                        item_tags = item.tags if isinstance(item.tags, list) else [item.tags]
                        if 'temporary' in [tag.lower() for tag in item_tags]:
                            is_orphaned = True
                    
                    # Check 3: No related feature services
                    if not is_orphaned:
                        try:
                            related_services = item.related_items(relationship_type='Service2Data', direction='reverse')
                            if not related_services:
                                # GeoJSON with no related feature service might be orphaned
                                # But be more careful - only if it follows our naming pattern
                                if service_name.lower() in item.title.lower():
                                    is_orphaned = True
                        except Exception:
                            pass
                    
                    if is_orphaned and item not in orphaned_items:
                        orphaned_items.append(item)
                        
            except Exception as search_error:
                logging.debug(f"Search pattern failed: {pattern}, error: {search_error}")
        
        # Clean up identified orphaned items
        for item in orphaned_items:
            logging.info(f"Cleaning up orphaned GeoJSON item: {item.title} ({item.itemid})")
            if _cleanup_geojson_item(item):
                cleaned_count += 1
            
        if cleaned_count > 0:
            logging.info(f"Cleaned up {cleaned_count} orphaned GeoJSON items")
        
    except Exception as e:
        logging.warning(f"Error during orphaned item cleanup: {e}")
    
    return cleaned_count


def _build_metadata_dict(tgt):
    """
    Build metadata dictionary from target configuration for item updates.
    
    Uses template-resolved metadata from YAML configuration.
    
    Args:
        tgt: Target configuration object with agol attributes
        
    Returns:
        Dictionary of metadata properties for item.update()
    """
    metadata = {}
    
    # Core template metadata fields
    metadata['title'] = tgt.agol.item_title
    metadata['snippet'] = tgt.agol.snippet
    metadata['description'] = tgt.agol.description
    
    # Handle tags (both list and string formats)
    if isinstance(tgt.agol.tags, list):
        metadata['tags'] = ','.join(tgt.agol.tags)
    else:
        metadata['tags'] = tgt.agol.tags
    
    # Enterprise metadata fields
    metadata['licenseInfo'] = tgt.agol.license_info
    metadata['accessInformation'] = tgt.agol.access_information
    
    return metadata


