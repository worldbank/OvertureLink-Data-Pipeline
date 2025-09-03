# publish.py
# Unified, high‑throughput publisher for ArcGIS Online
# - GPKG staging by default (robust, multi-layer)
# - Server-side append jobs with async + polling
# - Adaptive batching for very large layers (avoids 413s/timeouts)
# - Truncate+Append for overwrite (deterministic)
# - Name-safe matching (sanitizes both sides; strips 'main.')

from __future__ import annotations

import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional
import tempfile
from pathlib import Path

import geopandas as gpd
from arcgis.features import Feature, FeatureLayerCollection

# ----------------------------
# Global knobs (env‑tunable)
# ----------------------------
BATCH_THRESHOLD  = int(os.environ.get("BatchThreshold", "200000"))   # switch to batching above this many records
BATCH_SIZE       = int(os.environ.get("BatchSize", "500000"))        # starting part size per append job
BATCH_MIN        = int(os.environ.get("BatchMin", "50000"))          # floor for adaptive halves
SEED_SIZE        = int(os.environ.get("SeedSize", "2000"))           # small seed to establish schema
APPEND_TIMEOUT_S = int(os.environ.get("AppendTimeout", "14400"))
USE_ASYNC_APPEND = os.environ.get("USE_ASYNC_APPEND", "false").lower() == "true"  # Control async behavior

# Logging defaults
logger = logging.getLogger(__name__)


@dataclass
class StagingFormat:
    GPKG: str = "gpkg"
    GEOJSON: str = "geojson"
    FGDB: str = "fgdb"   # supported by Exporter if you implement it; default remains GPKG


class FeatureLayerManager:
    """
    Unified publisher for single or multi-layer Hosted Feature Layers.

    Usage:
      - Call publish_multi_layer_service(layer_data=..., service_name=..., metadata=..., mode="initial|overwrite|append")
      - `layer_data` can be a GeoDataFrame (single-layer) or dict[name -> GeoDataFrame] (multi-layer)
      - Creation/discovery of the service item should be handled by your existing pipeline,
        which passes the target FeatureLayerCollection (flc) when updating, or lets this
        function create new on initial if you prefer (optional stub provided).
    """

    def __init__(self, gis, mode: str = "auto", use_async: Optional[bool] = None):
        self.gis = gis
        self.mode = mode
        # CLI argument overrides environment variable
        self.use_async = use_async if use_async is not None else USE_ASYNC_APPEND

    # ----------------------------
    # Name + geometry helpers
    # ----------------------------
    def _sanitize_layer_name(self, name: str) -> str:
        """Stable AGOL-safe name for services and sublayers; strips 'main.' and punctuation."""
        if not name:
            return "layer"
        name = name.lower().strip()
        if name.startswith("main."):
            name = name[5:]
        name = name.replace(".", "_").replace("-", "_").replace(" ", "_")
        name = re.sub(r"[^a-z0-9_]", "", name)
        return name[:30]

    def _ensure_geodataframe_with_geometry(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Guarantee GeoDataFrame with active geometry and EPSG:4326; drop empties."""
        if not isinstance(df, gpd.GeoDataFrame):
            df = gpd.GeoDataFrame(df)

        geom_col = getattr(df, "_geometry_column_name", None)
        if not geom_col or geom_col not in df.columns:
            candidates = [c for c in df.columns if c.lower() in ("geometry", "geom", "the_geom", "wkb_geometry", "shape")]
            if not candidates:
                raise ValueError("No geometry column found.")
            df = df.set_geometry(candidates[0])

        # rename to conventional "geometry" where possible
        try:
            if df.geometry.name != "geometry":
                df = df.rename_geometry("geometry")
        except Exception:
            df = df.set_geometry(df.geometry.name)

        # drop invalid/empty
        df = df[df.geometry.notna() & ~df.geometry.is_empty].copy()

        # CRS → EPSG:4326
        if df.crs is None:
            df = df.set_crs(4326)
        else:
            try:
                if df.crs.to_epsg() != 4326:
                    df = df.to_crs(4326)
            except Exception:
                df = df.to_crs(4326)

        return df

    def _create_service_if_missing(
        self,
        service_name: str,
        layer_data_copy: dict[str, gpd.GeoDataFrame],
        metadata: dict | None = None,
        staging_format: str = "gpkg",
    ):
        """
        Create a new Hosted Feature Layer from a staged multi-layer file (GPKG by default),
        seeding each sublayer with a small sample to establish schema. Returns (item, flc).
        """
        if staging_format.lower() not in ("gpkg",):
            # For initial creation we stick to GPKG; once published we can append via any format.
            staging_format = "gpkg"

        # sanitize function for sublayer names we write
        def _safe(n: str) -> str:
            return self._sanitize_layer_name(n)

        # build a temporary multi-layer GeoPackage seeded with 1..SEED_SIZE rows per non-empty layer
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gpkg_path = tmpdir / f"{self._sanitize_layer_name(service_name)}.gpkg"

            wrote_any = False
            for raw_name, gdf in layer_data_copy.items():
                gdf = self._ensure_geodataframe_with_geometry(gdf)
                if len(gdf) == 0:
                    continue
                seed = gdf.iloc[: max(1, min(SEED_SIZE, len(gdf)))].copy()
                # Write/append layer table into the same GPKG
                seed.to_file(gpkg_path, layer=_safe(raw_name), driver="GPKG")
                wrote_any = True

            if not wrote_any:
                raise RuntimeError("No non-empty layers to seed; cannot create a new service.")

            # Prepare item properties
            item_title = (metadata or {}).get("title") or service_name
            item_props = {
                "type": "GeoPackage",
                "title": item_title,
                "tags": (metadata or {}).get("tags", "overture-pipeline,auto-created"),
                "snippet": (metadata or {}).get("snippet", ""),
                "description": (metadata or {}).get("description", ""),
            }

            folder_id = (metadata or {}).get("folder") or (metadata or {}).get("folder_id")

            # Upload the GPKG as a temporary item in AGOL
            if folder_id:
                src_item = self.gis.content.add(item_properties=item_props, data=str(gpkg_path), folder=folder_id)
            else:
                src_item = self.gis.content.add(item_properties=item_props, data=str(gpkg_path))

            if not src_item:
                raise RuntimeError("Failed to upload staging GeoPackage for initial publish.")

            # Publish the uploaded GPKG into a Hosted Feature Layer (multi-layer)
            published = None
            try:
                published = src_item.publish()
                flc = FeatureLayerCollection.fromitem(published)
                return published, flc
            finally:
                # Clean up the uploaded source GPKG item; the published HFL remains
                try:
                    src_item.delete()
                except Exception:
                    pass
    def _normalize_tags(self, tags) -> list[str] | None:
        """Accepts list/tuple/set or comma-separated string; returns a clean list or None."""
        if tags is None:
            return None
        if isinstance(tags, (list, tuple, set)):
            return [str(t).strip() for t in tags if str(t).strip()]
        if isinstance(tags, str):
            parts = [p.strip() for p in tags.split(",")]
            return [p for p in parts if p]
        return None

    def _update_item_metadata(self, item, metadata: dict | None) -> bool:
        """
        Update the AGOL item's title/snippet/description/tags if provided in metadata.
        Returns True if an update was sent.
        """
        if not isinstance(metadata, dict):
            return False

        desired_title = metadata.get("title")
        desired_snippet = metadata.get("snippet")
        desired_desc = metadata.get("description") or metadata.get("desc")
        desired_tags = self._normalize_tags(metadata.get("tags"))

        props = {}
        # Title
        if desired_title and desired_title != (item.title or ""):
            props["title"] = desired_title
        # Snippet
        if desired_snippet and desired_snippet != (getattr(item, "snippet", "") or ""):
            props["snippet"] = desired_snippet
        # Description
        if desired_desc and desired_desc != (getattr(item, "description", "") or ""):
            props["description"] = desired_desc
        # Tags
        if desired_tags is not None:
            current_tags = list(getattr(item, "tags", []) or [])
            if set(map(str, current_tags)) != set(map(str, desired_tags)):
                props["tags"] = desired_tags

        if not props:
            return False

        item.update(item_properties=props)
        logging.info(f"Updated item metadata fields: {', '.join(props.keys())}")
        return True

    # ----------------------------
    # Append via staged item (server-side)
    # ----------------------------
    def _target_append_fields(self, feature_layer, source_cols: list[str]) -> list[str]:
        reserved = {"OBJECTID", "objectid", "Shape", "shape", "GlobalID", "globalid", "geometry"}
        t_fields = [f["name"] for f in feature_layer.properties.fields]
        return [c for c in source_cols if c in t_fields and c not in reserved]

    def _poll_append_job(self, job, timeout_s: int) -> dict:
        start = time.time()
        while True:
            # Check if job has completed
            if hasattr(job, "done"):
                try:
                    if job.done():
                        return job.result() or {"success": True}
                except Exception as e:
                    logger.debug(f"Error checking job status: {e}")
                    # If we can't check status, wait a bit and retry
                    time.sleep(2)
                    continue
            
            # For synchronous jobs or jobs without done() method
            if hasattr(job, "status"):
                if job.status in ["completed", "succeeded", "CompletedSuccessfully"]:
                    return {"success": True}
                elif job.status in ["failed", "CompletedWithErrors"]:
                    raise RuntimeError(f"Append job failed with status: {job.status}")
            
            # Check timeout
            if time.time() - start > timeout_s:
                raise RuntimeError(f"Append job timed out after {timeout_s}s")
            
            time.sleep(2)

    def _append_via_item_hardened(
        self,
        feature_layer,
        gdf: gpd.GeoDataFrame,
        staging_format: str = StagingFormat.GPKG,
        use_async: Optional[bool] = None
    ) -> bool:
        """
        Stage gdf as temp item (GPKG/GeoJSON) and call feature_layer.append(item_id=...).
        Cleans up the temp item on success/failure.
        """
        # Import Exporter (support both project layouts)
        try:
            from .export import Exporter                   # layout A
        except Exception:
            try:
                from .pipeline.export import Exporter      # layout B
            except Exception as e:
                raise ImportError("Exporter not found. Ensure export.py is importable.") from e

        exporter = Exporter()
        layer_name = "features"

        fmt = staging_format.lower()
        if fmt == "gpkg":
            staging_file = exporter.create_staging_file(gdf, layer_name, "gpkg")
            item_type, upload_format, source_table_name = "GeoPackage", "geoPackage", layer_name
        elif fmt == "geojson":
            staging_file = exporter.create_staging_file(gdf, layer_name, "geojson")
            item_type, upload_format, source_table_name = "GeoJson", "geojson", layer_name
        elif fmt == "fgdb":
            staging_file = exporter.create_staging_file(gdf, layer_name, "fgdb")
            item_type, upload_format, source_table_name = "File Geodatabase", "filegdb", layer_name
        else:
            raise ValueError(f"Unsupported staging_format: {staging_format}")

        temp_item = None
        try:
            temp_title = f"o2agol_append_{uuid.uuid4().hex[:8]}"
            temp_item = self.gis.content.add(
                {"type": item_type, "title": temp_title, "tags": "temporary,overture-pipeline"},
                data=staging_file.name
            )
            if not temp_item:
                raise RuntimeError("Failed to create staging item")

            append_fields = self._target_append_fields(feature_layer, [c for c in gdf.columns if c != "geometry"])

            # Use parameter if provided, otherwise use instance setting
            actual_use_async = use_async if use_async is not None else self.use_async
            
            logger.info(f"Starting append operation with temp item {temp_item.id} ({len(gdf)} features)")
            
            job = feature_layer.append(
                item_id=temp_item.id,
                upload_format=upload_format,
                source_table_name=source_table_name,
                append_fields=append_fields,
                upsert=False,
                skip_updates=False,
                skip_inserts=False,
                update_geometry=True,
                rollback=True,
                return_messages=True,
                future=actual_use_async
            )

            if actual_use_async:
                logger.info(f"Async append started, polling for completion (timeout: {APPEND_TIMEOUT_S}s)")
                try:
                    self._poll_append_job(job, timeout_s=APPEND_TIMEOUT_S)
                    logger.info("Append job completed successfully")
                except Exception as e:
                    logger.error(f"Append job failed or timed out: {e}")
                    raise
            else:
                logger.info("Synchronous append completed")

            return True

        finally:
            # Clean up temp item from AGOL
            if temp_item:
                try:
                    logger.debug(f"Deleting temp item {temp_item.id} from AGOL")
                    temp_item.delete()
                    logger.debug("Temp item deleted successfully")
                except Exception as e:
                    logger.debug(f"Failed to delete temp item: {e}")
            
            # Clean up local staging file
            try:
                if staging_file and hasattr(staging_file, 'name'):
                    logger.debug(f"Removing local staging file {staging_file.name}")
                    os.unlink(staging_file.name)
            except Exception as e:
                logger.debug(f"Failed to remove staging file: {e}")

    def _append_via_batches(
        self,
        feature_layer,
        gdf: gpd.GeoDataFrame,
        batch_size: int,
        staging_format: str = StagingFormat.GPKG
    ) -> None:
        """Append large data in big server-side jobs, adapting on payload/time errors."""
        total = len(gdf)
        if total == 0:
            logger.info("No features to append (empty dataframe).")
            return

        idx = gdf.index.to_list()
        start = 0
        part = 0
        bs = max(1, batch_size)

        while start < total:
            end = min(start + bs, total)
            part += 1
            part_gdf = gdf.loc[idx[start:end]]

            logger.info(f"[append] part {part}: {len(part_gdf):,} features (window {start}:{end}, batch={bs:,})")
            try:
                self._append_via_item_hardened(feature_layer, part_gdf, staging_format=staging_format, use_async=self.use_async)
                start = end

            except Exception as e:
                msg = str(e)
                # adapt on payload/time errors
                if any(s in msg for s in ("413", "Request Entity Too Large", "timed out", "504", "502")) and bs > BATCH_MIN:
                    new_bs = max(BATCH_MIN, bs // 2)
                    logger.warning(f"Append part failed: {msg[:160]}... reducing batch {bs:,}→{new_bs:,} and retrying the same window.")
                    bs = new_bs
                    time.sleep(1.0)
                    continue
                raise

    # ----------------------------
    # Initial / Overwrite / Append
    # ----------------------------
    def _initial_with_seed_and_append(self, feature_layer, prepared_gdf: gpd.GeoDataFrame) -> None:
        total = len(prepared_gdf)
        if total == 0:
            logger.info("Nothing to publish (no features).")
            return

        seed_count = min(SEED_SIZE, total)
        seed = prepared_gdf.iloc[:seed_count].copy()
        rest = prepared_gdf.iloc[seed_count:].copy()

        logger.info(f"Seeding {seed_count:,} features to initialize schema...")
        self._append_via_item_hardened(feature_layer, seed, staging_format=StagingFormat.GPKG, use_async=self.use_async)

        if len(rest) > 0:
            if len(rest) >= BATCH_THRESHOLD:
                logger.info(f"Bulk appending remaining {len(rest):,} features in batches...")
                self._append_via_batches(feature_layer, rest, batch_size=BATCH_SIZE, staging_format=StagingFormat.GPKG)
            else:
                logger.info(f"Appending remaining {len(rest):,} features in a single job...")
                self._append_via_item_hardened(feature_layer, rest, staging_format=StagingFormat.GPKG, use_async=self.use_async)

    def publish_or_update(
        self,
        feature_layer,
        prepared_gdf: gpd.GeoDataFrame,
        mode: str = "auto"
    ) -> None:
        """
        Unified orchestrator per sublayer:
          - initial: seed + (batched) append
          - overwrite: truncate + (batched) append
          - append: (batched) append
        """
        gdf = self._ensure_geodataframe_with_geometry(prepared_gdf)

        m = (mode or self.mode or "auto").lower()
        if m == "initial":
            self._initial_with_seed_and_append(feature_layer, gdf)
            return

        if m == "overwrite":
            logger.info("Truncating layer prior to append (overwrite mode)...")
            feature_layer.manager.truncate()
            # big vs small
            if len(gdf) >= BATCH_THRESHOLD:
                self._append_via_batches(feature_layer, gdf, batch_size=BATCH_SIZE, staging_format=StagingFormat.GPKG)
            else:
                self._append_via_item_hardened(feature_layer, gdf, staging_format=StagingFormat.GPKG, use_async=self.use_async)
            return

        if m == "auto":
            logger.info("Truncating layer prior to append (auto mode on existing service)...")
            feature_layer.manager.truncate()
            
        if len(gdf) >= BATCH_THRESHOLD:
            self._append_via_batches(feature_layer, gdf, batch_size=BATCH_SIZE, staging_format=StagingFormat.GPKG)
        else:
            self._append_via_item_hardened(feature_layer, gdf, staging_format=StagingFormat.GPKG, use_async=self.use_async)

    # ----------------------------
    # Multi-layer entrypoint
    # ----------------------------
    def publish_multi_layer_service(
        self,
        layer_data: gpd.GeoDataFrame | Mapping[str, gpd.GeoDataFrame],
        service_name: str,
        metadata: Dict[str, Any] | None = None,
        mode: str = "auto",
        flc: FeatureLayerCollection | None = None,
        staging_format: str = StagingFormat.GPKG,
    ) -> str:
        """
        Unified multi-layer publisher.
        - layer_data: GeoDataFrame (treated as {service_name: gdf}) or dict[name -> gdf]
        - service_name: target service title (used for discovery if flc None)
        - flc: optional FeatureLayerCollection for existing service (skip discovery if provided)
        Returns the item id of the Feature Service.
        """

        # Normalize input to dict
        if isinstance(layer_data, gpd.GeoDataFrame):
            layer_data = {service_name: layer_data}

        # Sanitize our own layer keys for stability
        layer_data_copy: Dict[str, gpd.GeoDataFrame] = {
            self._sanitize_layer_name(k): v for k, v in layer_data.items()
        }

        # --- DISCOVERY: resolve existing service OR create it -----------------
        item = None

        # If caller provided a collection, keep it and load the item
        if flc is not None:
            item = flc.properties.get("serviceItemId")
            if isinstance(item, str):
                item = self.gis.content.get(item)

        # Accept explicit item id in metadata (fastest & safest)
        meta_id = None
        if isinstance(metadata, dict):
            meta_id = metadata.get("item_id") or metadata.get("service_id") or metadata.get("id")
        if item is None and meta_id:
            it = self.gis.content.get(str(meta_id))
            if not it:
                raise RuntimeError(f"Could not load Feature Service by id: {meta_id}")
            item = it
            flc = FeatureLayerCollection.fromitem(item)

        # Build expected human title and URL-safe service names
        meta_title = (metadata or {}).get("title")
        expected_titles = [t.strip() for t in [service_name, meta_title] if t]
        def _url_name(s: str) -> str:
            s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip())
            return s.strip("_")
        expected_url_names = { _url_name(t) for t in expected_titles }

        # Exact-match search by title (owner-agnostic but type-filtered)
        if item is None and expected_titles:
            def _search_exact(t: str):
                # Esri search is fuzzy; we post-filter exact title equality.
                res = self.gis.content.search(query=f'title:"{t}" AND type:"Feature Service"', max_items=100) or []
                return [it for it in res if (it.title or "").strip() == t]
            candidates = []
            for t in expected_titles:
                candidates.extend(_search_exact(t))
            # de-dup
            by_id = {}
            for it in candidates:
                by_id.setdefault(it.id, it)
            candidates = list(by_id.values())

            # If we found exact title matches, pick the newest one
            if candidates:
                item = max(candidates, key=lambda it: int(getattr(it, "modified", 0)) if getattr(it, "modified", None) else 0)
                try:
                    flc = FeatureLayerCollection.fromitem(item)
                except Exception:
                    item = None  # fall through to create

        # Fallback: look for exact URL 'name' match only (no fuzzy title)
        if item is None and expected_url_names:
            try:
                all_fs = self.gis.content.search(query='type:"Feature Service"', max_items=200) or []
            except Exception:
                all_fs = []
            def _item_url_name(it):
                n = getattr(it, "name", None)
                if n:
                    return str(n)
                u = getattr(it, "url", "") or ""
                m = re.search(r"/services/([^/]+)/FeatureServer", u, re.IGNORECASE)
                return m.group(1) if m else ""
            exact_url = [it for it in all_fs if _item_url_name(it) in expected_url_names]
            if exact_url:
                item = max(exact_url, key=lambda it: int(getattr(it, "modified", 0)) if getattr(it, "modified", None) else 0)
                try:
                    flc = FeatureLayerCollection.fromitem(item)
                except Exception:
                    item = None

        # If still not found and mode allows, auto-create the service
        desired_mode = (mode or self.mode or "auto").lower()
        if item is None or flc is None:
            if desired_mode in ("auto", "initial"):
                logger.info(f"Feature Service '{service_name}' not found. Creating a new service (auto-initial).")
                item, flc = self._create_service_if_missing(
                    service_name=service_name,
                    layer_data_copy=layer_data_copy,
                    metadata=metadata or {},
                    staging_format=staging_format,
                )
            else:
                raise RuntimeError(f"Feature Service '{service_name}' not found. Create it before publishing.")

        existing_id = item.id
        # --- END DISCOVERY ----------------------------------------------------


        # Keep the AGOL item metadata (title/snippet/description/tags) in sync with this run's metadata
        try:
            self._update_item_metadata(item, metadata or {})
        except Exception as e:
            logging.warning(f"Skipping item metadata update: {e}")

        # Map AGOL sublayers by sanitized name to be format-agnostic (e.g., 'main.*')
        target_layers_by_name = {
            self._sanitize_layer_name(lyr.properties.name): lyr
            for lyr in flc.layers
        }

        # Iterate each sublayer and publish/update according to mode
        for raw_name, gdf in layer_data_copy.items():
            key = self._sanitize_layer_name(raw_name)
            if key not in target_layers_by_name:
                raise RuntimeError(
                    f"Target sublayer '{raw_name}' not found. "
                    f"Available (sanitized): {list(target_layers_by_name.keys())}"
                )

            target_layer = target_layers_by_name[key]
            self.publish_or_update(target_layer, gdf, mode=mode)

        return existing_id
    
    def close(self):
        """Clean up GIS connection resources."""
        if hasattr(self, 'gis') and self.gis:
            try:
                # Force close any active GIS sessions
                if hasattr(self.gis, '_portal'):
                    portal = self.gis._portal
                    
                    # Close the session if it exists
                    if hasattr(portal, '_session') and portal._session:
                        try:
                            portal._session.close()
                            logger.debug("Closed GIS portal session")
                        except Exception:
                            pass
                    
                    # Clear any connection pools
                    if hasattr(portal, 'con') and hasattr(portal.con, '_session'):
                        try:
                            portal.con._session.close()
                            logger.debug("Closed portal connection session")
                        except Exception:
                            pass
                    
                    # Set to None to release references
                    self.gis._portal = None
                
                # Clear the GIS object reference
                self.gis = None
                logger.debug("GIS connection fully cleaned up")
                
            except Exception as e:
                logger.debug(f"Error during GIS cleanup: {e}")
                # Still try to clear the reference
                try:
                    self.gis = None
                except Exception:
                    pass
