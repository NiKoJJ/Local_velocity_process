#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optical_downloader.py
=====================
Universal optical imagery downloader for Sentinel-2 and Landsat 8/9.

Data source priority:
  1. Planetary Computer  (free, no auth required)
  2. AWS S3              (sentinel-cogs: free; usgs-landsat: Requester Pays)

Features
--------
- Sentinel-2 L2A and Landsat Collection 2 L2
- AOI from bbox or shapefile
- Flexible band selection (common names or sensor-native names)
- Optional AOI clipping via gdalwarp /vsis3/ (only downloads needed pixels)
- Cloud cover filtering
- Skip-existing deduplication

Requirements
------------
    pip install pystac-client planetary-computer geopandas shapely
    pip install requests tqdm contextily matplotlib
    # For clipping: GDAL 3.2+ with vsis3 support + AWS CLI

Usage
-----
    from optical_downloader import ImageryDownloader

    dl = ImageryDownloader(output_dir='./images', max_cloud_cover=20)

    results = dl.search(
        sensors=['sentinel-2-l2a', 'landsat-c2-l2'],
        bbox=[151.6, -69.0, 154.2, -68.5],
        date_range='2024-10-01/2025-03-31',
    )

    dl.download(results.head(4), bands=['red', 'green', 'blue', 'nir'], clip=True)
"""

import os
import re
import time
import json
import shutil
import warnings
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, shape, mapping

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import pystac_client
    import planetary_computer
    HAS_PC = True
except ImportError:
    HAS_PC = False
    warnings.warn(
        "pystac-client / planetary-computer not installed. "
        "Install with: pip install pystac-client planetary-computer"
    )

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ── Band configuration ────────────────────────────────────────────────────────
# Structure: common_name → (PC asset key, AWS COG filename)
#
# ⚠️  S2 FIX: Planetary Computer S2 L2A uses NATIVE band names as asset keys
#     (B02, B03, B04…), NOT common names (blue, green, red…).
#     Element84/AWS Earth Search also uses native names (B02 etc.).
#     The old mapping "blue" → ("blue", …) caused "asset 'blue' missing".

BAND_CONFIG: Dict[str, Dict[str, Tuple[str, str]]] = {
    "sentinel-2-l2a": {
        # common_name  →  (PC/E84 asset key,  AWS COG filename)
        "coastal":   ("B01",    "B01.tif"),
        "blue":      ("B02",    "B02.tif"),
        "green":     ("B03",    "B03.tif"),
        "red":       ("B04",    "B04.tif"),
        "rededge1":  ("B05",    "B05.tif"),
        "rededge2":  ("B06",    "B06.tif"),
        "rededge3":  ("B07",    "B07.tif"),
        "nir":       ("B08",    "B08.tif"),    # 10 m broad NIR
        "nir08":     ("B8A",    "B8A.tif"),    # 20 m narrow NIR
        "nir09":     ("B09",    "B09.tif"),
        "swir16":    ("B11",    "B11.tif"),
        "swir22":    ("B12",    "B12.tif"),
        "scl":       ("SCL",    "SCL.tif"),    # Scene Classification Layer
        "visual":    ("visual", "TCI.tif"),    # True-colour (PC uses "visual")
        "aot":       ("AOT",    "AOT.tif"),    # Aerosol optical thickness
        "wvp":       ("WVP",    "WVP.tif"),    # Water vapour
    },
    "landsat-c2-l2": {
        # PC & Element84 Landsat asset keys ARE the common names
        "coastal":        ("coastal",        "SR_B1.TIF"),
        "blue":           ("blue",           "SR_B2.TIF"),
        "green":          ("green",          "SR_B3.TIF"),
        "red":            ("red",            "SR_B4.TIF"),
        "nir08":          ("nir08",          "SR_B5.TIF"),
        "swir16":         ("swir16",         "SR_B6.TIF"),
        "swir22":         ("swir22",         "SR_B7.TIF"),
        "lwir11":         ("lwir11",         "ST_B10.TIF"),
        "qa_pixel":       ("qa_pixel",       "QA_PIXEL.TIF"),
        "qa_radsat":      ("qa_radsat",      "QA_RADSAT.TIF"),
        "atmos_opacity":  ("atmos_opacity",  "SR_ATMOS_OPACITY.TIF"),
        "cloud_qa":       ("cloud_qa",       "SR_CLOUD_QA.TIF"),
    },
}

# Per-collection fallback asset key chains to try (in order) when primary key fails.
# Handles catalog differences between PC, Element84, and older item versions.
ASSET_KEY_FALLBACKS: Dict[str, Dict[str, List[str]]] = {
    "sentinel-2-l2a": {
        "blue":     ["B02", "blue",   "B2"],
        "green":    ["B03", "green",  "B3"],
        "red":      ["B04", "red",    "B4"],
        "nir":      ["B08", "nir",    "B8",  "nir_broad"],
        "nir08":    ["B8A", "nir08",  "B8A"],
        "coastal":  ["B01", "coastal","B1"],
        "rededge1": ["B05", "rededge1","B5"],
        "rededge2": ["B06", "rededge2","B6"],
        "rededge3": ["B07", "rededge3","B7"],
        "nir09":    ["B09", "nir09",  "B9"],
        "swir16":   ["B11", "swir16", "B11"],
        "swir22":   ["B12", "swir22", "B12"],
        "scl":      ["SCL", "scl"],
        "visual":   ["visual", "TCI", "tci"],
        "aot":      ["AOT", "aot"],
        "wvp":      ["WVP", "wvp"],
    },
    "landsat-c2-l2": {
        "coastal":   ["coastal",   "SR_B1", "sr_b1"],
        "blue":      ["blue",      "SR_B2", "sr_b2"],
        "green":     ["green",     "SR_B3", "sr_b3"],
        "red":       ["red",       "SR_B4", "sr_b4"],
        "nir08":     ["nir08",     "SR_B5", "sr_b5"],
        "swir16":    ["swir16",    "SR_B6", "sr_b6"],
        "swir22":    ["swir22",    "SR_B7", "sr_b7"],
        "lwir11":    ["lwir11",    "ST_B10"],
        "qa_pixel":  ["qa_pixel",  "QA_PIXEL"],
    },
}

# Human-friendly aliases → canonical common name
BAND_ALIASES: Dict[str, str] = {
    # Sentinel-2 native names
    "b1": "coastal", "b2": "blue",  "b3": "green", "b4": "red",
    "b5": "rededge1", "b6": "rededge2", "b7": "rededge3",
    "b8": "nir", "b8a": "nir08", "b9": "nir09",
    "b11": "swir16", "b12": "swir22",
    "tci": "visual",
    # Landsat SR band names
    "sr_b1": "coastal", "sr_b2": "blue",  "sr_b3": "green",
    "sr_b4": "red",     "sr_b5": "nir08", "sr_b6": "swir16",
    "sr_b7": "swir22",
    # Generic aliases
    "nir": "nir",       # keep as-is (resolves to B08 for S2)
    "pan": "pan",
}

# Landsat AWS collection path components
LS_INSTRUMENT = {
    "LC08": ("oli-tirs", "LC08"),
    "LC09": ("oli-tirs", "LC09"),
    "LE07": ("etm",      "LE07"),
}


# ── Main class ────────────────────────────────────────────────────────────────

class ImageryDownloader:
    """
    Universal downloader for Sentinel-2 and Landsat imagery.

    Parameters
    ----------
    output_dir : str or Path
    max_cloud_cover : float
        Cloud cover upper limit (0–100).
    aws_profile : str, optional
        AWS CLI profile name. Required for Landsat (Requester Pays).
    use_pc : bool
        Attempt Planetary Computer (default True).
    use_aws : bool
        Use AWS S3 fallback (default True).
    """

    PC_URL     = "https://planetarycomputer.microsoft.com/api/stac/v1"
    ELEM84_URL = "https://earth-search.aws.element84.com/v1"   # free S2 STAC
    S2_BUCKET  = "sentinel-cogs"
    S2_PREFIX  = "sentinel-s2-l2a-cogs"
    LS_BUCKET  = "usgs-landsat"
    LS_PREFIX  = "collection02/level-2/standard"

    def __init__(
        self,
        output_dir:      str   = "./imagery",
        max_cloud_cover: float = 30.0,
        aws_profile:     Optional[str] = None,
        use_pc:          bool  = True,
        use_aws:         bool  = True,
    ):
        self.output_dir      = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_cloud_cover = max_cloud_cover
        self.aws_profile     = aws_profile
        self.use_pc          = use_pc and HAS_PC
        self.use_aws         = use_aws

        self._pc_catalog    = None
        self._elem84_catalog = None
        self._aoi_gdf       = None
        self._resolved_bbox = None
        self._search_results = None

        self._init_sources()

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_sources(self):
        if self.use_pc:
            try:
                self._pc_catalog = pystac_client.Client.open(
                    self.PC_URL,
                    modifier=planetary_computer.sign_inplace,
                )
                print("✓ Planetary Computer connected")
            except Exception as e:
                warnings.warn(f"PC connection failed ({e}). Will fall back to AWS.")

        if self.use_aws and HAS_PC:
            try:
                self._elem84_catalog = pystac_client.Client.open(self.ELEM84_URL)
                print("✓ Element84 Earth Search (AWS S2) connected")
            except Exception:
                pass

    # ── AOI helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_aoi(
        bbox:      Optional[List[float]] = None,
        shapefile: Optional[str]         = None,
    ) -> Tuple[List[float], gpd.GeoDataFrame]:
        """Return (bbox_wgs84, GeoDataFrame) for the AOI."""
        if shapefile is not None:
            gdf    = gpd.read_file(shapefile).to_crs("EPSG:4326")
            bounds = gdf.total_bounds          # [minx, miny, maxx, maxy]
            return list(bounds), gdf
        elif bbox is not None:
            gdf = gpd.GeoDataFrame(
                geometry=[box(*bbox)], crs="EPSG:4326"
            )
            return list(bbox), gdf
        else:
            raise ValueError("Provide either bbox=[minx,miny,maxx,maxy] or shapefile='path.shp'.")

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        sensors:    Union[str, List[str]] = "sentinel-2-l2a",
        bbox:       Optional[List[float]] = None,
        shapefile:  Optional[str]         = None,
        date_range: str  = "2024-01-01/2024-12-31",
        max_items:  int  = 500,
    ) -> gpd.GeoDataFrame:
        """
        Search for available scenes and return a GeoDataFrame with footprints.

        Parameters
        ----------
        sensors : str | list
            'sentinel-2-l2a', 'landsat-c2-l2', or both as a list.
        bbox : [minx, miny, maxx, maxy]
            Search region in WGS84 decimal degrees.
        shapefile : str
            Path to shapefile (alternative to bbox).
        date_range : str
            "YYYY-MM-DD/YYYY-MM-DD"
        max_items : int
            Maximum results per collection.

        Returns
        -------
        gpd.GeoDataFrame
            Columns: id, collection, datetime, cloud_cover, platform,
                     s2:mgrs_tile (S2) / wrs_path+wrs_row (Landsat), geometry
        """
        if isinstance(sensors, str):
            sensors = [sensors]

        self._resolved_bbox, self._aoi_gdf = self._resolve_aoi(bbox, shapefile)
        print(f"🔍 Searching {sensors}  |  {date_range}  |  cloud ≤ {self.max_cloud_cover}%")

        all_records = []

        for collection in sensors:
            if collection not in BAND_CONFIG:
                warnings.warn(f"Unknown collection '{collection}', skipping.")
                continue

            records = self._search_one(collection, date_range, max_items)
            print(f"   {collection}: {len(records)} scenes")
            all_records.extend(records)

        if not all_records:
            print("⚠  No scenes found. Try expanding date range, bbox, or cloud threshold.")
            return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
        gdf["datetime"] = pd.to_datetime(gdf["datetime"], utc=True)
        gdf = gdf.sort_values("datetime").reset_index(drop=True)

        self._search_results = gdf
        print(f"\n✓ Total: {len(gdf)} scenes")
        return gdf

    def _search_one(self, collection, date_range, max_items) -> List[dict]:
        """Search a single collection, trying PC → Element84 → raw AWS."""
        records = []

        # 1) Planetary Computer
        if self.use_pc and self._pc_catalog is not None:
            try:
                sr = self._pc_catalog.search(
                    collections=[collection],
                    bbox=self._resolved_bbox,
                    datetime=date_range,
                    query={"eo:cloud_cover": {"lte": self.max_cloud_cover}},
                    max_items=max_items,
                )
                for item in sr.items():
                    rec = self._item_to_record(item, source="PC")
                    if rec:
                        records.append(rec)
                if records:
                    return records
            except Exception as e:
                warnings.warn(f"PC search error: {e}")

        # 2) Element84 Earth Search (S2 free STAC on AWS)
        if (self.use_aws and self._elem84_catalog is not None
                and collection == "sentinel-2-l2a"):
            try:
                sr = self._elem84_catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=self._resolved_bbox,
                    datetime=date_range,
                    query={"eo:cloud_cover": {"lte": self.max_cloud_cover}},
                    max_items=max_items,
                )
                for item in sr.items():
                    rec = self._item_to_record(item, source="AWS_E84")
                    if rec:
                        records.append(rec)
            except Exception as e:
                warnings.warn(f"Element84 search error: {e}")

        return records

    def _item_to_record(self, item, source="PC") -> Optional[dict]:
        """Convert a pystac Item to a plain dict record."""
        try:
            props = item.properties
            geom  = shape(item.geometry) if isinstance(item.geometry, dict) else item.geometry

            # Store the raw item for download
            return {
                "id":           item.id,
                "collection":   item.collection_id,
                "datetime":     props.get("datetime", ""),
                "cloud_cover":  props.get("eo:cloud_cover", np.nan),
                "platform":     props.get("platform", ""),
                "s2:mgrs_tile": props.get("s2:mgrs_tile", ""),
                "wrs_path":     props.get("landsat:wrs_path", ""),
                "wrs_row":      props.get("landsat:wrs_row", ""),
                "source":       source,
                "_item":        item,     # keep raw item for download
                "geometry":     geom,
            }
        except Exception:
            return None

    # ── download ──────────────────────────────────────────────────────────────

    def download(
        self,
        items,
        bands:            Union[str, List[str]] = ["red", "green", "blue"],
        clip:             bool                  = True,
        bbox:             Optional[List[float]] = None,
        shapefile:        Optional[str]         = None,
        skip_existing:    bool                  = True,
        sleep_between:    float                 = 0.5,
        preferred_source: str                   = "auto",
    ) -> Dict[str, Dict[str, Path]]:
        """
        Download imagery bands for the given items.

        Parameters
        ----------
        items : GeoDataFrame or list of pystac Items
            Output of search() or a filtered subset.
        bands : str | list
            Band common names. Examples:
            - S2:  'blue','green','red','nir','nir08','swir16','swir22','scl','visual'
            - LS:  'blue','green','red','nir08','swir16','swir22','qa_pixel'
        clip : bool
            Clip output to AOI using gdalwarp (only transfers needed pixels).
        bbox, shapefile :
            Override clip AOI (defaults to search AOI).
        skip_existing : bool
            Skip files already present on disk.
        sleep_between : float
            Seconds to pause between item downloads (rate-limit courtesy).
        preferred_source : str
            Force a specific download backend:
            - "auto" — use whatever source the item was found from (default)
            - "pc"   — force Planetary Computer signed URLs
            - "aws"  — force AWS S3 (/vsis3/ for S2, AWS CLI for Landsat)
            Note: "aws" Landsat requires credentials (Requester Pays bucket).

        Returns
        -------
        dict  {item_id: {band_name: Path}}
        """
        if isinstance(bands, str):
            bands = [bands]
        # Normalise band names
        bands = [BAND_ALIASES.get(b.lower(), b.lower()) for b in bands]

        preferred_source = preferred_source.lower()
        if preferred_source not in ("auto", "pc", "aws"):
            raise ValueError("preferred_source must be 'auto', 'pc', or 'aws'")

        # Resolve clip AOI
        clip_gdf = None
        if clip:
            try:
                if bbox or shapefile:
                    _, clip_gdf = self._resolve_aoi(bbox, shapefile)
                elif self._aoi_gdf is not None:
                    clip_gdf = self._aoi_gdf
            except Exception:
                warnings.warn("Could not resolve clip AOI — downloading full scenes.")

        # Accept GeoDataFrame or raw list
        if isinstance(items, gpd.GeoDataFrame):
            item_rows = []
            for _, row in items.iterrows():
                item_id = row.get("id")
                match = self._search_results[self._search_results["id"] == item_id]
                if not match.empty:
                    row_obj = type("R", (), match.iloc[0].to_dict())()
                    item_rows.append(row_obj)
        else:
            item_rows = [type("R", (), {
                "_item": it, "source": getattr(it, "_source", "PC"),
                "collection": it.collection_id
            })() for it in items]

        results = {}
        total   = len(item_rows)

        for i, row in enumerate(item_rows, 1):
            item       = row._item
            source     = row.source          # original source: "PC" / "AWS_E84" / …
            collection = row.collection
            item_id    = item.id

            # ── resolve effective backend ──────────────────────────────────
            if preferred_source == "pc":
                effective = "PC"
                # Re-sign item if needed (PC tokens expire)
                if HAS_PC and self._pc_catalog is not None:
                    try:
                        import planetary_computer as pc_mod
                        item = pc_mod.sign(item)
                    except Exception:
                        pass
            elif preferred_source == "aws":
                effective = "AWS"
            else:
                # auto: S2 from URL (PC signed or Element84 public), LS from AWS CLI
                effective = "URL" if (source.startswith("PC") or source == "AWS_E84") else "AWS"

            print(f"\n[{i}/{total}] {item_id}  [src={source} → backend={effective}]")

            if effective in ("PC", "URL"):
                downloaded = self._dl_from_url(
                    item, bands, collection, clip_gdf, skip_existing, source
                )
            elif effective == "AWS" and collection == "sentinel-2-l2a":
                downloaded = self._dl_from_aws_cli(
                    item, bands, collection, clip_gdf, skip_existing
                )
            else:
                # Landsat: always AWS CLI (Requester Pays)
                downloaded = self._dl_from_aws_cli(
                    item, bands, collection, clip_gdf, skip_existing
                )

            results[item_id] = downloaded
            if sleep_between and i < total:
                time.sleep(sleep_between)

        return results

    # ── download backends ─────────────────────────────────────────────────────

    def _resolve_asset_key(self, item, band: str, collection: str) -> Optional[str]:
        """Find the actual asset key for a band, trying all known fallbacks.

        This handles the catalog inconsistency where:
          - PC S2 L2A uses native names: B02, B03, B04 …
          - Element84 S2 uses native names too but sometimes common names
          - Landsat uses common names: blue, green, red …

        Returns the resolved key, or None if not found.
        """
        available = set(item.assets.keys())

        # Build candidate list: primary key from BAND_CONFIG + all fallbacks
        primary_key = BAND_CONFIG.get(collection, {}).get(band, (band,))[0]
        fallbacks   = ASSET_KEY_FALLBACKS.get(collection, {}).get(band, [])
        candidates  = [primary_key] + [k for k in fallbacks if k != primary_key] + [band]

        for key in candidates:
            if key in available:
                return key

        return None

    def _dl_from_url(self, item, bands, collection, clip_gdf, skip_existing, source):
        """Download using signed HREF (PC) or public HREF (Element84).

        Uses _resolve_asset_key() to handle naming differences across catalogs.
        """
        downloaded = {}

        for band in bands:
            if band not in BAND_CONFIG.get(collection, {}):
                print(f"  ⚠  '{band}' not defined for collection '{collection}'")
                continue

            # ── find actual asset key (multi-fallback) ────────────────────
            resolved_key = self._resolve_asset_key(item, band, collection)
            if resolved_key is None:
                available_keys = list(item.assets.keys())
                print(f"  ✗  '{band}': no matching asset key. "
                      f"Available: {available_keys[:10]}")
                continue

            href     = item.assets[resolved_key].href
            out_path = self.output_dir / f"{item.id}_{band}.tif"

            if skip_existing and out_path.exists():
                print(f"  ✓  {band:12s}  (exists)")
                downloaded[band] = out_path
                continue

            print(f"  ⬇  {band:12s}", end="  ", flush=True)

            if clip_gdf is not None:
                clip_file = self._write_geojson(clip_gdf)
                ok = self._gdalwarp(href, out_path, clip_file)
                clip_file.unlink(missing_ok=True)
            else:
                ok = self._http_get(href, out_path)

            if ok:
                sz = out_path.stat().st_size / 1e6
                print(f"→  {out_path.name}  ({sz:.1f} MB)")
                downloaded[band] = out_path
            else:
                print("FAILED")

        return downloaded

    def _dl_from_aws_cli(self, item, bands, collection, clip_gdf, skip_existing):
        """Download Sentinel-2 directly from sentinel-cogs via AWS CLI."""
        band_config = BAND_CONFIG.get(collection, {})
        downloaded  = {}

        # Build S3 prefix from scene id
        # Scene IDs from elem84: S2B_56DNH_20250223_0_L2A
        # sentinel-cogs path:    sentinel-s2-l2a-cogs/56/D/NH/2025/2/S2B_56DNH_.../
        sid = item.id
        m   = re.match(r"S2[AB]_(\d{2})([A-Z])([A-Z]{2})_(\d{8})_", sid)
        if not m:
            warnings.warn(f"Cannot parse scene id: {sid}")
            return downloaded

        zone, lat_band, square, datestr = m.groups()
        zone_int = int(zone)
        year     = int(datestr[:4])
        month    = int(datestr[4:6])

        s3_base = (f"s3://{self.S2_BUCKET}/{self.S2_PREFIX}/"
                   f"{zone_int}/{lat_band}/{square}/{year}/{month}/{sid}")

        for band in bands:
            if band not in band_config:
                continue
            aws_file = band_config[band][1]
            s3_path  = f"{s3_base}/{aws_file}"
            out_path = self.output_dir / f"{sid}_{band}.tif"

            if skip_existing and out_path.exists():
                print(f"  ✓  {band:12s}  (exists)")
                downloaded[band] = out_path
                continue

            print(f"  ⬇  {band:12s}", end="  ", flush=True)

            if clip_gdf is not None:
                clip_file = self._write_geojson(clip_gdf)
                vsi       = s3_path.replace("s3://", "/vsis3/")
                ok = self._gdalwarp(
                    vsi, out_path, clip_file,
                    extra="--config AWS_NO_SIGN_REQUEST YES"
                )
                clip_file.unlink(missing_ok=True)
            else:
                profile_flag = (f"--profile {self.aws_profile}"
                                if self.aws_profile else "--no-sign-request")
                cmd = f"aws {profile_flag} s3 cp {s3_path} {out_path}"
                ok  = subprocess.run(cmd, shell=True, capture_output=True).returncode == 0

            if ok:
                sz = out_path.stat().st_size / 1e6 if out_path.exists() else 0
                print(f"→  {out_path.name}  ({sz:.1f} MB)")
                downloaded[band] = out_path
            else:
                print("FAILED")

        return downloaded

    # ── utilities ─────────────────────────────────────────────────────────────

    def _write_geojson(self, gdf: gpd.GeoDataFrame) -> Path:
        """Write GeoDataFrame to a temporary GeoJSON for gdalwarp cutline."""
        tmp = self.output_dir / "_tmp_aoi.geojson"
        gdf.to_file(tmp, driver="GeoJSON")
        return tmp

    def _gdalwarp(self, src: str, dst: Path, cutline: Path, extra: str = "") -> bool:
        """Clip-and-download using gdalwarp. Works with /vsis3/ and signed URLs."""
        cmd = (
            f'gdalwarp -overwrite '
            f'-cutline "{cutline}" -crop_to_cutline '
            f'{extra} '
            f'"{src}" "{dst}"'
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            warnings.warn(f"gdalwarp error: {r.stderr[-300:]}")
        return r.returncode == 0

    def _http_get(self, url: str, dst: Path, chunk: int = 65536) -> bool:
        """Stream-download a URL to disk."""
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for ch in r.iter_content(chunk_size=chunk):
                        f.write(ch)
            return True
        except Exception as e:
            warnings.warn(f"HTTP download failed: {e}")
            return False

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot_footprints(
        self,
        results: Optional[gpd.GeoDataFrame] = None,
        color_by:    str   = "cloud_cover",
        figsize:     tuple = (14, 8),
        add_basemap: bool  = True,
        title:       Optional[str] = None,
    ):
        """
        Plot scene footprints coloured by a property.

        Parameters
        ----------
        results : GeoDataFrame
            Output of search(). Uses last search if None.
        color_by : str
            Column for colour scale: 'cloud_cover', 'datetime', 'platform', …
        figsize : tuple
        add_basemap : bool
            Adds Esri NatGeo basemap via contextily (requires internet).
        title : str, optional
        """
        import matplotlib.pyplot as plt

        gdf = results if results is not None else self._search_results
        if gdf is None or gdf.empty:
            print("No search results to plot.")
            return None, None

        # Drop rows without geometry
        gdf = gdf[gdf.geometry.notna()].copy()

        fig, ax = plt.subplots(figsize=figsize)

        # Colour footprints
        cmap = "RdYlGn_r" if color_by == "cloud_cover" else "viridis"
        if color_by in gdf.columns and gdf[color_by].notna().any():
            col = gdf[color_by]
            if pd.api.types.is_datetime64_any_dtype(col):
                gdf = gdf.copy()
                gdf["_color_num"] = col.view("int64")
                color_by_plot = "_color_num"
            else:
                color_by_plot = color_by

            gdf.plot(
                column=color_by_plot,
                ax=ax, alpha=0.35, edgecolor="black", linewidth=0.7,
                legend=True, cmap=cmap,
                legend_kwds={"label": color_by, "shrink": 0.55},
            )
        else:
            gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.8)

        # AOI outline
        if self._aoi_gdf is not None:
            self._aoi_gdf.plot(
                ax=ax, facecolor="none",
                edgecolor="red", linewidth=2.5, linestyle="--",
                label="AOI",
            )
            ax.legend(loc="upper right")

        # Basemap
        if add_basemap:
            try:
                import contextily as ctx
                ctx.add_basemap(
                    ax, crs=gdf.crs.to_string(),
                    source=ctx.providers.Esri.NatGeoWorldMap,
                    attribution_size=5,
                )
            except Exception:
                pass   # contextily optional

        collections = gdf["collection"].unique().tolist()
        _title = title or (
            f"Scene footprints — {', '.join(collections)}\n"
            f"n={len(gdf)}  |  cloud ≤ {self.max_cloud_cover}%"
        )
        ax.set_title(_title, fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        return fig, ax

    def summary_table(
        self,
        results: Optional[gpd.GeoDataFrame] = None,
    ) -> pd.DataFrame:
        """Return a tidy summary table of search results."""
        gdf = results if results is not None else self._search_results
        if gdf is None or gdf.empty:
            return pd.DataFrame()

        cols = ["id", "collection", "datetime", "cloud_cover",
                "platform", "s2:mgrs_tile", "wrs_path", "wrs_row", "source"]
        cols = [c for c in cols if c in gdf.columns]
        df   = gdf[cols].copy()
        if "datetime" in df.columns:
            df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")
        if "cloud_cover" in df.columns:
            df["cloud_cover"] = df["cloud_cover"].round(1)
        return df.reset_index(drop=True)
