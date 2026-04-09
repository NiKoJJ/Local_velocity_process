#!/usr/bin/env python3
"""
gamma_preprocess.py  v2
=======================
Convert raw GAMMA POT offset-tracking outputs (range + azimuth displacement
maps) into the Vx / Vy / V velocity TIFs expected by the pipeline.

New in v2
---------
  1. Single global lv_phi.tif  (--lv-phi-file)
     One reference geometry file used for all pairs; skips per-date lookup.
     Its grid (dimensions / transform / CRS) becomes the reference grid.

  2. Auto-resample to lv_phi grid
     Range and azimuth arrays are reprojected to exactly match lv_phi
     before conversion, regardless of their original resolution or CRS.

  3. Pre-filter raw offsets  (--prefilter umt | mad | none)
     Spatial outlier filter applied to D_range / D_az BEFORE the rotation.
     A pixel flagged bad in either component is NaN-ed in both.
       umt — Universal Median Test (Westerweel & Scarano 2005)
       mad — Sliding-window Median + σ threshold  (paper Stage-1 logic)

Processing order per pair
--------------------------
  read range, azimuth
  → resample both to lv_phi reference grid   (bilinear)
  → pre-filter (umt | mad | none)            on raw displacements (m)
  → unit conversion  (m → m/yr)             D / Δt × 365.25
  → rotation to East/North                  using lv_phi
  → write Vx, Vy, V

Coordinate convention
---------------------
  lv_phi (GAMMA)  :  0 = East,  π/2 = North,  counter-clockwise, radians
  Rotation:
      Vx =  V_range · cos(α) − V_az · sin(α)
      Vy =  V_range · sin(α) + V_az · cos(α)
      V  =  √(Vx² + Vy²)

Output naming  (pipeline-compatible)
--------------------------------------
  <output_dir>/yyyymmdd-yyyymmdd-Vx.tif
  <output_dir>/yyyymmdd-yyyymmdd-Vy.tif
  <output_dir>/yyyymmdd-yyyymmdd-V.tif

Usage examples
--------------
# Single shared lv_phi + MAD pre-filter
python gamma_preprocess.py \\
    --input-root /data2/.../Cook_120m \\
    --output-dir /data2/.../velocity \\
    --lv-phi-file /data2/.../geometry/20220109.lv_phi.tif \\
    --prefilter mad \\
    --mad-window 5 --mad-sigma 5.0

# Single shared lv_phi + UMT pre-filter
python gamma_preprocess.py \\
    --input-root /data2/.../Cook_120m \\
    --output-dir /data2/.../velocity \\
    --lv-phi-file /data2/.../geometry/20220109.lv_phi.tif \\
    --prefilter umt \\
    --umt-window 5 --umt-eps 0.1 --umt-threshold 3.0 --umt-min-segment 25

# Per-date lv_phi from geometry dir (original behaviour) + MAD filter
python gamma_preprocess.py \\
    --input-root /data2/.../Cook_120m \\
    --output-dir /data2/.../velocity \\
    --geom-dir   /data2/.../geometry \\
    --prefilter mad --mad-sigma 5.0 \\
    --workers 8
"""

from __future__ import annotations
import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

log = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.25
_NUMERIC_FILL = np.float32(-32768.0)   # sentinel during bilinear reproject


# =============================================================================
# I/O helpers
# =============================================================================

def _read_tif(path: Path) -> Tuple[np.ndarray, rasterio.Affine, CRS]:
    """
    Read first band of a GeoTIFF as float32.
    All nodata / non-finite values are replaced with NaN.
    Returns (data, transform, crs).
    """
    with rasterio.open(str(path)) as src:
        data = src.read(1).astype(np.float32)
        tf   = src.transform
        crs  = src.crs
        nd   = src.nodata
    if nd is not None:
        data[data == float(nd)] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data, tf, crs


def _write_tif(
    path:      Path,
    data:      np.ndarray,
    transform: rasterio.Affine,
    crs:       CRS,
    nodata:    float = float("nan"),
) -> None:
    """Write a float32 single-band GeoTIFF (deflate, tiled 256×256)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = data.astype(np.float32)
    if np.isfinite(nodata):
        arr[~np.isfinite(arr)] = nodata
    with rasterio.open(
        str(path), "w", driver="GTiff", dtype="float32",
        count=1, height=arr.shape[0], width=arr.shape[1],
        crs=crs, transform=transform, nodata=nodata,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(arr, 1)


# =============================================================================
# Resampling — resample src array to reference grid
# =============================================================================

def _resample_to_ref(
    src_arr:       np.ndarray,
    src_transform: rasterio.Affine,
    src_crs:       CRS,
    ref_transform: rasterio.Affine,
    ref_h:         int,
    ref_w:         int,
    ref_crs:       CRS,
) -> np.ndarray:
    """
    Reproject src_arr onto the reference grid using bilinear resampling.

    NaN is replaced by a numeric sentinel before reprojection and restored
    afterwards so bilinear interpolation does not bleed fill into valid pixels.

    If source and reference grids already match (same shape + transform + CRS),
    returns a copy without reprojecting.

    Returns (ref_h, ref_w) float32, NaN where no valid source data.
    """
    if (src_arr.shape == (ref_h, ref_w)
            and src_transform == ref_transform
            and src_crs.to_epsg() == ref_crs.to_epsg()):
        return src_arr.copy()

    filled = np.where(np.isfinite(src_arr), src_arr, _NUMERIC_FILL)
    dst    = np.full((ref_h, ref_w), _NUMERIC_FILL, dtype=np.float32)

    reproject(
        source        = filled,
        destination   = dst,
        src_transform = src_transform,
        src_crs       = src_crs,
        dst_transform = ref_transform,
        dst_crs       = ref_crs,
        resampling    = Resampling.bilinear,
        src_nodata    = float(_NUMERIC_FILL),
        dst_nodata    = float(_NUMERIC_FILL),
    )
    dst[dst == _NUMERIC_FILL] = np.nan
    dst[~np.isfinite(dst)]    = np.nan
    return dst


# =============================================================================
# lv_phi loading
# =============================================================================

def _load_lv_phi_global(
    path: Path,
) -> Tuple[np.ndarray, rasterio.Affine, CRS]:
    """Load a single shared lv_phi file.  Its grid becomes the reference."""
    arr, tf, crs = _read_tif(path)
    log.info("Global lv_phi: %s  shape=%s", path.name, arr.shape)
    return arr, tf, crs


def _load_lv_phi_per_date(
    geom_dir:      Path,
    date1_str:     str,
    date2_str:     str,
    lv_suffix:     str,
    ref_transform: rasterio.Affine,
    ref_h:         int,
    ref_w:         int,
    ref_crs:       CRS,
) -> Optional[np.ndarray]:
    """
    Load and resample lv_phi for a pair from the geometry directory.

    Search: date1 → date2 → mean(date1, date2).
    Each candidate is resampled to the reference grid if its shape differs.
    Returns (ref_h, ref_w) float32 or None.
    """
    candidates = []
    for ds in (date1_str, date2_str):
        p = geom_dir / f"{ds}{lv_suffix}"
        if p.exists():
            try:
                arr, tf_s, crs_s = _read_tif(p)
                arr = _resample_to_ref(
                    arr, tf_s, crs_s,
                    ref_transform, ref_h, ref_w, ref_crs)
                candidates.append(arr)
            except Exception as exc:
                log.warning("  lv_phi read error %s: %s", p.name, exc)

    if not candidates:
        log.warning("  No lv_phi for %s or %s in %s",
                    date1_str, date2_str, geom_dir)
        return None
    if len(candidates) == 2:
        return np.nanmean(np.stack(candidates, 0), axis=0).astype(np.float32)
    return candidates[0]


# =============================================================================
# Pre-filters  (applied to raw displacement arrays, units = metres)
# =============================================================================

# ── Shared helper ─────────────────────────────────────────────────────────────

def _neighbour_stack(arr: np.ndarray, ws: int) -> np.ndarray:
    """
    Return (ws²-1, H, W) float32 stack of neighbourhood values for every
    pixel, centre pixel excluded, boundary padded with NaN.
    """
    H, W = arr.shape
    pad  = ws // 2
    padded = np.pad(arr.astype(np.float32), pad,
                    mode="constant", constant_values=np.nan)
    nb = []
    for dy in range(ws):
        for dx in range(ws):
            if dy == pad and dx == pad:
                continue
            nb.append(padded[dy:dy + H, dx:dx + W])
    return np.stack(nb, axis=0)


# ── UMT ───────────────────────────────────────────────────────────────────────

def _umt_bad_mask(
    arr:       np.ndarray,
    ws:        int,
    eps:       float,
    threshold: float,
) -> np.ndarray:
    """Boolean (H,W) mask: True = pixel fails UMT."""
    nb        = _neighbour_stack(arr, ws)
    nb_med    = np.nanmedian(nb, axis=0)
    resid_med = np.nanmedian(np.abs(nb - nb_med[np.newaxis]), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.abs(arr - nb_med) / (resid_med + np.float32(eps))
    return np.isfinite(arr) & np.isfinite(nb_med) & (score > threshold)


def _remove_small_segments(valid: np.ndarray, min_size: int) -> np.ndarray:
    """Zero out connected components smaller than min_size. Returns bool mask."""
    if min_size <= 1 or not np.any(valid):
        return valid
    from scipy.ndimage import label as nd_label
    lbl, n = nd_label(valid, structure=np.ones((3, 3), np.int8))
    if n == 0:
        return valid
    sizes = np.bincount(lbl.ravel())
    small = sizes < min_size
    small[0] = False
    out = valid.copy()
    out[small[lbl]] = False
    return out


def apply_umt(
    d_rg:        np.ndarray,
    d_az:        np.ndarray,
    ws:          int   = 5,
    eps:         float = 0.1,
    threshold:   float = 3.0,
    min_segment: int   = 25,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Universal Median Test on raw range / azimuth displacement maps.

    A pixel is removed from BOTH components when it fails UMT in range OR
    azimuth.  Small isolated valid-pixel segments are also removed.

    Parameters
    ----------
    ws          : odd window side (pixels), default 5
    eps         : normalisation floor, default 0.1
    threshold   : score threshold — lower = stricter (default 3.0)
    min_segment : min connected-component size to keep (pixels)

    Returns
    -------
    d_rg_f, d_az_f : filtered copies (NaN at removed pixels)
    stats          : diagnostic dict
    """
    bad = _umt_bad_mask(d_rg, ws, eps, threshold) | \
          _umt_bad_mask(d_az, ws, eps, threshold)

    d_rg_f = d_rg.copy(); d_rg_f[bad] = np.nan
    d_az_f = d_az.copy(); d_az_f[bad] = np.nan
    n_out  = int(bad.sum())

    valid  = np.isfinite(d_rg_f) & np.isfinite(d_az_f)
    valid2 = _remove_small_segments(valid, min_segment)
    small  = valid & ~valid2
    n_sml  = int(small.sum())
    d_rg_f[small] = np.nan
    d_az_f[small] = np.nan

    n_in  = int((np.isfinite(d_rg) & np.isfinite(d_az)).sum())
    total = n_out + n_sml
    pct   = round(100.0 * total / n_in, 2) if n_in else 0.0
    stats = dict(n_input_valid=n_in, n_removed=total, removal_pct=pct,
                 n_umt_outliers=n_out, n_small_seg=n_sml)
    log.debug("  UMT(ws=%d thr=%.1f eps=%.2f): removed %d outlier + %d small "
              "(%.1f%%)", ws, threshold, eps, n_out, n_sml, pct)
    return d_rg_f, d_az_f, stats


# ── MAD sliding-window filter ─────────────────────────────────────────────────

def _window_stats(
    arr:       np.ndarray,
    ws:        int,
    min_valid: int,
    tile_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pixel sliding-window median, population std, and MAD.
    Processed in horizontal tiles of `tile_rows` rows to cap peak memory.

    Returns median, std, mad — each (H,W) float32, NaN where window sparse.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    arr  = arr.astype(np.float32, copy=False)
    H, W = arr.shape
    pad  = ws // 2
    padded = np.pad(arr, pad, mode="constant", constant_values=np.nan)

    med_o = np.full((H, W), np.nan, np.float32)
    std_o = np.full((H, W), np.nan, np.float32)
    mad_o = np.full((H, W), np.nan, np.float32)

    for rs in range(0, H, tile_rows):
        re   = min(rs + tile_rows, H)
        th   = re - rs
        tile = padded[rs : re + 2 * pad, :]
        flat = sliding_window_view(tile, (ws, ws)).reshape(th, W, -1).copy()

        valid = np.isfinite(flat)
        nv    = valid.sum(axis=2).astype(np.int32)
        fn    = np.where(valid, flat, np.nan)
        tmed  = np.nanmedian(fn, axis=2).astype(np.float32)

        v0 = np.where(valid, flat, 0.0)
        s1 = v0.sum(axis=2)
        s2 = (v0 ** 2).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            mw = np.where(nv > 0, s1 / nv, np.nan)
            vw = np.where(nv > 1, np.maximum(s2 / nv - mw ** 2, 0.0), np.nan)
        tstd = np.sqrt(vw).astype(np.float32)

        adev = np.where(valid, np.abs(fn - tmed[:, :, np.newaxis]), np.nan)
        tmad = np.nanmedian(adev, axis=2).astype(np.float32)

        sp = nv < min_valid
        tmed[sp] = np.nan; tstd[sp] = np.nan; tmad[sp] = np.nan

        med_o[rs:re] = tmed
        std_o[rs:re] = tstd
        mad_o[rs:re] = tmad
        del flat, fn, adev, v0

    return med_o, std_o, mad_o


def apply_mad(
    d_rg:            np.ndarray,
    d_az:            np.ndarray,
    ws:              int   = 5,
    sigma_threshold: float = 5.0,
    min_valid:       int   = 4,
    tile_rows:       int   = 256,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Sliding-window MAD filter on raw range / azimuth displacement maps.

    Two criteria (evaluated independently per component, NaN-ed jointly):
      (a)  |d - median_window| > MAD_window       ← local outlier
      (b)  sigma_window > sigma_threshold (metres) ← high-variance window

    Parameters
    ----------
    sigma_threshold : std threshold in metres (raw displacement).
                      Typical values: 3–10 m for 12-day Sentinel-1 pairs.

    Returns
    -------
    d_rg_f, d_az_f : filtered copies
    stats          : diagnostic dict
    """
    med_rg, sig_rg, mad_rg = _window_stats(d_rg, ws, min_valid, tile_rows)
    med_az, sig_az, mad_az = _window_stats(d_az, ws, min_valid, tile_rows)

    with np.errstate(invalid="ignore"):
        crit_a = (np.abs(d_rg - med_rg) > mad_rg) | (np.abs(d_az - med_az) > mad_az)
        crit_b = (sig_rg > sigma_threshold) | (sig_az > sigma_threshold)

    is_valid = np.isfinite(d_rg) & np.isfinite(d_az)
    bad = (crit_a | crit_b) & is_valid

    d_rg_f = d_rg.copy(); d_rg_f[bad] = np.nan
    d_az_f = d_az.copy(); d_az_f[bad] = np.nan

    n_in  = int(is_valid.sum())
    n_out = int(bad.sum())
    n_a   = int((crit_a & is_valid).sum())
    n_b   = int((crit_b & ~crit_a & is_valid).sum())
    pct   = round(100.0 * n_out / n_in, 2) if n_in else 0.0

    stats = dict(n_input_valid=n_in, n_removed=n_out, removal_pct=pct,
                 n_crit_a=n_a, n_crit_b_only=n_b)
    log.debug("  MAD(ws=%d σ_thr=%.1f m): removed %d/%d (%.1f%%) "
              "[crit_a=%d crit_b_only=%d]",
              ws, sigma_threshold, n_out, n_in, pct, n_a, n_b)
    return d_rg_f, d_az_f, stats


def _prefilter(
    d_rg:    np.ndarray,
    d_az:    np.ndarray,
    method:  str,
    params:  dict,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Dispatch to the requested pre-filter."""
    if method == "none":
        n = int((np.isfinite(d_rg) & np.isfinite(d_az)).sum())
        return d_rg.copy(), d_az.copy(), {"n_removed": 0, "removal_pct": 0.0,
                                           "n_input_valid": n}
    if method == "umt":
        return apply_umt(d_rg, d_az,
                         ws=params["umt_window"], eps=params["umt_eps"],
                         threshold=params["umt_threshold"],
                         min_segment=params["umt_min_segment"])
    if method == "mad":
        return apply_mad(d_rg, d_az,
                         ws=params["mad_window"],
                         sigma_threshold=params["mad_sigma"],
                         min_valid=params["mad_min_valid"],
                         tile_rows=params["mad_tile_rows"])
    raise ValueError(f"Unknown prefilter: {method!r}")


# =============================================================================
# Core conversion: one pair
# =============================================================================

def _convert_pair(
    rg_path:          Path,
    az_path:          Path,
    out_dir:          Path,
    date1_str:        str,
    date2_str:        str,
    delta_days:       int,
    # lv_phi — pre-loaded global array OR None (→ load per-date)
    global_lv_phi:    Optional[np.ndarray],
    ref_transform:    rasterio.Affine,
    ref_h:            int,
    ref_w:            int,
    ref_crs:          CRS,
    geom_dir:         Optional[Path],
    lv_suffix:        str,
    prefilter_method: str,
    prefilter_params: dict,
    already_velocity: bool,
    nodata_out:       float,
    overwrite:        bool,
) -> dict:

    stem   = f"{date1_str}-{date2_str}"
    vx_out = out_dir / f"{stem}-Vx.tif"
    vy_out = out_dir / f"{stem}-Vy.tif"
    v_out  = out_dir / f"{stem}-V.tif"

    if not overwrite and all(p.exists() for p in (vx_out, vy_out, v_out)):
        log.debug("SKIP (exists): %s", stem)
        return {"stem": stem, "status": "skipped"}

    # ── 1. Read raw displacements ─────────────────────────────────────────────
    try:
        d_rg_raw, tf_rg, crs_rg = _read_tif(rg_path)
        d_az_raw, tf_az, crs_az = _read_tif(az_path)
    except Exception as exc:
        log.error("READ FAIL %s: %s", stem, exc)
        return {"stem": stem, "status": "error", "msg": str(exc)}

    # ── 2. Resample to lv_phi reference grid ─────────────────────────────────
    d_rg = _resample_to_ref(d_rg_raw, tf_rg, crs_rg,
                             ref_transform, ref_h, ref_w, ref_crs)
    d_az = _resample_to_ref(d_az_raw, tf_az, crs_az,
                             ref_transform, ref_h, ref_w, ref_crs)

    if np.all(np.isnan(d_rg)) or np.all(np.isnan(d_az)):
        log.warning("SKIP %s — no valid overlap with reference grid", stem)
        return {"stem": stem, "status": "skipped"}

    # ── 3. lv_phi for this pair ───────────────────────────────────────────────
    if global_lv_phi is not None:
        alpha = global_lv_phi
    else:
        alpha = _load_lv_phi_per_date(
            geom_dir, date1_str, date2_str, lv_suffix,
            ref_transform, ref_h, ref_w, ref_crs)
        if alpha is None:
            return {"stem": stem, "status": "error", "msg": "lv_phi missing"}

    # ── 4. Pre-filter raw displacements (metres) ──────────────────────────────
    d_rg, d_az, flt = _prefilter(d_rg, d_az, prefilter_method, prefilter_params)
    if flt["n_removed"]:
        log.info("  %s  [%s] removed=%d (%.1f%%)",
                 stem, prefilter_method, flt["n_removed"], flt["removal_pct"])

    # ── 5. m → m/yr ───────────────────────────────────────────────────────────
    if already_velocity:
        v_rg, v_az = d_rg, d_az
    else:
        if delta_days <= 0:
            return {"stem": stem, "status": "error",
                    "msg": f"delta_days={delta_days}"}
        s    = np.float32(DAYS_PER_YEAR / delta_days)
        v_rg = (d_rg * s).astype(np.float32)
        v_az = (d_az * s).astype(np.float32)

    # ── 6. Rotation: (range, azimuth) → (East, North) ───────────────────────
    #
    #   GAMMA lv_phi convention: 0 = East, π/2 = North, CCW, radians
    #
    #   [Vx]   [cos α  -sin α] [V_rg]
    #   [Vy] = [sin α   cos α] [V_az]
    #
    cos_a = np.cos(alpha).astype(np.float32)
    sin_a = np.sin(alpha).astype(np.float32)

    vx = v_rg * cos_a - v_az * sin_a
    vy = v_rg * sin_a + v_az * cos_a

    bad = ~np.isfinite(d_rg) | ~np.isfinite(d_az) | ~np.isfinite(alpha)
    vx[bad] = np.nan;  vy[bad] = np.nan
    v = np.sqrt(vx**2 + vy**2).astype(np.float32)
    v[bad] = np.nan

    # ── 7. Write ──────────────────────────────────────────────────────────────
    try:
        _write_tif(vx_out, vx, ref_transform, ref_crs, nodata_out)
        _write_tif(vy_out, vy, ref_transform, ref_crs, nodata_out)
        _write_tif(v_out,  v,  ref_transform, ref_crs, nodata_out)
    except Exception as exc:
        log.error("WRITE FAIL %s: %s", stem, exc)
        return {"stem": stem, "status": "error", "msg": str(exc)}

    n_v    = int(np.isfinite(vx).sum())
    pct    = 100.0 * n_v / vx.size
    v_mean = float(np.nanmean(v)) if n_v else float("nan")

    log.info("OK  %s  Δt=%dd  valid=%.1f%%  meanV=%.1f m/yr  "
             "filtered=%d px",
             stem, delta_days, pct, v_mean, flt["n_removed"])

    return dict(stem=stem, status="ok", valid_pct=pct,
                mean_v=v_mean, delta_days=delta_days,
                n_filter_removed=flt["n_removed"])


# =============================================================================
# Pair discovery
# =============================================================================

def _discover_pairs(
    input_root: Path,
    rg_dir:     str,
    az_dir:     str,
    rg_suffix:  str,
    az_suffix:  str,
) -> list:
    rg_root = input_root / rg_dir
    az_root = input_root / az_dir
    if not rg_root.exists():
        raise FileNotFoundError(f"Range dir not found: {rg_root}")
    if not az_root.exists():
        raise FileNotFoundError(f"Azimuth dir not found: {az_root}")

    rg_idx = {p.name[: -len(rg_suffix)]: p
               for p in sorted(rg_root.glob(f"*{rg_suffix}"))}

    pairs, miss = [], 0
    for stem, rg_path in rg_idx.items():
        parts = stem.split("-")
        if len(parts) != 2:
            continue
        d1s, d2s = parts
        az_path = az_root / f"{stem}{az_suffix}"
        if not az_path.exists():
            log.warning("No azimuth for %s — skip", stem)
            miss += 1
            continue
        try:
            d1 = datetime.strptime(d1s, "%Y%m%d")
            d2 = datetime.strptime(d2s, "%Y%m%d")
        except ValueError:
            log.warning("Bad dates in stem %s — skip", stem)
            continue
        if d1 > d2:
            d1, d2, d1s, d2s = d2, d1, d2s, d1s
        pairs.append(dict(stem=stem, date1_str=d1s, date2_str=d2s,
                          delta_days=(d2 - d1).days,
                          rg_path=rg_path, az_path=az_path))

    pairs.sort(key=lambda x: x["stem"])
    log.info("Discovered %d pairs (%d skipped: no azimuth)", len(pairs), miss)
    return pairs


# =============================================================================
# Worker (top-level for ProcessPoolExecutor pickle)
# =============================================================================

def _worker(kw: dict) -> dict:
    logging.basicConfig(
        level=kw["log_level"],
        format="%(asctime)s [%(levelname)s][%(process)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return _convert_pair(
            rg_path=kw["rg_path"], az_path=kw["az_path"],
            out_dir=kw["out_dir"],
            date1_str=kw["date1_str"], date2_str=kw["date2_str"],
            delta_days=kw["delta_days"],
            global_lv_phi=kw["global_lv_phi"],
            ref_transform=kw["ref_transform"],
            ref_h=kw["ref_h"], ref_w=kw["ref_w"], ref_crs=kw["ref_crs"],
            geom_dir=kw["geom_dir"], lv_suffix=kw["lv_suffix"],
            prefilter_method=kw["prefilter_method"],
            prefilter_params=kw["prefilter_params"],
            already_velocity=kw["already_velocity"],
            nodata_out=kw["nodata_out"], overwrite=kw["overwrite"],
        )
    except Exception as exc:
        return {"stem": kw["stem"], "status": "error", "msg": str(exc)}


# =============================================================================
# Main
# =============================================================================

def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    input_root = Path(args.input_root)
    out_dir    = Path(args.output_dir)

    # ── lv_phi source ─────────────────────────────────────────────────────────
    if args.lv_phi_file:
        p = Path(args.lv_phi_file)
        if not p.exists():
            log.error("--lv-phi-file not found: %s", p); sys.exit(1)
        global_lv_phi, ref_transform, ref_crs = _load_lv_phi_global(p)
        ref_h, ref_w = global_lv_phi.shape
        geom_dir     = None
        use_global   = True
    else:
        geom_dir      = Path(args.geom_dir) if args.geom_dir \
                        else input_root / "geometry"
        global_lv_phi = None
        ref_transform = ref_crs = None
        ref_h = ref_w = 0
        use_global    = False

    # ── pre-filter params ─────────────────────────────────────────────────────
    pf_params = dict(
        umt_window=args.umt_window, umt_eps=args.umt_eps,
        umt_threshold=args.umt_threshold, umt_min_segment=args.umt_min_segment,
        mad_window=args.mad_window, mad_sigma=args.mad_sigma,
        mad_min_valid=args.mad_min_valid, mad_tile_rows=args.mad_tile_rows,
    )

    log.info("=" * 65)
    log.info("  GAMMA Preprocess v2  —  range/azimuth → Vx/Vy/V")
    log.info("  input    : %s", input_root)
    log.info("  output   : %s", out_dir)
    log.info("  lv_phi   : %s",
             f"GLOBAL  {args.lv_phi_file}" if use_global
             else f"per-date  {geom_dir}")
    log.info("  prefilter: %s", args.prefilter)
    if args.prefilter == "umt":
        log.info("    UMT  ws=%d  eps=%.2f  thr=%.1f  min_seg=%d",
                 args.umt_window, args.umt_eps, args.umt_threshold,
                 args.umt_min_segment)
    elif args.prefilter == "mad":
        log.info("    MAD  ws=%d  sigma=%.1f m  min_valid=%d",
                 args.mad_window, args.mad_sigma, args.mad_min_valid)
    log.info("  units    : %s",
             "already m/yr" if args.already_velocity else "m → m/yr")
    log.info("=" * 65)

    # ── Discover & filter pairs ───────────────────────────────────────────────
    pairs = _discover_pairs(input_root, args.rg_dir, args.az_dir,
                            args.rg_suffix, args.az_suffix)
    if not pairs:
        log.error("No pairs found — check --rg-dir / --az-dir / suffixes")
        sys.exit(1)

    if args.date_start:
        t0 = datetime.strptime(args.date_start, "%Y-%m-%d")
        pairs = [p for p in pairs
                 if datetime.strptime(p["date1_str"], "%Y%m%d") >= t0]
    if args.date_end:
        t1 = datetime.strptime(args.date_end, "%Y-%m-%d")
        pairs = [p for p in pairs
                 if datetime.strptime(p["date2_str"], "%Y%m%d") <= t1]
    if args.max_days:
        pairs = [p for p in pairs if p["delta_days"] <= args.max_days]

    log.info("Processing %d pairs", len(pairs))

    if args.dry_run:
        for p in pairs:
            log.info("  DRY  %s  Δt=%dd", p["stem"], p["delta_days"])
        return

    # ── Per-date mode: determine reference grid from first available lv_phi ───
    if not use_global:
        found = False
        for p in pairs:
            for ds in (p["date1_str"], p["date2_str"]):
                cand = geom_dir / f"{ds}{args.lv_suffix}"
                if cand.exists():
                    arr, ref_transform, ref_crs = _read_tif(cand)
                    ref_h, ref_w = arr.shape
                    log.info("Reference grid from: %s  shape=(%d,%d)",
                             cand.name, ref_h, ref_w)
                    found = True
                    break
            if found:
                break
        if not found:
            log.error("No lv_phi found in %s to set reference grid", geom_dir)
            sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build work items ──────────────────────────────────────────────────────
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    work = [dict(
        stem=p["stem"], rg_path=p["rg_path"], az_path=p["az_path"],
        out_dir=out_dir, date1_str=p["date1_str"], date2_str=p["date2_str"],
        delta_days=p["delta_days"],
        global_lv_phi=global_lv_phi,
        ref_transform=ref_transform, ref_h=ref_h, ref_w=ref_w, ref_crs=ref_crs,
        geom_dir=geom_dir, lv_suffix=args.lv_suffix,
        prefilter_method=args.prefilter, prefilter_params=pf_params,
        already_velocity=args.already_velocity,
        nodata_out=args.nodata, overwrite=args.overwrite,
        log_level=log_level,
    ) for p in pairs]

    # ── Execute ───────────────────────────────────────────────────────────────
    results, n = [], len(work)

    if args.workers > 1 and n > 1:
        log.info("Parallel: %d workers × %d pairs", args.workers, n)
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futs = {pool.submit(_worker, w): w["stem"] for w in work}
            done = 0
            for fut in as_completed(futs):
                done += 1
                stem = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"stem": stem, "status": "error", "msg": str(e)}
                results.append(r)
                log.info("[%d/%d] %s → %s", done, n, stem, r["status"])
    else:
        for i, w in enumerate(work, 1):
            log.info("[%d/%d] %s", i, n, w["stem"])
            results.append(_worker(w))

    # ── Summary ───────────────────────────────────────────────────────────────
    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors  = [r for r in results if r["status"] == "error"]
    log.info("")
    log.info("=" * 65)
    log.info("  Done.  OK=%d  skipped=%d  errors=%d  / %d total",
             len(ok), len(skipped), len(errors), n)
    vs = [r["mean_v"] for r in ok if r.get("mean_v") is not None]
    nf = [r.get("n_filter_removed", 0) for r in ok]
    if vs:
        log.info("  Mean speed            : %.1f m/yr", float(np.mean(vs)))
    if nf and args.prefilter != "none":
        log.info("  Filtered (avg / pair) : %.0f px", float(np.mean(nf)))
    for r in errors:
        log.warning("  ERROR  %s — %s", r["stem"], r.get("msg", "?"))
    log.info("  Output: %s", out_dir)
    log.info("=" * 65)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gamma_preprocess",
        description="GAMMA range/azimuth → Vx/Vy/V (m/yr)  [v2]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--input-root", required=True,
                   help="Root dir containing --rg-dir and --az-dir subdirectories")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for Vx/Vy/V TIFs")

    # ── lv_phi source (mutually exclusive) ────────────────────────────────────
    lv = p.add_mutually_exclusive_group()
    lv.add_argument("--lv-phi-file", default=None, metavar="PATH",
                    help="Single shared lv_phi.tif for ALL pairs. "
                         "Its grid becomes the reference grid for resampling.")
    lv.add_argument("--geom-dir", default=None, metavar="DIR",
                    help="Directory with per-date lv_phi TIFs "
                         "(default: <input-root>/geometry). "
                         "Mutually exclusive with --lv-phi-file.")

    # ── Sub-directory / suffix ────────────────────────────────────────────────
    p.add_argument("--rg-dir",    default="range")
    p.add_argument("--az-dir",    default="azimuth")
    p.add_argument("--rg-suffix", default=".range.tif")
    p.add_argument("--az-suffix", default=".azi.tif")
    p.add_argument("--lv-suffix", default=".lv_phi.tif",
                   help="Suffix of per-date lv_phi files (with --geom-dir)")

    # ── Pre-filter ────────────────────────────────────────────────────────────
    p.add_argument("--prefilter", default="none",
                   choices=["none", "umt", "mad"],
                   help="Spatial outlier filter applied to raw D_range/D_az "
                        "BEFORE unit conversion and rotation")

    # UMT parameters
    p.add_argument("--umt-window",      type=int,   default=5,
                   help="[UMT] Odd window side (pixels)")
    p.add_argument("--umt-eps",         type=float, default=0.1,
                   help="[UMT] Normalisation floor ε")
    p.add_argument("--umt-threshold",   type=float, default=3.0,
                   help="[UMT] Score threshold — lower = stricter")
    p.add_argument("--umt-min-segment", type=int,   default=25,
                   help="[UMT] Remove connected segments smaller than this (px)")

    # MAD parameters
    p.add_argument("--mad-window",    type=int,   default=5,
                   help="[MAD] Odd window side (pixels)")
    p.add_argument("--mad-sigma",     type=float, default=5.0, metavar="METRES",
                   help="[MAD] Window std threshold in metres (raw displacement). "
                        "Typical: 3–10 m for 12-day S1 pairs")
    p.add_argument("--mad-min-valid", type=int,   default=4,
                   help="[MAD] Min finite pixels in window for stats")
    p.add_argument("--mad-tile-rows", type=int,   default=256,
                   help="[MAD] Rows per processing stripe (memory tuning)")

    # ── Units ─────────────────────────────────────────────────────────────────
    p.add_argument("--already-velocity", action="store_true", default=False,
                   help="Inputs are already m/yr — skip Δt normalisation")

    # ── Date / span filters ───────────────────────────────────────────────────
    p.add_argument("--date-start", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--date-end",   default=None, metavar="YYYY-MM-DD")
    p.add_argument("--max-days",   type=int, default=None,
                   help="Skip pairs with temporal baseline > N days")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--nodata",    type=float, default=float("nan"))
    p.add_argument("--overwrite", action="store_true", default=False)

    # ── Execution ─────────────────────────────────────────────────────────────
    p.add_argument("--workers",   type=int, default=1)
    p.add_argument("--dry-run",   action="store_true", default=False)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
