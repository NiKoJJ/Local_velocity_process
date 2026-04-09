#!/usr/bin/env python3
"""
gamma_preprocess.py
===================
Convert raw GAMMA POT offset-tracking outputs (range + azimuth displacement
maps) into the Vx / Vy / V velocity TIFs expected by the pipeline.

Directory layout assumed
------------------------
<input_root>/
azimuth/   yyyymmdd-yyyymmdd.azi.tif        �� along-track displacement (m)
range/     yyyymmdd-yyyymmdd.range.tif       �� ground-range displacement (m)
geometry/  yyyymmdd.lv_phi.tif              �� look-vector orientation (rad,
                                               0=East, ��/2=North, CCW)

Output layout (pipeline-compatible)
------------------------------------
<output_dir>/
yyyymmdd-yyyymmdd-Vx.tif    �� East  velocity  (m/yr, float32)
yyyymmdd-yyyymmdd-Vy.tif    �� North velocity  (m/yr, float32)
yyyymmdd-yyyymmdd-V.tif     �� Speed           (m/yr, float32)

Conversion
----------
1. Displacement (m) �� velocity (m/yr):
   V_range = D_range / ��t_days �� 365.25
   V_az    = D_az    / ��t_days �� 365.25

2. Range/Azimuth �� East/North (using lv_phi = ��, CCW from East):
   Vx = V_range �� cos(��) ? V_az �� sin(��)
   Vy = V_range �� sin(��) + V_az �� cos(��)
   V  = ��(Vx2 + Vy2)

lv_phi selection per pair
--------------------------
For pair date1�Cdate2:
Priority: 
1. --lv-phi-ref (if provided, use for ALL pairs + resample data to match)
2. date1.lv_phi �� date2.lv_phi �� average(date1, date2)

Usage
-----
# Basic
python gamma_preprocess.py \\
--input-root /data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Input/Cook_120m \\
--output-dir /data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Output/Cook_120m/velocity \\

# With unified lv_phi reference (ALL outputs will match this shape)
python gamma_preprocess.py \\
--input-root /data2/.../Cook_120m \\
--output-dir /data2/.../velocity \\
--lv-phi-ref /data2/.../geometry/20220109.lv_phi.tif \\

# With options
python gamma_preprocess.py \\
--input-root /data2/.../Cook_120m \\
--output-dir /data2/.../velocity \\
--az-suffix  .azi.tif \\
--rg-suffix  .range.tif \\
--lv-suffix  .lv_phi.tif \\
--nodata     -9999 \\
--workers    8 \\
--dry-run

# Override units �� if input files are already in m/yr, skip time normalisation
python gamma_preprocess.py --input-root ... --output-dir ... --already-velocity
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
from scipy.ndimage import zoom

log = logging.getLogger(__name__)

# ���� constants ����������������������������������������������������������������������������������������������������������������������������������
DAYS_PER_YEAR = 365.25
_SENTINEL     = np.float32(-9999.0)   # internal fill during processing


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# I/O helpers
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def _read_tif(path: Path) -> Tuple[np.ndarray, rasterio.Affine, CRS, float]:
    """
    Read first band of a GeoTIFF as float32.
    
    Returns
    -------
    data      : (H, W) float32  �� nodata replaced with NaN
    transform : rasterio Affine
    crs       : CRS
    nodata    : original nodata value (may be None �� 0.0 used)
    """
    with rasterio.open(str(path)) as src:
        data = src.read(1).astype(np.float32)
        tf   = src.transform
        crs  = src.crs
        nd   = src.nodata
        if nd is not None:
            data[data == float(nd)] = np.nan
        # Also mask zeros that look like fill (common in GAMMA products)
        # �� comment out the line below if zero is a legitimate velocity value
        # data[data == 0.0] = np.nan
        data[~np.isfinite(data)] = np.nan
        return data, tf, crs, nd


def _write_tif(
    path: Path,
    data: np.ndarray,        # (H, W) float32
    transform: rasterio.Affine,
    crs: CRS,
    nodata: float = float("nan"),
) -> None:
    """Write a float32 single-band GeoTIFF with deflate compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = data.astype(np.float32)
    if np.isfinite(nodata):
        arr[~np.isfinite(arr)] = nodata
    profile = dict(
        driver    = "GTiff",
        dtype     = "float32",
        count     = 1,
        height    = arr.shape[0],
        width     = arr.shape[1],
        crs       = crs,
        transform = transform,
        nodata    = nodata,
        compress  = "deflate",
        tiled     = True,
        blockxsize= 256,
        blockysize= 256,
    )
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr, 1)


def _resample_array(
    arr: np.ndarray,
    target_shape: Tuple[int, int],
    order: int = 1,
    mode: str = 'nearest',
) -> np.ndarray:
    """
    Resample a 2D array to target shape using scipy.ndimage.zoom.
    
    Parameters
    ----------
    arr : np.ndarray
        Input 2D array
    target_shape : Tuple[int, int]
        Target (height, width)
    order : int
        Interpolation order (0=nearest, 1=bilinear, 2=quadratic, 3=cubic)
        For displacement/velocity data, use order=1 (bilinear)
        For angles (lv_phi), use order=1 (bilinear) to avoid artifacts
    mode : str
        Boundary mode ('nearest', 'constant', 'reflect', 'wrap')
    
    Returns
    -------
    np.ndarray
        Resampled array with target_shape
    """
    if arr.shape == target_shape:
        return arr
    
    zoom_y = target_shape[0] / arr.shape[0]
    zoom_x = target_shape[1] / arr.shape[1]
    
    # Use bilinear interpolation for continuous data
    arr_resampled = zoom(arr, (zoom_y, zoom_x), order=order, mode=mode)
    
    # Ensure exact target shape (zoom may produce off-by-one due to rounding)
    if arr_resampled.shape != target_shape:
        h, w = target_shape
        h_curr, w_curr = arr_resampled.shape
        
        if h_curr > h:
            arr_resampled = arr_resampled[:h, :]
        elif h_curr < h:
            pad_h = h - h_curr
            arr_resampled = np.pad(arr_resampled, ((0, pad_h), (0, 0)), mode='nearest')
        
        if w_curr > w:
            arr_resampled = arr_resampled[:, :w]
        elif w_curr < w:
            pad_w = w - w_curr
            arr_resampled = np.pad(arr_resampled, ((0, 0), (0, pad_w)), mode='nearest')
    
    return arr_resampled


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# Geometry: lv_phi look-up
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def _find_lv_phi(
    geom_dir: Path,
    date1_str: str,
    date2_str: str,
    lv_suffix: str,
) -> Optional[Path]:
    """
    Return the best lv_phi file for a pair.
    Search order: date1 �� date2 �� (both present �� use date1, log warning)
    """
    p1 = geom_dir / f"{date1_str}{lv_suffix}"
    p2 = geom_dir / f"{date2_str}{lv_suffix}"
    if p1.exists():
        return p1
    if p2.exists():
        log.debug("  lv_phi: date1 not found, falling back to date2 for %s-%s",
                  date1_str, date2_str)
        return p2
    log.warning("  lv_phi not found for %s or %s in %s",
                date1_str, date2_str, geom_dir)
    return None


def _load_lv_phi_for_pair(
    geom_dir: Path,
    date1_str: str,
    date2_str: str,
    lv_suffix: str,
    ref_shape: Tuple[int, int],
    lv_phi_ref: Optional[Path] = None,  # �� ��������������������
) -> Optional[np.ndarray]:
    """
    Load lv_phi (radians, CCW from East) for a date pair.
    
    Priority:
    1. --lv-phi-ref (if provided, use for ALL pairs)
    2. date1.lv_phi �� date2.lv_phi �� average(date1, date2)
    
    If both dates have a file, average them (geometry rarely changes between
    Sentinel-1 acquisitions 6�C12 days apart, but averaging is robust).
    
    Returns
    -------
    (H, W) float32 array or None
    """
    # �� ��������������������������������������
    if lv_phi_ref is not None and lv_phi_ref.exists():
        arr, *_ = _read_tif(lv_phi_ref)
        if arr.shape == ref_shape:
            log.debug("  lv_phi: using unified reference %s (shape match)", 
                     lv_phi_ref.name)
            return arr.astype(np.float32)
        else:
            # �� ����������������������
            log.warning("  lv_phi_ref shape mismatch: %s has %s, expected %s",
                       lv_phi_ref.name, arr.shape, ref_shape)
            log.info("  Resampling lv_phi_ref to match target shape...")
            arr_resampled = _resample_array(arr, ref_shape, order=1)
            log.info("  Resampling successful: %s �� %s", arr.shape, arr_resampled.shape)
            return arr_resampled.astype(np.float32)
    
    # ��������������������
    p1 = geom_dir / f"{date1_str}{lv_suffix}"
    p2 = geom_dir / f"{date2_str}{lv_suffix}"
    arrays = []
    for p in (p1, p2):
        if p.exists():
            arr, *_ = _read_tif(p)
            if arr.shape == ref_shape:
                arrays.append(arr)
            else:
                # �� ������ per-pair ������������
                log.warning("  lv_phi shape mismatch: %s has %s, expected %s �� resampling",
                           p.name, arr.shape, ref_shape)
                arr_resampled = _resample_array(arr, ref_shape, order=1)
                arrays.append(arr_resampled)
    if not arrays:
        log.warning("  No usable lv_phi for %s-%s", date1_str, date2_str)
        return None
    if len(arrays) == 2:
        # Circular mean of two close angles (safe for small angular differences)
        alpha = np.nanmean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
    else:
        alpha = arrays[0]
    return alpha


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# Core conversion: one pair
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def _convert_pair(
    rg_path:   Path,
    az_path:   Path,
    geom_dir:  Path,
    out_dir:   Path,
    date1_str: str,
    date2_str: str,
    delta_days: int,
    lv_suffix:  str,
    nodata_out: float,
    already_velocity: bool,
    overwrite:  bool,
    lv_phi_ref: Optional[Path] = None,  # �� ��������������������
) -> dict:
    """
    Convert one range/azimuth pair to Vx, Vy, V and write TIFs.
    Returns a status dict.
    
    If lv_phi_ref is provided, ALL outputs will be resampled to match its shape.
    """
    stem   = f"{date1_str}-{date2_str}"
    vx_out = out_dir / f"{stem}-Vx.tif"
    vy_out = out_dir / f"{stem}-Vy.tif"
    v_out  = out_dir / f"{stem}-V.tif"
    if not overwrite and all(p.exists() for p in (vx_out, vy_out, v_out)):
        log.debug("  SKIP (exists): %s", stem)
        return {"stem": stem, "status": "skipped"}
    
    # ���� Read range and azimuth displacement (m or m/yr) ����������������������������������������������
    try:
        d_rg, tf, crs, _ = _read_tif(rg_path)
        d_az, *_          = _read_tif(az_path)
    except Exception as exc:
        log.error("  READ FAIL %s: %s", stem, exc)
        return {"stem": stem, "status": "error", "msg": str(exc)}
    
    H, W = d_rg.shape
    
    # ���� Load lv_phi ����������������������������������������������������������������������������������������������������������������������
    alpha = _load_lv_phi_for_pair(
        geom_dir, date1_str, date2_str, lv_suffix,
        ref_shape=(H, W),
        lv_phi_ref=lv_phi_ref,  # �� ����������������
    )
    if alpha is None:
        return {"stem": stem, "status": "error", "msg": "lv_phi missing"}
    
    # �� ���������� lv_phi_ref����������������������������
    target_shape = alpha.shape
    resampled = False
    if d_rg.shape != target_shape:
        log.debug("  Resampling range data: %s �� %s", d_rg.shape, target_shape)
        d_rg = _resample_array(d_rg, target_shape, order=1)
        resampled = True
    if d_az.shape != target_shape:
        log.debug("  Resampling azimuth data: %s �� %s", d_az.shape, target_shape)
        d_az = _resample_array(d_az, target_shape, order=1)
        resampled = True
    
    if resampled:
        H, W = target_shape
        log.info("  Data resampled to unified shape: %s", target_shape)
    
    # ���� Unit conversion: displacement (m) �� velocity (m/yr) ������������������������������������
    if already_velocity:
        v_rg = d_rg
        v_az = d_az
    else:
        if delta_days <= 0:
            log.error("  Invalid delta_days=%d for %s", delta_days, stem)
            return {"stem": stem, "status": "error", "msg": "delta_days <= 0"}
        scale = DAYS_PER_YEAR / delta_days
        v_rg  = (d_rg * scale).astype(np.float32)
        v_az  = (d_az * scale).astype(np.float32)
    
    # ���� Rotation: (range, azimuth) �� (East, North) ������������������������������������������������������
    #
    #  GAMMA lv_phi convention: 0 = East, ��/2 = North, counterclockwise
    #
    #  [Vx]   [cos ��   -sin ��] [V_rg]
    #  [Vy] = [sin ��    cos ��] [V_az]
    #
    cos_a = np.cos(alpha).astype(np.float32)
    sin_a = np.sin(alpha).astype(np.float32)
    vx = v_rg * cos_a - v_az * sin_a   # East  component
    vy = v_rg * sin_a + v_az * cos_a   # North component
    
    # Mask: any pixel where range, azimuth, OR lv_phi is NaN �� NaN in all bands
    bad = ~np.isfinite(d_rg) | ~np.isfinite(d_az) | ~np.isfinite(alpha)
    vx[bad] = np.nan
    vy[bad] = np.nan
    v = np.sqrt(vx**2 + vy**2).astype(np.float32)
    v[bad] = np.nan
    
    # ���� Write outputs ������������������������������������������������������������������������������������������������������������������
    try:
        _write_tif(vx_out, vx, tf, crs, nodata_out)
        _write_tif(vy_out, vy, tf, crs, nodata_out)
        _write_tif(v_out,  v,  tf, crs, nodata_out)
    except Exception as exc:
        log.error("  WRITE FAIL %s: %s", stem, exc)
        return {"stem": stem, "status": "error", "msg": str(exc)}
    
    n_valid = int(np.sum(np.isfinite(vx)))
    pct     = 100.0 * n_valid / vx.size
    v_mean  = float(np.nanmean(v)) if n_valid else float("nan")
    log.info("  OK  %s  valid=%.1f%%  mean_V=%.1f m/yr  ��t=%dd  shape=%s",
             stem, pct, v_mean, delta_days, vx.shape)
    return {"stem": stem, "status": "ok",
            "valid_pct": pct, "mean_v": v_mean, "delta_days": delta_days,
            "shape": vx.shape}


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# Pair discovery
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def _discover_pairs(
    input_root: Path,
    rg_dir:     str,
    az_dir:     str,
    rg_suffix:  str,
    az_suffix:  str,
) -> list:
    """
    Scan range and azimuth directories and return a sorted list of
    dicts describing each complete pair.
    A 'complete pair' has BOTH a range file and a matching azimuth file.
    """
    rg_root = input_root / rg_dir
    az_root = input_root / az_dir
    if not rg_root.exists():
        raise FileNotFoundError(f"Range directory not found: {rg_root}")
    if not az_root.exists():
        raise FileNotFoundError(f"Azimuth directory not found: {az_root}")
    
    # Build index: stem �� range path
    rg_index = {}
    for p in sorted(rg_root.glob(f"*{rg_suffix}")):
        stem = p.name[: -len(rg_suffix)]           # strip suffix
        rg_index[stem] = p
    
    pairs = []
    missing_az = 0
    for stem, rg_path in rg_index.items():
        # stem format: yyyymmdd-yyyymmdd
        parts = stem.split("-")
        if len(parts) != 2:
            log.debug("Unexpected stem format, skipping: %s", stem)
            continue
        d1_str, d2_str = parts
        az_path = az_root / f"{stem}{az_suffix}"
        if not az_path.exists():
            log.warning("No azimuth file for %s �� skipping", stem)
            missing_az += 1
            continue
        try:
            d1 = datetime.strptime(d1_str, "%Y%m%d")
            d2 = datetime.strptime(d2_str, "%Y%m%d")
        except ValueError:
            log.warning("Cannot parse dates from stem: %s �� skipping", stem)
            continue
        if d1 > d2:
            d1, d2, d1_str, d2_str = d2, d1, d2_str, d1_str
        delta = (d2 - d1).days
        pairs.append(dict(
            stem       = stem,
            date1_str  = d1_str,
            date2_str  = d2_str,
            delta_days = delta,
            rg_path    = rg_path,
            az_path    = az_path,
        ))
    pairs.sort(key=lambda x: x["stem"])
    log.info("Found %d complete pairs (%d skipped: no azimuth file)",
             len(pairs), missing_az)
    return pairs


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# Worker (top-level for ProcessPoolExecutor pickling)
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def _worker(kw: dict) -> dict:
    logging.basicConfig(
        level=kw["log_level"],
        format="%(asctime)s [%(levelname)s][%(process)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return _convert_pair(
            rg_path          = kw["rg_path"],
            az_path          = kw["az_path"],
            geom_dir         = kw["geom_dir"],
            out_dir          = kw["out_dir"],
            date1_str        = kw["date1_str"],
            date2_str        = kw["date2_str"],
            delta_days       = kw["delta_days"],
            lv_suffix        = kw["lv_suffix"],
            nodata_out       = kw["nodata_out"],
            already_velocity = kw["already_velocity"],
            overwrite        = kw["overwrite"],
            lv_phi_ref       = kw.get("lv_phi_ref"),  # �� ����������������
        )
    except Exception as exc:
        return {"stem": kw["stem"], "status": "error", "msg": str(exc)}


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# Main
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    input_root = Path(args.input_root)
    geom_dir   = Path(args.geom_dir) if args.geom_dir else input_root / "geometry"
    out_dir    = Path(args.output_dir)
    
    # �� ��������������������
    lv_phi_ref = Path(args.lv_phi_ref) if args.lv_phi_ref else None
    if lv_phi_ref and not lv_phi_ref.exists():
        log.error("lv_phi_ref file not found: %s", lv_phi_ref)
        sys.exit(1)
    
    # �� ���������� lv_phi_ref��������������������
    ref_shape_str = "N/A"
    if lv_phi_ref:
        try:
            arr, *_ = _read_tif(lv_phi_ref)
            ref_shape_str = str(arr.shape)
        except Exception:
            ref_shape_str = "unknown"
    
    log.info("=" * 65)
    log.info("  GAMMA Preprocess �� range/azimuth �� Vx/Vy/V")
    log.info("  input  : %s", input_root)
    log.info("  geom   : %s", geom_dir)
    if lv_phi_ref:
        log.info("  lv_phi : %s", lv_phi_ref)
        log.info("  target : %s (ALL outputs resampled to this shape)", ref_shape_str)
    else:
        log.info("  lv_phi : per-pair lookup (outputs may have different shapes)")
    log.info("  output : %s", out_dir)
    log.info("  units  : %s", "already m/yr" if args.already_velocity
             else "m (will normalise to m/yr)")
    log.info("=" * 65)
    
    if args.dry_run:
        log.info("DRY-RUN mode �� no files will be written")
    
    # ���� Discover pairs ����������������������������������������������������������������������������������������������������������������
    pairs = _discover_pairs(
        input_root, args.rg_dir, args.az_dir, args.rg_suffix, args.az_suffix)
    if not pairs:
        log.error("No pairs found �� check --rg-dir / --az-dir / --rg-suffix / --az-suffix")
        sys.exit(1)
    
    # ���� Filter by date range ����������������������������������������������������������������������������������������������������
    if args.date_start:
        t0 = datetime.strptime(args.date_start, "%Y-%m-%d")
        pairs = [p for p in pairs if datetime.strptime(p["date1_str"], "%Y%m%d") >= t0]
    if args.date_end:
        t1 = datetime.strptime(args.date_end, "%Y-%m-%d")
        pairs = [p for p in pairs if datetime.strptime(p["date2_str"], "%Y%m%d") <= t1]
    if args.max_days:
        pairs = [p for p in pairs if p["delta_days"] <= args.max_days]
    log.info("Processing %d pairs (after date/span filters)", len(pairs))
    
    if args.dry_run:
        for p in pairs:
            log.info("  DRY  %s  ��t=%dd", p["stem"], p["delta_days"])
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ���� Build work items ������������������������������������������������������������������������������������������������������������
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    work = [dict(
        stem             = p["stem"],
        rg_path          = p["rg_path"],
        az_path          = p["az_path"],
        geom_dir         = geom_dir,
        out_dir          = out_dir,
        date1_str        = p["date1_str"],
        date2_str        = p["date2_str"],
        delta_days       = p["delta_days"],
        lv_suffix        = args.lv_suffix,
        nodata_out       = args.nodata,
        already_velocity = args.already_velocity,
        overwrite        = args.overwrite,
        log_level        = log_level,
        lv_phi_ref       = lv_phi_ref,  # �� ����������������
    ) for p in pairs]
    
    # ���� Execute ������������������������������������������������������������������������������������������������������������������������������
    results = []
    n = len(work)
    if args.workers > 1 and n > 1:
        log.info("Parallel: %d workers �� %d pairs", args.workers, n)
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
                log.info("[%d/%d] %s �� %s", done, n, stem, r["status"])
    else:
        for i, w in enumerate(work, 1):
            log.info("[%d/%d] %s", i, n, w["stem"])
            r = _worker(w)
            results.append(r)
    
    # ���� Summary ������������������������������������������������������������������������������������������������������������������������������
    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors  = [r for p in results if p["status"] == "error"]
    log.info("")
    log.info("=" * 65)
    log.info("  Done.  OK=%d  skipped=%d  errors=%d  (total=%d)",
             len(ok), len(skipped), len(errors), n)
    if ok:
        vs = [r["mean_v"] for r in ok if r.get("mean_v") is not None]
        if vs:
            log.info("  Mean speed across all pairs: %.1f m/yr", float(np.mean(vs)))
        # �� ��������������������
        shapes = set(r.get("shape") for r in ok if r.get("shape"))
        if len(shapes) == 1:
            log.info("  All outputs have unified shape: %s", shapes.pop())
        else:
            log.warning("  Output shapes vary: %s", shapes)
    if errors:
        log.warning("  Failed pairs:")
        for r in errors:
            log.warning("    %s �� %s", r["stem"], r.get("msg", "unknown"))
    log.info("  Output: %s", out_dir)
    log.info("=" * 65)


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# CLI
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gamma_preprocess",
        description="Convert GAMMA range/azimuth offsets �� Vx/Vy/V (m/yr)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ���� Paths ����������������������������������������������������������������������������������������������������������������������������������
    p.add_argument("--input-root", required=True,
                   help="Root directory containing range/, azimuth/, geometry/")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for Vx/Vy/V TIFs")
    p.add_argument("--geom-dir", default=None,
                   help="Directory containing lv_phi TIFs "
                        "(default: <input-root>/geometry)")
    # �� ���������� lv_phi ��������
    p.add_argument("--lv-phi-ref", default=None,
                   help="Single lv_phi TIF file to use as reference shape. "
                        "ALL outputs will be resampled to match this file's shape.")
    # ���� Sub-directory names ������������������������������������������������������������������������������������������������������
    p.add_argument("--rg-dir",     default="range",
                   help="Subdirectory name for range (ground-range) files")
    p.add_argument("--az-dir",     default="azimuth",
                   help="Subdirectory name for azimuth files")
    # ���� File suffix ����������������������������������������������������������������������������������������������������������������������
    p.add_argument("--rg-suffix",  default=".range.tif",
                   help="Suffix of range files  (e.g. .range.tif)")
    p.add_argument("--az-suffix",  default=".azi.tif",
                   help="Suffix of azimuth files (e.g. .azi.tif)")
    p.add_argument("--lv-suffix",  default=".lv_phi.tif",
                   help="Suffix of lv_phi geometry files")
    # ���� Unit handling ������������������������������������������������������������������������������������������������������������������
    p.add_argument("--already-velocity", action="store_true", default=False,
                   help="Input files are already in m/yr �� skip ��t normalisation")
    # ���� Filters ������������������������������������������������������������������������������������������������������������������������������
    p.add_argument("--date-start", default=None, metavar="YYYY-MM-DD",
                   help="Process only pairs with date1 >= this date")
    p.add_argument("--date-end",   default=None, metavar="YYYY-MM-DD",
                   help="Process only pairs with date2 <= this date")
    p.add_argument("--max-days",   type=int, default=None,
                   help="Skip pairs whose temporal baseline exceeds this (days)")
    # ���� Output ��������������������������������������������������������������������������������������������������������������������������������
    p.add_argument("--nodata",     type=float, default=float("nan"),
                   help="NoData value written to output TIFs (default: NaN)")
    p.add_argument("--overwrite",  action="store_true", default=False,
                   help="Overwrite existing output files (default: skip)")
    # ���� Execution ��������������������������������������������������������������������������������������������������������������������������
    p.add_argument("--workers",    type=int,   default=1,
                   help="Parallel worker processes (default: 1)")
    p.add_argument("--dry-run",    action="store_true", default=False,
                   help="List pairs that would be processed, then exit")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
