#!/usr/bin/env python3
"""
plot_velocity.py
================
Plot Vx, Vy, V velocity TIFs from GAMMA preprocessing output.
Generates one 3-panel figure per date pair.

Usage
-----
# Basic: plot all pairs in a directory
python plot_velocity.py \
  --input-dir /data/velocity2 \
  --output-dir /data/figures

# Plot specific date range
python plot_velocity.py \
  --input-dir /data/velocity2 \
  --output-dir /data/figures \
  --date-start 2022-01-01 \
  --date-end 2022-06-30

# Customize colormap and resolution
python plot_velocity.py \
  --input-dir /data/velocity2 \
  --output-dir /data/figures \
  --dpi 300 \
  --vmax 1000 \
  --cmap-speed viridis \
  --cmap-component RdBu_r

# Plot only speed (V) component
python plot_velocity.py \
  --input-dir /data/velocity2 \
  --output-dir /data/figures \
  --components V

# Parallel processing
python plot_velocity.py \
  --input-dir /data/velocity2 \
  --output-dir /data/figures \
  --workers 4
"""
from __future__ import annotations
import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

log = logging.getLogger(__name__)

# ���� constants ����������������������������������������������������������������������������������������������������������������������������������
DEFAULT_CMAP_SPEED = 'viridis'
DEFAULT_CMAP_COMPONENT = 'RdBu_r'
DEFAULT_DPI = 150
DEFAULT_VMAX = None  # auto-scale
NODATA = np.nan

# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# I/O helpers
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def _read_tif(path: Path) -> Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    """Read first band of a GeoTIFF, mask nodata as NaN."""
    with rasterio.open(str(path)) as src:
        data = src.read(1).astype(np.float32)
        tf = src.transform
        crs = src.crs
        nd = src.nodata
        if nd is not None and np.isfinite(nd):
            data[data == nd] = np.nan
        data[~np.isfinite(data)] = np.nan
        return data, tf, crs


def _discover_pairs(
    input_dir: Path,
    suffix_vx: str = '-Vx.tif',
    suffix_vy: str = '-Vy.tif',
    suffix_v: str = '-V.tif',
) -> List[dict]:
    """Scan directory for complete Vx/Vy/V triplets and return sorted list."""
    pairs = {}
    for f in input_dir.glob(f'*{suffix_v}'):
        stem = f.name[:-len(suffix_v)]
        if '-' not in stem:
            continue
        # Check if all three components exist
        vx_path = input_dir / f"{stem}{suffix_vx}"
        vy_path = input_dir / f"{stem}{suffix_vy}"
        v_path = input_dir / f"{stem}{suffix_v}"
        if vx_path.exists() and vy_path.exists() and v_path.exists():
            pairs[stem] = dict(
                stem=stem,
                vx_path=vx_path,
                vy_path=vy_path,
                v_path=v_path,
            )
    result = sorted(pairs.values(), key=lambda x: x['stem'])
    log.info("Found %d complete Vx/Vy/V triplets", len(result))
    return result


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# Plotting
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def _add_scalebar(ax, transform, crs, location=4):
    """Add a scalebar to the axis if CRS is projected (in meters)."""
    if crs and crs.is_projected:
        # Get pixel size in meters from transform
        pixel_size = np.sqrt(abs(transform.a * transform.e))
        if 0 < pixel_size < 10000:  # Reasonable range for glacier studies
            scalebar = ScaleBar(
                pixel_size,
                units="m",
                location=location,
                box_alpha=0.5,
                color='white',
                font_properties={'size': 8}
            )
            ax.add_artist(scalebar)


def _plot_pair(
    vx_path: Path,
    vy_path: Path,
    v_path: Path,
    out_path: Path,
    components: List[str],
    vmax: Optional[float],
    cmap_speed: str,
    cmap_component: str,
    dpi: int,
    figsize: Tuple[int, int],
) -> dict:
    """Generate a 3-panel figure for one date pair."""
    stem = vx_path.name.split('-Vx.tif')[0]
    
    try:
        # Read data
        vx, tf, crs = _read_tif(vx_path)
        vy, _, _ = _read_tif(vy_path)
        v, _, _ = _read_tif(v_path)
    except Exception as exc:
        log.error("  READ FAIL %s: %s", stem, exc)
        return {"stem": stem, "status": "error", "msg": str(exc)}
    
    H, W = vx.shape
    
    # Determine which panels to plot
    panel_map = {'Vx': (vx, cmap_component, 'East-West Velocity'),
                 'Vy': (vy, cmap_component, 'North-South Velocity'),
                 'V':  (v,  cmap_speed,    'Speed')}
    
    n_panels = len(components)
    if n_panels == 0:
        log.warning("  No components selected for %s", stem)
        return {"stem": stem, "status": "skipped", "msg": "no components"}
    
    # Create figure
    fig, axes = plt.subplots(1, n_panels, figsize=(figsize[0]*n_panels, figsize[1]),
                            constrained_layout=True, dpi=dpi)
    if n_panels == 1:
        axes = [axes]
    
    for ax, comp in zip(axes, components):
        data, cmap, title = panel_map[comp]
        
        # Set normalization
        if comp == 'V':
            # Speed: always positive, use viridis-like
            vmin = 0
            vmax_use = vmax if vmax is not None else np.nanpercentile(data, 99)
            norm = Normalize(vmin=vmin, vmax=vmax_use)
        else:
            # Components: can be positive or negative, use diverging colormap
            vmax_abs = vmax if vmax is not None else np.nanpercentile(np.abs(data), 99)
            norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)
        
        # Plot
        im = ax.imshow(data, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f"{title}\n{stem}", fontsize=10, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('m/yr', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        # Add scalebar (only on first panel)
        if ax == axes[0]:
            _add_scalebar(ax, tf, crs)
        
        # Add N/S/E/W indicators
        if ax == axes[0] and crs and crs.is_projected:
            # Simple north arrow
            arrow_len = min(H, W) * 0.1
            ax.annotate('', 
                       xy=(W*0.95, H*0.1), 
                       xytext=(W*0.95, H*0.1 + arrow_len),
                       arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
            ax.text(W*0.96, H*0.1 + arrow_len*1.1, 'N', 
                   color='white', fontsize=8, ha='left', va='bottom')
    
    # Save figure
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    except Exception as exc:
        log.error("  SAVE FAIL %s: %s", stem, exc)
        plt.close(fig)
        return {"stem": stem, "status": "error", "msg": str(exc)}
    
    # Log summary stats
    v_valid = v[np.isfinite(v)]
    log.info("  OK  %s  V: min=%.1f max=%.1f mean=%.1f m/yr  saved: %s",
             stem, np.min(v_valid), np.max(v_valid), np.mean(v_valid), out_path)
    
    return {"stem": stem, "status": "ok", 
            "v_min": float(np.min(v_valid)), 
            "v_max": float(np.max(v_valid)),
            "v_mean": float(np.mean(v_valid))}


def _worker(kw: dict) -> dict:
    """Worker function for parallel execution."""
    logging.basicConfig(
        level=kw["log_level"],
        format="%(asctime)s [%(levelname)s][%(process)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    return _plot_pair(**{k: v for k, v in kw.items() 
                        if k not in ['log_level']})


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# Main
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)
    
    log.info("=" * 65)
    log.info("  Velocity Plotter �� Vx/Vy/V �� Figures")
    log.info("  input  : %s", input_dir)
    log.info("  output : %s", output_dir)
    log.info("  components: %s", ', '.join(args.components))
    if args.vmax:
        log.info("  vmax   : %.1f m/yr", args.vmax)
    else:
        log.info("  vmax   : auto (99th percentile)")
    log.info("  dpi    : %d", args.dpi)
    log.info("=" * 65)
    
    # Discover pairs
    pairs = _discover_pairs(input_dir)
    if not pairs:
        log.error("No complete Vx/Vy/V triplets found in %s", input_dir)
        sys.exit(1)
    
    # Filter by date range
    if args.date_start:
        t0 = datetime.strptime(args.date_start, "%Y-%m-%d")
        pairs = [p for p in pairs 
                if datetime.strptime(p['stem'].split('-')[0], "%Y%m%d") >= t0]
    if args.date_end:
        t1 = datetime.strptime(args.date_end, "%Y-%m-%d")
        pairs = [p for p in pairs 
                if datetime.strptime(p['stem'].split('-')[1], "%Y%m%d") <= t1]
    
    log.info("Processing %d pairs (after filters)", len(pairs))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build work items
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    work = []
    for p in pairs:
        out_name = f"{p['stem']}_velocity.png"
        work.append(dict(
            vx_path=p['vx_path'],
            vy_path=p['vy_path'],
            v_path=p['v_path'],
            out_path=output_dir / out_name,
            components=args.components,
            vmax=args.vmax,
            cmap_speed=args.cmap_speed,
            cmap_component=args.cmap_component,
            dpi=args.dpi,
            figsize=args.figsize,
            log_level=log_level,
        ))
    
    # Execute
    results = []
    n = len(work)
    
    if args.workers > 1 and n > 1:
        log.info("Parallel: %d workers �� %d figures", args.workers, n)
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futs = {pool.submit(_worker, w): w['out_path'].name for w in work}
            for i, fut in enumerate(as_completed(futs), 1):
                stem = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"stem": stem, "status": "error", "msg": str(e)}
                results.append(r)
                log.info("[%d/%d] %s �� %s", i, n, stem, r["status"])
    else:
        for i, w in enumerate(work, 1):
            log.info("[%d/%d] %s", i, n, w['out_path'].name)
            r = _worker(w)
            results.append(r)
    
    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    
    log.info("")
    log.info("=" * 65)
    log.info("  Done.  OK=%d  errors=%d  (total=%d)", len(ok), len(errors), n)
    if ok:
        v_means = [r["v_mean"] for r in ok if r.get("v_mean") is not None]
        if v_means:
            log.info("  Mean speed across all pairs: %.1f m/yr", np.mean(v_means))
    if errors:
        log.warning("  Failed plots:")
        for r in errors:
            log.warning("    %s �� %s", r["stem"], r.get("msg", "unknown"))
    log.info("  Output: %s", output_dir)
    log.info("=" * 65)


# ����������������������������������������������������������������������������������������������������������������������������������������������������������
# CLI
# ����������������������������������������������������������������������������������������������������������������������������������������������������������
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plot_velocity",
        description="Plot Vx/Vy/V velocity TIFs as figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Paths
    p.add_argument("--input-dir", required=True,
                   help="Directory containing Vx/Vy/V TIF files")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for PNG figures")
    
    # Components to plot
    p.add_argument("--components", nargs='+', default=['Vx', 'Vy', 'V'],
                   choices=['Vx', 'Vy', 'V'],
                   help="Which velocity components to plot")
    
    # Visualization
    p.add_argument("--vmax", type=float, default=None,
                   help="Max value for color scale (auto if None)")
    p.add_argument("--cmap-speed", default=DEFAULT_CMAP_SPEED,
                   help="Colormap for speed (V)")
    p.add_argument("--cmap-component", default=DEFAULT_CMAP_COMPONENT,
                   help="Colormap for components (Vx, Vy)")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                   help="Output image resolution")
    p.add_argument("--figsize", nargs=2, type=int, default=[5, 4],
                   metavar=('W', 'H'),
                   help="Figure size per panel in inches")
    
    # Filters
    p.add_argument("--date-start", default=None, metavar="YYYY-MM-DD",
                   help="Plot only pairs with date1 >= this date")
    p.add_argument("--date-end", default=None, metavar="YYYY-MM-DD",
                   help="Plot only pairs with date2 <= this date")
    
    # Execution
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel worker processes")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
