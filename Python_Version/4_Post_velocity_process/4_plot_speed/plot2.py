#!/usr/bin/env python3
"""
plot_monthly_profiles.py
========================
Plot monthly ice velocity profiles along a user-defined transect.

Three visualization modes (--mode):
  hovmoller      Time �� distance heatmap  [default, cleanest]
  mean_anomaly   Mean �� std + per-season anomaly panel
  small_multiples One sub-panel per month

Usage
-----
python plot_monthly_profiles.py \
    --data_dir ./monthly_V_Error \
    --points_latlon 158.5,-68.2 160.0,-67.8 162.0,-67.5 \
    --labels A B C \
    --mode hovmoller \
    --output profile.png

python plot_monthly_profiles.py \
    --data_dir ./monthly_V_Error \
    --points_3031 974000,-2100000 1050000,-2050000 1120000,-2020000 \
    --mode all \
    --output profile.png
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

plt.rcParams.update({
    "figure.facecolor": "#0d0d0d",
    "axes.facecolor":   "#111",
    "axes.edgecolor":   "#444",
    "text.color":       "#ddd",
    "axes.labelcolor":  "#ccc",
    "xtick.color":      "#999",
    "ytick.color":      "#999",
    "grid.color":       "#2a2a2a",
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": "#444",
    "legend.labelcolor":"#ccc",
})

SEASON_COLOR = {
    "12":"#e74c3c","01":"#e74c3c","02":"#e74c3c",   # Summer DJF  red
    "03":"#e67e22","04":"#e67e22","05":"#e67e22",   # Autumn MAM  orange
    "06":"#3498db","07":"#3498db","08":"#3498db",   # Winter JJA  blue
    "09":"#2ecc71","10":"#2ecc71","11":"#2ecc71",   # Spring SON  green
}


# ���� file discovery ������������������������������������������������������������������������������������������������������������������������

def discover_monthly_pairs(data_dir):
    pattern = re.compile(r'^(\d{4}-\d{2})_v\.tif$', re.IGNORECASE)
    pairs = []
    for p in sorted(data_dir.iterdir()):
        m = pattern.match(p.name)
        if not m:
            continue
        label    = m.group(1)
        err_path = data_dir / f"{label}_v_err.tif"
        if not err_path.exists():
            print(f"  [warn] no error file for {label}, skipping")
            continue
        pairs.append((label, p, err_path))
    pairs.sort(key=lambda x: x[0])
    return pairs


# ���� coordinate helpers ����������������������������������������������������������������������������������������������������������������

def latlon_to_3031(lons, lats):
    xs, ys = warp_transform(CRS.from_epsg(4326), CRS.from_epsg(3031), lons, lats)
    return list(xs), list(ys)


def build_profile(ctrl_xy, n_pts_per_seg):
    ctrl_xy = np.array(ctrl_xy)
    n_segs  = len(ctrl_xy) - 1
    px, py  = [], []
    ctrl_idx = [0]
    for s in range(n_segs):
        x1, y1 = ctrl_xy[s]
        x2, y2 = ctrl_xy[s + 1]
        n = n_pts_per_seg if np.isscalar(n_pts_per_seg) else n_pts_per_seg[s]
        seg_x = np.linspace(x1, x2, n)
        seg_y = np.linspace(y1, y2, n)
        if s > 0:
            seg_x = seg_x[1:]; seg_y = seg_y[1:]
        px.extend(seg_x); py.extend(seg_y)
        ctrl_idx.append(len(px) - 1)
    px, py = np.array(px), np.array(py)
    dist_km = np.concatenate([[0],
        np.cumsum(np.sqrt(np.diff(px)**2 + np.diff(py)**2))]) / 1e3
    return px, py, dist_km, dist_km[ctrl_idx]


# ���� raster sampling ����������������������������������������������������������������������������������������������������������������������

def sample_tif(tif_path, px, py):
    with rasterio.open(str(tif_path)) as src:
        data = src.read(1).astype(np.float64)
        nd   = src.nodata
        if nd is not None:
            data[data == nd] = np.nan
        data[~np.isfinite(data)] = np.nan
        h, w = data.shape
        tf   = src.transform
        xs   = tf.c + tf.a * (np.arange(w) + 0.5)
        ys   = tf.f + tf.e * (np.arange(h) + 0.5)
        if ys[0] > ys[-1]:
            ys   = ys[::-1]
            data = data[::-1, :]
        interp = RegularGridInterpolator(
            (ys, xs), data, method="linear",
            bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([py, px])).astype(np.float32)


# ���� Mode 1: Hovm?ller ������������������������������������������������������������������������������������������������������������������

def plot_hovmoller(dist_km, ctrl_dist_km, ctrl_labels, profiles,
                   output_path, unit="m/yr", vmin=None, vmax=None):
    """
    Clean time �� distance heatmap.
    Immediately shows WHERE speed is high and WHEN it changes.
    """
    labels  = [p[0] for p in profiles]
    n_t, n_d = len(profiles), len(dist_km)
    V_mat   = np.full((n_t, n_d), np.nan)
    Err_mat = np.full((n_t, n_d), np.nan)
    for i, (_, v, err) in enumerate(profiles):
        V_mat[i]   = v
        Err_mat[i] = err

    vlo = vmin if vmin is not None else np.nanpercentile(V_mat, 2)
    vhi = vmax if vmax is not None else np.nanpercentile(V_mat, 98)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})

    # ���� velocity heatmap ������������������������������������������������������������������������������������������������������������
    im = ax1.pcolormesh(dist_km, np.arange(n_t), V_mat,
                         cmap="RdYlBu_r", vmin=vlo, vmax=vhi, shading="auto")
    cb = fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.025)
    cb.set_label(f"Speed ({unit})", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    # Contour lines
    try:
        cs = ax1.contour(dist_km, np.arange(n_t), V_mat,
                          levels=6, colors="white", linewidths=0.5, alpha=0.4)
        ax1.clabel(cs, fmt="%.0f", fontsize=7, colors="white")
    except Exception:
        pass

    ax1.set_yticks(np.arange(n_t))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_ylabel("Month", fontsize=11)
    ax1.set_title("Hovm?ller diagram �� monthly ice velocity", fontsize=12)
    ax1.tick_params(labelbottom=False)

    for xc, lbl in zip(ctrl_dist_km, ctrl_labels):
        ax1.axvline(xc, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.text(xc + 0.2, -0.5, lbl,
                  ha="left", va="top", fontsize=8, color="white", fontweight="bold")

    # ���� mean error bar ����������������������������������������������������������������������������������������������������������������
    mean_err = np.nanmean(Err_mat, axis=0)
    ax2.fill_between(dist_km,
                      np.nanpercentile(Err_mat, 25, axis=0),
                      np.nanpercentile(Err_mat, 75, axis=0),
                      color="#e88c2e", alpha=0.3, linewidth=0)
    ax2.plot(dist_km, mean_err, color="#e88c2e", linewidth=1.5,
              label="Mean ��")
    ax2.set_ylabel(f"�� ({unit})", fontsize=9)
    ax2.set_xlabel("Distance along profile (km)", fontsize=11)
    ax2.set_xlim(0, dist_km[-1])
    ax2.legend(fontsize=8)
    for xc in ctrl_dist_km:
        ax2.axvline(xc, color="#888", linewidth=0.6, linestyle="--", alpha=0.4)

    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved (hovmoller) -> {output_path}")


# ���� Mode 2: Mean + anomaly ��������������������������������������������������������������������������������������������������������

def plot_mean_anomaly(dist_km, ctrl_dist_km, ctrl_labels, profiles,
                       output_path, unit="m/yr", vmin=None, vmax=None):
    """
    Top   : multi-month mean �� std; individual months as thin coloured lines.
    Bottom: per-month anomaly (month ? mean), colour-coded by season.
    """
    n_t   = len(profiles)
    V_mat = np.full((n_t, len(dist_km)), np.nan)
    for i, (_, v, _) in enumerate(profiles):
        V_mat[i] = v
    mean_v = np.nanmean(V_mat, axis=0)
    std_v  = np.nanstd(V_mat,  axis=0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [2, 1.2], "hspace": 0.1},
        sharex=True)

    # ���� mean �� std ������������������������������������������������������������������������������������������������������������������������
    ax1.fill_between(dist_km, mean_v - std_v, mean_v + std_v,
                      color="steelblue", alpha=0.22)
    for lbl, v, _ in profiles:
        mo  = lbl.split("-")[1]
        col = SEASON_COLOR.get(mo, "#888")
        ok  = np.isfinite(v)
        ax1.plot(dist_km[ok], v[ok], color=col, linewidth=0.8, alpha=0.55)
    ax1.plot(dist_km, mean_v, color="white", linewidth=2.2,
              label="Multi-month mean", zorder=5)

    season_legend = [
        Patch(color="#e74c3c", label="Summer (DJF)"),
        Patch(color="#e67e22", label="Autumn (MAM)"),
        Patch(color="#3498db", label="Winter (JJA)"),
        Patch(color="#2ecc71", label="Spring (SON)"),
        plt.Line2D([0],[0], color="white", lw=2, label="Mean"),
        Patch(color="steelblue", alpha=0.4, label="��1�� range"),
    ]
    ax1.legend(handles=season_legend, fontsize=8, loc="upper left")
    ax1.set_ylabel(f"Ice speed ({unit})", fontsize=11)
    ax1.set_title("Monthly velocity: mean �� variability + seasonal anomaly", fontsize=12)
    if vmin is not None: ax1.set_ylim(bottom=vmin)
    if vmax is not None: ax1.set_ylim(top=vmax)

    for xc, lbl in zip(ctrl_dist_km, ctrl_labels):
        ax1.axvline(xc, color="#888", linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.text(xc, ax1.get_ylim()[1], f" {lbl}", ha="left", va="top",
                  fontsize=9, color="#aaa", fontweight="bold")

    # ���� anomaly ������������������������������������������������������������������������������������������������������������������������������
    ax2.fill_between(dist_km, -std_v, std_v,
                      color="steelblue", alpha=0.12, label="��1��")
    for lbl, v, _ in profiles:
        mo   = lbl.split("-")[1]
        col  = SEASON_COLOR.get(mo, "#888")
        anom = v - mean_v
        ok   = np.isfinite(anom)
        ax2.plot(dist_km[ok], anom[ok], color=col, linewidth=0.9, alpha=0.75)
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_ylabel(f"Anomaly ({unit})", fontsize=10)
    ax2.set_xlabel("Distance along profile (km)", fontsize=11)
    for xc in ctrl_dist_km:
        ax2.axvline(xc, color="#888", linewidth=0.6, linestyle="--", alpha=0.4)

    for ax in (ax1, ax2):
        ax.set_xlim(0, dist_km[-1])
        ax.grid(alpha=0.2)

    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved (mean_anomaly) -> {output_path}")


# ���� Mode 3: Small multiples ������������������������������������������������������������������������������������������������������

def plot_small_multiples(dist_km, ctrl_dist_km, ctrl_labels, profiles,
                          output_path, unit="m/yr", vmin=None, vmax=None,
                          ncols=4):
    n_months = len(profiles)
    ncols    = min(ncols, n_months)
    nrows    = (n_months + ncols - 1) // ncols

    all_v  = np.concatenate([v for _, v, _ in profiles])
    vlo    = vmin if vmin is not None else np.nanpercentile(all_v, 1)
    vhi    = vmax if vmax is not None else np.nanpercentile(all_v, 99)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 2.5),
                              sharex=True, sharey=True)
    axes_flat = np.array(axes).flatten() if n_months > 1 else [axes]

    for idx, (lbl, v, err) in enumerate(profiles):
        ax  = axes_flat[idx]
        mo  = lbl.split("-")[1]
        col = SEASON_COLOR.get(mo, "#88aacc")

        valid = np.isfinite(v) & np.isfinite(err)
        if np.any(valid):
            dk = dist_km[valid]; vv = v[valid]; ee = err[valid]
            ax.fill_between(dk, vv - ee, vv + ee,
                             color=col, alpha=0.3, linewidth=0)
            ax.plot(dk, vv, color=col, linewidth=1.3)

        for xc, clbl in zip(ctrl_dist_km, ctrl_labels):
            ax.axvline(xc, color="#444", linewidth=0.6, linestyle="--")
            ax.text(xc, vhi * 0.97, clbl, ha="center", va="top",
                     fontsize=6, color="#777")

        ax.set_title(lbl, fontsize=9, color="white", pad=3)
        ax.set_ylim(vlo, vhi)
        ax.set_xlim(0, dist_km[-1])
        ax.grid(alpha=0.15)

    for idx in range(n_months, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.text(0.5, 0.01, "Distance along profile (km)",
              ha="center", fontsize=11)
    fig.text(0.01, 0.5, f"Ice speed ({unit})",
              va="center", rotation="vertical", fontsize=11)
    fig.suptitle("Monthly ice velocity profiles  (shading = ��1\u03c3)",
                  fontsize=13, y=1.01)
    fig.tight_layout(rect=[0.03, 0.03, 1, 1])
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved (small_multiples) -> {output_path}")


# ���� CLI ����������������������������������������������������������������������������������������������������������������������������������������������

def parse_xy(s):
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'x,y', got: {s!r}")
    return float(parts[0]), float(parts[1])


def main():
    ap = argparse.ArgumentParser(
        description="Monthly ice velocity profile visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--data_dir", required=True)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--points_3031", nargs="+", type=parse_xy, metavar="X,Y")
    grp.add_argument("--points_latlon", nargs="+", type=parse_xy, metavar="LON,LAT")

    ap.add_argument("--labels",  nargs="+", default=None)
    ap.add_argument("--n_pts",   type=int,  default=200)
    ap.add_argument("--output",  default="profile_monthly.png")
    ap.add_argument("--unit",    default="m/yr")
    ap.add_argument("--vmin",    type=float, default=None)
    ap.add_argument("--vmax",    type=float, default=None)
    ap.add_argument("--ncols",   type=int,   default=4,
                    help="Columns for small_multiples mode")
    ap.add_argument("--mode", default="hovmoller",
                    choices=["hovmoller","mean_anomaly","small_multiples","all"],
                    help="Visualization mode; 'all' saves three files")
    args = ap.parse_args()

    data_dir    = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Control points
    if args.points_latlon:
        lons = [p[0] for p in args.points_latlon]
        lats = [p[1] for p in args.points_latlon]
        xs, ys = latlon_to_3031(lons, lats)
        ctrl_xy = list(zip(xs, ys))
        print(f"Converted {len(ctrl_xy)} lat/lon points to EPSG:3031")
    else:
        ctrl_xy = list(args.points_3031)

    if len(ctrl_xy) < 2:
        sys.exit("Error: at least 2 control points required.")

    ctrl_labels = args.labels[:len(ctrl_xy)] if args.labels else \
                  [chr(65 + i) for i in range(len(ctrl_xy))]

    px, py, dist_km, ctrl_dist_km = build_profile(ctrl_xy, args.n_pts)
    print(f"Profile: {len(px)} pts,  {dist_km[-1]:.2f} km")

    pairs = discover_monthly_pairs(data_dir)
    if not pairs:
        sys.exit(f"No monthly pairs found in {data_dir}")
    print(f"Found {len(pairs)} monthly pairs ({pairs[0][0]} �C {pairs[-1][0]})")

    profiles = []
    for label, v_path, err_path in pairs:
        print(f"  Sampling {label} ...", end="", flush=True)
        v   = sample_tif(v_path,   px, py)
        err = sample_tif(err_path, px, py)
        err[err <= 0] = np.nan
        print(f"  {np.sum(np.isfinite(v))/len(v)*100:.0f}% valid")
        profiles.append((label, v, err))

    kw = dict(unit=args.unit, vmin=args.vmin, vmax=args.vmax)
    stem = output_path.stem

    modes = ["hovmoller","mean_anomaly","small_multiples"] \
            if args.mode == "all" else [args.mode]

    for mode in modes:
        out = output_path.parent / f"{stem}_{mode}.png" \
              if args.mode == "all" else output_path
        if mode == "hovmoller":
            plot_hovmoller(dist_km, ctrl_dist_km, ctrl_labels, profiles, out, **kw)
        elif mode == "mean_anomaly":
            plot_mean_anomaly(dist_km, ctrl_dist_km, ctrl_labels, profiles, out, **kw)
        elif mode == "small_multiples":
            plot_small_multiples(dist_km, ctrl_dist_km, ctrl_labels, profiles,
                                  out, ncols=args.ncols, **kw)

if __name__ == "__main__":
    main()
