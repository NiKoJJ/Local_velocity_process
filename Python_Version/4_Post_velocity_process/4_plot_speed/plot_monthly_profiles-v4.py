#!/usr/bin/env python3
"""
plot_monthly_profiles.py  (v4)
==============================
Ice velocity profiles along a transect, with error shading.
Supports **both monthly** (YYYY-MM_v.tif) and **yearly** (YYYY_v.tif) data.

New in v4
---------
* Auto-detects monthly vs yearly files; or force with --freq monthly|yearly
* Yearly mode uses a perceptually-uniform sequential colour ramp (viridis) 
  instead of season colours
* All lbl.split("-") calls are guarded for yearly labels
* plot_hovmoller y-axis label adapts (Month / Year)
* plot_mean_anomaly and plot_small_multiples adapt titles & colours

Usage
-----
# Monthly (unchanged):
python plot_monthly_profiles.py \\
    --data_dir ./monthly_V_Error \\
    --points_3031 1034892,-2059436 1087475,-2116546 \\
    --labels A B --n_pts 50 --output ./plot_res/monthly.png --mode all

# Yearly (new):
python plot_monthly_profiles.py \\
    --data_dir ./yearly_V_Error \\
    --points_3031 1034892,-2059436 1087475,-2116546 \\
    --labels A B --n_pts 50 --output ./plot_res/yearly.png --mode all

# Force frequency:
    --freq yearly          # override auto-detect
"""

import argparse, re, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
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

# Season colours used for monthly data
SEASON_COLOR = {
    "12":"#e74c3c","01":"#e74c3c","02":"#e74c3c",
    "03":"#e67e22","04":"#e67e22","05":"#e67e22",
    "06":"#3498db","07":"#3498db","08":"#3498db",
    "09":"#2ecc71","10":"#2ecc71","11":"#2ecc71",
}


# ─────────────────────────────────────────────────────────────────────────────
# Colour helper for yearly data
# ─────────────────────────────────────────────────────────────────────────────

def make_year_colormap(labels):
    """
    Return a dict {label: hex_color} using a viridis ramp over the year range.
    Works even when labels contains non-year strings (falls back to index).
    """
    cmap = cm.get_cmap("plasma", len(labels))
    return {lbl: matplotlib.colors.to_hex(cmap(i)) for i, lbl in enumerate(labels)}


def label_color(lbl, freq, year_cmap):
    """Return the appropriate colour for a profile label."""
    if freq == "monthly":
        mo = lbl.split("-")[1] if "-" in lbl else "01"
        return SEASON_COLOR.get(mo, "#888")
    else:
        return year_cmap.get(lbl, "#88aacc")


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_monthly_pairs(data_dir):
    """Find YYYY-MM_v.tif / YYYY-MM_v_err.tif pairs."""
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


def discover_yearly_pairs(data_dir):
    """Find YYYY_v.tif / YYYY_v_err.tif pairs."""
    pattern = re.compile(r'^(\d{4})_v\.tif$', re.IGNORECASE)
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


def discover_pairs(data_dir, freq="auto"):
    """
    Discover velocity/error pairs.  freq can be 'auto', 'monthly', 'yearly'.
    Auto-detect: if any YYYY-MM_v.tif found → monthly; else try yearly.
    Returns (pairs, detected_freq).
    """
    if freq in ("auto", "monthly"):
        pairs = discover_monthly_pairs(data_dir)
        if pairs:
            return pairs, "monthly"
        if freq == "monthly":
            return [], "monthly"

    if freq in ("auto", "yearly"):
        pairs = discover_yearly_pairs(data_dir)
        if pairs:
            return pairs, "yearly"

    return [], freq if freq != "auto" else "monthly"


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def latlon_to_3031(lons, lats):
    xs, ys = warp_transform(CRS.from_epsg(4326), CRS.from_epsg(3031), lons, lats)
    return list(xs), list(ys)


def build_profile(ctrl_xy, n_pts_per_seg):
    ctrl_xy  = np.array(ctrl_xy)
    n_segs   = len(ctrl_xy) - 1
    px, py   = [], []
    ctrl_idx = [0]
    for s in range(n_segs):
        x1, y1 = ctrl_xy[s]
        x2, y2 = ctrl_xy[s + 1]
        n = n_pts_per_seg if np.isscalar(n_pts_per_seg) else n_pts_per_seg[s]
        sx = np.linspace(x1, x2, n)
        sy = np.linspace(y1, y2, n)
        if s > 0:
            sx = sx[1:]; sy = sy[1:]
        px.extend(sx); py.extend(sy)
        ctrl_idx.append(len(px) - 1)
    px, py  = np.array(px), np.array(py)
    dist_km = np.concatenate([[0],
        np.cumsum(np.sqrt(np.diff(px)**2 + np.diff(py)**2))]) / 1e3
    return px, py, dist_km, dist_km[ctrl_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Raster sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_tif(tif_path, px, py):
    """Bilinear interpolation onto profile points. Returns float32 array."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_profile(arr, method, window):
    """
    Smooth a 1-D profile that may contain NaN values.

    Methods
    -------
    none        no smoothing
    moving_avg  boxcar rolling mean  (fast, obvious)
    savgol      Savitzky-Golay       (preserves peaks best, recommended)
    gaussian    Gaussian kernel      (gentler roll-off)

    NaN handling: all methods fill gaps via linear interpolation before
    smoothing, then restore NaN where the original was NaN.
    """
    if method == "none" or window <= 1:
        return arr.copy()

    out   = arr.copy()
    valid = np.isfinite(arr)
    if valid.sum() < 4:
        return out

    x_all      = np.arange(len(arr))
    arr_filled = np.interp(x_all, x_all[valid], arr[valid])

    if method == "moving_avg":
        kernel   = np.ones(window) / window
        smoothed = np.convolve(arr_filled, kernel, mode="same")
        norm     = np.convolve(np.ones_like(arr_filled), kernel, mode="same")
        smoothed = smoothed / norm

    elif method == "savgol":
        w = window if window % 2 == 1 else window + 1
        polyorder = min(3, w - 1)
        smoothed  = savgol_filter(arr_filled, window_length=w, polyorder=polyorder)

    elif method == "gaussian":
        sigma    = window / 4.0
        smoothed = gaussian_filter1d(arr_filled, sigma=sigma)

    else:
        return out

    out[valid] = smoothed[valid].astype(np.float32)
    return out


def apply_smooth(profiles, method, window):
    """Return new profiles list with smoothed v and err arrays."""
    if method == "none":
        return profiles
    smoothed = []
    for label, v, err in profiles:
        sv   = smooth_profile(v,   method, window)
        serr = smooth_profile(err, method, window)
        smoothed.append((label, sv, serr))
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

def _split_segments(mask):
    """Split boolean mask into (start, stop) index tuples for True runs."""
    segs   = []
    in_seg = False
    for i, val in enumerate(mask):
        if val and not in_seg:
            start = i; in_seg = True
        elif not val and in_seg:
            segs.append((start, i)); in_seg = False
    if in_seg:
        segs.append((start, len(mask)))
    return segs


def draw_band_and_line(ax, dist_km, v, err, color, lw=1.3, alpha_fill=0.3,
                       nan_policy="skip", interp_max_gap=5):
    """
    Draw error band (±1σ) and velocity line independently.

    nan_policy : "skip"   — draw each contiguous valid segment separately
                 "interp" — linearly interpolate over NaN runs < interp_max_gap
    """

    def _interp_short_gaps(arr, max_gap):
        out   = arr.copy().astype(float)
        valid = np.isfinite(out)
        if valid.sum() < 2:
            return out
        x_all  = np.arange(len(out))
        filled = np.interp(x_all, x_all[valid], out[valid])
        nan_mask = ~valid
        i = 0
        while i < len(nan_mask):
            if nan_mask[i]:
                j = i
                while j < len(nan_mask) and nan_mask[j]:
                    j += 1
                if j - i <= max_gap:
                    out[i:j] = filled[i:j]
                i = j
            else:
                i += 1
        return out

    v_work   = v.copy().astype(float)
    err_work = err.copy().astype(float)

    if nan_policy == "interp":
        v_work   = _interp_short_gaps(v_work,   interp_max_gap)
        err_work = _interp_short_gaps(err_work, interp_max_gap)

    ok_v   = np.isfinite(v_work)
    ok_err = np.isfinite(v_work) & np.isfinite(err_work) & (err_work > 0)

    for s, e in _split_segments(ok_err):
        if e - s < 2:
            continue
        sl = slice(s, e)
        ax.fill_between(dist_km[sl],
                         v_work[sl] - err_work[sl],
                         v_work[sl] + err_work[sl],
                         color=color, alpha=alpha_fill, linewidth=0)

    for s, e in _split_segments(ok_v):
        if e - s < 2:
            continue
        sl = slice(s, e)
        ax.plot(dist_km[sl], v_work[sl], color=color, linewidth=lw)


def add_ctrl_lines(ax, ctrl_dist_km, ctrl_labels, top_y=None):
    for xc, lbl in zip(ctrl_dist_km, ctrl_labels):
        ax.axvline(xc, color="#888", linewidth=0.7, linestyle="--", alpha=0.45)
        if top_y is not None:
            ax.text(xc, top_y, f" {lbl}", ha="left", va="top",
                     fontsize=8, color="#aaa", fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: Hovmöller
# ─────────────────────────────────────────────────────────────────────────────

def plot_hovmoller(dist_km, ctrl_dist_km, ctrl_labels, profiles,
                   output_path, freq="monthly", unit="m/yr",
                   vmin=None, vmax=None):
    labels   = [p[0] for p in profiles]
    n_t, n_d = len(profiles), len(dist_km)
    V_mat    = np.full((n_t, n_d), np.nan)
    Err_mat  = np.full((n_t, n_d), np.nan)
    for i, (_, v, err) in enumerate(profiles):
        V_mat[i]   = v
        Err_mat[i] = err

    vlo = vmin if vmin is not None else np.nanpercentile(V_mat, 2)
    vhi = vmax if vmax is not None else np.nanpercentile(V_mat, 98)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})

    im = ax1.pcolormesh(dist_km, np.arange(n_t), V_mat,
                         cmap="RdYlBu_r", vmin=vlo, vmax=vhi, shading="auto")
    cb = fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.025)
    cb.set_label(f"Speed ({unit})", fontsize=10)
    cb.ax.tick_params(labelsize=8)
    try:
        cs = ax1.contour(dist_km, np.arange(n_t), V_mat,
                          levels=6, colors="white", linewidths=0.5, alpha=0.4)
        ax1.clabel(cs, fmt="%.0f", fontsize=7, colors="white")
    except Exception:
        pass

    ax1.set_yticks(np.arange(n_t))
    ax1.set_yticklabels(labels, fontsize=8)
    time_label = "Month" if freq == "monthly" else "Year"
    ax1.set_ylabel(time_label, fontsize=11)
    ax1.set_title(
        f"Hovmöller diagram — {'monthly' if freq == 'monthly' else 'yearly'} ice velocity",
        fontsize=12)
    ax1.tick_params(labelbottom=False)
    for xc, lbl in zip(ctrl_dist_km, ctrl_labels):
        ax1.axvline(xc, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax1.text(xc + 0.2, -0.5, lbl,
                  ha="left", va="top", fontsize=8, color="white", fontweight="bold")

    mean_err = np.nanmean(Err_mat, axis=0)
    ax2.fill_between(dist_km,
                      np.nanpercentile(Err_mat, 25, axis=0),
                      np.nanpercentile(Err_mat, 75, axis=0),
                      color="#e88c2e", alpha=0.3, linewidth=0)
    ax2.plot(dist_km, mean_err, color="#e88c2e", linewidth=1.5, label="Mean σ")
    ax2.set_ylabel(f"σ ({unit})", fontsize=9)
    ax2.set_xlabel("Distance along profile (km)", fontsize=11)
    ax2.set_xlim(0, dist_km[-1]); ax2.legend(fontsize=8)
    for xc in ctrl_dist_km:
        ax2.axvline(xc, color="#888", linewidth=0.6, linestyle="--", alpha=0.4)

    ax1.set_xlim(0, dist_km[-1])
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved (hovmoller) -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: Mean + anomaly
# ─────────────────────────────────────────────────────────────────────────────

def plot_mean_anomaly(dist_km, ctrl_dist_km, ctrl_labels, profiles,
                       output_path, freq="monthly", unit="m/yr",
                       vmin=None, vmax=None):
    n_t   = len(profiles)
    V_mat = np.full((n_t, len(dist_km)), np.nan)
    for i, (_, v, _) in enumerate(profiles):
        V_mat[i] = v
    mean_v = np.nanmean(V_mat, axis=0)
    std_v  = np.nanstd(V_mat,  axis=0)

    labels     = [p[0] for p in profiles]
    year_cmap  = make_year_colormap(labels)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [2, 1.2], "hspace": 0.1},
        sharex=True)

    ax1.fill_between(dist_km, mean_v - std_v, mean_v + std_v,
                      color="steelblue", alpha=0.22)
    for lbl, v, _ in profiles:
        col = label_color(lbl, freq, year_cmap)
        ok  = np.isfinite(v)
        ax1.plot(dist_km[ok], v[ok], color=col, linewidth=0.8, alpha=0.55)
    ax1.plot(dist_km, mean_v, color="white", linewidth=2.2, zorder=5)

    if freq == "monthly":
        legend_handles = [
            Patch(color="#e74c3c", label="Summer (DJF)"),
            Patch(color="#e67e22", label="Autumn (MAM)"),
            Patch(color="#3498db", label="Winter (JJA)"),
            Patch(color="#2ecc71", label="Spring (SON)"),
            plt.Line2D([0],[0], color="white", lw=2, label="Mean"),
            Patch(color="steelblue", alpha=0.4, label="±1σ range"),
        ]
        time_label = "monthly"
    else:
        # Year-based legend: one entry per year
        legend_handles = [
            Patch(color=year_cmap[lbl], label=lbl) for lbl in labels
        ]
        legend_handles += [
            plt.Line2D([0],[0], color="white", lw=2, label="Mean"),
            Patch(color="steelblue", alpha=0.4, label="±1σ range"),
        ]
        time_label = "yearly"

    ax1.legend(handles=legend_handles, fontsize=8, loc="upper left",
               ncol=2 if freq == "yearly" and len(labels) > 6 else 1)
    ax1.set_ylabel(f"Ice speed ({unit})", fontsize=11)
    ax1.set_title(
        f"{'Monthly' if freq == 'monthly' else 'Yearly'} velocity: mean ± variability + anomaly",
        fontsize=12)
    if vmin is not None: ax1.set_ylim(bottom=vmin)
    if vmax is not None: ax1.set_ylim(top=vmax)
    add_ctrl_lines(ax1, ctrl_dist_km, ctrl_labels, top_y=ax1.get_ylim()[1])

    for lbl, v, _ in profiles:
        col  = label_color(lbl, freq, year_cmap)
        anom = v - mean_v
        ok   = np.isfinite(anom)
        ax2.plot(dist_km[ok], anom[ok], color=col, linewidth=0.9, alpha=0.75)
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.fill_between(dist_km, -std_v, std_v, color="steelblue", alpha=0.12)
    ax2.set_ylabel(f"Anomaly ({unit})", fontsize=10)
    ax2.set_xlabel("Distance along profile (km)", fontsize=11)
    add_ctrl_lines(ax2, ctrl_dist_km, ctrl_labels)

    for ax in (ax1, ax2):
        ax.set_xlim(0, dist_km[-1]); ax.grid(alpha=0.2)

    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved (mean_anomaly) -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3: Small multiples
# ─────────────────────────────────────────────────────────────────────────────

def plot_small_multiples(dist_km, ctrl_dist_km, ctrl_labels, profiles,
                          output_path, freq="monthly", unit="m/yr",
                          vmin=None, vmax=None, ncols=4):
    n_panels = len(profiles)
    ncols    = min(ncols, n_panels)
    nrows    = (n_panels + ncols - 1) // ncols

    labels    = [p[0] for p in profiles]
    year_cmap = make_year_colormap(labels)

    all_v = np.concatenate([v for _, v, _ in profiles])
    vlo   = vmin if vmin is not None else np.nanpercentile(all_v, 1)
    vhi   = vmax if vmax is not None else np.nanpercentile(all_v, 99)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 2.5),
                              sharex=True, sharey=True)
    axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

    for idx, (lbl, v, err) in enumerate(profiles):
        ax  = axes_flat[idx]
        col = label_color(lbl, freq, year_cmap)

        draw_band_and_line(ax, dist_km, v, err, col, lw=1.3, alpha_fill=0.35)

        for xc, clbl in zip(ctrl_dist_km, ctrl_labels):
            ax.axvline(xc, color="#444", linewidth=0.6, linestyle="--")
            ax.text(xc, vhi * 0.97, clbl, ha="center", va="top",
                     fontsize=6, color="#777")

        ax.set_title(lbl, fontsize=9, color="white", pad=3)
        ax.set_ylim(vlo, vhi)
        ax.set_xlim(0, dist_km[-1])
        ax.grid(alpha=0.15)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.text(0.5, 0.01, "Distance along profile (km)", ha="center", fontsize=11)
    fig.text(0.01, 0.5, f"Ice speed ({unit})",
              va="center", rotation="vertical", fontsize=11)
    time_word = "Monthly" if freq == "monthly" else "Yearly"
    fig.suptitle(f"{time_word} ice velocity profiles  (shading = ±1σ)",
                  fontsize=13, y=1.01)
    fig.tight_layout(rect=[0.03, 0.03, 1, 1])
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved (small_multiples) -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_xy(s):
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'x,y', got: {s!r}")
    return float(parts[0]), float(parts[1])


def main():
    ap = argparse.ArgumentParser(
        description="Monthly/yearly ice velocity profile visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--data_dir", required=True,
                    help="Directory with velocity .tif files "
                         "(YYYY-MM_v.tif for monthly, YYYY_v.tif for yearly)")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--points_3031",  nargs="+", type=parse_xy, metavar="X,Y",
                     help="Control points in EPSG:3031 metres")
    grp.add_argument("--points_latlon",nargs="+", type=parse_xy, metavar="LON,LAT",
                     help="Control points as lon,lat WGS-84")

    ap.add_argument("--labels",  nargs="+", default=None)
    ap.add_argument("--n_pts",   type=int,  default=200,
                    help="Sample points per segment")
    ap.add_argument("--output",  default="profile.png")
    ap.add_argument("--unit",    default="m/yr")
    ap.add_argument("--vmin",    type=float, default=None)
    ap.add_argument("--vmax",    type=float, default=None)
    ap.add_argument("--ncols",   type=int,   default=4)
    ap.add_argument("--mode",    default="hovmoller",
                    choices=["hovmoller","mean_anomaly","small_multiples","all"])
    ap.add_argument("--freq",    default="auto",
                    choices=["auto","monthly","yearly"],
                    help="File frequency: auto-detect, or force monthly/yearly")

    # ── smoothing ─────────────────────────────────────────────────────────────
    ap.add_argument("--smooth", default="savgol",
                    choices=["none","moving_avg","savgol","gaussian"],
                    help=(
                        "Profile smoothing:\n"
                        "  none       — raw values\n"
                        "  savgol     — Savitzky-Golay (recommended)\n"
                        "  moving_avg — boxcar rolling mean\n"
                        "  gaussian   — Gaussian kernel"))
    ap.add_argument("--smooth_window", type=int, default=11,
                    help="Smoothing window size in pixels (odd number for savgol)")

    args = ap.parse_args()

    data_dir    = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.points_latlon:
        lons = [p[0] for p in args.points_latlon]
        lats = [p[1] for p in args.points_latlon]
        xs, ys = latlon_to_3031(lons, lats)
        ctrl_xy = list(zip(xs, ys))
        print(f"Converted {len(ctrl_xy)} lat/lon -> EPSG:3031")
    else:
        ctrl_xy = list(args.points_3031)

    if len(ctrl_xy) < 2:
        sys.exit("Error: at least 2 control points required.")

    ctrl_labels = args.labels[:len(ctrl_xy)] if args.labels else \
                  [chr(65 + i) for i in range(len(ctrl_xy))]

    px, py, dist_km, ctrl_dist_km = build_profile(ctrl_xy, args.n_pts)
    print(f"Profile: {len(px)} pts, {dist_km[-1]:.2f} km")

    # ── discover files ──────────────────────────────────────────────────────
    pairs, freq = discover_pairs(data_dir, args.freq)
    if not pairs:
        sys.exit(
            f"No {'monthly' if args.freq == 'monthly' else 'yearly'} pairs found in {data_dir}.\n"
            f"  Monthly pattern : YYYY-MM_v.tif + YYYY-MM_v_err.tif\n"
            f"  Yearly  pattern : YYYY_v.tif    + YYYY_v_err.tif\n"
            f"  Files found     : {[p.name for p in sorted(data_dir.iterdir()) if p.suffix == '.tif'][:10]}")
    print(f"Detected frequency : {freq}")
    print(f"Found {len(pairs)} {freq} pairs ({pairs[0][0]} – {pairs[-1][0]})")

    # ── sample rasters ──────────────────────────────────────────────────────
    profiles_raw = []
    for label, v_path, err_path in pairs:
        print(f"  Sampling {label} ...", end="", flush=True)
        v   = sample_tif(v_path,   px, py)
        err = sample_tif(err_path, px, py)
        err[~np.isfinite(err)] = np.nan
        err[err < 0]           = np.nan
        err[err == 0]          = np.nan
        pct     = np.sum(np.isfinite(v))   / len(v)   * 100
        pct_err = np.sum(np.isfinite(err)) / len(err) * 100
        print(f"  v={pct:.0f}%  err={pct_err:.0f}%")
        profiles_raw.append((label, v, err))

    # ── smoothing ──────────────────────────────────────────────────────────
    if args.smooth != "none":
        print(f"\nSmoothing: method={args.smooth}, window={args.smooth_window}")
    profiles = apply_smooth(profiles_raw, args.smooth, args.smooth_window)

    kw    = dict(freq=freq, unit=args.unit, vmin=args.vmin, vmax=args.vmax)
    stem  = output_path.stem
    modes = (["hovmoller","mean_anomaly","small_multiples"]
             if args.mode == "all" else [args.mode])

    for mode in modes:
        out = (output_path.parent / f"{stem}_{mode}.png"
               if args.mode == "all" else output_path)
        if mode == "hovmoller":
            plot_hovmoller(dist_km, ctrl_dist_km, ctrl_labels, profiles, out, **kw)
        elif mode == "mean_anomaly":
            plot_mean_anomaly(dist_km, ctrl_dist_km, ctrl_labels, profiles, out, **kw)
        elif mode == "small_multiples":
            plot_small_multiples(dist_km, ctrl_dist_km, ctrl_labels, profiles,
                                  out, ncols=args.ncols, **kw)


if __name__ == "__main__":
    main()
