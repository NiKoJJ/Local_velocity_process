#!/usr/bin/env python3
"""
plot_profile_extras.py
======================
Three supplementary profile visualization modes for monthly ice velocity data:

  ridge         Stacked offset profiles — shows shape variation across months
  fan           Percentile fan (P5/P25/P50/P75/P95) — replaces messy lines
  delta         Month-on-month ΔV waterfall — highlights acceleration events

Input directory must contain:  YYYY-MM_v.tif  and  YYYY-MM_v_err.tif

Usage
-----
python plot_profile_extras.py \
    --data_dir ./monthly_V_Error \
    --points_latlon 158.5,-68.2 160.0,-67.8 162.0,-67.5 \
    --labels A B C \
    --mode all \
    --smooth savgol --smooth_window 15 \
    --output extras.png
"""

import argparse, re, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

# ── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0d0d", "axes.facecolor": "#111",
    "axes.edgecolor":   "#444",    "text.color":     "#ddd",
    "axes.labelcolor":  "#ccc",    "xtick.color":    "#999",
    "ytick.color":      "#999",    "grid.color":     "#2a2a2a",
    "legend.facecolor": "#1a1a1a", "legend.edgecolor":"#444",
    "legend.labelcolor":"#ccc",
})



def _split_segments(mask):
    """Return list of (start, stop) for each contiguous True run in mask."""
    segs = []
    in_seg = False
    for i, val in enumerate(mask):
        if val and not in_seg:
            start = i; in_seg = True
        elif not val and in_seg:
            segs.append((start, i)); in_seg = False
    if in_seg:
        segs.append((start, len(mask)))
    return segs

SEASON_COLOR = {
    "12":"#e74c3c","01":"#e74c3c","02":"#e74c3c",
    "03":"#e67e22","04":"#e67e22","05":"#e67e22",
    "06":"#3498db","07":"#3498db","08":"#3498db",
    "09":"#2ecc71","10":"#2ecc71","11":"#2ecc71",
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers  (same as base script)
# ─────────────────────────────────────────────────────────────────────────────

def discover_pairs(data_dir):
    pat = re.compile(r'^(\d{4}-\d{2})_v\.tif$', re.IGNORECASE)
    pairs = []
    for p in sorted(data_dir.iterdir()):
        m = pat.match(p.name)
        if not m: continue
        label = m.group(1)
        ep    = data_dir / f"{label}_v_err.tif"
        if ep.exists():
            pairs.append((label, p, ep))
    return sorted(pairs, key=lambda x: x[0])


def latlon_to_3031(lons, lats):
    xs, ys = warp_transform(CRS.from_epsg(4326), CRS.from_epsg(3031), lons, lats)
    return list(xs), list(ys)


def build_profile(ctrl_xy, n_pts):
    ctrl_xy = np.array(ctrl_xy)
    px, py  = [], []
    cidx    = [0]
    for s in range(len(ctrl_xy) - 1):
        x1,y1 = ctrl_xy[s]; x2,y2 = ctrl_xy[s+1]
        sx = np.linspace(x1, x2, n_pts)
        sy = np.linspace(y1, y2, n_pts)
        if s > 0: sx, sy = sx[1:], sy[1:]
        px.extend(sx); py.extend(sy)
        cidx.append(len(px)-1)
    px, py   = np.array(px), np.array(py)
    dist_km  = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(px)**2+np.diff(py)**2))]) / 1e3
    return px, py, dist_km, dist_km[cidx]


def sample_tif(path, px, py):
    with rasterio.open(str(path)) as src:
        d  = src.read(1).astype(np.float64)
        nd = src.nodata
        if nd is not None: d[d == nd] = np.nan
        d[~np.isfinite(d)] = np.nan
        h, w = d.shape
        tf   = src.transform
        xs   = tf.c + tf.a*(np.arange(w)+0.5)
        ys   = tf.f + tf.e*(np.arange(h)+0.5)
        if ys[0] > ys[-1]: ys = ys[::-1]; d = d[::-1]
        interp = RegularGridInterpolator((ys,xs), d, method="linear",
                                          bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([py, px])).astype(np.float32)


def smooth1d(arr, method, win):
    if method == "none" or win <= 1: return arr.copy()
    out   = arr.copy()
    valid = np.isfinite(arr)
    if valid.sum() < 4: return out
    x   = np.arange(len(arr))
    fil = np.interp(x, x[valid], arr[valid])
    if method == "moving_avg":
        k  = np.ones(win)/win
        sm = np.convolve(fil, k, "same") / np.convolve(np.ones_like(fil), k, "same")
    elif method == "savgol":
        w  = win if win%2 else win+1
        sm = savgol_filter(fil, window_length=w, polyorder=min(3,w-1))
    elif method == "gaussian":
        sm = gaussian_filter1d(fil, sigma=win/4)
    else:
        return out
    out[valid] = sm[valid].astype(np.float32)
    return out


def ctrl_vlines(ax, ctrl_dist_km, ctrl_labels, top_y=None):
    for xc, lb in zip(ctrl_dist_km, ctrl_labels):
        ax.axvline(xc, color="#888", lw=0.7, ls="--", alpha=0.5)
        if top_y is not None:
            ax.text(xc, top_y, f" {lb}", ha="left", va="top",
                    fontsize=8, color="#aaa", fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 : Ridge plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_ridge(dist_km, ctrl_dist_km, ctrl_labels, profiles,
               output_path, unit="m/yr", vmin=None, vmax=None):
    """
    Each month's profile is vertically offset so they stack like ridgelines.
    Color encodes season; fill extends down to the baseline of that row.
    """
    n   = len(profiles)
    all_v = np.concatenate([p[1] for p in profiles])
    span  = float(np.nanpercentile(all_v, 98) - np.nanpercentile(all_v, 2))
    step  = span * 0.6          # vertical offset per month

    fig, ax = plt.subplots(figsize=(14, 0.7*n + 2))

    for i, (lbl, v, err) in enumerate(profiles):
        offset = i * step
        mo     = lbl.split("-")[1]
        col    = SEASON_COLOR.get(mo, "#88aacc")
        ok     = np.isfinite(v)
        if ok.sum() < 2: continue

        vv  = v.copy(); vv[~ok] = np.nan
        base = np.full_like(vv, offset)

        # fill from offset baseline down to (offset) — segment by segment
        for s, e in _split_segments(ok):
            if e - s < 2: continue
            sl = slice(s, e)
            ax.fill_between(dist_km[sl], base[sl], vv[sl] + offset,
                            color=col, alpha=0.4, linewidth=0)
        for s, e in _split_segments(ok):
            if e - s < 2: continue
            sl = slice(s, e)
            ax.plot(dist_km[sl], vv[sl]+offset, color=col, lw=1.4)
        ax.text(-0.5, offset + float(np.nanmedian(vv)),
                lbl, ha="right", va="center", fontsize=8, color=col)

    # ctrl lines
    for xc, lb in zip(ctrl_dist_km, ctrl_labels):
        ax.axvline(xc, color="#666", lw=0.7, ls="--", alpha=0.5)
        ax.text(xc, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else step*n*0.98,
                f" {lb}", ha="left", va="top", fontsize=9,
                color="#aaa", fontweight="bold")

    season_legend = [
        Patch(color="#e74c3c", label="Summer (DJF)"),
        Patch(color="#e67e22", label="Autumn (MAM)"),
        Patch(color="#3498db", label="Winter (JJA)"),
        Patch(color="#2ecc71", label="Spring (SON)"),
    ]
    ax.legend(handles=season_legend, fontsize=8, loc="upper right")
    ax.set_xlabel("Distance along profile (km)", fontsize=11)
    ax.set_ylabel(f"Speed offset ({unit})", fontsize=10)
    ax.set_title("Ridge plot — monthly ice velocity profiles", fontsize=12)
    ax.set_xlim(0, dist_km[-1])
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved (ridge) -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 : Percentile fan chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_fan(dist_km, ctrl_dist_km, ctrl_labels, profiles,
             output_path, unit="m/yr", vmin=None, vmax=None):
    """
    Compress all monthly lines into percentile bands (P5/25/50/75/95).
    Far cleaner than 20 overlapping lines; still shows full variability.
    """
    n_t = len(profiles)
    n_d = len(dist_km)
    V   = np.full((n_t, n_d), np.nan, dtype=np.float32)
    for i, (_, v, _) in enumerate(profiles):
        V[i] = v

    p5   = np.nanpercentile(V,  5, axis=0)
    p25  = np.nanpercentile(V, 25, axis=0)
    p50  = np.nanpercentile(V, 50, axis=0)
    p75  = np.nanpercentile(V, 75, axis=0)
    p95  = np.nanpercentile(V, 95, axis=0)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Nested bands: outermost = lightest
    # Draw percentile bands segment-by-segment to handle NaN gaps
    ok95 = np.isfinite(p5)  & np.isfinite(p95)
    ok75 = np.isfinite(p25) & np.isfinite(p75)
    ok50 = np.isfinite(p50)
    for s, e in _split_segments(ok95):
        if e-s < 2: continue
        sl = slice(s,e)
        ax.fill_between(dist_km[sl], p5[sl], p95[sl],
                        color="#3498db", alpha=0.15, linewidth=0,
                        label="P5–P95" if s==_split_segments(ok95)[0][0] else "_")
    for s, e in _split_segments(ok75):
        if e-s < 2: continue
        sl = slice(s,e)
        ax.fill_between(dist_km[sl], p25[sl], p75[sl],
                        color="#3498db", alpha=0.30, linewidth=0,
                        label="P25–P75  (IQR)" if s==_split_segments(ok75)[0][0] else "_")
    for s, e in _split_segments(ok50):
        if e-s < 2: continue
        sl = slice(s,e)
        ax.plot(dist_km[sl], p50[sl], color="#e0e0e0", lw=2.2,
                label="Median (P50)" if s==_split_segments(ok50)[0][0] else "_", zorder=5)
    for arr, ls in [(p5,"--"),(p95,"--")]:
        ok_a = np.isfinite(arr)
        for s, e in _split_segments(ok_a):
            if e-s<2: continue
            ax.plot(dist_km[s:e], arr[s:e], color="#3498db", lw=0.7, alpha=0.6, ls=ls)

    # Individual extreme months (optional: show min/max)
    v_max_month = profiles[int(np.argmax([np.nanmean(p[1]) for p in profiles]))]
    v_min_month = profiles[int(np.argmin([np.nanmean(p[1]) for p in profiles]))]
    for lbl, v, _ in [v_max_month, v_min_month]:
        ok  = np.isfinite(v)
        mo  = lbl.split("-")[1]
        col = SEASON_COLOR.get(mo, "#888")
        ax.plot(dist_km[ok], v[ok], color=col, lw=1.0, alpha=0.7,
                ls=":", label=f"{lbl} (extreme)")

    ctrl_vlines(ax, ctrl_dist_km, ctrl_labels, ax.get_ylim()[1])
    if vmin is not None: ax.set_ylim(bottom=vmin)
    if vmax is not None: ax.set_ylim(top=vmax)

    ax.set_xlabel("Distance along profile (km)", fontsize=11)
    ax.set_ylabel(f"Ice speed ({unit})", fontsize=11)
    ax.set_title("Percentile fan chart — statistical spread of monthly velocity",
                 fontsize=12)
    ax.set_xlim(0, dist_km[-1])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved (fan) -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 : Waterfall / delta map
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta(dist_km, ctrl_dist_km, ctrl_labels, profiles,
               output_path, unit="m/yr", vmin=None, vmax=None):
    """
    Top panel : Hovmöller of ΔV = V_month − V_prev_month
                Red = acceleration, Blue = deceleration.
    Bottom    : cumulative anomaly (V_month − V_first_month) time series
                at five equal-spaced points along the profile.
    """
    labels = [p[0] for p in profiles]
    n_t    = len(profiles)
    n_d    = len(dist_km)

    V = np.full((n_t, n_d), np.nan, dtype=np.float32)
    for i, (_, v, _) in enumerate(profiles):
        V[i] = v

    # Month-on-month delta
    dV      = np.full_like(V, np.nan)
    dV[1:]  = V[1:] - V[:-1]    # shape (n_t, n_d); dV[0] = NaN

    # Cumulative anomaly vs first month
    anom = V - V[0][np.newaxis, :]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.1})

    # ── Hovmöller of delta ────────────────────────────────────────────────────
    vlim = float(np.nanpercentile(np.abs(dV[1:]), 95))
    im   = ax1.pcolormesh(dist_km, np.arange(n_t), dV,
                           cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto")
    cb   = fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.025)
    cb.set_label(f"\u0394V ({unit})", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    ax1.set_yticks(np.arange(n_t))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_ylabel("Month", fontsize=11)
    ax1.set_title("Month-on-month velocity change \u0394V  (red = acceleration)",
                  fontsize=12)
    ax1.tick_params(labelbottom=False)

    for xc, lb in zip(ctrl_dist_km, ctrl_labels):
        ax1.axvline(xc, color="white", lw=0.8, ls="--", alpha=0.45)
        ax1.text(xc+0.3, -0.5, lb, ha="left", va="top",
                 fontsize=8, color="white", fontweight="bold")

    # ── Cumulative anomaly time series at 5 points ────────────────────────────
    n_pts  = 5
    pt_idx = np.round(np.linspace(0, n_d-1, n_pts)).astype(int)
    cmap_t = plt.cm.get_cmap("plasma", n_pts)

    for j, idx in enumerate(pt_idx):
        col  = cmap_t(j)
        ok   = np.isfinite(anom[:, idx])
        xpos = [i for i,o in enumerate(ok) if o]
        ypos = anom[ok, idx]
        ax2.plot(xpos, ypos, color=col, lw=1.5, marker="o", ms=3,
                 label=f"{dist_km[idx]:.1f} km")

    ax2.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xticks(np.arange(n_t))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel(f"Cum. anomaly ({unit})", fontsize=10)
    ax2.set_xlabel("Month", fontsize=11)
    ax2.legend(title="profile km", fontsize=7, ncol=n_pts,
               loc="upper left", title_fontsize=7)
    ax2.grid(alpha=0.2)
    ax2.set_xlim(-0.5, n_t-0.5)

    ax1.set_xlim(0, dist_km[-1])

    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved (delta) -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_xy(s):
    a, b = s.replace(" ", "").split(",")
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--points_3031",  nargs="+", type=parse_xy, metavar="X,Y")
    grp.add_argument("--points_latlon",nargs="+", type=parse_xy, metavar="LON,LAT")
    ap.add_argument("--labels",        nargs="+", default=None)
    ap.add_argument("--n_pts",         type=int,  default=200)
    ap.add_argument("--output",        default="profile_extras.png")
    ap.add_argument("--unit",          default="m/yr")
    ap.add_argument("--vmin",          type=float, default=None)
    ap.add_argument("--vmax",          type=float, default=None)
    ap.add_argument("--mode", default="all",
                    choices=["ridge","fan","delta","all"])
    ap.add_argument("--smooth", default="savgol",
                    choices=["none","moving_avg","savgol","gaussian"])
    ap.add_argument("--smooth_window", type=int, default=11)
    args = ap.parse_args()

    data_dir    = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.points_latlon:
        lons = [p[0] for p in args.points_latlon]
        lats = [p[1] for p in args.points_latlon]
        xs, ys = latlon_to_3031(lons, lats)
        ctrl_xy = list(zip(xs, ys))
    else:
        ctrl_xy = list(args.points_3031)

    ctrl_labels = args.labels[:len(ctrl_xy)] if args.labels else \
                  [chr(65+i) for i in range(len(ctrl_xy))]

    px, py, dist_km, ctrl_dist_km = build_profile(ctrl_xy, args.n_pts)
    print(f"Profile: {len(px)} pts, {dist_km[-1]:.2f} km")

    pairs = discover_pairs(data_dir)
    if not pairs: sys.exit("No pairs found")
    print(f"Found {len(pairs)} pairs ({pairs[0][0]} – {pairs[-1][0]})")

    profiles = []
    for lbl, vp, ep in pairs:
        print(f"  {lbl} ...", end="", flush=True)
        v   = sample_tif(vp, px, py)
        err = sample_tif(ep, px, py)
        err[~np.isfinite(err)] = np.nan
        err[err <= 0]          = np.nan
        if args.smooth != "none":
            v   = smooth1d(v,   args.smooth, args.smooth_window)
            err = smooth1d(err, args.smooth, args.smooth_window)
        print(f" {np.sum(np.isfinite(v))/len(v)*100:.0f}%")
        profiles.append((lbl, v, err))

    kw   = dict(unit=args.unit, vmin=args.vmin, vmax=args.vmax)
    stem = output_path.stem
    modes = ["ridge","fan","delta"] if args.mode == "all" else [args.mode]

    for mode in modes:
        out = (output_path.parent / f"{stem}_{mode}.png"
               if args.mode == "all" else output_path)
        if   mode == "ridge": plot_ridge(dist_km, ctrl_dist_km, ctrl_labels, profiles, out, **kw)
        elif mode == "fan":   plot_fan  (dist_km, ctrl_dist_km, ctrl_labels, profiles, out, **kw)
        elif mode == "delta": plot_delta(dist_km, ctrl_dist_km, ctrl_labels, profiles, out, **kw)


if __name__ == "__main__":
    main()
