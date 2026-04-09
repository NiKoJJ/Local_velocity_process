#!/usr/bin/env python3
"""
ITS_LIVE Granule Search, Filter, Download & GeoTIFF Export
===========================================================
优化版本 v2.0:
1. 增强地图可视化：自动投影选择 + 双 bbox 显示 + 丰富底图
2. 改进坐标处理：支持 EPSG:3031/4326 自动转换
3. 优化错误处理：更完善的依赖检查和降级方案
"""
import json
import datetime
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import xarray as xr
from pystac_client import Client

# ============================================================
# ① 配置区域
# ============================================================
# ── BBox 配置 ────────────────────────────────────────────────────────────
BBOX_LATLON = [149.5, -69.5, 155.5, -68]  # [150.73, -69.133, 154.13, -68.245]  # WGS-84
# BBOX_PROJ   = [1032344, -2128060, 1084023, -2077435]                # EPSG:3031 (米)
PROJ_EPSG   = 3031
USE_PROJ_BBOX = False  # True → 使用 BBOX_PROJ；False → 使用 BBOX_LATLON

# ── 过滤参数 ────────────────────────────────────────────────────────────
FILTER_CONFIG = {
    'output_dir':        './itslive_download_2020_2026_cube',
    'collections':       ['itslive-granules'],     # 'itslive-cubes'  'itslive-granules'
    'datetime':          '2020-01-01/2026-03-18',
    'max_items':         300000, # 增加数量，确保有足够数据过滤
    'min_valid_pixels':  30,
    'max_date_dt':       33,
    'platforms': [
        'LC04', 'LC05', 'LC07', 'LC08', 'LC09',  # Landsat 系列
        'S1A', 'S1B', 'S1C',                      # Sentinel-1 系列
        'S2A', 'S2B',                              # Sentinel-2 系列
    ],
    'max_workers':       10,
}

# ── 传感器配置（用于地图标题） ─────────────────────────────────────────
SENSOR_CONFIG = {
    'S1A': {'label': 'Sentinel-1A'},
    'S1B': {'label': 'Sentinel-1B'},
    'S1C': {'label': 'Sentinel-1C'},
    'LC08': {'label': 'Landsat-8'},
    'LC09': {'label': 'Landsat-9'},
    'S2A': {'label': 'Sentinel-2A'},
    'S2B': {'label': 'Sentinel-2B'},
}

# ── 导出变量 ────────────────────────────────────────────────────────────
VARIABLES_TO_EXPORT = ['v', 'vx', 'vy', 'v_error']
BAND_NAMES = {
    'v': 'Ice Speed (m/yr)',
    'vx': 'X Velocity (m/yr)',
    'vy': 'Y Velocity (m/yr)',
    'v_error': 'Velocity Error (m/yr)',
}

ITS_LIVE_STAC = 'https://stac.itslive.cloud'

# ============================================================
# ② BBox 辅助函数
# ============================================================
def proj_to_latlon_bbox(x_min, y_min, x_max, y_max, epsg=3031):
    """将投影坐标 bbox 转换为 WGS-84 [lon_min, lat_min, lon_max, lat_max]。"""
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
        xs = [x_min, x_min, x_max, x_max]
        ys = [y_min, y_max, y_min, y_max]
        lons, lats = tr.transform(xs, ys)
        bbox = [min(lons), min(lats), max(lons), max(lats)]
        print(f'  [BBox] EPSG:{epsg} → WGS-84: {[round(v, 4) for v in bbox]}')
        return bbox
    except ImportError:
        raise RuntimeError('pyproj 未安装。请运行: pip install pyproj')

def resolve_bbox():
    """返回 (wgs84_bbox, proj_bbox_or_None, epsg_or_None)"""
    if USE_PROJ_BBOX:
        x_min, y_min, x_max, y_max = BBOX_PROJ
        wgs84 = proj_to_latlon_bbox(x_min, y_min, x_max, y_max, PROJ_EPSG)
        return wgs84, list(BBOX_PROJ), PROJ_EPSG
    else:
        return list(BBOX_LATLON), None, None

# ============================================================
# ③ 增强版 bbox 地图可视化 ⭐ 核心优化
# ============================================================
def plot_bbox_map(cfg: dict) -> Path:
    """
    在地图上绘制 bbox 位置，自动根据纬度范围选择投影：
      - 南极（纬度均值 < -45°）→ 南极极地立体投影
      - 北极（纬度均值 > +45°）→ 北极极地立体投影
      - 其他                  → 等经纬度投影（PlateCarree）
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as mticker
    except ImportError:
        print("  [警告] 未安装 matplotlib，无法绘图。运行: pip install matplotlib")
        return None

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False
        print("  [提示] 未安装 cartopy，将使用简化版地图（无海岸线）")
        print("         安装命令: pip install cartopy")

    bbox_ll     = cfg["bbox"]          # 经纬度 [W, S, E, N]
    bbox_native = cfg["bbox_native"]   # 原始坐标
    native_epsg = cfg["bbox_epsg"]     # 原始 EPSG

    lon_min, lat_min, lon_max, lat_max = bbox_ll
    cx = (lon_min + lon_max) / 2
    cy = (lat_min + lat_max) / 2

    # ── 选择地图投影 ─────────────────────────────────────────────────────
    if HAS_CARTOPY:
        if cy < -45:
            map_crs    = ccrs.SouthPolarStereo()
            proj_name  = "南极极地立体投影"
            extent_buf = 4
        elif cy > 45:
            map_crs    = ccrs.NorthPolarStereo()
            proj_name  = "北极极地立体投影"
            extent_buf = 4
        else:
            map_crs    = ccrs.PlateCarree()
            proj_name  = "等经纬度投影"
            extent_buf = 6
        
        fig = plt.figure(figsize=(10, 9), dpi=150)
        ax = fig.add_subplot(1, 1, 1, projection=map_crs)
        data_crs = ccrs.PlateCarree()

        # ── 底图要素 ──────────────────────────────────────────────────────
        ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                       facecolor="#10a1ef", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"),
                       facecolor="#b5b0a4", zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       linewidth=0.6, edgecolor="#555555", zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       linewidth=0.3, edgecolor="#999999", zorder=2)
        
        # 冰盖/冰川（仅极地有效）
        if cy < -45:
            try:
                ax.add_feature(cfeature.NaturalEarthFeature(
                    "physical", "antarctic_ice_shelves_polys", "50m"),
                    facecolor="#eaf4ff", edgecolor="#aaccee",
                    linewidth=0.4, zorder=1)
            except Exception:
                pass
        try:
            ax.add_feature(cfeature.NaturalEarthFeature(
                "physical", "glaciated_areas", "50m"),
                facecolor="#ddeeff", edgecolor="none", zorder=1)
        except Exception:
            pass

        # ── 设置地图范围 ──────────────────────────────────────────────────
        buf = extent_buf
        ax.set_extent([lon_min - buf, lon_max + buf,
                       max(lat_min - buf, -90),
                       min(lat_max + buf,  90)],
                      crs=data_crs)

        # ── 格网线 ───────────────────────────────────────────────────────
        try:
            gl = ax.gridlines(crs=data_crs, draw_labels=True,
                              linewidth=0.4, color="gray",
                              alpha=0.6, linestyle="--")
            gl.top_labels    = False
            gl.right_labels  = False
            gl.xlabel_style  = {"size": 7}
            gl.ylabel_style  = {"size": 7}
            if cy < -45 or cy > 45:
                gl.xlocator = mticker.MultipleLocator(30)
                gl.ylocator = mticker.MultipleLocator(5)
            else:
                gl.xlocator = mticker.MultipleLocator(2)
                gl.ylocator = mticker.MultipleLocator(2)
        except Exception:
            pass

        # ── 绘制 bbox ─────────────────────────────────────────────────────
        # 若投影坐标与经纬度框不同，先画经纬度外包矩形（橙色虚线）
        if native_epsg != 4326:
            ll_w, ll_s, ll_e, ll_n = bbox_ll
            ll_rect = mpatches.Rectangle(
                (ll_w, ll_s), ll_e - ll_w, ll_n - ll_s,
                linewidth=1.4, edgecolor="#ff8800",
                facecolor="none", linestyle="--",
                transform=data_crs, zorder=5,
                label=f"STAC 搜索框（经纬度）"
            )
            ax.add_patch(ll_rect)

            # 原始输入框（红色实线，沿边缘密集采样转换）
            try:
                from pyproj import Transformer
                t = Transformer.from_crs(
                    f"EPSG:{native_epsg}", "EPSG:4326", always_xy=True)
                xn, yn, xx, yx = bbox_native
                n = 50
                lin = np.linspace
                bx  = (list(lin(xn, xx, n)) + [xx]*n +
                       list(lin(xx, xn, n)) + [xn]*n)
                by  = ([yn]*n + list(lin(yn, yx, n)) +
                       [yx]*n + list(lin(yx, yn, n)))
                blons, blats = t.transform(bx, by)
                valid = np.isfinite(blons) & np.isfinite(blats)
                ax.plot(blons[valid], blats[valid],
                        color="#cc0000", linewidth=2.0,
                        transform=data_crs, zorder=6,
                        label=f"输入框（EPSG:{native_epsg}）")
                ax.fill(blons[valid], blats[valid],
                        color="#cc0000", alpha=0.12,
                        transform=data_crs, zorder=5)
            except Exception as e:
                print(f"  [警告] 投影坐标转换失败: {e}")
        else:
            # 经纬度直接画矩形
            rect = mpatches.Rectangle(
                (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                linewidth=2.0, edgecolor="#cc0000",
                facecolor="#cc0000", alpha=0.15, linestyle="-",
                transform=data_crs, zorder=6,
                label="输入 bbox"
            )
            ax.add_patch(rect)

        # 中心点
        ax.plot(cx, cy, marker="*", markersize=8,
                color="#cc0000", transform=data_crs, zorder=7)

    else:
        # ── 无 cartopy 简化版 ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
        buf = extent_buf if 'extent_buf' in dir() else 15
        ax.set_xlim(lon_min - buf, lon_max + buf)
        ax.set_ylim(lat_min - buf, lat_max + buf)
        ax.set_facecolor("#d0e8f5")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(True, linewidth=0.4, color="gray", alpha=0.5)
        rect = mpatches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            linewidth=2.0, edgecolor="#cc0000",
            facecolor="#cc0000", alpha=0.15
        )
        ax.add_patch(rect)
        ax.plot(cx, cy, marker="*", markersize=8, color="#cc0000")
        data_crs = None
        proj_name = "简化平面图（无 cartopy）"

    # ── 坐标信息文字框 ────────────────────────────────────────────────────
    if native_epsg != 4326:
        xn, yn, xx, yx = bbox_native
        info_lines = [
            f"输入坐标 (EPSG:{native_epsg})",
            f"  X: {xn:,.0f} ~ {xx:,.0f} m",
            f"  Y: {yn:,.0f} ~ {yx:,.0f} m",
            f"  宽: {abs(xx-xn)/1000:.1f} km  高: {abs(yx-yn)/1000:.1f} km",
            "",
            "经纬度外包矩形",
            f"  Lon: {lon_min:.4f}° ~ {lon_max:.4f}°",
            f"  Lat: {lat_min:.4f}° ~ {lat_max:.4f}°",
        ]
    else:
        info_lines = [
            "输入坐标 (EPSG:4326 经纬度)",
            f"  Lon: {lon_min:.4f}° ~ {lon_max:.4f}°",
            f"  Lat: {lat_min:.4f}° ~ {lat_max:.4f}°",
            f"  宽: {(lon_max-lon_min)*111*np.cos(np.radians(cy)):.1f} km",
            f"  高: {(lat_max-lat_min)*111:.1f} km",
        ]

    info_text = "\n".join(info_lines)
    
    # 字体处理：优先 Times New Roman
    try:
        import matplotlib.font_manager as _fm
        available_fonts = {f.name for f in _fm.fontManager.ttflist}
        text_font = "Times New Roman" if "Times New Roman" in available_fonts else None
    except Exception:
        text_font = None
    
    text_kwargs = dict(
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  alpha=0.85, edgecolor="#cccccc"),
        zorder=10
    )
    if text_font:
        text_kwargs["fontfamily"] = text_font
    
    ax.text(0.02, 0.02, info_text, **text_kwargs)

    # ── 图例 & 标题 ───────────────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right",
                  fontsize=7.5, framealpha=0.85)

    t_start, t_end = cfg["datetime"].split("/")
    sensor_str = ", ".join(SENSOR_CONFIG.get(s, {'label': s})['label'] 
                          for s in cfg["sensors"])
    
    title_kwargs = dict(fontsize=9, pad=10)
    if text_font:
        title_kwargs["fontfamily"] = text_font
    
    ax.set_title(
        f"ITS_LIVE Data Coverage  |  {proj_name}\n"
        f"Sensors: {sensor_str}  |  Time: {t_start} ~ {t_end}",
        **title_kwargs
    )

    # ── 保存 ─────────────────────────────────────────────────────────────
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bbox_map.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  [地图] 已保存: {out_path.resolve()}")
    return out_path

# ============================================================
# ④ CRS / 仿射变换提取
# ============================================================
def get_crs_and_transform(ds: xr.Dataset, epsg_hint=None):
    """从 ITS_LIVE xarray Dataset 提取 CRS 和仿射变换。"""
    try:
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
    except ImportError:
        print('    [警告] rasterio 未安装，跳过 GeoTIFF 转换')
        return None, None, False
    
    crs = None
    if 'mapping' in ds:
        m = ds['mapping']
        for attr in ('crs_wkt', 'spatial_ref', 'proj4text', 'proj4_params'):
            val = m.attrs.get(attr, '')
            if val:
                try:
                    crs = CRS.from_string(val)
                    print(f'    [CRS] mapping.{attr}: EPSG:{crs.to_epsg() or "custom"}')
                    break
                except Exception:
                    pass
    
    if crs is None and epsg_hint:
        crs = CRS.from_epsg(epsg_hint)
        print(f'    [CRS] from bbox hint: EPSG:{epsg_hint}')
    
    if crs is None:
        crs = CRS.from_epsg(3031)
        print('    [CRS] default EPSG:3031')
    
    x  = ds.coords['x'].values.astype(float)
    y  = ds.coords['y'].values.astype(float)
    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    west  = x[0] - dx / 2
    north = max(y[0], y[-1]) + dy / 2
    transform = from_origin(west, north, dx, dy)
    flip_rows = bool(y[0] < y[-1])
    
    return crs, transform, flip_rows

# ============================================================
# ⑤ NC → GeoTIFF 转换
# ============================================================
def array_to_geotiff(data: np.ndarray, crs, transform, out_path: Path,
                     flip_rows=False, nodata=None, band_desc=''):
    """将 2-D 数组写入压缩单波段 GeoTIFF。"""
    import rasterio
    if flip_rows:
        data = data[::-1].copy()
    
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress='deflate',
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
        if band_desc:
            dst.update_tags(1, description=band_desc)

def netcdf_to_geotiff(nc_path: Path, out_dir: Path, epsg_hint=None) -> list:
    """将 ITS_LIVE granule NetCDF 转换为逐变量 GeoTIFF。"""
    try:
        import rasterio
    except ImportError:
        print('    [警告] rasterio 未安装，跳过 GeoTIFF 转换')
        return []
    
    ds = xr.open_dataset(nc_path)
    saved = []
    
    if 'x' not in ds.coords or 'y' not in ds.coords:
        print(f'    [警告] 找不到 x/y 坐标，跳过：{nc_path.name}')
        ds.close()
        return []
    
    crs, transform, flip_rows = get_crs_and_transform(ds, epsg_hint)
    if crs is None:
        ds.close()
        return []
    
    for var in VARIABLES_TO_EXPORT:
        if var not in ds:
            continue
        data = ds[var].values
        if data.ndim == 3:
            data = data[0]
        if data.ndim != 2:
            continue
        
        nodata   = ds[var].attrs.get('_FillValue') or ds[var].attrs.get('missing_value')
        tif_path = out_dir / f'{nc_path.stem}_{var}.tif'
        
        array_to_geotiff(
            data, crs, transform, tif_path,
            flip_rows=flip_rows, nodata=nodata,
            band_desc=BAND_NAMES.get(var, var),
        )
        print(f'    [GeoTIFF] {tif_path.name}')
        saved.append(tif_path)
    
    ds.close()
    return saved

# ============================================================
# ⑥ 下载辅助函数
# ============================================================
def download_https(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """带进度条的 HTTPS 文件下载。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f'  [跳过] 已存在：{dest.name}')
        return dest
    
    print(f'  [下载] {dest.name}')
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total      = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f'\r  进度 [{dest.name[:30]}]: '
                        f'{downloaded / total * 100:5.1f}%  '
                        f'({downloaded >> 20} MB / {total >> 20} MB)',
                        end='',
                    )
        print()
    return dest

def download_item(item, output_dir: Path, epsg_hint=None) -> tuple:
    """下载单个 item 并转换为 GeoTIFF。"""
    url  = item.assets['data'].href
    dest = output_dir / f'{item.id}.nc'
    try:
        download_https(url, dest)
        tifs = netcdf_to_geotiff(dest, output_dir, epsg_hint)
        print(f'  [转换] 生成 {len(tifs)} 个 GeoTIFF')
        return item.id, True, ''
    except Exception as e:
        return item.id, False, str(e)

# ============================================================
# ⑦ 主程序
# ============================================================
def main():
    OUTPUT_DIR = Path(FILTER_CONFIG['output_dir'])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print('=' * 70)
    print('  ITS_LIVE Granule Search & Download  (优化版 v2.0)')
    print('=' * 70)
    
    # ── 解析 bbox ────────────────────────────────────────────────────────────
    wgs84_bbox, proj_bbox, epsg_hint = resolve_bbox()
    bbox_type = f'EPSG:{epsg_hint} → WGS-84' if USE_PROJ_BBOX else 'WGS-84'
    print(f'  BBox 类型 : {bbox_type}')
    print(f'  BBox      : {[round(v, 4) for v in wgs84_bbox]}')
    if proj_bbox:
        print(f'  投影 bbox : {proj_bbox}  (EPSG:{epsg_hint})')
    print()
    
    # ── 绘制增强版地图 ───────────────────────────────────────────────────────
    map_cfg = {
        "bbox":         wgs84_bbox,
        "bbox_native":  proj_bbox if proj_bbox else wgs84_bbox,
        "bbox_epsg":    epsg_hint if proj_bbox else 4326,
        "datetime":     FILTER_CONFIG['datetime'],
        "sensors":      FILTER_CONFIG['platforms'],
        "output_dir":   str(OUTPUT_DIR),
    }
    plot_bbox_map(map_cfg)
    
    # ── 连接 STAC ────────────────────────────────────────────────────────────
    catalog = Client.open(ITS_LIVE_STAC)
    
    # ── STAC 搜索 ────────────────────────────────────────────────────────────
    search = catalog.search(
        collections=FILTER_CONFIG['collections'],
        bbox=wgs84_bbox,
        datetime=FILTER_CONFIG['datetime'],
        query={
            'percent_valid_pixels': {'gte': FILTER_CONFIG['min_valid_pixels']},
            'date_dt':              {'lte': FILTER_CONFIG['max_date_dt']},
        },
        max_items=FILTER_CONFIG['max_items'],
    )
    
    all_items = list(search.items())
    print(f'\n🔍 服务端返回：{len(all_items)} 条数据')
    
    # ── 客户端二次过滤 ───────────────────────────────────────────────────────
    filtered_items = [
        item for item in all_items
        if (
            item.properties.get('platform', '') in FILTER_CONFIG['platforms']
            and item.properties.get('percent_valid_pixels', 0) >= FILTER_CONFIG['min_valid_pixels']
            and item.properties.get('date_dt', float('inf')) <= FILTER_CONFIG['max_date_dt']
        )
    ]
    print(f'\n✅ 客户端过滤后：{len(filtered_items)} 条数据')
    
    if not filtered_items:
        print('\n[提示] 无符合条件的数据，程序退出。')
        return
    
    # ── 导出结果文件 ───────────────────────────────────────────────────────
    # TXT
    with open(OUTPUT_DIR / 'filtered_itslive_results.txt', 'w', encoding='utf-8') as f:
        f.write('=' * 100 + '\n')
        f.write('ITS_LIVE 冰川流速数据过滤结果\n')
        f.write('=' * 100 + '\n')
        f.write(f"过滤条件：valid_pixels >= {FILTER_CONFIG['min_valid_pixels']}% | "
                f"date_dt <= {FILTER_CONFIG['max_date_dt']} 天 | "
                f"platforms: {', '.join(FILTER_CONFIG['platforms'])}\n")
        f.write(f"BBox ({bbox_type}): {wgs84_bbox}\n")
        if proj_bbox:
            f.write(f'投影 BBox (EPSG:{epsg_hint}): {proj_bbox}\n')
        f.write(f'数据总量：{len(filtered_items)} 条\n')
        f.write(f'生成时间：{datetime.datetime.now().isoformat()}\n')
        f.write('=' * 100 + '\n')
        
        header = (f"{'ID':<80} | {'Platform':<8} | {'Valid%':<6} | "
                  f"{'Date_dt':<8} | {'mid_datetime':<25}")
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')
        
        for item in filtered_items:
            p = item.properties
            f.write(f"{item.id:<80} | {p['platform']:<8} | "
                    f"{p['percent_valid_pixels']:<6} | "
                    f"{p['date_dt']:<8} | {p['mid_datetime']:<25}\n")
    
    print(f'\n✅ TXT 已导出：{OUTPUT_DIR / "filtered_itslive_results.txt"}')
    
    # JSON
    export_data = [
        {
            'id':                   item.id,
            'platform':             item.properties.get('platform'),
            'date_dt':              item.properties.get('date_dt'),
            'percent_valid_pixels': item.properties.get('percent_valid_pixels'),
            'mid_datetime':         item.properties.get('mid_datetime'),
            'proj_code':            item.properties.get('proj:code'),
            'bbox':                 item.bbox,
            'data_url':             item.assets['data'].href,
        }
        for item in filtered_items
    ]
    with open(OUTPUT_DIR / 'filtered_itslive_results.json', 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f'✅ JSON 已导出：{OUTPUT_DIR / "filtered_itslive_results.json"}')
    
    # ── 并发下载 + NC→GeoTIFF ───────────────────────────────────────────────
    print(f'\n🚀 开始并发下载 + GeoTIFF 转换（线程数：{FILTER_CONFIG["max_workers"]}）...')
    success_count = 0
    fail_count    = 0
    n_total       = len(filtered_items)
    
    with ThreadPoolExecutor(max_workers=FILTER_CONFIG['max_workers']) as executor:
        futures = {
            executor.submit(download_item, item, OUTPUT_DIR, epsg_hint): item
            for item in filtered_items
        }
        for i, future in enumerate(as_completed(futures), start=1):
            item_id, ok, err = future.result()
            if ok:
                success_count += 1
                print(f'  [完成 {i}/{n_total}] {item_id[:55]}')
            else:
                fail_count += 1
                print(f'  [失败 {i}/{n_total}] {item_id[:55]}  错误：{err}')
    
    print(f'\n✅ 下载完成：{success_count} 成功，{fail_count} 失败')
    
    # ── 汇总输出 ───────────────────────────────────────────────────────────
    n_nc  = len(list(OUTPUT_DIR.glob('*.nc')))
    n_tif = len(list(OUTPUT_DIR.glob('*.tif')))
    print('\n' + '=' * 70)
    print(f'  输出目录 : {OUTPUT_DIR.resolve()}')
    print(f'  NetCDF   : {n_nc} 个')
    print(f'  GeoTIFF  : {n_tif} 个')
    print(f'  AOI 地图 : {OUTPUT_DIR / "bbox_map.png"}')
    print('=' * 70)

if __name__ == '__main__':
    main()
