#!/usr/bin/env python3
"""
ITS_LIVE NC → GeoTIFF 批量转换 + 可选绘图
将 /data1/nc_files/ 下的 NC 文件转换为 GeoTIFF，并可选择性绘制结果图
"""
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import xarray as xr

# ============================================================
# ① 配置区域 ⚙️
# ============================================================
INPUT_DIR  = Path('/data1/nc_files')           # NC 文件输入目录
OUTPUT_DIR = Path('/data1/geotiff_output')     # GeoTIFF 输出目录
PLOT_DIR   = OUTPUT_DIR / 'plots'              # 绘图输出子目录

# 🔹 核心参数
VARIABLES_TO_EXPORT = ['v', 'vx', 'vy', 'v_error']
PLOT_RESULTS = True              # 🎨 是否绘制结果图（设为 False 可跳过绘图）
PLOT_DPI = 150                   # 输出图片分辨率
MAX_WORKERS = 8                  # 并发线程数

# 🔹 绘图样式配置
PLOT_CONFIG = {
    'v': {
        'cmap': 'viridis',
        'vmin': 0, 'vmax': 2000,
        'title': 'Ice Speed (m/yr)',
        'units': 'm/yr'
    },
    'vx': {
        'cmap': 'RdBu_r',
        'vmin': -1000, 'vmax': 1000,
        'title': 'X Velocity (m/yr)',
        'units': 'm/yr'
    },
    'vy': {
        'cmap': 'RdBu_r', 
        'vmin': -1000, 'vmax': 1000,
        'title': 'Y Velocity (m/yr)',
        'units': 'm/yr'
    },
    'v_error': {
        'cmap': 'magma',
        'vmin': 0, 'vmax': 500,
        'title': 'Velocity Error (m/yr)',
        'units': 'm/yr'
    },
}

BAND_NAMES = {
    'v': 'Ice Speed (m/yr)',
    'vx': 'X Velocity (m/yr)',
    'vy': 'Y Velocity (m/yr)',
    'v_error': 'Velocity Error (m/yr)',
}

# ============================================================
# ② CRS / 仿射变换提取（复用原代码逻辑）
# ============================================================
def get_crs_and_transform(ds: xr.Dataset, epsg_hint=3031):
    """从 ITS_LIVE xarray Dataset 提取 CRS 和仿射变换。"""
    try:
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
    except ImportError:
        print('    [错误] rasterio 未安装')
        return None, None, False
    
    crs = None
    if 'mapping' in ds:
        m = ds['mapping']
        for attr in ('crs_wkt', 'spatial_ref', 'proj4text'):
            val = m.attrs.get(attr, '')
            if val:
                try:
                    crs = CRS.from_string(val)
                    break
                except Exception:
                    pass
    
    if crs is None and epsg_hint:
        crs = CRS.from_epsg(epsg_hint)
    if crs is None:
        crs = CRS.from_epsg(3031)
    
    # 仿射变换
    x = ds.coords['x'].values.astype(float)
    y = ds.coords['y'].values.astype(float)
    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    west  = x[0] - dx / 2
    north = max(y[0], y[-1]) + dy / 2
    transform = from_origin(west, north, dx, dy)
    flip_rows = bool(y[0] < y[-1])
    
    return crs, transform, flip_rows

# ============================================================
# ③ 数组 → GeoTIFF
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

# ============================================================
# ④ 🎨 绘图功能：GeoTIFF → PNG 可视化
# ============================================================
def plot_geotiff(tif_path: Path, out_png: Path, var_name: str, cfg: dict):
    """绘制单个 GeoTIFF 文件并保存为 PNG。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
    except ImportError:
        print(f'    [警告] 绘图依赖缺失，跳过：{tif_path.name}')
        return False
    
    try:
        # 尝试使用 cartopy 进行极地投影
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
    except ImportError:
        USE_CARTOPY = False
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        crs = src.crs
        transform = src.transform
        nodata = src.nodatavals[0] if src.nodatavals else None
        
        # 掩膜 nodata
        if nodata is not None:
            data = np.ma.masked_equal(data, nodata)
        else:
            data = np.ma.masked_invalid(data)
        
        # 获取地理范围（用于标题）
        bounds = src.bounds
        lon_min, lat_min, lon_max, lat_max = None, None, None, None
        
        if USE_CARTOPY and crs:
            try:
                from pyproj import Transformer
                tr = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                lons, lats = tr.transform(
                    [bounds.left, bounds.right, bounds.left, bounds.right],
                    [bounds.top, bounds.top, bounds.bottom, bounds.bottom]
                )
                lon_min, lon_max = min(lons), max(lons)
                lat_min, lat_max = min(lats), max(lats)
            except:
                pass
    
    # ── 开始绘图 ─────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
    
    if USE_CARTOPY and crs and crs.to_epsg() == 3031:
        # 南极极地立体投影
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
        ax.set_extent([lon_min-1, lon_max+1, lat_min-1, lat_max+1], crs=ccrs.PlateCarree())
        
        # 底图
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#10a1ef', zorder=0)
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#b5b0a4', zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, zorder=2)
        
        # 重投影数据到显示坐标系
        dst_crs = ccrs.SouthPolarStereo()
        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, dst_crs, src.width, src.height, *bounds)
        
        data_reproj = np.empty((dst_height, dst_width), dtype=data.dtype)
        reproject(
            source=data,
            destination=data_reproj,
            src_transform=transform,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan
        )
        data_plot = np.ma.masked_invalid(data_reproj)
        extent = (dst_transform[2], dst_transform[2] + dst_width*dst_transform[0],
                  dst_transform[5] + dst_height*dst_transform[4], dst_transform[5])
        
        im = ax.imshow(data_plot, cmap=cfg['cmap'], vmin=cfg['vmin'], vmax=cfg['vmax'],
                      extent=extent, origin='upper', transform=dst_crs, zorder=3)
        
        # 格网
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, 
                         linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = plt.MaxNLocator(4)
        gl.ylocator = plt.MaxNLocator(4)
        
    else:
        # 简化版：直接显示数组
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(data, cmap=cfg['cmap'], vmin=cfg['vmin'], vmax=cfg['vmax'],
                      origin='upper' if flip_rows else 'lower')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # 颜色条 + 标题
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cfg['units'], fontsize=9)
    
    file_info = tif_path.stem.replace(f'_{var_name}', '')
    title = f"{cfg['title']}\n{file_info}"
    ax.set_title(title, fontsize=10, pad=10)
    
    # 保存
    plt.tight_layout()
    plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True

# ============================================================
# ⑤ 单个 NC 文件处理（转换 + 可选绘图）
# ============================================================
def process_nc_file(nc_path: Path, out_dir: Path, plot_dir: Path, 
                   plot_flag: bool, epsg_hint=3031) -> tuple:
    """处理单个 NC 文件：转换 GeoTIFF + 可选绘图。"""
    try:
        import rasterio
    except ImportError:
        return nc_path.name, False, 'rasterio missing'
    
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        return nc_path.name, False, f'打开失败：{e}'
    
    if 'x' not in ds.coords or 'y' not in ds.coords:
        ds.close()
        return nc_path.name, False, '缺少 x/y 坐标'
    
    crs, transform, flip_rows = get_crs_and_transform(ds, epsg_hint)
    if crs is None:
        ds.close()
        return nc_path.name, False, 'CRS 提取失败'
    
    saved_tifs = []
    saved_plots = []
    
    for var in VARIABLES_TO_EXPORT:
        if var not in ds:
            continue
        data = ds[var].values
        if data.ndim == 3:
            data = data[0]
        if data.ndim != 2:
            continue
        
        nodata = ds[var].attrs.get('_FillValue') or ds[var].attrs.get('missing_value')
        
        # → 保存 GeoTIFF
        tif_path = out_dir / f'{nc_path.stem}_{var}.tif'
        try:
            array_to_geotiff(data, crs, transform, tif_path,
                           flip_rows=flip_rows, nodata=nodata,
                           band_desc=BAND_NAMES.get(var, var))
            saved_tifs.append(tif_path)
        except Exception as e:
            print(f'    [警告] {var} GeoTIFF 保存失败：{e}')
            continue
        
        # → 可选绘图
        if plot_flag and var in PLOT_CONFIG:
            plot_dir.mkdir(parents=True, exist_ok=True)
            png_path = plot_dir / f'{nc_path.stem}_{var}.png'
            try:
                if plot_geotiff(tif_path, png_path, var, PLOT_CONFIG[var]):
                    saved_plots.append(png_path)
            except Exception as e:
                print(f'    [警告] {var} 绘图失败：{e}')
    
    ds.close()
    msg = f'GeoTIFF:{len(saved_tifs)}'
    if plot_flag:
        msg += f' | Plots:{len(saved_plots)}'
    return nc_path.name, len(saved_tifs) > 0, msg

# ============================================================
# ⑥ 主程序
# ============================================================
def main():
    print('=' * 70)
    print('  ITS_LIVE NC → GeoTIFF + 可选绘图 批量处理')
    print('=' * 70)
    
    if not INPUT_DIR.exists():
        print(f'[错误] 输入目录不存在：{INPUT_DIR}')
        return
    
    nc_files = list(INPUT_DIR.glob('*.nc'))
    if not nc_files:
        print(f'[错误] 未找到 .nc 文件：{INPUT_DIR}')
        return
    
    print(f'  输入目录 : {INPUT_DIR.resolve()}')
    print(f'  输出目录 : {OUTPUT_DIR.resolve()}')
    print(f'  绘图目录 : {PLOT_DIR.resolve() if PLOT_RESULTS else "（跳过）"}')
    print(f'  NC 文件数: {len(nc_files)}')
    print(f'  导出变量 : {VARIABLES_TO_EXPORT}')
    print(f'  并发线程 : {MAX_WORKERS}')
    print(f'  启用绘图 : {"✅ 是" if PLOT_RESULTS else "❌ 否"}')
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PLOT_RESULTS:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 并发处理
    success = fail = 0
    total_tifs = total_plots = 0
    
    print('🚀 开始批量处理...')
    print('-' * 70)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_nc_file, nc, OUTPUT_DIR, PLOT_DIR, 
                          PLOT_RESULTS): nc
            for nc in nc_files
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            name, ok, msg = future.result()
            if ok:
                success += 1
                if 'GeoTIFF:' in msg:
                    parts = msg.split(' | ')
                    for p in parts:
                        if 'GeoTIFF:' in p:
                            total_tifs += int(p.split(':')[1])
                        elif 'Plots:' in p:
                            total_plots += int(p.split(':')[1])
                print(f'  [{i}/{len(nc_files)}] ✅ {name[:45]} → {msg}')
            else:
                fail += 1
                print(f'  [{i}/{len(nc_files)}] ❌ {name[:45]} → {msg}')
    
    # 汇总
    print()
    print('=' * 70)
    print(f'  📊 处理完成！')
    print(f'  成功 NC 文件 : {success} / {len(nc_files)}')
    print(f'  失败 NC 文件 : {fail}')
    print(f'  生成 GeoTIFF : {total_tifs} 个')
    if PLOT_RESULTS:
        print(f'  生成结果图   : {total_plots} 个')
        print(f'  图片目录     : {PLOT_DIR.resolve()}')
    print(f'  输出目录     : {OUTPUT_DIR.resolve()}')
    print('=' * 70)

if __name__ == '__main__':
    main()