#!/usr/bin/env python3
"""
ITS_LIVE GeoTIFF 四合一可视化
将 v, vx, vy, v_error 四个变量绘制在同一张 2×2 子图中
"""
import numpy as np
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# ① 配置区域 ⚙️
# ============================================================
GEOTIFF_DIR = Path('/data1/geotiff_output')     # GeoTIFF 文件目录
PLOT_DIR    = Path('/data1/geotiff_output/plots')  # 绘图输出目录

# 🔹 变量配置（按 2×2 布局顺序）
VARIABLES_4IN1 = ['v', 'vx', 'vy', 'v_error']  # 顺序：左上→右上→左下→右下

# 🔹 绘图样式配置
PLOT_CONFIG = {
    'v': {
        'cmap': 'viridis',
        'vmin': 0, 'vmax': 2000,
        'title': 'Ice Speed',
        'units': 'm/yr',
        'colorbar_label': 'Speed (m/yr)'
    },
    'vx': {
        'cmap': 'RdBu_r',
        'vmin': -1000, 'vmax': 1000,
        'title': 'X Velocity',
        'units': 'm/yr',
        'colorbar_label': 'Vx (m/yr)'
    },
    'vy': {
        'cmap': 'RdBu_r', 
        'vmin': -1000, 'vmax': 1000,
        'title': 'Y Velocity',
        'units': 'm/yr',
        'colorbar_label': 'Vy (m/yr)'
    },
    'v_error': {
        'cmap': 'magma',
        'vmin': 0, 'vmax': 500,
        'title': 'Velocity Error',
        'units': 'm/yr',
        'colorbar_label': 'Error (m/yr)'
    },
}

# 🔹 绘图参数
PLOT_DPI = 150
PLOT_FIGSIZE = (14, 12)      # 更大的画布容纳 4 个子图
MAX_WORKERS = 4
USE_CARTOPY = True
ADD_BASEMAP = True

# ============================================================
# ② 🎨 核心绘图函数：四合一布局
# ============================================================
def plot_4in1_geotiff(tif_dict: dict, out_png: Path):
    """
    将四个变量的 GeoTIFF 绘制在同一张 2×2 子图中
    
    参数:
        tif_dict: {var_name: tif_path} 字典
        out_png:  输出 PNG 路径
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
    except ImportError as e:
        print(f'    [错误] 依赖缺失: {e}')
        return False
    
    if USE_CARTOPY:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            HAS_CARTOPY = True
        except ImportError:
            HAS_CARTOPY = False
            print(f'    [提示] cartopy 未安装，使用简化布局')
    else:
        HAS_CARTOPY = False
    
    # ── 读取所有变量数据 ─────────────────────────────────────
    data_dict = {}
    crs = None
    transform = None
    bounds = None
    
    for var, tif_path in tif_dict.items():
        try:
            with rasterio.open(tif_path) as src:
                data = src.read(1)
                nodata = src.nodatavals[0] if src.nodatavals else None
                
                if nodata is not None:
                    data = np.ma.masked_equal(data, nodata)
                else:
                    data = np.ma.masked_invalid(data)
                
                data_dict[var] = data
                if crs is None:
                    crs = src.crs
                    transform = src.transform
                    bounds = src.bounds
        except Exception as e:
            print(f'    [警告] 读取 {var} 失败: {e}')
            return False
    
    if not data_dict:
        return False
    
    # ── 开始绘图 ─────────────────────────────────────────────
    fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    
    # 检测是否为南极极地投影
    is_polar = (crs and crs.to_epsg() == 3031) if crs else False
    
    # 获取经纬度范围（用于标题）
    lon_min, lat_min, lon_max, lat_max = None, None, None, None
    if crs and bounds:
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
    
    # ── 2×2 子图布局 ─────────────────────────────────────────
    for idx, var in enumerate(VARIABLES_4IN1):
        if var not in data_dict:
            continue
        
        ax = fig.add_subplot(2, 2, idx + 1, 
                           projection=ccrs.SouthPolarStereo() if (HAS_CARTOPY and is_polar) else None)
        
        cfg = PLOT_CONFIG[var]
        data = data_dict[var]
        
        if HAS_CARTOPY and is_polar:
            # 极地投影设置
            if lon_min and lat_min:
                buf = 2
                ax.set_extent([lon_min-buf, lon_max+buf,
                              max(lat_min-buf, -90), min(lat_max+buf, -60)],
                             crs=ccrs.PlateCarree())
            
            # 底图
            if ADD_BASEMAP:
                ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                              facecolor='#10a1ef', zorder=0)
                ax.add_feature(cfeature.LAND.with_scale('50m'),
                              facecolor='#b5b0a4', zorder=1)
                ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                              linewidth=0.4, edgecolor='#555555', zorder=2)
            
            # 重投影数据
            dst_crs = ccrs.SouthPolarStereo()
            dst_transform, dst_width, dst_height = calculate_default_transform(
                crs, dst_crs, data.shape[1], data.shape[0], *bounds)
            
            data_reproj = np.empty((dst_height, dst_width), dtype=data.dtype)
            reproject(
                source=data, destination=data_reproj,
                src_transform=transform, src_crs=crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear, dst_nodata=np.nan
            )
            data_plot = np.ma.masked_invalid(data_reproj)
            extent = (dst_transform[2], dst_transform[2] + dst_width*dst_transform[0],
                      dst_transform[5] + dst_height*dst_transform[4], dst_transform[5])
            
            im = ax.imshow(data_plot, cmap=cfg['cmap'],
                          vmin=cfg['vmin'], vmax=cfg['vmax'],
                          extent=extent, origin='upper',
                          transform=dst_crs, zorder=3)
            
            # 格网
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                             linewidth=0.2, color='gray', alpha=0.4, linestyle='--')
            gl.xlocator = mticker.MultipleLocator(30)
            gl.ylocator = mticker.MultipleLocator(5)
            
        else:
            # 简化版
            im = ax.imshow(data, cmap=cfg['cmap'],
                          vmin=cfg['vmin'], vmax=cfg['vmax'],
                          origin='upper')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.2)
        
        # 子图标题
        ax.set_title(f"{cfg['title']} ({cfg['units']})", fontsize=10, pad=8)
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cfg['colorbar_label'], fontsize=8)
    
    # ── 主标题 ───────────────────────────────────────────────
    file_info = list(tif_dict.values())[0].stem.replace('_v', '')
    if lon_min and lat_min:
        loc_info = f"Lon: {lon_min:.2f}°~{lon_max:.2f}°, Lat: {lat_min:.2f}°~{lat_max:.2f}°"
        main_title = f"ITS_LIVE Velocity Components\n{file_info}\n{loc_info}"
    else:
        main_title = f"ITS_LIVE Velocity Components\n{file_info}"
    
    fig.suptitle(main_title, fontsize=13, fontweight='bold', y=1.02)
    
    # ── 保存 ─────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True

# ============================================================
# ③ 批量处理函数
# ============================================================
def find_4in1_groups(geotiff_dir: Path) -> dict:
    """
    查找同一场景的四个变量文件并分组
    
    返回:
        dict: {scene_prefix: {var: tif_path}}
    """
    tif_files = list(geotiff_dir.glob('*.tif'))
    if not tif_files:
        print(f'[警告] 未找到 GeoTIFF 文件：{geotiff_dir}')
        return {}
    
    # 按场景前缀分组
    groups = {}
    for tif in tif_files:
        stem = tif.stem
        # 提取变量名和场景前缀
        for var in VARIABLES_4IN1:
            if stem.endswith(f'_{var}'):
                prefix = stem[:-len(f'_{var}')]
                if prefix not in groups:
                    groups[prefix] = {}
                groups[prefix][var] = tif
                break
    
    # 只保留四个变量都存在的组
    complete_groups = {
        prefix: vars_dict 
        for prefix, vars_dict in groups.items() 
        if all(v in vars_dict for v in VARIABLES_4IN1)
    }
    
    print(f'  找到 {len(complete_groups)} 个完整的四变量场景')
    return complete_groups

def batch_plot_4in1(geotiff_dir: Path, plot_dir: Path):
    """批量绘制四合一图"""
    
    groups = find_4in1_groups(geotiff_dir)
    if not groups:
        print('[错误] 没有可绘制的完整四变量组')
        return 0, 0
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    success = fail = 0
    total = len(groups)
    
    print(f'\n🚀 开始批量绘制四合一图 (共 {total} 个场景)...')
    print('-' * 70)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for prefix, tif_dict in groups.items():
            out_png = plot_dir / f'{prefix}_4in1.png'
            future = executor.submit(plot_4in1_geotiff, tif_dict, out_png)
            futures[future] = prefix
        
        for future in as_completed(futures):
            prefix = futures[future]
            try:
                ok = future.result()
                if ok:
                    success += 1
                    print(f'  [{success+fail}/{total}] ✅ {prefix[:60]}_4in1.png')
                else:
                    fail += 1
                    print(f'  [{success+fail}/{total}] ❌ {prefix[:60]}_4in1.png')
            except Exception as e:
                fail += 1
                print(f'  [{success+fail}/{total}] ❌ {prefix[:60]} - {e}')
    
    return success, fail

# ============================================================
# ④ 命令行参数 & 主程序
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='ITS_LIVE 四合一可视化工具')
    parser.add_argument('-i', '--input', type=Path, default=GEOTIFF_DIR)
    parser.add_argument('-o', '--output', type=Path, default=PLOT_DIR)
    parser.add_argument('-d', '--dpi', type=int, default=PLOT_DPI)
    parser.add_argument('-w', '--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--no-cartopy', action='store_true')
    parser.add_argument('--no-basemap', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    
    global GEOTIFF_DIR, PLOT_DIR, PLOT_DPI, MAX_WORKERS, USE_CARTOPY, ADD_BASEMAP
    GEOTIFF_DIR = args.input
    PLOT_DIR = args.output
    PLOT_DPI = args.dpi
    MAX_WORKERS = args.workers
    USE_CARTOPY = not args.no_cartopy
    ADD_BASEMAP = not args.no_basemap
    
    print('=' * 70)
    print('  ITS_LIVE 四合一可视化工具 (v, vx, vy, v_error)')
    print('=' * 70)
    print(f'  输入目录 : {GEOTIFF_DIR.resolve()}')
    print(f'  输出目录 : {PLOT_DIR.resolve()}')
    print(f'  图片 DPI : {PLOT_DPI}')
    print(f'  并发线程 : {MAX_WORKERS}')
    print(f'  极地投影 : {"✅ 启用" if USE_CARTOPY else "❌ 禁用"}')
    print(f'  底图要素 : {"✅ 启用" if ADD_BASEMAP else "❌ 禁用"}')
    print()
    
    if not GEOTIFF_DIR.exists():
        print(f'[错误] 输入目录不存在：{GEOTIFF_DIR}')
        return
    
    success, fail = batch_plot_4in1(GEOTIFF_DIR, PLOT_DIR)
    
    print()
    print('=' * 70)
    print(f'  📊 绘图完成！')
    print(f'  成功：{success} 个')
    print(f'  失败：{fail} 个')
    print(f'  输出目录：{PLOT_DIR.resolve()}')
    print('=' * 70)

if __name__ == '__main__':
    main()