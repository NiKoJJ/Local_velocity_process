#!/usr/bin/env python3
"""
ITS_LIVE GeoTIFF 结果可视化
独立绘图脚本 - 用于批量绘制已转换好的 GeoTIFF 文件
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

# 🔹 变量过滤（只绘制指定的变量）
VARIABLES_TO_PLOT = ['v', 'vx', 'vy', 'v_error']  # 设为 [] 则绘制所有找到的变量

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
PLOT_DPI = 150           # 输出图片分辨率
PLOT_FIGSIZE = (10, 8)   # 图片尺寸 (宽, 高)
MAX_WORKERS = 4          # 并发线程数
USE_CARTOPY = True       # 是否使用 cartopy 进行极地投影
ADD_BASEMAP = True       # 是否添加海岸线/陆地底图

# ============================================================
# ② 🎨 核心绘图函数
# ============================================================
def plot_single_geotiff(tif_path: Path, out_png: Path, var_name: str, cfg: dict):
    """
    绘制单个 GeoTIFF 文件并保存为 PNG
    
    参数:
        tif_path: GeoTIFF 文件路径
        out_png:  输出 PNG 路径
        var_name: 变量名 (v, vx, vy, v_error)
        cfg:      绘图配置字典
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
    
    # 尝试加载 cartopy（用于极地投影）
    if USE_CARTOPY:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            HAS_CARTOPY = True
        except ImportError:
            HAS_CARTOPY = False
            print(f'    [提示] cartopy 未安装，使用简化投影')
    else:
        HAS_CARTOPY = False
    
    # ── 读取 GeoTIFF ─────────────────────────────────────────────
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            crs = src.crs
            transform = src.transform
            nodata = src.nodatavals[0] if src.nodatavals else None
            bounds = src.bounds
            
            # 掩膜 nodata
            if nodata is not None:
                data = np.ma.masked_equal(data, nodata)
            else:
                data = np.ma.masked_invalid(data)
            
            # 尝试获取经纬度范围（用于地图标题）
            lon_min, lat_min, lon_max, lat_max = None, None, None, None
            if crs:
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
    except Exception as e:
        print(f'    [错误] 读取 GeoTIFF 失败: {e}')
        return False
    
    # ── 开始绘图 ─────────────────────────────────────────────
    fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    
    # 检测是否为南极极地投影 (EPSG:3031)
    is_polar = (crs and crs.to_epsg() == 3031) if crs else False
    
    if HAS_CARTOPY and is_polar:
        # ── 南极极地立体投影 ─────────────────────────────────
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
        
        # 设置地图范围
        if lon_min and lat_min:
            buf = 2
            ax.set_extent([lon_min-buf, lon_max+buf, 
                          max(lat_min-buf, -90), min(lat_max+buf, -60)], 
                         crs=ccrs.PlateCarree())
        
        # 添加底图
        if ADD_BASEMAP:
            ax.add_feature(cfeature.OCEAN.with_scale('50m'), 
                          facecolor='#10a1ef', zorder=0)
            ax.add_feature(cfeature.LAND.with_scale('50m'), 
                          facecolor='#b5b0a4', zorder=1)
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), 
                          linewidth=0.5, edgecolor='#555555', zorder=2)
            
            # 南极冰架
            try:
                ax.add_feature(cfeature.NaturalEarthFeature(
                    'physical', 'antarctic_ice_shelves_polys', '50m'),
                    facecolor='#eaf4ff', edgecolor='#aaccee',
                    linewidth=0.4, zorder=1)
            except:
                pass
        
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
        
        im = ax.imshow(data_plot, cmap=cfg['cmap'], 
                      vmin=cfg['vmin'], vmax=cfg['vmax'],
                      extent=extent, origin='upper', 
                      transform=dst_crs, zorder=3)
        
        # 添加格网
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, 
                         linewidth=0.3, color='gray', alpha=0.5, 
                         linestyle='--')
        gl.xlocator = mticker.MultipleLocator(30)
        gl.ylocator = mticker.MultipleLocator(5)
        
    else:
        # ── 简化版：直接显示数组 ──────────────────────────────
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(data, cmap=cfg['cmap'], 
                      vmin=cfg['vmin'], vmax=cfg['vmax'],
                      origin='upper')
        ax.set_xlabel('Column (pixels)')
        ax.set_ylabel('Row (pixels)')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 添加坐标信息
        if crs:
            ax.text(0.02, 0.98, f'CRS: {crs.to_epsg() or "Unknown"}',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # ── 颜色条 ───────────────────────────────────────────────
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cfg.get('colorbar_label', cfg['units']), fontsize=9)
    
    # ── 标题 ─────────────────────────────────────────────────
    file_info = tif_path.stem.replace(f'_{var_name}', '')
    if lon_min and lat_min:
        loc_info = f"Lon: {lon_min:.1f}°~{lon_max:.1f}°, Lat: {lat_min:.1f}°~{lat_max:.1f}°"
        title = f"{cfg['title']} ({cfg['units']})\n{file_info}\n{loc_info}"
    else:
        title = f"{cfg['title']} ({cfg['units']})\n{file_info}"
    ax.set_title(title, fontsize=10, pad=10)
    
    # ── 保存 ─────────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig(out_png, dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True

# ============================================================
# ③ 批量处理函数
# ============================================================
def find_geotiff_files(geotiff_dir: Path, variables: list) -> dict:
    """
    查找 GeoTIFF 文件并按变量分组
    
    返回:
        dict: {var_name: [tif_path1, tif_path2, ...]}
    """
    result = {}
    
    # 获取所有 GeoTIFF 文件
    tif_files = list(geotiff_dir.glob('*.tif'))
    if not tif_files:
        print(f'[警告] 未找到任何 GeoTIFF 文件：{geotiff_dir}')
        return result
    
    print(f'  找到 {len(tif_files)} 个 GeoTIFF 文件')
    
    # 按变量分组
    for tif in tif_files:
        stem = tif.stem
        # 提取变量名（文件名最后一部分）
        parts = stem.rsplit('_', 1)
        if len(parts) == 2:
            var = parts[1]
            if not variables or var in variables:
                if var not in result:
                    result[var] = []
                result[var].append(tif)
    
    # 打印统计
    for var, files in result.items():
        print(f'    {var}: {len(files)} 个文件')
    
    return result

def batch_plot(geotiff_dir: Path, plot_dir: Path, variables: list):
    """批量绘制所有 GeoTIFF 文件"""
    
    # 查找文件
    var_files = find_geotiff_files(geotiff_dir, variables)
    if not var_files:
        print('[错误] 没有可绘制的文件')
        return 0, 0
    
    # 创建输出目录
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计
    success_count = 0
    fail_count = 0
    total_files = sum(len(files) for files in var_files.values())
    
    print(f'\n🚀 开始批量绘图 (共 {total_files} 个文件)...')
    print('-' * 70)
    
    # 并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        
        for var_name, tif_list in var_files.items():
            cfg = PLOT_CONFIG.get(var_name, {
                'cmap': 'viridis', 'vmin': 0, 'vmax': 1000,
                'title': var_name, 'units': '', 'colorbar_label': var_name
            })
            
            for tif_path in tif_list:
                out_png = plot_dir / f'{tif_path.stem}.png'
                future = executor.submit(
                    plot_single_geotiff, 
                    tif_path, out_png, var_name, cfg
                )
                futures.append((future, tif_path, out_png))
        
        # 处理结果
        for i, (future, tif_path, out_png) in enumerate(as_completed(futures), 1):
            # 注意：as_completed 不保持顺序，需要重新映射
            pass
        
        # 重新按顺序处理
        for future, tif_path, out_png in futures:
            try:
                ok = future.result()
                if ok:
                    success_count += 1
                    print(f'  [{success_count+fail_count}/{total_files}] ✅ {tif_path.name[:50]}')
                else:
                    fail_count += 1
                    print(f'  [{success_count+fail_count}/{total_files}] ❌ {tif_path.name[:50]}')
            except Exception as e:
                fail_count += 1
                print(f'  [{success_count+fail_count}/{total_files}] ❌ {tif_path.name[:50]} - {e}')
    
    return success_count, fail_count

# ============================================================
# ④ 命令行参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='ITS_LIVE GeoTIFF 批量可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python plot_geotiff_results.py
  python plot_geotiff_results.py -i /path/to/geotiff -o /path/to/plots
  python plot_geotiff_results.py -v v vx -d 300
  python plot_geotiff_results.py --no-cartopy
        '''
    )
    
    parser.add_argument('-i', '--input', type=Path, default=GEOTIFF_DIR,
                       help=f'GeoTIFF 输入目录 (默认: {GEOTIFF_DIR})')
    parser.add_argument('-o', '--output', type=Path, default=PLOT_DIR,
                       help=f'绘图输出目录 (默认: {PLOT_DIR})')
    parser.add_argument('-v', '--variables', nargs='+', choices=['v', 'vx', 'vy', 'v_error'],
                       help='要绘制的变量 (默认: 全部)')
    parser.add_argument('-d', '--dpi', type=int, default=PLOT_DPI,
                       help=f'输出图片 DPI (默认: {PLOT_DPI})')
    parser.add_argument('-w', '--workers', type=int, default=MAX_WORKERS,
                       help=f'并发线程数 (默认: {MAX_WORKERS})')
    parser.add_argument('--no-cartopy', action='store_true',
                       help='不使用 cartopy 极地投影')
    parser.add_argument('--no-basemap', action='store_true',
                       help='不添加海岸线/陆地底图')
    
    return parser.parse_args()

# ============================================================
# ⑤ 主程序
# ============================================================
def main():
    args = parse_args()
    
    # 应用命令行参数
    global GEOTIFF_DIR, PLOT_DIR, PLOT_DPI, MAX_WORKERS, USE_CARTOPY, ADD_BASEMAP
    GEOTIFF_DIR = args.input
    PLOT_DIR = args.output
    PLOT_DPI = args.dpi
    MAX_WORKERS = args.workers
    USE_CARTOPY = not args.no_cartopy
    ADD_BASEMAP = not args.no_basemap
    
    variables = args.variables if args.variables else VARIABLES_TO_PLOT
    
    print('=' * 70)
    print('  ITS_LIVE GeoTIFF 批量可视化工具')
    print('=' * 70)
    print(f'  输入目录 : {GEOTIFF_DIR.resolve()}')
    print(f'  输出目录 : {PLOT_DIR.resolve()}')
    print(f'  绘制变量 : {variables if variables else "全部"}')
    print(f'  图片 DPI : {PLOT_DPI}')
    print(f'  并发线程 : {MAX_WORKERS}')
    print(f'  极地投影 : {"✅ 启用" if USE_CARTOPY else "❌ 禁用"}')
    print(f'  底图要素 : {"✅ 启用" if ADD_BASEMAP else "❌ 禁用"}')
    print()
    
    # 检查输入目录
    if not GEOTIFF_DIR.exists():
        print(f'[错误] 输入目录不存在：{GEOTIFF_DIR}')
        return
    
    # 执行批量绘图
    success, fail = batch_plot(GEOTIFF_DIR, PLOT_DIR, variables)
    
    # 汇总
    print()
    print('=' * 70)
    print(f'  📊 绘图完成！')
    print(f'  成功：{success} 个')
    print(f'  失败：{fail} 个')
    print(f'  输出目录：{PLOT_DIR.resolve()}')
    print('=' * 70)

if __name__ == '__main__':
    main()