#!/usr/bin/env python3
"""
ITS_LIVE NC → GeoTIFF 批量转换
将 /data1/nc_files/ 下的所有 NC 文件转换为具有地理参考的 GeoTIFF
"""
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import xarray as xr

# ============================================================
# ① 配置区域
# ============================================================
INPUT_DIR  = Path('./r')           # NC 文件输入目录
OUTPUT_DIR = Path('./nc_tif')     # GeoTIFF 输出目录
VARIABLES_TO_EXPORT = ['v', 'vx', 'vy']  #, 'v_error'
MAX_WORKERS = 8  # 并发线程数

BAND_NAMES = {
    'v': 'Ice Speed (m/yr)',
    'vx': 'X Velocity (m/yr)',
    'vy': 'Y Velocity (m/yr)',
    #'v_error': 'Velocity Error (m/yr)',
}

# ============================================================
# ② CRS / 仿射变换提取（来自原代码）
# ============================================================
def get_crs_and_transform(ds: xr.Dataset, epsg_hint=3031):
    """从 ITS_LIVE xarray Dataset 提取 CRS 和仿射变换。"""
    try:
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
    except ImportError:
        print('    [错误] rasterio 未安装，请运行：pip install rasterio')
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
        print(f'    [CRS] from hint: EPSG:{epsg_hint}')
    
    if crs is None:
        crs = CRS.from_epsg(3031)
        print('    [CRS] default EPSG:3031')
    
    # 提取仿射变换参数
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
# ③ 数组 → GeoTIFF（来自原代码）
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
# ④ 单个 NC 文件转换
# ============================================================
def convert_nc_to_geotiff(nc_path: Path, out_dir: Path, epsg_hint=3031) -> tuple:
    """将单个 ITS_LIVE NetCDF 转换为逐变量 GeoTIFF。"""
    try:
        import rasterio
    except ImportError:
        print(f'    [错误] rasterio 未安装')
        return nc_path.name, False, 'rasterio missing'
    
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        return nc_path.name, False, f'打开失败：{e}'
    
    saved = []
    if 'x' not in ds.coords or 'y' not in ds.coords:
        ds.close()
        return nc_path.name, False, '缺少 x/y 坐标'
    
    crs, transform, flip_rows = get_crs_and_transform(ds, epsg_hint)
    if crs is None:
        ds.close()
        return nc_path.name, False, 'CRS 提取失败'
    
    for var in VARIABLES_TO_EXPORT:
        if var not in ds:
            continue
        data = ds[var].values
        if data.ndim == 3:
            data = data[0]
        if data.ndim != 2:
            continue
            
        var_suffix = var.capitalize()
        
        nodata = ds[var].attrs.get('_FillValue') or ds[var].attrs.get('missing_value')
        tif_path = out_dir / f'{nc_path.stem}-{var_suffix}.tif'
        
        try:
            array_to_geotiff(
                data, crs, transform, tif_path,
                flip_rows=flip_rows, nodata=nodata,
                band_desc=BAND_NAMES.get(var, var),
            )
            saved.append(tif_path)
        except Exception as e:
            print(f'    [警告] {var} 转换失败：{e}')
    
    ds.close()
    return nc_path.name, len(saved) > 0, f'生成 {len(saved)} 个 GeoTIFF'

# ============================================================
# ⑤ 主程序
# ============================================================
def main():
    print('=' * 70)
    print('  ITS_LIVE NC → GeoTIFF 批量转换')
    print('=' * 70)
    
    # 检查输入目录
    if not INPUT_DIR.exists():
        print(f'[错误] 输入目录不存在：{INPUT_DIR}')
        return
    
    # 获取所有 NC 文件
    nc_files = list(INPUT_DIR.glob('*.nc'))
    if not nc_files:
        print(f'[错误] 未找到任何 .nc 文件：{INPUT_DIR}')
        return
    
    print(f'  输入目录：{INPUT_DIR.resolve()}')
    print(f'  输出目录：{OUTPUT_DIR.resolve()}')
    print(f'  NC 文件数：{len(nc_files)}')
    print(f'  导出变量：{VARIABLES_TO_EXPORT}')
    print(f'  并发线程：{MAX_WORKERS}')
    print()
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 并发转换
    success_count = 0
    fail_count = 0
    total_tifs = 0
    
    print('🚀 开始批量转换...')
    print('-' * 70)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(convert_nc_to_geotiff, nc_file, OUTPUT_DIR): nc_file
            for nc_file in nc_files
        }
        
        for i, future in enumerate(as_completed(futures), start=1):
            nc_name, ok, msg = future.result()
            if ok:
                success_count += 1
                tifs_created = int(msg.split()[1]) if '生成' in msg else 0
                total_tifs += tifs_created
                print(f'  [{i}/{len(nc_files)}] ✅ {nc_name[:50]}  →  {msg}')
            else:
                fail_count += 1
                print(f'  [{i}/{len(nc_files)}] ❌ {nc_name[:50]}  →  {msg}')
    
    # 汇总
    print()
    print('=' * 70)
    print(f'  转换完成！')
    print(f'  成功：{success_count} 个 NC 文件')
    print(f'  失败：{fail_count} 个 NC 文件')
    print(f'  生成 GeoTIFF 总数：{total_tifs} 个')
    print(f'  输出目录：{OUTPUT_DIR.resolve()}')
    print('=' * 70)

if __name__ == '__main__':
    main()
