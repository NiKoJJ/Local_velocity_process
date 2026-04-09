"""
ITS_LIVE Sentinel-1 数据下载并转换为 GeoTIFF（含正确坐标系）
区域: 南极洲 (143.4787, -67.4588, 145.9545, -67.1506)
时间: 2015-2020
"""

from pystac_client import Client
import xarray as xr
import numpy as np
import re, requests
from pathlib import Path

# ── 输出目录 ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./itslive_sentinel1_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIABLES_TO_EXPORT = ["v", "vx", "vy", "v_error"]

BAND_NAMES = {
    "v":       "Ice Speed (m/yr)",
    "vx":      "X Velocity (m/yr)",
    "vy":      "Y Velocity (m/yr)",
    "v_error": "Velocity Error (m/yr)",
}


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def is_sentinel1(item) -> bool:
    props = item.properties
    for key in ("mission_img1", "mission_img2", "platform", "mission"):
        val = props.get(key, "")
        if val and re.search(r"sentinel.?1", str(val), re.IGNORECASE):
            return True
    if re.search(r"S1[AB]", item.id):
        return True
    return False


def download_https(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [跳过] 已存在: {dest.name}")
        return dest
    print(f"  [下载] {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  进度: {downloaded/total*100:5.1f}%  ({downloaded>>20} MB / {total>>20} MB)", end="")
        print()
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# 核心：读取 ITS_LIVE NetCDF 的真实 CRS 和仿射变换
# ─────────────────────────────────────────────────────────────────────────────

def get_crs_and_transform(ds: xr.Dataset):
    """
    从 ITS_LIVE NetCDF Dataset 提取 CRS 和仿射变换。

    ITS_LIVE granule 结构：
      - 坐标 x, y：单位为米的投影坐标（极地立体投影）
      - 变量 'mapping'：CF grid_mapping，含 crs_wkt / proj4 / epsg_code
      - 全局属性 'projection' 可能含 EPSG 字符串
    """
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    crs = None

    # 1. 从 'mapping' 变量读取 CRS
    if "mapping" in ds:
        m = ds["mapping"]
        for attr in ("crs_wkt", "spatial_ref", "proj4text", "proj4_params"):
            val = m.attrs.get(attr, "")
            if val:
                try:
                    crs = CRS.from_string(val)
                    print(f"  [CRS] 从 mapping.{attr} 读取: {crs.to_epsg() or crs.name}")
                    break
                except Exception:
                    pass
        if crs is None:
            for attr in ("epsg_code", "EPSG"):
                epsg = m.attrs.get(attr)
                if epsg:
                    try:
                        crs = CRS.from_epsg(int(str(epsg).replace("EPSG:", "")))
                        print(f"  [CRS] 从 mapping.{attr} 读取: EPSG:{crs.to_epsg()}")
                        break
                    except Exception:
                        pass

    # 2. 从全局属性读取
    if crs is None:
        for attr in ("projection", "crs", "proj4"):
            val = ds.attrs.get(attr, "")
            if val:
                try:
                    crs = CRS.from_string(str(val))
                    print(f"  [CRS] 从全局属性 '{attr}' 读取: {crs.to_epsg() or crs.name}")
                    break
                except Exception:
                    pass

    # 3. 南极数据默认 EPSG:3031
    if crs is None:
        print("  [CRS] 未能自动识别，默认使用 EPSG:3031（南极极地立体投影）")
        crs = CRS.from_epsg(3031)

    # 4. 构建仿射变换（像元中心 → 左上角边缘）
    x = ds.coords["x"].values.astype(float)
    y = ds.coords["y"].values.astype(float)
    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])

    # rasterio from_origin: 左上角坐标 + 像元尺寸（正值）
    west  = x[0]    - dx / 2
    north = y[0]    + dy / 2 if y[0] > y[-1] else y[-1] + dy / 2

    transform = from_origin(west, north, dx, dy)

    # 是否需要翻转行（y 从小到大时图像上下颠倒）
    flip_rows = bool(y[0] < y[-1])

    return crs, transform, flip_rows


def array_to_geotiff(data: np.ndarray, crs, transform, out_path: Path,
                     flip_rows: bool = False, nodata=None, band_desc: str = ""):
    """将 2D numpy 数组写入单波段 GeoTIFF，保留完整 CRS 信息。"""
    import rasterio

    if flip_rows:
        data = data[::-1].copy()

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
        if band_desc:
            dst.update_tags(1, description=band_desc)


def netcdf_to_geotiff(nc_path: Path, out_dir: Path) -> list:
    """将 ITS_LIVE granule NetCDF 转 GeoTIFF，保留原始投影坐标系。"""
    try:
        import rasterio
    except ImportError:
        print("  [警告] 未安装 rasterio，跳过 GeoTIFF 转换")
        return []

    ds = xr.open_dataset(nc_path)
    saved = []

    if "x" not in ds.coords or "y" not in ds.coords:
        print(f"  [警告] 找不到 x/y 坐标: {nc_path.name}")
        ds.close()
        return []

    crs, transform, flip_rows = get_crs_and_transform(ds)

    for var in VARIABLES_TO_EXPORT:
        if var not in ds:
            continue
        data = ds[var].values
        if data.ndim == 3:
            data = data[0]
        if data.ndim != 2:
            continue

        nodata = ds[var].attrs.get("_FillValue") or ds[var].attrs.get("missing_value")
        tif_path = out_dir / f"{nc_path.stem}_{var}.tif"
        array_to_geotiff(data, crs, transform, tif_path,
                         flip_rows=flip_rows, nodata=nodata,
                         band_desc=BAND_NAMES.get(var, var))
        print(f"  [保存] {tif_path.name}")
        saved.append(tif_path)

    ds.close()
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 第一部分：下载 Sentinel-1 granule
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("第一部分：搜索并下载 Sentinel-1 影像对 (granules)")
print("=" * 70)

catalog = Client.open("https://stac.itslive.cloud")

search = catalog.search(
    collections=["itslive-granules"],
    bbox=[151.78980703770634, -68.71210627544774, 153.47254985879806, -68.1319386038742], #[143.4787, -67.4588, 145.9545, -67.1506],
    datetime="2025-01-01/2025-12-14",
    max_items=5000,
)

sentinel1_items = []
print("\n正在筛选 Sentinel-1 数据...\n")
for item in search.items():
    if is_sentinel1(item):
        sentinel1_items.append(item)
        print(f"✓ ID   : {item.id}")
        print(f"  日期 : {item.properties.get('datetime', 'N/A')}")
        print(f"  传感器: {item.properties.get('mission_img1', 'S1')}")
        if len(sentinel1_items) >= 50:
            break

print(f"\n共找到 {len(sentinel1_items)} 个 Sentinel-1 影像对\n")

downloaded_nc = []
for item in sentinel1_items:
    asset = (item.assets.get("data") or
             item.assets.get("velocity") or
             next(iter(item.assets.values()), None))
    if asset is None:
        print(f"  [跳过] {item.id}：找不到资产")
        continue
    nc_file = OUTPUT_DIR / f"{item.id}.nc"
    try:
        download_https(asset.href, nc_file)
        downloaded_nc.append(nc_file)
        tifs = netcdf_to_geotiff(nc_file, OUTPUT_DIR)
        print(f"  → 生成 {len(tifs)} 个 GeoTIFF\n")
    except Exception as e:
        print(f"  [错误] {item.id}: {e}\n")

print(f"第一部分完成。已处理 {len(downloaded_nc)} 个 NetCDF 文件")


# ─────────────────────────────────────────────────────────────────────────────
# 第二部分：从 Zarr Cube 提取 Sentinel-1 时间步
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("第二部分：从 ITS_LIVE Zarr Cube 提取 Sentinel-1 并保存 GeoTIFF")
print("=" * 70)

search_cube = catalog.search(
    collections=["itslive-cubes"],
    bbox=[152.6964294, -68.8390986, 153.7930053, -68.7153849],
)

cube_item = next(search_cube.items(), None)
if cube_item is None:
    print("未找到对应区域的数据立方体，跳过。")
else:
    zarr_url = (cube_item.assets.get("zarr") or
                cube_item.assets.get("data")).href
    print(f"\n打开: {zarr_url}\n")

    # decode_timedelta=False 消除 FutureWarning
    ds = xr.open_zarr(zarr_url, consolidated=True, decode_timedelta=False)
    print(ds, "\n")

    # ── 筛选 Sentinel-1 时间步 ────────────────────────────────────────────────
    # satellite_img1 / mission_img1 是数据变量（1D），需 .values 加载
    s1_mask = None
    # ITS_LIVE Zarr cube 编码规则（由实际数据确认）：
    #   satellite_img1: '1A' / '1B' = Sentinel-1A/1B
    #                   '2A' / '2B' = Sentinel-2A/2B
    #                   '7' / '8' / '9' = Landsat 7/8/9
    #   mission_img1:   'S' = Sentinel,  'L' = Landsat
    S1_SATELLITE = {"1A", "1B"}   # satellite_img1 中 Sentinel-1 的值

    for var_name in ("satellite_img1", "mission_img1"):
        if var_name in ds:
            print(f"  加载 '{var_name}' 字段...")
            vals = ds[var_name].values   # 1D，全量加载（小，~100KB）

            if var_name == "satellite_img1":
                # 匹配 '1A' 或 '1B'
                s1_mask = np.array([str(v).strip() in S1_SATELLITE for v in vals])
            else:
                # mission_img1 = 'S' 表示 Sentinel（含 S-1 和 S-2），
                # 需结合 satellite_img1 再精确筛选；此处作为 fallback 先用 'S'
                s1_mask = np.array([str(v).strip() == "S" for v in vals])
                # 若同时有 satellite_img1，交叉过滤出纯 Sentinel-1
                if "satellite_img1" in ds:
                    sat_vals = ds["satellite_img1"].values
                    s1_mask &= np.array([str(v).strip() in S1_SATELLITE for v in sat_vals])

            count = int(s1_mask.sum())
            print(f"  共 {len(vals)} 个时间步，Sentinel-1: {count} 个")
            if count > 0:
                break

    if s1_mask is None or s1_mask.sum() == 0:
        print("  [提示] 未找到 Sentinel-1 时间步。")
        print("  可用的传感器类型：")
        for var_name in ("satellite_img1", "mission_img1"):
            if var_name in ds:
                unique_vals = np.unique(ds[var_name].values)
                print(f"    {var_name}: {unique_vals[:20]}")
        ds.close()
    else:
        from rasterio.crs import CRS
        from rasterio.transform import from_origin

        # ── 构建 CRS（从 URL 或属性中识别 EPSG）────────────────────────────
        proj_str = ds.attrs.get("projection", zarr_url)
        epsg_match = re.search(r"EPSG(\d{4,5})", proj_str, re.IGNORECASE)
        epsg = int(epsg_match.group(1)) if epsg_match else 3031
        crs = CRS.from_epsg(epsg)
        print(f"  [CRS] EPSG:{epsg} — {crs.to_string()}")

        x = ds.coords["x"].values.astype(float)
        y = ds.coords["y"].values.astype(float)
        dx = abs(x[1] - x[0])
        dy = abs(y[1] - y[0])
        west  = x[0]    - dx / 2
        north = max(y[0], y[-1]) + dy / 2
        transform = from_origin(west, north, dx, dy)
        flip_rows = bool(y[0] < y[-1])

        s1_indices = np.where(s1_mask)[0][:5]   # 只取前5个，避免数据量过大
        print(f"\n  导出前 {len(s1_indices)} 个 Sentinel-1 时间步...\n")

        for idx in s1_indices:
            ds_t  = ds.isel(mid_date=int(idx))
            date_str = str(ds.coords["mid_date"].values[idx])[:10]
            print(f"  时间步 {idx}  日期: {date_str}")

            for var in VARIABLES_TO_EXPORT:
                if var not in ds_t:
                    continue
                data = ds_t[var].values
                if data.ndim != 2:
                    continue
                nodata = ds[var].attrs.get("_FillValue")
                out_tif = OUTPUT_DIR / f"cube_S1_{date_str}_{var}.tif"
                array_to_geotiff(data.astype(np.float32), crs, transform,
                                 out_tif, flip_rows=flip_rows, nodata=nodata,
                                 band_desc=BAND_NAMES.get(var, var))
                print(f"    [保存] {out_tif.name}")

        ds.close()

print("\n" + "=" * 70)
print(f"全部完成！文件保存在: {OUTPUT_DIR.resolve()}")
print("=" * 70)
