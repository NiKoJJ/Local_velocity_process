"""
geogrid_builder.py — 构建 Geogrid 输入栅格并运行 Geogrid

替代 JPL shapefile 查找，采用 CautoRIFT 思路：
  - 从 DEM 计算坡度（dhdx, dhdy）
  - 按用户配置创建均匀 chip_size / search_range 栅格
  - 重采样 SSM（若提供）到 DEM 格网
  - 调用 hyp3_autorift.vend.testGeogrid.runGeogrid 输出 window_*.tif
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from osgeo import gdal

from .utils import (
    read_geotiff,
    write_geotiff,
    get_raster_info,
    ensure_dir,
)

gdal.UseExceptions()
log = logging.getLogger(__name__)


# ── 1. 构建 Geogrid 输入栅格 ───────────────────────────────────────────────────

def build_geogrid_inputs(
    dem_file: str,
    ssm_file: Optional[str],
    output_dir: str,
    chip_size_min_m: float = 240.0,
    chip_size_max_m: float = 960.0,
    search_range_m_yr: float = 3000.0,
) -> Dict[str, str]:
    """从 DEM + SSM 构建 Geogrid 所需全部输入栅格。

    Returns
    -------
    dict  键名与 runGeogrid 参数名一一对应，可直接解包：
          runGeogrid(info, info1, epsg=..., **geogrid_files)
    """
    ensure_dir(output_dir)
    out = Path(output_dir).resolve()  # ← 确保 output_dir 是绝对路径

    # ── 读取 DEM ────────────────────────────────────────────────────────────
    log.info("读取 DEM: %s", dem_file)
    dem_arr, gt, proj = read_geotiff(dem_file)
    dem_arr = dem_arr.astype(np.float32)
    rows, cols = dem_arr.shape
    dx = abs(gt[1])
    dy = abs(gt[5])
    log.info("DEM 尺寸: %d × %d  分辨率: %.1f m × %.1f m", rows, cols, dx, dy)

    # ── 坡度 ─────────────────────────────────────────────────────────────────
    log.info("计算坡度 dhdx / dhdy...")
    dhdx = (np.gradient(dem_arr, axis=1) / dx).astype(np.float32)
    dhdy = (np.gradient(dem_arr, axis=0) / dy).astype(np.float32)
    # 去除边缘异常梯度（>5 视为无效）
    dhdx[np.abs(dhdx) > 5.0] = 0.0
    dhdy[np.abs(dhdy) > 5.0] = 0.0

    # ── 均匀 chip size 栅格（单位 m）────────────────────────────────────────
    log.info("chip size: min=%.0f m, max=%.0f m", chip_size_min_m, chip_size_max_m)
    csmin = np.full((rows, cols), chip_size_min_m, dtype=np.uint16)
    csmax = np.full((rows, cols), chip_size_max_m, dtype=np.uint16)

    # ── 均匀搜索范围（单位 m/yr）────────────────────────────────────────────
    log.info("搜索范围: %.0f m/yr", search_range_m_yr)
    sr = np.full((rows, cols), search_range_m_yr, dtype=np.int16)

    # ── 零参考速度场 ─────────────────────────────────────────────────────────
    vx_zero = np.zeros((rows, cols), dtype=np.float32)
    vy_zero = np.zeros((rows, cols), dtype=np.float32)

    # ── 稳定地表掩膜 ─────────────────────────────────────────────────────────
    if ssm_file and Path(ssm_file).exists():
        log.info("读取 SSM: %s", ssm_file)
        ssm_arr = _resample_ssm(ssm_file, dem_file)
    else:
        log.info("未提供 SSM，使用全零掩膜")
        ssm_arr = np.zeros((rows, cols), dtype=np.uint8)

    # ── 写出 ─────────────────────────────────────────────────────────────────
    # 【修复】所有路径使用 .resolve() 转为绝对路径，防止后续 chdir 导致路径失效
    files = {
        "dem":    str(Path(dem_file).resolve()),
        "dhdx":   str((out / "dhdx.tif").resolve()),
        "dhdy":   str((out / "dhdy.tif").resolve()),
        "csminx": str((out / "csminx.tif").resolve()),
        "csminy": str((out / "csminy.tif").resolve()),
        "csmaxx": str((out / "csmaxx.tif").resolve()),
        "csmaxy": str((out / "csmaxy.tif").resolve()),
        "srx":    str((out / "srx.tif").resolve()),
        "sry":    str((out / "sry.tif").resolve()),
        "vx":     str((out / "vx.tif").resolve()),
        "vy":     str((out / "vy.tif").resolve()),
        "ssm":    str((out / "ssm.tif").resolve()),
    }

    write_geotiff(files["dhdx"],   [dhdx],    gt, proj, nodata=0.0,      band_names=["dhdx"])
    write_geotiff(files["dhdy"],   [dhdy],    gt, proj, nodata=0.0,      band_names=["dhdy"])
    write_geotiff(files["csminx"], [csmin],   gt, proj,                  band_names=["csminx"])
    write_geotiff(files["csminy"], [csmin],   gt, proj,                  band_names=["csminy"])
    write_geotiff(files["csmaxx"], [csmax],   gt, proj,                  band_names=["csmaxx"])
    write_geotiff(files["csmaxy"], [csmax],   gt, proj,                  band_names=["csmaxy"])
    write_geotiff(files["srx"],    [sr],      gt, proj,                  band_names=["srx"])
    write_geotiff(files["sry"],    [sr],      gt, proj,                  band_names=["sry"])
    write_geotiff(files["vx"],     [vx_zero], gt, proj, nodata=-32767.0, band_names=["vx"])
    write_geotiff(files["vy"],     [vy_zero], gt, proj, nodata=-32767.0, band_names=["vy"])
    write_geotiff(files["ssm"],    [ssm_arr], gt, proj, nodata=255,      band_names=["ssm"])

    log.info("Geogrid 输入栅格已写入: %s", output_dir)
    return files


def _resample_ssm(ssm_file: str, dem_file: str) -> np.ndarray:
    """将 SSM 重采样（最近邻）到 DEM 格网并二值化。"""
    info = get_raster_info(dem_file)
    warp_opts = gdal.WarpOptions(
        format="MEM",
        width=info["cols"],
        height=info["rows"],
        outputBounds=(info["x_min"], info["y_min"], info["x_max"], info["y_max"]),
        resampleAlg="near",
        dstSRS=info["projection"],
    )
    ds  = gdal.Warp("", ssm_file, options=warp_opts)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    arr[arr > 0] = 1
    ds  = None
    return arr


# ── 2. 运行 Geogrid ───────────────────────────────────────────────────────────

def run_geogrid_sar(
    meta_r,
    meta_s,
    epsg: int,
    geogrid_inputs: Dict[str, str],
    working_dir: str = ".",
) -> dict:
    """调用 hyp3_autorift.vend.testGeogrid.runGeogrid 生成 window_*.tif。"""
    from hyp3_autorift.vend.testGeogrid import runGeogrid

    orig_dir = os.getcwd()
    ensure_dir(working_dir)
    
    # 【修复】确保所有输入路径为绝对路径（防止 chdir 后路径解析错误）
    geogrid_inputs_abs = {}
    for k, v in geogrid_inputs.items():
        p = Path(v)
        # 如果文件存在且路径是相对的，转为绝对路径
        if p.exists() and not p.is_absolute():
            geogrid_inputs_abs[k] = str(p.resolve())
        else:
            geogrid_inputs_abs[k] = v  # 已是绝对路径或文件不存在（让 runGeogrid 报错）

    os.chdir(working_dir)

    try:
        log.info("运行 Geogrid（EPSG:%d）...", epsg)
        geogrid_run_info = runGeogrid(
            info=meta_r,
            info1=meta_s,
            optical_flag=0,     # SAR 模式
            epsg=epsg,
            **geogrid_inputs_abs,  # ← 使用绝对路径
        )
        gdal.AllRegister()      # Geogrid 有时会注销 GDAL 驱动
        ycount = geogrid_run_info.get("ycount", "?")
        xcount = geogrid_run_info.get("xcount", "?")
        log.info("Geogrid 完成。输出格网: %s × %s", ycount, xcount)
    finally:
        os.chdir(orig_dir)  # ← 确保恢复原目录，避免影响后续步骤

    _ensure_scale_factor(working_dir, geogrid_run_info)
    return geogrid_run_info


def _ensure_scale_factor(working_dir: str, geogrid_run_info: dict) -> None:
    """若 window_scale_factor.tif 不存在则创建全1占位栅格。"""
    sf_path  = Path(working_dir) / "window_scale_factor.tif"
    loc_path = Path(working_dir) / "window_location.tif"

    if sf_path.exists() or not loc_path.exists():
        return

    arr, gt, proj = read_geotiff(str(loc_path))
    if arr.ndim == 3:
        ycount, xcount = arr.shape[1], arr.shape[2]
    else:
        ycount, xcount = arr.shape

    ones = np.ones((ycount, xcount), dtype=np.float64)
    write_geotiff(str(sf_path), [ones, ones], gt, proj,
                  band_names=["scale_x", "scale_y"])
    log.info("已生成 window_scale_factor.tif（全1占位）")