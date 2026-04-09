"""
autorift_runner.py — 直接驱动 autoRIFT 并输出速度场 GeoTIFF

只使用 hyp3_autorift.vend.testautoRIFT（无 testGeogrid_ISCE 依赖）。
不生成 NetCDF，不裁剪，不依赖 JPL parameter_file。

【v2 修复】
  - _build_image_nodata_mask 返回 (noDataMask, zeroMask) 双掩膜
  - run_autorift 中设置 obj.zeroMask 供 Wallis 滤波使用
  - 强化边界检查，防止索引越界导致 OpenCV 崩溃

输出：
  velocity_{pair}.tif  → Band1=Vx  Band2=Vy  Band3=|V|  (m/yr, float32)
  offset_{pair}.tif    → Band1=DX  Band2=DY  Band3=InterpMask  Band4=ChipSizeX  (pixel)
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from osgeo import gdal

from .utils import read_geotiff, write_geotiff, ensure_dir

gdal.UseExceptions()
log = logging.getLogger(__name__)

NODATA_VEL    = -32767.0
NODATA_OFFSET = -32767.0


# ── 主入口 ────────────────────────────────────────────────────────────────────

def run_autorift(
    ref_tif: str,
    sec_tif: str,
    window_dir: str,
    output_dir: str,
    pair_name: str,
    filter_type: str = "WAL",
    filter_width: int = 21,
    oversample_ratio: int = 64,
    n_threads: int = 4,
    save_offsets: bool = True,
) -> dict:
    """运行 autoRIFT 并保存速度场 GeoTIFF。

    Parameters
    ----------
    ref_tif, sec_tif : str
        参考/次景幅度图（float32，来自 s1_processor.merge_burst_amplitudes）
    window_dir : str
        包含 window_*.tif 的 Geogrid 输出目录
    output_dir : str
        速度场输出目录
    pair_name : str
        文件名标识符
    filter_type : str
        'WAL'=Wallis, 'HPS'=高通, 'SOB'=Sobel, 'LAP'=Laplacian
    filter_width : int
        Wallis 滤波窗口宽度
    oversample_ratio : int
        亚像素过采样倍数（32 / 64 / 128）
    n_threads : int
        OpenMP 线程数（通过 OMP_NUM_THREADS 传递）
    save_offsets : bool
        是否同时保存像素偏移量

    Returns
    -------
    dict  {'velocity_tif': ..., 'offset_tif': ...（可选）}
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    ensure_dir(output_dir)

    try:
        from autoRIFT import autoRIFT
    except ImportError:
        raise ImportError(
            "请安装 autoRIFT：conda install -c conda-forge autorift\n"
            "或参考 README 中的安装说明。"
        )

    # ── 加载影像 ─────────────────────────────────────────────────────────────
    log.info("加载幅度图...")
    I1 = _load_image(ref_tif)
    I2 = _load_image(sec_tif)
    log.info("参考景尺寸: %s | 次景尺寸: %s", I1.shape, I2.shape)

    # ── 加载 Geogrid window_*.tif ────────────────────────────────────────────
    wd = Path(window_dir)
    log.info("加载 window_*.tif (%s)...", window_dir)

    xGrid, yGrid, geo_transform, projection = _load_grid_location(
        str(wd / "window_location.tif")
    )
    Dx0,    Dy0    = _load_2band(str(wd / "window_offset.tif"))
    SRx0,   SRy0   = _load_2band(str(wd / "window_search_range.tif"))
    CSMINx0, CSMINy0 = _load_2band_optional(str(wd / "window_chip_size_min.tif"))
    CSMAXx0, CSMAXy0 = _load_2band_optional(str(wd / "window_chip_size_max.tif"))
    noDataMask       = _load_ssm_mask(str(wd / "window_stable_surface_mask.tif"), xGrid.shape)

    offset2vx_path = str(wd / "window_rdr_off2vel_x_vec.tif")
    offset2vy_path = str(wd / "window_rdr_off2vel_y_vec.tif")
    NODATA_VAL     = -32767

    # ── 构建图像 no-data 掩膜 ────────────────────────────────────────────────
    log.info("构建 noDataMask（SAR 零幅度区域）...")
    # 【修复】接收两个返回值：noDataMask 和 zeroMask
    noDataMask, zeroMask = _build_image_nodata_mask(I1, I2, xGrid, yGrid, noDataMask, NODATA_VAL)

    # ── 初始化 autoRIFT ──────────────────────────────────────────────────────
    log.info("初始化 autoRIFT 对象...")
    obj = autoRIFT()
    obj.MultiThread = 0       # 线程数由 OMP_NUM_THREADS 控制
    obj.I1 = I1
    obj.I2 = I2
    obj.xGrid = xGrid.copy()
    obj.yGrid = yGrid.copy()

    # ── 搜索范围 ─────────────────────────────────────────────────────────────
    if SRx0 is not None:
        obj.SearchLimitX = SRx0.copy()
        obj.SearchLimitY = SRy0.copy()
        # 非零区域加 2 pixel 缓冲
        mask_nz = (obj.SearchLimitX != 0) & (obj.SearchLimitX != NODATA_VAL)
        obj.SearchLimitX[mask_nz] += 2
        obj.SearchLimitY[mask_nz] += 2
    else:
        obj.SearchLimitX = 15
        obj.SearchLimitY = 15

    # ── Chip size ─────────────────────────────────────────────────────────────
    if CSMINx0 is not None:
        obj.ChipSizeMinX = CSMINx0.copy()
        obj.ChipSizeMaxX = CSMAXx0.copy()
        valid = (CSMINx0 != NODATA_VAL) & (CSMINy0 != NODATA_VAL) & (CSMINx0 > 0)
        if valid.any():
            obj.ScaleChipSizeY = float(np.median(CSMINy0[valid] / CSMINx0[valid]))
        else:
            obj.ScaleChipSizeY = 1.0
    else:
        obj.ChipSizeMinX = 16
        obj.ChipSizeMaxX = 64

    # ── 初始偏移量（SAR 方位向取反）─────────────────────────────────────────
    if Dx0 is not None:
        obj.Dx0 = Dx0.copy()
        obj.Dy0 = -1.0 * Dy0.copy()   # ← SAR 方位向约定
    else:
        obj.Dx0 = np.zeros_like(xGrid, dtype=np.float32)
        obj.Dy0 = np.zeros_like(yGrid, dtype=np.float32)

    # ── 应用 noDataMask ──────────────────────────────────────────────────────
    obj.xGrid[noDataMask] = 0
    obj.yGrid[noDataMask] = 0
    obj.Dx0[noDataMask]   = 0
    obj.Dy0[noDataMask]   = 0
    if SRx0 is not None:
        obj.SearchLimitX[noDataMask] = 0
        obj.SearchLimitY[noDataMask] = 0
    if CSMINx0 is not None:
        obj.ChipSizeMinX[noDataMask] = 0
        obj.ChipSizeMaxX[noDataMask] = 0

    # ──【关键修复】设置 zeroMask 供 Wallis 滤波使用（模仿官方逻辑）────────────
    # autoRIFT 的 Wallis 滤波需要 zeroMask 来跳过无效像素，防止边界越界
    obj.zeroMask = zeroMask.astype(np.uint8)
    log.debug("zeroMask 设置: %d / %d 像素标记为零值", zeroMask.sum(), zeroMask.size)

    # ── 预处理滤波 ───────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("预处理滤波: %s (width=%d)...", filter_type, filter_width)
    _apply_filter(obj, filter_type, filter_width)
    obj.uniform_data_type()
    log.info("预处理完成 (%.1f s)", time.time() - t0)

    obj.sparseSearchSampleRate = 1
    obj.OverSampleRatio        = oversample_ratio

    # ── 运行 autoRIFT ────────────────────────────────────────────────────────
    t0 = time.time()
    log.info("运行 autoRIFT 特征追踪...")
    obj.runAutorift()
    log.info("autoRIFT 完成 (%.1f s)", time.time() - t0)

    # ── 扩张 noDataMask 消除边缘伪迹 ─────────────────────────────────────────
    kernel     = np.ones((3, 3), np.uint8)
    noDataMask = cv2.dilate(
        noDataMask.astype(np.uint8), kernel, iterations=1
    ).astype(bool)

    # ── 读取并整理结果 ────────────────────────────────────────────────────────
    grid_shape = xGrid.shape
    DX         = _pad_to(np.asarray(obj.Dx,         dtype=np.float32), grid_shape)
    DY         = _pad_to(np.asarray(obj.Dy,         dtype=np.float32), grid_shape)
    INTERPMASK = _pad_to(np.asarray(obj.InterpMask, dtype=np.float32), grid_shape)
    CHIPSIZEX  = _pad_to(np.asarray(obj.ChipSizeX,  dtype=np.float32), grid_shape)

    # 掩膜无效区
    DX[noDataMask]         = np.nan
    DY[noDataMask]         = np.nan
    INTERPMASK[noDataMask] = 0
    CHIPSIZEX[noDataMask]  = 0
    if SRx0 is not None:
        out_of_range = (SRx0 == 0) | (SRx0 == NODATA_VAL)
        DX[out_of_range]         = np.nan
        DY[out_of_range]         = np.nan
        INTERPMASK[out_of_range] = 0

    # ── 偏移量 → 速度 ─────────────────────────────────────────────────────────
    log.info("偏移量转换为速度场（m/yr）...")
    VX, VY = _offsets_to_velocity(DX, DY, offset2vx_path, offset2vy_path, NODATA_VAL)
    V_MAG  = np.full_like(VX, np.nan, dtype=np.float32)
    valid  = np.isfinite(VX) & np.isfinite(VY)
    V_MAG[valid] = np.sqrt(VX[valid] ** 2 + VY[valid] ** 2)

    log.info(
        "速度统计（有效像素=%d / 总=%d）: |V| mean=%.1f  max=%.1f  [m/yr]",
        int(valid.sum()), valid.size,
        float(np.nanmean(V_MAG)),
        float(np.nanmax(V_MAG)),
    )

    # ── 保存结果 ─────────────────────────────────────────────────────────────
    results = {}
    
    vel_tif = str(Path(output_dir) / f"velocity_{pair_name}.tif")
    _save_velocity(vel_tif, VX, VY, V_MAG, geo_transform, projection)
    results["velocity_tif"] = vel_tif

    if save_offsets:
        off_tif = str(Path(output_dir) / f"offset_{pair_name}.tif")
        _save_offsets(off_tif, DX, DY, INTERPMASK, CHIPSIZEX, geo_transform, projection)
        results["offset_tif"] = off_tif

    return results


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _load_image(path: str) -> np.ndarray:
    ds  = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"无法打开幅度图: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds  = None
    return arr


def _load_grid_location(path: str):
    arr, gt, proj = read_geotiff(path)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError(f"window_location.tif 应为 2 波段，实际 shape={arr.shape}")
    return arr[0].astype(np.int32), arr[1].astype(np.int32), gt, proj


def _load_2band(path: str):
    arr, _, _ = read_geotiff(path)
    if arr.ndim == 3 and arr.shape[0] >= 2:
        return arr[0].astype(np.float32), arr[1].astype(np.float32)
    if arr.ndim == 2:
        return arr.astype(np.float32), arr.astype(np.float32)
    return None, None


def _load_2band_optional(path: str):
    if not Path(path).exists():
        return None, None
    return _load_2band(path)


def _load_ssm_mask(ssm_path: str, shape: tuple) -> np.ndarray:
    if not Path(ssm_path).exists():
        return np.zeros(shape, dtype=bool)
    arr, _, _ = read_geotiff(ssm_path)
    mask = (arr != 0).astype(bool)
    # 对齐尺寸
    if mask.ndim == 3:
        mask = mask[0]
    if mask.shape != shape:
        mask = mask[:shape[0], :shape[1]]
    return mask


def _build_image_nodata_mask(
    I1: np.ndarray,
    I2: np.ndarray,
    xGrid: np.ndarray,
    yGrid: np.ndarray,
    init_mask: np.ndarray,
    nodata_val: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """在 SSM 掩膜基础上，增加幅度为零（SAR 无效区）的像素掩膜。
    
    【修复】返回 (noDataMask, zeroMask) 双掩膜：
      - noDataMask: 标记需要跳过的网格点（用于 autoRIFT 搜索）
      - zeroMask: 标记图像中幅度为 0 的像素（供 Wallis 滤波跳过无效区域）
    
    Returns
    -------
    noDataMask : bool array (rows, cols)
        标记需要跳过的网格点
    zeroMask : bool array (height, width)
        标记图像中幅度为 0 的像素（与 I1/I2 同尺寸）
    """
    mask = init_mask.copy()
    rows, cols = xGrid.shape
    h1, w1 = I1.shape  # I1/I2 是 (rows, cols) 格式
    
    # 【新增】构建 zeroMask：标记图像中值为 0 的像素（供 Wallis 滤波使用）
    zeroMask = (I1 == 0) | (I2 == 0)
    
    for ii in range(rows):
        for jj in range(cols):
            xi, yi = int(xGrid[ii, jj]), int(yGrid[ii, jj])
            
            # 【修复1】检查 nodata 和有效范围（坐标从1开始计数）
            if xi == nodata_val or yi == nodata_val or xi < 1 or yi < 1:
                mask[ii, jj] = True
                continue
            
            # 【修复2】转换为0基索引，并严格检查边界（>= 而非 >）
            r, c = yi - 1, xi - 1  # GDAL坐标(1,1) → numpy索引[0,0]
            if r >= h1 or c >= w1 or r < 0 or c < 0:  # ← 严格 >= 检查
                mask[ii, jj] = True  # 超出幅度图范围的标记为无效
                continue
                
            # 【修复3】安全访问像素值，用 try/except 兜底
            try:
                if I1[r, c] == 0 or I2[r, c] == 0:
                    mask[ii, jj] = True
            except IndexError:
                # 兜底：任何索引错误都标记为无效
                mask[ii, jj] = True
                
    return mask, zeroMask  # ← 返回两个掩膜


def _apply_filter(obj, filter_type: str, filter_width: int) -> None:
    ft = filter_type.upper()
    if ft == "WAL":
        obj.WallisFilterWidth = filter_width
        obj.preprocess_filt_wal()
    elif ft == "HPS":
        obj.preprocess_filt_hps()
    elif ft == "SOB":
        obj.preprocess_filt_sob()
    elif ft == "LAP":
        obj.preprocess_filt_lap()
    else:
        log.warning("未知滤波类型 '%s'，使用 HPS 高通滤波", filter_type)
        obj.preprocess_filt_hps()


def _offsets_to_velocity(
    DX: np.ndarray,
    DY: np.ndarray,
    vx_path: str,
    vy_path: str,
    nodata_val: float,
) -> tuple:
    """像素偏移 (DX, DY) → 地理速度 (Vx, Vy) m/yr。

    Vx = off2vx_1 * DX + off2vx_2 * DY
    Vy = off2vy_1 * DX + off2vy_2 * DY
    （系数来自 Geogrid 的 window_rdr_off2vel_*.tif）
    """
    def _load_coeff(path):
        arr, _, _ = read_geotiff(path)
        if arr.ndim == 3:
            b1 = arr[0].astype(np.float64)
            b2 = arr[1].astype(np.float64)
        else:
            b1 = arr.astype(np.float64)
            b2 = np.zeros_like(b1)
        b1[b1 == nodata_val] = np.nan
        b2[b2 == nodata_val] = np.nan
        return b1, b2

    vx1, vx2 = _load_coeff(vx_path)
    vy1, vy2 = _load_coeff(vy_path)

    DX64 = DX.astype(np.float64)
    DY64 = DY.astype(np.float64)

    VX = (vx1 * DX64 + vx2 * DY64).astype(np.float32)
    VY = (vy1 * DX64 + vy2 * DY64).astype(np.float32)
    VX[~np.isfinite(DX64)] = np.nan
    VY[~np.isfinite(DY64)] = np.nan
    return VX, VY


def _save_velocity(path, VX, VY, V, geo_transform, projection):
    def _fill(a):
        x = a.copy(); x[~np.isfinite(x)] = NODATA_VEL; return x.astype(np.float32)
    write_geotiff(
        path, [_fill(VX), _fill(VY), _fill(V)],
        geo_transform, projection,
        nodata=NODATA_VEL,
        band_names=["Vx_m_yr", "Vy_m_yr", "V_m_yr"],
    )
    log.info("速度场已写入: %s", path)


def _save_offsets(path, DX, DY, INTERPMASK, CHIPSIZEX, geo_transform, projection):
    def _fill(a, nv):
        x = a.copy(); x[~np.isfinite(x)] = nv; return x.astype(np.float32)
    write_geotiff(
        path,
        [_fill(DX, NODATA_OFFSET), _fill(DY, NODATA_OFFSET),
         _fill(INTERPMASK, 0),      _fill(CHIPSIZEX, 0)],
        geo_transform, projection,
        nodata=NODATA_OFFSET,
        band_names=["DX_px", "DY_px", "InterpMask", "ChipSizeX"],
    )
    log.info("偏移量已写入: %s", path)


def _pad_to(arr: np.ndarray, shape: tuple) -> np.ndarray:
    if arr.shape == shape:
        return arr
    out = np.full(shape, np.nan, dtype=np.float32)
    r = min(arr.shape[0], shape[0])
    c = min(arr.shape[1], shape[1])
    out[:r, :c] = arr[:r, :c]
    return out