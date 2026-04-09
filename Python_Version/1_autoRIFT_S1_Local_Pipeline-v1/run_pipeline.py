#!/usr/bin/env python
"""
run_pipeline.py — autoRIFT_S1_Local_Pipeline 主入口（多 Swath 版，对齐官方目录树）
【v3 新增 & 优化】
- 📂 统一目录路由：所有产物自动落盘至 results/{pair_name}/，批量任务互不干扰
- 🌐 SPS 参数文件解析（向后兼容：全 null 时自动回退均匀值）
- 🚀 速度阈值过滤（velocity_max_m_yr）
- 📊 CF-1.8 NetCDF 输出（含误差估计、CRS、元数据）
- ⚡ ChipSize 偶数约束自动修复
【目录对齐】
results/{pair_name}/
├── product/          : 参考景 Burst 产品
├── product_sec/      : 次景 Burst 产品
├── scratch/          : 参考景临时几何文件
├── s1_cslc.yaml      : COMPASS 配置（逐 Burst 覆盖写入）
├── reference.tif     : 合并后参考景幅度图
├── secondary.tif     : 合并后次景幅度图
├── geogrid_inputs/   : dhdx, srx, ssm, vx0 等 Geogrid 输入
├── window_*.tif      : Geogrid 输出
├── velocity_*.tif    : 最终速度场（GeoTIFF）
├── velocity_*.nc     : 最终速度场（NetCDF）
└── offset_*.tif      : 最终偏移量
"""
# ═══════════════════════════════════════════════════════════════════════════
#  预初始化 isce3 / s1reader，必须在 argparse 之前执行
# ═══════════════════════════════════════════════════════════════════════════
import sys as _sys
_argv_backup = _sys.argv[:]
_sys.argv = [_sys.argv[0]]
try:
    import isce3 as _isce3      # noqa: F401
    import s1reader as _s1r     # noqa: F401
except ImportError:
    pass
finally:
    _sys.argv = _argv_backup
# ═══════════════════════════════════════════════════════════════════════════

import argparse
import logging
from pathlib import Path
import yaml
from src.utils import setup_logging, ensure_dir, get_epsg_from_file
from src.s1_processor import (
    get_common_burst_ids,
    get_orbit_file,
    process_all_bursts_compass,
    merge_burst_amplitudes,
    load_slc_metadata_pair,
)
from src.geogrid_builder import build_geogrid_inputs, run_geogrid_sar
from src.autorift_runner import run_autorift

log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="autoRIFT_S1_Local_Pipeline — 本地多 Swath S1 速度场",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml", help="配置文件路径")
    p.add_argument(
        "--list-bursts", action="store_true",
        help="列出参考景所有可用 burst 后退出（无需提供轨道文件）",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    p.add_argument(
        "--keep-intermediates", action="store_true", default=False,
        help="保留所有中间文件（默认：仅保留 velocity_*.tif / velocity_*.nc，其余删除）",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    for key in ["reference_zip", "secondary_zip", "dem_file"]:
        val = cfg.get("input", {}).get(key, "")
        if not val or not Path(val).exists():
            raise FileNotFoundError(
                f"config.yaml → input.{key} = '{val}' 文件不存在，请检查路径。"
            )


def _date_from_zip(path: str) -> str:
    """从 S1 文件名提取日期 YYYYMMDD。"""
    try:
        return Path(path).name.split("_")[5][:8]
    except (IndexError, ValueError):
        return Path(path).stem[:8]


def _parse_sps_files(cfg: dict) -> dict:
    """从 config 解析 sps_params 节，返回路径字典（不存在则值为 None）。
    向后兼容：sps_params 节缺失 / 全为 null 时，返回空字典，
    build_geogrid_inputs 会自动回退到均匀值模式。
    """
    sps_cfg = cfg.get("sps_params") or {}
    keys = [
        "dhdx_file", "dhdy_file",
        "vx0_file", "vy0_file",
        "search_range_x_file", "search_range_y_file",
        "chip_size_min_x_file", "chip_size_min_y_file",
        "chip_size_max_x_file", "chip_size_max_y_file",
    ]
    sps = {}
    for k in keys:
        v = sps_cfg.get(k) or None
        sps[k] = v

    # 日志汇总
    provided = [k for k, v in sps.items() if v and Path(v).exists()]
    missing  = [k for k, v in sps.items() if v and not Path(v).exists()]
    if provided:
        log.info("SPS 参数文件 (%d 个): %s", len(provided), ", ".join(provided))
    if missing:
        log.warning("SPS 文件路径已配置但文件不存在（已忽略）: %s", ", ".join(missing))
    if not provided:
        log.info("未检测到有效 SPS 参数文件 → 将使用均匀回退值")
    return sps


# ── 中间文件清理 ──────────────────────────────────────────────────────────────

def _cleanup_intermediates(work_dir: str) -> None:
    """删除工作目录中的所有中间文件，仅保留 velocity_*.tif 和 velocity_*.nc。

    删除内容：
      • product/          — 参考景 COMPASS burst 产品
      • product_sec/      — 次景 COMPASS burst 产品
      • scratch/          — COMPASS 临时几何文件
      • geogrid_inputs/   — Geogrid 输入栅格（dhdx / srx / vx0 等）
      • reference.tif     — 合并后参考景幅度图
      • secondary.tif     — 合并后次景幅度图
      • s1_cslc.yaml      — COMPASS 配置副本
      • window_*.tif      — Geogrid 输出窗口栅格
      • offset_*.tif      — 原始像素偏移量
      • *.yaml            — 每对自动生成的配置文件副本
    保留内容：
      • velocity_*.tif
      • velocity_*.nc
    """
    import fnmatch
    import shutil

    keep_globs = ("velocity_*.tif", "velocity_*.nc", "*.yaml")  # 保留 velocity_*.tif / velocity_*.nc 和 *.yaml（如 s1_cslc.yaml）
    work_path  = Path(work_dir)
    n_files = n_dirs = 0

    for item in sorted(work_path.iterdir()):
        if item.is_dir():
            shutil.rmtree(item)
            n_dirs += 1
            log.debug("  删除子目录: %s", item.name)
        elif item.is_file():
            if not any(fnmatch.fnmatch(item.name, g) for g in keep_globs):
                item.unlink()
                n_files += 1
                log.debug("  删除文件: %s", item.name)

    kept = [p.name for p in work_path.iterdir() if p.is_file()]
    log.info(
        "中间文件清理完成：删除 %d 个文件 + %d 个子目录，保留: %s",
        n_files, n_dirs, kept if kept else "（无）",
    )


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log.info("加载配置: %s", args.config)
    cfg = load_config(args.config)

    inp  = cfg["input"]
    orb  = cfg["orbit"]
    proc = cfg["processing"]

    ref_zip  = inp["reference_zip"]
    sec_zip  = inp["secondary_zip"]
    dem_file = inp["dem_file"]
    ssm_file = inp.get("ssm_file") or None

    # ── 解析日期与像对名称 ───────────────────────────────────────────────
    date_ref  = _date_from_zip(ref_zip)
    date_sec  = _date_from_zip(sec_zip)
    pair_name = proc.get("pair_name") or f"{date_ref}_{date_sec}"
    log.info("像对名称: %s", pair_name)

    # ── 📂 统一工作目录路由（核心改造） ───────────────────────────────────
    # 默认 results/，可通过 config.yaml 的 processing.base_results_dir 覆盖
    base_results_dir = proc.get("base_results_dir", "results")
    ensure_dir(base_results_dir)

    # 每个像对独立子目录，包含所有中间文件与最终结果
    work_dir = str(Path(base_results_dir) / pair_name)
    ensure_dir(work_dir)
    log.info("📂 工作目录已设定: %s", work_dir)

    product_dir_ref = str(Path(work_dir) / "product")
    product_dir_sec = str(Path(work_dir) / "product_sec")
    amp_ref         = str(Path(work_dir) / "reference.tif")
    amp_sec         = str(Path(work_dir) / "secondary.tif")
    window_dir      = work_dir  # window_*.tif 直接输出到工作目录
    geogrid_dir     = str(Path(work_dir) / "geogrid_inputs")

    # ── --list-bursts 模式 ───────────────────────────────────────────────
    if args.list_bursts:
        try:
            orbit_ref = get_orbit_file(ref_zip, orb["orbit_dir"], allow_download=False)
        except FileNotFoundError:
            orbit_ref = None
        from src.s1_processor import get_all_bursts
        records = get_all_bursts(ref_zip, orbit_ref)
        print(f"\n参考景可用 Burst（共 {len(records)} 个）：{Path(ref_zip).name}\n")
        print(f"{'swath':>6}  {'burst_idx':>9}  {'ISCE3 ID':>24}  {'sensing_start':>22}")
        print("-" * 70)
        for r in records:
            print(
                f"  IW{r['swath']}  {r['burst_index']:>9}  "
                f"{r['isce3_id']:>24}  {r['sensing_start'][:22]}"
            )
        print()
        _sys.exit(0)

    # ── 解析 SPS 参数文件 ─────────────────────────────────────────────────────
    sps_files = _parse_sps_files(cfg)
    has_prior_v = bool(
        sps_files.get("vx0_file") and Path(sps_files["vx0_file"]).exists()
        and sps_files.get("vy0_file") and Path(sps_files["vy0_file"]).exists()
    )

    # ── Step 1: 轨道文件 ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 1: 获取轨道文件")
    orbit_ref = get_orbit_file(ref_zip, orb["orbit_dir"], allow_download=orb["allow_download"])
    orbit_sec = get_orbit_file(sec_zip, orb["orbit_dir"], allow_download=orb["allow_download"])

    # ── Step 2: Burst 交集 ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 2: 计算共视 Burst 交集")
    common_ids, swaths_used = get_common_burst_ids(ref_zip, sec_zip, orbit_ref, orbit_sec)
    log.info("共视 Burst: %d | 使用 Swath: %s", len(common_ids), swaths_used)

    # ── Step 3 + 4: COMPASS + 合并幅度图 ───────────────────────────────────────
    if Path(amp_ref).exists() and Path(amp_sec).exists():
        log.info("幅度图已存在，跳过 COMPASS 与合并步骤（断点续算）。")
    else:
        log.info("=" * 60)
        log.info("Step 3: COMPASS CSLC（%d burst × 2 景）", len(common_ids))
        process_all_bursts_compass(
            ref_zip=ref_zip,
            sec_zip=sec_zip,
            orbit_ref=orbit_ref,
            orbit_sec=orbit_sec,
            burst_ids_ref=common_ids,
            burst_ids_sec=common_ids,
            dem_file=dem_file,
            product_dir_ref=product_dir_ref,
            product_dir_sec=product_dir_sec,
            work_dir=work_dir,
        )
        log.info("=" * 60)
        log.info("Step 4: 合并 burst 幅度图 → reference.tif / secondary.tif")
        merge_burst_amplitudes(
            burst_ids=common_ids,
            ref_zip=ref_zip,
            orbit_ref=orbit_ref,
            product_dir_ref=product_dir_ref,
            product_dir_sec=product_dir_sec,
            amp_ref_path=amp_ref,
            amp_sec_path=amp_sec,
        )

    # ── Step 5: Geogrid 输入（SPS 模式 or 均匀回退）──────────────────────────
    log.info("=" * 60)
    log.info("Step 5: 构建 Geogrid 输入栅格")
    epsg = proc.get("epsg")
    if not epsg:
        epsg = get_epsg_from_file(dem_file)
    log.info("EPSG: %d", epsg)

    _csmin = proc.get("chip_size_min_m")
    _csmax = proc.get("chip_size_max_m")
    chip_size_min_m_val = float(_csmin) if _csmin is not None else None
    chip_size_max_m_val = float(_csmax) if _csmax is not None else None

    geogrid_files = build_geogrid_inputs(
        dem_file=dem_file,
        ssm_file=ssm_file,
        output_dir=geogrid_dir,
        chip_size_min_m=chip_size_min_m_val,
        chip_size_max_m=chip_size_max_m_val,
        search_range_m_yr=float(proc["search_range_m_yr"]),
        sps_files=sps_files,
    )

    # ── Step 6: Geogrid ───────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 6: runGeogrid（loadMetadataSlc，Swath %s）", swaths_used)
    meta_r, meta_s = load_slc_metadata_pair(
        ref_zip=ref_zip,
        sec_zip=sec_zip,
        orbit_ref=orbit_ref,
        orbit_sec=orbit_sec,
        swaths=swaths_used,
    )
    geogrid_run_info = run_geogrid_sar(
        meta_r=meta_r,
        meta_s=meta_s,
        epsg=epsg,
        geogrid_inputs=geogrid_files,
        working_dir=window_dir,
    )

    # ── Step 7: autoRIFT ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 7: autoRIFT 特征追踪")
    vx0_for_error = geogrid_files["vx"] if has_prior_v else None
    vy0_for_error = geogrid_files["vy"] if has_prior_v else None

    results = run_autorift(
        ref_tif=amp_ref,
        sec_tif=amp_sec,
        window_dir=window_dir,
        output_dir=work_dir,
        pair_name=pair_name,
        filter_type=proc.get("filter_type", "WAL"),
        filter_width=int(proc.get("filter_width", 21)),
        oversample_ratio=int(proc.get("oversample_ratio", 64)),
        n_threads=int(proc.get("n_threads", 4)),
        save_offsets=bool(proc.get("save_offsets", True)),
        # ── v3 新增参数 ─────────────────────────────────────────────────
        velocity_max_m_yr=float(proc.get("velocity_max_m_yr", 20000.0)),
        vx0_tif=vx0_for_error,
        vy0_tif=vy0_for_error,
        save_netcdf=bool(proc.get("output_netcdf", False)),
        date_ref=date_ref,
        date_sec=date_sec,
        epsg=epsg,
    )

    # ── 完成 ──────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("✓ 全部完成！")
    log.info("  速度场 GeoTIFF:  %s", results["velocity_tif"])
    if "offset_tif" in results:
        log.info("  偏移量:          %s", results["offset_tif"])
    if "netcdf" in results:
        log.info("  NetCDF CF-1.8:   %s", results["netcdf"])
    log.info("  Band: Vx(m/yr) / Vy(m/yr) / |V|(m/yr)  NoData=-32767")
    log.info(
        "  误差: Vx_err=%.2f  Vy_err=%.2f  V_err=%.2f  stable_count=%d  [m/yr]",
        results.get("vx_error",  float("nan")),
        results.get("vy_error",  float("nan")),
        results.get("v_error",   float("nan")),
        results.get("stable_count", 0),
    )
    if not has_prior_v:
        log.info("  （误差为稳定地表标准差估计；提供 SPS vx0/vy0 可获得更准确的误差）")

    # ── 中间文件清理 ──────────────────────────────────────────────────────────
    if not args.keep_intermediates:
        log.info("=" * 60)
        log.info("清理中间文件（保留 velocity_*.tif / velocity_*.nc）…")
        _cleanup_intermediates(work_dir)
    else:
        log.info("--keep-intermediates 已启用，跳过中间文件清理")


if __name__ == "__main__":
    main()