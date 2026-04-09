#!/usr/bin/env python
"""
run_pipeline.py — autoRIFT_S1_Local_Pipeline 主入口（多 Swath 版，对齐官方目录树）

【修复】isce3/pyre 框架会在 import 时尝试解析 sys.argv，
与 argparse 的 --config 参数冲突 → AttributeError: circular import。
解决方案：在 argparse 初始化前，临时清空 sys.argv 完成 isce3 预初始化。

【目录对齐 hyp3_tree.txt】
  - product/          : 参考景 Burst 产品
  - product_sec/      : 次景 Burst 产品
  - scratch/          : 参考景临时几何文件
  - s1_cslc.yaml      : COMPASS 配置（逐 Burst 覆盖写入）
  - reference.tif     : 合并后参考景幅度图
  - secondary.tif     : 合并后次景幅度图
  - window_*.tif      : Geogrid 输出（当前目录）
  - velocity.tif      : 最终速度场
  - offset.tif        : 最终偏移量

处理流程：
  Step 1  获取轨道文件
  Step 2  计算共视 Burst 交集（支持跨卫星/跨轨道容错）
  Step 3  逐 burst 运行 COMPASS（参考景 + 次景，官方路径逻辑）
  Step 4  两层合并幅度图：①单 Swath 内 burst → ②多 Swath 全景
  Step 5  构建 Geogrid 输入栅格（本地模式，无 JPL shapefile）
  Step 6  runGeogrid（loadMetadataSlc 全景元数据）
  Step 7  autoRIFT → velocity.tif（Vx/Vy/|V|，m/yr）
"""
# ═══════════════════════════════════════════════════════════════════════════
#  预初始化 isce3 / s1reader，必须在 argparse 之前执行
#  原因：pyre 框架初始化时会解析 sys.argv，与 --config 参数冲突导致循环导入
# ═══════════════════════════════════════════════════════════════════════════
import sys as _sys
_argv_backup = _sys.argv[:]
_sys.argv = [_sys.argv[0]]      # 临时只保留脚本名，防止 pyre 误解析
try:
    import isce3 as _isce3      # noqa: F401  触发 pyre 初始化
    import s1reader as _s1r     # noqa: F401  确保 s1reader 也完成初始化
except ImportError:
    pass                        # 后续步骤会给出更明确的错误
finally:
    _sys.argv = _argv_backup    # 恢复完整命令行
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
    """从 S1 文件名提取日期 YYYYMMDD（第6个下划线分隔段的前8位）。"""
    try:
        return Path(path).name.split("_")[5][:8]
    except (IndexError, ValueError):
        return Path(path).stem[:8]


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

    # 【对齐官方】使用当前工作目录作为运行根目录
    work_dir = "."
    product_dir_ref = str(Path(work_dir) / "product")
    product_dir_sec = str(Path(work_dir) / "product_sec")
    amp_ref         = str(Path(work_dir) / "reference.tif")
    amp_sec         = str(Path(work_dir) / "secondary.tif")
    window_dir      = "."  # 【对齐官方】window_*.tif 直接输出到当前目录
    geogrid_dir     = str(Path(work_dir) / "geogrid_inputs")

    # ── --list-bursts 模式 ────────────────────────────────────────────────────
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

    # ── 初始化 ───────────────────────────────────────────────────────────────
    date_ref  = _date_from_zip(ref_zip)
    date_sec  = _date_from_zip(sec_zip)
    pair_name = proc.get("pair_name") or f"{date_ref}_{date_sec}"
    log.info("像对名称: %s", pair_name)

    # ── Step 1: 轨道文件 ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 1: 获取轨道文件")
    orbit_ref = get_orbit_file(ref_zip, orb["orbit_dir"], allow_download=orb["allow_download"])
    orbit_sec = get_orbit_file(sec_zip, orb["orbit_dir"], allow_download=orb["allow_download"])

    # ── Step 2: 计算共视 Burst 交集（官方逻辑）──────────────────────────────────
    log.info("=" * 60)
    log.info("Step 2: 计算共视 Burst 交集")
    common_ids, swaths_used = get_common_burst_ids(ref_zip, sec_zip, orbit_ref, orbit_sec)
    log.info("共视 Burst: %d | 使用 Swath: %s", len(common_ids), swaths_used)

    # ── Step 3 + 4: COMPASS + 合并幅度图 ───────────────────────────────────────
    if Path(amp_ref).exists() and Path(amp_sec).exists():
        log.info("幅度图已存在，跳过 COMPASS 与合并步骤。")
    else:
        log.info("=" * 60)
        log.info("Step 3: COMPASS CSLC（%d burst × 2 景）", len(common_ids))
        process_all_bursts_compass(
            ref_zip=ref_zip,
            sec_zip=sec_zip,
            orbit_ref=orbit_ref,
            orbit_sec=orbit_sec,
            burst_ids_ref=common_ids,
            burst_ids_sec=common_ids,  # ← 使用交集
            dem_file=dem_file,
            product_dir_ref=product_dir_ref,   # ← 对齐官方: product/
            product_dir_sec=product_dir_sec,   # ← 对齐官方: product_sec/
            work_dir=work_dir,                 # ← 当前目录
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

    # ── Step 5: Geogrid 输入 ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 5: 构建 Geogrid 输入栅格（本地模式，无 JPL shapefile）")
    epsg = proc.get("epsg")
    if not epsg:
        epsg = get_epsg_from_file(dem_file)
    log.info("自动检测 EPSG: %d", epsg)

    geogrid_files = build_geogrid_inputs(
        dem_file=dem_file,
        ssm_file=ssm_file,
        output_dir=geogrid_dir,
        chip_size_min_m=float(proc["chip_size_min_m"]),
        chip_size_max_m=float(proc["chip_size_max_m"]),
        search_range_m_yr=float(proc["search_range_m_yr"]),
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
    # 【对齐官方】window_*.tif 直接输出到当前目录（working_dir="."）
    geogrid_run_info = run_geogrid_sar(
        meta_r=meta_r,
        meta_s=meta_s,
        epsg=epsg,
        geogrid_inputs=geogrid_files,
        working_dir=window_dir,  # ← 当前目录
    )

    # ── Step 7: autoRIFT ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Step 7: autoRIFT 特征追踪")
    results = run_autorift(
        ref_tif=amp_ref,
        sec_tif=amp_sec,
        window_dir=window_dir,   # ← 当前目录
        output_dir=work_dir,     # ← 当前目录
        pair_name=pair_name,
        filter_type=proc.get("filter_type", "WAL"),
        filter_width=int(proc.get("filter_width", 21)),
        oversample_ratio=int(proc.get("oversample_ratio", 64)),
        n_threads=int(proc.get("n_threads", 4)),
        save_offsets=bool(proc.get("save_offsets", True)),
    )

    # ── 完成 ──────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("✓ 全部完成！")
    log.info("  速度场:  %s", results["velocity_tif"])
    if "offset_tif" in results:
        log.info("  偏移量:  %s", results["offset_tif"])
    log.info("  波段:  Band1=Vx(m/yr)  Band2=Vy(m/yr)  Band3=|V|(m/yr)")
    log.info("  NoData: -32767")


if __name__ == "__main__":
    main()