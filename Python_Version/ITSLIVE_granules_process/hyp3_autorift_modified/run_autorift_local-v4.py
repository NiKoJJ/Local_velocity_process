#!/usr/bin/env python3
"""
run_autorift_local-v2.py
========================
批量用本地 Sentinel-1 SLC zip 文件运行 hyp3-autorift，跳过下载。

原理：
  1. 通过环境变量 LOCAL_SLC_DIR 告知修改版 s1_isce3._resolve_safe
     本地 zip 位置，isce3 配准阶段跳过网络下载。
  2. 在工作目录创建 zip 软链接，供 netcdf_output.py 写输出时读取
     SAFE 元数据（cal_swath_offset_bias → loadMetadata）。

用法：
  # 干运行，确认配对和命令
  python run_autorift_local-v2.py \
      --slc-dir /data1/PhD_Work1/Cook/Sentinel-1/SLC/SLC_2022_2025 \
      --output-dir /data1/PhD_Work1/AutoRIFT/results \
      --interval 12 --dry-run

  # 正式处理，60 线程，12 天间隔
  python run_autorift_local-v2.py \
      --slc-dir /data1/PhD_Work1/Cook/Sentinel-1/SLC/SLC_2022_2025 \
      --output-dir /data1/PhD_Work1/AutoRIFT/results \
      --interval 12 \
      --omp-threads 60 \
      --chip-size 64 \
      --search-range 32

  # 6 天间隔（S1A + S1C 混合）
  python run_autorift_local-v2.py \
      --slc-dir /data1/PhD_Work1/Cook/Sentinel-1/SLC/SLC_2022_2025 \
      --output-dir /data1/PhD_Work1/AutoRIFT/results \
      --interval 6 \
      --omp-threads 60
"""

from __future__ import annotations
import argparse
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

_SLC_RE = re.compile(
    r'^(S1[ABC]_IW_SLC__1S[A-Z]{2}_(\d{8})T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4})\.zip$',
    re.IGNORECASE,
)


def parse_slc_dir(slc_dir: Path) -> list[dict]:
    records = []
    for f in sorted(slc_dir.glob("*.zip")):
        m = _SLC_RE.match(f.name)
        if not m:
            log.warning("跳过: %s", f.name)
            continue
        records.append({
            "granule": m.group(1),
            "date":    datetime.strptime(m.group(2), "%Y%m%d"),
            "path":    f,
        })
    records.sort(key=lambda r: (r["date"], r["granule"]))
    log.info("找到 %d 个 SLC 文件", len(records))
    return records


def build_pairs(
    records:        list[dict],
    interval_days:  int,
    tolerance_days: int = 2,
    date_start:     datetime | None = None,
    date_end:       datetime | None = None,
) -> list[tuple[dict, dict]]:
    """按时间间隔配对，(ref_date, sec_date) 去重，同天优先选 S1A。"""
    if date_start:
        records = [r for r in records if r["date"] >= date_start]
    if date_end:
        records = [r for r in records if r["date"] <= date_end]

    by_date: dict[datetime, list[dict]] = defaultdict(list)
    for r in records:
        by_date[r["date"]].append(r)

    def pick_one(recs: list[dict]) -> dict:
        s1a = [r for r in recs if r["granule"].upper().startswith("S1A")]
        return s1a[0] if s1a else recs[0]

    unique_records = [pick_one(by_date[d]) for d in sorted(by_date)]

    pairs: list[tuple[dict, dict]] = []
    seen:  set[tuple] = set()

    for i, ref in enumerate(unique_records):
        target = ref["date"] + timedelta(days=interval_days)
        for sec in unique_records[i + 1:]:
            diff = abs((sec["date"] - target).days)
            if diff <= tolerance_days:
                key = (ref["date"], sec["date"])
                if key not in seen:
                    pairs.append((ref, sec))
                    seen.add(key)
                break
            if (sec["date"] - target).days > tolerance_days:
                break

    log.info("生成 %d 个配对（间隔 %d 天，容差 ±%d 天）",
             len(pairs), interval_days, tolerance_days)
    return pairs


def symlink_zip(src: Path, work_dir: Path) -> None:
    """
    在工作目录创建指向本地 zip 的软链接。

    原因：isce3 配准阶段通过 LOCAL_SLC_DIR 直接读取本地文件，
    但 netcdf_output.py 的 cal_swath_offset_bias → loadMetadata
    会在当前工作目录搜索 *.zip / *.SAFE 来读取元数据，
    若工作目录没有 zip 则返回空列表，导致 argmin of empty sequence。
    软链接不占额外磁盘空间，且对两处都透明。
    """
    dst = work_dir / src.name
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)
    log.debug("软链接: %s -> %s", dst.name, src.resolve())


def run_pair(
    ref:         dict,
    sec:         dict,
    output_dir:  Path,
    slc_dir:     Path,
    extra_args:  list[str],
    omp_threads: int = 4,
    dry_run:     bool = False,
) -> bool:
    pair_name = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
    out_sub   = output_dir / pair_name

    if list(out_sub.glob("*.nc")):
        log.info("已完成，跳过: %s", pair_name)
        return True

    span = (sec["date"] - ref["date"]).days
    log.info("处理: %s  (%dd)", pair_name, span)

    cmd = [
        "pixi", "run",
        "python", "-m", "hyp3_autorift",
        "++omp-num-threads", str(omp_threads),
        "++process", "hyp3_autorift",
        "--reference", ref["granule"],
        "--secondary", sec["granule"],
        "--parameter-file", "/home/junjie/Desktop/Cryosphere_Datasets/MEaSUREs/MEaSUREs Antarctic Boundaries/IceShelf_Antarctica_v02.shp",
    ] + extra_args

    env = os.environ.copy()
    env["LOCAL_SLC_DIR"] = str(slc_dir.resolve())

    if dry_run:
        print(f"[DRY] {' '.join(cmd)}")
        print(f"[DRY] CWD={out_sub}  LOCAL_SLC_DIR={env['LOCAL_SLC_DIR']}")
        return True

    out_sub.mkdir(parents=True, exist_ok=True)

    # 软链接 zip 到工作目录：
    #   - isce3 配准通过 LOCAL_SLC_DIR 读文件（绝对路径），不依赖此链接
    #   - netcdf_output.py 写 netCDF 时在 cwd 搜索 *.zip 读取 SAFE 元数据，
    #     必须有此链接否则报 ValueError: argmin of empty sequence
    for rec in [ref, sec]:
        src = slc_dir / f"{rec['granule']}.zip"
        if src.exists():
            symlink_zip(src, out_sub)
        else:
            log.warning("本地 zip 不存在，将尝试下载: %s", src)

    print(f"\n{'='*60}")
    print(f"[CMD] {' '.join(cmd)}")
    print(f"[CWD] {out_sub}")
    print(f"[ENV] LOCAL_SLC_DIR={env['LOCAL_SLC_DIR']}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(out_sub), env=env)

    if result.returncode != 0:
        log.error("失败: %s (exit %d)", pair_name, result.returncode)
        return False

    log.info("完成: %s", pair_name)
    return True


def main():
    p = argparse.ArgumentParser(
        description="批量用本地 S1 SLC 文件运行 hyp3-autorift（跳过下载）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--slc-dir",      required=True,
                   help="本地 SLC zip 文件目录（同时作为 LOCAL_SLC_DIR）")
    p.add_argument("--output-dir",   required=True,
                   help="结果输出目录（每对在独立子目录）")
    p.add_argument("--interval",     type=int, default=12,
                   help="配对时间间隔（天）")
    p.add_argument("--tolerance",    type=int, default=2,
                   help="时间间隔容差（天）")
    p.add_argument("--date-start",   default=None, help="YYYY-MM-DD")
    p.add_argument("--date-end",     default=None, help="YYYY-MM-DD")
    p.add_argument("--dry-run",      action="store_true",
                   help="只打印命令，不实际处理")
    p.add_argument("--omp-threads",  type=int, default=4,
                   help="OpenMP 线程数，传给 ++omp-num-threads（默认 4）")
    p.add_argument("--chip-size",    type=int, default=None,
                   help="模板匹配芯片大小（像素），如 32、64")
    p.add_argument("--search-range", type=int, default=None,
                   help="搜索窗口范围（像素），如 20、32")
    p.add_argument("--naming-scheme", default="ITS_LIVE_OD",
                   choices=["ITS_LIVE_OD", "ITS_LIVE_PROD"])
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    slc_dir    = Path(args.slc_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_slc_dir(slc_dir)
    if not records:
        log.error("没有找到 SLC 文件")
        sys.exit(1)

    date_start = datetime.strptime(args.date_start, "%Y-%m-%d") if args.date_start else None
    date_end   = datetime.strptime(args.date_end,   "%Y-%m-%d") if args.date_end   else None

    pairs = build_pairs(records, args.interval, args.tolerance, date_start, date_end)
    if not pairs:
        log.error("没有生成任何配对")
        sys.exit(1)

    print(f"\n共 {len(pairs)} 个配对：")
    for i, (ref, sec) in enumerate(pairs, 1):
        span = (sec["date"] - ref["date"]).days
        print(f"  {i:3d}. {ref['date']:%Y-%m-%d} -> {sec['date']:%Y-%m-%d}"
              f"  ({span}d)  {ref['granule'][:3]}/{sec['granule'][:3]}")
    print()

    extra_args = ["--naming-scheme", args.naming_scheme]
    if args.chip_size:
        extra_args += ["--chip-size", str(args.chip_size)]
    if args.search_range:
        extra_args += ["--search-range", str(args.search_range)]

    if args.dry_run:
        print("--dry-run 模式，打印命令但不实际处理：\n")
        for ref, sec in pairs:
            run_pair(ref, sec, output_dir, slc_dir, extra_args,
                     omp_threads=args.omp_threads, dry_run=True)
        return

    ok = fail = 0
    for ref, sec in pairs:
        if run_pair(ref, sec, output_dir, slc_dir, extra_args,
                    omp_threads=args.omp_threads, dry_run=False):
            ok += 1
        else:
            fail += 1

    print(f"\n完成: {ok} 成功  {fail} 失败  共 {len(pairs)} 对")


if __name__ == "__main__":
    main()
