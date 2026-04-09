#!/usr/bin/env python3
"""
batch_run.py — 批量运行 autoRIFT_S1_Local_Pipeline
功能：
  1. 自动扫描 SLC 目录，按时间间隔配对
  2. 为每对生成永久 config.yaml 并存放于 results/date1_date2/
  3. 实时流式输出子进程日志（解决终端“卡住”问题）
  4. 支持断点续算、并行处理、超时控制
  5. 完整兼容现有 run_pipeline.py 与 config.yaml 结构

用法：
  # 干运行，确认配对逻辑
  python batch_run.py --slc-dir /data1/SLC --orbit-dir /data2/Orbits \
    --output-dir ./results --interval 12 --dry-run

  # 串行调试（推荐首次使用）
  python batch_run.py --slc-dir /data1/SLC --orbit-dir /data2/Orbits \
    --output-dir ./results --interval 12 --verbose --log-file batch.log

  # 正式并行处理（4进程，超时3小时/对）
  python batch_run.py --slc-dir /data1/SLC --orbit-dir /data2/Orbits \
    --output-dir ./results --interval 12 --max-workers 4 --timeout-minutes 180
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 终端编码兼容 ─────────────────────────────────────────────────────
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

log = logging.getLogger(__name__)

# Sentinel-1 SLC 文件名正则（兼容 S1A/S1B/S1C）
_SLC_RE = re.compile(
    r'^(S1[ABC]_IW_SLC__1S[A-Z]{2}_(\d{8})T\d{6}_\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_[0-9A-F]{4})\.zip$',
    re.IGNORECASE,
)


def parse_slc_dir(slc_dir: Path) -> List[Dict]:
    """扫描 SLC 目录，解析可用 zip 文件。"""
    records = []
    for f in sorted(slc_dir.glob("*.zip")):
        m = _SLC_RE.match(f.name)
        if not m:
            log.warning("跳过非标准文件名: %s", f.name)
            continue
        records.append({
            "granule":   m.group(1),
            "date":      datetime.strptime(m.group(2), "%Y%m%d"),
            "satellite": m.group(1)[:3],  # S1A/S1B/S1C
            "path":      f.resolve(),
        })
    records.sort(key=lambda r: (r["date"], r["granule"]))
    log.info("找到 %d 个有效 SLC 文件", len(records))
    return records


def build_pairs(
    records: List[Dict],
    interval_days: int,
    tolerance_days: int = 2,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
) -> List[Tuple[Dict, Dict]]:
    """按时间间隔 + 容差配对，同天优先选 S1A，避免重叠链。"""
    if date_start:
        records = [r for r in records if r["date"] >= date_start]
    if date_end:
        records = [r for r in records if r["date"] <= date_end]

    # 按日期分组
    by_date: Dict[datetime, List[Dict]] = {}
    for r in records:
        by_date.setdefault(r["date"], []).append(r)

    # 同一天若有多个轨道，优先选 S1A
    unique = []
    for d in sorted(by_date):
        s1a = [r for r in by_date[d] if r["satellite"] == "S1A"]
        unique.append(s1a[0] if s1a else by_date[d][0])

    pairs = []
    seen = set()
    for i, ref in enumerate(unique):
        target = ref["date"] + timedelta(days=interval_days)
        for sec in unique[i + 1:]:
            diff = abs((sec["date"] - target).days)
            if diff <= tolerance_days:
                key = (ref["date"], sec["date"])
                if key not in seen:
                    pairs.append((ref, sec))
                    seen.add(key)
                break  # 每个参考景只匹配一个次景
            if (sec["date"] - target).days > tolerance_days:
                break

    log.info("生成 %d 个配对（间隔 %d±%d 天）", len(pairs), interval_days, tolerance_days)
    return pairs


def _generate_config(
    ref: Dict,
    sec: Dict,
    orbit_dir: str,
    base_config: str,
    output_dir: str,
    pair_name: str,
) -> str:
    """为当前配对生成永久 config.yaml（保存在 results/date1_date2/ 目录下）。"""
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 覆盖像对专属参数
    cfg["input"]["reference_zip"] = str(ref["path"])
    cfg["input"]["secondary_zip"] = str(sec["path"])
    cfg["orbit"]["orbit_dir"] = orbit_dir
    cfg["processing"]["output_dir"] = str(Path(output_dir) / pair_name)
    cfg["processing"]["pair_name"] = pair_name

    # 确保目录存在
    pair_dir = Path(output_dir) / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    config_path = pair_dir / f"{pair_name}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    log.debug("配置文件已写入: %s", config_path)
    return str(config_path)


def _is_completed(output_dir: str, pair_name: str) -> bool:
    """检查配对是否已完成（存在 velocity_*.tif 即认为成功）。"""
    out_path = Path(output_dir) / pair_name
    return any(out_path.glob("velocity_*.tif"))


def _run_pair(
    ref: Dict,
    sec: Dict,
    args: argparse.Namespace,
    base_config: str,
) -> Tuple[str, bool, Optional[str]]:
    """执行单个配对的处理（实时流式输出日志，保留永久配置）。"""
    pair_name = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
    
    # 断点续算：已存在速度场则跳过
    if _is_completed(args.output_dir, pair_name):
        return pair_name, True, None

    try:
        # 生成并保存永久 config.yaml
        config_path = _generate_config(
            ref=ref,
            sec=sec,
            orbit_dir=args.orbit_dir,
            base_config=base_config,
            output_dir=args.output_dir,
            pair_name=pair_name,
        )

        # 构建执行命令
        cmd = [
            sys.executable,
            str(Path(args.pipeline_dir) / "run_pipeline.py"),
            "--config", config_path,
        ]
        if args.verbose:
            cmd.append("--verbose")
        if args.keep_intermediates:
            cmd.append("--keep-intermediates")

        # 环境变量：强制无缓冲输出 + UTF-8 编码
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        log.info("[%s] 开始处理...", pair_name)

        # 实时流式捕获子进程输出
        with subprocess.Popen(
            cmd,
            cwd=str(args.pipeline_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        ) as proc:
            for line in proc.stdout:
                print(f"[{pair_name}] {line.rstrip()}", flush=True)
            proc.wait()

        # 清理？不，永久配置需保留供追溯/重跑
        if proc.returncode != 0:
            return pair_name, False, f"退出码 {proc.returncode}"
        
        log.info("[%s] ✓ 成功", pair_name)
        return pair_name, True, None

    except subprocess.TimeoutExpired:
        return pair_name, False, f"超时 ({args.timeout_minutes} 分钟)"
    except Exception as e:
        return pair_name, False, str(e)


def main():
    p = argparse.ArgumentParser(
        description="批量运行 autoRIFT_S1_Local_Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── 路径参数 ─────────────────────────────────────────────────────
    p.add_argument("--slc-dir", required=True, help="SLC zip 文件目录")
    p.add_argument("--orbit-dir", required=True, help="轨道文件目录 (.EOF)")
    p.add_argument("--output-dir", required=True, help="结果输出根目录（如 ./results）")
    p.add_argument("--pipeline-dir", default=".", help="pipeline 根目录（含 run_pipeline.py）")
    p.add_argument("--base-config", default="config.yaml", help="基础配置文件路径")

    # ── 配对参数 ─────────────────────────────────────────────────────
    p.add_argument("--interval", type=int, default=12, help="目标时间间隔（天）")
    p.add_argument("--tolerance", type=int, default=2, help="时间容差（±天）")
    p.add_argument("--date-start", help="起始日期 YYYY-MM-DD")
    p.add_argument("--date-end", help="结束日期 YYYY-MM-DD")

    # ── 执行参数 ─────────────────────────────────────────────────────
    p.add_argument("--dry-run", action="store_true", help="仅打印配对，不执行")
    p.add_argument("--max-workers", type=int, default=1, help="并行进程数（1=串行）")
    p.add_argument("--timeout-minutes", type=int, default=180, help="单对处理超时（分钟）")
    p.add_argument("--log-file", help="日志文件路径")
    p.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志")
    p.add_argument(
        "--keep-intermediates", action="store_true", default=False,
        help="保留所有中间文件（透传给 run_pipeline.py；默认：仅保留 velocity_*.tif / velocity_*.nc）",
    )

    args = p.parse_args()

    # ── 初始化日志 ───────────────────────────────────────────────────
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    # ── 路径校验 ─────────────────────────────────────────────────────
    slc_dir = Path(args.slc_dir)
    orbit_dir = Path(args.orbit_dir)
    output_dir = Path(args.output_dir)
    pipeline_dir = Path(args.pipeline_dir)
    base_config = Path(args.base_config)

    for p, name in [(slc_dir, "slc-dir"), (orbit_dir, "orbit-dir"),
                    (base_config, "base-config"), (pipeline_dir/"run_pipeline.py", "pipeline-dir")]:
        if not p.exists():
            log.error("%s 不存在: %s", name, p)
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 解析数据与配对 ───────────────────────────────────────────────
    records = parse_slc_dir(slc_dir)
    if not records:
        log.error("未找到有效 SLC 文件")
        sys.exit(1)

    date_start = datetime.strptime(args.date_start, "%Y-%m-%d") if args.date_start else None
    date_end   = datetime.strptime(args.date_end,   "%Y-%m-%d") if args.date_end   else None

    pairs = build_pairs(records, args.interval, args.tolerance, date_start, date_end)
    if not pairs:
        log.error("未生成任何配对")
        sys.exit(1)

    # ── 打印配对预览 ─────────────────────────────────────────────────
    print(f"\n共 {len(pairs)} 个待处理配对：")
    for i, (ref, sec) in enumerate(pairs, 1):
        span = (sec["date"] - ref["date"]).days
        status = "✓ 已完成" if _is_completed(str(output_dir), f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}") else "○ 待处理"
        print(f"  {i:3d}. {ref['date']:%Y-%m-%d} → {sec['date']:%Y-%m-%d} ({span}d) {ref['satellite']}/{sec['satellite']}  [{status}]")
    print()

    if args.dry_run:
        log.info("干运行模式，退出")
        return

    # ── 执行处理 ─────────────────────────────────────────────────────
    log.info("开始批量处理（并行度=%d）", args.max_workers)
    ok, fail, skip = 0, 0, 0
    failed_pairs = []

    if args.max_workers == 1:
        # 串行模式（日志顺序严格，便于调试）
        for ref, sec in pairs:
            pair_name = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
            if _is_completed(args.output_dir, pair_name):
                log.info("[%s] 已完成，跳过", pair_name)
                skip += 1
                continue
            
            pname, success, err = _run_pair(ref, sec, args, str(base_config))
            if success:
                ok += 1
            else:
                log.error("[%s] ✗ 失败: %s", pname, err)
                fail += 1
                failed_pairs.append((pname, err))
    else:
        # 并行模式
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_run_pair, ref, sec, args, str(base_config)): (ref, sec)
                for ref, sec in pairs
            }
            for fut in as_completed(futures):
                ref, sec = futures[fut]
                pair_name = f"{ref['date']:%Y%m%d}_{sec['date']:%Y%m%d}"
                try:
                    pname, success, err = fut.result()
                    if success:
                        ok += 1
                    else:
                        log.error("[%s] ✗ 失败: %s", pname, err)
                        fail += 1
                        failed_pairs.append((pname, err))
                except Exception as e:
                    log.error("[%s] ✗ 异常: %s", pair_name, e)
                    fail += 1
                    failed_pairs.append((pair_name, str(e)))

    # ── 汇总报告 ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"批量处理完成：{ok} 成功 | {fail} 失败 | {skip} 已跳过")
    if failed_pairs:
        print(f"\n失败配对（可重试）:")
        for pname, err in failed_pairs:
            print(f"  - {pname}: {err[:100]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()