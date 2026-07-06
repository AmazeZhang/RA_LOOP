#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# RA-LOOP  eval/parse_libero_plus_results.py
# ============================================================================
#
# 目的:
#   把 OpenVLA-OFT 评估 log 里的 per-task 结果, 按 LIBERO-Plus 的 7 个
#   category (Camera / Robot / Language / Light / Background / Noise / Layout)
#   聚合成 summary CSV.
#
# 输入:
#   --log_dir              : 评估日志目录 (eval/run_libero_plus.sh 输出)
#   --task_classification  : LIBERO-plus 的 task_classification.json
#                            位置: LIBERO-plus/libero/libero/benchmark/task_classification.json
#                            结构 (核实过): { "libero_spatial": [ {id, name, category, difficulty_level}, ... ], ... }
#
# 输出:
#   --output   CSV, 每行 (category, num_tasks, success_rate)
#
# 关键假设:
#   OpenVLA-OFT 的 run_libero_eval.py 会输出 per-task rollout log,
#   包含 task name 和 success bool. 我们从这些 log 里解析.
#
#   实际 log 格式可能有变化, Week 1 跑通 baseline 时再调整正则.
# ============================================================================

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# 从 OpenVLA-OFT 的 log 中解析 per-task success rate
#
# OpenVLA-OFT run_libero_eval.py 的 log 输出格式 (核实自源码):
#   [INFO] Task X succeeded: True/False
#   [INFO] Task <name>: N/M success rate 0.XX
#
# 我们抓 "Task <name>: N/M" 或 per-episode 的 success/fail 记录.
# 具体正则等 Week 1 baseline 跑完看真实 log 再校准.
# ---------------------------------------------------------------------------

# 匹配 log 中每个 task 的 success 记录 (估计格式, 需要 Week 1 校准)
PATTERN_TASK_RESULT = re.compile(
    r"Task\s+(?P<name>[\w_]+):\s+(?P<succ>\d+)/(?P<total>\d+)"
)


def parse_log(log_file: Path) -> dict[str, tuple[int, int]]:
    """
    Parse an OpenVLA-OFT eval log, return {task_name: (num_success, num_trials)}.

    如果 log 格式变了, 只需修改这个函数.
    """
    results: dict[str, tuple[int, int]] = {}
    if not log_file.exists():
        return results

    text = log_file.read_text(encoding="utf-8", errors="ignore")
    for match in PATTERN_TASK_RESULT.finditer(text):
        name = match.group("name")
        succ = int(match.group("succ"))
        total = int(match.group("total"))
        results[name] = (succ, total)

    return results


def load_task_classification(json_file: Path) -> dict[str, str]:
    """
    Load LIBERO-plus task_classification.json, return {task_name: category}.

    JSON 结构:
      {
        "libero_spatial": [
          {"id": 1, "name": "...", "category": "Background Textures", "difficulty_level": 2},
          ...
        ],
        "libero_object": [...],
        ...
      }
    """
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    mapping: dict[str, str] = {}
    for suite, tasks in data.items():
        for task in tasks:
            mapping[task["name"]] = task["category"]

    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--task_classification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # 加载 task → category 映射
    task2cat = load_task_classification(args.task_classification)
    print(f"[parse] Loaded {len(task2cat)} task→category mappings")

    # 遍历 log 目录下所有 log 文件
    all_results: dict[str, tuple[int, int]] = {}
    for log_file in sorted(args.log_dir.glob("plus_*.log")):
        r = parse_log(log_file)
        print(f"[parse] {log_file.name}: {len(r)} tasks parsed")
        all_results.update(r)

    # 按 category 聚合
    per_cat_succ: dict[str, int] = defaultdict(int)
    per_cat_total: dict[str, int] = defaultdict(int)
    unclassified: list[str] = []

    for task, (succ, total) in all_results.items():
        cat = task2cat.get(task)
        if cat is None:
            unclassified.append(task)
            continue
        per_cat_succ[cat] += succ
        per_cat_total[cat] += total

    # 写 CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "num_tasks", "success", "success_rate"])

        # 7 个官方 category (顺序对齐 LIBERO-Plus README)
        official_cats = [
            "Camera Viewpoints",
            "Robot Initial States",
            "Language Instructions",
            "Light Conditions",
            "Background Textures",
            "Sensor Noise",
            "Objects Layout",
        ]

        # 有些 log 里 category 名字可能微妙不同, 我们做 fuzzy 匹配
        def find_cat(target: str) -> str | None:
            for c in per_cat_total.keys():
                # 简单匹配: 关键词包含
                key = target.split()[0].lower()  # e.g. "camera"
                if key in c.lower():
                    return c
            return None

        total_succ = 0
        total_task = 0
        for cat in official_cats:
            match_cat = find_cat(cat) or cat
            succ = per_cat_succ.get(match_cat, 0)
            n = per_cat_total.get(match_cat, 0)
            rate = succ / n if n > 0 else 0.0
            writer.writerow([cat, n, succ, f"{rate:.4f}"])
            print(f"  {cat:<25} {n:>5} tasks, success rate {rate*100:.1f}%")
            total_succ += succ
            total_task += n

        # 总平均
        total_rate = total_succ / total_task if total_task > 0 else 0.0
        writer.writerow(["Total", total_task, total_succ, f"{total_rate:.4f}"])
        print(f"  {'Total':<25} {total_task:>5} tasks, success rate {total_rate*100:.1f}%")

    if unclassified:
        print(
            f"[parse] WARN: {len(unclassified)} tasks not in task_classification.json, "
            f"first 5: {unclassified[:5]}"
        )


if __name__ == "__main__":
    main()

# ============================================================================
# 成功预期:
#   打印 7 行 category + 1 行 Total, 数字非零
#
# 失败模式:
#   A. all_results 空 (no tasks parsed)
#      —— log 格式与 PATTERN_TASK_RESULT 正则不匹配.
#      —— Fix: 查看实际 log, 调整正则.
#         比如 log 可能是 "Success: True. Task: <name>", 就要改正则.
#
#   B. unclassified 数量 > 100
#      —— task_classification.json 与实际评估的 task 名不完全对齐.
#      —— 可能有 suffix 差异 (task_1 vs task_1_0). Fix: 加 rstrip regex.
#
#   C. 某些 category 数字为 0
#      —— 该 category 的 log 没跑到. 检查 eval/run_libero_plus.sh 是否漏了 suite.
# ============================================================================
