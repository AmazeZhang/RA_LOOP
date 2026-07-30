#!/usr/bin/env python3
"""Summarize a multi-task same-state language-sensitivity screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


KNOWN_EXCLUSIVE_GROUPS = {
    "bowl_destination_k3": (
        "put_the_bowl_on_the_plate",
        "put_the_bowl_on_the_stove",
        "put_the_bowl_on_top_of_the_cabinet",
    ),
    "wine_destination_k2": (
        "put_the_wine_bottle_on_the_rack",
        "put_the_wine_bottle_on_top_of_the_cabinet",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def rates_for_tasks(
    rows: list[dict[str, Any]], tasks: tuple[str, ...]
) -> dict[str, float | int | bool]:
    selected = [row for row in rows if row["instruction_task"] in tasks]
    hits = [
        bool(row["terminal_goal_truth"][row["instruction_task"]])
        for row in selected
    ]
    misses = [
        bool(row["terminal_goal_truth"][target])
        for row in selected
        for target in tasks
        if target != row["instruction_task"]
    ]
    hit_rate = mean(hits)
    miss_rate = mean(misses)
    lsg = hit_rate - miss_rate
    return {
        "n_rollouts": len(selected),
        "n_mismatch_checks": len(misses),
        "hit": hit_rate,
        "miss": miss_rate,
        "lsg": lsg,
        "high_hit": hit_rate >= 0.5,
        "language_deaf_candidate": hit_rate >= 0.5 and lsg <= 0.15,
    }


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["rows"]
    tasks = tuple(result["tasks"])
    per_target = {}
    for target in tasks:
        hit_values = [
            bool(row["terminal_goal_truth"][target])
            for row in rows
            if row["instruction_task"] == target
        ]
        miss_values = [
            bool(row["terminal_goal_truth"][target])
            for row in rows
            if row["instruction_task"] != target
        ]
        hit = mean(hit_values)
        miss = mean(miss_values)
        per_target[target] = {
            "hit": hit,
            "miss": miss,
            "lsg": hit - miss,
            "high_hit": hit >= 0.5,
            "language_deaf_candidate": hit >= 0.5 and hit - miss <= 0.15,
        }

    known_groups = {
        name: rates_for_tasks(rows, tuple(task for task in group if task in tasks))
        for name, group in KNOWN_EXCLUSIVE_GROUPS.items()
        if all(task in tasks for task in group)
    }
    candidates = [
        task for task, metrics in per_target.items() if metrics["language_deaf_candidate"]
    ]
    valid_group_candidates = [
        name
        for name, metrics in known_groups.items()
        if metrics["language_deaf_candidate"]
    ]
    return {
        "source_result": result.get("output_dir"),
        "num_init_states": result["num_init_states"],
        "num_tasks": len(tasks),
        "num_rollouts": len(rows),
        "all_tasks": rates_for_tasks(rows, tasks),
        "per_target": per_target,
        "known_exclusive_groups": known_groups,
        "raw_target_candidates_requiring_compatibility_review": candidates,
        "valid_exclusive_group_candidates": valid_group_candidates,
        "continue_training_direction": bool(valid_group_candidates),
    }


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    summary = summarize(result)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
