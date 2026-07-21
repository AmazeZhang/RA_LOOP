#!/usr/bin/env python3
"""Build and validate a deterministic LIBERO-Plus evaluation manifest.

This utility is deliberately simulation- and GPU-free.  It treats the official
``task_classification.json`` as metadata and cross-checks its ordering against
``libero_suite_task_map.py``, whose zero-based position is the task id consumed
by OpenVLA-OFT's evaluator.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


def default_benchmark_dir() -> Path:
    code_dir = Path(os.environ.get("CODE_DIR", Path.home() / "code"))
    return code_dir / "LIBERO-plus" / "libero" / "libero" / "benchmark"


def parse_args() -> argparse.Namespace:
    benchmark_dir = default_benchmark_dir()
    parser = argparse.ArgumentParser(
        description="Validate and select LIBERO-Plus tasks without loading a model or simulator."
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=benchmark_dir / "task_classification.json",
    )
    parser.add_argument(
        "--task-map",
        type=Path,
        default=benchmark_dir / "libero_suite_task_map.py",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=SUITES,
        help="Suite to include; repeat as needed. Default: all four suites.",
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=CATEGORIES,
        help="Category to include; repeat as needed. Default: all seven categories.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Hard cap applied after filtering and sharding.",
    )
    parser.add_argument(
        "--robot-init-per-base",
        type=int,
        choices=(1, 2, 3),
        help=(
            "Select 1-3 Robot Initial States rows per underlying base task. "
            "For 2/3 rows, choose deterministic difficulty extremes/median."
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional .jsonl or .csv manifest path.")
    parser.add_argument("--preview", type=int, default=8)
    return parser.parse_args()


def load_task_map(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                break
            return value
    raise ValueError(f"Could not find a literal libero_task_map in {path}")


def build_manifest(
    classification_path: Path, task_map_path: Path
) -> list[dict[str, Any]]:
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    task_map = load_task_map(task_map_path)
    rows: list[dict[str, Any]] = []

    if tuple(classification) != SUITES:
        raise ValueError(
            f"Unexpected classification suites/order: {tuple(classification)!r}; expected {SUITES!r}"
        )

    for suite in SUITES:
        metadata = classification[suite]
        names = task_map[suite]
        if len(metadata) != len(names):
            raise ValueError(
                f"{suite}: classification has {len(metadata)} tasks but task map has {len(names)}"
            )
        for task_id, (item, mapped_name) in enumerate(zip(metadata, names)):
            if item["name"] != mapped_name:
                raise ValueError(
                    f"{suite} task {task_id}: classification name {item['name']!r} "
                    f"does not match benchmark name {mapped_name!r}"
                )
            if item["category"] not in CATEGORIES:
                raise ValueError(
                    f"{suite} task {task_id}: unknown category {item['category']!r}"
                )
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "classification_id": item["id"],
                    "task_name": mapped_name,
                    "category": item["category"],
                    "difficulty_level": item["difficulty_level"],
                }
            )
    return rows


def fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise ValueError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    elif path.suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
                "suite", "task_id", "classification_id", "task_name", "category", "difficulty_level"
            ])
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("--output must end in .jsonl or .csv")


def print_counts(label: str, rows: list[dict[str, Any]]) -> None:
    suites = Counter(row["suite"] for row in rows)
    categories = Counter(row["category"] for row in rows)
    print(f"{label}: {len(rows)} tasks / {len(rows)} episodes")
    print("  suites: " + ", ".join(f"{key}={suites[key]}" for key in SUITES if suites[key]))
    print(
        "  categories: "
        + ", ".join(f"{key}={categories[key]}" for key in CATEGORIES if categories[key])
    )


def select_robot_init_per_base(
    rows: list[dict[str, Any]], per_base: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["category"] != "Robot Initial States":
            raise SystemExit(
                "--robot-init-per-base requires selecting only category 'Robot Initial States'"
            )
        if "_view_" not in row["task_name"]:
            raise SystemExit(f"Cannot derive Robot-init base task from {row['task_name']!r}")
        base_task = row["task_name"].split("_view_", 1)[0]
        groups.setdefault((row["suite"], base_task), []).append(row)

    selected: list[dict[str, Any]] = []
    for group_rows in groups.values():
        ordered = sorted(
            group_rows, key=lambda row: (row["difficulty_level"], row["task_id"])
        )
        if per_base == 1:
            indices = [len(ordered) // 2]
        elif per_base == 2:
            indices = [0, len(ordered) - 1]
        else:
            indices = [0, len(ordered) // 2, len(ordered) - 1]
        selected.extend(ordered[index] for index in indices)

    # Restore benchmark order so task ids and sharding remain easy to audit.
    return sorted(selected, key=lambda row: (SUITES.index(row["suite"]), row["task_id"]))


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < num-shards")
    if args.max_tasks is not None and args.max_tasks < 1:
        raise SystemExit("--max-tasks must be >= 1")
    if args.preview < 0:
        raise SystemExit("--preview must be >= 0")

    all_rows = build_manifest(args.classification, args.task_map)
    print_counts("Official manifest", all_rows)

    suites = set(args.suite or SUITES)
    categories = set(args.category or CATEGORIES)
    filtered = [
        row for row in all_rows if row["suite"] in suites and row["category"] in categories
    ]
    if args.robot_init_per_base is not None:
        filtered = select_robot_init_per_base(filtered, args.robot_init_per_base)
        print(
            f"Robot-init stratification: {len(filtered) // args.robot_init_per_base} "
            f"base task(s) x {args.robot_init_per_base} row(s)"
        )
    selected = [
        row for index, row in enumerate(filtered) if index % args.num_shards == args.shard_index
    ]
    if args.max_tasks is not None:
        selected = selected[: args.max_tasks]

    print_counts("Selected manifest", selected)
    print(f"  shard: {args.shard_index}/{args.num_shards}")
    print(f"  sha256: {fingerprint(selected)}")
    if not selected:
        raise SystemExit("Selection is empty")

    for row in selected[: args.preview]:
        print(
            f"  {row['suite']} task_id={row['task_id']} category={row['category']!r} "
            f"difficulty={row['difficulty_level']} name={row['task_name']}"
        )

    if args.output:
        write_manifest(args.output, selected)
        print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
