#!/usr/bin/env python3
"""Validate and summarize structured LIBERO-Plus per-task results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        action="append",
        required=True,
        help="Results JSONL; repeat to aggregate shards.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help=(
            "Selected manifest; repeat in the same order as --results for sharded runs. "
            "Enables exact coverage, identity, and run-metadata checks."
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero if any manifest task is missing from its paired results.",
    )
    parser.add_argument("--output", type=Path, help="Optional summary CSV path.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        rows.append(row)
    return rows


def key(row: dict[str, Any]) -> tuple[str, int]:
    try:
        return str(row["suite"]), int(row["task_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid task identity in row: {row}") from exc


def validate_results(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, int]] = set()
    for row in rows:
        identity = key(row)
        if identity in seen:
            raise SystemExit(f"Duplicate result for {identity[0]} task_id={identity[1]}")
        seen.add(identity)
        category = row.get("category")
        if category not in CATEGORIES:
            raise SystemExit(f"Unknown category {category!r} for {identity}")
        try:
            episodes = int(row["episodes"])
            successes = int(row["successes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid episode counts for {identity}: {row}") from exc
        if episodes < 1 or not 0 <= successes <= episodes:
            raise SystemExit(
                f"Invalid episodes/successes for {identity}: {episodes}/{successes}"
            )


def validate_coverage(
    results: list[dict[str, Any]], manifest: list[dict[str, Any]]
) -> tuple[int, int]:
    manifest_by_key = {key(row): row for row in manifest}
    if len(manifest_by_key) != len(manifest):
        raise SystemExit("Manifest contains duplicate suite/task_id entries")
    result_keys = {key(row) for row in results}
    unexpected = result_keys - manifest_by_key.keys()
    if unexpected:
        raise SystemExit(f"Results contain tasks outside the manifest: {sorted(unexpected)[:5]}")
    for row in results:
        expected = manifest_by_key[key(row)]
        for field in ("task_name", "category", "difficulty_level"):
            if row.get(field) != expected.get(field):
                raise SystemExit(
                    f"Metadata mismatch for {key(row)} field {field}: "
                    f"result={row.get(field)!r}, manifest={expected.get(field)!r}"
                )
    return len(result_keys), len(manifest_by_key)


def raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Run metadata not found: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid run metadata JSON in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SystemExit(f"Run metadata must be a JSON object: {path}")
    return metadata


def validate_shard_metadata(
    result_paths: list[Path],
    manifest_paths: list[Path],
    result_groups: list[list[dict[str, Any]]],
    manifest_groups: list[list[dict[str, Any]]],
) -> None:
    consistency_fields = ("checkpoint", "suite", "seed", "render_backend")
    reference: dict[str, Any] | None = None

    for index, (result_path, manifest_path, results, manifest) in enumerate(
        zip(result_paths, manifest_paths, result_groups, manifest_groups)
    ):
        metadata_path = result_path.parent / "run_metadata.json"
        metadata = load_run_metadata(metadata_path)
        expected_hash = raw_sha256(manifest_path)
        if metadata.get("manifest_sha256") != expected_hash:
            raise SystemExit(
                f"Shard {index} manifest hash mismatch: metadata={metadata.get('manifest_sha256')!r}, "
                f"actual={expected_hash!r}"
            )

        suites = {str(row.get("suite")) for row in manifest}
        if len(suites) != 1 or metadata.get("suite") != next(iter(suites)):
            raise SystemExit(
                f"Shard {index} suite mismatch between {metadata_path} and {manifest_path}"
            )
        for row in results:
            if row.get("seed") != metadata.get("seed"):
                raise SystemExit(
                    f"Shard {index} result seed mismatch for {key(row)}: "
                    f"result={row.get('seed')!r}, metadata={metadata.get('seed')!r}"
                )

        current = {field: metadata.get(field) for field in consistency_fields}
        missing = [field for field, value in current.items() if value is None]
        if missing:
            raise SystemExit(f"Shard {index} run metadata is missing fields: {missing}")
        if reference is None:
            reference = current
        elif current != reference:
            differences = {
                field: (reference[field], current[field])
                for field in consistency_fields
                if reference[field] != current[field]
            }
            raise SystemExit(f"Shard {index} run metadata differs from shard 0: {differences}")

    print(
        "Run metadata: "
        f"{len(result_paths)} shards consistent "
        f"(checkpoint={reference['checkpoint']}, seed={reference['seed']}, "
        f"suite={reference['suite']}, backend={reference['render_backend']})"
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_category: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        category = row["category"]
        per_category[category]["tasks"] += 1
        per_category[category]["episodes"] += int(row["episodes"])
        per_category[category]["successes"] += int(row["successes"])

    summary: list[dict[str, Any]] = []
    for category in CATEGORIES:
        counts = per_category[category]
        episodes = counts["episodes"]
        summary.append(
            {
                "category": category,
                "tasks": counts["tasks"],
                "episodes": episodes,
                "successes": counts["successes"],
                "success_rate": counts["successes"] / episodes if episodes else None,
            }
        )
    total_episodes = sum(int(row["episodes"]) for row in rows)
    total_successes = sum(int(row["successes"]) for row in rows)
    summary.append(
        {
            "category": "Total",
            "tasks": len(rows),
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": total_successes / total_episodes if total_episodes else None,
        }
    )
    return summary


def write_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"Refusing to overwrite existing summary: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("category", "tasks", "episodes", "successes", "success_rate"),
        )
        writer.writeheader()
        for row in summary:
            output_row = dict(row)
            rate = output_row["success_rate"]
            output_row["success_rate"] = "" if rate is None else f"{rate:.6f}"
            writer.writerow(output_row)


def main() -> None:
    args = parse_args()
    result_groups = [load_jsonl(path) for path in args.results]
    results = [row for group in result_groups for row in group]
    validate_results(results)
    if args.manifest:
        if len(args.results) > 1 and len(args.manifest) != len(args.results):
            raise SystemExit(
                "Sharded aggregation requires one --manifest per --results, in matching order"
            )
        manifest_groups = [load_jsonl(path) for path in args.manifest]
        manifests = [row for group in manifest_groups for row in group]
        complete, expected = validate_coverage(results, manifests)
        print(
            f"Combined coverage: {complete}/{expected} tasks "
            f"({complete / expected:.1%}); missing={expected - complete}"
        )
        if len(args.results) > 1:
            for index, (result_group, manifest_group) in enumerate(
                zip(result_groups, manifest_groups)
            ):
                shard_complete, shard_expected = validate_coverage(
                    result_group, manifest_group
                )
                print(
                    f"  shard {index}: {shard_complete}/{shard_expected} "
                    f"({shard_complete / shard_expected:.1%}); "
                    f"missing={shard_expected - shard_complete}"
                )
            validate_shard_metadata(
                args.results, args.manifest, result_groups, manifest_groups
            )
        if args.require_complete and complete != expected:
            raise SystemExit(
                f"Incomplete results: missing {expected - complete} of {expected} manifest tasks"
            )
    else:
        if args.require_complete:
            raise SystemExit("--require-complete requires at least one --manifest")
        print(f"Coverage: {len(results)} completed tasks (no manifest supplied)")

    summary = summarize(results)
    print(f"{'Category':<27} {'Tasks':>7} {'Episodes':>9} {'Success':>9} {'Rate':>9}")
    for row in summary:
        rate = row["success_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"{row['category']:<27} {row['tasks']:>7} {row['episodes']:>9} "
            f"{row['successes']:>9} {rate_text:>9}"
        )
    if args.output:
        write_csv(args.output, summary)
        print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
