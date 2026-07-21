#!/usr/bin/env python3
"""Bounded, resumable LIBERO-Plus evaluation driver.

The default mode is a CPU-only plan check.  Model/simulator imports happen only
after ``--execute`` is supplied, a single physical GPU id is specified, and all
safety checks pass.
"""

from __future__ import annotations

import argparse
import contextlib
import filecmp
import hashlib
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


SUITE_TO_UNNORM = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or execute a strictly bounded LIBERO-Plus manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--openvla-root",
        type=Path,
        default=Path(os.environ.get("CODE_DIR", Path.home() / "code")) / "openvla-oft",
    )
    parser.add_argument(
        "--libero-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".libero",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, help="Required physical GPU id in execute mode.")
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=5,
        help="Second hard cap, independent of the manifest size (default: 5).",
    )
    parser.add_argument(
        "--render-backend",
        choices=("osmesa",),
        default="osmesa",
        help="EGL is intentionally unavailable because of the recorded driver incident.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually load the model and simulator. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    required = {"suite", "task_id", "task_name", "category", "difficulty_level"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise SystemExit(f"Manifest row {index} is missing: {sorted(missing)}")
        if row["suite"] not in SUITE_TO_UNNORM:
            raise SystemExit(f"Manifest row {index} has unsupported suite: {row['suite']!r}")
        if not isinstance(row["task_id"], int) or row["task_id"] < 0:
            raise SystemExit(f"Manifest row {index} has invalid task_id: {row['task_id']!r}")
    if not rows:
        raise SystemExit("Manifest is empty")
    return rows, hashlib.sha256(raw).hexdigest()


def load_completed(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            completed.add((row["suite"], int(row["task_id"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Invalid results JSONL at line {line_number}: {exc}") from exc
    return completed


def validate_paths(args: argparse.Namespace) -> None:
    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    if not args.checkpoint.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {args.checkpoint}")
    eval_script = args.openvla_root / "experiments" / "robot" / "libero" / "run_libero_eval.py"
    if not eval_script.is_file():
        raise SystemExit(f"OpenVLA-OFT evaluator not found: {eval_script}")
    if not (args.libero_config / "config.yaml").is_file():
        raise SystemExit(f"LIBERO-Plus config not found: {args.libero_config / 'config.yaml'}")
    if args.max_tasks < 1:
        raise SystemExit("--max-tasks must be >= 1")
    if args.gpu_id is not None and args.gpu_id < 0:
        raise SystemExit("--gpu-id must be >= 0")


def validate_checkpoint_suite(checkpoint: Path, suite: str) -> None:
    """Fail before model loading when a checkpoint cannot unnormalize this suite."""
    statistics_path = checkpoint / "dataset_statistics.json"
    if not statistics_path.is_file():
        raise SystemExit(f"Checkpoint statistics not found: {statistics_path}")
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    expected_key = SUITE_TO_UNNORM[suite]
    if expected_key not in statistics:
        raise SystemExit(
            f"Checkpoint is incompatible with {suite}: expected normalization key "
            f"{expected_key!r}, available keys are {sorted(statistics)}"
        )


def validate_checkpoint_read_only_compatibility(checkpoint: Path, openvla_root: Path) -> None:
    """Verify everything OpenVLA would otherwise rewrite, without changing files."""
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_auto_map = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }
    if config.get("auto_map") != expected_auto_map:
        raise SystemExit(
            "Checkpoint auto_map is not ready for read-only loading. Refusing to let "
            f"OpenVLA rewrite {config_path}."
        )

    source_dir = openvla_root / "prismatic" / "extern" / "hf"
    for filename in ("modeling_prismatic.py", "configuration_prismatic.py"):
        source = source_dir / filename
        checkpoint_file = checkpoint / filename
        if not source.is_file() or not checkpoint_file.is_file():
            raise SystemExit(
                f"Read-only model logic check requires both files: {source} and {checkpoint_file}"
            )
        if filecmp.cmp(source, checkpoint_file, shallow=False):
            continue

        # The verified official OFT checkpoints differ from the current OpenVLA-OFT
        # tree only by this diffusion-scheduler initialization. Our bounded
        # evaluator uses L1 regression (use_diffusion=False), so the branch is
        # unreachable. Accept exactly this known difference while preserving
        # the checkpoint's bundled code; reject every other mismatch.
        if filename == "modeling_prismatic.py":
            checkpoint_text = checkpoint_file.read_text(encoding="utf-8")
            source_text = source.read_text(encoding="utf-8")
            known_diff = (
                "        # Set diffusion timestep values\n"
                "        action_head.noise_scheduler.set_timesteps(action_head.num_diffusion_steps)\n"
            )
            if checkpoint_text.count(known_diff) == 1 and checkpoint_text.replace(
                known_diff, "", 1
            ) == source_text:
                print(
                    "Checkpoint modeling logic differs only in the unused diffusion branch; "
                    "accepted for read-only L1 evaluation"
                )
                continue
        raise SystemExit(
            f"Model logic mismatch for {filename}; refusing OpenVLA's automatic checkpoint rewrite"
        )


@contextlib.contextmanager
def working_directory(path: Path):
    """Temporarily change cwd and always restore it."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def print_plan(args: argparse.Namespace, rows: list[dict[str, Any]], pending: list[dict[str, Any]]) -> None:
    print("LIBERO-Plus bounded evaluation plan")
    print(f"  mode: {'EXECUTE' if args.execute else 'DRY-RUN (no model/simulator/GPU imports)'}")
    print(f"  checkpoint: {args.checkpoint.resolve()}")
    print(f"  manifest: {args.manifest.resolve()}")
    print(f"  selected: {len(rows)} task(s), pending: {len(pending)} task(s), 1 episode/task")
    print(f"  suites: {dict(Counter(row['suite'] for row in pending))}")
    print(f"  categories: {dict(Counter(row['category'] for row in pending))}")
    print(f"  physical GPU: {args.gpu_id if args.gpu_id is not None else 'none'}")
    print(f"  render backend: {args.render_backend}")
    print(f"  expected peak VRAM: approximately 17-20 GiB (model-dependent)")
    print(f"  output: {args.output_dir.resolve()}")
    print("  process timeout: none inside evaluator (external launcher decides)")


def prepare_output(
    args: argparse.Namespace, manifest_hash: str, selected: list[dict[str, Any]]
) -> tuple[Path, set[tuple[str, int]]]:
    results_path = args.output_dir / "results.jsonl"
    metadata_path = args.output_dir / "run_metadata.json"

    if args.output_dir.exists() and not args.resume:
        raise SystemExit(
            f"Output directory already exists: {args.output_dir}; choose a new directory or pass --resume"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expected = {
        "manifest_sha256": manifest_hash,
        "checkpoint": str(args.checkpoint.resolve()),
        "suite": selected[0]["suite"],
        "seed": args.seed,
        "render_backend": args.render_backend,
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(
                    f"Resume metadata mismatch for {key}: existing={existing.get(key)!r}, requested={value!r}"
                )
    else:
        expected["created_at"] = datetime.now().astimezone().isoformat()
        metadata_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results_path, load_completed(results_path)


def execute(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    results_path: Path,
    completed: set[tuple[str, int]],
) -> None:
    # These environment variables must be set before importing torch, MuJoCo, or OpenVLA.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["MUJOCO_GL"] = args.render_backend
    os.environ["PYOPENGL_PLATFORM"] = args.render_backend
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.chdir(args.openvla_root)
    sys.path.insert(0, str(args.openvla_root))
    from experiments.robot.libero import run_libero_eval as upstream  # noqa: PLC0415
    from libero.libero import benchmark  # noqa: PLC0415
    from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: PLC0415
    from experiments.robot import openvla_utils  # noqa: PLC0415
    from tqdm.auto import tqdm as overall_progress  # noqa: PLC0415

    # The upstream loader always backs up and rewrites local checkpoint files.
    # Preflight validation has already proven these files are current, so turn
    # both mutators into no-ops for this process and keep the checkpoint read-only.
    openvla_utils.update_auto_map = lambda _checkpoint: None
    openvla_utils.check_model_logic_mismatch = lambda _checkpoint: None

    # Upstream writes videos below ./rollouts. Redirect only that call into this
    # run's output directory, restoring cwd immediately afterwards.
    upstream_save_rollout_video = upstream.save_rollout_video

    def bounded_save_rollout_video(*save_args, **save_kwargs):
        with working_directory(args.output_dir.resolve()):
            return upstream_save_rollout_video(*save_args, **save_kwargs)

    upstream.save_rollout_video = bounded_save_rollout_video

    # run_task creates a separate one-episode progress bar for every task.
    # Suppress those and provide one stable, resumable task-level bar below.
    upstream.tqdm.tqdm = lambda iterable, *progress_args, **progress_kwargs: iterable

    suite_name = selected[0]["suite"]
    cfg = upstream.GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        task_suite_name=suite_name,
        unnorm_key=SUITE_TO_UNNORM[suite_name],
        center_crop=True,
        num_trials_per_task=1,
        seed=args.seed,
        local_log_dir=str(args.output_dir / "upstream_logs"),
        use_wandb=False,
        run_id_note="bounded",
    )
    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = upstream.initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, _, _ = upstream.setup_logging(cfg)
    # LIBERO-Plus prints all ~2.5k task ids during construction; suppress only
    # that known diagnostic and emit a concise, verified summary instead.
    with contextlib.redirect_stdout(io.StringIO()):
        task_suite = benchmark.get_benchmark_dict()[suite_name]()
    print(f"Loaded {suite_name} benchmark with {task_suite.n_tasks} tasks")

    total_episodes = total_successes = 0
    try:
        with results_path.open("a", encoding="utf-8") as result_handle:
            pending_rows = [
                row
                for row in selected
                if (row["suite"], row["task_id"]) not in completed
            ]
            progress = overall_progress(
                pending_rows,
                total=len(selected),
                initial=len(selected) - len(pending_rows),
                desc=f"{suite_name} bounded eval",
                unit="task",
                dynamic_ncols=True,
            )
            for row in progress:
                actual_name = task_suite.get_task(row["task_id"]).name
                if actual_name != row["task_name"]:
                    raise RuntimeError(
                        f"Manifest/runtime mismatch at task {row['task_id']}: {row['task_name']!r} != {actual_name!r}"
                    )
                before_episodes, before_successes = total_episodes, total_successes
                total_episodes, total_successes = upstream.run_task(
                    cfg,
                    task_suite,
                    row["task_id"],
                    model,
                    resize_size,
                    processor,
                    action_head,
                    proprio_projector,
                    noisy_action_projector,
                    total_episodes,
                    total_successes,
                    log_file,
                )
                result = dict(row)
                result.update(
                    {
                        "episodes": total_episodes - before_episodes,
                        "successes": total_successes - before_successes,
                        "seed": args.seed,
                        "completed_at": datetime.now().astimezone().isoformat(),
                    }
                )
                result_handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                result_handle.flush()
                progress.set_postfix(
                    task_id=row["task_id"],
                    last_success=result["successes"],
                    refresh=True,
                )
    finally:
        log_file.close()


def main() -> None:
    args = parse_args()
    validate_paths(args)
    rows, manifest_hash = load_manifest(args.manifest)
    suites = {row["suite"] for row in rows}
    if len(suites) != 1:
        raise SystemExit(
            f"A bounded invocation must contain exactly one suite, got: {sorted(suites)}"
        )
    validate_checkpoint_suite(args.checkpoint, next(iter(suites)))
    validate_checkpoint_read_only_compatibility(args.checkpoint, args.openvla_root)
    selected = rows[: args.max_tasks]

    results_path = args.output_dir / "results.jsonl"
    completed = load_completed(results_path) if args.resume else set()
    pending = [row for row in selected if (row["suite"], row["task_id"]) not in completed]
    print_plan(args, selected, pending)

    if not args.execute:
        print("Dry-run complete. No output files were created and no GPU code was imported.")
        return
    if args.gpu_id is None:
        raise SystemExit("--gpu-id is required with --execute")
    if not pending:
        print("Nothing to do: all selected tasks are already complete.")
        return

    results_path, completed = prepare_output(args, manifest_hash, selected)
    execute(args, selected, results_path, completed)


if __name__ == "__main__":
    main()
