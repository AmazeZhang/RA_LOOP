#!/usr/bin/env python3
"""Run a K=3 same-state/different-instruction OpenVLA feasibility probe.

Safe default: print a validated plan without importing torch, LIBERO, or the
policy.  ``--execute`` requires exactly one launcher-visible GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import site
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_LIBERO_ROOT = Path("/home/imc/code/LIBERO-official")
RIPT_ROOT = Path("/home/imc/code/ript-vla")
OPENVLA_ROOT = Path("/home/imc/code/openvla-oft")
GOAL_CHECKPOINT = Path("/home/imc/models/ra-loop/openvla-oft-goal")
SCALE_HEADER = Path(
    "/home/imc/models/ra-loop/ript-vla/openvla_oft/scale_header/"
    "LIBERO_SPATIAL_scale_header.pth"
)
TASKS = (
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
)
ALL_GOAL_TASKS = (
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "turn_on_the_stove",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Goal-suite task to include; repeat for a broader screen.",
    )
    parser.add_argument(
        "--all-goal-tasks",
        action="store_true",
        help="Screen all ten official LIBERO-Goal instructions.",
    )
    parser.add_argument(
        "--source-task",
        default=TASKS[0],
        help="Task whose official init-state bank supplies the shared states.",
    )
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument(
        "--num-init-states",
        type=int,
        default=1,
        help="Run consecutive init states starting at --init-index.",
    )
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument("--early-action-steps", type=int, default=32)
    parser.add_argument(
        "--reference-result",
        type=Path,
        default=PROJECT_ROOT
        / "ci_grpo/artifacts/p0_openvla_k3_rollout/result.json",
        help="Optional same-init repeat used to calibrate the DTW noise floor.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "ci_grpo/artifacts/p0_openvla_k3_rollout",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.all_goal_tasks and args.tasks:
        raise SystemExit("--all-goal-tasks cannot be combined with --task")
    tasks = ALL_GOAL_TASKS if args.all_goal_tasks else tuple(args.tasks or TASKS)
    if len(tasks) < 2:
        raise SystemExit("at least two distinct tasks are required")
    if len(set(tasks)) != len(tasks):
        raise SystemExit("tasks must be distinct")
    unknown = sorted(set(tasks) - set(ALL_GOAL_TASKS))
    if unknown:
        raise SystemExit(f"tasks are not in the official LIBERO-Goal suite: {unknown}")
    if args.source_task not in ALL_GOAL_TASKS:
        raise SystemExit("--source-task must be in the official LIBERO-Goal suite")
    required = (
        OFFICIAL_LIBERO_ROOT,
        RIPT_ROOT,
        OPENVLA_ROOT,
        GOAL_CHECKPOINT / "config.json",
        GOAL_CHECKPOINT / "action_head--50000_checkpoint.pt",
        GOAL_CHECKPOINT / "proprio_projector--50000_checkpoint.pt",
        GOAL_CHECKPOINT / "dataset_statistics.json",
        SCALE_HEADER,
        PROJECT_ROOT / ".libero_official/config.yaml",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"required paths missing: {missing}")
    if args.init_index < 0 or args.init_index >= 50:
        raise SystemExit("--init-index must be in [0, 49]")
    if args.num_init_states < 1:
        raise SystemExit("--num-init-states must be positive")
    if args.init_index + args.num_init_states > 50:
        raise SystemExit("requested init-state range must stay in [0, 49]")
    if args.max_episode_length < 1 or args.max_episode_length > 300:
        raise SystemExit("--max-episode-length must be in [1, 300]")
    if args.early_action_steps < 1:
        raise SystemExit("--early-action-steps must be positive")
    if args.execute and (args.gpu_id is None or args.gpu_id < 0):
        raise SystemExit("--execute requires --gpu-id")
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")

    statistics = json.loads(
        (GOAL_CHECKPOINT / "dataset_statistics.json").read_text(encoding="utf-8")
    )
    if "libero_goal_no_noops" not in statistics:
        raise SystemExit("goal checkpoint lacks libero_goal_no_noops statistics")

    return {
        "probe": "CI-GRPO P0 OpenVLA K=3 rollout",
        "tasks": list(tasks),
        "shared_source_task": args.source_task,
        "init_index": args.init_index,
        "init_indices": list(
            range(args.init_index, args.init_index + args.num_init_states)
        ),
        "num_init_states": args.num_init_states,
        "max_episode_length": args.max_episode_length,
        "early_action_steps": args.early_action_steps,
        "reference_result": (
            str(args.reference_result) if args.reference_result.exists() else None
        ),
        "checkpoint": str(GOAL_CHECKPOINT),
        "checkpoint_config_sha256": sha256(GOAL_CHECKPOINT / "config.json"),
        "checkpoint_statistics_sha256": sha256(
            GOAL_CHECKPOINT / "dataset_statistics.json"
        ),
        "one_live_environment": True,
        "deterministic_action_mean": True,
        "gpu_id": args.gpu_id,
        "output_dir": str(args.output_dir),
        "execute": args.execute,
    }


def dtw_distance(first: Any, second: Any) -> dict[str, float | int]:
    import numpy as np

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"DTW expects [time, action_dim], got {a.shape} and {b.shape}")
    costs = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float64)
    lengths = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int64)
    costs[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            choices = (
                (costs[i - 1, j], lengths[i - 1, j]),
                (costs[i, j - 1], lengths[i, j - 1]),
                (costs[i - 1, j - 1], lengths[i - 1, j - 1]),
            )
            previous_cost, previous_length = min(choices, key=lambda item: item[0])
            costs[i, j] = previous_cost + float(np.linalg.norm(a[i - 1] - b[j - 1]))
            lengths[i, j] = previous_length + 1
    path_length = int(lengths[-1, -1])
    return {
        "total": float(costs[-1, -1]),
        "path_length": path_length,
        "mean_per_path_step": float(costs[-1, -1] / max(path_length, 1)),
    }


def diagnostic_metrics(
    rows: list[dict[str, Any]],
    *,
    tasks: tuple[str, ...] = TASKS,
    source_task: str = TASKS[0],
) -> dict[str, float | int | bool]:
    """Summarize hit/miss and instruction-swap behavior from cross-scored rows."""

    if not rows:
        raise ValueError("at least one rollout row is required")
    hits = [
        bool(row["terminal_goal_truth"][row["instruction_task"]]) for row in rows
    ]
    misses = [
        bool(value)
        for row in rows
        for task, value in row["terminal_goal_truth"].items()
        if task != row["instruction_task"]
    ]
    redirected = [
        bool(row["terminal_goal_truth"][row["instruction_task"]])
        for row in rows
        if row["instruction_task"] != source_task
    ]
    hit_rate = sum(hits) / len(hits)
    miss_rate = sum(misses) / len(misses)
    redirection_rate = sum(redirected) / len(redirected)
    lsg = hit_rate - miss_rate
    return {
        "n_rollouts": len(rows),
        "n_off_diagonal_checks": len(misses),
        "lsg_hit": hit_rate,
        "lsg_miss": miss_rate,
        "lsg": lsg,
        "instruction_swap_redirection_rate": redirection_rate,
        "baseline_language_deafness_premise_supported": bool(
            lsg <= 0.15 and redirection_rate <= 0.35
        ),
    }


def calibrate_dtw_threshold(
    between_instruction: list[float],
    within_instruction_repeat: list[float],
    *,
    effect_floor: float = 0.01,
    noise_multiplier: float = 5.0,
) -> dict[str, Any]:
    """Pre-registered R5b threshold from repeat noise plus an effect-size floor."""

    if not between_instruction:
        raise ValueError("between-instruction DTW values are required")
    noise_upper = max(within_instruction_repeat, default=0.0)
    threshold = max(effect_floor, noise_multiplier * noise_upper)
    return {
        "effect_floor": effect_floor,
        "noise_multiplier": noise_multiplier,
        "within_instruction_repeat_max": noise_upper,
        "threshold": threshold,
        "between_instruction_min": min(between_instruction),
        "between_instruction_median": float(
            __import__("numpy").median(between_instruction)
        ),
        "between_instruction_max": max(between_instruction),
        "all_between_instruction_pairs_pass": all(
            value >= threshold for value in between_instruction
        ),
    }


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBERO_CONFIG_PATH"] = str(PROJECT_ROOT / ".libero_official")
    os.environ["NUMBA_CACHE_DIR"] = "/tmp/ci_grpo_numba_cache"
    os.environ["MPLCONFIGDIR"] = "/tmp/ci_grpo_mpl_cache"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    user_site = str(site.getusersitepackages())
    sys.path[:] = [entry for entry in sys.path if str(entry) != user_site]
    sys.path[:0] = [
        str(PROJECT_ROOT),
        str(OFFICIAL_LIBERO_ROOT),
        str(RIPT_ROOT),
        str(OPENVLA_ROOT),
    ]

    import numpy as np
    import torch
    import libero.libero.envs.bddl_utils as BDDLUtils
    from ra_loop.ript_compat import InProcessOpenVLAOFTLiberoRunner
    from ript.algos.rl_optimizers.openvla_oft_interface import OpenVLA_OFT_Policy

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires exactly one launcher-visible GPU")

    tasks = tuple(plan["tasks"])
    torch.cuda.reset_peak_memory_stats()
    policy = OpenVLA_OFT_Policy(
        pretrained_checkpoint=str(GOAL_CHECKPOINT),
        header_checkpoint=str(SCALE_HEADER),
        task_suite_name="LIBERO_GOAL",
        lora_rank=32,
        lora_dropout=0.0,
        lora_adaptor_ckpt=None,
        device_id=0,
        seed=7,
        fix_scale_head=True,
        log_scale_clip=[-2.0, 0.5],
    )
    runner = InProcessOpenVLAOFTLiberoRunner(
        benchmark_name="LIBERO_GOAL",
        rollouts_per_env=1,
        num_parallel_envs=1,
        max_episode_length=args.max_episode_length,
        task_names_to_use=list(tasks),
        use_laplace_sampling=False,
        scale_factor=1.0,
    )

    task_ids = {task: runner.env_names.index(task) for task in tasks}
    parsed_goals = {}
    for task in tasks:
        task_info = runner.benchmark.get_task(task_ids[task])
        bddl_path = (
            OFFICIAL_LIBERO_ROOT
            / "libero/libero/bddl_files"
            / task_info.problem_folder
            / task_info.bddl_file
        )
        parsed_goals[task] = BDDLUtils.robosuite_parse_problem(str(bddl_path))[
            "goal_state"
        ]

    source_id = runner.env_names.index(plan["shared_source_task"])
    all_states = np.asarray(runner.benchmark.get_task_init_states(source_id))
    created_env = runner.create_env(plan["shared_source_task"])
    vector_env, _, _ = created_env
    control_env = vector_env.workers[0].env

    rows = []
    early_actions: dict[tuple[int, str], Any] = {}
    try:
        for init_index in plan["init_indices"]:
            shared_state = all_states[init_index : init_index + 1].copy()
            for task in tasks:
                control_env.env.parsed_problem["goal_state"] = parsed_goals[task]
                rollout = runner.run_policy_in_env(
                    task,
                    policy,
                    all_init_states=shared_state,
                    render=False,
                    created_env=created_env,
                    random_init=False,
                )
                success, total_reward, episode = next(rollout)
                normalized_chunks = [
                    np.asarray(chunk)
                    for chunk in episode["actions_normalized"]
                    if chunk is not None
                ]
                predicted_actions = (
                    np.concatenate(normalized_chunks, axis=0)
                    if normalized_chunks
                    else np.empty((0, 7), dtype=np.float32)
                )
                early_actions[(init_index, task)] = predicted_actions[
                    : args.early_action_steps
                ]
                terminal_state = np.asarray(control_env.get_sim_state())

                goal_truth = {}
                for scored_task in tasks:
                    previous = control_env.env.parsed_problem["goal_state"]
                    control_env.env.parsed_problem["goal_state"] = parsed_goals[
                        scored_task
                    ]
                    try:
                        goal_truth[scored_task] = bool(control_env.check_success())
                    finally:
                        control_env.env.parsed_problem["goal_state"] = previous

                np.save(args.output_dir.with_suffix(".pending.npy"), terminal_state)
                rows.append(
                    {
                        "init_index": init_index,
                        "instruction_task": task,
                        "language": runner.benchmark.get_task(
                            task_ids[task]
                        ).language,
                        "success_for_instruction_goal": bool(success),
                        "total_reward": float(total_reward),
                        "inference_chunks": len(normalized_chunks),
                        "predicted_action_steps": int(len(predicted_actions)),
                        "early_action_steps_used": int(
                            len(early_actions[(init_index, task)])
                        ),
                        "terminal_state_sha256": hashlib.sha256(
                            np.ascontiguousarray(terminal_state).tobytes()
                        ).hexdigest(),
                        "terminal_goal_truth": goal_truth,
                        "terminal_state": terminal_state.tolist(),
                        "early_normalized_actions": early_actions[
                            (init_index, task)
                        ].tolist(),
                    }
                )
    finally:
        vector_env.close()
        pending = args.output_dir.with_suffix(".pending.npy")
        if pending.exists():
            pending.unlink()

    pairwise_dtw = {}
    between_values = []
    for init_index in plan["init_indices"]:
        per_init = {}
        for left_index, left in enumerate(tasks):
            for right in tasks[left_index + 1 :]:
                distance = dtw_distance(
                    early_actions[(init_index, left)],
                    early_actions[(init_index, right)],
                )
                per_init[f"{left}::{right}"] = distance
                between_values.append(distance["mean_per_path_step"])
        pairwise_dtw[str(init_index)] = per_init

    within_repeat = []
    if plan["reference_result"] is not None and args.init_index == 0:
        reference = json.loads(args.reference_result.read_text(encoding="utf-8"))
        reference_rows = {
            row["instruction_task"]: row
            for row in reference["rows"]
            if row.get("init_index", reference.get("init_index")) == 0
        }
        for task in tasks:
            if task not in reference_rows:
                continue
            distance = dtw_distance(
                early_actions[(0, task)],
                reference_rows[task]["early_normalized_actions"],
            )
            within_repeat.append(distance["mean_per_path_step"])

    metrics = diagnostic_metrics(
        rows,
        tasks=tasks,
        source_task=plan["shared_source_task"],
    )
    dtw_calibration = calibrate_dtw_threshold(between_values, within_repeat)

    exclusivity_pass = all(
        row["success_for_instruction_goal"]
        and row["terminal_goal_truth"][row["instruction_task"]]
        and sum(row["terminal_goal_truth"].values()) == 1
        for row in rows
    )
    result = {
        **plan,
        "completed_at": datetime.now().astimezone().isoformat(),
        "rows": rows,
        "pairwise_early_action_dtw": pairwise_dtw,
        "within_instruction_repeat_dtw": within_repeat,
        "dtw_calibration": dtw_calibration,
        "language_sensitivity": metrics,
        "checks": {
            "all_instruction_rollouts_succeed": all(
                row["success_for_instruction_goal"] for row in rows
            ),
            "terminal_goals_pairwise_exclusive_empirically": exclusivity_pass,
            "all_pairs_pass_calibrated_early_dtw": dtw_calibration[
                "all_between_instruction_pairs_pass"
            ],
        },
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    result["p0_t0_4_empirical_pass"] = all(result["checks"].values())

    args.output_dir.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_dir / "result.json.incomplete"
    incomplete.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(args.output_dir / "result.json")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    plan = validate(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    execute(args, plan)


if __name__ == "__main__":
    main()
