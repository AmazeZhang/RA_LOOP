#!/usr/bin/env python3
"""Bounded mid-trajectory instruction-revision screen for OpenVLA Goal."""

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

_LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PROJECT_ROOT))

from ci_grpo.p0_openvla_k3_rollout import (
    GOAL_CHECKPOINT,
    OFFICIAL_LIBERO_ROOT,
    OPENVLA_ROOT,
    PROJECT_ROOT,
    RIPT_ROOT,
    SCALE_HEADER,
    TASKS,
    sha256,
)


CHECKPOINT_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "ci_grpo/artifacts/p1_temporal_regrounding_openvla",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def checkpoint_indices(
    trajectory_steps: int,
    fractions: tuple[float, ...] = CHECKPOINT_FRACTIONS,
) -> list[int]:
    """Return unique zero-based post-action checkpoint indices."""

    if trajectory_steps < 3:
        raise ValueError("trajectory must contain at least three control steps")
    if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("checkpoint fractions must lie strictly inside (0, 1)")
    indices = [
        min(trajectory_steps - 2, max(0, round(value * trajectory_steps) - 1))
        for value in fractions
    ]
    if len(set(indices)) != len(indices):
        raise ValueError("trajectory is too short for distinct checkpoints")
    return indices


def summarize_temporal_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    controls = [row for row in rows if row["is_control"]]
    switches = [row for row in rows if not row["is_control"]]
    valid_controls = [
        row
        for row in controls
        if row["checkpoint_all_goals_false"]
        and row["exact_restore"]
        and row.get("same_policy_start", True)
        and row["success_for_instruction_goal"]
    ]
    valid_cells = {
        (row["origin_task"], row["checkpoint_label"]) for row in valid_controls
    }
    valid_switches = [
        row
        for row in switches
        if (row["origin_task"], row["checkpoint_label"]) in valid_cells
        and row["checkpoint_all_goals_false"]
        and row["exact_restore"]
        and row.get("same_policy_start", True)
    ]

    def rate(values: list[bool]) -> float | None:
        return sum(values) / len(values) if values else None

    switch_successes = [
        bool(row["terminal_goal_truth"][row["instruction_task"]])
        for row in valid_switches
    ]
    inertia = [
        bool(row["terminal_goal_truth"][row["origin_task"]])
        for row in valid_switches
    ]
    inertia_cells = {
        (row["origin_task"], row["checkpoint_label"])
        for row in valid_switches
        if row["terminal_goal_truth"][row["origin_task"]]
    }
    control_rate = rate(
        [bool(row["success_for_instruction_goal"]) for row in controls]
    )
    switch_rate = rate(switch_successes)
    inertia_rate = rate(inertia)
    candidate = bool(
        len(valid_switches) >= 4
        and control_rate == 1.0
        and switch_rate is not None
        and switch_rate <= 0.5
        and inertia_rate is not None
        and inertia_rate >= 0.5
        and len(inertia_cells) >= 2
    )
    if candidate:
        decision = "provisional_candidate"
    elif controls and control_rate != 1.0:
        decision = "inconclusive_control_failure"
    else:
        decision = "no_go"
    return {
        "n_controls": len(controls),
        "n_valid_controls": len(valid_controls),
        "matched_control_success_rate": control_rate,
        "n_switches": len(switches),
        "n_valid_switches": len(valid_switches),
        "switch_success_rate": switch_rate,
        "original_goal_inertia_rate": inertia_rate,
        "switch_lsg": (
            switch_rate - inertia_rate
            if switch_rate is not None and inertia_rate is not None
            else None
        ),
        "n_distinct_inertia_checkpoints": len(inertia_cells),
        "temporal_inertia_candidate": candidate,
        "decision": decision,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    required = (
        OFFICIAL_LIBERO_ROOT,
        RIPT_ROOT,
        OPENVLA_ROOT,
        GOAL_CHECKPOINT / "config.json",
        GOAL_CHECKPOINT / "action_head--50000_checkpoint.pt",
        GOAL_CHECKPOINT / "proprio_projector--50000_checkpoint.pt",
        SCALE_HEADER,
        PROJECT_ROOT / ".libero_official/config.yaml",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"required paths missing: {missing}")
    if args.init_index < 0 or args.init_index >= 50:
        raise SystemExit("--init-index must be in [0, 49]")
    if args.max_episode_length < 1 or args.max_episode_length > 300:
        raise SystemExit("--max-episode-length must be in [1, 300]")
    if args.execute and (args.gpu_id is None or args.gpu_id <= 0):
        raise SystemExit("--execute requires a nonzero --gpu-id")
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")
    return {
        "probe": "Temporal Counterfactual Regrounding P0",
        "tasks": list(TASKS),
        "init_index": args.init_index,
        "checkpoint_fractions": list(CHECKPOINT_FRACTIONS),
        "maximum_rollouts": 21,
        "max_episode_length": args.max_episode_length,
        "checkpoint": str(GOAL_CHECKPOINT),
        "checkpoint_config_sha256": sha256(GOAL_CHECKPOINT / "config.json"),
        "gpu_id": args.gpu_id,
        "output_dir": str(args.output_dir),
        "execute": args.execute,
        "training_authorized": False,
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
        raise RuntimeError("screen requires exactly one launcher-visible GPU")

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
        task_names_to_use=list(TASKS),
        use_laplace_sampling=False,
        scale_factor=1.0,
    )
    task_ids = {task: runner.env_names.index(task) for task in TASKS}
    parsed_goals: dict[str, Any] = {}
    for task in TASKS:
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

    source_id = runner.env_names.index(TASKS[0])
    init_state = np.asarray(runner.benchmark.get_task_init_states(source_id))[
        args.init_index : args.init_index + 1
    ].copy()
    created_env = runner.create_env(TASKS[0])
    vector_env, _, _ = created_env
    control_env = vector_env.workers[0].env
    settle_steps = int(runner.num_steps_wait)

    def goal_truth() -> dict[str, bool]:
        truth = {}
        previous = control_env.env.parsed_problem["goal_state"]
        try:
            for scored_task in TASKS:
                control_env.env.parsed_problem["goal_state"] = parsed_goals[
                    scored_task
                ]
                truth[scored_task] = bool(control_env.check_success())
        finally:
            control_env.env.parsed_problem["goal_state"] = previous
        return truth

    baselines = []
    checkpoints: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    try:
        for origin_task in TASKS:
            captured_states = []
            original_step = vector_env.step

            def recording_step(actions: Any) -> Any:
                output = original_step(actions)
                captured_states.append(
                    np.asarray(control_env.get_sim_state()).copy()
                )
                return output

            vector_env.step = recording_step
            control_env.env.parsed_problem["goal_state"] = parsed_goals[origin_task]
            try:
                success, _, _ = next(
                    runner.run_policy_in_env(
                        origin_task,
                        policy,
                        all_init_states=init_state,
                        render=False,
                        created_env=created_env,
                        random_init=False,
                    )
                )
            finally:
                vector_env.step = original_step
            action_states = captured_states[settle_steps:]
            if not success:
                raise RuntimeError(f"baseline failed for {origin_task}")
            indices = checkpoint_indices(len(action_states))
            baselines.append(
                {
                    "origin_task": origin_task,
                    "success": bool(success),
                    "control_steps": len(action_states),
                    "checkpoint_indices": indices,
                }
            )
            for fraction, index in zip(CHECKPOINT_FRACTIONS, indices):
                state = action_states[index]
                control_env.set_init_state(state)
                restored = np.asarray(control_env.get_sim_state())
                bowl_id = control_env.env.obj_body_id["akita_black_bowl_1"]
                checkpoints.append(
                    {
                        "origin_task": origin_task,
                        "checkpoint_label": f"{fraction:.3f}",
                        "checkpoint_fraction": fraction,
                        "control_step": index + 1,
                        "state": state,
                        "state_sha256": hashlib.sha256(
                            np.ascontiguousarray(state).tobytes()
                        ).hexdigest(),
                        "exact_restore": bool(np.array_equal(state, restored)),
                        "restore_max_abs_delta": float(
                            np.max(np.abs(state - restored))
                        ),
                        "goal_truth": goal_truth(),
                        "bowl_position": control_env.env.sim.data.body_xpos[
                            bowl_id
                        ]
                        .copy()
                        .tolist(),
                    }
                )

        # The pinned upstream runner obtains its initial observation inside the
        # wait loop. One deterministic dummy action is therefore required.
        # We capture the resulting policy-start state and verify that it is
        # byte-identical across the three instruction variants.
        runner.num_steps_wait = 1
        for checkpoint in checkpoints:
            state = checkpoint["state"]
            origin_task = checkpoint["origin_task"]
            for instruction_task in (origin_task,) + tuple(
                task for task in TASKS if task != origin_task
            ):
                control_env.env.parsed_problem["goal_state"] = parsed_goals[
                    instruction_task
                ]
                policy_start_states = []
                original_step = vector_env.step

                def capture_policy_start(actions: Any) -> Any:
                    output = original_step(actions)
                    if not policy_start_states:
                        policy_start_states.append(
                            np.asarray(control_env.get_sim_state()).copy()
                        )
                    return output

                vector_env.step = capture_policy_start
                try:
                    success, total_reward, _ = next(
                        runner.run_policy_in_env(
                            instruction_task,
                            policy,
                            all_init_states=state[None, :],
                            render=False,
                            created_env=created_env,
                            random_init=False,
                        )
                    )
                finally:
                    vector_env.step = original_step
                policy_start_state = policy_start_states[0]
                terminal_truth = goal_truth()
                rows.append(
                    {
                        "origin_task": origin_task,
                        "instruction_task": instruction_task,
                        "is_control": instruction_task == origin_task,
                        "checkpoint_label": checkpoint["checkpoint_label"],
                        "checkpoint_fraction": checkpoint["checkpoint_fraction"],
                        "control_step": checkpoint["control_step"],
                        "checkpoint_state_sha256": checkpoint["state_sha256"],
                        "exact_restore": checkpoint["exact_restore"],
                        "restore_max_abs_delta": checkpoint[
                            "restore_max_abs_delta"
                        ],
                        "checkpoint_goal_truth": checkpoint["goal_truth"],
                        "checkpoint_all_goals_false": not any(
                            checkpoint["goal_truth"].values()
                        ),
                        "checkpoint_bowl_position": checkpoint["bowl_position"],
                        "policy_start_state_sha256": hashlib.sha256(
                            np.ascontiguousarray(policy_start_state).tobytes()
                        ).hexdigest(),
                        "_policy_start_state": policy_start_state,
                        "success_for_instruction_goal": bool(success),
                        "total_reward": float(total_reward),
                        "terminal_goal_truth": terminal_truth,
                    }
                )
    finally:
        vector_env.close()

    policy_start_hashes: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        key = (row["origin_task"], row["checkpoint_label"])
        policy_start_hashes.setdefault(key, set()).add(
            row["policy_start_state_sha256"]
        )
    for row in rows:
        key = (row["origin_task"], row["checkpoint_label"])
        row["same_policy_start"] = len(policy_start_hashes[key]) == 1
        row.pop("_policy_start_state")

    metrics = summarize_temporal_rows(rows)
    result = {
        **plan,
        "completed_at": datetime.now().astimezone().isoformat(),
        "settle_steps_excluded_from_checkpoint_timing": settle_steps,
        "baselines": baselines,
        "checkpoints": [
            {key: value for key, value in row.items() if key != "state"}
            for row in checkpoints
        ],
        "rows": rows,
        "metrics": metrics,
        "continue_to_reachability_audit": metrics["temporal_inertia_candidate"],
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
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
