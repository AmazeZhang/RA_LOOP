#!/usr/bin/env python3
"""Compare stale-queue and flush/replan handling of mid-chunk language changes."""

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


CHUNK_SIZE = 8
OFFSETS = (1, 4, 7)
TASK_PAIRS = (
    (TASKS[0], TASKS[1]),
    (TASKS[1], TASKS[2]),
    (TASKS[2], TASKS[0]),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "ci_grpo/artifacts/p2_semantic_chunk_interrupt_openvla",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def penultimate_chunk_index(n_generated_chunks: int) -> int:
    if n_generated_chunks < 2:
        raise ValueError("at least two generated chunks are required")
    return n_generated_chunks - 2


def summarize_interrupt_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["origin_task"], row["revised_task"], row["offset"])
        grouped.setdefault(key, {})[row["method"]] = row

    valid_pairs = []
    for key, methods in grouped.items():
        if set(methods) != {"stale", "flush"}:
            continue
        stale, flush = methods["stale"], methods["flush"]
        if not (
            stale["baseline_success"]
            and flush["baseline_success"]
            and stale["checkpoint_all_goals_false"]
            and flush["checkpoint_all_goals_false"]
            and stale["exact_restore"]
            and flush["exact_restore"]
            and stale["checkpoint_state_sha256"]
            == flush["checkpoint_state_sha256"]
            and sum(stale["terminal_goal_truth"].values()) <= 1
            and sum(flush["terminal_goal_truth"].values()) <= 1
        ):
            continue
        valid_pairs.append((key, stale, flush))

    def rate(values: list[bool]) -> float | None:
        return sum(values) / len(values) if values else None

    stale_success = rate(
        [
            bool(stale["terminal_goal_truth"][stale["revised_task"]])
            for _, stale, _ in valid_pairs
        ]
    )
    flush_success = rate(
        [
            bool(flush["terminal_goal_truth"][flush["revised_task"]])
            for _, _, flush in valid_pairs
        ]
    )
    stale_old_goal = rate(
        [
            bool(stale["terminal_goal_truth"][stale["origin_task"]])
            for _, stale, _ in valid_pairs
        ]
    )
    flush_old_goal = rate(
        [
            bool(flush["terminal_goal_truth"][flush["origin_task"]])
            for _, _, flush in valid_pairs
        ]
    )
    improvement = (
        flush_success - stale_success
        if flush_success is not None and stale_success is not None
        else None
    )
    candidate = bool(
        len(valid_pairs) >= 6
        and flush_success is not None
        and flush_success >= 2.0 / 3.0
        and improvement is not None
        and improvement >= 2.0 / 9.0
    )
    return {
        "n_rows": len(rows),
        "n_complete_pairs": len(grouped),
        "n_valid_pairs": len(valid_pairs),
        "stale_revised_goal_success_rate": stale_success,
        "flush_revised_goal_success_rate": flush_success,
        "flush_success_improvement": improvement,
        "stale_original_goal_rate": stale_old_goal,
        "flush_original_goal_rate": flush_old_goal,
        "mean_stale_response_latency_actions": (
            sum(stale["response_latency_actions"] for _, stale, _ in valid_pairs)
            / len(valid_pairs)
            if valid_pairs
            else None
        ),
        "mean_stale_action_jerk_l2": (
            sum(stale["switch_action_jerk_l2"] for _, stale, _ in valid_pairs)
            / len(valid_pairs)
            if valid_pairs
            else None
        ),
        "mean_flush_action_jerk_l2": (
            sum(flush["switch_action_jerk_l2"] for _, _, flush in valid_pairs)
            / len(valid_pairs)
            if valid_pairs
            else None
        ),
        "semantic_interrupt_candidate": candidate,
        "decision": "provisional_candidate" if candidate else "no_go",
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
    if args.max_episode_length < 16 or args.max_episode_length > 300:
        raise SystemExit("--max-episode-length must be in [16, 300]")
    if args.execute and (args.gpu_id is None or args.gpu_id <= 0):
        raise SystemExit("--execute requires a nonzero --gpu-id")
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")
    return {
        "probe": "Semantic Action-Chunk Interrupt P0",
        "task_pairs": [list(pair) for pair in TASK_PAIRS],
        "offsets": list(OFFSETS),
        "chunk_size": CHUNK_SIZE,
        "intervention_chunk_rule": "penultimate generated baseline chunk",
        "methods": ["stale", "flush"],
        "maximum_rollouts": 21,
        "init_index": args.init_index,
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
    from ript.env_runner.openvla_oft_libero_runner import (
        get_libero_dummy_action,
        get_vla_action_batch,
        prepare_observation,
        process_action,
    )

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
    descriptions = {
        task: runner.benchmark.get_task(task_ids[task]).language for task in TASKS
    }
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
        args.init_index
    ].copy()
    vector_env, _, _ = runner.create_env(TASKS[0])
    env = vector_env.workers[0].env

    cfg = policy.cfg
    model = policy.model.module if hasattr(policy.model, "module") else policy.model
    action_head = (
        policy.action_head.module
        if hasattr(policy.action_head, "module")
        else policy.action_head
    )
    scale_head = (
        policy.scale_head.module
        if hasattr(policy.scale_head, "module")
        else policy.scale_head
    )
    proprio_projector = (
        policy.proprio_projector.module
        if hasattr(policy.proprio_projector, "module")
        else policy.proprio_projector
    )

    def infer_chunk(obs: Any, task: str) -> list[Any]:
        prepared, _ = prepare_observation(obs, 224)
        with torch.inference_mode():
            actions, _, _, _, _ = get_vla_action_batch(
                cfg,
                vla=model,
                obs_batch=[prepared],
                task_label=descriptions[task],
                processor=policy.processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=None,
                scale_head=scale_head,
                use_film=cfg.use_film,
                use_laplace_sampling=False,
                scale_factor=1.0,
            )
        return [np.asarray(action).copy() for action in actions[0]]

    def set_goal(task: str) -> None:
        env.env.parsed_problem["goal_state"] = parsed_goals[task]

    def goal_truth() -> dict[str, bool]:
        truth = {}
        previous = env.env.parsed_problem["goal_state"]
        try:
            for task in TASKS:
                env.env.parsed_problem["goal_state"] = parsed_goals[task]
                truth[task] = bool(env.check_success())
        finally:
            env.env.parsed_problem["goal_state"] = previous
        return truth

    def reset_initial(task: str) -> Any:
        set_goal(task)
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
        return obs

    def bowl_position() -> Any:
        body_id = env.env.obj_body_id["akita_black_bowl_1"]
        return env.env.sim.data.body_xpos[body_id].copy()

    def run_baseline(task: str) -> dict[str, Any]:
        obs = reset_initial(task)
        raw_chunks: list[list[Any]] = []
        processed_actions = []
        states_after = []
        steps = 0
        queue: list[Any] = []
        while steps < args.max_episode_length:
            if not queue:
                queue = infer_chunk(obs, task)
                raw_chunks.append([action.copy() for action in queue])
            raw_action = queue.pop(0)
            action = np.asarray(process_action(raw_action, "openvla"))
            obs, _, _, _ = env.step(action)
            processed_actions.append(action.copy())
            states_after.append(np.asarray(env.get_sim_state()).copy())
            steps += 1
            if env.check_success():
                break
        return {
            "task": task,
            "success": bool(env.check_success()),
            "steps": steps,
            "raw_chunks": raw_chunks,
            "processed_actions": processed_actions,
            "states_after": states_after,
        }

    baselines = []
    switch_cases = []
    rows: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_dir / "result.json.incomplete"

    def write_progress() -> None:
        progress = {
            **plan,
            "baselines": baselines,
            "rows": rows,
            "progress_only": True,
        }
        incomplete.write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    try:
        for origin_task, revised_task in TASK_PAIRS:
            baseline = run_baseline(origin_task)
            if not baseline["success"]:
                raise RuntimeError(f"baseline failed for {origin_task}")
            chunk_index = penultimate_chunk_index(len(baseline["raw_chunks"]))
            chunk_start = chunk_index * CHUNK_SIZE
            baseline_row = {
                "task": origin_task,
                "success": baseline["success"],
                "steps": baseline["steps"],
                "generated_chunks": len(baseline["raw_chunks"]),
                "intervention_chunk_index": chunk_index,
                "intervention_chunk_start_step": chunk_start,
            }
            baselines.append(baseline_row)
            for offset in OFFSETS:
                state_index = chunk_start + offset - 1
                state = baseline["states_after"][state_index]
                env.set_init_state(state)
                switch_cases.append(
                    {
                        "origin_task": origin_task,
                        "revised_task": revised_task,
                        "offset": offset,
                        "switch_control_step": state_index + 1,
                        "state": state,
                        "state_sha256": hashlib.sha256(
                            np.ascontiguousarray(state).tobytes()
                        ).hexdigest(),
                        "checkpoint_goal_truth": goal_truth(),
                        "bowl_position": bowl_position(),
                        "remaining_raw_actions": [
                            action.copy()
                            for action in baseline["raw_chunks"][chunk_index][offset:]
                        ],
                        "last_prefix_action": baseline["processed_actions"][
                            state_index
                        ].copy(),
                        "baseline_success": baseline["success"],
                    }
                )
            write_progress()

        for case in switch_cases:
            for method in ("stale", "flush"):
                set_goal(case["revised_task"])
                env.reset()
                obs = env.set_init_state(case["state"])
                restored = np.asarray(env.get_sim_state()).copy()
                queue = (
                    [action.copy() for action in case["remaining_raw_actions"]]
                    if method == "stale"
                    else []
                )
                latency = len(queue)
                comparison_horizon = len(case["remaining_raw_actions"])
                first_action = None
                bowl_after_horizon = None
                steps = 0
                max_steps = max(
                    1, args.max_episode_length - case["switch_control_step"]
                )
                while steps < max_steps:
                    if not queue:
                        queue = infer_chunk(obs, case["revised_task"])
                    raw_action = queue.pop(0)
                    action = np.asarray(process_action(raw_action, "openvla"))
                    if first_action is None:
                        first_action = action.copy()
                    obs, _, _, _ = env.step(action)
                    steps += 1
                    if steps == comparison_horizon:
                        bowl_after_horizon = bowl_position()
                    if env.check_success():
                        break
                if bowl_after_horizon is None:
                    bowl_after_horizon = bowl_position()
                terminal_truth = goal_truth()
                rows.append(
                    {
                        "origin_task": case["origin_task"],
                        "revised_task": case["revised_task"],
                        "offset": case["offset"],
                        "method": method,
                        "baseline_success": case["baseline_success"],
                        "switch_control_step": case["switch_control_step"],
                        "checkpoint_state_sha256": case["state_sha256"],
                        "checkpoint_goal_truth": case["checkpoint_goal_truth"],
                        "checkpoint_all_goals_false": not any(
                            case["checkpoint_goal_truth"].values()
                        ),
                        "exact_restore": bool(
                            np.array_equal(case["state"], restored)
                        ),
                        "restore_max_abs_delta": float(
                            np.max(np.abs(case["state"] - restored))
                        ),
                        "stale_tail_actions": len(
                            case["remaining_raw_actions"]
                        ),
                        "response_latency_actions": latency,
                        "switch_action_jerk_l2": float(
                            np.linalg.norm(
                                first_action - case["last_prefix_action"]
                            )
                        ),
                        "bowl_displacement_during_comparison_horizon": (
                            bowl_after_horizon - case["bowl_position"]
                        ).tolist(),
                        "steps_after_switch": steps,
                        "success_for_revised_goal": bool(
                            terminal_truth[case["revised_task"]]
                        ),
                        "terminal_goal_truth": terminal_truth,
                    }
                )
                write_progress()
    finally:
        vector_env.close()

    metrics = summarize_interrupt_rows(rows)
    result = {
        **plan,
        "completed_at": datetime.now().astimezone().isoformat(),
        "baselines": baselines,
        "rows": rows,
        "metrics": metrics,
        "continue_method_direction": metrics["semantic_interrupt_candidate"],
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
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
