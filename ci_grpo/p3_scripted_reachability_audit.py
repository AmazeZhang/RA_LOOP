#!/usr/bin/env python3
"""Audit late switch-state reachability with a fixed Cartesian transport script."""

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
from ci_grpo.p2_semantic_chunk_interrupt_screen import (
    CHUNK_SIZE,
    OFFSETS,
    TASK_PAIRS,
    penultimate_chunk_index,
)


POSITION_SCALE = 0.05
WAYPOINT_TOLERANCE = 0.015
MAX_WAYPOINT_STEPS = 80
SAFE_CLEARANCE = 0.15
RELEASE_STEPS = 15
SETTLE_STEPS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "ci_grpo/artifacts/p3_scripted_reachability_audit",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def position_action(
    current: Any,
    target: Any,
    *,
    gripper: float,
    scale: float = POSITION_SCALE,
) -> Any:
    import numpy as np

    action = np.zeros(7, dtype=np.float64)
    action[:3] = np.clip(
        (np.asarray(target) - np.asarray(current)) / scale, -1.0, 1.0
    )
    action[6] = gripper
    return action


def transport_waypoints(
    current_bowl: Any,
    target_bowl: Any,
    eef_minus_bowl: Any,
    *,
    clearance: float = SAFE_CLEARANCE,
) -> list[Any]:
    import numpy as np

    current_bowl = np.asarray(current_bowl, dtype=np.float64)
    target_bowl = np.asarray(target_bowl, dtype=np.float64)
    offset = np.asarray(eef_minus_bowl, dtype=np.float64)
    safe_z = max(current_bowl[2], target_bowl[2]) + clearance
    lifted_bowl = current_bowl.copy()
    lifted_bowl[2] = safe_z
    above_target = target_bowl.copy()
    above_target[2] = safe_z
    return [
        lifted_bowl + offset,
        above_target + offset,
        target_bowl + offset,
    ]


def summarize_oracle_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    successes = [row for row in eligible if row["revised_goal_success"]]
    pairs_with_success = {
        (row["origin_task"], row["revised_task"]) for row in successes
    }
    required_pairs = {
        (row["origin_task"], row["revised_task"]) for row in rows
    }
    passed = bool(
        len(eligible) >= 6
        and len(successes) >= 6
        and pairs_with_success == required_pairs
    )
    return {
        "n_rows": len(rows),
        "n_eligible": len(eligible),
        "n_revised_goal_success": len(successes),
        "eligible_success_rate": (
            len(successes) / len(eligible) if eligible else None
        ),
        "n_pairs_with_success": len(pairs_with_success),
        "all_directed_pairs_have_success": pairs_with_success == required_pairs,
        "reachability_pass": passed,
        "decision": "reachable" if passed else "inconclusive_or_unreachable",
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    terminal_result = (
        PROJECT_ROOT / "ci_grpo/artifacts/p0_openvla_k3_rollout/result.json"
    )
    required = (
        OFFICIAL_LIBERO_ROOT,
        RIPT_ROOT,
        OPENVLA_ROOT,
        GOAL_CHECKPOINT / "config.json",
        GOAL_CHECKPOINT / "action_head--50000_checkpoint.pt",
        GOAL_CHECKPOINT / "proprio_projector--50000_checkpoint.pt",
        SCALE_HEADER,
        terminal_result,
        PROJECT_ROOT / ".libero_official/config.yaml",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"required paths missing: {missing}")
    if args.init_index != 0:
        raise SystemExit("audit is pre-registered only for --init-index 0")
    if args.max_episode_length != 300:
        raise SystemExit("audit is pre-registered only for 300 steps")
    if args.execute and (args.gpu_id is None or args.gpu_id <= 0):
        raise SystemExit("--execute requires a nonzero --gpu-id")
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")
    return {
        "probe": "Late-state scripted reachability audit",
        "task_pairs": [list(pair) for pair in TASK_PAIRS],
        "offsets": list(OFFSETS),
        "chunk_size": CHUNK_SIZE,
        "position_scale": POSITION_SCALE,
        "waypoint_tolerance": WAYPOINT_TOLERANCE,
        "max_waypoint_steps": MAX_WAYPOINT_STEPS,
        "safe_clearance": SAFE_CLEARANCE,
        "release_steps": RELEASE_STEPS,
        "settle_steps": SETTLE_STEPS,
        "maximum_states": 9,
        "init_index": args.init_index,
        "checkpoint": str(GOAL_CHECKPOINT),
        "checkpoint_config_sha256": sha256(GOAL_CHECKPOINT / "config.json"),
        "terminal_target_result": str(terminal_result),
        "gpu_id": args.gpu_id,
        "output_dir": str(args.output_dir),
        "execute": args.execute,
        "post_switch_policy": "fixed Cartesian scripted controller",
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
        raise RuntimeError("audit requires exactly one launcher-visible GPU")

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
        max_episode_length=300,
        task_names_to_use=list(TASKS),
        use_laplace_sampling=False,
        scale_factor=1.0,
    )
    task_ids = {task: runner.env_names.index(task) for task in TASKS}
    descriptions = {
        task: runner.benchmark.get_task(task_ids[task]).language for task in TASKS
    }
    parsed_goals = {}
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
    init_state = np.asarray(runner.benchmark.get_task_init_states(source_id))[0]
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

    def set_goal(task: str) -> None:
        env.env.parsed_problem["goal_state"] = parsed_goals[task]

    def goal_truth() -> dict[str, bool]:
        previous = env.env.parsed_problem["goal_state"]
        truth = {}
        try:
            for task in TASKS:
                set_goal(task)
                truth[task] = bool(env.check_success())
        finally:
            env.env.parsed_problem["goal_state"] = previous
        return truth

    def bowl_position() -> Any:
        body_id = env.env.obj_body_id["akita_black_bowl_1"]
        return env.env.sim.data.body_xpos[body_id].copy()

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

    def reconstruct_baseline(task: str) -> dict[str, Any]:
        set_goal(task)
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
        states_after = []
        gripper_actions_after = []
        raw_chunks = []
        queue = []
        steps = 0
        while steps < 300:
            if not queue:
                queue = infer_chunk(obs, task)
                raw_chunks.append([action.copy() for action in queue])
            action = process_action(queue.pop(0), "openvla")
            obs, _, _, _ = env.step(action)
            states_after.append(np.asarray(env.get_sim_state()).copy())
            gripper_actions_after.append(
                np.asarray(env.env.robots[0].gripper.current_action).copy()
            )
            steps += 1
            if env.check_success():
                break
        if not env.check_success():
            raise RuntimeError(f"checkpoint baseline failed for {task}")
        chunk_index = penultimate_chunk_index(len(raw_chunks))
        chunk_start = chunk_index * CHUNK_SIZE
        return {
            "task": task,
            "steps": steps,
            "generated_chunks": len(raw_chunks),
            "chunk_index": chunk_index,
            "chunk_start": chunk_start,
            "states": {
                offset: {
                    "sim_state": states_after[
                        chunk_start + offset - 1
                    ].copy(),
                    "gripper_current_action": gripper_actions_after[
                        chunk_start + offset - 1
                    ].copy(),
                }
                for offset in OFFSETS
            },
        }

    terminal_result = json.loads(
        Path(plan["terminal_target_result"]).read_text(encoding="utf-8")
    )
    terminal_states = {
        row["instruction_task"]: np.asarray(row["terminal_state"])
        for row in terminal_result["rows"]
    }
    target_bowl_positions = {}
    for task in TASKS:
        env.reset()
        env.set_init_state(terminal_states[task])
        set_goal(task)
        if not env.check_success():
            raise RuntimeError(f"stored terminal state no longer passes: {task}")
        target_bowl_positions[task] = bowl_position()

    baselines = {
        task: reconstruct_baseline(task) for task, _ in TASK_PAIRS
    }
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_dir / "result.json.incomplete"

    try:
        for origin_task, revised_task in TASK_PAIRS:
            for offset in OFFSETS:
                checkpoint = baselines[origin_task]["states"][offset]
                state = checkpoint["sim_state"]
                env.reset()
                obs = env.set_init_state(state)
                env.env.robots[0].gripper.current_action = checkpoint[
                    "gripper_current_action"
                ].copy()
                restored = np.asarray(env.get_sim_state()).copy()
                set_goal(revised_task)
                checkpoint_truth = goal_truth()
                bowl_object = env.env.objects_dict["akita_black_bowl_1"]
                grasped = bool(
                    env.env._check_grasp(
                        env.env.robots[0].gripper, bowl_object
                    )
                )
                exact_restore = bool(np.array_equal(state, restored))
                eligible = bool(
                    exact_restore and not any(checkpoint_truth.values()) and grasped
                )
                current_bowl = bowl_position()
                start_eef = np.asarray(obs["robot0_eef_pos"]).copy()
                target_bowl = target_bowl_positions[revised_task]
                waypoint_rows = []
                lost_grasp = False
                if eligible:
                    waypoints = transport_waypoints(
                        current_bowl, target_bowl, start_eef - current_bowl
                    )
                    for waypoint in waypoints:
                        reached = False
                        for step in range(MAX_WAYPOINT_STEPS):
                            current_eef = np.asarray(obs["robot0_eef_pos"])
                            error = float(np.linalg.norm(waypoint - current_eef))
                            if error <= WAYPOINT_TOLERANCE:
                                reached = True
                                break
                            obs, _, _, _ = env.step(
                                position_action(
                                    current_eef, waypoint, gripper=1.0
                                )
                            )
                            if not env.env._check_grasp(
                                env.env.robots[0].gripper, bowl_object
                            ):
                                lost_grasp = True
                        waypoint_rows.append(
                            {
                                "target": waypoint.tolist(),
                                "steps": step + 1,
                                "reached": reached,
                                "final_error": float(
                                    np.linalg.norm(
                                        waypoint
                                        - np.asarray(obs["robot0_eef_pos"])
                                    )
                                ),
                            }
                        )
                    for _ in range(RELEASE_STEPS):
                        obs, _, _, _ = env.step(
                            position_action(
                                np.asarray(obs["robot0_eef_pos"]),
                                np.asarray(obs["robot0_eef_pos"]),
                                gripper=-1.0,
                            )
                        )
                    for _ in range(SETTLE_STEPS):
                        obs, _, _, _ = env.step(
                            position_action(
                                np.asarray(obs["robot0_eef_pos"]),
                                np.asarray(obs["robot0_eef_pos"]),
                                gripper=-1.0,
                            )
                        )
                terminal_truth = goal_truth()
                row = {
                    "origin_task": origin_task,
                    "revised_task": revised_task,
                    "offset": offset,
                    "checkpoint_state_sha256": hashlib.sha256(
                        np.ascontiguousarray(state).tobytes()
                    ).hexdigest(),
                    "exact_restore": exact_restore,
                    "gripper_current_action": checkpoint[
                        "gripper_current_action"
                    ].tolist(),
                    "checkpoint_goal_truth": checkpoint_truth,
                    "checkpoint_all_goals_false": not any(
                        checkpoint_truth.values()
                    ),
                    "grasped_at_checkpoint": grasped,
                    "eligible": eligible,
                    "current_bowl_position": current_bowl.tolist(),
                    "target_bowl_position": target_bowl.tolist(),
                    "waypoints": waypoint_rows,
                    "lost_grasp": lost_grasp,
                    "terminal_bowl_position": bowl_position().tolist(),
                    "terminal_goal_truth": terminal_truth,
                    "revised_goal_success": bool(
                        terminal_truth[revised_task]
                    ),
                }
                rows.append(row)
                incomplete.write_text(
                    json.dumps(
                        {**plan, "rows": rows, "progress_only": True},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    finally:
        vector_env.close()

    metrics = summarize_oracle_rows(rows)
    result = {
        **plan,
        "completed_at": datetime.now().astimezone().isoformat(),
        "baselines": [
            {
                key: value
                for key, value in baseline.items()
                if key != "states"
            }
            for baseline in baselines.values()
        ],
        "target_bowl_positions": {
            task: position.tolist()
            for task, position in target_bowl_positions.items()
        },
        "rows": rows,
        "metrics": metrics,
        "temporal_training_gate_pass": metrics["reachability_pass"],
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
