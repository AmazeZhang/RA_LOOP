#!/usr/bin/env python3
"""Evaluate a recovery-SFT checkpoint on fixed late switches and retention."""

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
from ci_grpo.p2_semantic_chunk_interrupt_screen import OFFSETS, TASK_PAIRS


REFERENCE_INTERRUPT = (
    PROJECT_ROOT
    / "ci_grpo/artifacts/p2_semantic_chunk_interrupt_openvla_run3/result.json"
)
DATA_RESULT = (
    PROJECT_ROOT / "ci_grpo/artifacts/p4_oracle_recovery_data/result.json"
)
DATA_MANIFEST = PROJECT_ROOT / "data/p4_oracle_recovery_sft/manifest.json"
EVAL_CASES_MANIFEST = PROJECT_ROOT / "data/p4_eval_cases/manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def summarize_recovery_eval(
    retention_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    exact = [row for row in recovery_rows if row["exact_checkpoint_replay"]]
    successes = [row for row in exact if row["revised_goal_success"]]
    successful_pairs = {
        (row["origin_task"], row["revised_task"]) for row in successes
    }
    required_pairs = {
        (row["origin_task"], row["revised_task"]) for row in recovery_rows
    }
    retention_successes = sum(
        bool(row["original_goal_success"]) for row in retention_rows
    )
    passed = bool(
        len(exact) == 9
        and len(successes) >= 6
        and successful_pairs == required_pairs
        and retention_successes >= 2
    )
    return {
        "n_retention_rollouts": len(retention_rows),
        "n_retention_successes": retention_successes,
        "retention_success_rate": (
            retention_successes / len(retention_rows)
            if retention_rows
            else None
        ),
        "n_recovery_rollouts": len(recovery_rows),
        "n_exact_checkpoint_replays": len(exact),
        "n_revised_goal_successes": len(successes),
        "revised_goal_success_rate": (
            len(successes) / len(exact) if exact else None
        ),
        "all_directed_pairs_have_success": (
            successful_pairs == required_pairs
        ),
        "recovery_sft_gate_pass": passed,
        "decision": "pass" if passed else "fail",
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    required = (
        OFFICIAL_LIBERO_ROOT,
        RIPT_ROOT,
        OPENVLA_ROOT,
        GOAL_CHECKPOINT / "config.json",
        SCALE_HEADER,
        REFERENCE_INTERRUPT,
        DATA_RESULT,
        DATA_MANIFEST,
        EVAL_CASES_MANIFEST,
        args.adapter / "adapter_config.json",
        args.adapter / "adapter_model.safetensors",
        args.adapter / "openvla_headers.pt",
        PROJECT_ROOT / ".libero_official/config.yaml",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"required paths missing: {missing}")
    if args.max_episode_length != 300:
        raise SystemExit("evaluation is pre-registered only for 300 steps")
    if args.execute and (args.gpu_id is None or args.gpu_id <= 0):
        raise SystemExit("--execute requires a nonzero --gpu-id")
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")
    return {
        "experiment": "Late-state oracle recovery SFT evaluation",
        "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "headers_sha256": sha256(args.adapter / "openvla_headers.pt"),
        "reference_interrupt": str(REFERENCE_INTERRUPT),
        "reference_interrupt_sha256": sha256(REFERENCE_INTERRUPT),
        "data_manifest": str(DATA_MANIFEST),
        "data_manifest_sha256": sha256(DATA_MANIFEST),
        "eval_cases_manifest": str(EVAL_CASES_MANIFEST),
        "eval_cases_manifest_sha256": sha256(EVAL_CASES_MANIFEST),
        "task_pairs": [list(pair) for pair in TASK_PAIRS],
        "offsets": list(OFFSETS),
        "retention_tasks": list(TASKS),
        "maximum_rollouts": 12,
        "max_episode_length": args.max_episode_length,
        "gpu_id": args.gpu_id,
        "output_dir": str(args.output_dir),
        "execute": args.execute,
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
        raise RuntimeError("evaluation requires exactly one launcher-visible GPU")
    torch.cuda.reset_peak_memory_stats()

    reference = json.loads(REFERENCE_INTERRUPT.read_text(encoding="utf-8"))
    reference_hashes = {
        (row["origin_task"], row["revised_task"], row["offset"]): row[
            "checkpoint_state_sha256"
        ]
        for row in reference["rows"]
        if row["method"] == "flush"
    }
    eval_cases = json.loads(
        EVAL_CASES_MANIFEST.read_text(encoding="utf-8")
    )["cases"]

    policy = OpenVLA_OFT_Policy(
        pretrained_checkpoint=str(GOAL_CHECKPOINT),
        header_checkpoint=str(SCALE_HEADER),
        task_suite_name="LIBERO_GOAL",
        lora_rank=32,
        lora_dropout=0.0,
        lora_adaptor_ckpt=str(args.adapter),
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

    def reset_initial(task: str) -> Any:
        set_goal(task)
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
        return obs

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

    def run_policy(obs: Any, task: str, max_steps: int) -> tuple[int, bool]:
        queue = []
        steps = 0
        while steps < max_steps:
            if not queue:
                queue = infer_chunk(obs, task)
            action = process_action(queue.pop(0), "openvla")
            obs, _, _, _ = env.step(action)
            steps += 1
            if env.check_success():
                break
        return steps, bool(env.check_success())

    retention_rows = []
    recovery_rows = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_dir / "result.json.incomplete"

    def write_progress() -> None:
        incomplete.write_text(
            json.dumps(
                {
                    **plan,
                    "retention_rows": retention_rows,
                    "recovery_rows": recovery_rows,
                    "progress_only": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    try:
        for task in TASKS:
            obs = reset_initial(task)
            steps, success = run_policy(obs, task, args.max_episode_length)
            retention_rows.append(
                {
                    "task": task,
                    "steps": steps,
                    "original_goal_success": success,
                    "terminal_goal_truth": goal_truth(),
                }
            )
            write_progress()

        for case in eval_cases:
                origin_task = case["origin_task"]
                revised_task = case["revised_task"]
                offset = case["offset"]
                prefix_archive = np.load(case["path"], allow_pickle=False)
                processed_prefix = prefix_archive["processed_prefix"]
                prefix_archive.close()
                prefix_length = len(processed_prefix)
                obs = reset_initial(origin_task)
                for processed_action in processed_prefix:
                    obs, _, _, _ = env.step(processed_action.copy())
                replayed = np.asarray(env.get_sim_state()).copy()
                replayed_hash = hashlib.sha256(
                    np.ascontiguousarray(replayed).tobytes()
                ).hexdigest()
                expected_hash = case["checkpoint_state_sha256"]
                if expected_hash != reference_hashes[
                    (origin_task, revised_task, offset)
                ]:
                    raise RuntimeError("prepared case/reference hash mismatch")
                checkpoint_truth = goal_truth()
                exact = replayed_hash == expected_hash
                set_goal(revised_task)
                steps, success = run_policy(
                    obs,
                    revised_task,
                    args.max_episode_length - prefix_length,
                )
                recovery_rows.append(
                    {
                        "origin_task": origin_task,
                        "revised_task": revised_task,
                        "offset": offset,
                        "switch_control_step": prefix_length,
                        "expected_checkpoint_state_sha256": expected_hash,
                        "replayed_checkpoint_state_sha256": replayed_hash,
                        "exact_checkpoint_replay": exact,
                        "checkpoint_goal_truth": checkpoint_truth,
                        "checkpoint_all_goals_false": not any(
                            checkpoint_truth.values()
                        ),
                        "steps_after_switch": steps,
                        "revised_goal_success": success,
                        "terminal_goal_truth": goal_truth(),
                    }
                )
                write_progress()
    finally:
        vector_env.close()

    metrics = summarize_recovery_eval(retention_rows, recovery_rows)
    result = {
        **plan,
        "completed_at": datetime.now().astimezone().isoformat(),
        "retention_rows": retention_rows,
        "recovery_rows": recovery_rows,
        "metrics": metrics,
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
