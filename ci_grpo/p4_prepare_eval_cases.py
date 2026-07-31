#!/usr/bin/env python3
"""Prepare byte-exact P2 action prefixes for repeated recovery-SFT evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import site
import sys
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


REFERENCE = (
    PROJECT_ROOT
    / "ci_grpo/artifacts/p2_semantic_chunk_interrupt_openvla_run3/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/p4_eval_cases",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict[str, Any]:
    required = (
        GOAL_CHECKPOINT / "config.json",
        GOAL_CHECKPOINT / "action_head--50000_checkpoint.pt",
        GOAL_CHECKPOINT / "proprio_projector--50000_checkpoint.pt",
        SCALE_HEADER,
        REFERENCE,
        PROJECT_ROOT / ".libero_official/config.yaml",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"required paths missing: {missing}")
    if args.execute and (args.gpu_id is None or args.gpu_id <= 0):
        raise SystemExit("--execute requires a nonzero --gpu-id")
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")
    return {
        "experiment": "Prepare exact P2 recovery evaluation cases",
        "reference": str(REFERENCE),
        "reference_sha256": sha256(REFERENCE),
        "task_pairs": [list(pair) for pair in TASK_PAIRS],
        "offsets": list(OFFSETS),
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

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    expected = {
        (row["origin_task"], row["revised_task"], row["offset"]): row[
            "checkpoint_state_sha256"
        ]
        for row in reference["rows"]
        if row["method"] == "flush"
    }
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
    init_state = np.asarray(
        runner.benchmark.get_task_init_states(task_ids[TASKS[0]])
    )[0]
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

    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = []
    try:
        for origin_task, revised_task in TASK_PAIRS:
            env.env.parsed_problem["goal_state"] = parsed_goals[origin_task]
            env.reset()
            obs = env.set_init_state(init_state)
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
            raw_chunks = []
            processed_actions = []
            states_after = []
            queue = []
            steps = 0
            while steps < 300:
                if not queue:
                    queue = infer_chunk(obs, origin_task)
                    raw_chunks.append([action.copy() for action in queue])
                processed = np.asarray(
                    process_action(queue.pop(0), "openvla")
                )
                obs, _, _, _ = env.step(processed)
                processed_actions.append(processed.copy())
                states_after.append(np.asarray(env.get_sim_state()).copy())
                steps += 1
                if env.check_success():
                    break
            if not env.check_success():
                raise RuntimeError(f"base reconstruction failed: {origin_task}")
            chunk_start = penultimate_chunk_index(len(raw_chunks)) * CHUNK_SIZE
            for offset in OFFSETS:
                prefix_length = chunk_start + offset
                state = states_after[prefix_length - 1]
                state_hash = hashlib.sha256(
                    np.ascontiguousarray(state).tobytes()
                ).hexdigest()
                expected_hash = expected[(origin_task, revised_task, offset)]
                if state_hash != expected_hash:
                    raise RuntimeError(
                        f"P2 exact state mismatch for {origin_task} -> "
                        f"{revised_task} offset {offset}: "
                        f"{state_hash} != {expected_hash}"
                    )
                path = args.output_dir / (
                    f"{origin_task}__to__{revised_task}__offset_{offset}.npz"
                )
                np.savez(
                    path,
                    processed_prefix=np.stack(
                        processed_actions[:prefix_length]
                    ),
                )
                cases.append(
                    {
                        "origin_task": origin_task,
                        "revised_task": revised_task,
                        "offset": offset,
                        "prefix_length": prefix_length,
                        "checkpoint_state_sha256": state_hash,
                        "path": str(path),
                        "sha256": sha256(path),
                    }
                )
    finally:
        vector_env.close()

    manifest = {**plan, "cases": cases}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    plan = validate(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    execute(args, plan)


if __name__ == "__main__":
    main()
