#!/usr/bin/env python3
"""Run a bounded same-state instruction/goal matrix with SmolVLA-LIBERO."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_ROOT = Path("/home/imc/wangdi/lerobot_vla/lerobot")
CHECKPOINT = Path("/home/imc/models/ra-loop/smolvla-libero")
GROUPS = {
    "bowl_destination_k3": (1, 4, 8),
    "wine_destination_k2": (2, 9),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--source-task-id", type=int, default=8)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episode-length", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "ci_grpo/artifacts/p0_smolvla_same_state_screen",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def group_metrics(rows: list[dict[str, Any]], task_ids: tuple[int, ...]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["instruction_task_id"] in task_ids and row["scored_goal_task_id"] in task_ids
    ]
    hits = [
        row["success"]
        for row in selected
        if row["instruction_task_id"] == row["scored_goal_task_id"]
    ]
    misses = [
        row["success"]
        for row in selected
        if row["instruction_task_id"] != row["scored_goal_task_id"]
    ]
    hit = sum(hits) / len(hits)
    expected_mismatches = len(task_ids) * (len(task_ids) - 1)
    miss = sum(misses) / len(misses) if misses else 0.0
    mismatch_complete = len(misses) == expected_mismatches
    remaining = expected_mismatches - len(misses)
    max_possible_miss = (sum(misses) + remaining) / expected_mismatches
    return {
        "n_hit_rollouts": len(hits),
        "n_mismatch_rollouts": len(misses),
        "expected_mismatch_rollouts": expected_mismatches,
        "hit": hit,
        "miss": miss,
        "lsg": hit - miss,
        "high_hit": hit >= 0.5,
        "mismatch_complete": mismatch_complete,
        "max_possible_miss": max_possible_miss,
        "candidate_still_possible": hit >= 0.5 and hit - max_possible_miss <= 0.15,
        "language_deaf_candidate": (
            hit >= 0.5 and mismatch_complete and hit - miss <= 0.15
        ),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.execute and (args.gpu_id is None or args.gpu_id <= 0):
        raise SystemExit("--execute requires a physical --gpu-id greater than 0")
    if args.source_task_id not in {task for group in GROUPS.values() for task in group}:
        raise SystemExit("--source-task-id must belong to a screened group")
    if args.init_index < 0 or args.init_index >= 50:
        raise SystemExit("--init-index must be in [0, 49]")
    if args.max_episode_length < 1 or args.max_episode_length > 300:
        raise SystemExit("--max-episode-length must be in [1, 300]")
    if not CHECKPOINT.joinpath("config.json").is_file():
        raise SystemExit(f"checkpoint missing: {CHECKPOINT}")
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")
    return {
        "probe": "CI-GRPO bounded SmolVLA same-state language screen",
        "checkpoint": str(CHECKPOINT),
        "groups": {name: list(task_ids) for name, task_ids in GROUPS.items()},
        "source_task_id": args.source_task_id,
        "init_index": args.init_index,
        "seed": args.seed,
        "max_episode_length": args.max_episode_length,
        "gpu_id": args.gpu_id,
        "output_dir": str(args.output_dir),
        "execute": args.execute,
    }


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBERO_CONFIG_PATH"] = str(PROJECT_ROOT / ".libero-official")
    os.environ["NUMBA_CACHE_DIR"] = "/tmp/ci_grpo_numba_cache"
    os.environ["MPLCONFIGDIR"] = "/tmp/ci_grpo_mpl_cache"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(LEROBOT_ROOT / "src"))

    import gymnasium as gym
    import torch
    import libero.libero.envs.bddl_utils as BDDLUtils
    from libero.libero import benchmark, get_libero_path
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import LiberoEnv
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires exactly one launcher-visible GPU")

    set_seed(args.seed)
    env_cfg = LiberoEnvConfig(
        task="libero_goal",
        task_ids=[args.source_task_id],
        episode_length=args.max_episode_length,
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        control_mode="relative",
    )
    policy_cfg = PreTrainedConfig.from_pretrained(str(CHECKPOINT))
    policy_cfg.pretrained_path = CHECKPOINT
    policy_cfg.device = "cuda"
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    base_env = LiberoEnv(
        task_suite=suite,
        task_id=args.source_task_id,
        task_suite_name="libero_goal",
        episode_length=args.max_episode_length,
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        control_mode="relative",
        episode_index=args.init_index,
    )
    vector_env = gym.vector.SyncVectorEnv([lambda: base_env])

    task_ids = tuple(task for group in GROUPS.values() for task in group)
    descriptions = {task_id: suite.get_task(task_id).language for task_id in task_ids}
    goals = {}
    for task_id in task_ids:
        task = suite.get_task(task_id)
        bddl_path = (
            Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        goals[task_id] = BDDLUtils.robosuite_parse_problem(str(bddl_path))[
            "goal_state"
        ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_dir / "result.json.incomplete"
    rows = []
    try:
        for group_name, group_task_ids in GROUPS.items():
            pairs = [
                (task_id, task_id) for task_id in group_task_ids
            ] + [
                (goal_id, instruction_id)
                for goal_id in group_task_ids
                for instruction_id in group_task_ids
                if goal_id != instruction_id
            ]
            for pair_index, (scored_goal_id, instruction_id) in enumerate(pairs):
                # Run all diagonal hits first. During mismatches, stop as
                # soon as even making every remaining rollout succeed
                # cannot bring LSG down to the 15pp candidate threshold.
                if pair_index >= len(group_task_ids):
                    current = group_metrics(rows, group_task_ids)
                    if not current["candidate_still_possible"]:
                        print(
                            f"early-stop {group_name}: candidate mathematically impossible",
                            flush=True,
                        )
                        break
                base_env._env.env.parsed_problem["goal_state"] = goals[
                    scored_goal_id
                ]
                base_env.task_description = descriptions[instruction_id]
                base_env.task = suite.get_task(instruction_id).name
                print(
                    f"group={group_name} goal={scored_goal_id} "
                    f"instruction={instruction_id}",
                    flush=True,
                )
                # SmolVLA samples the flow-matching action trajectory. Reset
                # every RNG before every matrix cell so only the instruction
                # and scored predicate vary.
                set_seed(args.seed)
                result = eval_policy(
                    env=vector_env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=1,
                    start_seed=args.seed,
                )
                episode = result["per_episode"][0]
                row = {
                    "group": group_name,
                    "scored_goal_task_id": scored_goal_id,
                    "scored_goal_language": descriptions[scored_goal_id],
                    "instruction_task_id": instruction_id,
                    "instruction_language": descriptions[instruction_id],
                    "success": bool(episode["success"]),
                    "sum_reward": float(episode["sum_reward"]),
                }
                rows.append(row)
                partial = {
                    **plan,
                    "rows": rows,
                    "completed_rollouts": len(rows),
                }
                incomplete.write_text(
                    json.dumps(partial, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    finally:
        vector_env.close()

    metrics = {
        name: group_metrics(rows, task_ids) for name, task_ids in GROUPS.items()
    }
    result = {
        **plan,
        "completed_at": datetime.now().astimezone().isoformat(),
        "rows": rows,
        "metrics": metrics,
        "valid_exclusive_group_candidates": [
            name
            for name, values in metrics.items()
            if values["language_deaf_candidate"]
        ],
        "continue_training_direction": any(
            values["language_deaf_candidate"] for values in metrics.values()
        ),
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
