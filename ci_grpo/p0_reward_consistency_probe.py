#!/usr/bin/env python3
"""Compare CI-GRPO goal aggregation with LIBERO's official success evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_LIBERO_ROOT = Path("/home/imc/code/LIBERO-official")
TASKS = (
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
)


def score_goal_from_predicates(problem_env: Any, goal_state: list[Any]) -> bool:
    """CI-GRPO scorer: explicit conjunction of LIBERO predicate primitives."""
    return all(bool(problem_env._eval_predicate(predicate)) for predicate in goal_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-result",
        type=Path,
        default=PROJECT_ROOT / "ci_grpo/artifacts/p0_openvla_k3_rollout/result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "ci_grpo/artifacts/p0_reward_consistency_100",
    )
    parser.add_argument("--states-per-source", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")
    if not args.rollout_result.is_file():
        raise SystemExit(f"rollout result not found: {args.rollout_result}")
    if args.states_per_source < 1:
        raise SystemExit("--states-per-source must be positive")

    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBERO_CONFIG_PATH"] = str(PROJECT_ROOT / ".libero_official")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ci_grpo_numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ci_grpo_mpl_cache")
    sys.path.insert(0, str(OFFICIAL_LIBERO_ROOT))

    import numpy as np
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils

    rollout = json.loads(args.rollout_result.read_text(encoding="utf-8"))
    terminal_states = {
        row["instruction_task"]: np.asarray(row["terminal_state"], dtype=np.float64)
        for row in rollout["rows"]
    }
    bddl_dir = (
        OFFICIAL_LIBERO_ROOT / "libero/libero/bddl_files/libero_goal"
    )
    goals = {
        task: BDDLUtils.robosuite_parse_problem(str(bddl_dir / f"{task}.bddl"))[
            "goal_state"
        ]
        for task in TASKS
    }
    init_path = (
        OFFICIAL_LIBERO_ROOT
        / "libero/libero/init_files/libero_goal"
        / f"{TASKS[0]}.pruned_init"
    )
    init_states = torch.load(init_path, map_location="cpu", weights_only=False)
    source_states = {
        "initial": np.asarray(init_states[rollout["init_index"]], dtype=np.float64),
        **terminal_states,
    }

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_dir / f"{TASKS[0]}.bddl"),
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(0)
    rows = []
    disagreement_count = 0
    positive_counts = {task: 0 for task in TASKS}
    try:
        for source_name, source_state in source_states.items():
            env.reset()
            env.set_init_state(source_state)
            for local_step in range(args.states_per_source):
                if local_step:
                    env.step(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
                for task in TASKS:
                    goal_state = goals[task]
                    custom = score_goal_from_predicates(env.env, goal_state)
                    previous = env.env.parsed_problem["goal_state"]
                    env.env.parsed_problem["goal_state"] = goal_state
                    try:
                        official = bool(env.check_success())
                    finally:
                        env.env.parsed_problem["goal_state"] = previous
                    disagreement_count += int(custom != official)
                    positive_counts[task] += int(official)
                    rows.append(
                        {
                            "source": source_name,
                            "local_step": local_step,
                            "scored_task": task,
                            "custom": custom,
                            "official": official,
                        }
                    )
    finally:
        env.close()

    expected_states = len(source_states) * args.states_per_source
    expected_comparisons = expected_states * len(TASKS)
    if len(rows) != expected_comparisons:
        raise RuntimeError(f"expected {expected_comparisons} comparisons, got {len(rows)}")
    result = {
        "probe": "CI-GRPO P0 reward consistency",
        "completed_at": datetime.now().astimezone().isoformat(),
        "rollout_result": str(args.rollout_result.resolve()),
        "state_sources": list(source_states),
        "states_per_source": args.states_per_source,
        "sampled_states": expected_states,
        "goal_evaluations": expected_comparisons,
        "agreements": expected_comparisons - disagreement_count,
        "disagreements": disagreement_count,
        "agreement_rate": (expected_comparisons - disagreement_count)
        / expected_comparisons,
        "positive_counts": positive_counts,
        "p0_t0_3_pass": disagreement_count == 0
        and all(count > 0 for count in positive_counts.values()),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_dir / "result.json.incomplete"
    incomplete.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(args.output_dir / "result.json")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
