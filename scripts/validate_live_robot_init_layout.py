#!/usr/bin/env python3
"""Validate Robot-init addresses against one live official LIBERO environment."""

from __future__ import annotations

import json
import os

import numpy as np

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path

from ra_loop.robustness import (
    PANDA_ARM_JOINT_NAMES,
    apply_robot_init_perturbation,
    resolve_named_joint_layout,
)


TASK_NAME = "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"


def main() -> None:
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    matching = [suite.get_task(i) for i in range(suite.n_tasks) if suite.get_task(i).name == TASK_NAME]
    if len(matching) != 1:
        raise RuntimeError(f"expected one task named {TASK_NAME}, found {len(matching)}")
    task = matching[0]
    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=64,
        camera_widths=64,
    )
    try:
        env.seed(0)
        env.reset()
        model = env.sim.model
        raw_qpos_addresses = tuple(
            int(model.get_joint_qpos_addr(name)) for name in PANDA_ARM_JOINT_NAMES
        )
        layout = resolve_named_joint_layout(model, state_qpos_offset=1)
        if layout.qpos_indices != tuple(address + 1 for address in raw_qpos_addresses):
            raise AssertionError("flattened-state indices do not equal qpos addresses + 1")

        state = np.asarray(env.get_sim_state())
        original = state.copy()
        result = apply_robot_init_perturbation(
            state, layout=layout, strength=0.001, seed=20260720
        )
        changed = tuple(np.flatnonzero(result.state != original).tolist())
        if not changed or not set(changed).issubset(layout.qpos_indices):
            raise AssertionError(
                f"changed indices {changed} are not a non-empty subset of {layout.qpos_indices}"
            )
        if not np.array_equal(state, original):
            raise AssertionError("input simulator state was mutated")

        # In-memory round trip only: prove MuJoCo accepts the bounded state.
        env.set_init_state(result.state)
        roundtrip = np.asarray(env.get_sim_state())
        if not np.isfinite(roundtrip).all() or roundtrip.shape != original.shape:
            raise AssertionError("simulator rejected the perturbed state")

        print(
            json.dumps(
                {
                    "task": TASK_NAME,
                    "state_shape": list(original.shape),
                    "raw_qpos_addresses": list(raw_qpos_addresses),
                    "flat_state_indices": list(layout.qpos_indices),
                    "joint_lower": layout.lower.tolist(),
                    "joint_upper": layout.upper.tolist(),
                    "changed_indices": list(changed),
                    "max_abs_applied_noise": float(np.abs(result.noise).max()),
                    "simulator_roundtrip": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
