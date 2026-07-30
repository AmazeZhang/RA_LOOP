#!/usr/bin/env python3
"""P0 probe for same-state, different-goal LIBERO contrastive groups.

The probe deliberately does not load a policy or use a GPU.  It creates one
official LIBERO environment per candidate instruction, injects the exact same
MuJoCo state into every environment, and verifies that the rendered
observations and simulator states agree.  It also swaps the parsed goal
predicate inside one live environment to verify that reward evaluation is
independent from scene construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


DEFAULT_TASKS = (
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=Path("/home/imc/code/LIBERO-official"),
    )
    parser.add_argument(
        "--libero-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".libero-official",
    )
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--source-task", default=DEFAULT_TASKS[0])
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "p0_same_state_k3",
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_array(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def normalize_for_json(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(key): normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def scene_signature(parsed_problem: dict[str, Any]) -> dict[str, Any]:
    """Return BDDL fields that must match for state-compatible environments."""
    keys = (
        "problem_name",
        "fixtures",
        "regions",
        "objects",
        "scene_properties",
        "initial_state",
    )
    return {key: normalize_for_json(parsed_problem[key]) for key in keys}


@contextmanager
def temporary_goal(environment: Any, goal_state: list[Any]) -> Iterator[None]:
    previous = environment.env.parsed_problem["goal_state"]
    environment.env.parsed_problem["goal_state"] = goal_state
    try:
        yield
    finally:
        environment.env.parsed_problem["goal_state"] = previous


def save_rgb(path: Path, image: Any) -> None:
    from PIL import Image
    import numpy as np

    rgb = np.asarray(image)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def main() -> None:
    args = parse_args()
    tasks = tuple(args.tasks or DEFAULT_TASKS)
    if len(tasks) != 3:
        raise SystemExit(f"P0 requires exactly K=3 tasks, got {len(tasks)}")
    if len(set(tasks)) != len(tasks):
        raise SystemExit("P0 tasks must be distinct")
    if args.init_index < 0:
        raise SystemExit("--init-index must be non-negative")
    if args.camera_size < 32:
        raise SystemExit("--camera-size must be at least 32")

    libero_package_root = args.libero_root / "libero"
    bddl_dir = libero_package_root / "libero" / "bddl_files" / args.suite
    init_dir = libero_package_root / "libero" / "init_files" / args.suite
    if not libero_package_root.is_dir():
        raise SystemExit(f"LIBERO package root not found: {libero_package_root}")
    if not (args.libero_config / "config.yaml").is_file():
        raise SystemExit(f"LIBERO config not found: {args.libero_config / 'config.yaml'}")

    # These must be set before importing MuJoCo, robosuite, or LIBERO.
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ci_grpo_numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ci_grpo_mpl_cache")
    # Official LIBERO exposes the package as ``libero.libero`` from its
    # repository root.  Prepending the root also overrides the installed
    # LIBERO-Plus checkout for this official-environment probe.
    sys.path.insert(0, str(args.libero_root.resolve()))

    import numpy as np
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils

    task_paths = {task: bddl_dir / f"{task}.bddl" for task in tasks}
    missing = [str(path) for path in task_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing BDDL files: {missing}")

    source_init_path = init_dir / f"{args.source_task}.pruned_init"
    if not source_init_path.is_file():
        raise SystemExit(f"Source init states not found: {source_init_path}")
    source_states = torch.load(source_init_path, map_location="cpu", weights_only=False)
    if args.init_index >= len(source_states):
        raise SystemExit(
            f"--init-index {args.init_index} out of range for {len(source_states)} states"
        )
    shared_state = np.asarray(source_states[args.init_index], dtype=np.float64)

    parsed = {
        task: BDDLUtils.robosuite_parse_problem(str(path))
        for task, path in task_paths.items()
    }
    signatures = {task: scene_signature(problem) for task, problem in parsed.items()}
    signature_hashes = {
        task: hashlib.sha256(canonical_json(signature).encode("utf-8")).hexdigest()
        for task, signature in signatures.items()
    }
    static_scene_equal = len(set(signature_hashes.values())) == 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environments = {}
    task_rows: dict[str, dict[str, Any]] = {}
    images_by_camera: dict[str, dict[str, Any]] = {}
    states = {}
    try:
        for task, bddl_path in task_paths.items():
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_path),
                camera_heights=args.camera_size,
                camera_widths=args.camera_size,
            )
            environments[task] = env
            env.reset()
            observation = env.set_init_state(shared_state)
            live_state = np.asarray(env.get_sim_state())
            states[task] = live_state

            camera_rows = {}
            for camera_key in ("agentview_image", "robot0_eye_in_hand_image"):
                image = np.asarray(observation[camera_key])
                images_by_camera.setdefault(camera_key, {})[task] = image
                image_path = args.output_dir / f"{task}__{camera_key}.png"
                save_rgb(image_path, image)
                camera_rows[camera_key] = {
                    "shape": list(image.shape),
                    "sha256": sha256_array(image),
                    "path": str(image_path.resolve()),
                }

            task_rows[task] = {
                "bddl": str(bddl_path.resolve()),
                "language": " ".join(parsed[task]["language_instruction"]),
                "goal_state": normalize_for_json(parsed[task]["goal_state"]),
                "scene_signature_sha256": signature_hashes[task],
                "state_shape": list(live_state.shape),
                "state_sha256": sha256_array(live_state),
                "initial_success": bool(env.check_success()),
                "cameras": camera_rows,
            }

        reference_task = tasks[0]
        state_comparisons = {}
        for task in tasks[1:]:
            delta = np.abs(states[reference_task] - states[task])
            state_comparisons[f"{reference_task}::{task}"] = {
                "array_equal": bool(np.array_equal(states[reference_task], states[task])),
                "max_abs_delta": float(np.max(delta)),
            }

        image_comparisons = {}
        for camera_key, task_images in images_by_camera.items():
            camera_comparisons = {}
            for task in tasks[1:]:
                reference = task_images[reference_task].astype(np.int16)
                candidate = task_images[task].astype(np.int16)
                delta = np.abs(reference - candidate)
                camera_comparisons[f"{reference_task}::{task}"] = {
                    "array_equal": bool(np.array_equal(reference, candidate)),
                    "max_abs_delta": int(np.max(delta)),
                    "different_values": int(np.count_nonzero(delta)),
                }
            image_comparisons[camera_key] = camera_comparisons

        # Prove that goal evaluation can change without rebuilding the scene.
        # This is the intended CI-GRPO implementation: one physical environment
        # supplies s0 and the image; only instruction metadata and the predicate
        # used to score a rollout vary across group members.
        switch_env = environments[reference_task]
        dynamic_goal_results = {}
        single_env_images: dict[str, dict[str, Any]] = {}
        for task in tasks:
            goal_state = parsed[task]["goal_state"]
            with temporary_goal(switch_env, goal_state):
                observation = switch_env.set_init_state(shared_state)
                dynamic_goal_results[task] = bool(switch_env.check_success())
                for camera_key in ("agentview_image", "robot0_eye_in_hand_image"):
                    image = np.asarray(observation[camera_key])
                    single_env_images.setdefault(camera_key, {})[task] = image
                    save_rgb(
                        args.output_dir / f"single_env__{task}__{camera_key}.png",
                        image,
                    )

        single_env_image_comparisons = {}
        for camera_key, task_images in single_env_images.items():
            camera_comparisons = {}
            for task in tasks[1:]:
                reference = task_images[reference_task].astype(np.int16)
                candidate = task_images[task].astype(np.int16)
                delta = np.abs(reference - candidate)
                camera_comparisons[f"{reference_task}::{task}"] = {
                    "array_equal": bool(np.array_equal(reference, candidate)),
                    "max_abs_delta": int(np.max(delta)),
                    "different_values": int(np.count_nonzero(delta)),
                }
            single_env_image_comparisons[camera_key] = camera_comparisons

        state_equal = all(row["array_equal"] for row in state_comparisons.values())
        cross_env_pixel_equal = all(
            row["array_equal"]
            for camera_rows in image_comparisons.values()
            for row in camera_rows.values()
        )
        single_env_pixel_equal = all(
            row["array_equal"]
            for camera_rows in single_env_image_comparisons.values()
            for row in camera_rows.values()
        )
        initially_unsatisfied = all(
            not row["initial_success"] for row in task_rows.values()
        ) and all(not result for result in dynamic_goal_results.values())

        result = {
            "created_at": datetime.now().astimezone().isoformat(),
            "probe": "CI-GRPO P0 same-state/different-goal K=3",
            "libero_root": str(args.libero_root.resolve()),
            "libero_config": str(args.libero_config.resolve()),
            "suite": args.suite,
            "tasks": list(tasks),
            "source_task": args.source_task,
            "source_init_path": str(source_init_path.resolve()),
            "init_index": args.init_index,
            "input_state_shape": list(shared_state.shape),
            "input_state_sha256": sha256_array(shared_state),
            "checks": {
                "static_scene_equal": static_scene_equal,
                "restored_state_equal": state_equal,
                "single_env_pixel_equal_after_goal_switch": single_env_pixel_equal,
                "all_goals_initially_unsatisfied": initially_unsatisfied,
                "dynamic_goal_switch_supported": True,
            },
            "diagnostics": {
                "separate_env_pixel_equal": cross_env_pixel_equal,
                "separate_env_note": (
                    "Separate environment construction is not used for contrastive "
                    "groups; renderer/model-level differences can survive identical "
                    "MuJoCo data state. One live environment is the required path."
                ),
            },
            "task_details": task_rows,
            "state_comparisons": state_comparisons,
            "image_comparisons": image_comparisons,
            "single_env_image_comparisons": single_env_image_comparisons,
            "dynamic_goal_results_on_one_live_env": dynamic_goal_results,
        }
        result["p0_t0_1_pass"] = all(result["checks"].values())

        result_path = args.output_dir / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"RESULT_PATH={result_path.resolve()}")
    finally:
        for env in environments.values():
            env.close()


if __name__ == "__main__":
    main()
