#!/usr/bin/env python3
"""Paired anchor/fixed-L2 evaluation for one RA-LOOP checkpoint.

The safe default is a CPU-only path/config plan. ``--execute`` must be explicit
and is intended to be launched through the guarded shell wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path("/home/imc/yzy/RA_LOOP")
RIPT_ROOT = Path("/home/imc/code/ript-vla")
OFFICIAL_LIBERO_ROOT = Path("/home/imc/code/LIBERO-official")
BASE_MODEL = PROJECT_ROOT / "runtime/openvla-oft-spatial-smoke"
BASE_SCALE_HEADER = Path(
    "/home/imc/models/ra-loop/ript-vla/openvla_oft/scale_header/"
    "LIBERO_SPATIAL_scale_header.pth"
)
PILOT_ROOT = PROJECT_ROOT / (
    "outputs/ra_loop_spatial_overnight_pilot/libero_spatial/LIBERO_SPATIAL/"
    "openvla/RA-LOOP_spatial_robot_init_overnight_pilot/"
    "one_task_21step_k8_h220_fixed_l2_0p1_recovery_lr1e5/run_000"
)
AFTERNOON_ROOT = PROJECT_ROOT / (
    "outputs/ra_loop_spatial_afternoon_multitask/libero_spatial/LIBERO_SPATIAL/"
    "openvla/RA-LOOP_spatial_robot_init_afternoon_multitask/"
    "four_task_35step_k8_h220_fixed_l2_0p1_recovery_lr1e5_step5_warmstart/run_000"
)
DEFAULT_TASK_NAME = "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
VALID_TASKS = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    DEFAULT_TASK_NAME,
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
)
VALID_SOURCE_STEPS = {
    "pilot": (0, 5, 10, 15, 20),
    "afternoon": (5, 10, 15, 20, 25, 30),
}
VALID_STEPS = tuple(sorted({step for steps in VALID_SOURCE_STEPS.values() for step in steps}))


def adapter_dir(source: str, step: int) -> Path | None:
    if step == 0:
        return None
    root = PILOT_ROOT if source == "pilot" else AFTERNOON_ROOT
    return root / f"openvla_lora_step_{step:06d}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> dict[str, object]:
    required = [
        BASE_MODEL / "config.json",
        BASE_MODEL / "action_head--150000_checkpoint.pt",
        BASE_MODEL / "proprio_projector--150000_checkpoint.pt",
        BASE_SCALE_HEADER,
        OFFICIAL_LIBERO_ROOT,
        RIPT_ROOT,
    ]
    if args.checkpoint_step not in VALID_SOURCE_STEPS[args.checkpoint_source]:
        raise SystemExit(
            f"step {args.checkpoint_step} is invalid for source {args.checkpoint_source}"
        )
    adaptor = adapter_dir(args.checkpoint_source, args.checkpoint_step)
    if adaptor is not None:
        required.extend(
            [
                adaptor / "adapter_config.json",
                adaptor / "adapter_model.safetensors",
                adaptor / "openvla_headers.pt",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"required paths missing: {missing}")
    if args.num_pairs < 1 or args.num_pairs > 50:
        raise SystemExit("num-pairs must be in [1, 50]")
    if args.fixed_l2 <= 0:
        raise SystemExit("fixed-l2 must be positive")
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")

    return {
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_source": args.checkpoint_source,
        "adapter_dir": None if adaptor is None else str(adaptor),
        "base_model": str(BASE_MODEL),
        "base_scale_header": str(BASE_SCALE_HEADER),
        "task": args.task_name,
        "init_indices": list(range(args.num_pairs)),
        "modes": ["anchor", "fixed_l2"],
        "fixed_l2": args.fixed_l2,
        "perturb_seed": args.perturb_seed,
        "num_pairs": args.num_pairs,
        "episodes": 2 * args.num_pairs,
        "deterministic_action_mean": True,
        "output_dir": str(args.output_dir),
        "execute": args.execute,
    }


def execute(args: argparse.Namespace, plan: dict[str, object]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBERO_CONFIG_PATH"] = str(PROJECT_ROOT / ".libero_official")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    sys.path[:0] = [str(PROJECT_ROOT), str(OFFICIAL_LIBERO_ROOT), str(RIPT_ROOT)]

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from ra_loop.ript_compat import InProcessOpenVLAOFTLiberoRunner  # noqa: PLC0415
    from ra_loop.ript_recovery import resolve_in_process_joint_layout  # noqa: PLC0415
    from ra_loop.robustness import apply_robot_init_perturbation  # noqa: PLC0415
    from ript.algos.rl_optimizers.openvla_oft_interface import (  # noqa: PLC0415
        OpenVLA_OFT_Policy,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires exactly one launcher-visible GPU")

    adaptor = adapter_dir(args.checkpoint_source, args.checkpoint_step)
    policy = OpenVLA_OFT_Policy(
        pretrained_checkpoint=str(BASE_MODEL),
        header_checkpoint=str(BASE_SCALE_HEADER),
        task_suite_name="LIBERO_SPATIAL",
        lora_rank=32,
        lora_dropout=0.0,
        lora_adaptor_ckpt=None if adaptor is None else str(adaptor),
        device_id=0,
        seed=7,
        fix_scale_head=True,
        log_scale_clip=[-2.0, 0.5],
    )
    runner = InProcessOpenVLAOFTLiberoRunner(
        benchmark_name="LIBERO_SPATIAL",
        rollouts_per_env=args.num_pairs,
        num_parallel_envs=1,
        max_episode_length=220,
        task_names_to_use=[args.task_name],
        use_laplace_sampling=False,
        scale_factor=1.0,
    )

    created_env = runner.create_env(args.task_name)
    vector_env, env_id, _ = created_env
    try:
        all_states = np.asarray(runner.benchmark.get_task_init_states(env_id))
        init_indices = np.arange(args.num_pairs, dtype=np.int64)
        anchors = all_states[init_indices].copy()
        layout = resolve_in_process_joint_layout(created_env)
        perturbations = []
        perturbed_states = []
        for init_index, state in zip(init_indices, anchors):
            result = apply_robot_init_perturbation(
                state,
                layout=layout,
                strength=args.fixed_l2,
                seed=args.perturb_seed + int(init_index),
                sampling_mode="fixed_l2",
            )
            perturbed_states.append(result.state)
            perturbations.append(
                {
                    "init_index": int(init_index),
                    "seed": args.perturb_seed + int(init_index),
                    "joint_noise": result.noise.tolist(),
                    "joint_noise_l2": float(np.linalg.norm(result.noise)),
                }
            )
        perturbed = np.stack(perturbed_states)

        rows: list[dict[str, object]] = []
        for mode, states in (("anchor", anchors), ("fixed_l2", perturbed)):
            rollouts = runner.run_policy_in_env(
                args.task_name,
                policy,
                all_init_states=states,
                render=False,
                created_env=created_env,
                random_init=False,
            )
            for local_index, (success, total_reward, episode) in enumerate(rollouts):
                row = {
                    "checkpoint_step": args.checkpoint_step,
                    "mode": mode,
                    "init_index": int(init_indices[local_index]),
                    "success": bool(success),
                    "total_reward": float(total_reward),
                    "episode_action_steps": len(episode.get("actions", [])),
                }
                if mode == "fixed_l2":
                    row.update(perturbations[local_index])
                rows.append(row)
        if len(rows) != 2 * args.num_pairs:
            raise RuntimeError(f"expected {2 * args.num_pairs} rows, got {len(rows)}")
    finally:
        vector_env.close()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_dir / "results.jsonl.incomplete"
    final_results = args.output_dir / "results.jsonl"
    with incomplete.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    incomplete.replace(final_results)

    summary = {
        **plan,
        "completed_at": datetime.now().astimezone().isoformat(),
        "anchor_successes": sum(row["success"] for row in rows if row["mode"] == "anchor"),
        "perturbed_successes": sum(
            row["success"] for row in rows if row["mode"] == "fixed_l2"
        ),
        "result_rows": len(rows),
    }
    if adaptor is not None:
        summary["adapter_sha256"] = sha256(adaptor / "adapter_model.safetensors")
        summary["headers_sha256"] = sha256(adaptor / "openvla_headers.pt")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-step", type=int, choices=VALID_STEPS, required=True)
    parser.add_argument(
        "--checkpoint-source", choices=tuple(VALID_SOURCE_STEPS), default="pilot"
    )
    parser.add_argument("--task-name", choices=VALID_TASKS, default=DEFAULT_TASK_NAME)
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--fixed-l2", type=float, default=0.1)
    parser.add_argument("--perturb-seed", type=int, default=20260720)
    parser.add_argument("--gpu-id", type=int, choices=range(8), default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = validate(args)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.execute:
        execute(args, plan)


if __name__ == "__main__":
    main()
