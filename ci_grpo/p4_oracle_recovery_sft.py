#!/usr/bin/env python3
"""Run the pre-registered small-data late-state oracle recovery SFT."""

from __future__ import annotations

import argparse
import json
import os
import random
import site
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PROJECT_ROOT))

from ci_grpo.p0_openvla_k3_rollout import (
    GOAL_CHECKPOINT,
    OPENVLA_ROOT,
    PROJECT_ROOT,
    RIPT_ROOT,
    SCALE_HEADER,
    sha256,
)


CHUNK_SIZE = 8
ACTION_DIM = 7
LORA_RANK = 32
LEARNING_RATE = 1e-5
GRAD_ACCUMULATION_STEPS = 8
MAX_STEPS = 200
CHECKPOINT_STEPS = (25, 50, 100, 200)
SEED = 7
STATS_KEY = "libero_goal_no_noops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/p4_oracle_recovery_sft",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/p4_oracle_recovery_sft",
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def action_chunk(actions: Any, start: int, size: int = CHUNK_SIZE) -> Any:
    """Return a future action chunk, repeating the terminal action as padding."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"expected [time, {ACTION_DIM}] actions, got {actions.shape}")
    if start < 0 or start >= len(actions):
        raise IndexError(f"start {start} outside trajectory of length {len(actions)}")
    indices = np.minimum(np.arange(start, start + size), len(actions) - 1)
    return actions[indices].copy()


def normalize_action_chunk(chunk: Any, statistics: dict[str, Any]) -> Any:
    """Apply the checkpoint's q01/q99 action normalization exactly once."""
    chunk = np.asarray(chunk, dtype=np.float32)
    low = np.asarray(statistics["q01"], dtype=np.float32)
    high = np.asarray(statistics["q99"], dtype=np.float32)
    mask = np.asarray(
        statistics.get("mask", np.ones(ACTION_DIM, dtype=bool)), dtype=bool
    )
    normalized = np.where(
        mask,
        2.0 * (chunk - low) / (high - low + 1e-8) - 1.0,
        chunk,
    )
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def balanced_microbatch_schedule(
    recovery_size: int,
    retention_size: int,
    *,
    optimizer_steps: int,
    seed: int = SEED,
) -> list[tuple[str, int]]:
    """Create a deterministic 1:1 split schedule within every optimizer step."""
    if recovery_size < 1 or retention_size < 1:
        raise ValueError("both recovery and retention pools must be non-empty")
    rng = random.Random(seed)
    schedule = []
    for _ in range(optimizer_steps):
        pairs = [
            ("recovery", rng.randrange(recovery_size))
            for _ in range(GRAD_ACCUMULATION_STEPS // 2)
        ]
        pairs.extend(
            ("retention", rng.randrange(retention_size))
            for _ in range(GRAD_ACCUMULATION_STEPS // 2)
        )
        rng.shuffle(pairs)
        schedule.extend(pairs)
    return schedule


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.data_dir / "manifest.json"
    required = (
        GOAL_CHECKPOINT / "config.json",
        GOAL_CHECKPOINT / "dataset_statistics.json",
        GOAL_CHECKPOINT / "action_head--50000_checkpoint.pt",
        GOAL_CHECKPOINT / "proprio_projector--50000_checkpoint.pt",
        SCALE_HEADER,
        manifest,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"required paths missing: {missing}")
    if args.execute and (args.gpu_id is None or args.gpu_id <= 0):
        raise SystemExit("--execute requires a nonzero --gpu-id")
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    trajectories = manifest_data.get("trajectories", [])
    recovery = [row for row in trajectories if row.get("split") == "recovery"]
    retention = [row for row in trajectories if row.get("split") == "retention"]
    if len(recovery) != 9 or len(retention) != 3:
        raise SystemExit(
            f"expected 9 recovery and 3 retention trajectories, got "
            f"{len(recovery)} and {len(retention)}"
        )
    if not all(row.get("terminal_success") for row in trajectories):
        raise SystemExit("all training trajectories must have terminal success")
    for row in trajectories:
        path = Path(row["path"])
        if not path.exists() or sha256(path) != row["sha256"]:
            raise SystemExit(f"trajectory hash mismatch: {path}")

    return {
        "experiment": "Late-state oracle recovery SFT",
        "checkpoint": str(GOAL_CHECKPOINT),
        "checkpoint_config_sha256": sha256(GOAL_CHECKPOINT / "config.json"),
        "checkpoint_statistics_sha256": sha256(
            GOAL_CHECKPOINT / "dataset_statistics.json"
        ),
        "data_manifest": str(manifest),
        "data_manifest_sha256": sha256(manifest),
        "n_recovery_trajectories": len(recovery),
        "n_retention_trajectories": len(retention),
        "lora_rank": LORA_RANK,
        "learning_rate": LEARNING_RATE,
        "grad_accumulation_steps": GRAD_ACCUMULATION_STEPS,
        "effective_batch_size": GRAD_ACCUMULATION_STEPS,
        "max_optimizer_steps": MAX_STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "balanced_split_per_optimizer_step": True,
        "seed": SEED,
        "gpu_id": args.gpu_id,
        "smoke_only": args.smoke_only,
        "output_dir": str(args.output_dir),
        "execute": args.execute,
    }


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONNOUSERSITE"] = "1"
    user_site = str(site.getusersitepackages())
    sys.path[:] = [entry for entry in sys.path if str(entry) != user_site]
    sys.path[:0] = [str(PROJECT_ROOT), str(RIPT_ROOT), str(OPENVLA_ROOT)]

    import torch
    from experiments.robot.openvla_utils import (
        normalize_proprio,
        prepare_images_for_vla,
    )
    from prismatic.vla.constants import IGNORE_INDEX
    from ript.algos.rl_optimizers.openvla_oft_interface import OpenVLA_OFT_Policy
    from ript.env_runner.openvla_oft_libero_runner import (
        _regression_or_discrete_prediction_batch,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("training requires exactly one launcher-visible GPU")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()

    manifest = json.loads(
        (args.data_dir / "manifest.json").read_text(encoding="utf-8")
    )
    stats = json.loads(
        (GOAL_CHECKPOINT / "dataset_statistics.json").read_text(encoding="utf-8")
    )[STATS_KEY]
    pools: dict[str, list[dict[str, Any]]] = {
        "recovery": [],
        "retention": [],
    }
    opened = []
    for row in manifest["trajectories"]:
        archive = np.load(row["path"], allow_pickle=False)
        opened.append(archive)
        for index in range(len(archive["action"])):
            pools[row["split"]].append(
                {
                    "archive": archive,
                    "index": index,
                    "instruction": str(archive["instruction"]),
                }
            )

    policy = OpenVLA_OFT_Policy(
        pretrained_checkpoint=str(GOAL_CHECKPOINT),
        header_checkpoint=str(SCALE_HEADER),
        task_suite_name="LIBERO_GOAL",
        lora_rank=LORA_RANK,
        lora_dropout=0.0,
        lora_adaptor_ckpt=None,
        device_id=0,
        seed=SEED,
        fix_scale_head=True,
        log_scale_clip=[-2.0, 0.5],
    )
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
    model.eval()
    action_head.train()
    scale_head.eval()
    proprio_projector.eval()
    for module in (scale_head, proprio_projector):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    trainable = [
        parameter
        for parameter in list(model.parameters()) + list(action_head.parameters())
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=0.0
    )

    def forward_sample(sample: dict[str, Any]) -> tuple[Any, Any]:
        archive = sample["archive"]
        index = sample["index"]
        prompt = (
            "In: What action should the robot take to "
            f"{sample['instruction'].lower()}?\nOut:"
        )
        primary, wrist = prepare_images_for_vla(
            [archive["full_image"][index], archive["wrist_image"][index]],
            policy.cfg,
        )
        inputs = policy.processor([prompt], [primary]).to(
            model.device, dtype=torch.bfloat16
        )
        wrist_inputs = policy.processor([prompt], [wrist]).to(
            model.device, dtype=torch.bfloat16
        )
        inputs["pixel_values"] = torch.cat(
            [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if not torch.all(input_ids[:, -1] == 29871):
            empty_token = torch.full(
                (input_ids.shape[0], 1),
                29871,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, empty_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(empty_token)], dim=1
            )
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        num_prompt_tokens = input_ids.shape[-1] - 1
        input_ids, attention_mask = model._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        labels = model._prepare_labels_for_action_prediction(labels, input_ids)
        input_embeddings = model.get_input_embeddings()(input_ids)
        all_actions_mask = model._process_action_masks(labels)
        with torch.no_grad():
            language_embeddings = input_embeddings[
                ~all_actions_mask
            ].reshape(input_embeddings.shape[0], -1, input_embeddings.shape[2])
            projected = model._process_vision_features(
                inputs["pixel_values"], language_embeddings, False
            )
            proprio = normalize_proprio(
                archive["proprio"][index], stats["proprio"]
            )
            proprio_tensor = torch.as_tensor(
                proprio,
                device=projected.device,
                dtype=projected.dtype,
            ).unsqueeze(0)
            projected = model._process_proprio_features(
                projected, proprio_tensor, proprio_projector
            )
        num_patches = (
            model.vision_backbone.get_num_patches()
            * model.vision_backbone.get_num_images_in_input()
            + 1
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted, _ = _regression_or_discrete_prediction_batch(
                model,
                input_embeddings,
                all_actions_mask,
                projected,
                attention_mask,
                labels,
                num_patches,
                num_prompt_tokens,
                action_head,
                scale_head,
            )
            target_np = normalize_action_chunk(
                action_chunk(archive["action"], index), stats["action"]
            )
            target = torch.as_tensor(
                target_np, device=predicted.device, dtype=predicted.dtype
            ).unsqueeze(0)
            loss = torch.nn.functional.l1_loss(predicted, target)
        return loss, target

    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = balanced_microbatch_schedule(
        len(pools["recovery"]),
        len(pools["retention"]),
        optimizer_steps=MAX_STEPS,
    )
    log_path = args.output_dir / "training_log.jsonl"
    optimizer.zero_grad(set_to_none=True)

    if args.smoke_only:
        split, index = schedule[0]
        loss, _ = forward_sample(pools[split][index])
        loss.backward()
        smoke = {
            **plan,
            "completed_at": datetime.now().astimezone().isoformat(),
            "loss": float(loss.detach().float().cpu()),
            "gpu_peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
            ),
            "gpu_peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 1024**3
            ),
            "n_trainable_parameters": sum(
                parameter.numel() for parameter in trainable
            ),
            "backward_passed": True,
            "optimizer_step_executed": False,
        }
        (args.output_dir / "smoke_result.json").write_text(
            json.dumps(smoke, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return

    recent_losses = []
    for microstep, (split, index) in enumerate(schedule, start=1):
        loss, _ = forward_sample(pools[split][index])
        (loss / GRAD_ACCUMULATION_STEPS).backward()
        recent_losses.append(float(loss.detach().float().cpu()))
        if microstep % GRAD_ACCUMULATION_STEPS:
            continue
        optimizer_step = microstep // GRAD_ACCUMULATION_STEPS
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "optimizer_step": optimizer_step,
            "mean_microbatch_l1": float(np.mean(recent_losses[-8:])),
            "gpu_peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
            ),
            "gpu_peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 1024**3
            ),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if optimizer_step in CHECKPOINT_STEPS:
            checkpoint_dir = (
                args.output_dir / f"openvla_lora_step_{optimizer_step:06d}"
            )
            model.save_pretrained(checkpoint_dir)
            torch.save(
                {
                    "action_header": action_head.state_dict(),
                    "scale_header": scale_head.state_dict(),
                },
                checkpoint_dir / "openvla_headers.pt",
            )
            (checkpoint_dir / "training_state.json").write_text(
                json.dumps({**plan, **row}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    result = {
        **plan,
        "completed_at": datetime.now().astimezone().isoformat(),
        "final_mean_microbatch_l1": float(np.mean(recent_losses[-8:])),
        "n_trainable_parameters": sum(
            parameter.numel() for parameter in trainable
        ),
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "completed_optimizer_steps": MAX_STEPS,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
