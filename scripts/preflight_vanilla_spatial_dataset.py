#!/usr/bin/env python3
"""CPU-only construction check for the bounded Spatial smoke dataset."""

from __future__ import annotations

import argparse

import torch

import ript.utils.libero_utils as libero_utils


TASK_NAME = "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.demos <= 50:
        parser.error("--demos must be in [1, 50]")

    # Dataset construction normally downloads/loads CLIP only to embed one fixed task
    # string. Replace that unrelated step so this remains a deterministic CPU I/O test.
    libero_utils.get_task_embs = lambda _format, descriptions: torch.zeros(
        (len(descriptions), 512), dtype=torch.float32
    )

    shape_meta = {
        "action_dim": 7,
        "observation": {
            "rgb": {
                "agentview_rgb": [3, 128, 128],
                "eye_in_hand_rgb": [3, 128, 128],
            },
            "lowdim": {
                "joint_states": 7,
                "ee_pos": 3,
                "gripper_states": 2,
            },
        },
        "task": {"type": "vector", "dim": 512},
    }

    dataset = libero_utils.build_dataset(
        data_prefix="/home/imc/data/ra-loop/libero-datasets",
        suite_name=".",
        benchmark_name="LIBERO_SPATIAL",
        mode="all",
        seq_len=600,
        frame_stack=1,
        shape_meta=shape_meta,
        n_demos=args.demos,
        load_next_obs=True,
        get_pad_mask=True,
        task_names_to_use=[TASK_NAME],
        obs_seq_len=1,
        load_obs=True,
        task_embedding_format="clip",
        pad_seq_length=False,
        load_state=True,
    )
    assert len(dataset) > 0, "constructed dataset is empty"
    sample = dataset[0]
    assert sample["actions"].shape[-1] == 7
    assert "init_state" in sample
    assert sample["task_id"] == 0
    print(
        "DATASET_PREFLIGHT_OK",
        f"demos={args.demos}",
        f"sequences={len(dataset)}",
        f"action_shape={tuple(sample['actions'].shape)}",
        f"task_id={sample['task_id']}",
    )


if __name__ == "__main__":
    main()
