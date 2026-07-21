#!/usr/bin/env python3
"""Read-only integrity checks for the RIPT LIBERO Spatial training assets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py


SCALE_HEADER = Path(
    "/home/imc/models/ra-loop/ript-vla/openvla_oft/scale_header/"
    "LIBERO_SPATIAL_scale_header.pth"
)
SCALE_SIZE = 335_805_208
SCALE_SHA256 = "388c491761523d41a28f62b2262d52611c98c1cb89dae09a7290dc1a1c1097b8"

DATA_DIR = Path("/home/imc/data/ra-loop/libero-datasets/libero_spatial")
EXPECTED_SIZES = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5": 508_779_600,
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5": 589_630_943,
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5": 748_271_833,
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5": 632_345_992,
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5": 597_677_074,
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5": 671_583_385,
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5": 507_189_857,
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5": 581_087_722,
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5": 711_715_594,
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5": 688_768_764,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    assert SCALE_HEADER.stat().st_size == SCALE_SIZE, "scale header size mismatch"
    actual_hash = sha256(SCALE_HEADER)
    assert actual_hash == SCALE_SHA256, "scale header SHA-256 mismatch"
    print(f"scale_header OK size={SCALE_SIZE} sha256={actual_hash}")

    actual_names = {path.name for path in DATA_DIR.glob("*.hdf5")}
    assert actual_names == set(EXPECTED_SIZES), "HDF5 file set mismatch"

    total_bytes = 0
    total_demos = 0
    total_steps = 0
    for name, expected_size in sorted(EXPECTED_SIZES.items()):
        path = DATA_DIR / name
        actual_size = path.stat().st_size
        assert actual_size == expected_size, f"size mismatch: {name}"
        total_bytes += actual_size

        with h5py.File(path, "r") as handle:
            assert "data" in handle, f"missing /data: {name}"
            demos = sorted(handle["data"].keys())
            assert demos, f"no demos: {name}"
            file_steps = 0
            for demo_name in demos:
                demo = handle["data"][demo_name]
                assert "actions" in demo, f"missing actions: {name}/{demo_name}"
                assert "obs" in demo, f"missing obs: {name}/{demo_name}"
                assert len(demo["actions"]) > 0, f"empty actions: {name}/{demo_name}"
                file_steps += len(demo["actions"])
            total_demos += len(demos)
            total_steps += file_steps
        print(f"hdf5 OK demos={len(demos):3d} steps={file_steps:6d} size={actual_size:10d} {name}")

    print(
        f"ALL_OK files={len(EXPECTED_SIZES)} demos={total_demos} "
        f"steps={total_steps} bytes={total_bytes}"
    )


if __name__ == "__main__":
    main()
