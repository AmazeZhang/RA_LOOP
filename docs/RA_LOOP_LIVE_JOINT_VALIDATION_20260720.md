# RA-LOOP live LIBERO joint validation — 2026-07-20

## Outcome

The Robot-init named-joint mapping was validated against a live official LIBERO
Spatial environment using CPU rendering through OSMesa. No OpenVLA model,
rollout, CUDA device, checkpoint, or dataset write was involved.

The validation also found and fixed an important representation boundary before
training integration: MuJoCo joint `qposadr` is relative to qpos, while LIBERO's
initialization vector is `MjSimState.flatten()` and starts with simulation time.
The mapping into the 92-dimensional LIBERO state therefore requires an explicit
offset of one.

## Live result

Task:

```text
pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
```

Observed mapping:

```text
state shape:             (92,)
MuJoCo qpos addresses:   [0, 1, 2, 3, 4, 5, 6]
LIBERO state indices:    [1, 2, 3, 4, 5, 6, 7]
changed by perturbation: [1, 2, 3, 4, 5, 6, 7]
maximum applied noise:   0.0017903548335325092 rad
simulator round-trip:    passed
```

Live Panda joint limits:

```text
robot0_joint1  [-2.8973,  2.8973]
robot0_joint2  [-1.7628,  1.7628]
robot0_joint3  [-2.8973,  2.8973]
robot0_joint4  [-3.0718, -0.0698]
robot0_joint5  [-2.8973,  2.8973]
robot0_joint6  [-0.0175,  3.7525]
robot0_joint7  [-2.8973,  2.8973]
```

With a deterministic `0.001 rad` Gaussian perturbation, all seven named arm
joints changed, no other flattened-state entry changed, the input array remained
unchanged, and official LIBERO/MuJoCo accepted the bounded state and returned a
finite state of the same shape.

## Verification assets

- Live validator: `scripts/validate_live_robot_init_layout.py`
- Core mapping and perturbation: `ra_loop/robustness.py`
- Unit tests now explicitly cover qpos-only offset 0 versus flattened-state
  offset 1.

After the offset correction, targeted core tests passed 13/13 and the isolated
official-LIBERO project regression passed 16/16.

## Next boundary

The state mapping gate is passed. The core is still not connected to RIPT. The
next CPU-only step is an exactly-once rollout adapter that injects the paired
states and metadata without copying or patching the upstream training entry
point. It must be tested with fake rollouts before any GPU process is proposed.
