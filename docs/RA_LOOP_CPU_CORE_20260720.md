# RA-LOOP Robot-init CPU core — 2026-07-20

## Outcome

The first recovery-only RA-LOOP core is implemented under `ra_loop/robustness.py`
and passes all CPU tests. It is deliberately independent of RIPT, LIBERO,
MuJoCo, torch, Hydra, and CUDA. It is not connected to the training entry point
yet and cannot start or alter training by itself.

## Implemented contract

- Interleaved anchor/Robot-init rollout pairs with deterministic independent
  seeds and explicit pair identifiers.
- Positive perturbation strength is mandatory; a zero-strength episode cannot
  be labelled perturbed.
- Panda arm qpos addresses are resolved from the simulator's seven named joints
  (`robot0_joint1` through `robot0_joint7`), never from a hard-coded state slice.
- Missing, duplicate, non-scalar, unlimited, or invalid named joints fail closed.
- Gaussian Robot-init noise is deterministic, clipped to simulator joint limits,
  and changes only resolved arm-joint entries; input arrays are not mutated.
- Only `none` and `robot_init` are accepted. Camera, light, layout, or arbitrary
  labels are rejected rather than treated as completed perturbations.
- Materialized episode metadata reports what was actually applied.
- Recovery reward is

  ```text
  R_total = R_success + lambda_recovery * I(robot_init_applied and success)
  ```

  and reports anchor/perturbed success separately using the valid mask.
- `lambda_recovery=0` preserves base rewards exactly.

## Verification

Targeted core suite:

```text
12 passed in 0.08s
```

Full project test suite in the pinned `ript_vla_openvla_oft` environment, with
official LIBERO explicitly selected and CUDA hidden:

```text
15 passed in 20.18s
```

The first full-suite attempt exposed a read-only Numba cache path and the second
exposed the shell's default LIBERO-Plus path. Neither was a product-code failure.
The successful regression used the same isolated official-LIBERO, `/tmp` cache,
and `PYTHONNOUSERSITE=1` settings as the verified training launcher. No Conda or
upstream file was modified.

## Boundary of this step

This result initially validated pure planning, perturbation, metadata, and reward
logic against a fake MuJoCo model interface. The subsequent live official-LIBERO
check found the required flattened-state time offset, added it explicitly, and
passed with real qpos addresses and limits; see
`docs/RA_LOOP_LIVE_JOINT_VALIDATION_20260720.md`. Exactly-once rollout integration
with RIPT remains the next CPU-only gate before any RA-LOOP GPU smoke.
