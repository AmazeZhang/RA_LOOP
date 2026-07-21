# RA-LOOP bounded Spatial Hydra preflight — 2026-07-20

## Outcome

The complete recovery-only RA-LOOP configuration composed successfully from the
pinned upstream Spatial config, and all three project-local factories
instantiated with environment creation disabled. This closes the CPU
configuration gate before the first GPU smoke.

The preflight script intentionally has no GPU execution mode.

## Resolved bounded profile

```text
suite/task:             LIBERO_SPATIAL / one selected task
training steps:         1
RLOO K:                 4 (2 anchor/Robot-init pairs)
rollouts per env:       4
parallel envs:          1, in-process target
episode horizon:        10
Robot-init strength:    0.001 rad
perturbation seed:      20260720
lambda_recovery:        0.5
lambda_consistency:     absent/disabled
PPO epochs:             1
gradient accumulation:  1
gradient clipping:      model 1.0 / header 1.0
Laplace scale factor:   2.0
periodic evaluation:    disabled
checkpoint save:        effectively disabled for the one-step smoke
W&B:                    disabled
```

Resolved targets:

```text
env runner:  ra_loop.ript_compat.InProcessOpenVLAOFTLiberoRunner
rollout:     ra_loop.ript_recovery.RobotInitRecoveryRolloutGenerator
reward:      ra_loop.ript_recovery.RobotInitRecoveryReward
optimizer:   ra_loop.ript_recovery.RobotInitRecoveryOptimizer
```

## Factory gate

The preflight instantiated:

- `RobotInitRecoveryReward(lambda_recovery=0.5)`;
- the partial rollout factory as a `RobotInitRecoveryRolloutGenerator` using a
  no-environment runner and `create_env=False`;
- the partial optimizer factory as a `RobotInitRecoveryOptimizer`;
- upstream's misplaced `enable_rollout_stats_tracking=True` flag and verified
  that it reached the rollout generator.

No dataset, official LIBERO simulator, policy, checkpoint tensor, or CUDA model
was opened. The upstream rollout counter was created under `/tmp` for constructor
compatibility and cleaned by the preflight.

TensorFlow printed a CUDA driver probe failure because `CUDA_VISIBLE_DEVICES` was
explicitly empty. This confirms no CUDA device was available to the process; it
is not a training error and no GPU was occupied.

## Verification

```text
Hydra --cfg job --resolve:       passed
factory instantiation:           passed
environment creation:            false
full project regression:         27 passed in 20.23s
py_compile:                      passed
```

Assets:

- `train/ra_loop_spatial_preflight.sh`
- `scripts/preflight_ra_hydra_factories.py`

No Conda, upstream, model, checkpoint, or dataset file was modified.

## Next boundary

The next step is to create a separate explicit GPU launcher from this exact
profile, inspect its command without executing it, and only then launch a
one-task, one-step, K=4, horizon-10 RA-LOOP connectivity smoke after user
confirmation. It should validate real metadata/reward/PPO plumbing, not learning
quality; horizon 10 is expected to produce failures and may yield zero advantage.
