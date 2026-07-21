# RA-LOOP recovery reward/optimizer adapter — 2026-07-20

## Outcome

The recovery-only reward and optimizer adapter are implemented in
`ra_loop/ript_recovery.py`. CPU fake-PPO tests prove that the pinned upstream PPO
optimizer and rollout generator are each called exactly once, while recovery
metrics are derived afterward from that same captured rollout batch.

This step did not compose Hydra, create an environment, load a model, or use a
GPU.

## Reward semantics

`RobotInitRecoveryReward` returns:

```text
R_total = I(success) + lambda_recovery * I(robot_init_applied and success)
```

The core metadata validator rejects a false perturbation label, an unapplied
Robot-init label, or an unsupported perturbation type. A successful anchor has
reward 1, a successful applied Robot-init episode has reward
`1 + lambda_recovery`, and a failed episode has reward 0.

## Exactly-once optimizer behavior

`RobotInitRecoveryOptimizer` temporarily wraps the rollout generator, then calls
the pinned upstream `RLOptimizerOpenVLAOFT.optimize()` exactly once. The wrapper:

- captures exactly one `(episodes, task_ids, valid_mask, samples_checked)` tuple;
- rejects a second rollout request or malformed result;
- restores the generator method in `finally`, including PPO failure paths;
- delegates all advantage, PPO loss, backward, clipping, optimizer-step, and
  cleanup logic to upstream unchanged.

Metric correction is currently intentionally gated to one GPU. This matches the
first bounded RA pilot and prevents silently incorrect cross-rank aggregation.

## Metric semantics

The upstream optimizer treats reward output as `all_scores`, so shaped rewards
would otherwise inflate fields named score or task success. The adapter corrects
the same captured valid episodes after PPO:

```text
mean_scores:                 true binary success, valid episodes only
mean_R_success:              true binary success, valid episodes only
mean_R_recovery:             successful applied perturbation indicator
mean_R_total:                augmented reward, valid episodes only
mean_rlhf_reward:            same augmented reward as mean_R_total
anchor_success_rate:         binary success on valid anchors
perturbed_success_rate:      binary success on valid Robot-init episodes
valid_anchor_count:          valid anchor count
valid_perturbed_count:       valid perturbed count
rl_train_succeess_rate/*:    corrected binary task success
```

Padding is excluded from every corrected statistic. Advantage and PPO metrics
remain those actually computed by upstream from augmented rewards.

## Verification

The reward/optimizer tests cover reward truth table, exactly-one PPO/rollout
call, padding exclusion, metric correction, generator restoration after PPO
failure, and rejection of a second rollout request.

```text
RIPT recovery tests: 10 passed
full project tests:  26 passed in 20.61s
py_compile:          passed
```

The regression used the pinned `ript_vla_openvla_oft` environment with official
LIBERO isolated, CUDA hidden, and caches under `/tmp`. No Conda, upstream,
checkpoint, or dataset file was modified.

## Next boundary

The rollout, reward, and optimizer adapters are now individually CPU-tested.
The next step is a CPU-only Hydra compose using the verified bounded Spatial
profile, followed by factory instantiation with environment creation disabled.
That step must prove all targets and values resolve correctly before a GPU smoke
can be proposed.
