# Vanilla LOOP Spatial mixed-reward probe — 2026-07-20

## Outcome

The one-step `mixed_reward_probe` completed successfully on physical GPU 7 and
exited cleanly with `[info] Finished training`. Its eight rollout rewards were

```text
[1, 0, 1, 1, 0, 1, 0, 0]
```

This is the first bounded run that validates an actual vanilla LOOP learning
signal in the local OpenVLA-OFT + RIPT path: mixed binary rewards produced
non-zero per-sample leave-one-out advantages, a non-zero PPO loss, and non-zero
gradients. The mean advantage is approximately zero because leave-one-out
advantages are centered; it is not evidence of a zero update.

This remains a one-step mechanism probe. It does not establish policy
improvement or reproduce a paper-level training result.

## Configuration

```text
suite:                 LIBERO_SPATIAL (official LIBERO, isolated from Plus)
task:                  pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
checkpoint:            runtime/openvla-oft-spatial-smoke
training steps:        1
RLOO K:                8
rollouts per env:      8, sequential
parallel envs:         1, in-process
max episode length:    220 policy steps
scale factor:          5
PPO epochs:            1
W&B:                   disabled
periodic evaluation:   disabled
checkpoint save:       disabled
physical GPU:          7
```

## Timing and results

```text
rollout generation:    108.83 s
PPO update:             57.67 s (8/8 batches, 1 epoch)
total training step:   166.85 s
successes:             4/8
mean reward:           0.5
```

Final metrics:

```text
pg_clipfrac_stats:             0.396075576543808
pg_loss_stats:                 0.002401747740805149
pg_ratio_stats:                0.9605819582939148
mean_scores:                   0.5
mean_advantage:               -2.9802322387695312e-08
mean_rlhf_reward:              0.5
rollout_checked:               1
gradient_norm_model_stats:     1.99267578125
gradient_norm_header_stats:    85.5
non_zero_adv_ratio:            1.0
task success rate:             0.5
```

The action-header gradient norm and PPO clip fraction were audited after this
run. The gradient metric is the total norm returned by `clip_grad_norm_` before
the configured 1.0 clipping is applied. The clip fraction is the fraction of PPO
elements selecting the clipped surrogate; it is not itself an error, but 0.3961
is a warning against scaling this sequential update schedule without further
control. See `docs/RA_LOOP_SEMANTIC_AUDIT_20260720.md`.

## Safety and artifacts

- The process exited normally and physical GPU 7 was released.
- Checkpoint saving was disabled, so no trained model artifact was produced.
- Original checkpoint weights were not modified. Only the isolated runtime
  mirror's mutable metadata may be updated by OpenVLA loading.
- No longer vanilla training should be launched until the gradient/clip metrics
  are audited and the RA-LOOP objective semantics are implemented and tested.

## Decision

The vanilla pipeline gate is passed: connectivity and non-zero learning signal
are both demonstrated. The next step is the RA-LOOP semantic refactor and CPU
unit tests, performed incrementally before any longer GPU training.
