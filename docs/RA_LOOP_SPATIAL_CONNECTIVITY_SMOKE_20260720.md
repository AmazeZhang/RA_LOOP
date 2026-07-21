# RA-LOOP Spatial connectivity smoke — 2026-07-20

## Outcome

The first real recovery-only RA-LOOP run completed successfully on physical GPU
7 and exited with `[info] Finished training` and tmux marker `[RA_LOOP_EXIT] 0`.
GPU 7 returned to 18 MiB used and 0% utilization afterward.

This passes the end-to-end connectivity gate for live named-joint Robot-init
pairing, episode metadata, recovery reward, exactly-once upstream PPO, corrected
metrics, cleanup, and process exit. It is not a learning experiment.

## Execution

```text
tmux session:           ra_loop_spatial_smoke
tmux window:            run1
physical GPU:           7
log:                    logs/ra_loop_spatial_connectivity_smoke_20260720/run1.log
exit code:              0
```

The tmux session remains available at a shell prompt:

```bash
tmux attach -t ra_loop_spatial_smoke
```

## Bounded configuration

```text
suite/task:             one LIBERO Spatial task
training steps:         1
RLOO K:                 4
pairs:                  2 anchor/Robot-init pairs
parallel envs:          1, in-process
episode horizon:        10
Robot-init strength:    0.001 rad
perturbation seed:      20260720
lambda_recovery:        0.5
PPO epochs:             1
gradient accumulation:  1
Laplace scale factor:   2.0
W&B/eval/save:          disabled
```

## Observed milestones

```text
model allocated memory:       15.75 GB
model reserved memory:        17.12 GB
environment creation:          4.43 s
four rollout generation:      44.65 s
PPO update:                    2.99 s
one training step:            48.01 s
Closed environment 8
[info] Finished training
[RA_LOOP_EXIT] 0
```

The runtime mirror created a new mutable metadata backup
`config.json.back.20260720_231436`; original checkpoint weight targets were not
modified.

## Results

All four horizon-10 rollouts failed. This matches the earlier vanilla h10 smoke
and produces a deliberately zero learning update:

```text
mean_scores:                  0.0
mean_R_success:               0.0
mean_R_recovery:              0.0
mean_R_total:                 0.0
mean_advantage:               0.0
non_zero_adv_ratio:           0.0
pg_loss_stats:                0.0
gradient_norm_model_stats:    0.0
gradient_norm_header_stats:   0.0
pg_ratio_stats:               1.0001912117004395
pg_clipfrac_stats:            0.0
```

RA-specific structure was present and correctly separated:

```text
valid_anchor_count:           2.0
valid_perturbed_count:        2.0
anchor_success_rate:          0.0
perturbed_success_rate:       0.0
lambda_r_effective:           0.5
```

The run output directory contained no files or checkpoint directories. The only
persistent run artifact is the explicit project log and the runtime metadata
backup.

## Decision

The RA-LOOP connectivity gate is passed. The `0.001 rad` perturbation was chosen
only for safety and should not be treated as a meaningful robustness-training
distribution. Before a realistic h220/K8 learning-signal probe, the next step is
a CPU-only calibration of actual LIBERO-Plus Robot-init state shifts and safe
Panda joint limits. That calibration should determine the pilot perturbation
scale instead of guessing it.
