# Vanilla LOOP Spatial bounded smoke — 2026-07-20

## Outcome

Run `run6` completed successfully on one RTX 4090D and exited cleanly with
`[info] Finished training`. This validates the OpenVLA-OFT + RIPT vanilla LOOP
execution path through model loading, official LIBERO dataset loading,
environment interaction, K-sampling, leave-one-out reward construction, PPO
forward/backward plumbing, metric reporting, cleanup, and distributed teardown.

This was a connectivity smoke, not a learning experiment. Both deliberately
short rollouts failed, so reward, advantage, loss, and gradient norms were zero.

## Bounded configuration

```text
suite:                 LIBERO_SPATIAL (official LIBERO, isolated from Plus)
task:                  pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
checkpoint:            runtime/openvla-oft-spatial-smoke
scale header:          LIBERO_SPATIAL_scale_header.pth
training steps:        1
demo batch size:       1
RLOO K:                2
rollouts per env:      2
parallel envs:         1, in-process DummyVectorEnv
max episode length:    10 policy steps (+10 stabilization steps upstream)
PPO epochs:            1
W&B:                   disabled
periodic evaluation:   disabled
checkpoint save:       disabled for this one-step run
physical GPU:          7
```

## Observed milestones

```text
Starting training loop
eval_loop_num: 2
rank 0 finished generate rollout episodes in 14.68 seconds
PPO Batch: 2/2
PPO Epochs: 1/1
Training with 1 GPUs: 1/1 [16.63 seconds]
Closed environment 8
[info] Finished training
```

Final metrics:

```text
pg_clipfrac_stats: 0.0
pg_loss_stats: 0.0
pg_ratio_stats: 1.0002521276474
mean_scores: 0.0
mean_advantage: 0.0
mean_rlhf_reward: 0.0
rollout_checked: 1
gradient_norm_model_stats: 0.0
gradient_norm_header_stats: 0.0
non_zero_adv_ratio: 0.0
task success rate: 0.0
```

The zero update is expected because K=2 produced rewards `[0, 0]`; leave-one-out
advantages are therefore both zero. A later learning-validation run needs a
realistic horizon and enough initializations/rollouts to obtain mixed rewards.

## Isolated compatibility work required by pinned upstream

- `PYTHONNOUSERSITE=1` prevents user-site PEFT/Transformers from shadowing the
  pinned conda environment.
- Official LIBERO is selected with an isolated `PYTHONPATH` and
  `.libero_official/config.yaml`; LIBERO-Plus remains available for robustness
  evaluation.
- `task.dataset.suite_name=.` avoids RIPT duplicating the suite component that
  official `get_task_demonstration()` already returns.
- `RLOptimizerOpenVLAOFTCompat` routes upstream's misplaced
  `enable_rollout_stats_tracking` argument to the rollout generator.
- `InProcessOpenVLAOFTLiberoRunner` avoids torchrun rendezvous variables leaking
  into a spawned one-environment worker.
- A runtime checkpoint mirror confines OpenVLA's automatic `config.json` writes
  while referencing the original 15 GB weights through symlinks.

No RIPT, OpenVLA-OFT, official LIBERO, LIBERO-Plus, or original model weight
source file was modified by these compatibility changes.
