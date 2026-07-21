# RA-LOOP minimal GPU launcher review — 2026-07-20

## Outcome

The separate launcher `train/ra_loop_spatial_connectivity_smoke.sh` is ready for
the first one-step RA-LOOP GPU connectivity smoke. It was only printed and
statically reviewed in this step; no GPU status query, CUDA initialization, tmux
session, environment, model, or training process was started.

## Safe interface

Default behavior is non-executing:

```bash
bash train/ra_loop_spatial_connectivity_smoke.sh
bash train/ra_loop_spatial_connectivity_smoke.sh --print-command
```

Both only print the final command. Execution requires an explicit physical GPU:

```bash
bash train/ra_loop_spatial_connectivity_smoke.sh --run <GPU_ID>
```

The launcher rejects malformed arguments and GPU IDs outside 0 through 7.

## Pre-run safety checks

Before exposing a CUDA device, the run path checks that all required code,
checkpoint metadata, scale header, and Spatial dataset directories exist. It
then queries only the selected physical GPU and refuses to start when any of the
following is true:

```text
memory used > 1024 MiB
GPU utilization > 10%
temperature > 75 C
```

Only after those checks pass does it set `CUDA_VISIBLE_DEVICES`. Official LIBERO
is isolated, OSMesa is selected, user-site packages are disabled, caches use
`/tmp`, W&B is disabled, and the process uses one standalone torchrun worker.

## Fixed bounded configuration

```text
task:                  one verified LIBERO Spatial task
training steps:        1
RLOO K:                4
pairs:                 2 anchor/Robot-init pairs
parallel environments: 1, in-process
episode horizon:       10
Robot-init strength:   0.001 rad
lambda_recovery:       0.5
PPO epochs:            1
gradient accumulation: 1
Laplace scale factor:  2.0
periodic evaluation:   disabled
W&B:                   disabled
checkpoint saving:     disabled for this one-step run
timeout:               none
```

The 45 substantive Hydra overrides match the passed CPU preflight exactly. Only
the experiment label and output directory differ. `bash -n` passed, optional
shellcheck produced no warning, and the default invocation printed without
performing any system or GPU operation.

## Expected interpretation

This is a connectivity smoke, not a learning experiment. Horizon 10 is likely to
produce four failures and zero advantage, as the earlier vanilla smoke did. The
gate is instead:

- real joint layout resolution and two bounded Robot-init states;
- four exactly-once rollouts with correct pair metadata;
- recovery reward/metrics present and semantically separated;
- upstream PPO completes once and exits cleanly;
- no checkpoint is written and the selected GPU is released.

After explicit confirmation, it should be started inside a shared tmux window on
an idle GPU so the user can attach and inspect progress without polling.
