# RA-LOOP exactly-once RIPT rollout adapter — 2026-07-20

## Outcome

A thin project-local RIPT adapter is implemented in `ra_loop/ript_recovery.py`.
It reuses the pinned upstream `RolloutGenerator.generate_rollouts()` exactly
once and does not copy, patch, or edit the upstream rollout loop.

This is a CPU fake-rollout integration gate. It has not been selected by Hydra,
has not loaded a policy, and has not used a GPU.

## Mechanism

During the single upstream call, the adapter temporarily wraps
`env_runner.run_policy_in_env`:

1. Validate one in-process environment and require `rollouts_per_env == RLOO K`.
2. Require an even K and K identical finite base states.
3. Resolve the live Panda joint layout from the directly inspectable worker.
4. Replace the K copies with interleaved anchor/Robot-init pairs.
5. Call the original environment method once.
6. Tag each returned episode with pair, type, strength, seed, applied flag, and
   actual joint noise.
7. Restore the original environment method in `finally`.

Mixed validation initializations, random initialization, subprocess workers,
parallel environments, wrong rollout counts, non-identical bases, unsupported
metadata collisions, and too few/many returned episodes fail closed.

The perturbation seed includes optimizer step, distributed rank, and base-state
call index, while remaining deterministic for a fixed run.

## Verification

Six adapter tests passed, covering:

- one parent call and one environment call;
- correct anchor/Robot-init ordering and pair metadata;
- only named arm joint state entries changing;
- original method restoration after success and parent exception;
- rejection of non-identical base states and unsafe runner configuration;
- detection of incomplete environment episode output.

Results:

```text
adapter tests:       6 passed
full project tests: 22 passed in 19.85s
py_compile:          passed
```

The regression ran in the pinned `ript_vla_openvla_oft` environment with
official LIBERO isolated, CUDA hidden, and caches under `/tmp`. No Conda,
upstream, model, or dataset file was modified.

## Next boundary

The adapter currently injects rollout states and metadata only. Recovery reward
still needs a thin optimizer/reward integration that calls upstream PPO exactly
once, preserves true task-success metrics rather than reporting augmented reward
as success, and respects valid masks. After those CPU tests pass, the complete
Hydra configuration can be composed without loading a model. GPU remains gated.
