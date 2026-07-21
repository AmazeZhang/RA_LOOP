# RA-LOOP semantic audit — 2026-07-20

## Decision

The existing `code/ra_optimizer.py` must not be used for GPU training. The first
RA-LOOP pilot should target the project's primary Robot-init weakness with a
**recovery-only reward**. Action-consistency reward must initially be disabled
for Robot-init because changing the robot configuration changes the physical
state and can require a different correct action.

Consistency can later be tested separately for observation-only perturbations
such as camera or lighting, once those perturbations are genuinely implemented
with identical underlying physical state.

No GPU, model, simulator, checkpoint, or dataset write was used in this audit.

## Existing implementation blockers

1. `code.ra_optimizer` is not a safe import target: `code/` is not a package and
   collides with Python's standard-library `code` module. New implementation
   must live under the existing unique `ra_loop` package.
2. The current launcher expects an obsolete patch to the upstream entry point.
   The pinned RIPT commit already exposes Hydra rollout-generator and optimizer
   factories, so integration must use project-local Hydra `_target_` overrides
   and leave upstream untouched.
3. Camera and light groups currently change nothing but are labelled perturbed,
   allowing a false recovery bonus.
4. Group 0 is assigned a Robot-init perturbation but is subsequently labelled as
   the unperturbed anchor. After warmup this makes the anchor label false.
5. Consistency is computed within repeated samples of one perturbation group,
   not between an anchor and perturbations. It therefore does not implement the
   stated cross-perturbation hypothesis.
6. The reward uses sampled rollout actions aligned only by time index. Once
   trajectories diverge, the same time index need not represent the same state;
   penalizing different actions can punish correct state-dependent behavior.
7. The first optimization step has zero perturbation strength because both
   perturbation and reward warmup start at step 0, while non-anchor episodes can
   still be labelled perturbed and receive recovery reward.
8. The custom optimizer runs and caches a rollout before calling the parent
   optimizer. This coupling needs targeted tests for exactly-once rollout,
   valid-mask handling, reward lookup order, cleanup, and exception restoration.
9. The current `train_ra_loop.yaml` is stale relative to the pinned upstream
   factory/config structure and is still configured for LIBERO-Long rather than
   the verified bounded Spatial path.

## State-vector evidence

A read-only inspection of the official Spatial demonstration used by the probe
showed `states.shape == (115, 92)`. Its first values are:

```text
[0.25, 0.00515, -0.15058, 0.00517, -2.43120,
 -0.00142, 2.22734, 0.79991, 0.03406, -0.03405, ...]
```

The existing implementation perturbs `state[:7]` as seven arm joints and starts
object poses at index 9. The observed state is inconsistent with both assumptions:
the plausible seven Panda joint values occupy indices 1 through 7, indices 8 and
9 are the two gripper joints, and the following state layout is model-dependent.
Hard-coded slices can therefore move the wrong degrees of freedom.

The refactor must resolve joint qpos addresses by simulator/model joint names and
validate limits, shape, finiteness, and non-target invariance. It must fail closed
if those addresses cannot be resolved.

## Metric audit from the vanilla probe

- `gradient_norm_header_stats=85.5` is the **pre-clipping total norm**. The pinned
  upstream records the return value of `torch.nn.utils.clip_grad_norm_`, whose
  return contract is the total norm before clipping. The configured 1.0 clipping
  was applied, so 85.5 is not the norm used by the optimizer step; it does show
  that the header update was clipped very strongly.
- `pg_clipfrac_stats=0.3961` is not a crash condition. It is the fraction of PPO
  elements for which the clipped surrogate exceeded the unclipped surrogate.
  In this probe, gradient accumulation was 1 and the eight episodes were updated
  sequentially in shuffled order, so later episodes were evaluated after earlier
  optimizer steps against rollout-time reference log probabilities. A high mean
  clip fraction is therefore plausible but warns against carrying this update
  schedule unchanged into a longer run.

## Implementation contract for the next step

The first CPU-tested RA slice will be deliberately narrow:

1. Put all new code in `ra_loop`, with no upstream or system file edits.
2. Support `none` and `robot_init` only; reject camera, light, and layout instead
   of silently treating them as implemented.
3. Build an explicit rollout plan containing an actual unperturbed anchor plus
   Robot-init episodes, with `perturb_type`, `strength`, `seed`, `is_perturbed`,
   and pair/group identifiers derived from what was actually applied.
4. Apply Robot-init noise only through validated named joint qpos addresses,
   enforce joint limits, and prove all non-target state entries are unchanged.
5. Use

   ```text
   R_total = R_success + lambda_recovery * I(robot_init_applied and success)
   ```

   for the first pilot. Set `lambda_consistency=0` and report success/recovery
   separately for anchor and perturbed episodes.
6. Preserve exact vanilla reward behavior when perturbation or recovery weight is
   disabled.
7. Unit-test determinism, anchor correctness, unsupported-type rejection,
   address validation, clipping/limits, no-op backward compatibility, reward
   values, valid-mask behavior, and exactly-once rollout before Hydra composition.

Only after those CPU gates pass should a one-task, one-step Spatial Robot-init GPU
smoke be proposed. No full training is authorized by this document.
