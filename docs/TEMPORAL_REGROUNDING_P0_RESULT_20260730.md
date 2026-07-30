# Temporal Counterfactual Regrounding P0 — result

Date: 2026-07-30

Preregistration:
`docs/TEMPORAL_REGROUNDING_P0_PREREG_20260730.md`

Machine-readable result:
`ci_grpo/artifacts/p1_temporal_regrounding_openvla/result.json`

## Outcome

Overall decision: **inconclusive because late matched controls failed; do not
train.** The valid early subgroup is a clear no-go for early temporal inertia.

The fixed 21-rollout budget completed:

- three original full trajectories succeeded;
- six checkpoints were restored with exact MuJoCo state equality;
- the actual policy-start state after the upstream runner's one deterministic
  dummy step was byte-identical across all three instruction variants;
- none of the three goals was true at any checkpoint;
- 18 restored continuations completed.

At every one-third checkpoint:

- all 3/3 matched original-instruction controls succeeded;
- all 6/6 revised-instruction continuations succeeded for their new goal;
- revised-goal success was 100%;
- original-goal inertia was 0%.

This falsifies the proposed early-execution inertia failure on this
backbone/group. OpenVLA can redirect among plate, stove, and cabinet after it
has already begun acting.

At every two-thirds checkpoint:

- all 3/3 matched original-instruction controls failed;
- all 6/6 revised-instruction continuations also failed;
- every terminal goal vector was all-false.

These rows cannot diagnose language use. The matched controls show that merely
restoring a late state and restarting action-chunk inference changes the
rollout enough to lose the task. The checkpoints were selected by primitive
control-step fraction rather than policy action-chunk boundary, so discarded
queue context is a plausible technical cause. No rollout completed the old
goal, hence there is no positive goal-inertia signal.

Primary machine-readable metrics:

- `matched_control_success_rate = 0.5`;
- `n_valid_controls = 3/6`;
- `n_valid_switches = 6/12`;
- valid `switch_success_rate = 1.0`;
- valid `original_goal_inertia_rate = 0.0`;
- `n_distinct_inertia_checkpoints = 0`;
- `temporal_inertia_candidate = false`.

## Decision

Do not start temporal CI-GRPO training. The only scientifically valid subgroup
shows perfect redirection, while the late subgroup fails its control gate.
Under the preregistered bounded-search discipline, no second checkpoint-search
or timing sweep is launched.

The reusable result is methodological: evaluating mid-trajectory instruction
revision for chunked VLA policies must preserve or explicitly reset action
queue phase, and matched original-instruction restoration controls are
mandatory. A future project may study chunk-state/history dependence itself,
but that is a different hypothesis from language inertia.
