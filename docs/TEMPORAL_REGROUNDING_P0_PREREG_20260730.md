# Temporal Counterfactual Regrounding P0 — preregistration

Date: 2026-07-30

## Question

The completed CI-GRPO P0 showed that OpenVLA-OFT-Goal is perfectly sensitive
to reachable conflicting instructions when the instruction is supplied at the
initial state. This experiment asks a different question:

> After a policy has begun executing one valid goal, does it follow a revised
> instruction from the exact same intermediate physical state, or continue the
> original plan?

This is a diagnostic screen. It does not authorize training.

## Fixed task group and state

The mutually exclusive bowl-destination group is fixed before execution:

1. `put_the_bowl_on_the_plate`
2. `put_the_bowl_on_the_stove`
3. `put_the_bowl_on_top_of_the_cabinet`

All runs use official LIBERO-Goal init-state index 0 and one live environment.
Physical GPU 0 is forbidden; physical GPU 7 is used.

## Rollout matrix

For each of the three original instructions:

1. run one successful baseline trajectory and capture every post-settling
   simulator state;
2. select checkpoints at one-third and two-thirds of the baseline control
   trajectory;
3. restore each checkpoint exactly;
4. run one matched continuation with the original instruction and two
   counterfactual continuations with the other instructions;
5. cross-score every terminal state with all three official goal predicates.

Maximum budget: 3 baseline rollouts plus 18 restored continuations, for 21
rollouts total. No seed, checkpoint, task, prompt, or time-step search follows
this matrix.

## Validity checks

A checkpoint enters the primary analysis only if:

- none of the three destination goals is already true at the checkpoint;
- the simulator state after restoration is exactly equal to the saved state;
- because the pinned upstream runner initializes observations through one
  standard deterministic dummy action, the actual policy-start state after
  that action is byte-identical across all three instruction variants;
- the matched original-instruction continuation succeeds;
- the three goal predicates remain empirically mutually exclusive.

A failed revised-instruction rollout is evidence of temporal goal inertia only
when its terminal state satisfies the original goal. Failure with no terminal
goal is reported separately and is not sufficient to claim language inertia.
Physical reachability of a revised goal after substantial object motion is a
remaining confound; any positive signal is provisional until an independent
reachability audit passes.

## Metrics and decision rules

For valid checkpoint/instruction revisions:

- `switch_success`: terminal success for the revised goal;
- `original_goal_inertia`: terminal success for the pre-revision goal;
- `switch_LSG = switch_success - original_goal_inertia`;
- exact-restoration and matched-control success rates.

The idea passes this bounded screen only if all of the following hold:

- at least four valid revised rollouts exist;
- matched-control success is 100%;
- revised-goal success is at most 50%;
- original-goal inertia is at least 50%;
- the inertia pattern occurs at two or more distinct checkpoints.

Interpretation:

- **pass/provisional candidate:** perform a separate physical-reachability
  audit before proposing training;
- **no-go:** do not develop temporal CI-GRPO on this backbone/group;
- **inconclusive:** restoration/control validity fails; fix the harness once,
  rerun the identical matrix, and do not change scientific thresholds.

Even a pass does not authorize optimization. Training is considered only after
the reachability audit and replication on another backbone or task group.
