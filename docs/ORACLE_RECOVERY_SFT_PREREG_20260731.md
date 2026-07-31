# Late-state oracle recovery SFT — preregistration

Date: 2026-07-31

> **Post-run provenance amendment:** the original preregistration below assumed
> that the independently reconstructed P3 demonstration states were byte-exact
> matches for the corrected P2 prefix states. A post-run hash audit found 0/9
> matches. The original text is preserved as preregistered, but “same nine
> states” must be read as the intended design, not an achieved invariant. See
> `ci_grpo/artifacts/p4_p2_p3_state_provenance/result.json`.

## Question

The corrected temporal-interrupt screen produced a sharp late-state failure:
the frozen OpenVLA policy reached the old goal in 9/9 branches and the revised
goal in 0/9, while a fixed non-VLA Cartesian oracle reached the revised goal
from the same nine states in 9/9. This experiment asks whether a deliberately
small amount of supervised recovery data can make the policy follow the
revised instruction without destroying its original behavior.

This is a recovery-feasibility experiment, not yet CI-GRPO. Sparse-reward GRPO
is excluded at this stage because 0/9 revised-goal success supplies no positive
on-policy reward signal.

## Fixed data

The recovery split contains exactly the nine valid switch states from the
corrected reachability audit:

1. plate → stove;
2. stove → cabinet;
3. cabinet → plate;
4. offsets 1, 4, and 7 within the pre-registered penultimate action chunk.

For every state, the already fixed three-waypoint Cartesian controller supplies
the post-switch trajectory. No waypoint, gain, tolerance, task pair, state, or
success-based trajectory search is permitted.

The retention split contains one deterministic frozen-policy trajectory for
each of the three original instructions at official init-state index 0. These
are behavior-distillation samples, not additional oracle demonstrations.

Each stored transition contains the exact 224×224 agent-view and wrist image,
the 8-D proprioceptive input, the language instruction, and the action in the
checkpoint's pre-normalization convention. Recovery controller gripper actions
are converted from LIBERO's `+1=close, -1=open` convention to the training
convention `0=close, 1=open`. Continuous action dimensions are normalized only
at training time using the immutable `libero_goal_no_noops` q01/q99 statistics.

Eight-step targets use the current action plus the next seven actions. At the
end of a trajectory, the last available action is repeated. Recovery and
retention classes are sampled 1:1 so the larger recovery set cannot silently
change the intended mixture.

## Fixed optimization budget

- Starting checkpoint: `/home/imc/models/ra-loop/openvla-oft-goal`.
- Architecture: existing continuous L1 OpenVLA-OFT head, two images, proprio.
- Trainable components: a new rank-32 LoRA adapter and the existing action
  head; the scale head, proprio projector, and base weights stay frozen.
- Optimizer: AdamW, learning rate `1e-5`, no weight decay.
- Effective batch size: 8 via batch size 1 and eight-step accumulation.
- Maximum: 200 optimizer steps.
- Checkpoints/evaluation points: steps 25, 50, 100, and 200.
- No hyperparameter search follows. A single one-backward-pass memory smoke
  test may stop the run before optimization if physical GPU 7 cannot fit.
- Physical GPU 0 is forbidden. Only physical GPU 7 may be exposed.

The first checkpoint that passes both gates below is selected. This
earliest-passing rule prevents choosing a checkpoint after inspecting all
test outcomes. If none passes, the experiment fails and no larger SFT is
started automatically.

## Evaluation gates

Late recovery is evaluated on the same nine switch states and exact revised
instructions used by the corrected temporal screen. Original retention is
evaluated by the three deterministic official-init rollouts used in the
baseline.

The SFT feasibility gate passes only if:

- revised-goal success is at least 6/9;
- every directed task pair has at least one revised-goal success;
- original-goal retention is at least 2/3;
- no rollout is counted unless exact checkpoint replay/restoration checks pass.

The strongest possible conclusion is narrow: small supervised recovery can
repair the tested late-state temporal grounding failure while retaining most
tested initial-task behavior. Failure does not prove that temporal grounding
is unlearnable; it only rejects this fixed small-data/budget intervention.
