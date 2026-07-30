# Late-state scripted reachability audit — result

Date: 2026-07-30

Preregistration:
`docs/SCRIPTED_REACHABILITY_AUDIT_PREREG_20260730.md`

Machine-readable result:
`ci_grpo/artifacts/p3_scripted_reachability_audit_run3/result.json`

## Outcome

Decision: **reachable; temporal training gate passes.**

- Nine of nine reconstructed switch states were eligible.
- Every state restored exactly and began with the bowl grasped.
- The fixed three-waypoint scripted controller reached the revised official
  goal in 9/9 states.
- Every directed pair passed at all three offsets:
  plate → stove, stove → cabinet, and cabinet → plate.
- No post-switch VLA decisions were used.

The result exceeds the preregistered gate of at least 6/9 successes with at
least one success for every directed pair.

Target bowl positions were not tuned. They came from previously saved,
successful official OpenVLA terminal states and independently passed the
corresponding LIBERO predicates.

## Controller-state correction

Runs 1 and 2 exposed a hidden restoration issue. MuJoCo's flattened state does
not include `PandaGripper.current_action`; resetting the environment clears
this internal close command even when finger qpos and contact are restored.

- Run 1 additionally stopped after the first transient contact loss.
- Run 2 removed that undeclared stop but was interrupted after confirming the
  missing gripper command caused the bowl to leave the grasp.
- Run 3 saved and restored the source checkpoint's gripper command before the
  first scripted action. It is the valid audit.

This correction also motivated the semantic-interrupt run 3, which reconstructs
each branch by replaying its exact original action prefix instead of relying on
MuJoCo state alone.

## Combined scientific implication

From the same nine late states:

- corrected OpenVLA stale queue: revised goal 0/9, original goal 9/9;
- corrected OpenVLA immediate flush: revised goal 0/9, original goal 9/9;
- fixed scripted controller: revised goal 9/9.

The revised goals are physically reachable. The OpenVLA failure is therefore a
policy conditioning/control failure rather than an impossible late revision.
This supplies the missing feasibility gate for designing a temporal
counterfactual grounding treatment. It does not authorize training by itself.
