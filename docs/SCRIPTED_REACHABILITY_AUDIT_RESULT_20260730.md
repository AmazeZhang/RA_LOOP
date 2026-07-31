# Late-state scripted reachability audit — result

Date: 2026-07-30

Preregistration:
`docs/SCRIPTED_REACHABILITY_AUDIT_PREREG_20260730.md`

Machine-readable result:
`ci_grpo/artifacts/p3_scripted_reachability_audit_run3/result.json`

## Integrity amendment (2026-07-31)

Later byte-exact recovery-SFT evaluation found that the P3 reconstructed state
hashes do not match the corrected P2 prefix-replay hashes. For example, the
first P3 state begins `b44081...`, whereas the corresponding corrected P2 state
begins `4d5a61...`; baseline episode lengths also differ.

Therefore P3 establishes 9/9 reachability for its own reconstructed states,
which use the same checkpoint, tasks, offsets, and penultimate-chunk rule, but
not for the exact nine P2 states. Statements below that say “the same nine late
states” are superseded by this amendment. Exact P2-state scripted reachability
remains unverified.

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

From two mechanically matched but not byte-identical sets of late states:

- corrected OpenVLA stale queue: revised goal 0/9, original goal 9/9;
- corrected OpenVLA immediate flush: revised goal 0/9, original goal 9/9;
- fixed scripted controller: revised goal 9/9.

The revised goals are physically reachable from the P3 reconstructed states.
Because P2 and P3 state hashes differ, this does not exclude exact P2-state
unreachability or state-provenance effects. The stronger combined policy-failure
claim is withdrawn pending an oracle audit on the byte-exact P2 prefixes.
