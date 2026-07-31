# Semantic Action-Chunk Interrupt P0 — result

Date: 2026-07-30

Preregistration:
`docs/SEMANTIC_CHUNK_INTERRUPT_P0_PREREG_20260730.md`

Machine-readable result:
`ci_grpo/artifacts/p2_semantic_chunk_interrupt_openvla_run3/result.json`

## Outcome

Decision for the tested inference method: **no-go**.

The fixed matrix completed with three successful original baselines and 18
post-revision branches. All nine stale/flush pairs passed the preregistered
validity checks:

- no destination goal was true at any switch state;
- exact processed-action prefix replay reproduced every switch-state MuJoCo
  vector byte-for-byte while preserving gripper and OSC controller history;
- all three original continuous baselines succeeded;
- terminal goal truth remained mutually exclusive.

Primary results:

| Metric | Stale queue | Immediate flush |
|---|---:|---:|
| Revised-goal success | 0/9 (0%) | 0/9 (0%) |
| Original-goal terminal rate | 9/9 (100%) | 9/9 (100%) |
| Mean response latency | 4.0 actions | 0 actions |
| Mean switch-action jerk L2 | 0.1249 | 0.2829 |

Immediate queue invalidation produced zero additional revised-goal successes,
failed the required `flush >= 6/9` ability gate, and more than doubled the mean
action discontinuity. It is therefore not a viable intervention on this
setting.

For plate → stove, stove → cabinet, and cabinet → plate, both methods failed
the revised goal at offsets 1, 4, and 7. Every one of the 18 branches completed
the original goal. All nine stale/flush pairs therefore had identical terminal
truth.

## Interpretation

The simple hypothesis “mid-chunk language failure is caused by unconsumed stale
actions” is falsified. Discarding every stale action and immediately querying
OpenVLA with the new instruction does not redirect the late-stage behavior.

The experiment exposes a stronger, separate diagnostic pattern:

- at the official initial state, the previous all-Goal screen measured perfect
  instruction redirection;
- at a mechanically selected penultimate action chunk, revised-goal success is
  0%, while the original goal remains the dominant terminal outcome.

The separate scripted reachability audit reached the revised official goal
from 9/9 mechanically matched P3 reconstructions. A 2026-07-31 provenance audit
found that their hashes differ from the corrected P2 states, so they must not be
described as the same byte-exact switch states. The valid combined result is:

- original instruction from the initial state: 100% redirection in the prior
  all-Goal screen;
- revised instruction at the penultimate chunk: 0/9 revised-goal success and
  9/9 original-goal completion for both stale and flush;
- fixed non-VLA scripted controller: 9/9 revised-goal success on the analogous
  P3 reconstruction set.

This establishes that stale queued actions do not explain the tested P2
failure. It does not yet distinguish policy grounding/control failure from
exact-state reachability or state-provenance effects, because scripted
reachability has not been verified on the byte-exact P2 states.

## Literature position and next gate

Recent action-chunk work already studies entropy-adaptive horizons, visual
deviation-triggered correction, real-time observation correction, and
cross-chunk smoothing. The potentially distinct contribution here is not
generic chunk correction; it is controlled evaluation of instruction revision
after a task has become visually and dynamically committed.

The analogous-state physical-reachability gate passed, but the exact P2-state
gate remains open. Immediate queue flushing is not a viable treatment.

The first launch terminated after three branch rows because robosuite's
internal episode counter was not reset between restored branches. Run 2
completed, but the reachability audit later showed that state-only restoration
omitted `PandaGripper.current_action`; it is retained only as an operational
diagnostic. Final run 3 replaced state-only branching with exact original
action-prefix replay. It preserved all controller history, reproduced all nine
switch states exactly, and is the only run used for the final result.
