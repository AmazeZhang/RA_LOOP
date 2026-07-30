# Semantic Action-Chunk Interrupt P0 — result

Date: 2026-07-30

Preregistration:
`docs/SEMANTIC_CHUNK_INTERRUPT_P0_PREREG_20260730.md`

Machine-readable result:
`ci_grpo/artifacts/p2_semantic_chunk_interrupt_openvla_run2/result.json`

## Outcome

Decision for the tested inference method: **no-go**.

The fixed matrix completed with three successful original baselines and 18
post-revision branches. All nine stale/flush pairs passed the preregistered
validity checks:

- no destination goal was true at any switch state;
- all MuJoCo state restorations were exact;
- all three original continuous baselines succeeded;
- terminal goal truth remained mutually exclusive.

Primary results:

| Metric | Stale queue | Immediate flush |
|---|---:|---:|
| Revised-goal success | 0/9 (0%) | 0/9 (0%) |
| Original-goal terminal rate | 7/9 (77.8%) | 7/9 (77.8%) |
| Mean response latency | 4.0 actions | 0 actions |
| Mean switch-action jerk L2 | 0.1249 | 0.2828 |

Immediate queue invalidation produced zero additional revised-goal successes,
failed the required `flush >= 6/9` ability gate, and more than doubled the mean
action discontinuity. It is therefore not a viable intervention on this
setting.

Per directed revision:

- plate → stove: both methods failed the revised goal at offsets 1, 4, and 7;
  all six branches completed the original plate goal;
- stove → cabinet: both methods failed all revised goals; stale completed the
  old goal in 2/3 cases and flush in 3/3;
- cabinet → plate: both methods failed all revised goals; stale completed the
  old goal in 2/3 cases and flush in 1/3.

Seven of nine paired cases had identical stale/flush terminal truth. The two
differences changed only whether the terminal state was old-goal or all-false;
neither produced the revised goal.

## Interpretation

The simple hypothesis “mid-chunk language failure is caused by unconsumed stale
actions” is falsified. Discarding every stale action and immediately querying
OpenVLA with the new instruction does not redirect the late-stage behavior.

The experiment exposes a stronger, separate diagnostic pattern:

- at the official initial state, the previous all-Goal screen measured perfect
  instruction redirection;
- at a mechanically selected penultimate action chunk, revised-goal success is
  0%, while the original goal remains the dominant terminal outcome.

This is consistent with late-trajectory goal commitment or visual-state
inertia, but it is not yet proof of linguistic blindness. At these late states,
independent physical reachability of the revised destination has not been
established by an oracle controller. A model failure cannot serve as its own
reachability test.

## Literature position and next gate

Recent action-chunk work already studies entropy-adaptive horizons, visual
deviation-triggered correction, real-time observation correction, and
cross-chunk smoothing. The potentially distinct contribution here is not
generic chunk correction; it is controlled evaluation of instruction revision
after a task has become visually and dynamically committed.

No training or additional policy sweep follows this P0. Before treating the
late-state pattern as a research direction, the next required gate is a
non-policy physical-reachability audit from the exact nine switch states. Only
if an oracle can reach the revised goals from those states does a temporal
language-grounding treatment become scientifically justified.

The first launch terminated after three branch rows because robosuite's
internal episode counter was not reset between restored branches. No complete
matrix was produced. The one allowed technical repair added `env.reset()`
before each exact state restoration; run 2 then completed the identical
pre-registered matrix.
