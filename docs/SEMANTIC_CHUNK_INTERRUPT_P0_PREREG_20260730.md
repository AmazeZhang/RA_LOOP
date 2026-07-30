# Semantic Action-Chunk Interrupt P0 — preregistration

Date: 2026-07-30

## Question

When a user revises an instruction while an eight-action OpenVLA chunk is
already executing, does immediately invalidating the stale action queue improve
completion of the revised goal relative to blindly finishing that queue?

This is a bounded inference experiment. It does not authorize training, a
third method, or a search over intervention times.

## Fixed task pairs and intervention points

The three directed instruction revisions are fixed:

1. plate → stove;
2. stove → cabinet;
3. cabinet → plate.

For each original instruction, one deterministic successful baseline is run
from official LIBERO-Goal init-state index 0. The intervention chunk is the
penultimate generated action chunk of that baseline. Revisions occur after
executing offsets 1, 4, and 7 of its eight actions.

The penultimate chunk is chosen mechanically, not by inspecting outcomes. It
places the revision late enough for the original plan to be behaviorally
committed while leaving a complete later chunk in the successful baseline.

## Compared execution rules

Every saved switch state is restored exactly and scored against the revised
goal.

- `stale`: execute the unconsumed tail of the original eight-action chunk,
  then query the policy with the revised instruction.
- `flush`: discard the unconsumed tail immediately and query the policy with
  the revised instruction from the same state.

Both methods then use normal eight-action chunks under the revised instruction.
There are 3 baseline rollouts and 18 branch rollouts, for a fixed maximum of 21.
Physical GPU 0 is forbidden; physical GPU 7 is used.

## Validity and measurements

A pair enters the primary analysis only if:

- its original baseline succeeds;
- none of the three destination goals is true at the switch state;
- restored MuJoCo state equality is exact for both methods;
- stale and flush start from the same saved state;
- terminal goal predicates remain mutually exclusive.

Reported measurements:

- revised-goal terminal success;
- original-goal terminal success;
- number of stale actions executed after the language revision;
- action response latency;
- L2 action discontinuity between the last prefix action and first
  post-revision action;
- bowl displacement during the stale-tail comparison horizon.

## Decision rule

Let `F` and `S` be revised-goal success rates for flush and stale over valid
pairs. A semantic-interrupt method candidate exists only if:

- at least six of nine pairs are valid;
- `F >= 2/3`;
- `F - S >= 2/9` (at least two additional successes in the fixed matrix).

Otherwise the direction is a no-go on this backbone/group. Original-goal rate,
jerk, and short-horizon displacement diagnose the trade-off but are not used to
move the efficacy threshold after execution.

If the harness fails before producing branch outcomes, one technical repair is
allowed and the identical matrix is rerun. No timing, task-pair, seed, horizon,
or threshold sweep follows a completed matrix.

## Controller-state restoration amendment

The completed scripted reachability audit exposed a hidden-state confound in
the original branch implementation. MuJoCo's flattened state omits the Panda
gripper's internal `current_action`; `env.reset()` clears that command even
when `set_init_state` restores closed finger qpos. Thus a branch can begin from
the correct physical vector while receiving an incorrect opening actuator
command.

The scientifically valid semantic-interrupt run reconstructs every branch by
resetting to the official initial state and replaying the exact processed
original-policy action prefix through the switch offset. Stale and flush then
branch from this live replay state. This preserves gripper and OSC controller
history without adding an intervention parameter. Exact equality between the
original captured MuJoCo state and replayed switch state is still required.
Earlier state-only branch runs remain operational diagnostics and are not used
for the final language result.
