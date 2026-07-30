# Late-state scripted reachability audit — preregistration

Date: 2026-07-30

## Purpose

The semantic-interrupt P0 found 0/9 revised-goal successes after a
penultimate-chunk instruction revision. This audit asks whether those revised
goals are physically reachable from the exact switch states without using a
VLA policy for post-switch decisions.

Success proves reachability for the tested state. Script failure does not prove
unreachability; it may reflect controller limitations.

## Fixed states and targets

The nine switch states are reconstructed deterministically using the same
OpenVLA checkpoint, official Goal init-state index 0, task pairs, penultimate
chunk rule, and offsets 1, 4, and 7 used in the semantic-interrupt P0.

Revised targets remain fixed:

1. plate → stove;
2. stove → cabinet;
3. cabinet → plate.

For each target, the desired resting bowl position is read from the previously
successful OpenVLA terminal state in
`ci_grpo/artifacts/p0_openvla_k3_rollout/result.json`. These terminal states
already pass the corresponding official LIBERO predicate.

## Eligibility

A state is eligible only if:

- no bowl-destination goal is true at the switch;
- exact MuJoCo restoration passes;
- LIBERO/robosuite contact checks report the bowl grasped by both gripper
  fingers.

Non-grasped states are reported but not replaced or searched.

## Fixed scripted controller

The controller preserves the checkpoint gripper orientation and bowl-to-EEF
offset. With the gripper closed, it follows three Cartesian waypoints:

1. lift vertically to a bowl height 0.15 m above the higher of current and
   target bowl positions;
2. translate above the target while maintaining that safe height;
3. descend until the held bowl matches the target terminal position.

Each waypoint uses the environment's OSC position action, component-wise
`clip(position_error / 0.05, -1, 1)`, at most 80 steps, with 0.015 m tolerance.
The script then opens the gripper for 15 steps and settles for 20 steps. No
orientation, waypoint, gain, tolerance, or target-position search follows.

## Decision

Reachability passes only if:

- at least six of nine states are eligible;
- at least six of nine eligible states reach the revised official goal;
- every directed task pair has at least one successful state.

If the gate passes, the 0/9 VLA result becomes evidence of a late-state policy
grounding/control failure and temporal training may be designed. If it fails,
the result is `inconclusive_or_unreachable`; no training starts.

The model is used only to reconstruct the pre-registered checkpoint states.
All post-switch actions in this audit come from the fixed scripted controller.
Physical GPU 0 is forbidden; checkpoint reconstruction uses physical GPU 7.

## Technical restoration amendment

The first two launches revealed that MuJoCo's flattened state does not contain
`PandaGripper.current_action`. After `env.reset()`, this internal actuator
command returns to zero even when `set_init_state` restores closed finger
qpos/contact. Consequently the first commanded motion can open the grasp and
does not reproduce the source checkpoint's controller state.

The valid audit therefore records `current_action` alongside every checkpoint
and restores it before the first scripted action. This is a controller-state
restoration correction, not a waypoint or threshold change. Run 1 stopped on
an undeclared transient-contact guard after one action. Run 2 removed that
guard but was interrupted after confirming the missing gripper command caused
contact loss. Neither launch is used scientifically.
