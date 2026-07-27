# RA-LOOP soft counterfactual final trial — 2026-07-26

## Decision

The hard pair gate did not pass independent evaluation at either fixed-L2
`0.1 rad` or the independently selected `0.2 rad` evaluation strength. This is
the final planned algorithm iteration before a go/no-go decision.

## Fixed change

All four anchor/perturbed pairs in a rollout group share one base simulator
state. The new estimator uses:

```text
c_hat(x) = mean(anchor successes for the state)
A_soft,i = c_hat(x) * [S_perturbed,i - LOO_mean(S_perturbed,-i)]
```

Consequences:

- an individual stochastic anchor failure no longer hard-removes its paired
  perturbation when the state has positive aggregate competence;
- an all-failed anchor state produces zero recovery signal;
- uniform perturbed outcomes produce zero signal;
- each perturbed advantage group still sums exactly to zero;
- NPC and the global warm-start calibration barrier are unchanged.

There is no new threshold or tunable reward coefficient.

## Execution plan

1. One-task, 3-step GPU smoke: one calibration step and two active
   opportunities.
2. If smoke proves a real nonzero soft-CRA update, run exactly two seeds:
   `10000` and `20000`.
3. Each seed uses four tasks, K=8, fixed-L2 training strength `0.1`, 51 total
   steps, calibration=3/task, and step-50 as the preselected checkpoint.
4. Evaluate both step-50 checkpoints on the already frozen `0.1` and `0.2`
   independent protocols. Do not select another checkpoint after seeing
   results.

The direction stops if the two seeds do not show consistent positive paired
gains, roughly >=5 points at the informative `0.2` strength, with anchor drop
no more than 2 points.
