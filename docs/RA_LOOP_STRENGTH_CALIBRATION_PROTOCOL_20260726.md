# RA-LOOP stronger-intervention calibration protocol — 2026-07-26

## Purpose

The completed fixed-L2 `0.1 rad` held-out evaluation has a 94.5% warm-start
perturbed-success ceiling.  This calibration selects one more informative
Robot-init strength using only the warm-start policy, before comparing A4/A5.

## Calibration set

- Model: warm-start (`pilot`, step 5) only.
- LIBERO-Spatial: all ten tasks.
- Init indices: `[26, 36)`, ten pairs per task.
- Strengths: `0.15` and `0.20 rad`.
- Perturbation seed: `20260811`.
- The same task/init/seed produces the same perturbation direction at both
  strengths.
- Total: 2 strengths × 10 tasks × 10 pairs × 2 modes = 400 episodes.

## Selection rule

Among strengths whose pooled warm-start fixed-L2 success is in `[65%, 85%]`,
select the one closest to 75%. If both are equally close, select the lower
strength. If neither lies in the interval, select the one closest to 75% and
record that the desired difficulty band was missed.

No A4/A5 result may be inspected before this selection is frozen.

## Subsequent comparison set

The selected strength will be evaluated on disjoint init indices `[36, 50)`,
with perturbation seed `20260821`, for:

- warm-start (`pilot`, step 5);
- A4 CRA-only (step 40);
- A5 CRA+NPC (step 40).

This subsequent comparison contains 3 × 10 × 14 × 2 = 840 episodes. Its Gate
is fixed-L2 improvement of roughly 5 percentage points, anchor drop no more
than 2 points, and clearly positive paired wins minus losses.
