# ADR-0008: Assign hands to the nearer pose wrist

Status: accepted · 2026-10-05 · Phase 0 steps 5–6
Builds on ADR-0004 (hand labels as measured) and ADR-0006 (geometry checks).

## Context

ADR-0004 used MediaPipe Tasks' own Left/Right handedness label for each detected hand, after a
manual check (raising the right hand reported "right"). The Kaggle training data comes from legacy
Holistic, which does not classify handedness: it attaches each hand to the pose wrist it was found
from, so a hand on the subject's right wrist is always `right_hand`.

Real browser recordings (6 usable clips, several people, 540 frames) showed the label is not
reliable on its own:

- Left-hand check (hand wrist nearer the pose left wrist than the pose right wrist): **0.928**
  pooled (166 frames with a left hand); right hand 0.994. Kaggle's own agreement is 0.991 / 0.987.
- All 12 failing frames are unambiguous: the "Left" hand sits on the pose **right** wrist (3.4× to
  157× nearer to it than to the left wrist), and the other hand is not in view. A lone hand's label
  is wrong in about 7% of such frames.
- The held pose (pose runs every 2nd frame) does not explain it: agreement is 0.927 on frames with a
  fresh pose and 0.929 on stale ones.

A hand in the wrong slot moves a whole hand's 21 landmarks to the other half of the input, so it is
worse than a missing hand.

## Decision

`assembleFrame` (`packages/landmarks/src/frame.ts`) assigns hands with `handAssignment: "wrist"`,
the default:

1. One hand: to the nearer of the pose left wrist (BlazePose 15) and right wrist (16), by distance
   from the hand's wrist landmark (hand index 0) in image x, y.
2. Two hands: the pairing with the smaller total distance (straight vs crossed).
3. Fallback to MediaPipe's label (with `swapHandedness`, default false) when either pose wrist is
   missing or non-finite, when there is no pose, or on an exact tie.

`handAssignment: "label"` keeps the old behaviour. The pose wrist can be one frame stale when pose
runs every 2nd frame; that did not matter in the data above.

Unit tests cover a lone mislabelled hand on each wrist, two swapped hands, two correct hands, each
fallback, the tie and `"label"` mode.

## Consequences and open points

- On the serve side the hand-side geometry check is now true by construction (it is the same nearest
  wrist rule). It stays useful on Kaggle data and for the face and orientation checks.
- The recordings made so far were captured with label-based assignment, so they still show 0.928; a
  fresh recording should show about 1.0 for both hands. The pooled recordings threshold stays at 0.90
  until fresh clips replace them.
- A hand far from both wrists (someone else's hand in view) is still assigned to the nearer one.
- If BlazePose itself swaps its wrists when the hands cross, this rule inherits that error, as
  Holistic would.
