# Phase 1 step 4: first Kaggle sweep (six one-change variants)

Run on Kaggle, commit `c722dc9`, config `ml/configs/baseline.yaml` plus one override each, split
`ml/splits/kaggle-islr-v1.json`, 30 epochs, 250 signs + "other". Raw results:
`D:\signlnk-data\kaggle-islr\signlnk-step4-results\` (`compare.txt`, `runs/<name>/report.*`).
Ranked on val; test is for reference only.

| run | change | params | train top-1 | val top-1 | val top-5 | test top-1 | test top-5 | worst test signer |
|---|---|---|---|---|---|---|---|---|
| baseline | none | 1,063,291 | 0.898 | **0.612** | 0.837 | 0.473 | 0.676 | 0.258 |
| z | `input.use_z=true` | 1,156,731 | 0.909 | 0.611 | 0.833 | 0.470 | 0.680 | 0.261 |
| velocity | `input.velocity=true` | 1,250,171 | 0.899 | 0.610 | 0.831 | 0.476 | 0.677 | 0.273 |
| augment | rotate 25°, scale 0.8–1.25, frame drop 0.2 | 1,063,291 | 0.876 | 0.610 | 0.831 | 0.473 | 0.681 | 0.267 |
| dropout | `model.dropout=0.3` | 1,063,291 | 0.854 | 0.609 | 0.833 | 0.480 | 0.683 | 0.274 |
| big | `d_model=256`, `n_layers=4` | 4,195,067 | 0.989 | 0.605 | 0.811 | 0.461 | 0.657 | 0.253 |

## What the numbers say

1. **No variant helped.** Val top-1 spans 0.605 to 0.612. The earlier baseline run (commit `6876f76`,
   same config) scored val 0.607 and test 0.469, so identical runs differ by about 0.5 points: every
   gap in the table is inside that noise. Test differences (0.461 to 0.480) are noise too.
2. **The model overfits to its training signers, it does not underfit.** Baseline train top-1 is 0.898
   against val 0.612. A 4x bigger model reaches 0.989 on train and 0.605 on val. Dropout and stronger
   augmentation lowered train top-1 (0.854, 0.876) and left val unchanged. So capacity and these
   regularisers are not the lever.
3. **Velocity and z do not help.** Both are dropped; the baseline input (x, y, presence mask) stays.
   This also avoids having to reproduce frame differences in the exported model and live decoder.
4. **One test signer is the problem, in every run.** Test per signer is about 0.68–0.70 (2044), 0.25–0.27
   (29302) and 0.45–0.48 (34503) in all six runs. A model change moves none of them, which points at
   the signer's data, not the model.

## Why signer 29302 may be hard (local sample, [VERIFY] on full data)

From the 525-sequence local sample (25 sequences per signer, so indicative only): among right-dominant
signers 29302 has the lowest hand-detection rate (right hand present in 0.46 of frames; the other
right-dominant signers 0.54–0.83) and one of the smallest shoulder widths in frame (0.476; others
0.49–0.64), i.e. further from the camera. Missing hand frames carry no information. Handedness itself
is not the cause: 29302 is right-dominant like most of training.

## What the errors look like

Baseline test, most confused pairs: nap -> sleep (26), awake -> wake (24), look -> face (23), give ->
gift (22), dryer -> dry (21). Val: awake -> wake (45), goose -> bird (40), that -> stay (22), duck ->
goose (17). These are near-synonym or related pairs that landmarks separate poorly. The 250-sign set
contains many of them, so 85% over all 250 may be unrealistic with this data; a chosen 20–50 sign
meeting vocabulary can avoid such pairs. This is a hypothesis until the vocabulary exists.

## Conclusions

- The step 4 target (85% top-1) is **not met**: best val 0.612, test 0.473 on 250 signs. No result for
  the chosen vocabulary exists, because the vocabulary is not chosen (G0).
- More of the same (bigger model, more dropout, stronger geometric augmentation) is not worth more
  Kaggle time.
- Levers that could still matter, none tested yet:
  1. **More and more varied signers**: 15 training signers is the likely ceiling. Train the final model
     on train + val (18 signers); Studio recordings (step 11) add new signers. Research-track data
     (ASL Citizen etc.) could measure the ceiling but can never ship.
  2. **Signer-invariant input**: per-hand normalization (hand relative to its wrist, scaled by hand size),
     dropping most face points, bridging short gaps where a hand is missing.
  3. **Averaging models** (seeds or variants) and mirror test-time augmentation.
  4. **Evaluate on the realistic target**: restrict scoring to 20/50-sign subsets to see what accuracy a
     chosen vocabulary could reach. Needs saved val/test scores (see below).
- Cheap enabler: have `train` save the val and test softmax scores (about 7 MB each) so subset
  evaluation, ensembling and a full confusion matrix can be done on the laptop without another Kaggle run.
- Not recorded in the logs: time per epoch, and whether the GPU was used.
