# Phase 1 step 4: three seeds, ensemble and vocabulary-subset analysis

Three runs of the baseline config (`ml/configs/baseline.yaml`, 250 signs + "other", 30 epochs) with
seeds 1, 2, 3 on Kaggle at commit `e72ff37`. Raw files:
`D:\signlnk-data\kaggle-islr\seeds\runs\seed{1,2,3}` (`report.md`, `scores.npz`, ...). Analysis on the
laptop: `uv run python -m signlnk_ml.training.analyze <runs> --vocab ml/vocab/draft-meeting-wishlist.yaml`.

## Run-to-run noise

| run | train top-1 | val top-1 | val top-5 | test top-1 | test top-5 |
|---|---|---|---|---|---|
| seed 1 | 0.894 | 0.612 | 0.830 | 0.467 | 0.674 |
| seed 2 | 0.904 | 0.609 | 0.828 | 0.467 | 0.674 |
| seed 3 | 0.897 | 0.611 | 0.832 | 0.481 | 0.681 |
| ensemble of 3 (mean of softmax) | n/a | 0.639 | 0.852 | 0.499 | 0.699 |

- Seed noise is about 0.3 points on val and 1.4 points on test (range over three seeds). Every
  difference in the first sweep (`phase1-step4-sweep.md`) was inside that.
- **Averaging three models adds about 3 points**: val 0.639 (single 0.609-0.612), test 0.499 (single
  0.467-0.481).

## Per signer (test): one signer carries the gap

| signer | seed 1 | seed 2 | seed 3 | ensemble |
|---|---|---|---|---|
| 2044 | 0.675 | 0.677 | 0.691 | 0.713 |
| 29302 | 0.257 | 0.261 | 0.269 | 0.284 |
| 34503 | 0.465 | 0.462 | 0.480 | 0.497 |

Without signer 29302 the test ensemble scores 0.608 (9,427 windows), the same level as val (0.639). Val
signers (ensemble): 0.584, 0.582, 0.742.

## What the model does and does not separate (test, ensemble)

- Only 7 of 250 signs reach 85% top-1, 46 reach 70%, 48 are below 30%. Accuracy is very uneven by sign.
- Worst: zipper 0.05, give 0.05, yucky 0.06, after 0.09, pen 0.09, look 0.09.
- Most confused pairs: nap -> sleep (26), awake -> wake (25), give -> gift (23), look -> face (23), dryer
  -> dry (21), pen -> pencil (20), mouth -> lips (18), listen -> hear (17). Near-synonyms.
- Full 251x251 test confusion matrix: `D:\signlnk-data\kaggle-islr\seeds\test_confusion_ensemble.csv`.

## What a smaller vocabulary would score (proxy)

Only the chosen signs compete, windows of other signs are dropped. The models were trained on all 250
signs, so this is a proxy, not a model trained on the list (which could score higher, because fewer
signs compete, or lower, because it sees less data).

**Random subsets** (200 random draws of N signs, mean and 10th percentile to best):

| N signs | val, single seed | val, ensemble | test, single seed | test, ensemble |
|---|---|---|---|---|
| 20 | 0.845-0.847 | 0.866 (0.83-0.93) | 0.703-0.711 | 0.731 (0.68-0.83) |
| 50 | 0.774-0.777 | 0.800 (0.77-0.85) | 0.622-0.632 | 0.652 (0.62-0.71) |
| 100 | 0.707-0.711 | 0.736 (0.72-0.78) | 0.556-0.568 | 0.587 (0.57-0.62) |

**Draft meeting wishlist** (`ml/vocab/draft-meeting-wishlist.yaml`, 60 signs from the Kaggle labels,
priority order written before looking at any accuracy, NOT approved by advisors). Scores on the first k
signs of that list, ensemble of 3 (single seeds in brackets):

| first k signs | val top-1 | test top-1 |
|---|---|---|
| 20 | 0.873 (0.851-0.865) | 0.772 (0.742-0.772) |
| 30 | 0.794 | 0.691 |
| 40 | 0.762 | 0.654 |
| 50 | 0.750 | 0.625 |
| 60 | 0.768 | 0.633 |

Weakest list signs on val: go 0.24, listen 0.34, give 0.34, fast 0.44, why 0.47, after 0.49. Most
confused inside the list on val: listen -> hear (21), talk -> quiet (21), why -> sick (21), go -> later
(17), after -> finish (16), look -> see (16). Advisors can drop one of each pair.

## What this means for the 85% target

- On **val signers** the first 20 draft signs already reach 0.85-0.87 with a model that was never trained
  on that list. On the **test signers** the same list reaches 0.74-0.77, held down by signer 29302.
- So the target looks reachable for about 20 well-chosen, mutually distinct signs for most signers, and
  not yet for every signer. Both numbers are proxies until a model is trained on the chosen list.
- The list is a draft of mine from a mostly children's vocabulary. The real vocabulary, and whether
  it is 20 or 50 signs, is the advisors' decision (G0).

## Next experiment

Train directly on the vocabulary (`vocabulary.file` + `vocabulary.size` overrides, added in this
branch), 3 seeds each for 20, 30, 40 and 60 signs, and compare with the proxy above. Recipe:
`docs/training.md`, section "Train on the draft vocabulary".
