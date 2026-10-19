# Phase 1 step 4: training on the draft vocabulary vs restricting the 250-sign model

Twelve Kaggle runs: the first k signs (k = 20, 30, 40, 60) of the **draft, not advisor-approved**
wishlist `ml/vocab/draft-meeting-wishlist.yaml`, 3 seeds each, trained only on those signs plus "other"
(`vocabulary.file` / `vocabulary.size`, same model and augmentation as the baseline, epochs
`4000 / k` = 200, 133, 100, 66 to keep about 4,000 optimizer steps). Raw files:
`D:\signlnk-data\kaggle-islr\vocab\runs\v{k}_seed{s}`. Scores below are signs-only top-1 (windows of the
"other" class excluded), as in the proxy.

## Result: no gain from training on the list

| signs | train top-1 | val, ensemble of 3 | test, ensemble of 3 | proxy: 250-sign ensemble restricted to the list (val / test) |
|---|---|---|---|---|
| 20 | 1.000 | 0.839 (top-5 0.978) | 0.758 (top-5 0.917) | 0.873 / 0.772 |
| 30 | 1.000 | 0.764 (top-5 0.928) | 0.667 (top-5 0.869) | 0.794 / 0.691 |
| 40 | 0.999 | 0.731 (top-5 0.919) | 0.618 (top-5 0.834) | 0.762 / 0.654 |
| 60 | 0.993-0.994 | 0.746 (top-5 0.914) | 0.604 (top-5 0.801) | 0.768 / 0.633 |

Single runs (seeds 1, 2, 3):

| signs | val | test |
|---|---|---|
| 20 | 0.813, 0.834, 0.839 | 0.738, 0.737, 0.757 |
| 30 | 0.750, 0.768, 0.750 | 0.657, 0.648, 0.641 |
| 40 | 0.716, 0.717, 0.706 | 0.595, 0.597, 0.605 |
| 60 | 0.731, 0.718, 0.730 | 0.597, 0.576, 0.579 |

- **Every size scores 2-4 points below the proxy** on both val and test, with the same ensemble recipe.
  The proxy is the 250-sign model (trained on all the data) with only the list's signs allowed to
  compete.
- The small models fit their training data completely (train top-1 0.99-1.00) and still generalise to
  new signers worse. Other signs act as extra training data for the shared encoder.
- Caveat: one recipe was tried (epochs set to hold the step count, no extra regularisation). Another
  recipe might close the gap; this one did not beat the proxy.

## Per signer (test, ensemble of 3, trained on the list)

| signs | signer 2044 | signer 29302 | signer 34503 |
|---|---|---|---|
| 20 | 0.906 | 0.542 | 0.828 |
| 30 | 0.803 | 0.444 | 0.756 |
| 40 | 0.769 | 0.394 | 0.693 |
| 60 | 0.755 | 0.380 | 0.677 |

Two of the three test signers reach 0.83-0.91 on 20 signs; signer 29302 stays near 0.54.

## Decision (proposed): train on all 250 signs, restrict at inference

- Keep training the recognizer on all Kaggle signs (the baseline recipe plus ensembling), and apply the
  chosen vocabulary as a **mask on the output at inference** (only the list's signs may be predicted,
  everything else is "no sign / other"). It costs nothing at run time and scored higher than the
  list-only models in every size tested.
- It also fits the plan: the encoder's 256-d embedding is meant to serve Studio prototypes, which
  benefits from training on more signs.
- Consequence for step 5: export the full 251-class model; the decoder (step 6) takes the vocabulary
  list as configuration. No list-specific retraining is needed to change the vocabulary.
- A formal ADR is not written yet: `/adr` can only be started by the user.

## Where this leaves the 85% target

- On the first 20 draft signs the ensemble reaches val 0.84-0.87 and test 0.76-0.77. The 85% target is
  **not met on the test signers** and borderline on the val signers; signer 29302 is the gap.
- 30 or more signs of this draft list score below 80% on val.
- The vocabulary itself (and so the real target) is still the advisors' choice (G0).
