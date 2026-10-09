# Phase 1, steps 3 and 4: status summary

Written 2026-10-07. Covers the baseline model (step 3), the iteration toward 85% (step 4) and the
Kaggle runs that both depend on. Numbers come from `docs/reports/phase1-baseline-kaggle.md` and the
repo; anything not measured is marked **[VERIFY]**.

## 1. Where things stand

| | State |
|---|---|
| Step 2: signer-independent splits | Done, merged (PR #9) |
| Step 3: baseline model + training pipeline + first report | Done, merged (PR #10). Report exists |
| Step 4: iterate to ≥85% top-1 on unseen signers | **In progress.** Tooling built and pushed (branch `feat/phase1-iterate`, `c722dc9`, no PR yet). No new accuracy result yet |
| Blocker for a final step 4 verdict | The target is for the _chosen vocabulary_ (20–50 signs). That list needs Deaf advisors (gate G0). None recruited |

Headline: the baseline reaches **test top-1 0.469** (target 0.85, a gap of 38 points) and **val top-1
0.607**. The next move is a six-run sweep on Kaggle that only you can run.

## 2. What happened, in order

1. **Step 2.** Kaggle ISLR has 21 participants. They were split once and committed in
   `ml/splits/kaggle-islr-v1.json`: 15 train, 3 val, 3 test. Val and test each hold one
   left-dominant and one right-dominant signer (a reviewer caught that the first draw had put all
   right-handers in val). `splits --check` asserts zero participant overlap on the full `train.csv`.
2. **Step 3 built.** A pipeline that turns each Kaggle sequence into a 64-frame normalized window, adds
   an "other" (no sign) class from stretches where neither hand is detected, augments, trains a
   1,063,291-parameter conv + Transformer model and writes a signer-independent report. The
   release/research data-track check (ADR-0003) is now enforced in code and in CI.
3. **Local smoke test only.** The laptop has no GPU and only a 525-sequence sample of the data, so a
   CPU run on the sample (8 epochs, train loss 5.65 → 4.66) proved the pipeline works. Accuracy at
   that size means nothing and the report said so.
4. **First Kaggle run (you).** A 2-epoch check, then 30 epochs on the full data, at commit `6876f76`.
   Results below.
5. **Step 3 closed.** Report recorded in `docs/reports/`, PROJECT_CONTEXT and CHANGELOG; PR #10 merged.
6. **Step 4 tooling.** Diagnostics, new options and a comparison tool were built so the next Kaggle
   session answers "why is it at 47%?" and not just "what is it now?". Pushed, awaiting the sweep.

## 3. The baseline result

| | top-1 | top-5 | per signer top-1 |
|---|---|---|---|
| val (3 signers, 13,651 windows) | 0.607 | 0.828 | 0.561 / 0.542 / 0.713 |
| test (3 signers, 14,190 windows) | 0.469 | 0.673 | 0.681 / 0.256 / 0.466 |

Final train loss 1.518 (with label smoothing 0.1). The 2-epoch check run scored val 0.269 and test 0.196.

What this does and does not say:

- The model learns across signers (far above chance, which is about 0.4% for 251 classes) but is
  nowhere near 85%.
- Val and test differ by 14 points, mostly because **one test signer (29302) scores 0.256**. With 3
  signers per split, one hard signer moves the whole number. Treat gaps of about 1 point as ties.
- The val curve climbs to about 0.60 by epoch 23 and then flattens, so simply training longer will not
  close the gap.
- **We do not know the train accuracy.** The first report did not record it, so we cannot yet say
  whether the model underfits (cannot fit the data) or overfits (fits, but does not transfer to new
  signers). That single number decides which fixes are worth trying, and the sweep supplies it.
- The "other" class false-activation rate printed as 0.000 is not meaningful: those windows are all
  hands-absent, which is trivially easy to tell apart.
- Not recorded in the logs: time per epoch, and whether the GPU was used.

## 4. Step 4: goal and what is built

**Goal (PLAN §7):** ≥85% top-1 on unseen signers for the chosen vocabulary; if stuck, escalate to
Opus with the training curves and confusion matrix.

**Built, no result yet:**

- Every report now includes train top-1 (20,000 train windows, no augmentation), the worst classes
  and the most-confused sign pairs, by name.
- `input.velocity` (frame-to-frame differences as extra input).
- `--set section.key=value` overrides, so one notebook can run many variants without new files. They
  are checked against the data-track rule before training starts.
- A guard that refuses to reuse a cache built with different window settings.
- `compare`: a table of runs **ranked on val**, with test shown for reference only. Choosing a
  configuration by its test score would turn the test signers into a second validation set.
- The sweep recipe in `docs/training.md`: baseline, velocity, z on, bigger model, dropout 0.3, stronger
  augmentation. One change each.

**Cannot do yet:**

- Restrict training or evaluation to a subset of signs. The 85% target is defined for the chosen
  20–50 signs; the 250-sign sweep is only a proxy until that list exists.
- Show a result. Nothing here has been run on the full data since the baseline.

## 5. The Kaggle part: how it works

Why Kaggle: there is no local GPU, and the full dataset is about 40 GB, far more than the laptop holds.
Kaggle gives a free GPU and has the competition data attached.

What happens in a run:

1. The notebook clones the repo from GitHub (`feat/phase1-iterate`, public).
2. `train` reads `train.csv` and the sequence files from `/kaggle/input/competitions/asl-signs`, builds a
   window cache of about 5 GB in `/kaggle/working/cache` (first run only; later runs reuse it), then
   trains.
3. Each run writes `report.md`, `report.json`, `history.csv` and `model.pt` to its own folder.
4. `compare` prints the table of all runs.

**Exact steps for you:**

1. Notebook settings: Accelerator = GPU, Internet = On (Kaggle may require phone verification for this
   **[VERIFY]**). Under Add Input, attach the competition "Google - Isolated Sign Language Recognition"
   (accept its rules first). Confirm with `!ls /kaggle/input/competitions/asl-signs`.
2. Cell 1: `!git clone -b feat/phase1-iterate https://github.com/T4fhim/signlink.git /kaggle/working/signlink`
   then `!pip -q install pyyaml pydantic`.
3. Cell 2: paste the whole `%%bash` cell from the "Step 4 sweep" section of `docs/training.md`.
   `%%bash` must be its first line; the `cd /kaggle/working/signlink` is needed because `PYTHONPATH=ml`
   is relative.
4. Run it. For a long job use Save Version, then Save & Run All (Commit), so a disconnect does not kill
   it. Watch the first run to learn the epoch time before trusting the session budget.
5. Download `runs/baseline/report.md` and the table `compare` prints at the end. Skip the cache folder.
6. Send both to me, or drop them in `D:\signlnk-data\kaggle-islr\signlink-training-results\`.

Watch for: a session timeout or GPU quota running out mid-sweep (each run is independent, so finished
runs are still usable); a `cache has ...` error means the window settings changed and the cache must be
rebuilt in a new cache folder.

## 6. Who has to do what

| Who | Needs to do |
|---|---|
| **You** | Run the Kaggle sweep and send back the results. Decide when to open the step 4 PR |
| **You / project owner** | Recruit at least two fluent Deaf ASL signers (advisors), who choose the final 20–50 signs (G0, step 1) |
| **Claude** | Read the sweep results, pick the next experiments, build features or fixes, restrict training to the chosen vocabulary once it exists, keep docs and PRs current |
| **Opus (you switch the model)** | The PLAN's escalation if step 4 is stuck: send it the training curves and confusion matrix. The model-routing note in PROJECT_CONTEXT puts hard training problems on Opus |
| **Deaf advisors** | G0 vocabulary sign-off; later G1 before any public demo of captions |

## 7. How progress gets made

After the sweep, read the **train top-1** of each run next to its **val top-1**. These are rules of
thumb, not measured facts:

- **Train top-1 is low too:** the model is not fitting. Try capacity (the "big" run), longer training, a
  higher learning rate, better input features (velocity, z).
- **Train top-1 is high but val is low:** the model memorises signers. Try stronger augmentation,
  dropout and weight decay, fewer or cleaner input landmarks, and per-hand normalization.
- **Check the worst classes and confusions:** if the same sign pairs dominate the errors, look at
  whether they differ mainly by something the input does not carry (finger shape, face) and add
  features for that.
- **Combine** the changes that helped on val into one final run, then read the test score once.
- **Variance:** val and test have 3 signers each. Small differences are noise.
- **If stuck** after one or two rounds, escalate to Opus with `history.csv` and the confusions, as the
  PLAN says.
- **In parallel, without waiting on Kaggle:** advisor outreach (step 1), which unblocks the
  vocabulary and therefore the real 85% target. After that, step 5 (ONNX export) and step 6 (live
  decoder, which also owns the fps carry-over).

## 8. Open risks and unknowns

- Epoch time and GPU use on Kaggle: not measured **[VERIFY]**. One CPU batch step of 256 took 2.8 s on
  the laptop, which extrapolates to hours per run, so the GPU is needed.
- The 85% target may be easier on a 20–50 sign vocabulary than on 250 signs, or harder if the chosen
  signs are visually similar. We cannot know until the list exists.
- If `input.velocity` is kept, the exported model and the live decoder must compute the same frame
  differences.
- Browser hand depth (z) looked about half the Kaggle value in earlier recordings **[VERIFY]**. The
  baseline therefore uses x and y only; the sweep includes a z run as the test of this.
- Kaggle rules and the dataset licence must be re-checked on the live page before any public release
  (carried **[VERIFY]**).

## 9. Where everything is

| What | Where |
|---|---|
| Baseline report and curve | `docs/reports/phase1-baseline-kaggle.md`, `…-history.csv` |
| Training recipe, Kaggle steps, sweep | `docs/training.md` |
| Code | `ml/signlnk_ml/training/` (`train`, `cache`, `model`, `evaluate`, `compare`, `config`) |
| Config and dataset registry | `ml/configs/baseline.yaml`, `ml/datasets.yaml` |
| Splits | `ml/splits/kaggle-islr-v1.json` |
| PRs | #9 (splits), #10 (baseline), both merged. Step 4: branch `feat/phase1-iterate`, no PR yet |
| History | `docs/CHANGELOG.md`, `PROJECT_CONTEXT.md` |
