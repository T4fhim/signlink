# Training the baseline (Phase 1 step 3)

The baseline is a 1D conv + Transformer encoder over `[T, N, 3]` landmark windows, 250 Kaggle ISLR
signs plus an "other" (no sign) class (`ml/configs/baseline.yaml`, `track: release`). Splits are by
participant (`ml/splits/kaggle-islr-v1.json`, ADR-0003 for the data track). Code: `ml/signlnk_ml/training/`.

The step's **Done when** ("a top-1/top-5 report on the test signers exists") needs the full dataset,
so it comes from a Kaggle or Colab run. The laptop has no GPU and only a 525-sequence sample.

## Local CPU smoke run (pipeline check, not a result)

```
uv sync --all-packages --extra train
SIGNLNK_DATA_DIR=D:\signlnk-data uv run python -m signlnk_ml.training.train \
  --cache-dir D:\signlnk-data\cache-sample --out-dir D:\signlnk-data\runs\smoke --sample --epochs 8
```

`uv sync` without `--extra train` removes torch again. The report is marked SAMPLE.

## Full run on Kaggle (free tier)

1. New notebook, add the competition data (Google "Isolated Sign Language Recognition"; accept its rules
   first). It appears at `/kaggle/input/asl-signs` with `train.csv`, `sign_to_prediction_index_map.json`
   and `train_landmark_files/`.
2. Turn on a GPU accelerator (training picks `cuda` automatically; the GPU path is **[VERIFY]**, it
   could not be run on the laptop). On CPU one batch step of 256 took 2.8 s here, which extrapolates
   to roughly 14 min per epoch and hours for 30 epochs, so use the GPU.
   Get the repo into the notebook with `git clone`, not an upload: the report cites the commit hash
   from git and says `unknown` without it. Do not
   `pip install -e ml`: it pins `numpy==2.4.6` and would replace the notebook's numpy. The code needs only
   numpy, pyarrow, pydantic, pyyaml and torch, which the image already has (`pip install pyyaml` if not).
3. Run, from the repo root:

```
PYTHONPATH=ml python -m signlnk_ml.training.train --config ml/configs/baseline.yaml \
  --data-dir /kaggle/input/asl-signs --cache-dir /kaggle/working/cache --out-dir /kaggle/working/run
```

   Try `--epochs 2` first: the cache build and epoch time on the full data are **[VERIFY]** (not yet
   measured). The cache is about 5 GB of float16 windows.
4. Download `report.md`, `report.json` and `history.csv` from `/kaggle/working/run`. Record the numbers
   in `docs/CHANGELOG.md` with the commit hash printed in the report.

Colab works the same way with the competition data downloaded through the Kaggle CLI.

## Step 4 sweep: one change at a time

`--set section.key=value` overrides any config value without a new file (repeatable; values are YAML).
The cache is reused between runs as long as the `window` settings do not change. One Kaggle cell
(`%%bash` must be its first line, and the `cd` makes `PYTHONPATH=ml` resolve; adjust it to where you cloned):

```
%%bash
cd /kaggle/working/signlnk
D=/kaggle/input/competitions/asl-signs; C=/kaggle/working/cache; R=/kaggle/working/runs
run() { name=$1; shift; PYTHONPATH=ml python -m signlnk_ml.training.train --data-dir $D --cache-dir $C --out-dir $R/$name "$@"; }
run baseline
run velocity --set input.velocity=true
run z        --set input.use_z=true
run big      --set model.d_model=256 --set model.n_layers=4
run dropout  --set model.dropout=0.3
run augment  --set augment.rotate_degrees=25 --set "augment.scale_range=[0.8,1.25]" --set augment.drop_probability=0.2
PYTHONPATH=ml python -m signlnk_ml.training.compare $R/*
```

Each run takes about as long as the baseline run did (epoch time is **[VERIFY]**, not in the logs), so
check that budget before starting all six. `compare` ranks on **val** top-1 and shows test only for
reference: a configuration chosen by its test score turns the test signers into a second validation
set. Combine the changes that helped on val into one final run, then read its test score once.
Every report now has the train top-1 (no augmentation) too: a low train top-1 means the model does
not fit (more capacity, longer training); a high train top-1 with a low val top-1 means it does not
transfer to new signers (more augmentation, regularisation, better features).

## Seeds, ensemble and vocabulary-subset analysis

Every run now also saves `scores.npz` (val and test softmax scores, about 7 MB each). That lets the
laptop answer three questions without another Kaggle run. Three runs of the same config with different
seeds (one Kaggle cell, same shape as the sweep above):

```
%%bash
cd /kaggle/working/signlnk
D=/kaggle/input/competitions/asl-signs; C=/kaggle/working/cache; R=/kaggle/working/runs
run() { name=$1; shift; PYTHONPATH=ml python -m signlnk_ml.training.train --data-dir $D --cache-dir $C --out-dir $R/$name "$@"; }
run seed1 --set train.seed=1
run seed2 --set train.seed=2
run seed3 --set train.seed=3
```

Download each run's `scores.npz` (and `report.md`) into one folder per run, then on the laptop:

```
uv run python -m signlnk_ml.training.analyze <dir>\seed1 <dir>\seed2 <dir>\seed3 --confusion test_confusion.csv
```

It prints top-1 per run and for the averaged **ensemble**, and top-1 when only a random subset of 20, 50
or 100 signs is kept (mean, 10th percentile and best of 200 draws). The spread between seeds is the
real run-to-run noise. Subset scores are a proxy for a chosen vocabulary: the model was trained on all
250 signs and a random subset is not the advisors' list.

## What the report contains

Top-1 and top-5 on the val and test participants, per-participant top-1 (lowest signer shown), top-1
on signs only, and the false activation rate: the share of "other" windows predicted as a sign.
The test participants are scored once, after training, and never used to choose anything.

## Known limits of this baseline

- The "other" class comes only from hands-absent runs of Kaggle sequences; Studio recordings add better
  negatives later.
- Mirroring drops the face (no verified left/right face pairing); hands and pose are swapped properly.
- Inputs are x, y only (`input.use_z: false`) until the browser/Kaggle hand z difference is settled.
- Vocabulary is all 250 Kaggle signs; the advisors' list (G0) narrows it later. Nothing yet restricts
  training or evaluation to a subset of signs: build that once the list exists.
- If `input.velocity` is kept, the serving side (ONNX export, live decoder) must compute the same frame
  differences, or they must go inside the exported graph.
- Val and test have 3 signers each: treat val differences of about 1 point between runs as ties.
