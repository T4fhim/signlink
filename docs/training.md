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

## What the report contains

Top-1 and top-5 on the val and test participants, per-participant top-1 (lowest signer shown), top-1
on signs only, and the false activation rate: the share of "other" windows predicted as a sign.
The test participants are scored once, after training, and never used to choose anything.

## Known limits of this baseline

- The "other" class comes only from hands-absent runs of Kaggle sequences; Studio recordings add better
  negatives later.
- Mirroring drops the face (no verified left/right face pairing); hands and pose are swapped properly.
- Inputs are x, y only (`input.use_z: false`) until the browser/Kaggle hand z difference is settled.
- Vocabulary is all 250 Kaggle signs; the advisors' list (G0) narrows it later.
