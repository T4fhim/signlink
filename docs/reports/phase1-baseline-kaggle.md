# Baseline report (full Kaggle ISLR)

- config: `ml/configs/baseline.yaml` · commit `6876f764d841a98de7dcf727c922ffc2e48f1dda` · split `ml/splits/kaggle-islr-v1.json`
- command: `python -m signlnk_ml.training.train --data-dir /kaggle/input/competitions/asl-signs --cache-dir /kaggle/working/cache --out-dir /kaggle/working/run --epochs 30`
- model: 1,063,291 parameters, 30 epochs, final train loss 1.518

## val signers (3 participants, 13651 windows)
- top-1 0.607 · top-5 0.828 · signs-only top-1 0.604 · false activation 0.000 (hands-absent windows only)
- lowest signer top-1 0.542; per signer: 25571 0.561, 36257 0.542, 37779 0.713

## test signers (3 participants, 14190 windows)
- top-1 0.469 · top-5 0.673 · signs-only top-1 0.464 · false activation 0.000 (hands-absent windows only)
- lowest signer top-1 0.256; per signer: 2044 0.681, 29302 0.256, 34503 0.466
