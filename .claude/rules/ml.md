---
paths:
  - "ml/**/*.py"
  - "ml/configs/**"
  - "ml/pyproject.toml"
---
<!-- Loads when Claude works on Python ML code or training configs. -->
# ML rules

- Every `ml/configs/*.yaml` declares `track: release|research` (a Stop hook blocks without it).
  Release configs use only release-track datasets listed in `docs/datasets.md`.
- New dataset, pretrained weights or lexicon source: run `/dataset-license` first. Never state a
  licence, dataset size or version you haven't read at the source; mark unknowns `[VERIFY]`.
- Splits are signer-independent (participant IDs, zero overlap asserted). Report top-1/top-5 on
  unseen signers, never only a random split; add subgroup breakdowns when available.
- Never quote a metric without the command, data split and commit that produced it.
- Python 3.11 compatible (numpy pinned at 2.4.6); mypy strict; type hints everywhere.
- Training runs on Kaggle/Colab free tiers; local runs are CPU-only smoke tests.
