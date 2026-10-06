# Datasets

One entry per dataset: source, licence as found, version, download date, where it lives, and which
data track it belongs to (`release` = permissive only; `research` = non-commercial allowed). Nothing
in the `research` track may reach a release model or bundle (CLAUDE.md rule 4).

## Kaggle ISLR — Google "Isolated Sign Language Recognition" (`asl-signs`)

| Field | Value |
|---|---|
| Source | https://www.kaggle.com/competitions/asl-signs |
| Content | 94,477 sequences, 250 signs, 21 participants; landmarks extracted with legacy MediaPipe Holistic (face 468, left_hand 21, pose 33, right_hand 21 per frame) |
| Files used | `train.csv`, `sign_to_prediction_index_map.json`, and a sample of `train_landmark_files/<participant>/<sequence>.parquet` |
| Dataset date | files dated 2023-02-17 on Kaggle |
| Downloaded | 2026-10-04 via the `kaggle` CLI after accepting the competition rules |
| Location | `D:\signlnk-data\kaggle-islr` (set `SIGNLNK_DATA_DIR=D:\signlnk-data`); never committed (`data/` is gitignored) |
| Track | **release** |

**Licence as found** (Rules → "A. Data Access and Use", text supplied by the project owner on
2026-10-04): the Competition Data may be used "for any purpose, whether commercial or
non-commercial, including for participating in the Competition and on Kaggle.com forums, and for
academic research and education"; it "is also subject to the following terms and conditions:
CC-By 4.0"; where the CC terms conflict with the Rules, the Rules govern. The Sponsor may disqualify
participants who use the data other than as the Rules permit.

Consequences:
- Release-track use is allowed. CC BY 4.0 needs **attribution**: credit "Google – Isolated Sign
  Language Recognition (Kaggle)" in the model card and public site. Wording to be settled in Phase 1.
- Keep the Rules text with the model licence files so the basis for the release track is on record.
- Only a sample is on disk (about 40 GB for the full set; the laptop's C: drive has little space).
  The sample is participant-balanced: the same number of sequences per participant, spread over
  signs (`uv run python -m signlnk_ml.data.fetch_islr --per-participant 25`).

**[VERIFY]** the licence wording on the live Rules page before the first public release, and
re-check whether the Kaggle competition page lists a separate data licence field.

## MediaPipe landmarker models (`hand_landmarker`, `pose_landmarker_lite`, `face_landmarker`)

| Field | Value |
|---|---|
| Source | `storage.googleapis.com/mediapipe-models/` (URLs and SHA-256 in `scripts/mediapipe-assets.json`) |
| Content | Pretrained `.task` bundles run in the browser by `@mediapipe/tasks-vision@1.0.1` (npm package licence Apache-2.0, read from its `package.json` 2026-10-06) |
| Location | `apps/web/public/mediapipe/models` (fetched by `pnpm assets`, gitignored) |
| Track | Release (runtime component, not training data) |

**Licence as found:** none stated for the model files. The Hand Tracking model card (checked
2026-10-06 by `license-auditor`) names no licence; the developers.google.com Hand Landmarker page
licenses only its own content (CC BY 4.0) and code samples (Apache 2.0). Pose and face model cards were
not checked.

**[VERIFY]** an explicit licence for each `.task` file before Phase 1 step 7 (the PWA caches and serves
them from our own origin). If none exists, download them at first run instead of redistributing.

## Not yet downloaded

ASL Citizen, Sem-Lex, WLASL and How2Sign are research-track (see PLAN §5.1) and have no entries
until they are used. Record the licence, version and download date here when they are.
