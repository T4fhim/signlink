# Runbook for Claude in Chrome: three-seed Kaggle training run

Paste everything from "## Task" down to the end of "## Final report" into Claude in Chrome. It is
written to be followed literally. Where it says STOP, do not improvise: tell the user what you see and
wait.

## Why this exists (context for the human)

The project (SignLnk, repo `https://github.com/T4fhim/signlink`, branch `feat/phase1-iterate`, commit
`e72ff37`) needs three training runs of the same model with seeds 1, 2 and 3 on Kaggle (free GPU, the
competition data is attached there). Each run saves `scores.npz`. The results are analysed afterwards
on the user's laptop. Nothing here changes the code.

## Task

You are operating the user's own Kaggle account in their Chrome. Goal: run three training runs in a
Kaggle notebook and download one small zip with the results. Follow the phases in order. After each
phase, check its "Done when" line before moving on.

### Ground rules

1. **Never** type, read aloud or store a password, 2FA code or phone number. If Kaggle asks you to log
   in, solve a CAPTCHA, verify a phone number or confirm an email: **STOP** and ask the user to do it,
   then continue.
2. **Never** accept a competition's rules, terms of use or any legal dialog on the user's behalf. If one
   appears: **STOP** and ask the user.
3. **Never** click Submit, Submit to Competition, Publish, Make Public, Share, or anything that changes
   visibility. The notebook must stay **private**.
4. **Do not edit the repository code** or the commands below. Copy them exactly. The only allowed
   change is the data path in Phase 3 if the folder name is different, and you must tell the user.
5. **Do not delete** anything except the paths named in Phase 4 (Cell A) and Phase 6.
6. Take a page snapshot (or screenshot) before and after every click on an unfamiliar control. Kaggle's
   interface changes; if a label below does not exist, look for the closest equivalent, say so in your
   report, and if you are not sure, **STOP** and ask.
7. If any step fails twice, **STOP** and report the exact error text and where you were.
8. Keep this tab open and active while the runs go. Do not close it, navigate away, or let the computer
   sleep. Interactive Kaggle sessions can stop when idle (limit **[VERIFY]**).

### Phase 0: preconditions

1. Open `https://www.kaggle.com` in a new tab.
2. Confirm you are signed in (an avatar at the top right, not a "Sign in" button). If not: **STOP**, ask
   the user to sign in, then continue.
3. Ask the user, if you do not already know: "Which Kaggle notebook did you use for the earlier SignLnk
   runs, or should I create a new one?" Wait for the answer. If the user says "create a new one" go to
   Phase 1 step 2. If they give a name or link, use it in Phase 1 step 1.

**Done when:** signed in, and you know whether to reuse or create the notebook.

### Phase 1: open the notebook

1. **Reuse:** open `https://www.kaggle.com/code`, choose "Your Work", and open the notebook the user
   named. If several notebooks look possible, **STOP** and ask which one.
2. **Create:** on `https://www.kaggle.com/code` click "New Notebook". Wait until the editor shows an
   empty code cell.
3. Check the title near the top left. Do not rename it unless the user asks.
4. If the page offers "Save & Run All (Commit)" and "Run" style options, do **not** use them yet.

**Done when:** a notebook editor with at least one code cell is open.

### Phase 2: session settings (GPU, Internet, language)

1. Open the right-hand sidebar. If it is collapsed, expand it (the panel is labelled "Session options" or
   shows "Settings"; look for a gear or "Notebook options").
2. Set **Language** to **Python** (it is the default).
3. Set **Accelerator** to a **GPU** option. Any single GPU (for example "GPU P100" or "GPU T4 x2") is
   fine; **not** "None" and **not** TPU. If the notebook restarts after the change, wait for the session
   to come back.
4. Set **Internet** to **On**. If Kaggle says phone verification is required to enable internet:
   **STOP** and ask the user to complete it, then continue.
5. Leave **Persistence** as it is.
6. Confirm by reading the sidebar again: Accelerator shows a GPU, Internet shows On.

**Done when:** GPU selected and Internet On, both read back from the sidebar.

### Phase 3: attach the competition data

1. In the right sidebar find the "Input" (or "Data") section and click **+ Add Input** (or "Add Data").
2. In the search box type `Isolated Sign Language Recognition`.
3. Choose the **Competition** result titled **"Google - Isolated Sign Language Recognition"** (not a
   dataset with a similar name). Click **Add** (or the "+" on its card).
4. If a dialog says you must **join the competition / accept the rules**, **STOP**: the user must accept
   it themselves. Ask them, wait, then continue.
5. Confirm the input appears in the sidebar list.
6. Run the first check cell (below) and compare to the expected output.

Check cell (new code cell, run it):

```
!ls /kaggle/input
!ls /kaggle/input/competitions/asl-signs | head
```

Expected: a folder `competitions` containing `asl-signs`, and inside it `train.csv`,
`sign_to_prediction_index_map.json` and `train_landmark_files` (plus possibly more).

If `/kaggle/input/competitions/asl-signs` does not exist but `/kaggle/input/asl-signs` does, use that path
instead, **tell the user in your final report**, and use it for `D=` in Phase 5 (change only that one
variable, nothing else).

**Done when:** the listing shows `train.csv`, `sign_to_prediction_index_map.json`, `train_landmark_files`.

### Phase 4: get the code and check the GPU

Run these as two separate code cells, one at a time, waiting for each to finish.

Cell A (start fresh so an old copy cannot interfere; it deletes only the old clone folder
`/kaggle/working/signlink` and nothing else):

```
!rm -rf /kaggle/working/signlink
!git clone -b feat/phase1-iterate https://github.com/T4fhim/signlink.git /kaggle/working/signlink
!cd /kaggle/working/signlink && git log -1 --oneline
!pip -q install pyyaml pydantic
```

Expected last git line: starts with `e72ff37` (`docs: seed/ensemble recipe and analyze command`). If the
hash is different, the branch changed: report the hash you see and **STOP** unless the user says to go
on. If the clone fails with a network error, Internet is off: go back to Phase 2 step 4.

Cell B (GPU check):

```
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
```

Expected: `cuda available: True`. If it prints `False`, the accelerator did not apply: **STOP**, go back to
Phase 2 step 3, then re-run Cell B. Do not start training on CPU (it would take many hours).

**Done when:** commit `e72ff37` printed, `cuda available: True`.

### Phase 5: run the three seeds

Create **one** new code cell. Paste **exactly** the following, including the first line `%%bash` (it must
be the very first line of the cell, with nothing above it). Do not add a leading `!` to any line.

```
%%bash
cd /kaggle/working/signlink
D=/kaggle/input/competitions/asl-signs; C=/kaggle/working/cache; R=/kaggle/working/runs
run() { name=$1; shift; PYTHONPATH=ml python -m signlnk_ml.training.train --data-dir $D --cache-dir $C --out-dir $R/$name "$@"; }
run seed1 --set train.seed=1
run seed2 --set train.seed=2
run seed3 --set train.seed=3
```

(Only change `D=` if Phase 3 told you to.)

1. Run the cell (click its Run button, or Shift+Enter with the cell selected).
2. The first run first builds a data cache (several lines starting `cache train:`, `cache val:`,
   `cache test:`), then trains for 30 epochs, then prints a report starting `# Training report (full
   Kaggle ISLR)`. Then seed2 and seed3 follow and reuse the cache.
3. This is long and the total time is **[VERIFY]**: it is not known. While it runs:
   - Check the page about every 5 minutes with a snapshot. Do not click the cell again and do not
     interrupt it.
   - Note the clock time when it started and when each `# Training report` appears. The user wants the
     time per run in your final report.
4. If the cell stops with an error, copy the **last 30 lines** of the output and **STOP** (see
   "If something goes wrong" below). Do not edit the commands to work around it.
5. If the session disconnects or is stopped for inactivity, **STOP**, tell the user which seeds had
   already printed a report, and wait for instructions. Do not silently restart everything.

**Done when:** three `# Training report (full Kaggle ISLR)` blocks have printed, and each says
`overrides: train.seed=1`, `train.seed=2`, `train.seed=3` respectively, and the cell finished without error.

Write down, for each report, the two lines under "val signers" and "test signers" that begin
`- top-1` (you will repeat them in the final report).

### Phase 6: tidy and package (only after Phase 5 is done)

The data cache is about 5 GB and must not end up in the download. Run this as a new code cell. It
deletes exactly one folder (`/kaggle/working/cache`) and creates one zip. Nothing else is deleted.

```
%%bash
rm -rf /kaggle/working/cache
python -c "import shutil; shutil.make_archive('/kaggle/working/seeds', 'zip', '/kaggle/working', 'runs')"
ls -la /kaggle/working/seeds.zip
find /kaggle/working/runs -type f | sort
```

Expected: `seeds.zip` of roughly 15 to 60 MB (not gigabytes), and a file list showing, for each of
`seed1`, `seed2`, `seed3`: `history.csv`, `model.pt`, `report.json`, `report.md`, `scores.npz`.

If `seeds.zip` is larger than about 200 MB, **STOP** and report the size and the file list.

**Done when:** the zip exists, is small, and all 15 expected files are listed.

### Phase 7: download

1. Open the notebook's file browser: in the right sidebar use **Output** (under "Data"), or the
   "/kaggle/working" listing. Click the refresh icon if `seeds.zip` is not shown.
2. Download **only** `seeds.zip`: use the three-dot menu or right-click on the file and choose
   **Download**. Do not use "Download All" if it would include other large files.
3. Chrome saves to its default download folder (normally `C:\Users\<user>\Downloads`). You cannot move
   files on the user's computer. Note the **exact file name** and, if visible, the folder.
4. Confirm the download finished (Chrome's download bar or `chrome://downloads`), and note the size.

**Done when:** `seeds.zip` is fully downloaded.

### Do not do

- Do not click "Save Version", "Save & Run All (Commit)", "Submit", "Publish", "Share" or "Make Public".
- Do not change any Kaggle account setting.
- Do not add a second competition, dataset or notebook.
- Do not run any other training, and do not change `--set` values or seeds.

### If something goes wrong

| Symptom | What to do |
|---|---|
| Sign-in, CAPTCHA, phone or email verification | STOP, ask the user |
| Competition rules / terms dialog | STOP, ask the user |
| `git clone` fails or "Could not resolve host" | Internet is off: Phase 2 step 4, then redo Cell A |
| `cuda available: False` | Accelerator not applied: Phase 2 step 3, then redo Cell B |
| `ls` cannot find `asl-signs` | Re-do Phase 3; use the real folder name for `D=` and tell the user |
| `ModuleNotFoundError: yaml` or `pydantic` | Re-run the `pip install` line in Cell A |
| `CUDA out of memory` | STOP, report the last 30 lines; do not change settings |
| `cache has ... but the config wants ...` | STOP, report; do not delete anything yet |
| Quota exceeded / "GPU quota" message | STOP, report which seeds finished |
| Cell finished but a report is missing | STOP, report which seeds are missing |

## Final report

Reply to the user with exactly these items:

1. Which notebook you used (its title and URL) and whether it was reused or new. Confirm it is private.
2. The accelerator shown in the sidebar and the Internet setting.
3. The data path used for `D=`.
4. The commit hash printed in Phase 4 (expected `e72ff37`).
5. For seed1, seed2 and seed3: the clock time the run finished, the time it took, and the two `- top-1`
   lines (val and test) from its report.
6. The size of `seeds.zip` and the file list from Phase 6.
7. The exact name and folder of the downloaded `seeds.zip`, and its size on disk.
8. Anything that differed from this runbook, any warnings in the output, and any step you were unsure
   about.
9. This closing sentence: "Please extract seeds.zip into D:\signlnk-data\kaggle-islr\signlnk-step4-seeds\
   so that seed1, seed2 and seed3 are folders inside a runs folder, then tell Claude Code the path."

## Variant B: vocabulary runs (use ONLY when the user says "variant B")

Everything above applies, with these three replacements.

**Phase 4, Cell A:** the expected commit hash is the one the user gives you (the branch moved). If the
user gave none, report the hash you see and STOP until they confirm it.

**Phase 5 cell (replaces the seed cell).** Paste exactly, `%%bash` first:

```
%%bash
cd /kaggle/working/signlink
D=/kaggle/input/competitions/asl-signs; R=/kaggle/working/runs
for k in 20 30 40 60; do
  for s in 1 2 3; do
    PYTHONPATH=ml python -m signlnk_ml.training.train --data-dir $D --cache-dir /kaggle/working/cache-v$k \
      --out-dir $R/v${k}_seed$s --set vocabulary.file=ml/vocab/draft-meeting-wishlist.yaml \
      --set vocabulary.size=$k --set train.seed=$s --set train.epochs=$((4000 / k))
  done
done
```

Done when: twelve `# Training report (full Kaggle ISLR)` blocks have printed, named by their `--out-dir`
`v20_seed1` to `v60_seed3`. Each run is much shorter than the earlier baseline runs; note the start and
end time of the whole cell.

**Phase 6 cell (replaces the cleanup cell):** it deletes only the four cache folders.

```
%%bash
rm -rf /kaggle/working/cache-v20 /kaggle/working/cache-v30 /kaggle/working/cache-v40 /kaggle/working/cache-v60
python -c "import shutil; shutil.make_archive('/kaggle/working/vocab', 'zip', '/kaggle/working', 'runs')"
ls -la /kaggle/working/vocab.zip
find /kaggle/working/runs -type f | sort
```

Expected: `vocab.zip` of roughly 20 to 80 MB, and for each of the 12 run folders the five files
`history.csv`, `model.pt`, `report.json`, `report.md`, `scores.npz`. Download only `vocab.zip`. In the final
report replace "seeds.zip" with "vocab.zip" and list, for every one of the 12 runs, the two `- top-1` lines.

## For the human, after the download

Extract the zip so the folders are `D:\signlnk-data\kaggle-islr\signlnk-step4-seeds\runs\seed1`,
`...\seed2` and `...\seed3`, then tell Claude Code. It will run:

```
uv run python -m signlnk_ml.training.analyze <dir>\runs\seed1 <dir>\runs\seed2 <dir>\runs\seed3 --confusion test_confusion.csv
```
