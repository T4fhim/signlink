---
name: dataset-license
description: Verify and record the licence of a dataset, pretrained model or lexicon source for SignLnk. Use whenever one is proposed, downloaded, or used for training, before relying on it.
argument-hint: "[dataset or model name]"
---
# Licence check: $ARGUMENTS

SignLnk rule: never invent a licence, size or version. Every dataset in use is recorded in
`docs/datasets.md` with its track (`release` = permissive only; `research` = non-commercial
allowed). Research-track data never reaches a release model or bundle.

1. Find the **primary source** (the authors' page, the dataset card, the competition rules, the
   licence file in the repository). Secondary blog posts don't count.
2. Quote the licence text that decides each of: commercial use, redistribution of data, release of
   trained model weights, attribution. Record the URL and today's date.
3. If any of the four is unclear, mark it `[VERIFY]` and say exactly what would confirm it (for
   example "email the authors" or "accept the Kaggle rules and read §A").
4. Decide the track. If a licence is unclear, the default is `research`.
5. Which sign language is it (ASL `ase`, or another)? Datasets for other sign languages are not
   interchangeable with ASL.
6. Add or update the entry in `docs/datasets.md` with the same fields as the Kaggle ISLR entry
   (source, content, files used, dataset date, downloaded, location, track, licence as found,
   consequences). Edits to that file ask for my approval.
7. For non-trivial cases, hand the evidence to the `license-auditor` subagent for a second look.
