"""Offline analysis of the scores a training run saves (`scores.npz`); numpy only, laptop-friendly.

    uv run python -m signlnk_ml.training.analyze <run dir> [<run dir> ...]
        [--sizes 20 50 100] [--draws 200] [--confusion test_confusion.csv]

For each run it prints top-1 on val and test, and top-1 when only a random subset of N signs is kept
(the other signs are removed as competitors and their windows are dropped). With several run dirs it
also scores their averaged ensemble. A subset score is a proxy for a chosen vocabulary: the model
was trained on all signs, and a random subset is not the advisors' list.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from signlnk_ml.data.fetch_islr import data_dir

Scores = npt.NDArray[np.floating[Any]]
Split = tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.str_]]


def subset_top1(
    scores: Scores, labels: npt.NDArray[np.integer[Any]], classes: Sequence[int]
) -> float:
    """Top-1 over windows whose label is in `classes`, choosing only among `classes`."""
    if len(classes) == 0:
        raise ValueError("the subset is empty")
    keep = np.isin(labels, classes)
    if not keep.any():
        raise ValueError("no windows have a label in the subset")
    columns = np.asarray(classes)
    predicted = columns[scores[keep][:, columns].argmax(axis=1)]
    return float((predicted == labels[keep]).mean())


def random_subset_top1(  # noqa: PLR0913, PLR0917
    scores: Scores,
    labels: npt.NDArray[np.integer[Any]],
    size: int,
    draws: int,
    seed: int,
    exclude: Iterable[int] = (),
) -> dict[str, float]:
    """Summary of `subset_top1` over `draws` random subsets of `size` classes (minus `exclude`)."""
    present = {int(c) for c in labels.tolist() if 0 <= c < scores.shape[1]}
    classes = sorted(present - set(exclude))  # a class with no windows cannot be scored
    if not 1 <= size <= len(classes):
        raise ValueError(f"size must be between 1 and {len(classes)}, got {size}")
    rng = np.random.default_rng(seed)
    values = np.array(
        [
            subset_top1(scores, labels, sorted(rng.choice(classes, size=size, replace=False)))
            for _ in range(draws)
        ]
    )
    return {
        "size": size,
        "draws": draws,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "max": float(values.max()),
    }


def load_vocabulary(path: Path, sign_map: Mapping[str, int]) -> list[str]:
    """Sign names from a vocabulary file (`signs: [...]`, in priority order), checked for typos."""
    signs: list[str] = yaml.safe_load(path.read_text(encoding="utf-8"))["signs"]
    not_text = [s for s in signs if not isinstance(s, str)]
    if not_text:  # YAML reads an unquoted yes / no as True / False
        raise ValueError(f"{path.name}: entries must be strings (quote yes and no): {not_text}")
    unknown = [s for s in signs if s not in sign_map]
    if unknown:
        raise ValueError(f"{path.name}: not in the sign map: {unknown}")
    repeated = sorted({s for s in signs if signs.count(s) > 1})
    if repeated:
        raise ValueError(f"{path.name}: duplicate sign(s) {repeated}")
    return signs


def vocabulary_report(
    scores: Scores,
    labels: npt.NDArray[np.integer[Any]],
    signs: Sequence[str],
    sign_map: Mapping[str, int],
) -> dict[str, Any]:
    """Top-1 when only `signs` compete, per-sign top-1 (worst first) and the wrong pairs."""
    classes = [sign_map[s] for s in signs]
    names = {sign_map[s]: s for s in signs}
    keep = np.isin(labels, classes)
    if not keep.any():
        raise ValueError("no windows have a label in the vocabulary")
    columns = np.asarray(classes)
    predicted = columns[scores[keep][:, columns].argmax(axis=1)]
    true = labels[keep]
    per_sign = [
        {
            "sign": names[c],
            "top1": float((predicted[true == c] == c).mean()),
            "n": int((true == c).sum()),
        }
        for c in classes
        if (true == c).any()
    ]
    wrong = predicted != true
    pairs: dict[tuple[int, int], int] = {}
    for t, p in zip(true[wrong].tolist(), predicted[wrong].tolist(), strict=True):
        pairs[(t, p)] = pairs.get((t, p), 0) + 1
    ranked = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "n": int(keep.sum()),
        "top1": float((predicted == true).mean()),
        "per_sign": sorted(per_sign, key=lambda r: (r["top1"], r["sign"])),
        "confusions": [
            {"true": names[t], "predicted": names[p], "count": n} for (t, p), n in ranked
        ],
    }


def confusion_matrix(
    scores: Scores, labels: npt.NDArray[np.integer[Any]], n_classes: int
) -> npt.NDArray[np.int64]:
    """Counts [true, predicted]."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, scores.argmax(axis=1)), 1)
    return matrix


def ensemble(score_sets: Sequence[Scores]) -> npt.NDArray[np.float64]:
    """Mean of score arrays that describe the same windows in the same order."""
    if len({s.shape for s in score_sets}) != 1:
        raise ValueError("all score arrays must have the same shape")
    stacked = np.stack([np.asarray(s, dtype=np.float64) for s in score_sets])
    mean: npt.NDArray[np.float64] = stacked.mean(axis=0)
    return mean


def save_scores(path: Path, splits: Mapping[str, tuple[Scores, Any, Any]]) -> None:
    """Writes {name: (scores, labels, participants)} compactly (scores as float16)."""
    arrays: dict[str, Any] = {}
    for name, (scores, labels, participants) in splits.items():
        arrays[f"{name}_scores"] = np.asarray(scores, dtype=np.float16)
        arrays[f"{name}_labels"] = np.asarray(labels, dtype=np.int16)
        arrays[f"{name}_participants"] = np.asarray(participants, dtype=str)
    np.savez_compressed(path, **arrays)


def load_scores(path: Path) -> dict[str, Split]:
    with np.load(path) as data:
        names = sorted(k[: -len("_scores")] for k in data.files if k.endswith("_scores"))
        return {
            name: (
                data[f"{name}_scores"].astype(np.float32),
                data[f"{name}_labels"].astype(np.int64),
                data[f"{name}_participants"],
            )
            for name in names
        }


def _row(  # noqa: PLR0913, PLR0917
    name: str, split: str, scores: Scores, labels: Any, sizes: Sequence[int], draws: int
) -> str:
    other = scores.shape[1] - 1  # the 'other' class is last
    signs = labels != other
    top1 = float((scores.argmax(axis=1) == labels)[signs].mean())
    cells = [name, split, f"{top1:.3f}"]
    for size in sizes:
        r = random_subset_top1(scores[signs], labels[signs], size, draws, seed=0, exclude=[other])
        cells.append(f"{r['mean']:.3f} ({r['p10']:.2f}-{r['max']:.2f})")
    return "| " + " | ".join(cells)


def _vocabulary_text(
    vocab: Path,
    sign_map_path: Path,
    sizes: Sequence[int],
    combined: Mapping[str, tuple[Scores, Any]],
    n_runs: int,
) -> str:
    sign_map: dict[str, int] = json.loads(sign_map_path.read_text(encoding="utf-8"))
    signs = load_vocabulary(vocab, sign_map)
    lines = [
        "",
        f"Vocabulary `{vocab.name}`: {len(signs)} signs in priority order, "
        f"{'ensemble of ' + str(n_runs) + ' runs' if n_runs > 1 else 'single run'}. "
        "Proxy: the model was trained on all signs, not on this list.",
        "| first k signs | val top-1 | test top-1 | test windows",
        "|---|---|---|---|",
    ]
    for k in sorted({min(k, len(signs)) for k in sizes} | {len(signs)}):
        val = vocabulary_report(*combined["val"], signs[:k], sign_map)
        test = vocabulary_report(*combined["test"], signs[:k], sign_map)
        lines.append(f"| {k} | {val['top1']:.3f} | {test['top1']:.3f} | {test['n']}")
    full = vocabulary_report(*combined["val"], signs, sign_map)  # decisions use val, not test
    lines.append("")
    lines.append(
        "Weakest signs on val (top-1, n): "
        + ", ".join(f"{s['sign']} {s['top1']:.2f} ({s['n']})" for s in full["per_sign"][:12])
    )
    lines.append(
        "Most confused inside the vocabulary (true -> predicted): "
        + ", ".join(f"{c['true']} -> {c['predicted']} x{c['count']}" for c in full["confusions"])
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--confusion", type=Path, default=None, help="write the test confusion CSV")
    parser.add_argument("--vocab", type=Path, default=None, help="score a vocabulary file")
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[20, 30, 40, 50, 60])
    parser.add_argument(
        "--sign-map",
        type=Path,
        default=data_dir() / "sign_to_prediction_index_map.json",
        help="sign name -> class index (Kaggle ISLR file)",
    )
    args = parser.parse_args()

    runs = {folder.name: load_scores(folder / "scores.npz") for folder in args.runs}
    header = [
        "run",
        "split",
        "signs-only top-1",
        *[f"{n} random: mean (p10-max)" for n in args.sizes],
    ]
    lines = ["| " + " | ".join(header), "|" + "---|" * len(header)]
    last_test: tuple[Scores, Any] | None = None
    for split in ("val", "test"):
        per_run = [(name, r[split]) for name, r in runs.items()]
        for name, (scores, labels, _) in per_run:
            lines.append(_row(name, split, scores, labels, args.sizes, args.draws))
            last_test = (scores, labels) if split == "test" else last_test
        if len(per_run) > 1:
            labels0 = per_run[0][1][1]
            if any(not np.array_equal(labels0, lab) for _, (_, lab, _) in per_run):
                raise SystemExit("runs do not score the same windows: cannot ensemble")
            mean = ensemble([s for _, (s, _, _) in per_run])
            lines.append(_row("ensemble", split, mean, labels0, args.sizes, args.draws))
            last_test = (mean, labels0) if split == "test" else last_test
    print("\n".join(lines))
    print(
        "\nSubset scores are a proxy: the model was trained on all signs, the subsets are random."
    )
    if args.vocab is not None:
        first = next(iter(runs.values()))
        combined = {
            split: (ensemble([r[split][0] for r in runs.values()]), first[split][1])
            for split in ("val", "test")
        }
        print(_vocabulary_text(args.vocab, args.sign_map, args.vocab_sizes, combined, len(runs)))
    if args.confusion and last_test is not None:
        scores, labels = last_test
        matrix = confusion_matrix(scores, labels, scores.shape[1])
        np.savetxt(args.confusion, matrix, fmt="%d", delimiter=",")
        print(f"wrote {args.confusion}")


if __name__ == "__main__":
    main()
