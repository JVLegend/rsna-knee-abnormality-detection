"""CPU-only rank stack over validated outputs from our completed kernels.

This notebook deliberately consumes kernel outputs rather than pixels. It is a
low-cost robustness test while the GPU quota is exhausted; no competition data or
predictions are stored in the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
EXPECTED = ["StudyInstanceUID", *TARGETS]

# H-29 is the only promoted member. The small B0 vote is intentionally conservative:
# H-27 is a different visual head and H-33 contributes an independent text signal, but
# neither is allowed to overturn H-29 on its own. These are fixed before looking at a
# new leaderboard score.
SOURCE_WEIGHTS = {"h29": 0.80, "h27": 0.15, "h33": 0.05}
SOURCE_TOKENS = {
    "h29": ("h-29", "h29"),
    "h27": ("v4-plane-target", "h-27", "h27"),
    "h33": ("weak-text-plane", "h-33", "h33"),
}


def find_test_csv() -> Path:
    roots = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("/kaggle/input"),
        Path("."),
    ]
    for root in roots:
        candidate = root / "test.csv"
        if candidate.is_file() and "StudyInstanceUID" in pd.read_csv(candidate, nrows=0).columns:
            return candidate
    raise FileNotFoundError("competition test.csv was not mounted")


def candidate_files() -> list[Path]:
    # Never recurse through the competition mount: its train/test_series directories
    # contain millions of DICOM files. Kernel-source outputs are mounted at one or two
    # shallow levels, so enumerate directory entries only and inspect CSVs at depth <= 2.
    input_root = Path("/kaggle/input")
    roots = [Path(".")]
    if input_root.is_dir():
        roots.append(input_root)
        first_level = [p for p in input_root.iterdir() if p.is_dir()]
        roots.extend(first_level)
        roots.extend(
            child
            for parent in first_level
            for child in parent.iterdir()
            if child.is_dir() and child.name not in {"train_series", "test_series"}
        )
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.glob("submission*.csv"):
            if path.is_file() and not path.name.lower().startswith("sample"):
                found.add(path)
    return sorted(found)


def source_for(path: Path) -> str | None:
    text = str(path).lower()
    matches = [
        source
        for source, tokens in SOURCE_TOKENS.items()
        if any(token in text for token in tokens)
    ]
    return matches[0] if len(matches) == 1 else None


def read_sources(test_ids: pd.Series) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for path in candidate_files():
        source = source_for(path)
        if source in loaded:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"skip unreadable {path}: {type(exc).__name__}: {exc}")
            continue
        if list(frame.columns) != EXPECTED or len(frame) != len(test_ids):
            continue
        if not frame["StudyInstanceUID"].is_unique:
            continue
        if not frame["StudyInstanceUID"].reset_index(drop=True).equals(test_ids.reset_index(drop=True)):
            continue
        values = frame[TARGETS].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            continue
        loaded[source] = frame
        print(f"accepted {source}: {path}")
    return loaded


def rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # ROC AUC reads order, so percent ranks remove harmless calibration differences
    # between model families while preserving every target's ordering.
    ranks = frame[TARGETS].rank(method="average", pct=True)
    return ranks.astype(np.float64)


def main() -> None:
    test_path = find_test_csv()
    test = pd.read_csv(test_path, usecols=["StudyInstanceUID"])
    sources = read_sources(test["StudyInstanceUID"])
    if "h29" not in sources:
        raise RuntimeError("H-29 output is not mounted; refusing to create a fallback submission")

    active = {name: weight for name, weight in SOURCE_WEIGHTS.items() if name in sources}
    normalizer = sum(active.values())
    if normalizer <= 0:
        raise RuntimeError("no positive source weight")
    print(f"active_sources={sorted(active)} weights={active}")

    blended = sum(
        (weight / normalizer) * rank_frame(sources[name])
        for name, weight in active.items()
    )
    output = pd.concat([test, blended], axis=1)
    output[TARGETS] = output[TARGETS].clip(0.0, 1.0)
    if list(output.columns) != EXPECTED or not np.isfinite(output[TARGETS]).all().all():
        raise AssertionError("rank stack failed the submission contract")

    output.to_csv("submission.csv", index=False)
    output.to_csv("submission_h35_rank_stack.csv", index=False)
    print(json.dumps({
        "rows": len(output),
        "columns": len(output.columns),
        "sources": sorted(active),
        "weights": {k: active[k] / normalizer for k in active},
        "output": "submission.csv",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
