#!/usr/bin/env python3
"""Avalia um baseline visual linear sobre embeddings por estudo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_baseline.constants import KEY_COLUMN, TARGET_COLUMNS
from rsna_knee_baseline.data import find_data_dir, load_competition_tables


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_visual_rows(index_path: Path, train: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = index.get("records", [])
    if not records:
        raise ValueError("O índice de embeddings não contém records.")

    embedding_path = index_path.parent / str(index.get("embedding_path", "embeddings.npy"))
    matrix = np.load(embedding_path)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError(f"Matriz incompatível com o índice: shape={matrix.shape}, records={len(records)}.")
    if not np.isfinite(matrix).all():
        raise ValueError("A matriz de embeddings contém NaN ou infinito.")

    train_by_study = train.copy()
    train_by_study[KEY_COLUMN] = train_by_study[KEY_COLUMN].astype(str)
    if train_by_study[KEY_COLUMN].duplicated().any():
        raise ValueError("train.csv contém estudos duplicados.")

    study_uids = [str(record["study_uid"]) for record in records]
    if len(set(study_uids)) != len(study_uids):
        raise ValueError("O índice contém estudos duplicados.")
    missing = sorted(set(study_uids) - set(train_by_study[KEY_COLUMN]))
    if missing:
        raise ValueError(f"Estudos de embedding ausentes no train.csv: {missing[:3]}")
    frame = train_by_study.set_index(KEY_COLUMN).loc[study_uids].reset_index()
    return matrix.astype(np.float32), frame, index


def _fold_count(y: np.ndarray, requested: int) -> int:
    if requested < 2:
        raise ValueError("--folds precisa ser pelo menos 2.")
    counts = np.bincount(y.astype(int), minlength=2)
    folds = min(requested, int(counts.min()))
    if folds < 2:
        raise ValueError(
            f"Rótulo insuficiente para validação estratificada: negativos={counts[0]}, positivos={counts[1]}."
        )
    return folds


def evaluate_embeddings(
    matrix: np.ndarray,
    frame: pd.DataFrame,
    folds: int = 5,
    seed: int = 42,
    c: float = 1.0,
) -> dict[str, object]:
    if matrix.shape[0] != len(frame):
        raise ValueError("Número de embeddings e estudos não coincide.")
    if c <= 0:
        raise ValueError("C precisa ser positivo.")

    target_results: list[dict[str, object]] = []
    aucs: list[float] = []
    for target in TARGET_COLUMNS:
        labels = pd.to_numeric(frame.get(target), errors="coerce")
        labeled = labels.notna().to_numpy()
        y = labels.loc[labeled].to_numpy(dtype=np.int64)
        x = matrix[labeled]
        used_folds = _fold_count(y, folds)
        splitter = StratifiedKFold(n_splits=used_folds, shuffle=True, random_state=seed)
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for train_positions, validation_positions in splitter.split(x, y):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear"),
            )
            model.fit(x[train_positions], y[train_positions])
            oof[validation_positions] = model.predict_proba(x[validation_positions])[:, 1]
        auc = float(roc_auc_score(y, oof))
        aucs.append(auc)
        target_results.append(
            {
                "target": target,
                "labeled": int(len(y)),
                "positive": int((y == 1).sum()),
                "negative": int((y == 0).sum()),
                "folds_used": used_folds,
                "auc": auc,
            }
        )

    return {
        "model": "visual_embedding_logistic_regression",
        "embedding_shape": list(matrix.shape),
        "study_count": int(len(frame)),
        "requested_folds": folds,
        "seed": seed,
        "c": c,
        "study_level_split": True,
        "macro_auc": float(np.mean(aucs)),
        "targets": target_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--index", type=Path, default=Path("data/processed/dicom_embeddings_efficientnet_b0_labeled/index.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("reports/visual_embeddings_cv.json"))
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    tables = load_competition_tables(data_dir)
    train = tables["train"].reset_index(drop=True)
    matrix, frame, index = _load_visual_rows(_resolve(args.index), train)
    result = evaluate_embeddings(matrix, frame, args.folds, args.seed, args.c)
    result["source_index"] = str(_resolve(args.index))
    result["weights_sha256"] = index.get("weights_sha256")

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"studies={result['study_count']} embedding_shape={result['embedding_shape']} folds={args.folds} seed={args.seed}")
    print(f"macro_auc={result['macro_auc']:.6f}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
