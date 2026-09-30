#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — mede sinal visual contra labels weak, com alerta de seleção."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def load_study_embeddings(index_path: Path) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, object]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = index.get("records", [])
    if not records:
        raise ValueError("O índice de embeddings não contém records.")
    matrix = np.load(index_path.parent / str(index.get("embedding_path", "embeddings.npy"))).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError(f"Matriz incompatível: shape={matrix.shape}, records={len(records)}.")
    if not np.isfinite(matrix).all():
        raise ValueError("A matriz de embeddings contém NaN ou infinito.")

    rows_by_study: OrderedDict[str, list[int]] = OrderedDict()
    for row, record in enumerate(records):
        rows_by_study.setdefault(str(record["study_uid"]), []).append(row)
    if any(len(rows) != 3 for rows in rows_by_study.values()):
        raise ValueError("Cada estudo precisa ter exatamente três séries para esta avaliação.")

    mean_matrix = np.stack([matrix[rows].mean(axis=0) for rows in rows_by_study.values()])
    concat_matrix = np.stack([matrix[rows].reshape(-1) for rows in rows_by_study.values()])
    return list(rows_by_study), mean_matrix, concat_matrix, index


def _evaluate_matrix(
    matrix: np.ndarray,
    labels: pd.DataFrame,
    targets: list[str],
    threshold: float,
    folds: int,
    seed: int,
    c: float,
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for target in targets:
        values = pd.to_numeric(labels[target], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        y = (values[valid] >= threshold).astype(np.int8)
        x = matrix[valid]
        counts = np.bincount(y, minlength=2)
        if counts.min() < 2:
            continue
        used_folds = min(folds, int(counts.min()))
        oof = np.full(len(y), np.nan, dtype=np.float64)
        splitter = StratifiedKFold(n_splits=used_folds, shuffle=True, random_state=seed)
        for train_positions, validation_positions in splitter.split(x, y):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                ),
            )
            model.fit(x[train_positions], y[train_positions])
            oof[validation_positions] = model.predict_proba(x[validation_positions])[:, 1]
        results[target] = {
            "positive": int(counts[1]),
            "negative": int(counts[0]),
            "folds": used_folds,
            "auc": float(roc_auc_score(y, oof)),
        }
    return results


def evaluate(
    index_path: Path,
    teacher_path: Path,
    threshold: float = 0.5,
    folds: int = 5,
    seed: int = 2026,
    c: float = 0.5,
) -> dict[str, object]:
    study_ids, mean_matrix, concat_matrix, index = load_study_embeddings(index_path)
    teacher = pd.read_csv(teacher_path).set_index("StudyInstanceUID")
    missing = sorted(set(study_ids) - set(teacher.index.astype(str)))
    if missing:
        raise ValueError(f"Teacher sem {len(missing)} estudos locais; exemplos: {missing[:3]}")
    teacher.index = teacher.index.astype(str)
    labels = teacher.loc[study_ids]
    targets = [column for column in labels.columns if column != "StudyInstanceUID"]

    target_results: dict[str, dict[str, object]] = {}
    for name, matrix in (("mean", mean_matrix), ("concat", concat_matrix)):
        target_results[name] = _evaluate_matrix(matrix, labels, targets, threshold, folds, seed, c)

    macro = {
        name: float(np.mean([row["auc"] for row in rows.values()]))
        for name, rows in target_results.items()
    }
    return {
        "tags": ["RSNA", "Kaggle", "Dados"],
        "format": "weak-visual-budget-diagnostic-v0",
        "selection_bias_warning": (
            "Os estudos foram selecionados por extremos dos próprios weak labels; este resultado é diagnóstico, "
            "não substitui CV nos 58 rótulos oficiais."
        ),
        "source_index": str(index_path),
        "source_teacher": str(teacher_path),
        "studies": len(study_ids),
        "series": len(index["records"]),
        "threshold": threshold,
        "folds": folds,
        "seed": seed,
        "c": c,
        "features": {"mean": list(mean_matrix.shape), "concat": list(concat_matrix.shape)},
        "macro_auc": macro,
        "targets": target_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("reports/weak_visual_budget_diagnostic.json"))
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.folds < 2 or args.c <= 0:
        raise ValueError("threshold precisa estar em (0,1), folds >= 2 e C > 0.")

    index_path = _resolve(args.index)
    teacher_path = _resolve(args.teacher)
    result = evaluate(index_path, teacher_path, args.threshold, args.folds, args.seed, args.c)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"studies={result['studies']} series={result['series']} features={result['features']}")
    print(f"macro_auc_mean={result['macro_auc']['mean']:.6f}")
    print(f"macro_auc_concat={result['macro_auc']['concat']:.6f}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
