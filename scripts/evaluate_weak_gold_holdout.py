#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — testa weak visual em estudos com rótulo oficial independente."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_index(path: Path) -> tuple[np.ndarray, list[str], OrderedDict[str, list[int]]]:
    index = json.loads(path.read_text(encoding="utf-8"))
    records = index.get("records", [])
    matrix = np.load(path.parent / str(index.get("embedding_path", "embeddings.npy"))).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records) or not np.isfinite(matrix).all():
        raise ValueError(f"Índice/matriz inválidos: shape={matrix.shape}, records={len(records)}")
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for row, record in enumerate(records):
        groups.setdefault(str(record["study_uid"]), []).append(row)
    return matrix, list(groups), groups


def evaluate(
    train_index: Path,
    gold_index: Path,
    teacher_path: Path,
    official_path: Path,
    pooling: str = "mean",
    threshold: float = 0.5,
    c: float = 0.5,
) -> dict[str, object]:
    train_matrix, train_ids, train_groups = _load_index(train_index)
    gold_matrix, gold_ids, gold_groups = _load_index(gold_index)
    teacher = pd.read_csv(teacher_path).set_index("StudyInstanceUID")
    teacher.index = teacher.index.astype(str)
    official = pd.read_csv(official_path).set_index("StudyInstanceUID")
    official.index = official.index.astype(str)
    missing_teacher = sorted(set(train_ids) - set(teacher.index))
    missing_official = sorted(set(gold_ids) - set(official.index))
    if missing_teacher or missing_official:
        raise ValueError(f"IDs ausentes: teacher={len(missing_teacher)}, official={len(missing_official)}")

    if pooling == "mean":
        train_x = np.stack([train_matrix[rows].mean(axis=0) for rows in train_groups.values()])
        gold_x = np.stack([gold_matrix[rows].mean(axis=0) for rows in gold_groups.values()])
        train_labels = teacher.loc[train_ids]
        mode_description = "mean das séries disponíveis por estudo"
    elif pooling == "series":
        train_x = train_matrix
        gold_x = gold_matrix
        train_ids = [study for study in train_ids for _ in train_groups[study]]
        train_labels = teacher.loc[train_ids]
        mode_description = "treino por série e média das previsões no estudo"
    else:
        raise ValueError("pooling precisa ser mean ou series")

    targets = list(teacher.columns)
    results: dict[str, object] = {
        "tags": ["RSNA", "Kaggle", "Dados"],
        "format": "weak-train-official-gold-holdout-v0",
        "selection_bias_warning": (
            "O treino usa 700 estudos escolhidos por extremos dos weak labels; o gold é independente, "
            "mas este teste não substitui CV estratificado nos 58 rótulos oficiais."
        ),
        "pooling": pooling,
        "pooling_description": mode_description,
        "train_studies": len(train_groups),
        "train_rows": len(train_x),
        "gold_studies": len(gold_groups),
        "gold_rows": len(gold_x),
        "train_feature_shape": list(train_x.shape),
        "gold_feature_shape": list(gold_x.shape),
        "threshold": threshold,
        "c": c,
        "targets": {},
    }
    aucs: list[float] = []
    for target in targets:
        y_train = (teacher.loc[train_ids, target].to_numpy(dtype=float) >= threshold).astype(np.int8)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear"),
        )
        model.fit(train_x, y_train)
        series_predictions = model.predict_proba(gold_x)[:, 1]
        if pooling == "series":
            prediction_by_study = {
                study: series_predictions[rows].mean() for study, rows in gold_groups.items()
            }
            predictions = np.asarray([prediction_by_study[study] for study in gold_groups])
        else:
            predictions = series_predictions
        y_gold = official.loc[gold_ids, target].to_numpy(dtype=float)
        auc = float(roc_auc_score(y_gold, predictions))
        aucs.append(auc)
        results["targets"][target] = {
            "auc": auc,
            "train_positive": int(y_train.sum()),
            "train_negative": int((1 - y_train).sum()),
            "gold_positive": int(y_gold.sum()),
            "gold_negative": int((1 - y_gold).sum()),
        }
    results["macro_auc"] = float(np.mean(aucs))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--pooling", choices=("mean", "series"), default="mean")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("reports/weak_gold_holdout.json"))
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.c <= 0:
        raise ValueError("threshold precisa estar em (0,1) e C > 0.")
    result = evaluate(
        _resolve(args.train_index),
        _resolve(args.gold_index),
        _resolve(args.teacher),
        _resolve(args.official),
        args.pooling,
        args.threshold,
        args.c,
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pooling={result['pooling']} train={result['train_studies']} gold={result['gold_studies']}")
    print(f"macro_auc={result['macro_auc']:.6f}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
