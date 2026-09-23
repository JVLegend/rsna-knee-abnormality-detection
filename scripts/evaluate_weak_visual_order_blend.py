#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — mistura fixa de representações visuais no holdout gold.

Treina o mesmo classificador weak-supervised em dois índices de embeddings
(por exemplo, ordem lexicográfica e ordem anatômica) e mistura as previsões no
gold oficial. O peso ``0,5`` é a candidata pré-especificada; a grade opcional
serve somente para diagnosticar complementaridade e não deve ser tratada como
calibração válida do leaderboard.
"""

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


def _load_bundle(path: Path) -> dict[str, object]:
    index = json.loads(path.read_text(encoding="utf-8"))
    records = [dict(record) for record in index.get("records", [])]
    matrix = np.load(path.parent / str(index.get("embedding_path", "embeddings.npy"))).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records) or not np.isfinite(matrix).all():
        raise ValueError(f"Índice/matriz inválidos em {path}: shape={matrix.shape}, records={len(records)}")
    groups: OrderedDict[str, list[int]] = OrderedDict()
    row_studies: list[str] = []
    for row, record in enumerate(records):
        study_uid = str(record["study_uid"])
        row_studies.append(study_uid)
        groups.setdefault(study_uid, []).append(row)
    return {
        "path": str(path),
        "matrix": matrix,
        "records": records,
        "groups": groups,
        "row_studies": row_studies,
        "study_ids": list(groups),
    }


def _study_mean(matrix: np.ndarray, groups: OrderedDict[str, list[int]]) -> np.ndarray:
    return np.stack([matrix[rows].mean(axis=0) for rows in groups.values()])


def _fit_predictions(
    train_bundle: dict[str, object],
    gold_bundle: dict[str, object],
    teacher: pd.DataFrame,
    pooling: str,
) -> dict[str, np.ndarray]:
    if pooling not in {"mean", "series"}:
        raise ValueError("pooling precisa ser mean ou series")
    train_matrix = train_bundle["matrix"]
    gold_matrix = gold_bundle["matrix"]
    train_groups = train_bundle["groups"]
    gold_groups = gold_bundle["groups"]
    if pooling == "mean":
        train_matrix = _study_mean(train_matrix, train_groups)
        gold_matrix = _study_mean(gold_matrix, gold_groups)
        train_ids = list(train_groups)
    else:
        train_ids = train_bundle["row_studies"]

    predictions: dict[str, np.ndarray] = {}
    for target in teacher.columns:
        y_train = (teacher.loc[train_ids, target].to_numpy(dtype=float) >= 0.5).astype(np.int8)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, solver="liblinear"),
        )
        model.fit(train_matrix, y_train)
        values = model.predict_proba(gold_matrix)[:, 1]
        if pooling == "series":
            values = np.asarray(
                [values[gold_groups[study_uid]].mean() for study_uid in gold_bundle["study_ids"]]
            )
        predictions[target] = values
    return predictions


def _auc_report(predictions: dict[str, np.ndarray], official: pd.DataFrame, gold_ids: list[str]) -> dict[str, object]:
    target_results = {}
    aucs: list[float] = []
    for target, values in predictions.items():
        y_gold = official.loc[gold_ids, target].to_numpy(dtype=float)
        auc = float(roc_auc_score(y_gold, values))
        target_results[target] = {"auc": auc}
        aucs.append(auc)
    return {"macro_auc": float(np.mean(aucs)), "targets": target_results}


def evaluate(
    train_a_index: Path,
    train_b_index: Path,
    gold_a_index: Path,
    gold_b_index: Path,
    teacher_path: Path,
    official_path: Path,
    pooling: str,
    alpha: float,
    grid: list[float],
) -> dict[str, object]:
    train_a = _load_bundle(train_a_index)
    train_b = _load_bundle(train_b_index)
    gold_a = _load_bundle(gold_a_index)
    gold_b = _load_bundle(gold_b_index)
    if train_a["study_ids"] != train_b["study_ids"] or gold_a["study_ids"] != gold_b["study_ids"]:
        raise ValueError("Os índices A/B não têm a mesma cobertura de estudos")

    teacher = pd.read_csv(teacher_path).set_index("StudyInstanceUID")
    teacher.index = teacher.index.astype(str)
    official = pd.read_csv(official_path).set_index("StudyInstanceUID")
    official.index = official.index.astype(str)
    for study_uid in train_a["study_ids"]:
        if study_uid not in teacher.index:
            raise ValueError(f"Teacher sem estudo de treino: {study_uid}")
    gold_ids = gold_a["study_ids"]
    missing_official = sorted(set(gold_ids) - set(official.index))
    if missing_official:
        raise ValueError(f"Oficial sem estudos gold: {len(missing_official)}")
    official = official.loc[gold_ids, teacher.columns].apply(pd.to_numeric, errors="coerce")
    if official.isna().any().any():
        raise ValueError("O gold oficial tem rótulos ausentes")

    pred_a = _fit_predictions(train_a, gold_a, teacher, pooling)
    pred_b = _fit_predictions(train_b, gold_b, teacher, pooling)
    pred_fixed = {target: (1.0 - alpha) * pred_a[target] + alpha * pred_b[target] for target in teacher.columns}
    fixed_report = _auc_report(pred_fixed, official, gold_ids)
    branch_a_report = _auc_report(pred_a, official, gold_ids)
    branch_b_report = _auc_report(pred_b, official, gold_ids)

    grid_reports = []
    for value in grid:
        if not 0 <= value <= 1:
            raise ValueError("Todos os pesos da grade precisam estar em [0,1]")
        predictions = {
            target: (1.0 - value) * pred_a[target] + value * pred_b[target]
            for target in teacher.columns
        }
        report = _auc_report(predictions, official, gold_ids)
        grid_reports.append({"alpha_b": value, **report})

    return {
        "tags": ["RSNA", "Kaggle", "Dados"],
        "format": "weak-visual-order-blend-gold-v1",
        "pooling": pooling,
        "alpha_b_fixed": alpha,
        "train_studies": len(train_a["study_ids"]),
        "gold_studies": len(gold_ids),
        "classifier": "StandardScaler + LogisticRegression(C=0.5, class_weight=balanced, solver=liblinear)",
        "branch_a": {"index": train_a["path"], **branch_a_report},
        "branch_b": {"index": train_b["path"], **branch_b_report},
        "fixed_blend": fixed_report,
        "grid_diagnostic": grid_reports,
        "warnings": [
            "O peso fixo é a candidata pré-especificada; a grade usa o gold e serve apenas para medir complementaridade.",
            "O treino usa 700 estudos selecionados por weak labels; o gold tem 58 estudos oficiais independentes.",
            "AUC local não é score de leaderboard e não autoriza substituir a referência sem teste Kaggle.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-a-index", type=Path, required=True)
    parser.add_argument("--train-b-index", type=Path, required=True)
    parser.add_argument("--gold-a-index", type=Path, required=True)
    parser.add_argument("--gold-b-index", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--pooling", choices=("mean", "series"), default="mean")
    parser.add_argument("--alpha-b", type=float, default=0.5)
    parser.add_argument("--grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--output", type=Path, default=Path("reports/weak_visual_order_blend_gold.json"))
    args = parser.parse_args()
    grid = [float(value.strip()) for value in args.grid.split(",") if value.strip()]
    if not 0 <= args.alpha_b <= 1:
        raise ValueError("--alpha-b precisa estar em [0,1]")
    result = evaluate(
        _resolve(args.train_a_index),
        _resolve(args.train_b_index),
        _resolve(args.gold_a_index),
        _resolve(args.gold_b_index),
        _resolve(args.teacher),
        _resolve(args.official),
        args.pooling,
        args.alpha_b,
        grid,
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"pooling={result['pooling']} train={result['train_studies']} gold={result['gold_studies']} "
        f"branch_a={result['branch_a']['macro_auc']:.6f} "
        f"branch_b={result['branch_b']['macro_auc']:.6f} "
        f"fixed_blend={result['fixed_blend']['macro_auc']:.6f}"
    )
    print(f"report={output}")


if __name__ == "__main__":
    main()
