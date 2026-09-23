#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — audita texto weak + visual no gold oficial.

O ramo textual é treinado nos 700 estudos weak escolhidos para o benchmark,
usando o teacher target-wise como rótulo binário. O ramo visual usa as mesmas
2.100 séries header dos 700 estudos e compara média global com ensemble por
plano. Os 58 estudos gold ficam exclusivamente na avaliação.

Este é um gate local de direção, não uma estimativa de leaderboard: o lote
weak foi selecionado por labels públicos e a grade de pesos é diagnóstica.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_baseline.model import KneeReportBaseline
from scripts.audit_report_hash_groups import report_hash
from scripts.evaluate_plane_presence_holdout import (
    PLANES,
    _load_index,
    _pooled_features,
    _study_plane_rows,
)


KEY_COLUMN = "StudyInstanceUID"
TARGETS = (
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
)
H27_TARGETWISE_WEIGHTS = {
    "ACL": 0.5,
    "MCL": 0.5,
    "Medial Meniscus": 0.4,
    "Lateral Meniscus": 0.1,
    "Medial OA": 0.0,
    "Lateral OA": 0.1,
    "PF OA": 0.0,
    "Effusion": 0.5,
    "Synovitis": 0.0,
    "Baker's": 0.4,
    "Contusion": 0.2,
    "Fracture": 0.1,
}


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _visual_predictions(
    train_by_plane: dict[str, dict[str, np.ndarray]],
    gold_by_plane: dict[str, dict[str, np.ndarray]],
    train_ids: list[str],
    gold_ids: list[str],
    teacher: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    predictions: dict[str, dict[str, np.ndarray]] = {"mean_all": {}, "plane_ensemble": {}}
    # ``train_by_plane``/``gold_by_plane`` hold one pooled vector per study and
    # plane. Build the global mean explicitly to avoid re-reading arrays.
    train_global = np.stack(
        [np.vstack([train_by_plane[study][plane] for plane in PLANES]).mean(axis=0) for study in train_ids]
    )
    gold_global = np.stack(
        [np.vstack([gold_by_plane[study][plane] for plane in PLANES]).mean(axis=0) for study in gold_ids]
    )
    for target in TARGETS:
        y_train = (teacher.loc[train_ids, target].to_numpy(dtype=float) >= 0.5).astype(np.int8)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, solver="liblinear"),
        )
        model.fit(train_global, y_train)
        predictions["mean_all"][target] = model.predict_proba(gold_global)[:, 1]

        per_plane: list[np.ndarray] = []
        for plane in PLANES:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, solver="liblinear"),
            )
            model.fit(np.stack([train_by_plane[study][plane] for study in train_ids]), y_train)
            values = np.full(len(gold_ids), np.nan, dtype=float)
            present = [index for index, study in enumerate(gold_ids) if plane in gold_by_plane[study]]
            if present:
                values[present] = model.predict_proba(
                    np.stack([gold_by_plane[gold_ids[index]][plane] for index in present])
                )[:, 1]
            per_plane.append(values)
        predictions["plane_ensemble"][target] = np.nanmean(np.stack(per_plane), axis=0)
    return predictions


def _macro_report(
    predictions: dict[str, np.ndarray], official: pd.DataFrame, gold_ids: list[str]
) -> dict[str, object]:
    targets: dict[str, float] = {}
    for target in TARGETS:
        y = official.loc[gold_ids, target].to_numpy(dtype=float)
        targets[target] = float(roc_auc_score(y, predictions[target]))
    return {"macro_auc": float(np.mean(list(targets.values()))), "targets": targets}


def evaluate(
    train_index: Path,
    gold_index: Path,
    teacher_path: Path,
    official_path: Path,
    train_path: Path,
    alphas: tuple[float, ...],
) -> dict[str, object]:
    train_matrix_raw, train_records = _load_index(train_index)
    gold_matrix_raw, gold_records = _load_index(gold_index)
    train = pd.read_csv(train_path).set_index(KEY_COLUMN)
    train.index = train.index.astype(str)
    teacher = pd.read_csv(teacher_path).set_index(KEY_COLUMN)
    teacher.index = teacher.index.astype(str)
    official = pd.read_csv(official_path).set_index(KEY_COLUMN)
    official.index = official.index.astype(str)

    train_groups_all = _study_plane_rows(train_records)
    gold_groups = _study_plane_rows(gold_records)
    all_train_ids = list(train_groups_all)
    gold_ids = list(gold_groups)
    for name, ids, frame in (("teacher", all_train_ids, teacher), ("official", gold_ids, official)):
        missing = sorted(set(ids) - set(frame.index))
        if missing:
            raise ValueError(f"{name} sem {len(missing)} estudos; exemplo={missing[:3]}")

    weak_hashes = train.loc[all_train_ids, "Report"].map(report_hash)
    gold_hashes = train.loc[gold_ids, "Report"].map(report_hash)
    overlap_hashes = set(weak_hashes) & set(gold_hashes)
    blocked_train_ids = [study for study in all_train_ids if weak_hashes.loc[study] in overlap_hashes]
    train_ids = [study for study in all_train_ids if study not in set(blocked_train_ids)]
    train_groups = {study: train_groups_all[study] for study in train_ids}
    _, train_by_plane, train_masks = _pooled_features(train_matrix_raw, train_groups, train_ids)
    _, gold_by_plane, gold_masks = _pooled_features(gold_matrix_raw, gold_groups, gold_ids)

    train_frame = train.loc[train_ids].reset_index()
    gold_frame = train.loc[gold_ids].reset_index()
    series = pd.read_csv(Path(train_path).parent / "train_series.csv")
    train_series = series[series[KEY_COLUMN].astype(str).isin(train_ids)].copy()
    gold_series = series[series[KEY_COLUMN].astype(str).isin(gold_ids)].copy()
    weak_frame = train_frame.copy()
    for target in TARGETS:
        weak_frame[target] = (teacher.loc[train_ids, target].to_numpy(dtype=float) >= 0.5).astype(np.int8)
    text_model = KneeReportBaseline(c=32.0, use_lexicon=True, lexicon_weight=1.0)
    text_model.fit(weak_frame, train_series)
    text_frame = text_model.predict(gold_frame, gold_series)
    text_predictions = {target: text_frame[target].to_numpy(dtype=float) for target in TARGETS}

    visual = _visual_predictions(
        train_by_plane,
        gold_by_plane,
        train_ids,
        gold_ids,
        teacher,
    )
    models: dict[str, dict[str, object]] = {
        "text_weak": _macro_report(text_predictions, official, gold_ids),
        "visual_mean_all": _macro_report(visual["mean_all"], official, gold_ids),
        "visual_plane_ensemble": _macro_report(visual["plane_ensemble"], official, gold_ids),
    }
    for alpha in alphas:
        if not 0 <= alpha <= 1:
            raise ValueError("Pesos alpha precisam estar em [0,1].")
        blended = {
            target: (1.0 - alpha) * text_predictions[target] + alpha * visual["plane_ensemble"][target]
            for target in TARGETS
        }
        models[f"blend_plane_alpha_{alpha:g}"] = _macro_report(blended, official, gold_ids)
    h27_blend = {
        target: (1.0 - H27_TARGETWISE_WEIGHTS[target]) * text_predictions[target]
        + H27_TARGETWISE_WEIGHTS[target] * visual["plane_ensemble"][target]
        for target in TARGETS
    }
    models["blend_h27_targetwise"] = _macro_report(h27_blend, official, gold_ids)
    return {
        "tags": ["RSNA", "Kaggle", "Pesquisa"],
        "format": "weak-gold-text-visual-blend-v1",
        "warning": (
            "Gate local: os estudos weak foram selecionados por labels públicos; "
            "hashes de laudo compartilhados com o gold foram removidos; os gold "
            "são usados somente na avaliação; alpha não é score Kaggle."
        ),
        "train_studies": len(train_ids),
        "gold_studies": len(gold_ids),
        "train_studies_before_hash_block": len(all_train_ids),
        "blocked_train_studies_by_report_hash": blocked_train_ids,
        "report_hash_overlap_count": len(overlap_hashes),
        "train_series": sum(len(rows) for rows in train_groups.values()),
        "gold_series": len(gold_records),
        "train_plane_coverage": {plane: int(train_masks[:, index].sum()) for index, plane in enumerate(PLANES)},
        "gold_plane_coverage": {plane: int(gold_masks[:, index].sum()) for index, plane in enumerate(PLANES)},
        "alphas": list(alphas),
        "models": models,
        "h27_targetwise_weights": H27_TARGETWISE_WEIGHTS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, default=Path("data/external_labels/targetwise_teacher.csv"))
    parser.add_argument("--official", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--alphas", default="0,0.1,0.2,0.25,0.4,0.5")
    parser.add_argument("--output", type=Path, default=Path("reports/weak_gold_text_visual_blend_20260820.json"))
    args = parser.parse_args()
    alphas = tuple(float(value.strip()) for value in args.alphas.split(",") if value.strip())
    if not alphas:
        raise ValueError("--alphas não pode ser vazio.")
    result = evaluate(
        _resolve(args.train_index),
        _resolve(args.gold_index),
        _resolve(args.teacher),
        _resolve(args.official),
        _resolve(args.train),
        alphas,
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"train={result['train_studies']} gold={result['gold_studies']}")
    for name, report in result["models"].items():
        print(f"{name} macro_auc={report['macro_auc']:.6f}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
