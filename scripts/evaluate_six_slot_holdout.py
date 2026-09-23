#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — gate local para slots adicionais de aquisição.

Compara o ensemble H-27 de uma série fluido/FS por plano com agregações que
usam todas as séries observadas ou cabeças separadas por plano/categoria. O
treino continua nos 700 estudos weak e o gold são os 58 estudos oficiais;
portanto o resultado é uma ablação de direção, não uma estimativa do
leaderboard.
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
PLANES = ("Sagittal", "Coronal", "Axial")
SLOTS = tuple(f"{plane}_{category}" for plane in PLANES for category in ("FLUID_FS", "NONFLUID"))


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _category(fluid_sensitive: object, fat_suppression: object) -> str:
    fluid = int(float(fluid_sensitive or 0))
    fat = int(float(fat_suppression or 0))
    if fluid == 1 and fat == 1:
        return "FLUID_FS"
    if fluid == 0 and fat == 0:
        return "NONFLUID"
    return "OTHER"


def _slot(record: dict[str, object]) -> str:
    return f"{record['anatomical_plane']}_{_category(record.get('fluid_sensitive'), record.get('fat_suppression'))}"


def _load_bundle(index_path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    matrix_path = index_path.parent / str(payload.get("embedding_path", "embeddings.npy"))
    matrix = np.load(matrix_path).astype(np.float32)
    records = list(payload.get("records", []))
    source_value = payload.get("source_index")
    if source_value is None:
        raise ValueError(f"Índice sem source_index: {index_path}")
    source = Path(str(source_value))
    if not source.is_absolute():
        source = ROOT / source
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    source_records = list(source_payload.get("records", []))
    if len(records) != len(source_records) or matrix.shape[0] != len(records):
        raise ValueError(f"Bundle desalinhado: matrix={matrix.shape}, records={len(records)}, source={len(source_records)}")
    enriched: list[dict[str, object]] = []
    for row, (record, source_record) in enumerate(zip(records, source_records, strict=True)):
        if str(record.get("study_uid")) != str(source_record.get("study_uid")):
            raise ValueError(f"Study UID desalinhado na linha {row}")
        enriched.append(
            {
                "study_uid": str(source_record["study_uid"]),
                "series_uid": str(source_record["series_uid"]),
                "anatomical_plane": str(source_record.get("anatomical_plane", "")),
                "fluid_sensitive": source_record.get("fluid_sensitive"),
                "fat_suppression": source_record.get("fat_suppression"),
            }
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Embeddings não finitos: {index_path}")
    return matrix, enriched


def _groups(records: list[dict[str, object]]) -> OrderedDict[str, dict[str, list[int]]]:
    groups: OrderedDict[str, dict[str, list[int]]] = OrderedDict()
    for row, record in enumerate(records):
        study = str(record["study_uid"])
        plane = str(record["anatomical_plane"])
        if plane not in PLANES:
            raise ValueError(f"Plano inválido: {plane!r}")
        slot = _slot(record)
        groups.setdefault(study, {key: [] for key in (*PLANES, *SLOTS)})
        groups[study][plane].append(row)
        if slot in SLOTS:
            groups[study][slot].append(row)
    return groups


def _mean_feature(matrix: np.ndarray, rows: list[int]) -> np.ndarray:
    if not rows:
        raise ValueError("Não é possível calcular média sem linhas")
    return matrix[rows].mean(axis=0)


def _study_features(
    matrix: np.ndarray,
    groups: OrderedDict[str, dict[str, list[int]]],
    study_ids: list[str],
    key: str,
    fallback: str | None = None,
    allow_missing: bool = False,
    first_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    present: list[float] = []
    for study in study_ids:
        selected = groups[study][key]
        if not selected and fallback is not None:
            selected = groups[study][fallback]
        if not selected and allow_missing:
            rows.append(np.zeros(matrix.shape[1], dtype=np.float32))
            present.append(0.0)
            continue
        if first_only:
            selected = selected[:1]
        rows.append(_mean_feature(matrix, selected))
        present.append(float(bool(selected)))
    return np.stack(rows), np.asarray(present, dtype=np.float32)


def _model(c: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear"),
    )


def _auc(y_gold: np.ndarray, predictions: np.ndarray) -> float:
    if np.unique(y_gold).size < 2:
        return float("nan")
    return float(roc_auc_score(y_gold, predictions))


def evaluate(
    train_index: Path,
    gold_index: Path,
    teacher_path: Path,
    official_path: Path,
    threshold: float,
    c: float,
) -> dict[str, object]:
    train_matrix, train_records = _load_bundle(train_index)
    gold_matrix, gold_records = _load_bundle(gold_index)
    train_groups = _groups(train_records)
    gold_groups = _groups(gold_records)
    train_ids = list(train_groups)
    gold_ids = list(gold_groups)

    teacher = pd.read_csv(teacher_path).set_index("StudyInstanceUID")
    teacher.index = teacher.index.astype(str)
    official = pd.read_csv(official_path).set_index("StudyInstanceUID")
    official.index = official.index.astype(str)
    targets = [column for column in official.columns if column in teacher.columns]
    if len(targets) != 12:
        raise ValueError(f"Esperava 12 alvos comuns; encontrados {len(targets)}")
    if set(train_ids) - set(teacher.index) or set(gold_ids) - set(official.index):
        raise ValueError("IDs ausentes em teacher ou official")

    methods = ("plane_preferred", "plane_all_series", "slot_ensemble", "slot_fallback")
    results: dict[str, object] = {
        "tags": ["RSNA", "Kaggle", "Dados"],
        "format": "six-slot-weak-train-official-gold-v0",
        "selection_bias_warning": (
            "Ablação local nos 700 estudos weak e 58 oficiais; o treino não contém todas as categorias "
            "de aquisição e o gold pequeno não substitui o leaderboard."
        ),
        "train_studies": len(train_ids),
        "gold_studies": len(gold_ids),
        "train_series": len(train_records),
        "gold_series": len(gold_records),
        "threshold": threshold,
        "c": c,
        "slots": list(SLOTS),
        "train_slot_coverage": {slot: int(sum(bool(train_groups[study][slot]) for study in train_ids)) for slot in SLOTS},
        "gold_slot_coverage": {slot: int(sum(bool(gold_groups[study][slot]) for study in gold_ids)) for slot in SLOTS},
        "targets": {},
    }

    train_plane_preferred: dict[str, np.ndarray] = {}
    gold_plane_preferred: dict[str, np.ndarray] = {}
    train_plane_all: dict[str, np.ndarray] = {}
    gold_plane_all: dict[str, np.ndarray] = {}
    train_slot: dict[str, np.ndarray] = {}
    gold_slot: dict[str, np.ndarray] = {}
    train_slot_presence: dict[str, np.ndarray] = {}
    gold_slot_presence: dict[str, np.ndarray] = {}
    for plane in PLANES:
        train_plane_preferred[plane], _ = _study_features(
            train_matrix, train_groups, train_ids, f"{plane}_FLUID_FS", plane, first_only=True
        )
        gold_plane_preferred[plane], _ = _study_features(
            gold_matrix, gold_groups, gold_ids, f"{plane}_FLUID_FS", plane, first_only=True
        )
        train_plane_all[plane], _ = _study_features(train_matrix, train_groups, train_ids, plane)
        gold_plane_all[plane], _ = _study_features(gold_matrix, gold_groups, gold_ids, plane)
    for slot in SLOTS:
        train_slot[slot], train_slot_presence[slot] = _study_features(
            train_matrix, train_groups, train_ids, slot, allow_missing=True
        )
        gold_slot[slot], gold_slot_presence[slot] = _study_features(
            gold_matrix, gold_groups, gold_ids, slot, allow_missing=True
        )

    for target in targets:
        y_train = (teacher.loc[train_ids, target].to_numpy(dtype=float) >= threshold).astype(np.int8)
        y_gold = official.loc[gold_ids, target].to_numpy(dtype=float)
        preferred_models: dict[str, object] = {}
        all_models: dict[str, object] = {}
        for plane in PLANES:
            preferred_models[plane] = _model(c).fit(train_plane_preferred[plane], y_train)
            all_models[plane] = _model(c).fit(train_plane_all[plane], y_train)
        slot_models: dict[str, object | None] = {}
        for slot in SLOTS:
            train_rows = train_slot_presence[slot] > 0
            if train_rows.any() and np.unique(y_train[train_rows]).size >= 2:
                slot_models[slot] = _model(c).fit(train_slot[slot][train_rows], y_train[train_rows])
            else:
                slot_models[slot] = None

        predictions: dict[str, list[float]] = {method: [] for method in methods}

        for row_idx in range(len(gold_ids)):
            plane_values_preferred: list[float] = []
            plane_values_all: list[float] = []
            slot_values: list[float] = []
            fallback_values: list[float] = []
            for plane in PLANES:
                plane_values_preferred.append(float(preferred_models[plane].predict_proba(gold_plane_preferred[plane][row_idx : row_idx + 1])[:, 1][0]))
                plane_values_all.append(float(all_models[plane].predict_proba(gold_plane_all[plane][row_idx : row_idx + 1])[:, 1][0]))

            for slot in SLOTS:
                if gold_slot_presence[slot][row_idx] == 0:
                    continue
                plane = slot.split("_", 1)[0]
                slot_model = slot_models[slot]
                if slot_model is not None:
                    value = float(slot_model.predict_proba(gold_slot[slot][row_idx : row_idx + 1])[:, 1][0])
                    slot_values.append(value)
                    fallback_values.append(value)
                else:
                    fallback_values.append(
                        float(preferred_models[plane].predict_proba(gold_plane_preferred[plane][row_idx : row_idx + 1])[:, 1][0])
                    )

            predictions["plane_preferred"].append(float(np.mean(plane_values_preferred)))
            predictions["plane_all_series"].append(float(np.mean(plane_values_all)))
            predictions["slot_ensemble"].append(float(np.mean(slot_values)) if slot_values else 0.5)
            predictions["slot_fallback"].append(float(np.mean(fallback_values)) if fallback_values else 0.5)

        results["targets"][target] = {
            "train_positive": int(y_train.sum()),
            "train_negative": int((1 - y_train).sum()),
            "gold_positive": int(y_gold.sum()),
            "gold_negative": int((1 - y_gold).sum()),
            "auc": {method: _auc(y_gold, np.asarray(predictions[method])) for method in methods},
        }

    results["macro_auc"] = {
        method: float(np.nanmean([results["targets"][target]["auc"][method] for target in targets]))
        for method in methods
    }
    results["delta_vs_plane_preferred"] = {
        method: float(results["macro_auc"][method] - results["macro_auc"]["plane_preferred"])
        for method in methods
        if method != "plane_preferred"
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("reports/six_slot_holdout.json"))
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.c <= 0:
        raise ValueError("threshold precisa estar em (0,1) e C > 0")
    result = evaluate(
        _resolve(args.train_index),
        _resolve(args.gold_index),
        _resolve(args.teacher),
        _resolve(args.official),
        args.threshold,
        args.c,
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"train={result['train_studies']} gold={result['gold_studies']}")
    print(f"train_slot_coverage={result['train_slot_coverage']}")
    print(f"gold_slot_coverage={result['gold_slot_coverage']}")
    print(f"macro_auc={result['macro_auc']}")
    print(f"delta_vs_plane_preferred={result['delta_vs_plane_preferred']}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
