#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — testa representação por plano com máscara de presença.

O treino usa embeddings visuais fracos de 700 estudos e o holdout usa os 58
estudos com rótulo oficial local. O teste é deliberadamente separado do
backbone: compara pooling global, concatenação Sagittal/Coronal/Axial com
presença explícita e um ensemble de modelos por plano.
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


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _source_index_path(embedding_index_path: Path, payload: dict[str, object]) -> Path:
    source = Path(str(payload["source_index"]))
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise FileNotFoundError(f"source_index não encontrado: {source}")
    return source


def _load_index(path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("records", []))
    matrix = np.load(path.parent / str(payload.get("embedding_path", "embeddings.npy"))).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records) or not np.isfinite(matrix).all():
        raise ValueError(f"Índice/matriz inválidos: shape={matrix.shape}, records={len(records)}")

    source_path = _source_index_path(path, payload)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_records = list(source_payload.get("records", []))
    if len(source_records) != len(records):
        raise ValueError(
            f"Embedding/source desalinhados: embeddings={len(records)}, source={len(source_records)}"
        )

    enriched: list[dict[str, object]] = []
    for row, (record, source_record) in enumerate(zip(records, source_records, strict=True)):
        for key in ("study_uid", "series_uid"):
            if str(record.get(key)) != str(source_record.get(key)):
                raise ValueError(f"UID desalinhado na linha {row}: {key}")
        plane = source_record.get("anatomical_plane")
        if plane not in PLANES:
            raise ValueError(f"Plano inválido na linha {row}: {plane!r}")
        enriched.append(
            {
                "study_uid": str(record["study_uid"]),
                "series_uid": str(record["series_uid"]),
                "anatomical_plane": str(plane),
            }
        )
    return matrix, enriched


def _study_plane_rows(records: list[dict[str, object]]) -> OrderedDict[str, dict[str, list[int]]]:
    groups: OrderedDict[str, dict[str, list[int]]] = OrderedDict()
    for row, record in enumerate(records):
        study = str(record["study_uid"])
        plane = str(record["anatomical_plane"])
        groups.setdefault(study, {plane_name: [] for plane_name in PLANES})[plane].append(row)
    return groups


def _pooled_features(
    matrix: np.ndarray,
    groups: OrderedDict[str, dict[str, list[int]]],
    study_ids: list[str],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Retorna média global, médias por plano e máscara de presença."""

    all_rows: list[np.ndarray] = []
    per_study: dict[str, dict[str, np.ndarray]] = {}
    masks: list[np.ndarray] = []
    for study in study_ids:
        plane_vectors: dict[str, np.ndarray] = {}
        mask = np.zeros(len(PLANES), dtype=np.float32)
        for plane_idx, plane in enumerate(PLANES):
            rows = groups[study][plane]
            if rows:
                plane_vectors[plane] = matrix[rows].mean(axis=0)
                mask[plane_idx] = 1.0
        per_study[study] = plane_vectors
        all_rows.append(np.vstack(list(plane_vectors.values())).mean(axis=0))
        masks.append(mask)
    return np.stack(all_rows), per_study, np.stack(masks)


def _concat_presence(
    per_study: dict[str, dict[str, np.ndarray]],
    masks: np.ndarray,
    study_ids: list[str],
    feature_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row_idx, study in enumerate(study_ids):
        chunks = [per_study[study].get(plane, np.zeros(feature_size, dtype=np.float32)) for plane in PLANES]
        rows.append(np.concatenate([*chunks, masks[row_idx]]))
    return np.stack(rows)


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
    threshold: float = 0.5,
    c: float = 0.5,
) -> dict[str, object]:
    train_matrix, train_records = _load_index(train_index)
    gold_matrix, gold_records = _load_index(gold_index)
    train_groups = _study_plane_rows(train_records)
    gold_groups = _study_plane_rows(gold_records)
    train_ids = list(train_groups)
    gold_ids = list(gold_groups)

    teacher = pd.read_csv(teacher_path).set_index("StudyInstanceUID")
    teacher.index = teacher.index.astype(str)
    official = pd.read_csv(official_path).set_index("StudyInstanceUID")
    official.index = official.index.astype(str)
    missing_teacher = sorted(set(train_ids) - set(teacher.index))
    missing_official = sorted(set(gold_ids) - set(official.index))
    if missing_teacher or missing_official:
        raise ValueError(f"IDs ausentes: teacher={len(missing_teacher)}, official={len(missing_official)}")
    targets = [column for column in official.columns if column in teacher.columns]
    if len(targets) != 12:
        raise ValueError(f"Esperava 12 alvos comuns entre official e teacher; encontrados {len(targets)}")

    train_all, train_by_plane, train_masks = _pooled_features(train_matrix, train_groups, train_ids)
    gold_all, gold_by_plane, gold_masks = _pooled_features(gold_matrix, gold_groups, gold_ids)
    feature_size = train_all.shape[1]
    train_concat = _concat_presence(train_by_plane, train_masks, train_ids, feature_size)
    gold_concat = _concat_presence(gold_by_plane, gold_masks, gold_ids, feature_size)

    methods = ("mean_all", "concat_presence", "plane_ensemble")
    results: dict[str, object] = {
        "tags": ["RSNA", "Kaggle", "Dados"],
        "format": "plane-presence-weak-train-official-gold-v0",
        "selection_bias_warning": (
            "O treino usa 700 estudos escolhidos por extremos dos weak labels; o gold é independente, "
            "mas o holdout tem somente 58 estudos e não contém série Axial. O resultado é um gate local, "
            "não uma estimativa de leaderboard."
        ),
        "train_studies": len(train_ids),
        "gold_studies": len(gold_ids),
        "train_series": len(train_records),
        "gold_series": len(gold_records),
        "planes": list(PLANES),
        "train_plane_coverage": {plane: int(train_masks[:, idx].sum()) for idx, plane in enumerate(PLANES)},
        "gold_plane_coverage": {plane: int(gold_masks[:, idx].sum()) for idx, plane in enumerate(PLANES)},
        "threshold": threshold,
        "c": c,
        "targets": targets,
        "train_feature_shapes": {
            "mean_all": list(train_all.shape),
            "concat_presence": list(train_concat.shape),
        },
        "gold_feature_shapes": {
            "mean_all": list(gold_all.shape),
            "concat_presence": list(gold_concat.shape),
        },
        "targets": {},
    }

    method_predictions: dict[str, dict[str, np.ndarray]] = {method: {} for method in methods}
    for target in targets:
        y_train = (teacher.loc[train_ids, target].to_numpy(dtype=float) >= threshold).astype(np.int8)
        y_gold = official.loc[gold_ids, target].to_numpy(dtype=float)

        global_model = _model(c)
        global_model.fit(train_all, y_train)
        method_predictions["mean_all"][target] = global_model.predict_proba(gold_all)[:, 1]

        concat_model = _model(c)
        concat_model.fit(train_concat, y_train)
        method_predictions["concat_presence"][target] = concat_model.predict_proba(gold_concat)[:, 1]

        plane_predictions: list[list[float]] = [[] for _ in gold_ids]
        for plane in PLANES:
            plane_train = np.stack([train_by_plane[study][plane] for study in train_ids])
            plane_model = _model(c)
            plane_model.fit(plane_train, y_train)
            present_rows = [
                row_idx for row_idx, study in enumerate(gold_ids) if plane in gold_by_plane[study]
            ]
            if present_rows:
                gold_plane = np.stack([gold_by_plane[gold_ids[row_idx]][plane] for row_idx in present_rows])
                probabilities = plane_model.predict_proba(gold_plane)[:, 1]
                for row_idx, probability in zip(present_rows, probabilities, strict=True):
                    plane_predictions[row_idx].append(float(probability))
        # Todos os estudos do gold têm ao menos um plano. Estudos sem Axial
        # simplesmente não contribuem com uma previsão inexistente.
        ensemble = np.asarray([np.mean(values) for values in plane_predictions], dtype=float)
        if ensemble.shape != (len(gold_ids),):
            raise ValueError("Predições do ensemble por plano desalinhadas")
        method_predictions["plane_ensemble"][target] = ensemble

        results["targets"][target] = {
            "train_positive": int(y_train.sum()),
            "train_negative": int((1 - y_train).sum()),
            "gold_positive": int(y_gold.sum()),
            "gold_negative": int((1 - y_gold).sum()),
            "auc": {method: _auc(y_gold, method_predictions[method][target]) for method in methods},
        }

    for method in methods:
        aucs = [results["targets"][target]["auc"][method] for target in targets]
        results.setdefault("macro_auc", {})[method] = float(np.nanmean(aucs))
    results["delta_vs_mean_all"] = {
        method: float(results["macro_auc"][method] - results["macro_auc"]["mean_all"])
        for method in methods
        if method != "mean_all"
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("reports/plane_presence_holdout.json"))
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.c <= 0:
        raise ValueError("threshold precisa estar em (0,1) e C > 0.")
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
    print(f"gold_plane_coverage={result['gold_plane_coverage']}")
    for method, macro_auc in result["macro_auc"].items():
        print(f"{method} macro_auc={macro_auc:.6f}")
    print(f"delta_vs_mean_all={result['delta_vs_mean_all']}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
