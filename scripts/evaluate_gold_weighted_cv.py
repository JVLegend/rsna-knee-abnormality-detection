#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — testa peso dos 58 labels gold com weak supervision.

O treino local combina embeddings de 700 estudos weak com os estudos gold do
fold de treino; o fold gold de validação permanece separado. Qualquer estudo
weak que compartilhe ``report_hash`` com o fold de validação é removido antes
do ajuste. O resultado é um proxy de estudo, não uma estimativa direta do
leaderboard.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_report_hash_groups import report_hash


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


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_bundle(path: Path) -> tuple[list[str], np.ndarray]:
    payload = json.loads((path / "index.json").read_text(encoding="utf-8"))
    matrix = np.load(path / str(payload.get("embedding_path", "embeddings.npy"))).astype(np.float32)
    records = list(payload.get("records", []))
    if matrix.ndim != 2 or matrix.shape[0] != len(records) or not np.isfinite(matrix).all():
        raise ValueError(f"Bundle inválido em {path}: shape={matrix.shape}, records={len(records)}")
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for row, record in enumerate(records):
        groups[str(record[KEY_COLUMN.lower() if KEY_COLUMN.lower() in record else "study_uid"])].append(row)
    study_ids = sorted(groups)
    study_matrix = np.stack([matrix[groups[study_id]].mean(axis=0) for study_id in study_ids])
    return study_ids, study_matrix


def _fit_model(c: float, x_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> object:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=2000),
    )
    model.fit(x_train, y_train, logisticregression__sample_weight=weights)
    return model


def evaluate(
    weak_index: Path,
    gold_index: Path,
    train_path: Path,
    teacher_path: Path,
    gold_weights: list[float],
    seeds: list[int],
    folds: int,
    c: float,
) -> dict[str, object]:
    weak_ids, weak_matrix = _load_bundle(weak_index)
    gold_ids, gold_matrix = _load_bundle(gold_index)
    if set(weak_ids) & set(gold_ids):
        raise ValueError("Weak e gold compartilham estudos; o holdout não é independente.")

    train = pd.read_csv(train_path).set_index(KEY_COLUMN)
    train.index = train.index.astype(str)
    teacher = pd.read_csv(teacher_path).set_index(KEY_COLUMN)
    teacher.index = teacher.index.astype(str)
    for name, ids, frame in (("weak", weak_ids, teacher), ("gold", gold_ids, train)):
        missing = sorted(set(ids) - set(frame.index))
        if missing:
            raise ValueError(f"{name}: {len(missing)} estudos sem labels; exemplos={missing[:3]}")

    weak_hashes = train.loc[weak_ids, "Report"].map(report_hash).to_numpy()
    gold_hashes = train.loc[gold_ids, "Report"].map(report_hash).to_numpy()
    hash_overlap = sorted(set(weak_hashes) & set(gold_hashes))
    if len(gold_ids) < folds:
        raise ValueError("O número de estudos gold precisa ser >= folds.")

    results: dict[str, object] = {
        "tags": ["RSNA", "Kaggle", "Pesquisa"],
        "format": "rsna-gold-weighted-weak-cv-v1",
        "weak_studies": len(weak_ids),
        "gold_studies": len(gold_ids),
        "feature_dim": int(weak_matrix.shape[1]),
        "hash_overlap_weak_gold": len(hash_overlap),
        "gold_hash_groups": len(set(gold_hashes)),
        "blocked_weak_rows_total": 0,
        "method": "study-level mean embedding + StandardScaler/LogisticRegression",
        "warning": "Proxy local; não é score Kaggle e o lote weak foi selecionado por labels públicos.",
        "weights": {},
    }

    for gold_weight in gold_weights:
        seed_results: dict[str, object] = {}
        macro_values: list[float] = []
        for seed in seeds:
            predictions = np.full((len(gold_ids), len(TARGETS)), np.nan, dtype=np.float64)
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            for target_index, target in enumerate(TARGETS):
                y_gold = pd.to_numeric(train.loc[gold_ids, target], errors="coerce").to_numpy(float)
                y_weak = (pd.to_numeric(teacher.loc[weak_ids, target], errors="coerce").to_numpy(float) >= 0.5).astype(np.int8)
                if np.isnan(y_gold).any() or np.isnan(y_weak).any():
                    raise ValueError(f"Labels ausentes para {target}")
                rows = np.arange(len(gold_ids))
                for train_rows, valid_rows in splitter.split(rows, y_gold, gold_hashes):
                    blocked = set(gold_hashes[valid_rows])
                    keep_weak = ~np.isin(weak_hashes, list(blocked))
                    results["blocked_weak_rows_total"] += int((~keep_weak).sum())
                    x_train = np.vstack((weak_matrix[keep_weak], gold_matrix[train_rows]))
                    y_train = np.concatenate((y_weak[keep_weak], y_gold[train_rows].astype(np.int8)))
                    sample_weights = np.concatenate(
                        (np.ones(int(keep_weak.sum()), dtype=np.float64), np.full(len(train_rows), gold_weight))
                    )
                    model = _fit_model(c, x_train, y_train, sample_weights)
                    predictions[valid_rows, target_index] = model.predict_proba(gold_matrix[valid_rows])[:, 1]
            target_auc = {
                target: float(roc_auc_score(train.loc[gold_ids, target].to_numpy(float), predictions[:, index]))
                for index, target in enumerate(TARGETS)
            }
            macro = float(np.mean(list(target_auc.values())))
            macro_values.append(macro)
            seed_results[str(seed)] = {"macro_auc": macro, "target_auc": target_auc}
        results["weights"][str(gold_weight)] = {
            "seeds": seed_results,
            "macro_auc_mean": float(np.mean(macro_values)),
            "macro_auc_std": float(np.std(macro_values, ddof=0)),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-index", type=Path, required=True)
    parser.add_argument("--gold-index", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--teacher", type=Path, default=Path("data/external_labels/targetwise_teacher.csv"))
    parser.add_argument("--weights", default="1,4,8")
    parser.add_argument("--seeds", default="42,2026")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("reports/gold_weighted_cv.json"))
    args = parser.parse_args()
    weights = [float(value.strip()) for value in args.weights.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not weights or not seeds or any(value <= 0 for value in weights) or args.folds < 2 or args.c <= 0:
        raise ValueError("Pesos/seeds/folds/C inválidos.")
    result = evaluate(
        _resolve(args.weak_index),
        _resolve(args.gold_index),
        _resolve(args.train),
        _resolve(args.teacher),
        weights,
        seeds,
        args.folds,
        args.c,
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for weight, values in result["weights"].items():
        print(f"gold_weight={weight} macro_mean={values['macro_auc_mean']:.6f} std={values['macro_auc_std']:.6f}")
    print(f"hash_overlap_weak_gold={result['hash_overlap_weak_gold']} report={output}")


if __name__ == "__main__":
    main()
