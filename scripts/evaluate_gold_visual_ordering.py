#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — ablação gold da ordenação de cortes 2.5D.

Compara duas representações extraídas exatamente dos mesmos estudos e séries:

* ``filename_order``: ordem lexicográfica dos arquivos DICOM;
* ``header_instance_number``: ordem por ``InstanceNumber`` do cabeçalho DICOM.

A comparação usa apenas os 58 estudos locais com os 12 rótulos oficiais da
competição, agrupa as séries por estudo e calcula AUC out-of-fold com
``StratifiedKFold`` separado por alvo. O objetivo é decidir se a ordenação
anatômica merece ser escalada para um conjunto maior; não é uma estimativa
confiável do leaderboard com este gold pequeno.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
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


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_source_records(embedding_index: dict[str, object], index_path: Path) -> list[dict[str, object]]:
    source_index_value = embedding_index.get("source_index")
    if not source_index_value:
        return []
    source_path = _resolve(Path(str(source_index_value)))
    if not source_path.exists():
        source_path = index_path.parent / str(source_index_value)
    if not source_path.exists():
        return []
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return [dict(record) for record in source.get("records", [])]


def _load_variant(index_path: Path) -> dict[str, object]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = [dict(record) for record in index.get("records", [])]
    matrix_path = index_path.parent / str(index.get("embedding_path", "embeddings.npy"))
    matrix = np.load(matrix_path).astype(np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records) or not np.isfinite(matrix).all():
        raise ValueError(
            f"Índice/matriz inválidos em {index_path}: shape={matrix.shape}, records={len(records)}"
        )

    groups: OrderedDict[str, list[int]] = OrderedDict()
    for row, record in enumerate(records):
        groups.setdefault(str(record["study_uid"]), []).append(row)
    study_ids = sorted(groups)
    study_matrix = np.stack([matrix[groups[study_id]].mean(axis=0) for study_id in study_ids])

    source_records = _load_source_records(index, index_path)
    planes: Counter[str] = Counter()
    plane_by_series: dict[tuple[str, str], str] = {}
    for record in source_records:
        study_uid = str(record.get("study_uid", ""))
        series_uid = str(record.get("series_uid", ""))
        plane = str(record.get("anatomical_plane", "Unknown"))
        plane_by_series[(study_uid, series_uid)] = plane
    for record in records:
        key = (str(record["study_uid"]), str(record["series_uid"]))
        planes[plane_by_series.get(key, "Unknown")] += 1

    return {
        "index_path": str(index_path),
        "matrix": study_matrix,
        "study_ids": study_ids,
        "series_count": len(records),
        "plane_counts": dict(sorted(planes.items())),
        "series_per_study": {study_id: len(groups[study_id]) for study_id in study_ids},
    }


def _load_gold_labels(train_path: Path, study_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(train_path).set_index("StudyInstanceUID")
    frame.index = frame.index.astype(str)
    missing = sorted(set(study_ids) - set(frame.index))
    if missing:
        raise ValueError(f"{len(missing)} estudos sem linha em {train_path}")
    gold = frame.loc[study_ids, TARGETS].apply(pd.to_numeric, errors="coerce")
    incomplete = gold.isna().any(axis=1)
    if incomplete.any():
        examples = list(gold.index[incomplete][:3])
        raise ValueError(f"Gold contém rótulos incompletos; exemplos: {examples}")
    return gold


def _evaluate_variant(
    matrix: np.ndarray,
    labels: pd.DataFrame,
    seeds: list[int],
    folds: int,
    c: float,
) -> dict[str, object]:
    by_seed: dict[str, object] = {}
    target_aucs: dict[str, list[float]] = {target: [] for target in TARGETS}
    macro_aucs: list[float] = []

    for seed in seeds:
        seed_targets: dict[str, object] = {}
        seed_aucs: list[float] = []
        for target in TARGETS:
            y = labels[target].to_numpy(dtype=np.int8)
            positive = int(y.sum())
            negative = int(len(y) - positive)
            n_splits = min(folds, positive, negative)
            if n_splits < 2:
                raise ValueError(f"Alvo {target!r} não permite validação estratificada: {positive}/{negative}")

            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            oof = np.full(len(y), np.nan, dtype=np.float64)
            for train_rows, valid_rows in splitter.split(matrix, y):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear"),
                )
                model.fit(matrix[train_rows], y[train_rows])
                oof[valid_rows] = model.predict_proba(matrix[valid_rows])[:, 1]
            if not np.isfinite(oof).all():
                raise ValueError(f"OOF incompleto para {target!r}, seed={seed}")
            auc = float(roc_auc_score(y, oof))
            seed_targets[target] = {
                "auc": auc,
                "positive": positive,
                "negative": negative,
                "folds": n_splits,
            }
            target_aucs[target].append(auc)
            seed_aucs.append(auc)
        macro_auc = float(np.mean(seed_aucs))
        macro_aucs.append(macro_auc)
        by_seed[str(seed)] = {"macro_auc": macro_auc, "targets": seed_targets}

    return {
        "seeds": by_seed,
        "macro_auc_mean": float(np.mean(macro_aucs)),
        "macro_auc_std": float(np.std(macro_aucs, ddof=0)),
        "target_summary": {
            target: {
                "mean_auc": float(np.mean(aucs)),
                "std_auc": float(np.std(aucs, ddof=0)),
            }
            for target, aucs in target_aucs.items()
        },
    }


def evaluate(
    filename_index: Path,
    header_index: Path,
    train_path: Path,
    seeds: list[int],
    folds: int,
    c: float,
) -> dict[str, object]:
    filename = _load_variant(filename_index)
    header = _load_variant(header_index)
    filename_ids = filename["study_ids"]
    header_ids = header["study_ids"]
    if filename_ids != header_ids:
        raise ValueError("As duas variantes não têm exatamente os mesmos estudos na mesma cobertura")
    labels = _load_gold_labels(train_path, filename_ids)

    filename_result = _evaluate_variant(filename["matrix"], labels, seeds, folds, c)
    header_result = _evaluate_variant(header["matrix"], labels, seeds, folds, c)
    seed_comparison = {}
    for seed in seeds:
        filename_macro = float(filename_result["seeds"][str(seed)]["macro_auc"])
        header_macro = float(header_result["seeds"][str(seed)]["macro_auc"])
        seed_comparison[str(seed)] = {
            "filename_macro_auc": filename_macro,
            "header_macro_auc": header_macro,
            "delta_header_minus_filename": header_macro - filename_macro,
        }
    target_comparison = {
        target: {
            "filename_mean_auc": filename_result["target_summary"][target]["mean_auc"],
            "header_mean_auc": header_result["target_summary"][target]["mean_auc"],
            "delta_header_minus_filename": (
                header_result["target_summary"][target]["mean_auc"]
                - filename_result["target_summary"][target]["mean_auc"]
            ),
        }
        for target in TARGETS
    }

    return {
        "tags": ["RSNA", "Kaggle", "Dados"],
        "format": "gold_visual_ordering_ablation-v1",
        "gold_studies": len(filename_ids),
        "gold_series": {
            "filename_order": filename["series_count"],
            "header_instance_number": header["series_count"],
        },
        "plane_counts": {
            "filename_order": filename["plane_counts"],
            "header_instance_number": header["plane_counts"],
        },
        "series_per_study": filename["series_per_study"],
        "feature_shape": {
            "filename_order": list(filename["matrix"].shape),
            "header_instance_number": list(header["matrix"].shape),
        },
        "gold_label_source": str(train_path),
        "validation": {
            "method": "study-grouped stratified out-of-fold AUC, separado por alvo",
            "seeds": seeds,
            "requested_folds": folds,
            "classifier": "StandardScaler + LogisticRegression(class_weight=balanced, solver=liblinear)",
            "c": c,
        },
        "limitations": [
            "São apenas 58 estudos oficiais locais; a variância entre seeds é relevante.",
            "A cobertura tem 56 séries Sagittal e 3 Coronal, sem Axial; pooling por três planos não foi testado.",
            "O resultado é uma ablação de representação, não uma estimativa direta do leaderboard.",
        ],
        "variants": {
            "filename_order": filename_result,
            "header_instance_number": header_result,
        },
        "comparison": {
            "seed_macro": seed_comparison,
            "target_mean": target_comparison,
            "mean_delta_header_minus_filename": (
                header_result["macro_auc_mean"] - filename_result["macro_auc_mean"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename-index", type=Path, required=True)
    parser.add_argument("--header-index", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--seeds", default="42,2026")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--left-label", default="filename_order")
    parser.add_argument("--right-label", default="header_instance_number")
    parser.add_argument("--output", type=Path, default=Path("reports/gold_visual_ordering_ablation.json"))
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or args.folds < 2 or args.c <= 0:
        raise ValueError("Informe ao menos uma seed, folds >= 2 e C > 0.")

    result = evaluate(
        _resolve(args.filename_index),
        _resolve(args.header_index),
        _resolve(args.train),
        seeds,
        args.folds,
        args.c,
    )
    result["variant_labels"] = {"left": args.left_label, "right": args.right_label}
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"gold_studies={result['gold_studies']} gold_series={result['gold_series']} "
        f"planes={result['plane_counts']}"
    )
    for seed, values in result["comparison"]["seed_macro"].items():
        print(
            f"seed={seed} {args.left_label}_macro={values['filename_macro_auc']:.6f} "
            f"{args.right_label}_macro={values['header_macro_auc']:.6f} "
            f"delta={values['delta_header_minus_filename']:+.6f}"
        )
    print(f"mean_delta_header_minus_filename={result['comparison']['mean_delta_header_minus_filename']:+.6f}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
