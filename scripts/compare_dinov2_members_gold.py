#!/usr/bin/env python3
"""Compara o blend público DINOv2 members com a âncora H-38 no gold.

O script só combina artefatos já produzidos por
``evaluate_dinov2_members_gold.py``. Como os checkpoints foram treinados com
o treino da competição, todas as métricas aqui são diagnósticas e não OOF.
Nenhum peso é promovido automaticamente para submissão.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


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
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)


def ranks(matrix: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.asarray(matrix, dtype=float)).rank(method="average", pct=True).to_numpy(dtype=float)


def auc_by_target(labels: np.ndarray, matrix: np.ndarray) -> dict[str, float]:
    return {
        target: float(roc_auc_score(labels[:, index], matrix[:, index]))
        for index, target in enumerate(TARGETS)
    }


def macro_auc(labels: np.ndarray, matrix: np.ndarray) -> float:
    return float(np.mean(list(auc_by_target(labels, matrix).values())))


def bootstrap_delta(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    seed: int = 20260903,
    count: int = 2000,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(count, dtype=float)
    for index in range(count):
        sample = rng.integers(0, len(labels), size=len(labels))
        per_target = []
        for target_index in range(len(TARGETS)):
            y = labels[sample, target_index]
            if np.unique(y).size < 2:
                continue
            per_target.append(
                roc_auc_score(y, candidate[sample, target_index])
                - roc_auc_score(y, reference[sample, target_index])
            )
        deltas[index] = np.mean(per_target) if per_target else np.nan
    valid = deltas[np.isfinite(deltas)]
    return {
        "mean": float(np.mean(valid)),
        "ci95_low": float(np.quantile(valid, 0.025)),
        "ci95_high": float(np.quantile(valid, 0.975)),
        "n": int(len(valid)),
    }


def load_ensemble(path: Path, family: str, expected: int) -> tuple[np.ndarray, dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    values = np.asarray(report["families"][family]["ensemble_predictions"], dtype=float)
    if values.shape != (expected, len(TARGETS)) or not np.isfinite(values).all():
        raise ValueError(f"predições inválidas em {path}: {values.shape}")
    return values, {
        "report": str(path),
        "family": family,
        "coverage": report.get("slot_coverage_decoded"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--champ-report", type=Path, required=True)
    parser.add_argument("--llm-report", type=Path, required=True)
    parser.add_argument("--h36-report", type=Path, required=True)
    parser.add_argument("--dino-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-count", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_count < 1:
        raise ValueError("bootstrap-count precisa ser positivo")

    train = pd.read_csv(args.data_dir / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[train[TARGETS].notna().all(axis=1)].copy()
    labels = gold[TARGETS].to_numpy(dtype=float)
    ids = gold["StudyInstanceUID"].astype(str).tolist()
    champ_raw, champ_meta = load_ensemble(args.champ_report, "champ", len(ids))
    llm_raw, llm_meta = load_ensemble(args.llm_report, "llm199e30", len(ids))

    h36 = json.loads(args.h36_report.read_text(encoding="utf-8"))
    h36_values = h36.get("predictions", {}).get("h36", {})
    if set(h36_values) != set(ids):
        raise ValueError("IDs do H-36 não coincidem com os 58 gold")
    h36_rank = ranks(np.asarray([h36_values[uid] for uid in ids], dtype=float))
    dino = np.load(args.dino_npz, allow_pickle=False)
    if dino["ids"].astype(str).tolist() != ids:
        raise ValueError("IDs do DINOv3 não coincidem com os 58 gold")
    h38_rank = ranks(0.80 * h36_rank + 0.20 * np.asarray(dino["base_rank"], dtype=float))
    champ_rank = ranks(champ_raw)
    llm_rank = ranks(llm_raw)

    individual = {
        "h38": {"macro_auc": macro_auc(labels, h38_rank), "auc_by_target": auc_by_target(labels, h38_rank)},
        "champ": {"macro_auc": macro_auc(labels, champ_rank), "auc_by_target": auc_by_target(labels, champ_rank)},
        "llm199e30": {"macro_auc": macro_auc(labels, llm_rank), "auc_by_target": auc_by_target(labels, llm_rank)},
    }
    blend_results: dict[str, object] = {}
    for champ_weight in WEIGHTS:
        llm_weight = 1.0 - champ_weight
        blend = ranks(champ_weight * champ_rank + llm_weight * llm_rank)
        blend_results[f"champ_{champ_weight:.2f}_llm199e30_{llm_weight:.2f}"] = {
            "champ_weight": champ_weight,
            "llm199e30_weight": llm_weight,
            "macro_auc": macro_auc(labels, blend),
            "auc_by_target": auc_by_target(labels, blend),
        }

    comparisons: dict[str, object] = {}
    for name, weight in (("champ", 1.0), ("llm199e30", 1.0)):
        candidate = champ_rank if name == "champ" else llm_rank
        comparisons[name] = {
            f"h38_{1.0 - weight:.2f}_{name}_{weight:.2f}": {
                "macro_auc": macro_auc(labels, candidate),
                "delta_vs_h38_bootstrap": bootstrap_delta(
                    labels,
                    candidate,
                    h38_rank,
                    seed=20260903 + (1 if name == "champ" else 2),
                    count=args.bootstrap_count,
                ),
            }
        }
    exp056 = ranks(0.20 * champ_rank + 0.80 * llm_rank)
    comparisons["exp056_20_80_vs_h38"] = {
        "macro_auc": macro_auc(labels, exp056),
        "auc_by_target": auc_by_target(labels, exp056),
        "delta_vs_h38_bootstrap": bootstrap_delta(
            labels, exp056, h38_rank, seed=20260903 + 20, count=args.bootstrap_count
        ),
    }

    correlations = {
        target: {
            "champ_llm199e30": float(np.corrcoef(champ_rank[:, index], llm_rank[:, index])[0, 1]),
            "champ_h38": float(np.corrcoef(champ_rank[:, index], h38_rank[:, index])[0, 1]),
            "llm199e30_h38": float(np.corrcoef(llm_rank[:, index], h38_rank[:, index])[0, 1]),
        }
        for index, target in enumerate(TARGETS)
    }
    result = {
        "format": "rsna-dinov2-members-gold-blend-comparison-v1",
        "diagnostic_only": True,
        "gold_studies": len(ids),
        "models": {"champ": champ_meta, "llm199e30": llm_meta},
        "individual": individual,
        "fixed_blends": blend_results,
        "h38_comparisons": comparisons,
        "correlations_by_target": correlations,
        "decision_rule": "não promover pelo gold; exigir OOF ou score Kaggle novo acima de H-38",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"individual": individual, "exp056_20_80_vs_h38": comparisons["exp056_20_80_vs_h38"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
