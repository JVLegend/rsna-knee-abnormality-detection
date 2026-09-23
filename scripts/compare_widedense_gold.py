"""#RSNA #Kaggle #Pesquisa — compara WideDense e H-36 no gold.

Os pesos dos blends são uma grade diagnóstica pré-especificada. O script não
escolhe um peso para deployment: o gold tem 58 estudos e serve apenas para
medir complementaridade e refutar hipóteses frágeis.
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
BLEND_FULL_WEIGHTS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.0)
BOOTSTRAPS = 2000


def rank_columns(matrix: np.ndarray) -> np.ndarray:
    return pd.DataFrame(matrix).rank(method="average", pct=True).to_numpy(dtype=float)


def auc_by_target(matrix: np.ndarray, gold: pd.DataFrame) -> dict[str, float]:
    result = {}
    for index, target in enumerate(TARGETS):
        result[target] = float(roc_auc_score(gold[target].to_numpy(dtype=float), matrix[:, index]))
    return result


def macro_auc(matrix: np.ndarray, gold: pd.DataFrame) -> float:
    return float(np.mean(list(auc_by_target(matrix, gold).values())))


def bootstrap_delta(candidate: np.ndarray, reference: np.ndarray, gold: pd.DataFrame) -> dict[str, float]:
    rng = np.random.default_rng(20260902)
    deltas = np.empty(BOOTSTRAPS, dtype=float)
    y = gold[TARGETS].to_numpy(dtype=float)
    n = len(gold)
    for index in range(BOOTSTRAPS):
        sample = rng.integers(0, n, size=n)
        scores = []
        for target_index in range(len(TARGETS)):
            target = y[sample, target_index]
            if np.unique(target).size < 2:
                continue
            scores.append(
                roc_auc_score(target, candidate[sample, target_index])
                - roc_auc_score(target, reference[sample, target_index])
            )
        deltas[index] = np.mean(scores) if scores else np.nan
    valid = deltas[np.isfinite(deltas)]
    return {
        "mean": float(np.mean(valid)),
        "ci95_low": float(np.quantile(valid, 0.025)),
        "ci95_high": float(np.quantile(valid, 0.975)),
        "n": int(len(valid)),
    }


def load_report(path: Path) -> tuple[list[str], np.ndarray, dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    predictions = report["predictions"]
    if len(predictions) != 1:
        raise ValueError(f"relatório deve conter um modelo: {path}")
    name, values = next(iter(predictions.items()))
    ids = list(values)
    matrix = np.asarray([values[study] for study in ids], dtype=float)
    if matrix.shape != (len(ids), len(TARGETS)) or not np.isfinite(matrix).all():
        raise ValueError(f"predições inválidas em {path}")
    return ids, matrix, {"name": name, "metadata": report.get("models", {}).get(name, {})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--h36-report", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser()
    train = pd.read_csv(data_dir / "train.csv")
    gold = train[train[TARGETS].notna().all(axis=1)].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    ids_h36, h36, h36_meta = load_report(args.h36_report.expanduser())
    ids_full, full, full_meta = load_report(args.full_report.expanduser())
    gold_ids = gold["StudyInstanceUID"].tolist()
    if ids_h36 != ids_full or ids_h36 != gold_ids:
        raise ValueError("ordem/IDs dos relatórios não coincide com o gold oficial")

    h36_rank = rank_columns(h36)
    full_rank = rank_columns(full)
    individual = {
        "h36_raw": {"macro_auc": macro_auc(h36, gold), "auc_by_target": auc_by_target(h36, gold)},
        "full_raw": {"macro_auc": macro_auc(full, gold), "auc_by_target": auc_by_target(full, gold)},
    }
    reference = h36_rank
    blends = {}
    for full_weight in BLEND_FULL_WEIGHTS:
        candidate = (1.0 - full_weight) * h36_rank + full_weight * full_rank
        name = f"rank_h36_{1.0 - full_weight:.2f}_full_{full_weight:.2f}"
        blends[name] = {
            "h36_weight": 1.0 - full_weight,
            "full_weight": full_weight,
            "macro_auc": macro_auc(candidate, gold),
            "auc_by_target": auc_by_target(candidate, gold),
            "delta_vs_h36_rank_bootstrap": bootstrap_delta(candidate, reference, gold),
        }

    correlations = {}
    for index, target in enumerate(TARGETS):
        correlations[target] = {
            "pearson_raw": float(np.corrcoef(h36[:, index], full[:, index])[0, 1]),
            "pearson_rank": float(np.corrcoef(h36_rank[:, index], full_rank[:, index])[0, 1]),
        }

    result = {
        "format": "rsna-widedense-gold-comparison-v1",
        "gold_studies": int(len(gold)),
        "bootstrap": {"seed": 20260902, "n": BOOTSTRAPS},
        "models": {"h36": h36_meta, "full": full_meta},
        "individual": individual,
        "blends_diagnostic_only": blends,
        "correlations_by_target": correlations,
        "decision_rule": "não escolher peso pelo gold; só promover blend após validação fold-conditioned independente e teste Kaggle",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"individual": individual, "blends": blends}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
