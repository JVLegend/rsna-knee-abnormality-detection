"""#RSNA #Kaggle #Pesquisa — testa WideDense contra a âncora H-38 no gold.

O H-38 é reconstruído exatamente como no deployment: rank por alvo de 80% do
H-36 e 20% do rank DINOv3 público. A grade WideDense é diagnóstica; nenhum
peso é escolhido automaticamente para submissão.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_widedense_gold import (
    TARGETS,
    auc_by_target,
    bootstrap_delta,
    load_report,
    macro_auc,
    rank_columns,
)


FULL_WEIGHTS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--h36-report", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--dino-npz", type=Path, required=True)
    parser.add_argument(
        "--aux-label",
        default="widedense_full",
        help="nome do braço auxiliar no relatório (permite auditar full e SWA com o mesmo código)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    gold = train[train[TARGETS].notna().all(axis=1)].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    gold_ids = gold["StudyInstanceUID"].tolist()
    ids_h36, h36_raw, h36_meta = load_report(args.h36_report)
    ids_full, full_raw, full_meta = load_report(args.full_report)
    dino = np.load(args.dino_npz, allow_pickle=False)
    ids_dino = dino["ids"].astype(str).tolist()
    dino_rank = np.asarray(dino["base_rank"], dtype=float)
    if ids_h36 != gold_ids or ids_full != gold_ids or ids_dino != gold_ids:
        raise ValueError("IDs/ordem não coincidem entre gold, H-36, WideDense e DINOv3")
    if dino_rank.shape != (len(gold_ids), len(TARGETS)) or not np.isfinite(dino_rank).all():
        raise ValueError(f"rank DINOv3 inválido: {dino_rank.shape}")

    h36_rank = rank_columns(h36_raw)
    full_rank = rank_columns(full_raw)
    h38_rank = rank_columns(0.80 * h36_rank + 0.20 * dino_rank)
    aux_label = str(args.aux_label)
    individual = {
        "h36_rank": {"macro_auc": macro_auc(h36_rank, gold), "auc_by_target": auc_by_target(h36_rank, gold)},
        "h38_reconstructed_rank": {"macro_auc": macro_auc(h38_rank, gold), "auc_by_target": auc_by_target(h38_rank, gold)},
        f"{aux_label}_rank": {"macro_auc": macro_auc(full_rank, gold), "auc_by_target": auc_by_target(full_rank, gold)},
    }
    blends = {}
    for full_weight in FULL_WEIGHTS:
        candidate = rank_columns((1.0 - full_weight) * h38_rank + full_weight * full_rank)
        name = f"rank_h38_{1.0 - full_weight:.2f}_{aux_label}_{full_weight:.2f}"
        blends[name] = {
            "h38_weight": 1.0 - full_weight,
            "full_weight": full_weight,
            "macro_auc": macro_auc(candidate, gold),
            "auc_by_target": auc_by_target(candidate, gold),
            "delta_vs_h38_bootstrap": bootstrap_delta(candidate, h38_rank, gold),
        }

    correlations = {}
    for index, target in enumerate(TARGETS):
        correlations[target] = {
            "pearson_h38_full_rank": float(np.corrcoef(h38_rank[:, index], full_rank[:, index])[0, 1]),
            "pearson_h36_full_rank": float(np.corrcoef(h36_rank[:, index], full_rank[:, index])[0, 1]),
        }
    result = {
        "format": "rsna-widedense-h38-gold-comparison-v1",
        "gold_studies": len(gold_ids),
        "h38_formula": {"h36_rank_weight": 0.80, "dino_rank_weight": 0.20, "dino_npz": str(args.dino_npz)},
        "models": {"h36": h36_meta, aux_label: full_meta},
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
