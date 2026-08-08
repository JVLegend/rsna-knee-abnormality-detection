#!/usr/bin/env python3
"""Compara pesos da feature lexical em seeds fixadas e validação por estudo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rsna_knee_baseline.data import find_data_dir, load_competition_tables
from scripts.evaluate_baseline import _evaluate


def _numbers(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=32.0)
    parser.add_argument("--weights", default="0.5,1,2,4")
    parser.add_argument("--seeds", default="42,2026")
    parser.add_argument("--output", default="reports/v0_2_lexicon_weight_grid.json")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    tables = load_competition_tables(data_dir)
    train = tables["train"].reset_index(drop=True)
    if train.empty:
        raise RuntimeError("train.csv é necessário para a grade.")

    results: list[dict[str, object]] = []
    for weight in _numbers(args.weights):
        for seed in [int(value) for value in _numbers(args.seeds)]:
            result = _evaluate(train, tables["train_series"], args.folds, seed, args.c, True, weight)
            results.append(result)
            print(f"weight={weight} seed={seed} macro_auc={result['macro_auc']:.6f}", flush=True)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={output}")


if __name__ == "__main__":
    main()
