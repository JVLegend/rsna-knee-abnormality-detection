#!/usr/bin/env python3
"""Inspeciona os CSVs da competição sem abrir os DICOMs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee_baseline.data import find_data_dir, label_coverage, load_competition_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    tables = load_competition_tables(data_dir)
    print(f"data_dir={data_dir}")
    for name, table in tables.items():
        print(f"{name}: rows={len(table):,} columns={len(table.columns)}")
        if not table.empty:
            print(f"  columns={list(table.columns)}")
    if not tables["train"].empty:
        print("\nCobertura de rótulos:")
        print(label_coverage(tables["train"]).to_string(index=False))


if __name__ == "__main__":
    main()
