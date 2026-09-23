#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — audita blends contínuos de teachers no holdout gold.

O objetivo é testar H-24 sem tocar no treino visual: as fontes são combinadas
por estudo e alvo, e o resultado é comparado ao mapa target-wise já usado no
H-22. A auditoria é direcional, pois os 58 estudos oficiais continuam sendo
um conjunto pequeno; não transforma o score local em previsão de leaderboard.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


KEY_COLUMN = "StudyInstanceUID"
TARGET_COLUMNS = [
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

TARGETWISE_SOURCE = {
    "ACL": "pilkwang",
    "MCL": "pilkwang",
    "Medial Meniscus": "v2",
    "Lateral Meniscus": "v4_masked",
    "Medial OA": "v4_masked",
    "Lateral OA": "v4_masked",
    "PF OA": "v4_masked",
    "Effusion": "v4_masked",
    "Synovitis": "v4_masked",
    "Baker's": "v4_masked",
    "Contusion": "v4_masked",
    "Fracture": "pilkwang",
}


def _load_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if KEY_COLUMN not in frame.columns:
        raise ValueError(f"Fonte sem {KEY_COLUMN}: {path}")
    frame = frame.copy()
    frame[KEY_COLUMN] = frame[KEY_COLUMN].astype(str)
    if frame[KEY_COLUMN].duplicated().any():
        raise ValueError(f"Fonte com UIDs duplicados: {path}")
    frame = frame.set_index(KEY_COLUMN)
    result = pd.DataFrame(index=frame.index)
    for target in TARGET_COLUMNS:
        if target not in frame.columns:
            raise ValueError(f"Fonte sem o alvo {target!r}: {path}")
        score = pd.to_numeric(frame[target], errors="coerce").clip(0.0, 1.0)
        verdict_column = f"{target}__verdict"
        if verdict_column in frame.columns:
            verdict = frame[verdict_column].fillna("UNK").astype(str).str.upper()
            score = score.mask(verdict.eq("UNK"), 0.5)
        result[target] = score
    result.attrs["path"] = str(path)
    return result


def _combine(frames: list[pd.DataFrame], method: str) -> pd.DataFrame:
    result: dict[str, pd.Series] = {}
    for target in TARGET_COLUMNS:
        values = pd.concat([frame[target] for frame in frames], axis=1)
        if method == "mean":
            result[target] = values.mean(axis=1, skipna=True)
        elif method == "rankmean":
            ranks = values.rank(method="average", pct=True)
            result[target] = ranks.mean(axis=1, skipna=True)
        else:
            raise ValueError(f"Método de combinação desconhecido: {method}")
    return pd.DataFrame(result, index=frames[0].index)


def _targetwise(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            target: frames[source][target]
            for target, source in TARGETWISE_SOURCE.items()
        },
        index=frames["v4_masked"].index,
    )


def _auc_rows(gold: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate, frame in predictions.items():
        for target in TARGET_COLUMNS:
            y = gold[target]
            score = frame[target].reindex(gold.index)
            valid = y.notna() & score.notna()
            auc = None
            if valid.sum() and y.loc[valid].nunique() > 1:
                auc = float(roc_auc_score(y.loc[valid].astype(float), score.loc[valid].astype(float)))
            rows.append(
                {
                    "candidate": candidate,
                    "target": target,
                    "n_gold": int(valid.sum()),
                    "auc": auc,
                }
            )
    return rows


def _macro(rows: list[dict[str, object]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    return (
        table.groupby("candidate", as_index=False)["auc"]
        .mean()
        .rename(columns={"auc": "macro_auc"})
        .sort_values("macro_auc", ascending=False)
    )


def _bootstrap_delta(
    gold: pd.DataFrame,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    seed: int,
    repeats: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    ids = np.arange(len(gold))
    deltas: list[float] = []
    for _ in range(repeats):
        sample = rng.choice(ids, size=len(ids), replace=True)
        target_deltas: list[float] = []
        for target in TARGET_COLUMNS:
            y = gold[target].to_numpy(dtype=float)[sample]
            ref = reference[target].reindex(gold.index).to_numpy(dtype=float)[sample]
            cand = candidate[target].reindex(gold.index).to_numpy(dtype=float)[sample]
            valid = np.isfinite(y) & np.isfinite(ref) & np.isfinite(cand)
            if valid.sum() < 2 or np.unique(y[valid]).size < 2:
                continue
            target_deltas.append(
                roc_auc_score(y[valid], cand[valid])
                - roc_auc_score(y[valid], ref[valid])
            )
        if target_deltas:
            deltas.append(float(np.mean(target_deltas)))
    if not deltas:
        raise ValueError("Não foi possível calcular o bootstrap pareado.")
    quantiles = np.quantile(deltas, [0.025, 0.5, 0.975])
    return {
        "seed": seed,
        "repeats": repeats,
        "valid_repeats": len(deltas),
        "delta_mean": float(np.mean(deltas)),
        "delta_p025": float(quantiles[0]),
        "delta_median": float(quantiles[1]),
        "delta_p975": float(quantiles[2]),
    }


def audit(
    data_dir: Path,
    v2_path: Path,
    v4_path: Path,
    pilkwang_path: Path,
    lixin_path: Path,
    *,
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 20260816,
) -> dict[str, object]:
    train = pd.read_csv(data_dir / "train.csv")
    train[KEY_COLUMN] = train[KEY_COLUMN].astype(str)
    gold = train.loc[train[TARGET_COLUMNS].notna().any(axis=1), [KEY_COLUMN, *TARGET_COLUMNS]].copy()
    gold = gold.set_index(KEY_COLUMN)

    v2 = _load_source(v2_path)
    v4 = _load_source(v4_path)
    pilkwang = _load_source(pilkwang_path)
    lixin = _load_source(lixin_path)
    v4_masked = v4[TARGET_COLUMNS].where(v2[TARGET_COLUMNS].sub(0.5).abs().gt(1e-9), 0.5)
    v4_masked.attrs.update(v4.attrs)
    frames = {
        "v2": v2[TARGET_COLUMNS],
        "v4": v4[TARGET_COLUMNS],
        "v4_masked": v4_masked,
        "pilkwang": pilkwang[TARGET_COLUMNS],
        "lixin": lixin[TARGET_COLUMNS],
    }

    predictions = {
        "v4_masked": frames["v4_masked"],
        "targetwise_current": _targetwise(frames),
        "mean_v4_masked_v2_pilkwang": _combine(
            [frames["v4_masked"], frames["v2"], frames["pilkwang"]], "mean"
        ),
        "rankmean_v4_masked_v2_pilkwang": _combine(
            [frames["v4_masked"], frames["v2"], frames["pilkwang"]], "rankmean"
        ),
        "mean_v4_masked_pilkwang_lixin": _combine(
            [frames["v4_masked"], frames["pilkwang"], frames["lixin"]], "mean"
        ),
        "mean_v4_masked_v2_pilkwang_lixin": _combine(
            [frames["v4_masked"], frames["v2"], frames["pilkwang"], frames["lixin"]], "mean"
        ),
        "rankmean_v4_masked_v2_pilkwang_lixin": _combine(
            [frames["v4_masked"], frames["v2"], frames["pilkwang"], frames["lixin"]], "rankmean"
        ),
    }
    rows = _auc_rows(gold, predictions)
    macro = _macro(rows)
    reference = predictions["targetwise_current"]
    simple_candidates = [name for name in predictions if name not in {"v4_masked", "targetwise_current"}]
    bootstrap = {
        name: _bootstrap_delta(
            gold,
            reference,
            predictions[name],
            seed=bootstrap_seed,
            repeats=bootstrap_repeats,
        )
        for name in simple_candidates
    }
    macro_lookup = macro.set_index("candidate")["macro_auc"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "format": "rsna-teacher-blend-audit-v1",
        "gold_studies": int(len(gold)),
        "gold_rows_by_target": {target: int(gold[target].notna().sum()) for target in TARGET_COLUMNS},
        "source_paths": {
            "v2": str(v2_path),
            "v4": str(v4_path),
            "pilkwang": str(pilkwang_path),
            "lixin": str(lixin_path),
        },
        "targetwise_source": TARGETWISE_SOURCE,
        "macro_auc": [
            {
                "candidate": str(row["candidate"]),
                "macro_auc": float(row["macro_auc"]),
                "delta_vs_targetwise": float(row["macro_auc"] - macro_lookup["targetwise_current"]),
            }
            for row in macro.to_dict("records")
        ],
        "per_target_auc": rows,
        "bootstrap_delta_vs_targetwise": bootstrap,
        "interpretation": (
            "Blends contínuos são evidência direcional. O mapa target-wise permanece o fallback; "
            "pesos escolhidos diretamente nos 58 estudos seriam suscetíveis a overfit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v4", type=Path, required=True)
    parser.add_argument("--pilkwang", type=Path, required=True)
    parser.add_argument("--lixin", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/teacher_blend_audit.json"))
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        raise ValueError("Use pelo menos 100 repetições para o bootstrap.")
    result = audit(
        args.data_dir,
        args.v2,
        args.v4,
        args.pilkwang,
        args.lixin,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in result["macro_auc"]:
        print(f"{row['candidate']}: macro_auc={row['macro_auc']:.6f} delta={row['delta_vs_targetwise']:+.6f}")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
