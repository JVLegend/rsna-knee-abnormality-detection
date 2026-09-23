#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — V01: partição agrupada imutável, sem GPU.

Inventaria caches existentes, não certifica integridade dos volumes DICOM.
O JSON congela labels suaves; 0.5 permanece incerto, nunca positivo implícito.
Não importa nem executa código ou checkpoints de terceiros.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

KEY = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
PLANES = {"Sagittal", "Coronal", "Axial"}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def group_hash(report: str) -> str:
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", report).casefold()).strip()
    if not normalized:
        raise ValueError("Laudo vazio não permite agrupamento confiável")
    return hashlib.sha256(normalized.encode()).hexdigest()


def rows_by_id(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        # CSV Kaggle: vírgula confirmada no cabeçalho, não locale brasileiro.
        rows = list(csv.DictReader(handle))
    ids = [r[KEY] for r in rows]
    if any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"IDs vazios/duplicados: {path}")
    return dict(zip(ids, rows))


def assign_groups(studies: list[dict], dev: int, confirm: int, seed: int) -> dict:
    if dev < 1 or confirm < 1 or len(studies) <= dev + confirm:
        raise ValueError("Cobertura insuficiente para treino/dev/confirmação")
    groups = defaultdict(list)
    for study in studies:
        groups[study["report_hash"]].append(study)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    result = {"development": [], "confirmation": [], "train": []}
    # Sem busca de seed ou seleção por desempenho/prevalência.
    for key in keys:
        split = ("development" if len(result["development"]) < dev else
                 "confirmation" if len(result["confirmation"]) < confirm else "train")
        result[split].extend(groups[key])
    if not result["train"] or len(result["confirmation"]) < confirm:
        raise ValueError("Grupos grandes inviabilizam as cotas; não dividir o grupo")
    return {name: sorted(rows, key=lambda r: r[KEY]) for name, rows in result.items()}


def class_counts(rows: list[dict]) -> dict:
    return {target: {
        "positive_gt_0.5": sum(r["labels"][target] > .5 for r in rows),
        "negative_lt_0.5": sum(r["labels"][target] < .5 for r in rows),
        "uncertain_eq_0.5": sum(r["labels"][target] == .5 for r in rows),
    } for target in TARGETS}


def build(train_path: Path, teacher_path: Path, cache_index: Path,
          dev: int = 250, confirm: int = 150, seed: int = 2026) -> dict:
    train, teacher = rows_by_id(train_path), rows_by_id(teacher_path)
    cache = json.loads(cache_index.read_text())
    gold_ids = {uid for uid, row in train.items() if any(row[t].strip() for t in TARGETS)}
    hashes = {}
    for uid, row in train.items():
        try:
            hashes[uid] = group_hash(row["Report"])
        except ValueError:
            # Conservador: nunca incluir laudo vazio em weak.
            pass
    gold_groups = {hashes[uid] for uid in gold_ids if uid in hashes}
    records = defaultdict(list)
    seen_series = set()
    for record in cache["records"]:
        key = (record["study_uid"], record["series_uid"])
        if key in seen_series:
            raise ValueError(f"Série duplicada no cache: {key}")
        seen_series.add(key)
        records[key[0]].append(record)
    eligible, excluded = [], {}
    existing_arrays = 0
    root = cache_index.parent.resolve()
    for uid, series in sorted(records.items()):
        reason = None
        if uid not in train or uid not in teacher:
            reason = "missing_metadata_or_teacher"
        elif uid not in hashes:
            reason = "blank_report"
        elif uid in gold_ids or hashes[uid] in gold_groups:
            reason = "gold_or_shared_gold_report"
        coverage = set()
        arrays = []
        for record in series:
            path = (root / record["array_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Array fora da raiz do cache")
            present = path.is_file() and path.stat().st_size > 0
            existing_arrays += int(present)
            if present:
                coverage.add(record["anatomical_plane"])
                arrays.append({"series_uid": record["series_uid"],
                               "plane": record["anatomical_plane"],
                               "array_path": record["array_path"],
                               "bytes": path.stat().st_size,
                               "slices_recorded": record["n_slices"],
                               "fat_suppression": record.get("fat_suppression")})
        if not PLANES.issubset(coverage):
            reason = reason or "missing_plane_cache"
        labels = {}
        if reason is None:
            try:
                labels = {t: float(teacher[uid][t]) for t in TARGETS}
                if any(not math.isfinite(v) or not 0 <= v <= 1 for v in labels.values()):
                    raise ValueError("Label fora do intervalo")
            except (ValueError, KeyError, TypeError):
                reason = "invalid_teacher_labels"
        if reason:
            excluded[uid] = reason
        else:
            eligible.append({KEY: uid, "report_hash": hashes[uid],
                             "labels": labels, "series": arrays})
    splits = assign_groups(eligible, dev, confirm, seed)
    counts = {name: class_counts(rows) for name, rows in splits.items()}
    failures = [f"{name}/{target}" for name, targets in counts.items()
                for target, c in targets.items()
                if not c["positive_gt_0.5"] or not c["negative_lt_0.5"]]
    group_sets = [{r["report_hash"] for r in rows} for rows in splits.values()]
    if any(a & b for i, a in enumerate(group_sets) for b in group_sets[i + 1:]):
        raise AssertionError("Vazamento entre partições")
    return {
        "format": "rsna-weak-grouped-validation-v1", "seed": seed,
        "requested_sizes": {"development": dev, "confirmation": confirm},
        "sources": {name: {"path": str(path), "sha256": digest(path)}
                    for name, path in [("train", train_path), ("teacher", teacher_path),
                                       ("cache_index", cache_index)]},
        "coverage": {"cached_studies": len(records), "cached_series": len(seen_series),
                     "nonempty_arrays": existing_arrays, "eligible_studies": len(eligible),
                     "gold_studies_excluded_globally": len(gold_ids),
                     "eligible_groups": len({r["report_hash"] for r in eligible}),
                     "excluded_reasons": dict(Counter(excluded.values()))},
        "limitations": ["Existência/tamanho do cache verificados; pixels/DICOM não revalidados.",
                        "Três planos não equivalem a seis protocolos completos.",
                        "Laudo igual agrupa, mas não identifica paciente; duplicatas visuais não auditadas.",
                        "Rótulos weak medem concordância com professor, não verdade clínica.",
                        "0.5 é incerto: excluir da AUC binária; preservar label suave para treino.",
                        "Confirmação nova neste protocolo, não dados historicamente virgens.",
                        "Só avaliar pesos sem exposição aos grupos reservados (pretreino genérico)."],
        "class_counts": counts, "class_gate_passed": not failures,
        "class_gate_failures": failures, "excluded": excluded, "splits": splits,
        "confirmation_model_evaluations": 0,
    }


def freeze(path: Path, payload: dict) -> bool:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise ValueError("Manifesto congelado difere: usar nova versão, nunca sobrescrever")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--teacher", type=Path, default=Path("data/external_labels/targetwise_teacher.csv"))
    parser.add_argument("--cache-index", type=Path,
                        default=Path("data/processed/dicom_25d_budget50_header_v1/index.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.train, args.teacher, args.cache_index)
    created = freeze(args.output, payload)
    print(json.dumps({"created": created, "sha256": digest(args.output),
                      "coverage": payload["coverage"],
                      "sizes": {k: len(v) for k, v in payload["splits"].items()},
                      "class_gate_passed": payload["class_gate_passed"],
                      "class_gate_failures": payload["class_gate_failures"]}, indent=2))


if __name__ == "__main__":
    main()
