"""Sinais lexicais multilíngues auditáveis para relatórios de joelho."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .constants import TARGET_COLUMNS


LEXICON: dict[str, tuple[str, ...]] = {
    "ACL": ("acl", "lca", "anterior cruciate", "ligamento cruzado anterior", "ligament croise anterieur"),
    "MCL": ("mcl", "lcm", "medial collateral", "ligamento colateral medial", "ligament collateral medial"),
    "Medial Meniscus": ("medial meniscus", "meniscus medialis", "menisco medial", "menisque medial", "menisco interno"),
    "Lateral Meniscus": ("lateral meniscus", "meniscus lateralis", "menisco lateral", "menisque lateral", "menisco externo"),
    "Medial OA": ("medial osteoarthritis", "medial arthrosis", "medial osteoarthrosis", "medial compartment", "compartimento medial", "compartiment medial"),
    "Lateral OA": ("lateral osteoarthritis", "lateral arthrosis", "lateral osteoarthrosis", "lateral compartment", "compartimento lateral", "compartiment lateral"),
    "PF OA": ("patellofemoral osteoarthritis", "patellofemoral arthrosis", "patellofemoral compartment", "patellofemoral", "femoropatellar", "femoro-patellar"),
    "Effusion": ("joint effusion", "effusion", "derrame articular", "derrame", "epanchement"),
    "Synovitis": ("synovitis", "sinovitis", "synovite"),
    "Baker's": ("baker", "popliteal cyst", "cisto popliteo", "kyste poplite"),
    "Contusion": ("bone contusion", "bone bruise", "contusion", "contusao ossea", "contusion ossea", "contusion oseuse", "bone marrow edema"),
    "Fracture": ("fracture", "fractura", "fratura"),
}

NEGATION_CUES = ("no", "not", "without", "intact", "normal", "preserved", "absent", "sin", "sem", "aucun", "aucune", "sans")


def normalize(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.lower().split())


def compile_patterns(terms: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(rf"(?<!\w){re.escape(normalize(term))}(?!\w)") for term in terms)


def score_report(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    """Retorna 1 para menção não negada, -1 para menção negada e 0 para ausência."""

    normalized = normalize(text)
    positive = False
    negative = False
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            context = normalized[max(0, match.start() - 90) : match.start()]
            if any(re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", context) for cue in NEGATION_CUES):
                negative = True
            else:
                positive = True
    if positive:
        return 1
    if negative:
        return -1
    return 0


def build_lexicon_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Retorna uma coluna {-1, 0, 1} por alvo, sem consultar rótulos."""

    reports = frame.get("Report", pd.Series("", index=frame.index)).fillna("").astype(str)
    result = pd.DataFrame(index=frame.index)
    for target in TARGET_COLUMNS:
        patterns = compile_patterns(LEXICON[target])
        result[f"lexicon_{target}"] = [score_report(report, patterns) for report in reports]
    return result.astype(float)
