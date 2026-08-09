"""Entry point autossuficiente para copiar para um notebook Kaggle.

O arquivo não importa o pacote do repositório: a execução da competição fica
sem internet e precisa continuar funcionando quando este código for colado
como células no notebook. O pacote em ``src/`` continua sendo a fonte de
desenvolvimento local; este arquivo é o snapshot de submissão da família v0.
Para a candidata v0.2, use ``--c 32 --use-lexicon`` ou chame
``run(..., c=32, use_lexicon=True)``.
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

KEY_COLUMN = "StudyInstanceUID"
TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]

LEXICON = {
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


def _reports(frame: pd.DataFrame) -> pd.Series:
    report = frame.get("Report", pd.Series("", index=frame.index)).fillna("").astype(str)
    report = report.map(lambda value: " ".join(value.split()))
    sex = frame.get("PatientSex", pd.Series("Unknown", index=frame.index))
    sex = sex.fillna("Unknown").astype(str).replace({"": "Unknown"})
    return report + " [SEX=" + sex + "]"


def _normalize(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.lower().split())


def _lexicon_score(text: str, terms: tuple[str, ...]) -> int:
    normalized = _normalize(text)
    positive = False
    negative = False
    for term in terms:
        pattern = re.compile(rf"(?<!\w){re.escape(_normalize(term))}(?!\w)")
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


def _lexicon_features(frame: pd.DataFrame) -> pd.DataFrame:
    reports = frame.get("Report", pd.Series("", index=frame.index)).fillna("").astype(str)
    result = {}
    for target in TARGET_COLUMNS:
        result[f"lexicon_{target}"] = [_lexicon_score(report, LEXICON[target]) for report in reports]
    return pd.DataFrame(result, index=range(len(frame))).astype(float)


def _metadata(frame: pd.DataFrame, series: pd.DataFrame, use_lexicon: bool = False, lexicon_weight: float = 1.0) -> pd.DataFrame:
    sex = frame.get("PatientSex", pd.Series("Unknown", index=frame.index))
    sex = sex.fillna("Unknown").astype(str).replace({"": "Unknown"})
    result = pd.get_dummies(sex, prefix="sex", dtype=float)
    result.index = range(len(frame))

    if series.empty or KEY_COLUMN not in series.columns:
        if use_lexicon:
            result = result.join(_lexicon_features(frame) * lexicon_weight)
        return result.fillna(0).astype(float)

    work = series.copy()
    work[KEY_COLUMN] = work[KEY_COLUMN].astype(str)
    grouped = work.groupby(KEY_COLUMN, dropna=False)
    keyed = pd.DataFrame(index=work[KEY_COLUMN].drop_duplicates())
    keyed["n_series"] = grouped.size()
    for column in ("Fluid_Sensitive", "Fat_Suppression"):
        if column in work.columns:
            values = pd.to_numeric(work[column], errors="coerce").fillna(0)
            keyed[f"n_{column.lower()}"] = values.groupby(work[KEY_COLUMN]).sum()
    if "Anatomical_Plane" in work.columns:
        planes = pd.crosstab(work[KEY_COLUMN], work["Anatomical_Plane"])
        planes.columns = [f"plane_{str(column).lower()}" for column in planes.columns]
        keyed = keyed.join(planes, how="left")
    keyed = np.log1p(keyed.fillna(0).astype(float))

    ids = frame[KEY_COLUMN].astype(str)
    for column in keyed.columns:
        result[column] = ids.map(keyed[column]).fillna(0).to_numpy()
    if use_lexicon:
        result = result.join(_lexicon_features(frame) * lexicon_weight)
    return result.fillna(0).astype(float)


def _validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    expected = [KEY_COLUMN, *TARGET_COLUMNS]
    assert list(submission.columns) == expected, "Colunas da submissão fora de ordem."
    assert submission[KEY_COLUMN].astype(str).tolist() == test[KEY_COLUMN].astype(str).tolist(), "UIDs fora de ordem."
    values = submission[TARGET_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(values).all(), "A submissão contém NaN ou infinito."
    assert ((values >= 0) & (values <= 1)).all(), "A submissão contém valores fora de [0, 1]."
    assert not submission[KEY_COLUMN].duplicated().any(), "A submissão contém UIDs duplicados."


def _resolve_data_dir(data_dir: Path) -> Path:
    required = ("train.csv", "test.csv")
    if all((data_dir / filename).is_file() for filename in required):
        print(f"data_dir={data_dir}")
        return data_dir

    input_root = Path("/kaggle/input")
    candidates: set[Path] = set()
    if input_root.is_dir():
        search_patterns = ("train.csv", "*/train.csv", "*/*/train.csv")
        for pattern in search_patterns:
            for train_path in input_root.glob(pattern):
                candidate = train_path.parent
                if (candidate / "test.csv").is_file():
                    candidates.add(candidate)

    if len(candidates) == 1:
        resolved = next(iter(candidates))
        print(f"data_dir solicitado={data_dir}; usando fonte montada={resolved}")
        return resolved

    available = []
    if input_root.is_dir():
        available = sorted(path.name for path in input_root.iterdir())[:50]
    raise FileNotFoundError(
        "Dados da competição não encontrados. "
        f"data_dir solicitado={data_dir}; candidatos={sorted(map(str, candidates))}; "
        f"entradas em /kaggle/input={available}"
    )


def run(data_dir: Path, output: Path, c: float = 2.0, use_lexicon: bool = False, lexicon_weight: float = 1.0) -> pd.DataFrame:
    if lexicon_weight <= 0:
        raise ValueError("lexicon_weight precisa ser positivo.")
    data_dir = _resolve_data_dir(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    train_series = pd.read_csv(data_dir / "train_series.csv") if (data_dir / "train_series.csv").exists() else pd.DataFrame()
    test_series = pd.read_csv(data_dir / "test_series.csv") if (data_dir / "test_series.csv").exists() else pd.DataFrame()

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode", max_features=120_000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True, max_features=120_000)
    x_word = word.fit_transform(_reports(train))
    x_char = char.fit_transform(_reports(train))
    train_meta = _metadata(train, train_series, use_lexicon=use_lexicon, lexicon_weight=lexicon_weight)
    test_meta = _metadata(test, test_series, use_lexicon=use_lexicon, lexicon_weight=lexicon_weight).reindex(columns=train_meta.columns, fill_value=0)
    x_train = hstack([x_word, x_char, csr_matrix(train_meta.to_numpy())], format="csr")
    x_test = hstack([word.transform(_reports(test)), char.transform(_reports(test)), csr_matrix(test_meta.to_numpy())], format="csr")

    submission = pd.DataFrame({KEY_COLUMN: test[KEY_COLUMN].astype(str).to_numpy()})
    for target in TARGET_COLUMNS:
        labels = pd.to_numeric(train.get(target, pd.Series(np.nan, index=train.index)), errors="coerce")
        labeled = labels.notna().to_numpy()
        values = labels.loc[labeled].astype(float).to_numpy()
        if values.size == 0:
            predictions = np.full(len(test), 0.5)
        elif np.unique(values).size < 2:
            predictions = np.full(len(test), values.mean())
        else:
            classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=800, solver="liblinear")
            classifier.fit(x_train[labeled], values)
            predictions = classifier.predict_proba(x_test)[:, 1]
        submission[target] = np.clip(predictions, 1e-6, 1 - 1e-6)

    _validate_submission(submission[[KEY_COLUMN, *TARGET_COLUMNS]], test)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission[[KEY_COLUMN, *TARGET_COLUMNS]].to_csv(output, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("RSNA_DATA_DIR", "/kaggle/input/rsna-knee-abnormality-detection"))
    parser.add_argument("--c", type=float, default=32.0)
    parser.add_argument("--use-lexicon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lexicon-weight", type=float, default=1.0)
    parser.add_argument("--output", default=os.environ.get("RSNA_OUTPUT", "/kaggle/working/submission.csv"))
    args = parser.parse_args()
    submission = run(Path(args.data_dir), Path(args.output), c=args.c, use_lexicon=args.use_lexicon, lexicon_weight=args.lexicon_weight)
    print(f"submission gravada em {args.output} com {len(submission)} linhas; C={args.c}; lexicon={args.use_lexicon}; lexicon_weight={args.lexicon_weight}")


if __name__ == "__main__":
    main()
