"""Entry point autossuficiente para copiar para um notebook Kaggle.

O arquivo não importa o pacote do repositório: a execução da competição fica
sem internet e precisa continuar funcionando quando este código for colado
como células no notebook. O pacote em ``src/`` continua sendo a fonte de
desenvolvimento local; este arquivo é o snapshot de submissão da v0.
"""

from __future__ import annotations

import argparse
import os
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


def _reports(frame: pd.DataFrame) -> pd.Series:
    report = frame.get("Report", pd.Series("", index=frame.index)).fillna("").astype(str)
    report = report.map(lambda value: " ".join(value.split()))
    sex = frame.get("PatientSex", pd.Series("Unknown", index=frame.index))
    sex = sex.fillna("Unknown").astype(str).replace({"": "Unknown"})
    return report + " [SEX=" + sex + "]"


def _metadata(frame: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    sex = frame.get("PatientSex", pd.Series("Unknown", index=frame.index))
    sex = sex.fillna("Unknown").astype(str).replace({"": "Unknown"})
    result = pd.get_dummies(sex, prefix="sex", dtype=float)
    result.index = range(len(frame))

    if series.empty or KEY_COLUMN not in series.columns:
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
    return result.fillna(0).astype(float)


def run(data_dir: Path, output: Path) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    train_series = pd.read_csv(data_dir / "train_series.csv") if (data_dir / "train_series.csv").exists() else pd.DataFrame()
    test_series = pd.read_csv(data_dir / "test_series.csv") if (data_dir / "test_series.csv").exists() else pd.DataFrame()

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode", max_features=120_000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True, max_features=120_000)
    x_word = word.fit_transform(_reports(train))
    x_char = char.fit_transform(_reports(train))
    train_meta = _metadata(train, train_series)
    test_meta = _metadata(test, test_series).reindex(columns=train_meta.columns, fill_value=0)
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
            classifier = LogisticRegression(C=2.0, class_weight="balanced", max_iter=800, solver="liblinear")
            classifier.fit(x_train[labeled], values)
            predictions = classifier.predict_proba(x_test)[:, 1]
        submission[target] = np.clip(predictions, 1e-6, 1 - 1e-6)

    output.parent.mkdir(parents=True, exist_ok=True)
    submission[[KEY_COLUMN, *TARGET_COLUMNS]].to_csv(output, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("RSNA_DATA_DIR", "/kaggle/input/rsna-knee-abnormality-detection"))
    parser.add_argument("--output", default="/kaggle/working/submission.csv")
    args = parser.parse_args()
    submission = run(Path(args.data_dir), Path(args.output))
    print(f"submission gravada em {args.output} com {len(submission)} linhas")


if __name__ == "__main__":
    main()
