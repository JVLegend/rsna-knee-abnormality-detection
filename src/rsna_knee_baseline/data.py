"""Leitura e atributos leves para os CSVs da competição."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import KEY_COLUMN, TARGET_COLUMNS


def find_data_dir(explicit: str | Path | None = None) -> Path:
    """Encontra a pasta de dados local ou montada no Kaggle."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            Path("/kaggle/input/rsna-knee-abnormality-detection"),
            Path("/kaggle/input/rsna-knee-abnormalities-detection"),
            Path.cwd() / "data" / "raw",
        ]
    )

    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Não encontrei train.csv e test.csv. Pastas verificadas: {checked}")


def _read_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_competition_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Carrega CSVs sem tentar ler os DICOMs."""

    root = Path(data_dir)
    return {
        "train": _read_csv(root, "train.csv"),
        "test": _read_csv(root, "test.csv"),
        "train_series": _read_csv(root, "train_series.csv"),
        "test_series": _read_csv(root, "test_series.csv"),
        "sample_submission": _read_csv(root, "sample_submission.csv"),
    }


def normalize_report(value: object) -> str:
    """Normaliza apenas espaços; preserva acentos e idioma do laudo."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).split())


def _series_features(series: pd.DataFrame, study_ids: Iterable[object]) -> pd.DataFrame:
    """Cria atributos de sequência estáveis e baratos por estudo."""

    ids = pd.Index(study_ids, name=KEY_COLUMN)
    if series.empty or KEY_COLUMN not in series.columns:
        return pd.DataFrame(index=ids)

    work = series.copy()
    work[KEY_COLUMN] = work[KEY_COLUMN].astype(str)
    grouped = work.groupby(KEY_COLUMN, dropna=False)
    result = pd.DataFrame(index=ids.astype(str))
    result["n_series"] = grouped.size()

    for column in ("Fluid_Sensitive", "Fat_Suppression"):
        if column in work.columns:
            values = pd.to_numeric(work[column], errors="coerce").fillna(0)
            sums = values.groupby(work[KEY_COLUMN]).sum()
            result[f"n_{column.lower()}"] = sums

    if "Anatomical_Plane" in work.columns:
        planes = pd.crosstab(work[KEY_COLUMN], work["Anatomical_Plane"])
        planes.columns = [f"plane_{str(column).lower()}" for column in planes.columns]
        result = result.join(planes, how="left")

    result = result.fillna(0)
    for column in result.columns:
        result[column] = np.log1p(pd.to_numeric(result[column], errors="coerce").fillna(0))
    result.index = result.index.astype(str)
    return result


def build_metadata_features(
    frame: pd.DataFrame,
    series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Retorna atributos tabulares alinhados ao índice do frame."""

    result = pd.DataFrame(index=frame.index)
    sex = frame.get("PatientSex", pd.Series("Unknown", index=frame.index))
    sex = sex.fillna("Unknown").astype(str).replace({"": "Unknown"})
    sex_dummies = pd.get_dummies(sex, prefix="sex", dtype=float)
    result = result.join(sex_dummies)

    if series is not None:
        series_features = _series_features(series, frame[KEY_COLUMN].astype(str))
        series_features.index = frame[KEY_COLUMN].astype(str).values
        series_features.index.name = None
        result = result.reset_index(drop=True).join(series_features.reset_index(drop=True))

    return result.fillna(0).astype(float)


def build_text_frame(frame: pd.DataFrame) -> pd.Series:
    """Retorna laudos com sexo para permitir que o modelo use ambos os campos."""

    reports = frame.get("Report", pd.Series("", index=frame.index)).map(normalize_report)
    sex = frame.get("PatientSex", pd.Series("Unknown", index=frame.index))
    sex = sex.fillna("Unknown").astype(str).replace({"": "Unknown"})
    return reports + " [SEX=" + sex + "]"


def label_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume quantas etiquetas válidas existem por alvo."""

    rows: list[dict[str, int | str]] = []
    for target in TARGET_COLUMNS:
        if target not in frame.columns:
            rows.append({"target": target, "labeled": 0, "positive": 0, "negative": 0})
            continue
        values = pd.to_numeric(frame[target], errors="coerce").dropna()
        rows.append(
            {
                "target": target,
                "labeled": int(values.size),
                "positive": int((values == 1).sum()),
                "negative": int((values == 0).sum()),
            }
        )
    return pd.DataFrame(rows)
