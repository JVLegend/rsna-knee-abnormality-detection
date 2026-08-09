#!/usr/bin/env python3
"""Candidata visual v2 multi-view para o notebook Kaggle.

Combina o baseline de laudos/metadados com embeddings EfficientNet-B0 de uma
representação 2.5D DICOM. O código é autocontido para o modo Notebook-only:
ele não clona o repositório, não instala pacotes e não usa internet.
"""

from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import efficientnet_b0


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
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
SAMPLE_QUANTILES = (0.25, 0.5, 0.75)
WEIGHTS_FILENAME = "efficientnet_b0_rwightman-7f5810bc.pth"
TARGETWISE_MODE = False
TARGETWISE_VISUAL_WEIGHTS = {
    "ACL": 0.5,
    "MCL": 0.5,
    "Medial Meniscus": 0.4,
    "Lateral Meniscus": 0.1,
    "Medial OA": 0.0,
    "Lateral OA": 0.1,
    "PF OA": 0.0,
    "Effusion": 0.5,
    "Synovitis": 0.0,
    "Baker's": 0.4,
    "Contusion": 0.2,
    "Fracture": 0.1,
}


def _resolve_data_dir(requested: Path) -> Path:
    if (requested / "train.csv").is_file() and (requested / "test.csv").is_file():
        return requested
    root = Path("/kaggle/input")
    candidates: set[Path] = set()
    if root.is_dir():
        for pattern in ("train.csv", "*/train.csv", "*/*/train.csv"):
            for train_path in root.glob(pattern):
                candidate = train_path.parent
                if (candidate / "test.csv").is_file():
                    candidates.add(candidate)
    if len(candidates) == 1:
        return next(iter(candidates))
    available = sorted(path.name for path in root.iterdir())[:50] if root.is_dir() else []
    raise FileNotFoundError(
        f"Dados não encontrados: solicitado={requested}; candidatos={sorted(map(str, candidates))}; "
        f"/kaggle/input={available}"
    )


def _find_weights() -> Path:
    explicit = os.environ.get("RSNA_EFFICIENTNET_WEIGHTS")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    root = Path("/kaggle/input")
    if root.is_dir():
        candidates.extend(root.glob("*.pth"))
        candidates.extend(root.glob("*/*.pth"))
        candidates.extend(root.glob("*/*/*.pth"))
    candidates.append(Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_FILENAME)
    for candidate in candidates:
        if candidate.is_file() and candidate.name == WEIGHTS_FILENAME:
            return candidate
    raise FileNotFoundError(
        f"Pesos {WEIGHTS_FILENAME} não encontrados; fontes verificadas={list(map(str, candidates))[:20]}"
    )


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


def _metadata(frame: pd.DataFrame, series: pd.DataFrame, use_lexicon: bool = True) -> pd.DataFrame:
    sex = frame.get("PatientSex", pd.Series("Unknown", index=frame.index))
    sex = sex.fillna("Unknown").astype(str).replace({"": "Unknown"})
    result = pd.get_dummies(sex, prefix="sex", dtype=float)
    result.index = range(len(frame))
    if not series.empty and KEY_COLUMN in series.columns:
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
        result = result.join(_lexicon_features(frame))
    return result.fillna(0).astype(float)


def _fit_probabilities(x_train, labels: pd.Series, x_test, c: float) -> np.ndarray:
    values = pd.to_numeric(labels, errors="coerce")
    labeled = values.notna().to_numpy()
    y = values.loc[labeled].to_numpy(dtype=float)
    if y.size == 0:
        return np.full(x_test.shape[0], 0.5, dtype=float)
    if np.unique(y).size < 2:
        return np.full(x_test.shape[0], float(y.mean()), dtype=float)
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, solver="liblinear")
    model.fit(x_train[labeled], y)
    return np.clip(model.predict_proba(x_test)[:, 1], 1e-6, 1 - 1e-6)


def text_predictions(train: pd.DataFrame, test: pd.DataFrame, train_series: pd.DataFrame, test_series: pd.DataFrame) -> pd.DataFrame:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode", max_features=120_000)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True, max_features=120_000)
    x_word = word.fit_transform(_reports(train))
    x_char = char.fit_transform(_reports(train))
    train_meta = _metadata(train, train_series)
    test_meta = _metadata(test, test_series).reindex(columns=train_meta.columns, fill_value=0)
    x_train = hstack([x_word, x_char, csr_matrix(train_meta.to_numpy())], format="csr")
    x_test = hstack([word.transform(_reports(test)), char.transform(_reports(test)), csr_matrix(test_meta.to_numpy())], format="csr")
    result = pd.DataFrame({KEY_COLUMN: test[KEY_COLUMN].astype(str).to_numpy()})
    for target in TARGET_COLUMNS:
        result[target] = _fit_probabilities(x_train, train.get(target, pd.Series(np.nan, index=train.index)), x_test, c=32.0)
    return result


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_key(item: tuple[Path, object]) -> tuple[object, ...]:
    path, dataset = item
    instance = _number(getattr(dataset, "InstanceNumber", None))
    if instance is not None:
        return (0, instance, path.name)
    position = getattr(dataset, "ImagePositionPatient", None)
    z_position = _number(position[-1]) if position is not None and len(position) >= 3 else None
    if z_position is not None:
        return (1, z_position, path.name)
    return (2, path.name)


def _preferred_series(series: pd.DataFrame, study_uid: str) -> str | None:
    if series.empty or KEY_COLUMN not in series.columns or "SeriesInstanceUID" not in series.columns:
        return None
    candidates = series[series[KEY_COLUMN].astype(str).eq(study_uid)].copy()
    if candidates.empty:
        return None
    fluid = candidates["Fluid_Sensitive"] if "Fluid_Sensitive" in candidates.columns else pd.Series(0, index=candidates.index)
    fat = candidates["Fat_Suppression"] if "Fat_Suppression" in candidates.columns else pd.Series(0, index=candidates.index)
    plane = candidates["Anatomical_Plane"] if "Anatomical_Plane" in candidates.columns else pd.Series("", index=candidates.index)
    candidates["_fluid"] = pd.to_numeric(fluid, errors="coerce").fillna(0)
    candidates["_fat"] = pd.to_numeric(fat, errors="coerce").fillna(0)
    plane_order = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
    candidates["_plane"] = plane.map(plane_order).fillna(9)
    candidates = candidates.sort_values(["_fluid", "_fat", "_plane", "SeriesInstanceUID"], ascending=[False, False, True, True])
    return str(candidates.iloc[0]["SeriesInstanceUID"])


def _preferred_series_uids(series: pd.DataFrame, study_uid: str) -> list[str]:
    """Escolhe até uma série fluido-sensível por plano anatômico."""

    if series.empty or KEY_COLUMN not in series.columns or "SeriesInstanceUID" not in series.columns:
        return []
    candidates = series[series[KEY_COLUMN].astype(str).eq(study_uid)].copy()
    if candidates.empty:
        return []
    fluid = candidates["Fluid_Sensitive"] if "Fluid_Sensitive" in candidates.columns else pd.Series(0, index=candidates.index)
    fat = candidates["Fat_Suppression"] if "Fat_Suppression" in candidates.columns else pd.Series(0, index=candidates.index)
    plane = candidates["Anatomical_Plane"] if "Anatomical_Plane" in candidates.columns else pd.Series("", index=candidates.index)
    candidates["_fluid"] = pd.to_numeric(fluid, errors="coerce").fillna(0)
    candidates["_fat"] = pd.to_numeric(fat, errors="coerce").fillna(0)
    candidates["_plane_name"] = plane.fillna("").astype(str)
    selected: list[str] = []
    for plane_name in ("Sagittal", "Coronal", "Axial"):
        subset = candidates[candidates["_plane_name"].eq(plane_name)].sort_values(
            ["_fluid", "_fat", "SeriesInstanceUID"], ascending=[False, False, True]
        )
        if not subset.empty:
            selected.append(str(subset.iloc[0]["SeriesInstanceUID"]))
    if selected:
        return selected
    fallback = candidates.sort_values(["_fluid", "_fat", "SeriesInstanceUID"], ascending=[False, False, True])
    return [str(fallback.iloc[0]["SeriesInstanceUID"])]


def _sample_indices(n_slices: int) -> tuple[int, ...]:
    if n_slices < 1:
        raise ValueError("A série DICOM não contém fatias.")
    return tuple(int(round(value * (n_slices - 1))) for value in SAMPLE_QUANTILES)


def _normalize_slice(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.0))
    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    result = np.clip((values - low) / (high - low), 0.0, 1.0)
    result[~np.isfinite(result)] = 0.0
    return result.astype(np.float32)


def _read_pixels(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(path, force=True)
    pixels = dataset.pixel_array.astype(np.float32)
    slope = _number(getattr(dataset, "RescaleSlope", 1.0)) or 1.0
    intercept = _number(getattr(dataset, "RescaleIntercept", 0.0)) or 0.0
    pixels = pixels * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        pixels = float(np.max(pixels)) - pixels
    return pixels


def _series_image(data_dir: Path, split: str, study_uid: str, series_uid: str, size: int = 224) -> tuple[np.ndarray, bool]:
    series_dir = data_dir / f"{split}_series" / study_uid / series_uid
    paths = sorted(series_dir.glob("*.dcm"))
    if not paths:
        return np.zeros((3, size, size), dtype=np.uint8), False
    records: list[tuple[Path, object]] = []
    for path in paths:
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            records.append((path, header))
        except Exception:
            continue
    if not records:
        return np.zeros((3, size, size), dtype=np.uint8), False
    records.sort(key=_sort_key)
    channels: list[np.ndarray] = []
    for index in _sample_indices(len(records)):
        try:
            normalized = _normalize_slice(_read_pixels(records[index][0]))
            image = Image.fromarray(np.rint(normalized * 255).astype(np.uint8), mode="L")
            channels.append(np.asarray(image.resize((size, size), resample=Image.Resampling.BILINEAR), dtype=np.uint8))
        except Exception:
            return np.zeros((3, size, size), dtype=np.uint8), False
    return np.stack(channels, axis=0), True


def study_image(data_dir: Path, split: str, study_uid: str, series: pd.DataFrame, size: int = 224) -> tuple[np.ndarray, bool]:
    series_uid = _preferred_series(series, study_uid)
    if series_uid is None:
        return np.zeros((3, size, size), dtype=np.uint8), False
    return _series_image(data_dir, split, study_uid, series_uid, size)


def study_images(data_dir: Path, split: str, study_uid: str, series: pd.DataFrame, size: int = 224) -> tuple[list[np.ndarray], bool]:
    images: list[np.ndarray] = []
    for series_uid in _preferred_series_uids(series, study_uid):
        image, valid = _series_image(data_dir, split, study_uid, series_uid, size)
        if valid:
            images.append(image)
    return images, bool(images)


def _load_encoder(device: str) -> tuple[torch.nn.Module, str, Path]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = _find_weights()
    model = efficientnet_b0(weights=None)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    encoder = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten()).to(device).eval()
    return encoder, device, weights_path


def visual_embeddings(
    data_dir: Path,
    train_labeled: pd.DataFrame,
    test: pd.DataFrame,
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    batch_size: int = 16,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    encoder, device, weights_path = _load_encoder(device)
    records = [("train", row, train_series) for _, row in train_labeled.iterrows()]
    records.extend(("test", row, test_series) for _, row in test.iterrows())
    images: list[np.ndarray] = []
    image_owners: list[int] = []
    valid_flags: list[bool] = []
    study_embeddings: list[list[np.ndarray]] = [[] for _ in records]
    mean = IMAGE_MEAN.to(device)
    std = IMAGE_STD.to(device)

    def flush() -> None:
        if not images:
            return
        batch = torch.from_numpy(np.stack(images).astype(np.float32) / 255.0)
        with torch.inference_mode():
            values = encoder(batch.to(device).sub(mean).div(std)).cpu().numpy().astype(np.float32)
        for owner, value in zip(image_owners, values):
            study_embeddings[owner].append(value)
        images.clear()
        image_owners.clear()

    for position, (split, row, series) in enumerate(records, start=1):
        owner = position - 1
        study_images_for_row, valid = study_images(data_dir, split, str(row[KEY_COLUMN]), series)
        for image in study_images_for_row:
            images.append(image)
            image_owners.append(owner)
        valid_flags.append(valid)
        if len(images) >= batch_size:
            flush()
        if position % 25 == 0 or position == len(records):
            print(f"visual={position}/{len(records)} valid={sum(valid_flags)} views={sum(len(values) for values in study_embeddings)}")
    flush()
    matrix = np.stack(
        [np.mean(values, axis=0) if values else np.zeros(1280, dtype=np.float32) for values in study_embeddings]
    ).astype(np.float32)
    valid = np.asarray(valid_flags, dtype=bool)
    matrix[~valid] = 0.0
    train_count = len(train_labeled)
    meta = {
        "device": device,
        "weights_path": str(weights_path),
        "weights_filename": weights_path.name,
        "study_count_train": train_count,
        "study_count_test": len(test),
        "valid_train": int(valid[:train_count].sum()),
        "valid_test": int(valid[train_count:].sum()),
        "views_total": int(sum(len(values) for values in study_embeddings)),
        "views_mean_valid_study": float(np.mean([len(values) for values in study_embeddings if values])) if any(study_embeddings) else 0.0,
        "max_views_study": int(max((len(values) for values in study_embeddings), default=0)),
        "series_selection": "one_best_fluid_fat_series_per_anatomical_plane",
        "embedding_shape": list(matrix.shape),
    }
    return matrix[:train_count], matrix[train_count:], meta


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    expected = [KEY_COLUMN, *TARGET_COLUMNS]
    assert list(submission.columns) == expected, "Colunas da submissão fora de ordem."
    assert submission[KEY_COLUMN].astype(str).tolist() == test[KEY_COLUMN].astype(str).tolist(), "UIDs fora de ordem."
    values = submission[TARGET_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(values).all(), "A submissão contém NaN ou infinito."
    assert ((values >= 0) & (values <= 1)).all(), "A submissão contém valores fora de [0, 1]."
    assert not submission[KEY_COLUMN].duplicated().any(), "A submissão contém UIDs duplicados."


def run(
    data_dir: Path,
    output: Path,
    visual_weight: float = 0.4,
    batch_size: int = 16,
    device: str = "auto",
    targetwise: bool = False,
) -> pd.DataFrame:
    if not 0 <= visual_weight <= 1:
        raise ValueError("visual_weight precisa estar entre 0 e 1.")
    if targetwise and set(TARGETWISE_VISUAL_WEIGHTS) != set(TARGET_COLUMNS):
        raise ValueError("Os pesos targetwise precisam cobrir exatamente os 12 alvos.")
    started = time.perf_counter()
    data_dir = _resolve_data_dir(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    train_series = pd.read_csv(data_dir / "train_series.csv") if (data_dir / "train_series.csv").exists() else pd.DataFrame()
    test_series = pd.read_csv(data_dir / "test_series.csv") if (data_dir / "test_series.csv").exists() else pd.DataFrame()
    labeled = train[TARGET_COLUMNS].notna().any(axis=1)
    train_labeled = train.loc[labeled].reset_index(drop=True)
    if train_labeled.empty:
        raise ValueError("train.csv não contém estudos rotulados.")

    text = text_predictions(train, test, train_series, test_series)
    train_visual, test_visual, visual_meta = visual_embeddings(
        data_dir,
        train_labeled,
        test,
        train_series,
        test_series,
        batch_size=batch_size,
        device=device,
    )
    visual = pd.DataFrame({KEY_COLUMN: test[KEY_COLUMN].astype(str).to_numpy()})
    for target in TARGET_COLUMNS:
        labels = train_labeled[target]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=1500, solver="liblinear"),
        )
        values = pd.to_numeric(labels, errors="coerce")
        mask = values.notna().to_numpy()
        y = values.loc[mask].to_numpy(dtype=float)
        if y.size == 0:
            visual[target] = 0.5
        elif np.unique(y).size < 2:
            visual[target] = float(y.mean())
        else:
            model.fit(train_visual[mask], y)
            visual[target] = np.clip(model.predict_proba(test_visual)[:, 1], 1e-6, 1 - 1e-6)

    submission = text.copy()
    for target in TARGET_COLUMNS:
        alpha = TARGETWISE_VISUAL_WEIGHTS[target] if targetwise else visual_weight
        submission[target] = np.clip(
            (1 - alpha) * text[target].to_numpy(dtype=float)
            + alpha * visual[target].to_numpy(dtype=float),
            1e-6,
            1 - 1e-6,
        )
    validate_submission(submission, test)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"data_dir={data_dir}")
    print(f"labeled_train={len(train_labeled)} test={len(test)} visual_weight={visual_weight} targetwise={targetwise}")
    if targetwise:
        print(f"targetwise_visual_weights={TARGETWISE_VISUAL_WEIGHTS}")
    print(f"visual_meta={visual_meta}")
    print(f"submission gravada em {output} com {len(submission)} linhas; elapsed={time.perf_counter() - started:.1f}s")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("RSNA_DATA_DIR", "/kaggle/input/rsna-knee-abnormality-detection"))
    parser.add_argument("--visual-weight", type=float, default=0.4)
    parser.add_argument("--targetwise", action="store_true", default=TARGETWISE_MODE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default=os.environ.get("RSNA_OUTPUT", "/kaggle/working/submission.csv"))
    args = parser.parse_args()
    run(
        Path(args.data_dir),
        Path(args.output),
        visual_weight=args.visual_weight,
        batch_size=args.batch_size,
        device=args.device,
        targetwise=args.targetwise,
    )


if __name__ == "__main__":
    main()
