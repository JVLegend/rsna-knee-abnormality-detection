#!/usr/bin/env python3
"""Candidata v4 agressiva: dense 2.5D + pooling visual por alvo.

Esta variante mantém o controle v3 (EfficientNet-B0, Steven v4 mascarado pelo
v2 e blend texto-imagem), mas explora duas hipóteses do documento vivo:

* seis fatias por série em vez de três quantis;
* um classificador visual por alvo treinado em views e agregado por mean/top-k
  conforme o achado é difuso ou focal.

O entrypoint é autocontido para Notebook-only: não clona o repositório, não
instala pacotes e não usa internet. Use ``--slice-profile quantile3
--view-pooling mean`` para reproduzir o controle v3.
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
SLICE_PROFILES = {
    "quantile3": (0.25, 0.5, 0.75),
    # Mesmo número de centros do controle, mas cada centro vira um slab 2.5D
    # com vizinhas imediatas. Esta é a ablação rápida da v4: mantém os três
    # planos e reduz o custo de 18 para 9 views por estudo.
    "adjacent3": (0.25, 0.5, 0.75),
    "dense6": (0.10, 0.25, 0.40, 0.60, 0.75, 0.90),
    "dense9": (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 0.95),
}
SLICE_PROFILE = os.environ.get("RSNA_SLICE_PROFILE", "dense6")
VIEW_POOLING = os.environ.get("RSNA_VIEW_POOLING", "target")
TEACHER_PROFILE = os.environ.get("RSNA_TEACHER_PROFILE", "targetwise")
PILKWANG_LABEL_FILENAME = "report_labels_v2.csv"
TEACHER_BY_TARGET = {
    "ACL": "pilkwang",
    "MCL": "pilkwang",
    "Medial Meniscus": "steven_v2",
    "Lateral Meniscus": "steven_v4",
    "Medial OA": "steven_v4",
    "Lateral OA": "steven_v4",
    "PF OA": "steven_v4",
    "Effusion": "steven_v4",
    "Synovitis": "steven_v4",
    "Baker's": "steven_v4",
    "Contusion": "steven_v4",
    "Fracture": "pilkwang",
}
TARGET_VIEW_POOLING = {
    "ACL": "topk",
    "MCL": "topk",
    "Medial Meniscus": "topk",
    "Lateral Meniscus": "topk",
    "Medial OA": "mean",
    "Lateral OA": "mean",
    "PF OA": "mean",
    "Effusion": "mean",
    "Synovitis": "mean",
    "Baker's": "topk",
    "Contusion": "topk",
    "Fracture": "topk",
}
WEIGHTS_FILENAME = "efficientnet_b0_rwightman-7f5810bc.pth"
TARGETWISE_MODE = False
WEAK_VISUAL_MODE = True
WEAK_VISUAL_THRESHOLD = 0.85
WEAK_VISUAL_SAMPLE_WEIGHT = 0.10
EXTERNAL_LABEL_V2_FILENAME = "llm_labels_v2.csv"
EXTERNAL_LABEL_V4_FILENAME = "llm_labels_v4_blend.csv"
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
        # Public/private dataset mounts can include a version directory or a
        # nested artifact folder; do not assume a fixed input depth.
        candidates.extend(root.rglob(WEIGHTS_FILENAME))
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


def _series_index(series: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if series.empty or KEY_COLUMN not in series.columns:
        return {}
    work = series.copy()
    work[KEY_COLUMN] = work[KEY_COLUMN].astype(str)
    return {
        str(study_uid): group.reset_index(drop=True)
        for study_uid, group in work.groupby(KEY_COLUMN, sort=False)
    }


def _series_for_study(
    series: pd.DataFrame | dict[str, pd.DataFrame],
    study_uid: str,
) -> pd.DataFrame:
    if isinstance(series, dict):
        return series.get(study_uid, pd.DataFrame())
    if series.empty or KEY_COLUMN not in series.columns:
        return pd.DataFrame()
    return series[series[KEY_COLUMN].astype(str).eq(study_uid)].copy()


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


def _preferred_series(series: pd.DataFrame | dict[str, pd.DataFrame], study_uid: str) -> str | None:
    if isinstance(series, dict):
        candidates = series.get(study_uid, pd.DataFrame()).copy()
    else:
        candidates = _series_for_study(series, study_uid)
    if candidates.empty or "SeriesInstanceUID" not in candidates.columns:
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


def _preferred_series_uids(series: pd.DataFrame | dict[str, pd.DataFrame], study_uid: str) -> list[str]:
    """Escolhe até uma série fluido-sensível por plano anatômico."""

    if isinstance(series, dict):
        candidates = series.get(study_uid, pd.DataFrame()).copy()
    else:
        candidates = _series_for_study(series, study_uid)
    if candidates.empty or "SeriesInstanceUID" not in candidates.columns:
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


def _sample_indices(n_slices: int, slice_profile: str = SLICE_PROFILE) -> tuple[int, ...]:
    if n_slices < 1:
        raise ValueError("A série DICOM não contém fatias.")
    if slice_profile not in SLICE_PROFILES:
        raise ValueError(f"Perfil de fatias desconhecido: {slice_profile}")
    indices = [int(round(value * (n_slices - 1))) for value in SLICE_PROFILES[slice_profile]]
    return tuple(dict.fromkeys(indices))


def _normalize_slice(pixels: np.ndarray, percentile_stride: int = 1) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float32)
    if percentile_stride < 1:
        raise ValueError("percentile_stride precisa ser >= 1.")
    sampled = values
    if percentile_stride > 1:
        sampled = values[::percentile_stride, ::percentile_stride] if values.ndim >= 2 else values[::percentile_stride]
    finite = sampled[np.isfinite(sampled)]
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


def _series_image(
    data_dir: Path,
    split: str,
    study_uid: str,
    series_uid: str,
    size: int = 224,
    slice_profile: str = SLICE_PROFILE,
    fast_preprocess: bool = False,
) -> tuple[list[np.ndarray], bool]:
    series_dir = data_dir / f"{split}_series" / study_uid / series_uid
    paths = sorted(series_dir.glob("*.dcm"))
    if not paths:
        return [], False
    records: list[tuple[Path, object]] = []
    for path in paths:
        try:
            if fast_preprocess:
                header = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    force=True,
                    specific_tags=["InstanceNumber", "ImagePositionPatient"],
                )
            else:
                header = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            records.append((path, header))
        except Exception:
            continue
    if not records:
        return [], False
    records.sort(key=_sort_key)
    centers = _sample_indices(len(records), slice_profile)
    percentile_stride = 4 if fast_preprocess else 1

    if slice_profile == "quantile3":
        slabs: list[np.ndarray] = []
        try:
            channels = []
            for index in centers:
                normalized = _normalize_slice(_read_pixels(records[index][0]), percentile_stride)
                image = Image.fromarray(np.rint(normalized * 255).astype(np.uint8))
                channels.append(np.asarray(image.resize((size, size), resample=Image.Resampling.BILINEAR), dtype=np.uint8))
            slabs.append(np.stack(channels, axis=0))
        except Exception:
            return [], False
        return slabs, True

    slabs = []
    resized_cache: dict[int, np.ndarray] = {}

    def resized_slice(index: int) -> np.ndarray:
        if index not in resized_cache:
            normalized = _normalize_slice(_read_pixels(records[index][0]), percentile_stride)
            image = Image.fromarray(np.rint(normalized * 255).astype(np.uint8))
            resized_cache[index] = np.asarray(
                image.resize((size, size), resample=Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        return resized_cache[index]

    for center in centers:
        indices = (max(0, center - 1), center, min(len(records) - 1, center + 1))
        try:
            slabs.append(np.stack([resized_slice(index) for index in indices], axis=0))
        except Exception:
            continue
    return slabs, bool(slabs)


def study_image(
    data_dir: Path,
    split: str,
    study_uid: str,
    series: pd.DataFrame | dict[str, pd.DataFrame],
    size: int = 224,
    slice_profile: str = SLICE_PROFILE,
    fast_preprocess: bool = False,
) -> tuple[list[np.ndarray], bool]:
    series_uid = _preferred_series(series, study_uid)
    if series_uid is None:
        return [], False
    return _series_image(data_dir, split, study_uid, series_uid, size, slice_profile, fast_preprocess)


def study_images(
    data_dir: Path,
    split: str,
    study_uid: str,
    series: pd.DataFrame | dict[str, pd.DataFrame],
    size: int = 224,
    slice_profile: str = SLICE_PROFILE,
    fast_preprocess: bool = False,
) -> tuple[list[np.ndarray], bool]:
    images: list[np.ndarray] = []
    for series_uid in _preferred_series_uids(series, study_uid):
        series_images, valid = _series_image(
            data_dir,
            split,
            study_uid,
            series_uid,
            size,
            slice_profile,
            fast_preprocess,
        )
        if valid:
            images.extend(series_images)
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


def _pool_embedding_values(values: list[np.ndarray], pooling: str) -> np.ndarray:
    if not values:
        return np.zeros(1280, dtype=np.float32)
    matrix = np.stack(values).astype(np.float32)
    if pooling == "mean":
        return matrix.mean(axis=0)
    if pooling == "max":
        return matrix.max(axis=0)
    if pooling == "topk":
        k = max(1, int(np.ceil(matrix.shape[0] * 0.25)))
        # A feature-wise top-k is a cheap, deterministic proxy for attention;
        # target-specific probability pooling is implemented below.
        return np.sort(matrix, axis=0)[-k:].mean(axis=0)
    raise ValueError(f"Pooling visual desconhecido: {pooling}")


def visual_embeddings(
    data_dir: Path,
    train_labeled: pd.DataFrame,
    test: pd.DataFrame,
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    batch_size: int = 16,
    device: str = "auto",
    slice_profile: str = SLICE_PROFILE,
    pooling: str = "mean",
    fast_preprocess: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], list[list[np.ndarray]], list[list[np.ndarray]]]:
    if slice_profile not in SLICE_PROFILES:
        raise ValueError(f"Perfil de fatias desconhecido: {slice_profile}")
    if pooling not in {"mean", "max", "topk"}:
        raise ValueError(f"Pooling de embedding desconhecido: {pooling}")
    encoder, device, weights_path = _load_encoder(device)
    series_indexes = {"train": _series_index(train_series), "test": _series_index(test_series)}
    records = [("train", row, series_indexes["train"]) for _, row in train_labeled.iterrows()]
    records.extend(("test", row, series_indexes["test"]) for _, row in test.iterrows())
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
        study_images_for_row, valid = study_images(
            data_dir,
            split,
            str(row[KEY_COLUMN]),
            series,
            slice_profile=slice_profile,
            fast_preprocess=fast_preprocess,
        )
        for image in study_images_for_row:
            images.append(image)
            image_owners.append(owner)
        valid_flags.append(valid)
        if len(images) >= batch_size:
            flush()
        if position % 25 == 0 or position == len(records):
            print(f"visual={position}/{len(records)} valid={sum(valid_flags)} views={sum(len(values) for values in study_embeddings)}")
    flush()
    matrix = np.stack([_pool_embedding_values(values, pooling) for values in study_embeddings]).astype(np.float32)
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
        "slice_profile": slice_profile,
        "fast_preprocess": fast_preprocess,
        "percentile_stride": 4 if fast_preprocess else 1,
        "embedding_pooling": pooling,
        "valid_train_mask": valid[:train_count].tolist(),
        "series_selection": "one_best_fluid_fat_series_per_anatomical_plane",
        "embedding_shape": list(matrix.shape),
    }
    return matrix[:train_count], matrix[train_count:], meta, study_embeddings[:train_count], study_embeddings[train_count:]


def _find_external_label_dir(requested: Path | None = None) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested.expanduser())
    configured = os.environ.get("RSNA_EXTERNAL_LABELS_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    root = Path("/kaggle/input")
    if root.is_dir():
        candidates.extend(path for path in root.glob("*") if path.is_dir())
        candidates.append(root)
    for candidate in candidates:
        if (candidate / EXTERNAL_LABEL_V2_FILENAME).is_file() and (candidate / EXTERNAL_LABEL_V4_FILENAME).is_file():
            return candidate
    checked = ", ".join(str(path) for path in candidates[:30])
    raise FileNotFoundError(
        f"Labels externos não encontrados; esperados {EXTERNAL_LABEL_V2_FILENAME} e "
        f"{EXTERNAL_LABEL_V4_FILENAME}; pastas verificadas={checked}"
    )


def _label_search_dirs(requested: Path | None = None) -> list[Path]:
    candidates: list[Path] = []

    def add_with_siblings(path: Path) -> None:
        path = path.expanduser()
        candidates.append(path)
        if path.parent != path:
            candidates.append(path.parent)
            if path.parent.is_dir():
                candidates.extend(child for child in path.parent.glob("*") if child.is_dir())

    if requested is not None:
        add_with_siblings(requested)
    configured = os.environ.get("RSNA_EXTERNAL_LABELS_DIR")
    if configured:
        add_with_siblings(Path(configured))
    root = Path("/kaggle/input")
    if root.is_dir():
        candidates.extend(path for path in root.glob("*") if path.is_dir())
        candidates.append(root)
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _find_label_file(filename: str, requested: Path | None = None) -> Path:
    for directory in _label_search_dirs(requested):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    # Kaggle can mount dataset versions below /kaggle/input/datasets/<slug>/
    # rather than exposing the CSV at the first directory level. Resolve the
    # exact filename recursively, while keeping the shallow lookup above for
    # the common case and for local smoke paths.
    root = Path("/kaggle/input")
    if root.is_dir():
        # Never recurse through the competition mount: it contains hundreds
        # of thousands of DICOMs. External datasets are mounted below a
        # non-competition input folder, commonly /kaggle/input/datasets.
        search_roots = [
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.lower() not in {"competition", "competitions"}
        ]
        for search_root in search_roots:
            matches = sorted(path for path in search_root.rglob(filename) if path.is_file())
            if matches:
                return matches[0]
    checked = ", ".join(str(path) for path in _label_search_dirs(requested)[:30])
    raise FileNotFoundError(f"Arquivo de labels não encontrado: {filename}; pastas={checked}")


def _state_scores(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if KEY_COLUMN not in frame.columns:
        raise ValueError(f"Labels externos {name} sem {KEY_COLUMN}.")
    if frame[KEY_COLUMN].duplicated().any():
        raise ValueError(f"Labels externos {name} têm UIDs duplicados.")
    frame = frame.copy()
    frame[KEY_COLUMN] = frame[KEY_COLUMN].astype(str)
    frame = frame.set_index(KEY_COLUMN)
    result = pd.DataFrame(index=frame.index)
    for target in TARGET_COLUMNS:
        if target not in frame.columns:
            raise ValueError(f"Labels externos {name} sem o alvo {target!r}.")
        score = pd.to_numeric(frame[target], errors="coerce").clip(0.0, 1.0)
        verdict_column = f"{target}__verdict"
        if verdict_column in frame.columns:
            verdict = frame[verdict_column].fillna("UNK").astype(str).str.upper()
            score = score.mask(verdict.eq("UNK"), 0.5)
        confidence_column = f"{target}__conf"
        if confidence_column in frame.columns:
            confidence = pd.to_numeric(frame[confidence_column], errors="coerce").clip(0.0, 1.0)
            if verdict_column in frame.columns:
                confidence = confidence.mask(verdict.eq("UNK"), 0.0)
        else:
            confidence = score.sub(0.5).abs().mul(2.0)
        result[target] = score
        result[f"{target}__confidence"] = confidence
    return result


def _external_teacher(
    train: pd.DataFrame,
    requested_dir: Path | None = None,
    profile: str = TEACHER_PROFILE,
) -> tuple[pd.DataFrame, Path]:
    if profile not in {"steven_v4", "targetwise"}:
        raise ValueError(f"Perfil de teacher desconhecido: {profile}")
    v2_path = _find_label_file(EXTERNAL_LABEL_V2_FILENAME, requested_dir)
    v4_path = _find_label_file(EXTERNAL_LABEL_V4_FILENAME, requested_dir)
    v2 = _state_scores(v2_path, "steven_v2")
    v4 = _state_scores(v4_path, "steven_v4")
    source_frames = {"steven_v2": v2, "steven_v4": v4}
    file_paths = [v2_path, v4_path]
    if profile == "targetwise":
        pilkwang_path = _find_label_file(PILKWANG_LABEL_FILENAME, requested_dir)
        source_frames["pilkwang"] = _state_scores(pilkwang_path, "pilkwang")
        file_paths.append(pilkwang_path)

    ids = train[KEY_COLUMN].astype(str)
    result = pd.DataFrame({KEY_COLUMN: ids.to_numpy()})
    for target in TARGET_COLUMNS:
        source_name = "steven_v4" if profile == "steven_v4" else TEACHER_BY_TARGET[target]
        source = source_frames[source_name]
        score = ids.map(source[target])
        # v4 can assign a value even when v2 says that the report did not
        # address the target. Keep that state neutral for either profile.
        if source_name == "steven_v4":
            not_addressed = ids.map(v2[target]).sub(0.5).abs().le(1e-9)
            score = score.mask(not_addressed, 0.5)
            confidence = score.sub(0.5).abs().mul(2.0)
        else:
            confidence = ids.map(source[f"{target}__confidence"])
        score = score.clip(0.0, 1.0)
        result[target] = score.to_numpy()
        result[f"{target}__confidence"] = confidence.clip(0.0, 1.0).to_numpy()
    result.attrs["label_dir"] = str(v4_path.parent)
    result.attrs["label_files"] = [str(path) for path in file_paths]
    result.attrs["teacher_profile"] = profile
    return result, v4_path.parent


def _weak_target_arrays(
    train: pd.DataFrame,
    teacher: pd.DataFrame,
    threshold: float = WEAK_VISUAL_THRESHOLD,
    sample_weight: float = WEAK_VISUAL_SAMPLE_WEIGHT,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Usa scores externos somente quando a confiança supera o limiar.

    As linhas gold usam o rótulo oficial e peso 1. Os demais estudos entram
    apenas quando o labeler externo está suficientemente distante de 0,5;
    ausência de menção foi previamente convertida em 0,5 pelo adaptador.
    """

    if not 0.5 < threshold < 1:
        raise ValueError("threshold precisa estar entre 0,5 e 1.")
    if sample_weight <= 0:
        raise ValueError("sample_weight precisa ser positivo.")
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for target in TARGET_COLUMNS:
        gold_values = pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
        gold = np.isfinite(gold_values)
        probabilities = pd.to_numeric(teacher[target], errors="coerce").to_numpy(dtype=float)
        confidence_column = f"{target}__confidence"
        confidence = pd.to_numeric(teacher.get(confidence_column), errors="coerce").to_numpy(dtype=float)
        if len(probabilities) != len(train) or len(confidence) != len(train):
            raise ValueError(f"Professor externo inválido para {target!r}.")
        available = np.isfinite(probabilities) & np.isfinite(confidence)
        weak = (~gold) & available & (confidence >= threshold)
        included = gold | weak
        labels = np.where(gold, gold_values, np.where(probabilities >= 0.5, 1.0, 0.0))
        weights = np.zeros(len(train), dtype=float)
        weights[gold] = 1.0
        weights[weak] = sample_weight * np.clip(confidence[weak], 0.0, 1.0)
        result[target] = labels, weights, included
    return result


def _pool_probabilities(probabilities: np.ndarray, pooling: str) -> float:
    if probabilities.size == 0:
        return 0.5
    if pooling == "mean":
        return float(probabilities.mean())
    if pooling == "max":
        return float(probabilities.max())
    if pooling == "topk":
        k = max(1, int(np.ceil(probabilities.size * 0.25)))
        return float(np.sort(probabilities)[-k:].mean())
    raise ValueError(f"Pooling de probabilidade desconhecido: {pooling}")


def _resolve_target_pooling(target: str, override: str | None = None) -> str:
    pooling = override or TARGET_VIEW_POOLING[target]
    if pooling not in {"mean", "max", "topk"}:
        raise ValueError(f"Pooling de probabilidade desconhecido: {pooling}")
    return pooling


def _fit_target_view_model(
    target: str,
    train_views: list[list[np.ndarray]],
    test_views: list[list[np.ndarray]],
    labels: np.ndarray,
    fit_mask: np.ndarray,
    sample_weights: np.ndarray | None,
    target_pooling: str | None = None,
) -> np.ndarray:
    """Treina em views repetidas e agrega probabilidades por alvo.

    O rótulo continua sendo de estudo; repetir o rótulo nas views é uma
    aproximação MIL deliberadamente agressiva. A agregação top-k reduz o risco
    de uma única fatia espúria dominar achados focais.
    """

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for index, values in enumerate(train_views):
        if not fit_mask[index] or not values:
            continue
        block = np.stack(values).astype(np.float32)
        x_parts.append(block)
        y_parts.append(np.full(block.shape[0], labels[index], dtype=float))
        if sample_weights is not None:
            weight_parts.append(np.full(block.shape[0], sample_weights[index], dtype=float))

    if not x_parts:
        return np.full(len(test_views), 0.5, dtype=float)
    x_train = np.vstack(x_parts)
    y_train = np.concatenate(y_parts)
    if np.unique(y_train).size < 2:
        constant = float(y_train.mean())
        return np.full(len(test_views), constant, dtype=float)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=1500, solver="liblinear"),
    )
    if sample_weights is None:
        model.fit(x_train, y_train)
    else:
        model.fit(
            x_train,
            y_train,
            logisticregression__sample_weight=np.concatenate(weight_parts),
        )

    pooling = _resolve_target_pooling(target, target_pooling)
    predictions: list[float] = []
    for values in test_views:
        if not values:
            predictions.append(0.5)
            continue
        probabilities = model.predict_proba(np.stack(values).astype(np.float32))[:, 1]
        predictions.append(_pool_probabilities(probabilities, pooling))
    return np.clip(np.asarray(predictions, dtype=float), 1e-6, 1 - 1e-6)


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
    weak_visual: bool = WEAK_VISUAL_MODE,
    weak_threshold: float = WEAK_VISUAL_THRESHOLD,
    weak_sample_weight: float = WEAK_VISUAL_SAMPLE_WEIGHT,
    external_labels_dir: Path | None = None,
    slice_profile: str = SLICE_PROFILE,
    view_pooling: str = VIEW_POOLING,
    teacher_profile: str = TEACHER_PROFILE,
    fast_preprocess: bool = False,
    target_pooling: str | None = None,
) -> pd.DataFrame:
    if not 0 <= visual_weight <= 1:
        raise ValueError("visual_weight precisa estar entre 0 e 1.")
    if targetwise and set(TARGETWISE_VISUAL_WEIGHTS) != set(TARGET_COLUMNS):
        raise ValueError("Os pesos targetwise precisam cobrir exatamente os 12 alvos.")
    if slice_profile not in SLICE_PROFILES:
        raise ValueError(f"Perfil de fatias desconhecido: {slice_profile}")
    if view_pooling not in {"mean", "max", "topk", "target"}:
        raise ValueError(f"Pooling visual desconhecido: {view_pooling}")
    if teacher_profile not in {"steven_v4", "targetwise"}:
        raise ValueError(f"Perfil de teacher desconhecido: {teacher_profile}")
    if target_pooling is not None and target_pooling not in {"mean", "max", "topk"}:
        raise ValueError(f"Pooling de alvo desconhecido: {target_pooling}")
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

    weak_targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    label_dir: Path | None = None
    if weak_visual:
        # Validate all external inputs before the expensive DICOM pass. This
        # prevents a missing Kaggle attachment from wasting a full worker run.
        teacher, label_dir = _external_teacher(train, external_labels_dir, teacher_profile)
        weak_targets = _weak_target_arrays(
            train,
            teacher,
            threshold=weak_threshold,
            sample_weight=weak_sample_weight,
        )
    text = text_predictions(train, test, train_series, test_series)
    visual_train_frame = train.reset_index(drop=True) if weak_visual else train_labeled
    embedding_pooling = "mean" if view_pooling == "target" else view_pooling
    train_visual, test_visual, visual_meta, train_views, test_views = visual_embeddings(
        data_dir,
        visual_train_frame,
        test,
        train_series,
        test_series,
        batch_size=batch_size,
        device=device,
        slice_profile=slice_profile,
        pooling=embedding_pooling,
        fast_preprocess=fast_preprocess,
    )
    valid_train = np.asarray(
        visual_meta.get("valid_train_mask", [True] * len(visual_train_frame)),
        dtype=bool,
    )
    if len(valid_train) != len(visual_train_frame):
        raise ValueError("A máscara de validade visual não coincide com o treino.")
    visual = pd.DataFrame({KEY_COLUMN: test[KEY_COLUMN].astype(str).to_numpy()})
    weak_meta: dict[str, dict[str, int]] = {}
    for target in TARGET_COLUMNS:
        if weak_visual:
            assert weak_targets is not None
            labels_array, sample_weights, included = weak_targets[target]
            fit_mask = included & valid_train
            labels_for_views = labels_array
            weak_meta[target] = {
                "gold": int(np.isfinite(pd.to_numeric(train[target], errors="coerce")).sum()),
                "weak": int((included & ~train[target].notna().to_numpy()).sum()),
                "fit": int(fit_mask.sum()),
            }
        else:
            labels = pd.to_numeric(train_labeled[target], errors="coerce")
            fit_mask = labels.notna().to_numpy() & valid_train
            labels_for_views = labels.to_numpy(dtype=float)
            sample_weights = None
        if view_pooling == "target":
            visual[target] = _fit_target_view_model(
                target,
                train_views,
                test_views,
                labels_for_views,
                fit_mask,
                sample_weights,
                target_pooling=target_pooling,
            )
        else:
            y = labels_for_views[fit_mask]
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=1500, solver="liblinear"),
            )
            if y.size == 0:
                visual[target] = 0.5
            elif np.unique(y).size < 2:
                visual[target] = float(y.mean())
            else:
                if sample_weights is None:
                    model.fit(train_visual[fit_mask], y)
                else:
                    model.fit(
                        train_visual[fit_mask],
                        y,
                        logisticregression__sample_weight=sample_weights[fit_mask],
                    )
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
    print(
        f"labeled_train={len(train_labeled)} visual_train={len(visual_train_frame)} "
        f"test={len(test)} visual_weight={visual_weight} targetwise={targetwise} "
        f"weak_visual={weak_visual} external_labels={label_dir} "
        f"slice_profile={slice_profile} view_pooling={view_pooling} "
        f"target_pooling={target_pooling or 'per-target'} teacher_profile={teacher_profile}"
    )
    if targetwise:
        print(f"targetwise_visual_weights={TARGETWISE_VISUAL_WEIGHTS}")
    if weak_visual:
        print(f"weak_meta={weak_meta}")
    visual_meta_log = dict(visual_meta)
    visual_meta_log.pop("valid_train_mask", None)
    print(f"visual_meta={visual_meta_log}")
    print(f"submission gravada em {output} com {len(submission)} linhas; elapsed={time.perf_counter() - started:.1f}s")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("RSNA_DATA_DIR", "/kaggle/input/rsna-knee-abnormality-detection"))
    parser.add_argument("--visual-weight", type=float, default=0.4)
    parser.add_argument("--targetwise", action="store_true", default=TARGETWISE_MODE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--weak-visual", action="store_true", default=WEAK_VISUAL_MODE)
    parser.add_argument("--weak-threshold", type=float, default=WEAK_VISUAL_THRESHOLD)
    parser.add_argument("--weak-sample-weight", type=float, default=WEAK_VISUAL_SAMPLE_WEIGHT)
    parser.add_argument("--external-labels-dir", default=os.environ.get("RSNA_EXTERNAL_LABELS_DIR"))
    parser.add_argument("--slice-profile", choices=tuple(SLICE_PROFILES), default=SLICE_PROFILE)
    parser.add_argument("--view-pooling", choices=("mean", "max", "topk", "target"), default=VIEW_POOLING)
    parser.add_argument("--teacher-profile", choices=("steven_v4", "targetwise"), default=TEACHER_PROFILE)
    parser.add_argument("--target-pooling", choices=("default", "mean", "max", "topk"), default="default")
    parser.add_argument("--fast-preprocess", action="store_true")
    parser.add_argument("--output", default=os.environ.get("RSNA_OUTPUT", "/kaggle/working/submission.csv"))
    args = parser.parse_args()
    run(
        Path(args.data_dir),
        Path(args.output),
        visual_weight=args.visual_weight,
        batch_size=args.batch_size,
        device=args.device,
        targetwise=args.targetwise,
        weak_visual=args.weak_visual,
        weak_threshold=args.weak_threshold,
        weak_sample_weight=args.weak_sample_weight,
        external_labels_dir=Path(args.external_labels_dir) if args.external_labels_dir else None,
        slice_profile=args.slice_profile,
        view_pooling=args.view_pooling,
        teacher_profile=args.teacher_profile,
        fast_preprocess=args.fast_preprocess,
        target_pooling=None if args.target_pooling == "default" else args.target_pooling,
    )


if __name__ == "__main__":
    main()
