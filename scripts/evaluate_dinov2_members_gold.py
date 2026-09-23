#!/usr/bin/env python3
"""Gate local para os ensembles públicos DINOv2 ``champ`` e ``llm199e30``.

Os checkpoints são públicos e foram encontrados no HD externo, mas o notebook
de treino original não está disponível para execução. Este script reproduz a
arquitetura persistida no checkpoint e a receita pública recuperada do código
H-37: seis slots por protocolo, três slabs 2.5D por slot, crop físico de
130 mm, DINOv2-Small e ``SlotHead``. O resultado é explicitamente uma
reprodução compatível/diagnóstica no conjunto-ouro de 58 estudos; não é OOF e
não escolhe automaticamente uma submissão.

Exemplos:

    python3 scripts/evaluate_dinov2_members_gold.py \
      --root . --family champ --device mps

    python3 scripts/evaluate_dinov2_members_gold.py \
      --root . --family both --device mps --batch-size 1

Os DICOMs, checkpoints, caches e JSONs gerados estão cobertos pelo
``.gitignore``. Nenhum identificador de estudo é impresso no log resumido,
mas os IDs permanecem nos artefatos locais para permitir comparação segura.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


TARGETS = [
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

# This is the recovered six-slot scheme from the public H-37 family. The
# binary competition flags alone cannot distinguish the two non-fat-sat
# sagittal sequences, so those slots are resolved from DICOM protocol tags.
SLOT_SPECS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
N_SLOT = len(SLOT_SPECS)
N_GROUP = 3
GROUP = 3
IMAGE_SIZE = 224
CROP_MM = 130.0
WINDOW = (0.35, 0.65)
BOOTSTRAPS = 2000
cv2.setNumThreads(1)

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
    r"water excit|\btirm\b|\bsting\b",
    re.IGNORECASE,
)
T1_RX = re.compile(r"\bt1\b|\bt1w\b", re.IGNORECASE)
T2_RX = re.compile(r"\bt2\b|\bt2w\b", re.IGNORECASE)
PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens", re.IGNORECASE)
SEPARATORS_RX = re.compile(r"[_\-.]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"rank_columns espera matriz 2D, recebeu {array.shape}")
    return pd.DataFrame(array).rank(method="average", pct=True).to_numpy(dtype=float)


def auc_by_target(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        target: float(roc_auc_score(labels[:, index], predictions[:, index]))
        for index, target in enumerate(TARGETS)
        if np.unique(labels[:, index]).size >= 2
    }


def macro_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    values = list(auc_by_target(labels, predictions).values())
    if not values:
        return float("nan")
    return float(np.mean(values))


def bootstrap_delta(
    candidate: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    seed: int,
    count: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(count, dtype=float)
    n = len(labels)
    for index in range(count):
        sample = rng.integers(0, n, size=n)
        per_target: list[float] = []
        for target_index in range(len(TARGETS)):
            y = labels[sample, target_index]
            if np.unique(y).size < 2:
                continue
            per_target.append(
                roc_auc_score(y, candidate[sample, target_index])
                - roc_auc_score(y, reference[sample, target_index])
            )
        deltas[index] = np.mean(per_target) if per_target else np.nan
    valid = deltas[np.isfinite(deltas)]
    return {
        "mean": float(np.mean(valid)),
        "ci95_low": float(np.quantile(valid, 0.025)),
        "ci95_high": float(np.quantile(valid, 0.975)),
        "n": int(len(valid)),
    }


def _float(value: object, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _header_value(header: pydicom.dataset.FileDataset, name: str) -> str:
    value = getattr(header, name, "")
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "|".join(str(item) for item in value)
    return str(value or "")


def _descriptor(record: dict[str, Any]) -> str:
    value = " ".join(
        str(record.get(field, ""))
        for field in ("SeriesDescription", "ProtocolName", "SequenceName")
    )
    return SEPARATORS_RX.sub(" ", value).lower()


def _annotate_record(record: dict[str, Any]) -> dict[str, Any]:
    description = _descriptor(record)
    scan_options = str(record.get("ScanOptions", "")).upper()
    option_tokens = {token.strip() for token in scan_options.split("|")}
    fatsat = bool(FATSAT_RX.search(description)) or bool(option_tokens & FATSAT_OPTS)
    tr = _float(record.get("RepetitionTime"))
    te = _float(record.get("EchoTime"))
    scanning = str(record.get("ScanningSequence", "")).upper()
    gre = "GR" in scanning
    t1 = bool(T1_RX.search(description))
    t2 = bool(T2_RX.search(description))
    pdw = bool(PD_RX.search(description))
    if t1 and not t2 and not pdw:
        weight = "T1"
    elif t2 and not pdw:
        weight = "T2"
    elif pdw:
        weight = "PD"
    elif gre:
        weight = "GRE"
    elif tr is not None and tr < 800:
        weight = "T1"
    elif te is not None and te > 60:
        weight = "T2"
    elif tr is not None and tr >= 800:
        weight = "PD"
    else:
        weight = "UNK"
    result = dict(record)
    result["fatsat"] = fatsat
    result["weight"] = weight
    result["fluid"] = weight in {"PD", "T2"}
    return result


def _probe_series(
    row: dict[str, Any],
    series_root: Path,
) -> dict[str, Any]:
    study_uid = str(row["StudyInstanceUID"])
    series_uid = str(row["SeriesInstanceUID"])
    directory = series_root / study_uid / series_uid
    files = sorted(path.name for path in directory.glob("*.dcm"))
    record: dict[str, Any] = {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "plane": str(row.get("Anatomical_Plane", "")),
        "fluid_flag": int(_float(row.get("Fluid_Sensitive"), 0.0) or 0),
        "fat_flag": int(_float(row.get("Fat_Suppression"), 0.0) or 0),
        "files": files,
        "n_slices": len(files),
        "pixel_spacing": None,
    }
    if not files:
        return _annotate_record(record)
    try:
        header_path = directory / files[len(files) // 2]
        header = pydicom.dcmread(header_path, stop_before_pixels=True, force=True)
        for field in (
            "SeriesDescription",
            "ProtocolName",
            "SequenceName",
            "ScanOptions",
            "ScanningSequence",
            "RepetitionTime",
            "EchoTime",
        ):
            record[field] = _header_value(header, field)
        spacing = getattr(header, "PixelSpacing", None)
        if spacing is not None and len(spacing) >= 2:
            record["pixel_spacing"] = [float(spacing[0]), float(spacing[1])]
    except Exception as exc:  # pragma: no cover - depends on a corrupt DICOM
        record["probe_error"] = type(exc).__name__
    return _annotate_record(record)


def _select_slots(records: list[dict[str, Any]], study_ids: list[str]) -> dict[str, Any]:
    by_study: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_study.setdefault(str(record["study_uid"]), []).append(record)
    selected: dict[str, Any] = {}
    coverage = {name: 0 for name, *_ in SLOT_SPECS}
    for study_uid in study_ids:
        study_slots: dict[str, dict[str, Any] | None] = {}
        candidates = by_study.get(study_uid, [])
        for name, plane, fluid, fatsat in SLOT_SPECS:
            matches = [
                record
                for record in candidates
                if record.get("plane") == plane
                and bool(record.get("fluid")) == fluid
                and bool(record.get("fatsat")) == fatsat
                and int(record.get("n_slices", 0)) > 0
            ]
            # The public recipe picks the longest series within a slot. UID is
            # a deterministic tie-breaker for this local reproduction.
            matches.sort(key=lambda item: (-int(item["n_slices"]), str(item["series_uid"])))
            chosen = matches[0] if matches else None
            study_slots[name] = chosen
            coverage[name] += int(chosen is not None)
        selected[study_uid] = study_slots
    return {"study_ids": study_ids, "coverage": coverage, "studies": selected}


def _manifest_path(cache_dir: Path, order: str) -> Path:
    return cache_dir / f"selection_manifest_{order}.json"


def load_or_build_manifest(
    root: Path,
    study_ids: list[str],
    cache_dir: Path,
    workers: int,
) -> dict[str, Any]:
    path = _manifest_path(cache_dir, "physical")
    if path.is_file():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("study_ids") == study_ids:
                print(f"selection manifest cache={path}", flush=True)
                return manifest
        except (OSError, ValueError):
            pass

    series_path = root / "train_series.csv"
    series_root = root / "train_series"
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    selected_ids = set(study_ids)
    rows = series[series["StudyInstanceUID"].astype(str).isin(selected_ids)].to_dict("records")
    records: list[dict[str, Any]] = []
    print(f"probing {len(rows)} gold series headers with workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_probe_series, row, series_root) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index == 1 or index % 64 == 0 or index == len(futures):
                print(f"probed {index}/{len(futures)}", flush=True)
    records.sort(key=lambda item: (str(item["study_uid"]), str(item["series_uid"])))
    manifest = _select_slots(records, study_ids)
    manifest["records"] = records
    manifest["selection_policy"] = {
        "slot_specs": SLOT_SPECS,
        "protocol_annotation": "H-37 recovered fatsat/fluid regex from DICOM tags",
        "series_selection": "longest matching series, then SeriesInstanceUID",
        "note": "This is compatible reproduction; original Steven notebook source was unavailable.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest


def _image_headers(
    directory: Path,
    files: list[str],
    order: str,
) -> list[str]:
    rows: list[tuple[float | None, float | None, str]] = []
    for name in files:
        path = directory / name
        position: float | None = None
        instance: float | None = None
        try:
            header = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                force=True,
                specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"],
            )
            raw_instance = _float(getattr(header, "InstanceNumber", None))
            instance = raw_instance
            if order == "physical":
                ipp = np.asarray(getattr(header, "ImagePositionPatient"), dtype=float)
                iop = np.asarray(getattr(header, "ImageOrientationPatient"), dtype=float)
                if ipp.size >= 3 and iop.size >= 6 and np.isfinite(ipp[:3]).all() and np.isfinite(iop[:6]).all():
                    position = float(np.dot(ipp[:3], np.cross(iop[:3], iop[3:6])))
        except Exception:
            pass
        rows.append((position, instance, name))
    if order == "physical" and all(item[0] is not None for item in rows):
        rows.sort(key=lambda item: (float(item[0]), item[2]))
    elif all(item[1] is not None for item in rows):
        rows.sort(key=lambda item: (float(item[1]), item[2]))
    else:
        rows.sort(key=lambda item: item[2])
    return [item[2] for item in rows]


def _read_dicom(path: Path) -> tuple[np.ndarray, tuple[float, float] | None]:
    dataset = pydicom.dcmread(path, force=True)
    pixels = dataset.pixel_array.astype(np.float32)
    slope = _float(getattr(dataset, "RescaleSlope", 1.0), 1.0) or 1.0
    intercept = _float(getattr(dataset, "RescaleIntercept", 0.0), 0.0) or 0.0
    pixels = pixels * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        pixels = float(np.max(pixels)) - pixels
    spacing = getattr(dataset, "PixelSpacing", None)
    if spacing is None or len(spacing) < 2:
        return pixels, None
    try:
        return pixels, (float(spacing[0]), float(spacing[1]))
    except (TypeError, ValueError):
        return pixels, None


def _crop_center(volume: np.ndarray, spacing: tuple[float, float] | None, crop_mm: float) -> np.ndarray:
    if spacing is None:
        return volume
    row_mm, col_mm = spacing
    crop_rows = int(round(crop_mm / max(row_mm, 1e-3)))
    crop_cols = int(round(crop_mm / max(col_mm, 1e-3)))
    crop_rows = min(volume.shape[-2], max(1, crop_rows))
    crop_cols = min(volume.shape[-1], max(1, crop_cols))
    y0 = max(0, (volume.shape[-2] - crop_rows) // 2)
    x0 = max(0, (volume.shape[-1] - crop_cols) // 2)
    return volume[..., y0 : y0 + crop_rows, x0 : x0 + crop_cols]


def _render_group(
    raw: list[np.ndarray],
    spacings: list[tuple[float, float] | None],
    common_spacing: list[float] | None,
) -> np.ndarray:
    if not raw:
        raise ValueError("slab vazio")
    shape = raw[0].shape
    if any(array.shape != shape for array in raw):
        raise ValueError("dimensões divergentes no slab")
    spacing = None
    if common_spacing is not None and len(common_spacing) >= 2:
        spacing = (float(common_spacing[0]), float(common_spacing[1]))
    elif spacings and spacings[0] is not None:
        spacing = spacings[0]
    volume = _crop_center(np.stack(raw), spacing, CROP_MM)
    low, high = np.percentile(volume, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((volume - low) / (high - low), 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(normalized[None])
    resized = F.interpolate(
        tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0].numpy()
    return np.rint(np.clip(resized, 0.0, 1.0) * 255.0).astype(np.uint8)


def _decode_slot(
    record: dict[str, Any],
    series_root: Path,
    order: str,
) -> np.ndarray | None:
    directory = series_root / str(record["study_uid"]) / str(record["series_uid"])
    files = _image_headers(directory, list(record.get("files", [])), order)
    if len(files) < GROUP + 2:
        return None
    centers = [int(round(fraction * (len(files) - 1))) for fraction in np.linspace(WINDOW[0], WINDOW[1], N_GROUP)]
    centers = list(dict.fromkeys(centers))
    if len(centers) != N_GROUP:
        return None
    groups: list[np.ndarray] = []
    common_spacing = record.get("pixel_spacing")
    for center in centers:
        indices = [center - 1, center, center + 1]
        if min(indices) < 0 or max(indices) >= len(files):
            return None
        raw: list[np.ndarray] = []
        spacings: list[tuple[float, float] | None] = []
        try:
            for index in indices:
                pixels, spacing = _read_dicom(directory / files[index])
                raw.append(pixels)
                spacings.append(spacing)
            groups.append(_render_group(raw, spacings, common_spacing))
        except Exception:
            return None
    return np.stack(groups).astype(np.uint8)


def _decode_study(
    study_uid: str,
    slots: dict[str, dict[str, Any] | None],
    series_root: Path,
    order: str,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, Any]]:
    images = np.zeros((N_SLOT, N_GROUP, GROUP, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask = np.zeros(N_SLOT, dtype=np.uint8)
    selected: dict[str, Any] = {}
    for slot_index, (name, _plane, _fluid, _fatsat) in enumerate(SLOT_SPECS):
        record = slots.get(name)
        selected[name] = None if record is None else {
            "series_uid": str(record["series_uid"]),
            "n_slices": int(record.get("n_slices", 0)),
            "description": str(record.get("SeriesDescription", "")),
            "weight": str(record.get("weight", "")),
            "fatsat": bool(record.get("fatsat")),
        }
        if record is None:
            continue
        decoded = _decode_slot(record, series_root, order)
        if decoded is not None and decoded.shape == images[slot_index].shape:
            images[slot_index] = decoded
            mask[slot_index] = 1
    return study_uid, images, mask, {"selected": selected, "filled_slots": int(mask.sum())}


def _cache_paths(cache_dir: Path, order: str) -> dict[str, Path]:
    prefix = cache_dir / f"gold_dinov2_members_{order}"
    return {
        "images": prefix.with_suffix(".npz"),
        "diagnostics": prefix.with_name(prefix.name + "_diagnostics.json"),
    }


def load_or_build_images(
    manifest: dict[str, Any],
    series_root: Path,
    cache_dir: Path,
    order: str,
    workers: int,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    paths = _cache_paths(cache_dir, order)
    study_ids = [str(value) for value in manifest["study_ids"]]
    if paths["images"].is_file():
        try:
            cached = np.load(paths["images"], allow_pickle=False)
            cached_ids = cached["ids"].astype(str).tolist()
            if cached_ids == study_ids:
                print(f"image cache={paths['images']} shape={cached['images'].shape}", flush=True)
                diagnostics = {}
                if paths["diagnostics"].is_file():
                    diagnostics = json.loads(paths["diagnostics"].read_text(encoding="utf-8"))
                return study_ids, cached["images"], cached["masks"], diagnostics
        except (OSError, ValueError, KeyError):
            pass

    output = np.zeros((len(study_ids), N_SLOT, N_GROUP, GROUP, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    masks = np.zeros((len(study_ids), N_SLOT), dtype=np.uint8)
    diagnostics: dict[str, Any] = {}
    index_by_id = {study_uid: index for index, study_uid in enumerate(study_ids)}
    study_cache = cache_dir / f"gold_dinov2_members_{order}_studies"
    study_cache.mkdir(parents=True, exist_ok=True)
    missing: list[tuple[int, str]] = []
    for index, study_uid in enumerate(study_ids):
        partial_path = study_cache / f"study_{index:04d}.npz"
        if not partial_path.is_file():
            missing.append((index, study_uid))
            continue
        try:
            cached = np.load(partial_path, allow_pickle=False)
            cached_uid = str(cached["study_uid"].item())
            cached_images = cached["images"]
            cached_mask = cached["mask"]
            if (
                cached_uid != study_uid
                or cached_images.shape != (N_SLOT, N_GROUP, GROUP, IMAGE_SIZE, IMAGE_SIZE)
                or cached_mask.shape != (N_SLOT,)
            ):
                missing.append((index, study_uid))
                continue
            output[index] = cached_images
            masks[index] = cached_mask
            diagnostics[study_uid] = {"filled_slots": int(cached_mask.sum()), "from_partial_cache": True}
        except (OSError, ValueError, KeyError):
            missing.append((index, study_uid))
    print(
        f"decoding {len(missing)}/{len(study_ids)} gold studies with workers={workers}; "
        f"partial_cache={len(study_ids) - len(missing)}",
        flush=True,
    )
    if not missing:
        np.savez_compressed(paths["images"], ids=np.asarray(study_ids, dtype="U80"), images=output, masks=masks)
        paths["diagnostics"].write_text(json.dumps(diagnostics, ensure_ascii=False), encoding="utf-8")
        return study_ids, output, masks, diagnostics
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _decode_study,
                study_uid,
                manifest["studies"][study_uid],
                series_root,
                order,
            )
            for _index, study_uid in missing
        ]
        for done, future in enumerate(as_completed(futures), 1):
            study_uid, images, mask, diagnostic = future.result()
            output[index_by_id[study_uid]] = images
            masks[index_by_id[study_uid]] = mask
            diagnostics[study_uid] = diagnostic
            index = index_by_id[study_uid]
            np.savez_compressed(
                study_cache / f"study_{index:04d}.npz",
                study_uid=np.asarray(study_uid, dtype="U80"),
                images=images,
                mask=mask,
            )
            if done == 1 or done % 8 == 0 or done == len(futures):
                print(f"decoded {done}/{len(futures)} filled_slots={diagnostic['filled_slots']}", flush=True)
    paths["images"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["images"], ids=np.asarray(study_ids, dtype="U80"), images=output, masks=masks)
    paths["diagnostics"].write_text(json.dumps(diagnostics, ensure_ascii=False), encoding="utf-8")
    return study_ids, output, masks, diagnostics


class EncoderWrapper(nn.Module):
    """Names the HF body as ``encoder.module`` like the public checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model

        config = Dinov2Config(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            image_size=518,
            patch_size=14,
            hidden_act="gelu",
            qkv_bias=True,
            layerscale_value=1.0,
            use_mask_token=True,
        )
        self.module = Dinov2Model(config)


class SlotHead(nn.Module):
    """SlotHead recovered from the public H-37 model family."""

    def __init__(self, dim: int = 768, n_slot: int = N_SLOT, n_out: int = len(TARGETS)) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, 256) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, 256) * 0.02)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(256, n_out)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(features) + self.slot_emb.unsqueeze(0)
        attention = torch.einsum("bsh,oh->bos", hidden, self.query) / math.sqrt(256.0)
        attention = attention.masked_fill(~mask.bool().unsqueeze(1), -10000.0).softmax(dim=-1)
        context = self.drop(torch.einsum("bos,bsh->boh", attention, hidden))
        return (context * self.out.weight.unsqueeze(0)).sum(dim=-1) + self.out.bias.unsqueeze(0)


class DinoMember(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = EncoderWrapper()
        self.head = SlotHead()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, slots, groups, channels, height, width = images.shape
        values = images.reshape(batch * slots * groups, channels, height, width).float().div_(255.0)
        values = (values - self.mean) / self.std
        tokens = self.encoder.module(pixel_values=values, interpolate_pos_encoding=True).last_hidden_state
        features = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)
        features = features.reshape(batch, slots, groups, -1).mean(dim=2)
        return self.head(features, mask)


def load_member(path: Path, device: torch.device) -> tuple[DinoMember, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DinoMember()
    model.load_state_dict(checkpoint["model"], strict=True)
    metadata = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
        "fold": int(checkpoint.get("fold", -1)),
        "fingerprint": checkpoint.get("fingerprint", {}),
        "provenance": checkpoint.get("provenance", {}),
    }
    del checkpoint
    model.eval().to(device)
    return model, metadata


@torch.no_grad()
def infer_member(
    model: DinoMember,
    images: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        batch_images = torch.from_numpy(np.asarray(images[start:stop])).to(device)
        batch_masks = torch.from_numpy(np.asarray(masks[start:stop])).to(device)
        logits = model(batch_images, batch_masks)
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())
        del batch_images, batch_masks, logits
        if start == 0 or stop == len(images) or stop % max(batch_size * 8, 1) == 0:
            print(f"infer {stop}/{len(images)}", flush=True)
    return np.concatenate(predictions, axis=0)


def family_paths(model_dir: Path, family: str) -> list[Path]:
    patterns = {
        "champ": "champ_fold*.pt",
        "llm199e30": "llm199e30_fold*.pt",
    }
    subdirectories = {
        "champ": "champ-members-only",
        "llm199e30": "llm199-e30-members",
    }
    candidate_dirs = [model_dir]
    nested = model_dir / subdirectories[family]
    if nested != model_dir:
        candidate_dirs.insert(0, nested)
    paths: list[Path] = []
    for candidate_dir in candidate_dirs:
        paths = sorted(candidate_dir.glob(patterns[family]))
        if paths:
            break
    if len(paths) != 5:
        raise FileNotFoundError(f"{family}: esperava cinco checkpoints em {model_dir}, encontrei {len(paths)}")
    return paths


def load_h38_reference(
    root: Path,
    study_ids: list[str],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if len(study_ids) != 58:
        return None
    h36_path = root.parent / "reports" / "h36_gold_gate_20260902.json"
    dino_path = root / "processed" / "dinov3_v16_gold_public336" / "gold_dinov3_v16_predictions.npz"
    if not h36_path.is_file() or not dino_path.is_file():
        return None
    report = json.loads(h36_path.read_text(encoding="utf-8"))
    predictions = report.get("predictions", {}).get("h36", {})
    if set(predictions) != set(study_ids):
        return None
    h36_raw = np.asarray([predictions[study_uid] for study_uid in study_ids], dtype=float)
    dino = np.load(dino_path, allow_pickle=False)
    if dino["ids"].astype(str).tolist() != study_ids:
        return None
    h36_rank = rank_columns(h36_raw)
    dino_rank = np.asarray(dino["base_rank"], dtype=float)
    h38 = rank_columns(0.80 * h36_rank + 0.20 * dino_rank)
    return h38, {
        "formula": "rank(0.80 * rank(H36) + 0.20 * DINOv3_base_rank)",
        "h36_report": str(h36_path),
        "dino_npz": str(dino_path),
    }


def compare_against_h38(
    labels: np.ndarray,
    h38: np.ndarray,
    candidate: np.ndarray,
    bootstrap_count: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0):
        blended = rank_columns((1.0 - weight) * h38 + weight * candidate)
        results[f"h38_{1.0 - weight:.2f}_candidate_{weight:.2f}"] = {
            "h38_weight": 1.0 - weight,
            "candidate_weight": weight,
            "macro_auc": macro_auc(labels, blended),
            "auc_by_target": auc_by_target(labels, blended),
            "delta_vs_h38_bootstrap": bootstrap_delta(
                blended,
                h38,
                labels,
                seed=20260903 + int(weight * 1000),
                count=bootstrap_count,
            ),
        }
    return results


def evaluate_family(
    family: str,
    model_dir: Path,
    images: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    fold_predictions: list[np.ndarray] = []
    model_metadata: list[dict[str, Any]] = []
    for path in family_paths(model_dir, family):
        print(f"loading {family}/{path.name} on {device}", flush=True)
        model, metadata = load_member(path, device)
        prediction = infer_member(model, images, masks, device, batch_size)
        if prediction.shape != (len(images), len(TARGETS)) or not np.isfinite(prediction).all():
            raise RuntimeError(f"predição inválida de {path}")
        fold_predictions.append(prediction)
        metadata["macro_auc"] = macro_auc(labels, prediction)
        metadata["auc_by_target"] = auc_by_target(labels, prediction)
        model_metadata.append(metadata)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
    folds = np.stack(fold_predictions, axis=0)
    ensemble = folds.mean(axis=0)
    ensemble_rank = rank_columns(ensemble)
    result = {
        "checkpoints": model_metadata,
        "ensemble": {
            "macro_auc_raw": macro_auc(labels, ensemble),
            "auc_by_target_raw": auc_by_target(labels, ensemble),
            "macro_auc_rank": macro_auc(labels, ensemble_rank),
            "auc_by_target_rank": auc_by_target(labels, ensemble_rank),
        },
        "fold_predictions": folds.tolist(),
        "ensemble_predictions": ensemble.tolist(),
        "ensemble_rank": ensemble_rank.tolist(),
    }
    return result, ensemble_rank


def resolve_device(value: str) -> torch.device:
    if value == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS solicitado, mas não está disponível")
        return torch.device("mps")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--family", choices=("champ", "llm199e30", "both"), default="champ")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="limita estudos para smoke test")
    parser.add_argument("--order", choices=("physical", "instance"), default="physical")
    parser.add_argument("--bootstrap-count", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1 or args.bootstrap_count < 1:
        raise ValueError("batch-size, workers e bootstrap-count precisam ser positivos")

    project_root = args.root.expanduser().resolve()
    data_root = project_root / "data" / "raw"
    if not (data_root / "train.csv").is_file():
        raise FileNotFoundError(f"train.csv não encontrado em {data_root}")
    model_root = (args.model_dir or project_root / "data" / "models").expanduser().resolve()
    cache_dir = (args.cache_dir or project_root / "data" / "processed" / "dinov2_members_gold_224").expanduser().resolve()
    output_path = (args.output or project_root / "reports" / f"dinov2_members_{args.family}_gold.json").expanduser().resolve()
    device = resolve_device(args.device)
    print(f"device={device} torch={torch.__version__} root={project_root}", flush=True)

    train = pd.read_csv(data_root / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[train[TARGETS].notna().all(axis=1)].copy()
    if args.limit is not None:
        gold = gold.head(args.limit).copy()
    study_ids = gold["StudyInstanceUID"].astype(str).tolist()
    labels = gold[TARGETS].to_numpy(dtype=np.float32)
    workers = min(args.workers, max(1, len(study_ids)))
    manifest = load_or_build_manifest(data_root, study_ids, cache_dir, workers)
    _, images, masks, diagnostics = load_or_build_images(
        manifest,
        data_root / "train_series",
        cache_dir,
        args.order,
        workers,
    )
    if images.shape != (len(study_ids), N_SLOT, N_GROUP, GROUP, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"cache de imagens com shape inesperado: {images.shape}")
    print(f"slot coverage manifest={manifest['coverage']} decoded={masks.sum(axis=0).astype(int).tolist()}", flush=True)

    families = ("champ", "llm199e30") if args.family == "both" else (args.family,)
    family_results: dict[str, Any] = {}
    family_ranks: dict[str, np.ndarray] = {}
    started = time.time()
    for family in families:
        family_result, family_rank = evaluate_family(
            family,
            model_root,
            images,
            masks,
            labels,
            device,
            args.batch_size,
        )
        family_results[family] = family_result
        family_ranks[family] = family_rank

    h38_loaded = load_h38_reference(project_root / "data", study_ids)
    comparisons: dict[str, Any] = {}
    h38_rank = None
    if h38_loaded is not None:
        h38_rank, h38_metadata = h38_loaded
        comparisons["h38_reference"] = {
            "macro_auc": macro_auc(labels, h38_rank),
            "auc_by_target": auc_by_target(labels, h38_rank),
            "metadata": h38_metadata,
        }
        for family, family_rank in family_ranks.items():
            comparisons[family] = compare_against_h38(labels, h38_rank, family_rank, args.bootstrap_count)
            correlations = {
                target: float(np.corrcoef(h38_rank[:, index], family_rank[:, index])[0, 1])
                for index, target in enumerate(TARGETS)
            }
            family_results[family]["correlation_vs_h38_rank"] = correlations

    output = {
        "format": "rsna-dinov2-public-members-gold-compatible-gate-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "diagnostic_only": True,
        "study_count": len(study_ids),
        "study_ids": study_ids,
        "gold_label_policy": "all 12 official labels non-null",
        "device": str(device),
        "preprocessing": {
            "slots": SLOT_SPECS,
            "image_size": IMAGE_SIZE,
            "crop_mm": CROP_MM,
            "group_channels": GROUP,
            "n_groups_per_slot": N_GROUP,
            "center_window": WINDOW,
            "slice_order": args.order,
            "intensity_window": "per_group percentile 1-99",
            "normalization": "checkpoint mean/std (ImageNet values in audited weights)",
            "laterality_flip": False,
        },
        "selection_policy": manifest["selection_policy"],
        "slot_coverage_manifest": manifest["coverage"],
        "slot_coverage_decoded": dict(zip((name for name, *_ in SLOT_SPECS), masks.sum(axis=0).astype(int).tolist())),
        "families": family_results,
        "h38_comparisons_diagnostic_only": comparisons,
        "decision_rule": "não escolher peso pelo gold; exigir reprodução Kaggle e/ou validação OOF antes de promover",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output": str(output_path),
        "study_count": len(study_ids),
        "slot_coverage_decoded": output["slot_coverage_decoded"],
        "families": {
            family: result["ensemble"]
            for family, result in family_results.items()
        },
        "h38_macro_auc": None if h38_rank is None else macro_auc(labels, h38_rank),
        "elapsed_seconds": output["elapsed_seconds"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
