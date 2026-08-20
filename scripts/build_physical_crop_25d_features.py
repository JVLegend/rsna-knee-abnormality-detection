#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — ablação 2.5D com crop físico central.

Esta sonda implementa a hipótese observada em baselines públicos: recortar um
campo central de aproximadamente 130 mm e renderizá-lo em 336 px. Ela não
altera o pipeline Kaggle ainda; produz arrays locais no mesmo formato do
builder 2.5D para uma comparação pareada contra a representação H-23.

O centro é uma hipótese operacional, não uma segmentação anatômica. O crop é
calculado por ``PixelSpacing`` de cada série e limitado ao tamanho disponível.
As três fatias usam a mesma janela de intensidade p1–p99 da série, para não
confundir crop com normalização independente por fatia.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from scripts.build_dicom_25d_features import (
    DEFAULT_QUANTILES,
    ROOT,
    _manifest_entries,
    _pixel_array,
    read_series,
    sample_indices,
)


PIXEL_TAGS = (
    "InstanceNumber",
    "PixelData",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PhotometricInterpretation",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "PixelSpacing",
    "RescaleSlope",
    "RescaleIntercept",
)


def _series_window_bounds(pixels: list[np.ndarray]) -> tuple[float, float]:
    finite = [values[np.isfinite(values)] for values in pixels if np.isfinite(values).any()]
    if not finite:
        return 0.0, 1.0
    sampled = np.concatenate(finite)
    low = float(np.percentile(sampled, 1.0))
    high = float(np.percentile(sampled, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(sampled))
        high = float(np.max(sampled))
    return low, high


def _normalize(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def _crop_center(normalized: np.ndarray, pixel_spacing: object, crop_mm: float) -> tuple[np.ndarray, tuple[int, int]]:
    if crop_mm <= 0:
        raise ValueError("crop_mm precisa ser positivo.")
    try:
        spacing = np.asarray(pixel_spacing, dtype=np.float64).reshape(-1)
        row_mm, col_mm = float(spacing[0]), float(spacing[1])
    except (TypeError, ValueError, IndexError):
        return normalized, (normalized.shape[0], normalized.shape[1])
    if not np.isfinite(row_mm) or not np.isfinite(col_mm) or row_mm <= 0 or col_mm <= 0:
        return normalized, (normalized.shape[0], normalized.shape[1])
    crop_rows = min(normalized.shape[0], max(1, int(round(crop_mm / row_mm))))
    crop_cols = min(normalized.shape[1], max(1, int(round(crop_mm / col_mm))))
    row_start = max(0, (normalized.shape[0] - crop_rows) // 2)
    col_start = max(0, (normalized.shape[1] - crop_cols) // 2)
    cropped = normalized[row_start : row_start + crop_rows, col_start : col_start + crop_cols]
    return cropped, (crop_rows, crop_cols)


def _resize(values: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(np.rint(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    return np.asarray(image.resize((size, size), resample=Image.Resampling.BILINEAR), dtype=np.uint8)


def build_entry(
    entry: dict[str, object],
    data_dir: Path,
    output_dir: Path,
    size: int,
    crop_mm: float,
    quantiles: tuple[float, ...],
    overwrite: bool = False,
    sort_mode: str = "header",
) -> dict[str, object]:
    study_uid = str(entry["study_uid"])
    series_uid = str(entry["series_uid"])
    series_dir = data_dir / "train_series" / study_uid / series_uid
    records = read_series(series_dir, sort_by_header=True, sort_mode=sort_mode)
    indices = sample_indices(len(records), quantiles)
    raw: list[np.ndarray] = []
    spacing: object = None
    selected: list[dict[str, object]] = []
    for index in indices:
        path, _ = records[index]
        dataset = pydicom.dcmread(path, specific_tags=PIXEL_TAGS, force=False)
        pixels = _pixel_array(dataset)
        raw.append(pixels)
        if spacing is None:
            spacing = getattr(dataset, "PixelSpacing", None)
        selected.append(
            {
                "index": index,
                "file": path.name,
                "instance_number": getattr(dataset, "InstanceNumber", None),
                "input_shape": list(pixels.shape),
                "pixel_spacing": list(getattr(dataset, "PixelSpacing", [])) if hasattr(dataset, "PixelSpacing") else None,
            }
        )
    bounds = _series_window_bounds(raw)
    channels: list[np.ndarray] = []
    crop_shapes: list[list[int]] = []
    for pixels in raw:
        normalized = _normalize(pixels, bounds)
        cropped, shape = _crop_center(normalized, spacing, crop_mm)
        channels.append(_resize(cropped, size))
        crop_shapes.append(list(shape))
    image = np.stack(channels, axis=0).astype(np.uint8)
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    array_path = arrays_dir / f"{study_uid}__{series_uid}.npz"
    if overwrite or not array_path.exists():
        np.savez_compressed(array_path, image=image, sample_indices=np.asarray(indices, dtype=np.int16))
    return {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "selected_for": entry.get("selected_for", []),
        "labels": entry.get("labels", {}),
        "fluid_sensitive": entry.get("fluid_sensitive"),
        "fat_suppression": entry.get("fat_suppression"),
        "anatomical_plane": entry.get("anatomical_plane", ""),
        "n_slices": len(records),
        "sample_quantiles": list(quantiles),
        "sample_indices": list(indices),
        "intensity_bounds": list(bounds),
        "crop_mm": crop_mm,
        "crop_shapes": crop_shapes,
        "selected": selected,
        "output_shape": list(image.shape),
        "output_dtype": str(image.dtype),
        "array_path": str(array_path.relative_to(output_dir)),
    }


def build_features(
    manifest_paths: list[Path],
    data_dir: Path,
    output_dir: Path,
    size: int = 336,
    crop_mm: float = 130.0,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    overwrite: bool = False,
    workers: int = 4,
    sort_mode: str = "header",
) -> dict[str, object]:
    if size < 16 or crop_mm <= 0 or workers < 1:
        raise ValueError("size/crop_mm/workers inválidos.")
    entries = _manifest_entries(manifest_paths)
    if sort_mode not in {"header", "physical", "filename"}:
        raise ValueError(f"sort_mode desconhecido: {sort_mode}")
    if workers == 1:
        records = [
            build_entry(entry, data_dir, output_dir, size, crop_mm, quantiles, overwrite, sort_mode)
            for entry in entries
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    build_entry,
                    entry,
                    data_dir,
                    output_dir,
                    size,
                    crop_mm,
                    quantiles,
                    overwrite,
                    sort_mode,
                )
                for entry in entries
            ]
            records = [future.result() for future in futures]
    output_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "format": "rsna-knee-dicom-25d-physical-crop-v1",
        "source_manifests": [str(path) for path in manifest_paths],
        "size": size,
        "crop_mm": crop_mm,
        "channels": len(quantiles),
        "quantiles": list(quantiles),
        "sort_mode": sort_mode,
        "studies": len({str(record["study_uid"]) for record in records}),
        "records": records,
    }
    (output_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", dest="manifests", action="append", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dicom_25d_gold_physical_crop_130mm_336"))
    parser.add_argument("--size", type=int, default=336)
    parser.add_argument("--crop-mm", type=float, default=130.0)
    parser.add_argument("--quantiles", default="0.25,0.5,0.75")
    parser.add_argument("--sort-mode", choices=("header", "physical", "filename"), default="header")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifests = [path.expanduser() if path.is_absolute() else ROOT / path for path in args.manifests]
    data_dir = args.data_dir.expanduser()
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    index = build_features(
        manifests,
        data_dir,
        output_dir,
        size=args.size,
        crop_mm=args.crop_mm,
        quantiles=tuple(float(part.strip()) for part in args.quantiles.split(",") if part.strip()),
        overwrite=args.overwrite,
        workers=args.workers,
        sort_mode=args.sort_mode,
    )
    print(f"studies={index['studies']} series={len(index['records'])} shape=({index['channels']},{index['size']},{index['size']})")
    print(f"crop_mm={index['crop_mm']} sort_mode={index['sort_mode']} output={output_dir}")


if __name__ == "__main__":
    main()
