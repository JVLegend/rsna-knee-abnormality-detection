#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — constrói uma representação visual 2.5D de manifestos locais."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUANTILES = (0.25, 0.5, 0.75)
HEADER_TAGS = ("InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient")
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
    "RescaleSlope",
    "RescaleIntercept",
)


def sample_indices(n_slices: int, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> tuple[int, ...]:
    """Retorna índices determinísticos; repete o centro quando a série é curta."""

    if n_slices < 1:
        raise ValueError("A série precisa ter pelo menos uma fatia.")
    if not quantiles or any(value < 0 or value > 1 for value in quantiles):
        raise ValueError("Os quantis precisam estar no intervalo [0, 1].")
    return tuple(int(round(value * (n_slices - 1))) for value in quantiles)


def normalize_slice(pixels: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> tuple[np.ndarray, float, float]:
    """Converte uma fatia para [0, 1] usando percentis robustos."""

    values = np.asarray(pixels, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Esperava imagem 2D; recebi shape={values.shape}.")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32), 0.0, 0.0

    low = float(np.percentile(finite, lower))
    high = float(np.percentile(finite, upper))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32), low, high
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32), low, high


def resize_slice(normalized: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona uma fatia normalizada para uma imagem grayscale uint8."""

    if size < 16:
        raise ValueError("--size precisa ser pelo menos 16.")
    image = Image.fromarray(np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8), mode="L")
    resized = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _number(value: object, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sort_key(item: tuple[Path, pydicom.dataset.FileDataset]) -> tuple[object, ...]:
    path, dataset = item
    instance = _number(getattr(dataset, "InstanceNumber", None))
    if instance is not None:
        return (0, instance, path.name)
    position = getattr(dataset, "ImagePositionPatient", None)
    z_position = _number(position[-1]) if position is not None and len(position) >= 3 else None
    if z_position is not None:
        return (1, z_position, path.name)
    return (2, path.name)


def _physical_sort_key(item: tuple[Path, pydicom.dataset.FileDataset]) -> tuple[object, ...]:
    """Ordena pela posição física ao longo da normal da aquisição.

    ``InstanceNumber`` é um fallback útil, mas pode ser inconsistente entre
    séries. Quando IPP/IOP existem, a projeção de ``ImagePositionPatient`` na
    normal derivada de ``ImageOrientationPatient`` fornece uma ordenação
    determinística no eixo real de aquisição.
    """

    path, dataset = item
    try:
        position = np.asarray(getattr(dataset, "ImagePositionPatient"), dtype=np.float64)
        orientation = np.asarray(getattr(dataset, "ImageOrientationPatient"), dtype=np.float64)
        if position.shape != (3,) or orientation.shape != (6,):
            raise ValueError
        normal = np.cross(orientation[:3], orientation[3:])
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError
        projection = float(np.dot(position, normal / norm))
        if not np.isfinite(projection):
            raise ValueError
        instance = _number(getattr(dataset, "InstanceNumber", None))
        return (0, projection, instance if instance is not None else float("inf"), path.name)
    except (TypeError, ValueError):
        return (1, *_sort_key(item))


def read_series(
    series_dir: Path,
    sort_by_header: bool = True,
    sort_mode: str | None = None,
) -> list[tuple[Path, pydicom.dataset.FileDataset | None]]:
    """Lê somente os headers de uma série e ordena por InstanceNumber/posição."""

    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .dcm encontrado em {series_dir}")
    if not sort_by_header:
        return [(path, None) for path in files]
    effective_sort_mode = sort_mode or "header"
    if effective_sort_mode not in {"header", "physical", "filename"}:
        raise ValueError(f"sort_mode desconhecido: {effective_sort_mode}")
    if effective_sort_mode == "filename":
        return [(path, None) for path in files]
    records = [
        (path, pydicom.dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS, force=False))
        for path in files
    ]
    records.sort(key=_physical_sort_key if effective_sort_mode == "physical" else _sort_key)
    return records


def _pixel_array(dataset: pydicom.dataset.FileDataset) -> np.ndarray:
    pixels = dataset.pixel_array.astype(np.float32)
    slope = _number(getattr(dataset, "RescaleSlope", 1.0), 1.0) or 1.0
    intercept = _number(getattr(dataset, "RescaleIntercept", 0.0), 0.0) or 0.0
    pixels = pixels * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        pixels = float(np.max(pixels)) - pixels
    return pixels


def _manifest_entries(manifest_paths: list[Path]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    seen_series: set[tuple[str, str]] = set()
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.expanduser().read_text(encoding="utf-8"))
        for entry in payload.get("studies", []):
            key = (str(entry["study_uid"]), str(entry["series_uid"]))
            if key in seen_series:
                raise ValueError(f"Série repetida entre manifestos: {key}")
            seen_series.add(key)
            entries.append(entry)
    if not entries:
        raise ValueError("Nenhum estudo encontrado nos manifestos.")
    return entries


def build_entry(
    entry: dict[str, object],
    data_dir: Path,
    output_dir: Path,
    size: int,
    quantiles: tuple[float, ...],
    overwrite: bool = False,
    sort_by_header: bool = True,
    sort_mode: str | None = None,
) -> dict[str, object]:
    study_uid = str(entry["study_uid"])
    series_uid = str(entry["series_uid"])
    series_dir = data_dir / "train_series" / study_uid / series_uid
    records = read_series(series_dir, sort_by_header=sort_by_header, sort_mode=sort_mode)
    indices = sample_indices(len(records), quantiles)
    channels: list[np.ndarray] = []
    selected: list[dict[str, object]] = []

    for index in indices:
        path, header = records[index]
        dataset = pydicom.dcmread(path, specific_tags=PIXEL_TAGS, force=False)
        pixels = _pixel_array(dataset)
        normalized, low, high = normalize_slice(pixels)
        channels.append(resize_slice(normalized, size))
        selected.append(
            {
                "index": index,
                "file": path.name,
                "instance_number": getattr(dataset, "InstanceNumber", None),
                "input_shape": list(pixels.shape),
                "pixel_min": float(np.min(pixels)),
                "pixel_max": float(np.max(pixels)),
                "percentile_low": low,
                "percentile_high": high,
            }
        )

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
        "selected": selected,
        "output_shape": list(image.shape),
        "output_dtype": str(image.dtype),
        "array_path": str(array_path.relative_to(output_dir)),
    }


def build_features(
    manifest_paths: list[Path],
    data_dir: Path,
    output_dir: Path,
    size: int = 224,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    overwrite: bool = False,
    workers: int = 4,
    sort_by_header: bool = True,
    sort_mode: str | None = None,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers precisa ser positivo.")
    entries = _manifest_entries(manifest_paths)
    effective_sort_mode = "filename" if not sort_by_header else (sort_mode or "header")
    if effective_sort_mode not in {"header", "physical", "filename"}:
        raise ValueError(f"sort_mode desconhecido: {effective_sort_mode}")
    if workers == 1:
        records = [
            build_entry(
                entry,
                data_dir,
                output_dir,
                size,
                quantiles,
                overwrite,
                sort_by_header,
                effective_sort_mode,
            )
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
                    quantiles,
                    overwrite,
                    sort_by_header,
                    effective_sort_mode,
                )
                for entry in entries
            ]
            records = [future.result() for future in futures]
    output_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "format": "rsna-knee-dicom-25d-v0",
        "source_manifests": [str(path) for path in manifest_paths],
        "size": size,
        "channels": len(quantiles),
        "quantiles": list(quantiles),
        "sort_mode": effective_sort_mode,
        "studies": len(records),
        "records": records,
    }
    (output_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def _parse_quantiles(value: str) -> tuple[float, ...]:
    quantiles = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not quantiles:
        raise ValueError("--quantiles não pode ser vazio.")
    return quantiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", dest="manifests", action="append", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/dicom_25d_v0"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--quantiles", default="0.25,0.5,0.75")
    parser.add_argument("--sort-mode", choices=("header", "physical", "filename"), default="header")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--fast-file-order",
        action="store_true",
        help="Ablação rápida: amostra a ordem lexicográfica sem ler headers de todas as fatias.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_paths = [path.expanduser() if path.is_absolute() else ROOT / path for path in args.manifests]
    data_dir = args.data_dir.expanduser()
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    index = build_features(
        manifest_paths,
        data_dir,
        output_dir,
        size=args.size,
        quantiles=_parse_quantiles(args.quantiles),
        overwrite=args.overwrite,
        workers=args.workers,
        sort_by_header=not args.fast_file_order,
        sort_mode="filename" if args.fast_file_order else args.sort_mode,
    )
    for position, record in enumerate(index["records"], start=1):
        print(
            f"{position}/{index['studies']} study={record['study_uid']} "
            f"slices={record['n_slices']} samples={record['sample_indices']}"
        )
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
