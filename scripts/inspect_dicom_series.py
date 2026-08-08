#!/usr/bin/env python3
"""Audita uma série DICOM local sem copiar imagens para o repositório."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pydicom


def summarize(series_dir: Path) -> dict[str, object]:
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .dcm encontrado em {series_dir}")

    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()
    photometric: set[str] = set()
    series_uids: set[str] = set()
    pixel_spacings: set[tuple[str, ...]] = set()
    instances: list[int] = []
    minimums: list[int] = []
    maximums: list[int] = []

    for path in files:
        dataset = pydicom.dcmread(path)
        pixels = dataset.pixel_array
        shapes.add(tuple(int(value) for value in pixels.shape))
        dtypes.add(str(pixels.dtype))
        photometric.add(str(getattr(dataset, "PhotometricInterpretation", "")))
        series_uids.add(str(getattr(dataset, "SeriesInstanceUID", "")))
        spacing = getattr(dataset, "PixelSpacing", [])
        pixel_spacings.add(tuple(str(value) for value in spacing))
        instances.append(int(getattr(dataset, "InstanceNumber", -1)))
        minimums.append(int(np.min(pixels)))
        maximums.append(int(np.max(pixels)))

    return {
        "series_dir": str(series_dir),
        "files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "shapes": sorted([list(shape) for shape in shapes]),
        "dtypes": sorted(dtypes),
        "photometric": sorted(photometric),
        "series_uids": len(series_uids),
        "pixel_spacings": sorted([list(spacing) for spacing in pixel_spacings]),
        "instance_min": min(instances),
        "instance_max": max(instances),
        "intensity_min": min(minimums),
        "intensity_max": max(maximums),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("series_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = summarize(args.series_dir.expanduser())
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
