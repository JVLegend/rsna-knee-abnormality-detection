#!/usr/bin/env python3
"""Baixa somente as séries presentes em um manifesto local."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError


def _series_prefix(entry: dict[str, object]) -> str:
    return f"train_series/{entry['study_uid']}/{entry['series_uid']}/"


def _list_series_files(
    api: KaggleApi,
    competition: str,
    prefixes: set[str],
    page_size: int,
    max_pages: int,
) -> dict[str, list[tuple[str, int]]]:
    found: dict[str, list[tuple[str, int]]] = {prefix: [] for prefix in prefixes}
    token = None
    pages = 0
    retries = 0
    while True:
        if pages >= max_pages:
            raise RuntimeError(f"Limite de {max_pages} páginas atingido antes de localizar todas as séries.")
        try:
            response = api.competition_list_files(competition, page_token=token, page_size=page_size)
            retries = 0
        except HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in {429, 500, 502, 503, 504} or retries >= 3:
                raise
            delay = min(30, 5 * (2**retries))
            retries += 1
            print(f"api_status={status}; retry={retries}/3; aguardando={delay}s", flush=True)
            time.sleep(delay)
            continue
        pages += 1
        for file in response.files:
            for prefix in prefixes:
                if file.name.startswith(prefix):
                    found[prefix].append((file.name, int(getattr(file, "total_bytes", 0))))
        if all(found.values()):
            break
        token = response.next_page_token
        if not token:
            break
    missing = [prefix for prefix, files in found.items() if not files]
    if missing:
        raise RuntimeError(f"Séries não encontradas na API após {pages} páginas: {missing}")
    print(f"api_pages_scanned={pages}")
    return found


def download_manifest(
    manifest: dict[str, object],
    data_dir: Path,
    competition: str,
    page_size: int = 200,
    max_pages: int = 5000,
    dry_run: bool = False,
) -> None:
    entries = manifest.get("studies", [])
    if not entries:
        raise ValueError("Manifesto sem estudos.")

    api = KaggleApi()
    api.authenticate()
    prefixes = {_series_prefix(entry) for entry in entries}
    files_by_prefix = _list_series_files(api, competition, prefixes, page_size, max_pages)
    total_files = sum(len(files) for files in files_by_prefix.values())
    total_bytes = sum(size for files in files_by_prefix.values() for _, size in files)
    print(f"series={len(files_by_prefix)} files={total_files} api_bytes={total_bytes}")
    if dry_run:
        return

    with TemporaryDirectory(prefix="rsna-dicom-") as staging_name:
        staging = Path(staging_name)
        completed = 0
        for prefix, files in files_by_prefix.items():
            destination = data_dir / prefix
            destination.mkdir(parents=True, exist_ok=True)
            for name, _ in files:
                basename = Path(name).name
                target = destination / basename
                if not target.exists():
                    api.competition_download_file(competition, name, path=str(staging), quiet=True)
                    downloaded = staging / basename
                    if not downloaded.exists():
                        raise FileNotFoundError(downloaded)
                    shutil.move(str(downloaded), str(target))
                completed += 1
                print(f"{completed}/{total_files} {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--competition", default="rsna-knee-abnormality-detection")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.expanduser().read_text(encoding="utf-8"))
    download_manifest(manifest, args.data_dir.expanduser(), args.competition, args.page_size, args.max_pages, args.dry_run)


if __name__ == "__main__":
    main()
