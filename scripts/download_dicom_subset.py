#!/usr/bin/env python3
"""Baixa somente as séries presentes em um manifesto local."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError, RequestException

try:
    from kagglesdk.competitions.types.competition_api_service import ApiListDataTreeFilesRequest
except ImportError:  # Kaggle CLI anterior à API de listagem em árvore.
    ApiListDataTreeFilesRequest = None


def _series_prefix(entry: dict[str, object]) -> str:
    return f"train_series/{entry['study_uid']}/{entry['series_uid']}/"


def _list_series_files(
    api: KaggleApi,
    competition: str,
    prefixes: set[str],
    page_size: int,
    max_pages: int,
    file_cache: Path | None = None,
    tree_delay: float = 0.5,
    tree_retry_attempts: int = 8,
) -> dict[str, list[tuple[str, int]]]:
    cached: dict[str, list[tuple[str, int]]] = {}
    if file_cache is not None and file_cache.is_file():
        raw_cache = json.loads(file_cache.read_text(encoding="utf-8"))
        cached = {
            str(prefix): [(str(name), int(size)) for name, size in files]
            for prefix, files in raw_cache.items()
        }
    if ApiListDataTreeFilesRequest is not None:
        return _list_series_files_tree(
            api,
            competition,
            prefixes,
            page_size,
            max_pages,
            cached,
            file_cache,
            tree_delay,
            tree_retry_attempts,
        )

    return _list_series_files_flat(api, competition, prefixes, page_size, max_pages)


def _list_series_files_tree(
    api: KaggleApi,
    competition: str,
    prefixes: set[str],
    page_size: int,
    max_pages: int,
    cached: dict[str, list[tuple[str, int]]],
    file_cache: Path | None,
    tree_delay: float,
    tree_retry_attempts: int,
) -> dict[str, list[tuple[str, int]]]:
    """Lista diretamente cada diretório de série (Kaggle CLI >= 2.2.2)."""

    found: dict[str, list[tuple[str, int]]] = {
        prefix: list(cached.get(prefix, [])) for prefix in prefixes
    }
    pages = 0
    with api.build_kaggle_client() as client:
        for prefix in sorted(prefixes):
            if found[prefix]:
                continue
            token = None
            while True:
                if pages >= max_pages:
                    raise RuntimeError(f"Limite de {max_pages} páginas atingido antes de localizar todas as séries.")
                request = ApiListDataTreeFilesRequest()
                request.competition_name = competition
                request.path = prefix.rstrip("/")
                request.page_size = page_size
                request.page_token = token
                response = None
                for attempt in range(1, tree_retry_attempts + 1):
                    if tree_delay:
                        time.sleep(tree_delay)
                    try:
                        response = client.competitions.competition_api_client.list_data_tree_files(request)
                        break
                    except HTTPError as exc:
                        status = getattr(exc.response, "status_code", None)
                        if status not in {429, 500, 502, 503, 504} or attempt >= tree_retry_attempts:
                            raise RuntimeError(
                                f"API de árvore retornou HTTP {status} após {attempt} tentativas; "
                                f"cache preservado em {file_cache}"
                            ) from exc
                        delay = min(120, 10 * (2 ** (attempt - 1)))
                        retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After")
                        if retry_after:
                            try:
                                delay = max(delay, float(retry_after))
                            except (TypeError, ValueError):
                                pass
                        print(
                            f"tree_status={status}; retry={attempt}/{tree_retry_attempts}; "
                            f"aguardando={delay:g}s; prefix={prefix}",
                            flush=True,
                        )
                        time.sleep(delay)
                assert response is not None
                pages += 1
                for file in response.files:
                    found[prefix].append((f"{prefix}{file.name}", int(getattr(file, "total_bytes", 0))))
                if file_cache is not None:
                    file_cache.parent.mkdir(parents=True, exist_ok=True)
                    file_cache.write_text(json.dumps(found), encoding="utf-8")
                token = response.next_page_token
                if not token:
                    break

    missing = [prefix for prefix, files in found.items() if not files]
    if missing:
        raise RuntimeError(f"Séries não encontradas na API de árvore: {missing}")
    print(f"api_tree_series={len(found)} api_tree_pages={pages}")
    return found


def _list_series_files_flat(
    api: KaggleApi,
    competition: str,
    prefixes: set[str],
    page_size: int,
    max_pages: int,
) -> dict[str, list[tuple[str, int]]]:
    """Fallback para versões antigas do Kaggle CLI sem listagem em árvore."""

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
    workers: int = 1,
    request_delay: float = 0.75,
    retry_attempts: int = 6,
    file_cache: Path | None = None,
    tree_delay: float = 0.5,
    tree_retry_attempts: int = 8,
) -> None:
    entries = manifest.get("studies", [])
    if not entries:
        raise ValueError("Manifesto sem estudos.")
    if workers < 1:
        raise ValueError("--workers precisa ser positivo.")
    if request_delay < 0:
        raise ValueError("--request-delay não pode ser negativo.")
    if retry_attempts < 1:
        raise ValueError("--retry-attempts precisa ser positivo.")

    api = KaggleApi()
    api.authenticate()
    prefixes = {_series_prefix(entry) for entry in entries}
    files_by_prefix = _list_series_files(
        api,
        competition,
        prefixes,
        page_size,
        max_pages,
        file_cache=file_cache,
        tree_delay=tree_delay,
        tree_retry_attempts=tree_retry_attempts,
    )
    total_files = sum(len(files) for files in files_by_prefix.values())
    total_bytes = sum(size for files in files_by_prefix.values() for _, size in files)
    print(f"series={len(files_by_prefix)} files={total_files} api_bytes={total_bytes}")
    if dry_run:
        return

    with TemporaryDirectory(prefix="rsna-dicom-") as staging_name:
        staging = Path(staging_name)
        pending: list[tuple[str, Path]] = []
        skipped = 0
        for prefix, files in files_by_prefix.items():
            destination = data_dir / prefix
            destination.mkdir(parents=True, exist_ok=True)
            for name, _ in files:
                basename = Path(name).name
                target = destination / basename
                if target.exists():
                    skipped += 1
                else:
                    pending.append((name, target))

        print(
            f"pending={len(pending)} skipped={skipped} workers={workers} "
            f"request_delay={request_delay}s retry_attempts={retry_attempts}",
            flush=True,
        )
        if not pending:
            return

        local_state = threading.local()

        def worker_download(item: tuple[str, Path]) -> str:
            name, target = item
            if not hasattr(local_state, "api"):
                worker_api = KaggleApi()
                worker_api.authenticate()
                worker_dir = staging / f"worker-{threading.get_ident()}"
                worker_dir.mkdir(parents=True, exist_ok=True)
                local_state.api = worker_api
                local_state.directory = worker_dir

            worker_api = local_state.api
            worker_dir = local_state.directory
            basename = Path(name).name
            for attempt in range(1, retry_attempts + 1):
                try:
                    if request_delay:
                        time.sleep(request_delay)
                    worker_api.competition_download_file(competition, name, path=str(worker_dir), quiet=True)
                    downloaded = worker_dir / basename
                    if not downloaded.exists():
                        raise FileNotFoundError(downloaded)
                    shutil.move(str(downloaded), str(target))
                    return name
                except RequestException as exc:
                    status = getattr(exc.response, "status_code", None)
                    if status not in {429, 500, 502, 503, 504} or attempt >= retry_attempts:
                        raise
                    delay = min(30, 5 * (2 ** (attempt - 1)))
                    retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except (TypeError, ValueError):
                            pass
                    print(
                        f"download_status={status}; retry={attempt}/{retry_attempts}; "
                        f"aguardando={delay:g}s",
                        flush=True,
                    )
                    time.sleep(delay)
            raise RuntimeError(f"Falha inesperada ao baixar {name}")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_download, item) for item in pending]
            for completed, future in enumerate(as_completed(futures), start=skipped + 1):
                name = future.result()
                print(f"{completed}/{total_files} {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--competition", default="rsna-knee-abnormality-detection")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--request-delay", type=float, default=0.75)
    parser.add_argument("--retry-attempts", type=int, default=6)
    parser.add_argument("--file-cache", type=Path, default=None)
    parser.add_argument("--tree-delay", type=float, default=0.5)
    parser.add_argument("--tree-retry-attempts", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.expanduser().read_text(encoding="utf-8"))
    download_manifest(
        manifest,
        args.data_dir.expanduser(),
        args.competition,
        args.page_size,
        args.max_pages,
        args.dry_run,
        args.workers,
        args.request_delay,
        args.retry_attempts,
        args.file_cache.expanduser() if args.file_cache else None,
        args.tree_delay,
        args.tree_retry_attempts,
    )


if __name__ == "__main__":
    main()
