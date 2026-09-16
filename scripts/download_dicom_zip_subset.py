#!/usr/bin/env python3
"""#RSNA #Kaggle #Dados — extrai séries selecionadas usando HTTP Range no ZIP."""

from __future__ import annotations

import argparse
import binascii
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import zlib
from dataclasses import dataclass

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest


COMPETITION = "rsna-knee-abnormality-detection"
LOCAL_HEADER = struct.Struct("<4s5H3L2H")
CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")


@dataclass(frozen=True)
class ZipEntry:
    filename: str
    flags: int
    compression: int
    crc: int
    compressed_size: int
    file_size: int
    header_offset: int

    @property
    def is_dir(self) -> bool:
        return self.filename.endswith("/")


def _zip_url(competition: str) -> tuple[str, int]:
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = competition
        response = client.competitions.competition_api_client.download_data_files(request)
    return response.url, int(response.headers["Content-Length"])


def _prefix(entry: dict[str, object]) -> str:
    return f"train_series/{entry['study_uid']}/{entry['series_uid']}/"


def _load_manifest(path: Path) -> set[str]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    entries = payload.get("studies", [])
    prefixes = {_prefix(entry) for entry in entries}
    if not prefixes:
        raise ValueError("Manifesto sem séries")
    return prefixes


def _fetch_range(url: str, start: int, end: int, retries: int) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(30, 600),
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"Servidor ignorou Range: HTTP {response.status_code}")
            expected = end - start + 1
            if len(response.content) != expected:
                raise RuntimeError(f"Range incompleto: {len(response.content)} != {expected} bytes")
            return response.content
        except (requests.RequestException, RuntimeError) as exc:
            if attempt >= retries:
                raise
            delay = min(120, 2 ** attempt)
            print(f"range_retry={attempt}/{retries} start={start} end={end} error={exc}; sleep={delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError("Falha inesperada no Range")


def _zip64_extra(
    extra: bytes,
    need_file_size: bool,
    need_compressed_size: bool,
    need_offset: bool,
) -> tuple[int | None, int | None, int | None]:
    file_size = compressed_size = offset = None
    position = 0
    while position + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, position)
        payload = extra[position + 4 : position + 4 + field_size]
        position += 4 + field_size
        if field_id != 0x0001:
            continue
        cursor = 0
        if need_file_size:
            file_size = struct.unpack_from("<Q", payload, cursor)[0]
            cursor += 8
        if need_compressed_size:
            compressed_size = struct.unpack_from("<Q", payload, cursor)[0]
            cursor += 8
        if need_offset:
            offset = struct.unpack_from("<Q", payload, cursor)[0]
        break
    return file_size, compressed_size, offset


def _central_directory(
    url: str,
    archive_size: int,
    prefixes: set[str],
    retries: int,
) -> dict[str, list[ZipEntry]]:
    tail_size = min(8 * 1024 * 1024, archive_size)
    tail_start = archive_size - tail_size
    tail = _fetch_range(url, tail_start, archive_size - 1, retries)
    eocd_position = tail.rfind(b"PK\x05\x06")
    if eocd_position < 0:
        raise RuntimeError("End of central directory não encontrado")
    _, _, _, entries_disk, entries_total, central_size_32, central_offset_32, _ = struct.unpack_from(
        "<4s4H2LH", tail, eocd_position
    )
    if entries_total == 0xFFFF or central_size_32 == 0xFFFFFFFF or central_offset_32 == 0xFFFFFFFF:
        locator_position = tail.rfind(b"PK\x06\x07", 0, eocd_position)
        if locator_position < 0:
            raise RuntimeError("ZIP64 locator não encontrado")
        zip64_offset = struct.unpack_from("<Q", tail, locator_position + 8)[0]
        zip64 = _fetch_range(url, zip64_offset, zip64_offset + 55, retries)
        fields = struct.unpack_from("<4sQ2H2L4Q", zip64)
        entries_total = fields[7]
        central_size = fields[8]
        central_offset = fields[9]
    else:
        central_size = central_size_32
        central_offset = central_offset_32
    central = _fetch_range(url, central_offset, central_offset + central_size - 1, retries)
    selected: dict[str, list[ZipEntry]] = {prefix: [] for prefix in prefixes}
    position = 0
    parsed = 0
    while position + CENTRAL_HEADER.size <= len(central):
        fields = CENTRAL_HEADER.unpack_from(central, position)
        if fields[0] != b"PK\x01\x02":
            break
        (
            _, _, _, flags, compression, _, _, crc, compressed_size_32, file_size_32,
            name_length, extra_length, comment_length, _, _, _, header_offset_32,
        ) = fields
        name_start = position + CENTRAL_HEADER.size
        name_end = name_start + name_length
        extra_end = name_end + extra_length
        name_bytes = central[name_start:name_end]
        filename = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        parts = filename.split("/")
        prefix = "/".join(parts[:3]) + "/" if len(parts) >= 4 else ""
        if prefix in selected and not filename.endswith("/"):
            extra = central[name_end:extra_end]
            file_size_64, compressed_size_64, offset_64 = _zip64_extra(
                extra,
                file_size_32 == 0xFFFFFFFF,
                compressed_size_32 == 0xFFFFFFFF,
                header_offset_32 == 0xFFFFFFFF,
            )
            file_size = file_size_64 if file_size_32 == 0xFFFFFFFF else file_size_32
            compressed_size = compressed_size_64 if compressed_size_32 == 0xFFFFFFFF else compressed_size_32
            header_offset = offset_64 if header_offset_32 == 0xFFFFFFFF else header_offset_32
            if file_size is None or compressed_size is None or header_offset is None:
                raise RuntimeError(f"ZIP64 extra incompleto: {filename}")
            selected[prefix].append(
                ZipEntry(filename, flags, compression, crc, compressed_size, file_size, header_offset)
            )
        position = extra_end + comment_length
        parsed += 1
    if parsed != entries_total:
        raise RuntimeError(f"Central directory incompleto: parsed={parsed} expected={entries_total}")
    return selected


def _extract_entry(blob: bytes, range_start: int, info: ZipEntry) -> bytes:
    local_offset = info.header_offset - range_start
    if local_offset < 0 or local_offset + LOCAL_HEADER.size > len(blob):
        raise RuntimeError(f"Header fora do Range: {info.filename}")
    fields = LOCAL_HEADER.unpack_from(blob, local_offset)
    signature, _, flags, compression, _, _, _, _, _, name_length, extra_length = fields
    if signature != b"PK\x03\x04":
        raise RuntimeError(f"Assinatura ZIP local inválida: {info.filename}")
    data_start = local_offset + LOCAL_HEADER.size + name_length + extra_length
    data_end = data_start + info.compressed_size
    compressed = blob[data_start:data_end]
    if len(compressed) != info.compressed_size:
        raise RuntimeError(f"Dados comprimidos incompletos: {info.filename}")
    if flags & 0x1:
        raise RuntimeError(f"Entrada ZIP criptografada não suportada: {info.filename}")
    if compression != info.compression:
        raise RuntimeError(f"Compressão local/central divergente: {info.filename}")
    if compression == 0:
        raw = compressed
    elif compression == 8:
        raw = zlib.decompress(compressed, -15)
    elif compression == 12:
        import bz2

        raw = bz2.decompress(compressed)
    elif compression == 14:
        import lzma

        raw = lzma.decompress(compressed)
    else:
        raise RuntimeError(f"Compressão ZIP não suportada ({compression}): {info.filename}")
    if len(raw) != info.file_size:
        raise RuntimeError(f"Tamanho inválido após extração: {info.filename}")
    if (binascii.crc32(raw) & 0xFFFFFFFF) != info.crc:
        raise RuntimeError(f"CRC inválido após extração: {info.filename}")
    return raw


def _write_atomic(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == len(content):
        return
    with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".part", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _download_series(
    url: str,
    data_dir: Path,
    prefix: str,
    infos: list[ZipEntry],
    retries: int,
) -> tuple[str, int, int]:
    pending = [info for info in infos if not ((data_dir / info.filename).is_file() and (data_dir / info.filename).stat().st_size == info.file_size)]
    if not pending:
        return prefix, 0, sum(info.file_size for info in infos)
    start = min(info.header_offset for info in pending)
    end = max(
        info.header_offset + LOCAL_HEADER.size + len(info.filename.encode("utf-8")) + 65535 + info.compressed_size - 1
        for info in pending
    )
    blob = _fetch_range(url, start, end, retries)
    written = 0
    for info in pending:
        content = _extract_entry(blob, start, info)
        _write_atomic(data_dir / info.filename, content)
        written += 1
    return prefix, written, sum(info.file_size for info in pending)


def download_manifest(
    manifest: Path,
    data_dir: Path,
    competition: str = COMPETITION,
    workers: int = 4,
    retries: int = 5,
) -> None:
    prefixes = _load_manifest(manifest)
    url, zip_size = _zip_url(competition)
    print(f"remote_zip_bytes={zip_size}", flush=True)
    selected = _central_directory(url, zip_size, prefixes, retries)
    missing = [prefix for prefix, infos in selected.items() if not infos]
    if missing:
        raise RuntimeError(f"Séries ausentes no ZIP: {missing[:5]}")
    total_files = sum(len(infos) for infos in selected.values())
    total_bytes = sum(info.file_size for infos in selected.values() for info in infos)
    print(f"selected_series={len(selected)} selected_files={total_files} selected_bytes={total_bytes}", flush=True)

    completed_files = 0
    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_download_series, url, data_dir, prefix, infos, retries)
            for prefix, infos in selected.items()
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            prefix, written, bytes_written = future.result()
            completed_files += written
            completed_bytes += bytes_written
            print(
                f"series={index}/{len(futures)} files_written={completed_files}/{total_files} "
                f"bytes_written={completed_bytes} prefix={prefix}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1:
        raise ValueError("workers e retries precisam ser positivos")
    download_manifest(
        args.manifest.expanduser(),
        args.data_dir.expanduser(),
        args.competition,
        args.workers,
        args.retries,
    )


if __name__ == "__main__":
    main()
