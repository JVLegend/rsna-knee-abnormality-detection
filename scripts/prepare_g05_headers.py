"""#RSNA #Kaggle #Dados — freeze a CPU-only full-header job, no patient attestation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.resolution_comparison import contract_hash, digest, load_manifest, V05_SHA


def assemble():
    manifest = load_manifest()
    rows = [dict(StudyInstanceUID=r['StudyInstanceUID'], split=split, series=r['series'])
            for split in ('train', 'development', 'confirmation') for r in manifest['splits'][split]]
    runtime = Path('scripts/g05_header_runtime.py').read_text()
    spec = {'name': 'G05_exhaustive_headers_v1', 'rows': rows, 'v05_sha256': V05_SHA,
            'series_metadata_sha256': digest(Path('data/raw/train_series.csv')),
            'guard_seconds': 3300, 'timeout_seconds': 3600,
            'runtime_sha256': hashlib.sha256(runtime.encode()).hexdigest(),
            'confirmation_evaluated': False, 'gpu': False}
    spec['contract_hash'] = contract_hash(spec)
    source = 'HEADER_SPEC = '+repr(spec)+'\n'+runtime
    ast.parse(source)
    if len(source.encode()) >= 1_000_000:
        raise ValueError('Source budget exceeded')
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = assemble()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(source)
    print(json.dumps({'source_sha256': digest(args.output), 'bytes': len(source.encode())}))
