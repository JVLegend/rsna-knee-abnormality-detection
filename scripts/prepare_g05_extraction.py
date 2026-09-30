"""#RSNA #Kaggle #Pesquisa — pinned paired extraction builder; no dispatch/training.

Same frozen split/recipe. Technical additions are frozen before any quality
evaluation. Pack the large immutable geometry contract, not arbitrary code.
"""
import argparse
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path

from scripts.prepare_g04_preflight import sampling_identity
from scripts.prepare_v04_confirmation import V03_BUILD, V03_SHA, literal
from scripts.resolution_comparison import (contract_hash, digest, identity_gate,
                                         load_manifest, RECIPE, V05_SHA)
from scripts.g05_pair_runtime import checked_header_evidence
from scripts.g05_duplicate_audit import DUPLICATE_RECIPE


def geometry_contract():
    if digest(V03_BUILD) != V03_SHA:
        raise ValueError('Pinned V03 builder drift')
    base = literal(V03_BUILD.read_text(), 'V03')
    first = Path('reports/avance_av023_g01_v1/g01_geometry.json')
    extra = Path('reports/avance_av024_v03_v1/v03_geometry.json')
    receipt = json.loads(Path('reports/avance_av024_v03_v1/v03_receipt.json').read_text())
    if digest(first) != base['geometry_sha256'] or digest(extra) != receipt['geometry_sha256']:
        raise ValueError('Frozen geometry hashes changed')
    expected = {(r['study'], r['series']): {'study': r['study'], 'series': r['series'],
                'selected': sampling_identity(r['arms']['physical_adjacent']),
                'pixel_sha256': r['arms']['physical_adjacent']['pixel_sha256']}
                for r in json.loads(first.read_text())['series']}
    expected.update({(r['study'], r['series']): {'study': r['study'], 'series': r['series'],
                    'selected': sampling_identity(r['selected']), 'pixel_sha256': r['pixel_sha256']}
                    for r in json.loads(extra.read_text())})
    manifest = load_manifest()
    return [expected[(r['StudyInstanceUID'], s['series_uid'])] for r in manifest['splits']['train'] for s in r['series']]


def check_local_headers(path, rows):
    # Runtime sha helper is intentionally injected, never replaced with an
    # unverified header JSON boolean.
    import scripts.g05_pair_runtime as runtime
    runtime.sha = digest
    return checked_header_evidence(path, digest(path), {'rows': rows, 'v05_sha256': V05_SHA})


def assemble(stage, headers, identity=None, pixels=None):
    manifest = load_manifest()
    protocol = json.loads(Path('reports/avance_av036_g05/protocol_v2.json').read_text())
    if protocol['recipe'] != RECIPE or contract_hash({k: v for k, v in protocol.items() if k != 'contract_hash'}) != protocol['contract_hash']:
        raise ValueError('Preregistered quality contract changed')
    rows = [dict(StudyInstanceUID=r['StudyInstanceUID'], split=split, series=r['series'])
            for split in ('train', 'development', 'confirmation') for r in manifest['splits'][split]]
    check_local_headers(headers, rows)
    source = V03_BUILD.read_text()
    wanted = {'sha', 'normalize_cache_compatible', 'physical_plan', 'discover_inputs'}
    definitions = [ast.get_source_segment(source, n) for n in ast.parse(source).body
                   if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(definitions) != len(wanted):
        raise ValueError('Pinned helper inventory changed')
    duplicate_source = Path('scripts/g05_duplicate_audit.py').read_text()
    runtime = Path('scripts/g05_pair_runtime.py').read_text()
    expected = geometry_contract()
    spec = {'name': 'G05_paired_extraction_v2', 'payload_encoding': 'lzma_base85_v2', 'stage': stage, 'rows': rows,
            'v05_sha256': V05_SHA, 'headers_sha256': digest(headers),
            'protocol_hash': protocol['contract_hash'], 'expected_geometry': expected,
            'model_sha256': RECIPE['model_sha256'], 'config_sha256': RECIPE['config_sha256'],
            'baseline_archive_sha256': RECIPE['baseline_archive_sha256'],
            'pilot_archive_sha256': '3175b1a08d08a80fd371e8c9c10ed1dde2fd4d93b0ed27d1215764dfe45d9948',
            'pilot_geometry_sha256': '524893ce96b2fb74ef57924a7d1b01dc791c60df1c6ee7500a2ed29a83f0ac84',
            'duplicate_recipe': DUPLICATE_RECIPE,
            'pixel_guard_seconds': 6900, 'feature_guard_seconds': 6600,
            'runtime_sha256': hashlib.sha256(runtime.encode()).hexdigest(),
            'duplicate_source_sha256': hashlib.sha256(duplicate_source.encode()).hexdigest(),
            'confirmation_evaluated': False, 'submission_eligible': False}
    if stage == 'features':
        if identity is None or pixels is None:
            raise ValueError('Verified identity and pixel receipt required before GPU build')
        evidence = json.loads(identity.read_text())
        identity_gate(manifest['splits'], evidence)
        from scripts.assess_g05_pixels import verify
        independent = verify(pixels.parent, headers)
        if not independent['paired_pixel_gate_passed']:
            raise ValueError('Independent pixel/duplicate replay did not pass')
        receipt = json.loads(pixels.read_text())
        if (receipt['status'] != 'COMPLETE_G05_PIXELS_NOT_FEATURES_OR_PATIENT_CERTIFICATION' or
                not receipt['duplicate_screen_passed'] or receipt['studies'] != 1600 or
                receipt['headers_sha256'] != digest(headers) or
                receipt['spec']['protocol_hash'] != protocol['contract_hash']):
            raise ValueError('Paired extraction preconditions not met')
        spec.update(identity_sha256=digest(identity), pixel_receipt_sha256=digest(pixels),
                    pixel_contract_hash=receipt['contract_hash'],
                    split_identity_rows={s: [{k: r[k] for k in ['StudyInstanceUID', 'series', 'report_hash']}
                                            for r in manifest['splits'][s]]
                                         for s in ['train', 'development', 'confirmation']})
    else:
        if stage != 'pixels':
            raise ValueError('Unknown extraction stage')
    spec['contract_hash'] = contract_hash(spec)
    # zlib's 32 KiB window cannot reuse far-apart geometry/identity records.
    # LZMA is stdlib and keeps both CPU and full GPU sources below 1 MB without
    # dropping provenance or changing the frozen scientific recipe.
    packed = base64.b85encode(lzma.compress(json.dumps(spec, sort_keys=True).encode(), preset=6)).decode()
    imports = "import time\nG05_CODE_STARTED = time.perf_counter()\nimport os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\nos.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\nimport base64,lzma,hashlib,json\nfrom pathlib import Path\nimport numpy as np\n"
    result = (imports+'G05_SPEC = json.loads(lzma.decompress(base64.b85decode('+repr(packed)+')))\n'
              +'CACHE_SOURCE = '+repr(literal(source, 'CACHE_SOURCE'))+'\n'
              +'\n\n'.join(definitions)+'\n'+duplicate_source+'\n')
    if stage == 'features':
        common = Path('scripts/resolution_comparison.py').read_text()
        tree = ast.parse(common)
        functions = [ast.get_source_segment(common, n) for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name in {'identity_gate', 'contract_hash'}]
        result += 'V05_SHA = '+repr(V05_SHA)+'\n'+'\n\n'.join(functions)+'\n'
    result += runtime
    ast.parse(result)
    if len(result.encode()) >= 1_000_000:
        raise ValueError('Packed source exceeds Kaggle budget')
    return result, spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['pixels', 'features'])
    p.add_argument('--headers', type=Path, required=True)
    p.add_argument('--identity', type=Path); p.add_argument('--pixels', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, spec = assemble(args.stage, args.headers, args.identity, args.pixels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(source)
    print(json.dumps({'source_sha256': digest(args.output), 'bytes': len(source.encode()),
                      'contract_hash': spec['contract_hash'], 'stage': args.stage}))
