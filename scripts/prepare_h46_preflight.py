"""#RSNA #Kaggle #Pesquisa — CPU asset audit for pinned public 0.943 source.

Does not execute/import downloaded notebook code, pickle or model weights.
Only own standard-library hashing routines run remotely. No submission output.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE = Path('reports/research_20260923/maverick-v3/rsna-knee-restructured-version-3.ipynb')
SOURCE_SHA = '7dc49666e01c46e5017d4975960b06b359e869d8fd916d1be41cb90561beb522'
PINS = [
    ('coatnet_pairfilm_manifest.json', '7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d'),
    ('coatnet_global96_top3_manifest.json', '015f09030e76f86274cadae40777b3c3cf5c19f8c835d5ad88e15db41903779e'),
]


def inspect_source(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Public source changed; new audit required')
    nb = json.loads(raw)
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    for cell in cells:
        ast.parse(cell)
    config = next(c for c in cells if '\nRUN = {' in c)
    run_node = next(n for n in ast.parse(config).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'RUN' for t in n.targets))
    run = ast.literal_eval(run_node.value)
    if run['raptor_k_eval'] != 94 or run['coatnet_w']['__default__'] != .65:
        raise ValueError('Unexpected fixed recipe')
    source = '\n'.join(cells)
    if any(pin not in source for _, pin in PINS):
        raise ValueError('Missing pinned new-family manifest')
    return {'source_sha256': SOURCE_SHA, 'recipe': run,
            'status': 'STATIC_SOURCE_PINNED_NOT_RUNTIME_VALIDATED',
            'notebook_version': 1, 'script_version_id': 351863321,
            'required_family': ['resgated_top3', 'global96_top3', 'd4_swa3'],
            'required_reduction': 'rank_of_member_probability_mean',
            'known_risks': ['Child failure can drop a family member',
                           'Missing probabilities can change aggregation',
                           'Final author receipt still describes two family members',
                           'Gold58 used in public checkpoint selection; not independent validation'],
            'submission_eligible': False}


def assemble(raw):
    spec = inspect_source(raw)
    own_runtime = Path('scripts/h43_integrity.py').read_text()
    # The old asset audit is reused only for shared inventory; its source ID
    # is explicitly overridden and numerical publication code is never called.
    extra = '''
def h46_preflight():
    started = time.monotonic()
    report = {'status': 'CHECKING', 'source_sha256': H46_SPEC['source_sha256'],
              'spec': H46_SPEC, 'families': {}, 'submission_eligible': False,
              'gpu_used': False, 'source_executed': False, 'weights_loaded': False}
    try:
        shared = h43_preflight(gpu=False)
        report['shared'] = shared
        inventory = h43_inventory('/kaggle/input')
        for name, pin in H46_PINS:
            cached = {}
            manifest = h43_select(inventory, name, pin, cached)
            payload = json.loads(manifest.read_text())
            h43_require(payload['labels'] == H43_TARGETS, 'new family label schema')
            h43_require(bool(payload.get('files')), 'empty new-family manifest')
            for relative, sha in payload['files'].items():
                path = (manifest.parent / relative).resolve()
                h43_require(path.is_relative_to(manifest.parent.resolve()), 'manifest path escape')
                h43_select([path] if path.is_file() else [], path.name, sha, cached)
            report['families'][name] = {'manifest_sha256': pin, 'files': cached,
                                       'file_count': len(payload['files'])}
        report['status'] = 'PASSED_H46_ASSET_AUDIT_NOT_INFERENCE'
    except Exception as exc:
        report['status'] = 'FAILED_H46_ASSET_AUDIT'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        Path('/kaggle/working/h46_preflight.json').write_text(json.dumps(report, indent=2)+'\\n')
        print(json.dumps({k:v for k,v in report.items() if k not in {'shared','families','spec'}}), flush=True)

h46_preflight()
'''
    source = (own_runtime + '\nH46_SPEC = ' + repr(spec) + '\nH46_PINS = ' + repr(PINS)
              + '\nH43_SOURCE_SHA = ' + repr(SOURCE_SHA) + '\n' + extra)
    ast.parse(source)
    return source, spec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, spec = assemble(SOURCE.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle: handle.write(source)
    with args.output.with_suffix('.audit.json').open('x') as handle:
        json.dump(spec, handle, indent=2)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(source.encode()).hexdigest(),
                      'bytes': len(source.encode()), 'status': spec['status']}))
