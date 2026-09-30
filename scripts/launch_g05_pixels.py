"""#RSNA #Kaggle #Dados — one private CPU pixel audit, never GPU/train/submit.

Requires independently assessed full headers. Unknown PatientID semantics remain
a training blocker, not a reason to claim pixel screening proves independence.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zlib

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

from scripts.launch_h46_exact_source import quota_record

SLUG = 'jvlegend/rsna-knee-g05-paired-pixels-v1'
HEADER_SLUG = 'jvlegend/rsna-knee-g05-full-headers-v1'


def source_spec(source):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == 'G05_SPEC' for t in n.targets))
    packed = ast.literal_eval(node.value.args[0].args[0].args[0])
    spec = json.loads(zlib.decompress(base64.b85decode(packed)))
    expected = hashlib.sha256(json.dumps({k: v for k, v in spec.items() if k != 'contract_hash'},
                              sort_keys=True, allow_nan=False).encode()).hexdigest()
    if (expected != spec['contract_hash'] or spec['stage'] != 'pixels' or len(spec['rows']) != 1600 or
            spec['confirmation_evaluated'] or spec['submission_eligible'] or
            {s: sum(r['split'] == s for r in spec['rows']) for s in ('train', 'development', 'confirmation')} !=
            {'train': 1000, 'development': 300, 'confirmation': 300}):
        raise ValueError('Paired CPU-only scope/source contract drift')
    return spec


def main(args):
    if args.receipt.exists():
        raise FileExistsError('Existing dispatch intent: reconcile, do not duplicate')
    source = args.source.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != args.sha256:
        raise ValueError('Frozen CPU pixel source SHA changed')
    spec = source_spec(source)
    audit_raw = args.header_audit.read_bytes()
    if hashlib.sha256(audit_raw).hexdigest() != args.header_audit_sha256:
        raise ValueError('Header audit SHA changed')
    audit = json.loads(audit_raw)
    if (not audit['full_header_technical_audit_passed'] or audit['studies'] != 1600 or
            audit['headers_sha256'] != spec['headers_sha256'] or audit['gpu_used'] or audit['confirmation_labels_scored']):
        raise ValueError('Exhaustive header precondition not passed')
    api = KaggleApi(); api.authenticate()
    header = api.kernels_status(HEADER_SLUG)
    if getattr(header.status, 'name', str(header.status)) != 'COMPLETE':
        raise ValueError('Reconcile running/failed CPU headers; never race attachment generation')
    if api.kernels_list(mine=True, search='RSNA Knee G05 Paired Pixels', page_size=100):
        raise ValueError('Existing/ambiguous paired CPU kernel: do not overwrite')
    recent = api.kernels_list(mine=True, sort_by='dateRun', page_size=5)
    if not recent or any(not k.ref.startswith('jvlegend/') for k in recent):
        raise ValueError('Authenticated ownership not verified')
    metadata = json.loads(Path('reports/avance_av035_g04/parent_metadata/kernel-metadata.json').read_text())
    request = ApiSaveKernelRequest()
    request.slug = SLUG; request.new_title = 'RSNA Knee G05 Paired Pixels v1'
    request.text = source; request.language = 'python'; request.kernel_type = 'script'
    request.is_private = True; request.enable_gpu = False; request.enable_tpu = False
    request.enable_internet = False; request.session_timeout_seconds = 7200
    request.dataset_data_sources = []; request.kernel_data_sources = [HEADER_SLUG]; request.model_data_sources = []
    request.competition_data_sources = ['rsna-knee-abnormality-detection']
    request.docker_image = metadata['docker_image']; request.docker_image_pinning_type = 'original'
    record = {'slug': SLUG, 'utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': args.sha256,
              'source_contract_hash': spec['contract_hash'], 'header_audit_sha256': args.header_audit_sha256,
              'gpu': False, 'internet': False, 'timeout_seconds': 7200, 'kernel_sources': [HEADER_SLUG],
              'quota': quota_record(api), 'state': 'PREPARED_NOT_DISPATCHED', 'no_auto_submit': True,
              'confirmation_pixel_screen_only': True, 'submission_eligible': False}
    if not args.launch:
        print(json.dumps(record, indent=2)); return
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    record['state'] = 'DISPATCH_ATTEMPTED_RECONCILE_IF_UNKNOWN'
    with args.receipt.open('x') as handle: json.dump(record, handle, indent=2)
    try:
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.save_kernel(request)
        record.update(kernel_id=response.kernel_id, version_number=response.version_number,
                      response=str(response), state='RESPONSE_RECEIVED_CHECK_STATUS')
    finally:
        with args.receipt.open('w') as handle: json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--header-audit', type=Path, required=True); p.add_argument('--header-audit-sha256', required=True)
    p.add_argument('--receipt', type=Path, required=True); p.add_argument('--launch', action='store_true')
    main(p.parse_args())
