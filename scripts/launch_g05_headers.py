"""#RSNA #Kaggle #Dados — single private CPU audit; no training/GPU/submission.

Persist dispatch intent first. Any existing slug/receipt prevents a second job.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

SLUG = 'jvlegend/rsna-knee-g05-full-headers-v1'


def main(args):
    if args.receipt.exists():
        raise FileExistsError('Existing dispatch receipt: reconcile; never retry blindly')
    # Builder dependencies unavailable in the lean CLI runtime: caller freezes and
    # tests source locally; here verify its explicit SHA and parsed CPU scope.
    source = args.source.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != args.sha256:
        raise ValueError('Header source SHA changed')
    import ast
    tree = ast.parse(source)
    spec = ast.literal_eval(tree.body[0].value)
    if spec['gpu'] or spec['confirmation_evaluated'] or len(spec['rows']) != 1600:
        raise ValueError('Header-only frozen population required')
    api = KaggleApi(); api.authenticate()
    owned = api.kernels_list(mine=True, search='RSNA Knee G05 Full Headers', page_size=100)
    if owned:
        raise ValueError('Existing/ambiguous G05 audit kernel: reconcile before creation')
    # Owned search + successful recent-owned query avoids treating a permission
    # denial on an unknown new slug as proof of absence.
    recent = api.kernels_list(mine=True, sort_by='dateRun', page_size=5)
    if not recent or any(not k.ref.startswith('jvlegend/') for k in recent):
        raise ValueError('Authenticated ownership not verified')
    from scripts.launch_h46_exact_source import quota_record
    quota = quota_record(api)
    metadata = json.loads(Path('reports/avance_av035_g04/parent_metadata/kernel-metadata.json').read_text())
    request = ApiSaveKernelRequest()
    request.slug = SLUG; request.new_title = 'RSNA Knee G05 Full Headers v1'
    request.text = source; request.language = 'python'; request.kernel_type = 'script'
    request.is_private = True; request.enable_gpu = False; request.enable_tpu = False
    request.enable_internet = False; request.session_timeout_seconds = 3600
    request.dataset_data_sources = []; request.kernel_data_sources = []; request.model_data_sources = []
    request.competition_data_sources = ['rsna-knee-abnormality-detection']
    request.docker_image = metadata['docker_image']; request.docker_image_pinning_type = 'original'
    record = {'slug': SLUG, 'utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': args.sha256, 'source_contract_hash': spec['contract_hash'],
              'gpu': False, 'internet': False, 'timeout_seconds': 3600,
              'quota': quota, 'state': 'PREPARED_NOT_DISPATCHED', 'no_auto_submit': True}
    if not args.launch:
        print(json.dumps(record, indent=2)); return
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    record['state'] = 'DISPATCH_ATTEMPTED_RECONCILE_IF_UNKNOWN'
    with args.receipt.open('x') as handle:
        json.dump(record, handle, indent=2)
    try:
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.save_kernel(request)
        record.update(kernel_id=response.kernel_id, version_number=response.version_number,
                      response=str(response), state='RESPONSE_RECEIVED_CHECK_STATUS')
    finally:
        with args.receipt.open('w') as handle:
            json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--launch', action='store_true')
    main(p.parse_args())
