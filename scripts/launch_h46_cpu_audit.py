"""#RSNA #Kaggle #Pesquisa — explicitly launch own CPU hashing audit, never GPU.

No configuration file is changed. Source/dependencies are recorded privately.
Refuses an existing slug or ambiguous status. No automatic retry after save.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from requests import HTTPError
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from scripts.prepare_h46_preflight import SOURCE, SOURCE_SHA, assemble

SLUG = 'jvlegend/rsna-knee-h46-cpu-assets-v1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch', action='store_true')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    expected, _ = assemble(SOURCE.read_bytes())
    if a.source.read_text() != expected:
        raise ValueError('Generated source does not match current own builder')
    meta = json.loads(SOURCE.with_name('kernel-metadata.json').read_text())
    api = KaggleApi(); api.authenticate()
    quota = str(api.quota_view())
    try:
        state = api.kernels_status(SLUG)
    except HTTPError as exc:
        if exc.response is None or exc.response.status_code != 404: raise
        state = None
    except ValueError as exc:
        # Kaggle can answer kernels.get/403 for a NEW own slug. Do not try
        # reading a private source or another account; require authenticated
        # ownership and a separate empty search before allowing creation.
        if "Permission 'kernels.get' was denied" not in str(exc): raise
        recent = api.kernels_list(mine=True, sort_by='dateRun', page_size=3)
        matches = api.kernels_list(mine=True, search='RSNA Knee H46', page_size=100)
        if not recent or any(not k.ref.startswith('jvlegend/') for k in recent) or matches:
            raise ValueError('Ambiguous own-slug existence; manual reconciliation required') from exc
        state = None
    if state is not None:
        raise ValueError(f'Existing kernel: reconcile before launch: {state}')
    request = ApiSaveKernelRequest()
    request.slug = SLUG
    request.new_title = 'RSNA Knee H46 CPU Assets v1'
    request.text = expected
    request.language = 'python'; request.kernel_type = 'script'
    request.is_private = True; request.enable_gpu = False; request.enable_tpu = False
    request.enable_internet = False; request.session_timeout_seconds = 600
    request.dataset_data_sources = meta['dataset_sources']
    request.kernel_data_sources = meta['kernel_sources']
    request.model_data_sources = meta['model_sources']
    request.competition_data_sources = meta['competition_sources']
    request.docker_image = meta['docker_image']
    request.docker_image_pinning_type = 'original'
    record = {'slug': SLUG, 'utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashlib.sha256(expected.encode()).hexdigest(),
              'parent_sha256': SOURCE_SHA, 'source_path': str(a.source),
              'gpu': False, 'internet': False, 'timeout_seconds': 600,
              'quota_raw': quota, 'dependencies': {k:meta[k] for k in
                  ['dataset_sources','kernel_sources','model_sources','competition_sources']},
              'state': 'PREPARED_NOT_DISPATCHED'}
    if not a.launch:
        print(json.dumps(record, indent=2)); return
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    # Persist before dispatch. A crash after this point requires reconciliation.
    with a.receipt.open('x') as handle: json.dump(record, handle, indent=2)
    record['state'] = 'DISPATCH_ATTEMPTED_RECONCILE_IF_UNKNOWN'
    try:
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.save_kernel(request)
        record['response'] = str(response)
        record['state'] = 'RESPONSE_RECEIVED_CHECK_STATUS'
    finally:
        with a.receipt.open('w') as handle: json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
