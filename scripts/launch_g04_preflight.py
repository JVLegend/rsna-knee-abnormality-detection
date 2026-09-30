"""#RSNA #Kaggle #Pesquisa — launch the frozen twenty-study resolution pilot.

No competition submission. Existing jobs must be reconciled, never overwritten.
The source builder rechecks identities, archived pixels and training membership.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from requests import HTTPError

from scripts.launch_h46_exact_source import quota_record
from scripts.prepare_g04_preflight import assemble

SLUG = 'jvlegend/rsna-knee-g04-resolution-preflight'
SOURCE = Path('reports/avance_av035_g04/preflight_v2.py')
SOURCE_SHA = '37b721468b9a5273403014f9569707b32b23a023c3703baaef6e70346fb0d528'
METADATA = Path('reports/avance_av035_g04/parent_metadata/kernel-metadata.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--previous-failed-receipt', type=Path, required=True)
    parser.add_argument('--receipt', type=Path,
                        default=Path('reports/avance_av035_g04/launch_v2.json'))
    args = parser.parse_args()
    source = SOURCE.read_text()
    sha = hashlib.sha256(source.encode()).hexdigest()
    if sha != SOURCE_SHA or source != assemble():
        raise ValueError('Frozen G04 source changed')
    meta = json.loads(METADATA.read_text())
    if meta['id'] != 'jvlegend/rsna-knee-v03-scale1000' or meta['id_no'] != 134854744:
        raise ValueError('V03 environment metadata identity changed')
    api = KaggleApi()
    api.authenticate()
    state = api.kernels_status(SLUG)
    status = getattr(state.status, 'name', str(state.status).rsplit('.',1)[-1])
    previous = json.loads(args.previous_failed_receipt.read_text())
    if (status != 'ERROR' or previous.get('slug') != SLUG
            or previous.get('kernel_id') != 135300098 or previous.get('version_number') != 1
            or previous.get('source_sha256') !=
            'ca3dd37aea22cdacfc7899938d2a68f6475fc68bf55b90aa645546a857dfeb21'):
        raise ValueError('Expected reconciled failed G04 version1 before schema repair')
    quota = quota_record(api)
    if quota['remaining_gpu_seconds'] < 960:
        raise ValueError(f'Insufficient pilot quota: {quota}')
    request = ApiSaveKernelRequest()
    request.slug = SLUG
    request.new_title = 'RSNA Knee G04 Resolution Preflight'
    request.text = source
    request.language = 'python'
    request.kernel_type = 'script'
    request.is_private = True
    request.enable_gpu = True
    request.enable_tpu = False
    request.enable_internet = False
    request.session_timeout_seconds = 480
    request.machine_shape = 'NvidiaTeslaT4'
    request.dataset_data_sources = []
    request.kernel_data_sources = ['jvlegend/rsna-knee-v03-scale1000']
    request.model_data_sources = meta['model_sources']
    request.competition_data_sources = meta['competition_sources']
    request.docker_image = meta['docker_image']
    request.docker_image_pinning_type = 'original'
    record = dict(slug=SLUG, utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha, source_path=str(SOURCE), timeout_seconds=480,
                  machine_shape='NvidiaTeslaT4', quota=quota,
                  kernel_sources=list(request.kernel_data_sources),
                  state='PREPARED_NOT_DISPATCHED', submission_eligible=False)
    if not args.launch:
        print(json.dumps(record, indent=2))
        return
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    record['state'] = 'DISPATCH_ATTEMPTED_RECONCILE_IF_UNKNOWN'
    with args.receipt.open('x') as handle:
        json.dump(record, handle, indent=2)
    try:
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.save_kernel(request)
        record.update(response=str(response), kernel_id=response.kernel_id,
                      version_number=response.version_number,
                      state='RESPONSE_RECEIVED_CHECK_STATUS')
    finally:
        with args.receipt.open('w') as handle:
            json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
