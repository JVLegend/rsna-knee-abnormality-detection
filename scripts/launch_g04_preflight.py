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
SOURCE = Path('reports/avance_av029_g04/preflight_v1.py')
SOURCE_SHA = 'ca3dd37aea22cdacfc7899938d2a68f6475fc68bf55b90aa645546a857dfeb21'
METADATA = Path('reports/avance_av035_g04/parent_metadata/kernel-metadata.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--receipt', type=Path,
                        default=Path('reports/avance_av035_g04/launch.json'))
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
    try:
        state = api.kernels_status(SLUG)
    except HTTPError as exc:
        if exc.response is None or exc.response.status_code != 404:
            raise
    else:
        raise ValueError(f'Existing G04 job: reconcile {state}')
    matches = api.kernels_list(mine=True, search='G04', page_size=100)
    # Kaggle includes an anonymous [Private Notebook] placeholder with empty
    # ref and timestamp 2010; the direct own-slug check above still requires404.
    if any(kernel.ref for kernel in matches):
        raise ValueError('Ambiguous G04 kernel search; reconcile before launch')
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
