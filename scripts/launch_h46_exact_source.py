"""#RSNA #Kaggle #Pesquisa — launch the pinned public H46 source unchanged.

This is the recovery path after three hidden reruns of the audited derivative
threw exceptions.  The notebook bytes are the public 0.943 source; no preflight,
release gate, cell edit, output rewrite or arithmetic patch is injected.

The launcher still enforces Kaggle's code-competition operational constraints:
T4 x2, internet disabled and a 12-hour session.  It does not submit to the
competition.  Submission is allowed only after the public run completes and
its output is audited.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest


SOURCE = Path('reports/research_20260923/maverick-v3/rsna-knee-restructured-version-3.ipynb')
METADATA = SOURCE.with_name('kernel-metadata.json')
SOURCE_SHA256 = '7dc49666e01c46e5017d4975960b06b359e869d8fd916d1be41cb90561beb522'
SLUG = 'jvlegend/rsna-knee-h46-exact-source'
TITLE = 'RSNA Knee H46 Exact Public Source'
SESSION_TIMEOUT_SECONDS = 12 * 60 * 60
MINIMUM_GPU_SECONDS = 6 * 60 * 60
FAILED_SUBMISSION_REFS = [56640374, 56652369, 56662611]


def seconds(value):
    if hasattr(value, 'total_seconds'):
        return float(value.total_seconds())
    text = str(value)
    match = re.fullmatch(r'(\d+(?:\.\d+)?)s', text)
    if not match:
        match = re.fullmatch(r'(\d+(?:\.\d+)?)\.0s', text)
    if not match:
        raise ValueError(f'Unparseable Kaggle duration: {text!r}')
    return float(match.group(1))


def validate_source():
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f'Public H46 source drift: {digest}')
    notebook = json.loads(raw)
    metadata = json.loads(METADATA.read_text())
    if metadata.get('id') != 'maverickss26/rsna-knee-restructured-version-3':
        raise ValueError('Unexpected public H46 metadata identity')
    if metadata.get('machine_shape') != 'NvidiaTeslaT4' or not metadata.get('enable_gpu'):
        raise ValueError('Unexpected public H46 accelerator metadata')
    code = '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells']
                     if cell.get('cell_type') == 'code')
    forbidden = ('PASSED_H46_SOURCE_COMPATIBLE_GATE', 'H46_RELEASE = validate_h46_release',
                 'H43_ARTIFACT_LOCK =')
    if any(token in code for token in forbidden):
        raise ValueError('Source is not the untouched public H46 notebook')
    return raw.decode(), metadata, digest


def quota_record(api):
    quota = api.quota_view()
    gpu = quota.gpu_quota
    allowed = seconds(gpu.total_time_allowed)
    used = seconds(gpu.time_used)
    reserved = seconds(gpu.time_reserved)
    remaining = allowed - used - reserved
    return {
        'allowed_gpu_seconds': allowed,
        'used_gpu_seconds': used,
        'reserved_gpu_seconds': reserved,
        'remaining_gpu_seconds': remaining,
        'quota_refresh_time': str(quota.quota_refresh_time),
        'eligible': remaining >= MINIMUM_GPU_SECONDS,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-receipt', type=Path,
                        default=Path('reports/avance_av034_h46_exact/launch.json'))
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()

    text, metadata, digest = validate_source()
    api = KaggleApi()
    api.authenticate()
    existing = api.kernels_list(mine=True, search='RSNA Knee H46 Exact Public Source',
                                page_size=100)
    if any(kernel.ref == SLUG for kernel in existing):
        raise ValueError(f'Refusing to overwrite existing exact-source kernel: {SLUG}')
    quota = quota_record(api)
    record = {
        'slug': SLUG,
        'title': TITLE,
        'utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': digest,
        'source_identity': metadata['id'],
        'source_script_version_id': 351863321,
        'notebook_cells_modified': 0,
        'internet': False,
        'gpu': True,
        'machine_shape': 'NvidiaTeslaT4',
        'timeout_seconds': SESSION_TIMEOUT_SECONDS,
        'failed_derivative_submission_refs': FAILED_SUBMISSION_REFS,
        'quota': quota,
        'state': 'READY' if quota['eligible'] else 'BLOCKED_GPU_QUOTA',
    }
    if not args.launch:
        print(json.dumps(record, indent=2))
        return
    if not quota['eligible']:
        raise ValueError('GPU quota has not reset: ' + json.dumps(quota, sort_keys=True))

    request = ApiSaveKernelRequest()
    request.slug = SLUG
    request.new_title = TITLE
    request.text = text
    request.language = 'python'
    request.kernel_type = 'notebook'
    request.is_private = True
    request.enable_gpu = True
    request.enable_tpu = False
    request.enable_internet = False
    request.session_timeout_seconds = SESSION_TIMEOUT_SECONDS
    request.machine_shape = 'NvidiaTeslaT4'
    request.dataset_data_sources = metadata['dataset_sources']
    request.kernel_data_sources = metadata['kernel_sources']
    request.model_data_sources = metadata['model_sources']
    request.competition_data_sources = metadata['competition_sources']
    request.docker_image = metadata['docker_image']
    request.docker_image_pinning_type = 'original'

    args.launch_receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.launch_receipt.open('x') as handle:
        json.dump({**record, 'state': 'DISPATCHING'}, handle, indent=2)
    try:
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.save_kernel(request)
        record.update(response=str(response), kernel_id=response.kernel_id,
                      version_number=response.version_number,
                      state='RESPONSE_RECEIVED_CHECK_STATUS')
    finally:
        with args.launch_receipt.open('w') as handle:
            json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
