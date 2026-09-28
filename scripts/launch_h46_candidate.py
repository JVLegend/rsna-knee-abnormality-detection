"""#RSNA #Kaggle #Pesquisa — launch the frozen H46 notebook once on T4 x2.

The script refuses source drift, an existing slug, insufficient quota or an
uncommitted launch implementation. It records intent before dispatch. It does
not submit to the competition; submission requires a completed audited output.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from requests import HTTPError
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from scripts.prepare_h46_candidate import build, SOURCE

SLUG = 'jvlegend/rsna-knee-h46-speedy-fixed-v1'
EXPECTED_SHA = '9c9924f9ec55ebf193661e5b96e7f072b988e285f5a1d2a1cfc91e52d7cf58eb'
PREVIOUS_V2_SHA = '71e22cce751a5fc1e379180a117aa73daa7933ac2f238f23d31bfa332f9f51ef'
FAILED_V1_SHA = 'd3029315a5f7beb419aeba943f3d4d0b68603078e59777c68b56c2acea196b53'
PARENT_SHA = '7dc49666e01c46e5017d4975960b06b359e869d8fd916d1be41cb90561beb522'


def seconds(value):
    if hasattr(value, 'total_seconds'):
        return float(value.total_seconds())
    text = str(value)
    match = re.fullmatch(r'(\d+(?:\.\d+)?)s', text)
    if not match:
        # Current SDK renders fractional seconds as e.g. 3055.185440.0s.
        match = re.fullmatch(r'(\d+(?:\.\d+)?)\.0s', text)
    if not match:
        raise ValueError(f'Unparseable Kaggle duration: {text!r}')
    return float(match.group(1))


def own_slug_state(api):
    try:
        return api.kernels_status(SLUG)
    except HTTPError as exc:
        if exc.response is None or exc.response.status_code != 404: raise
    except ValueError as exc:
        if "Permission 'kernels.get' was denied" not in str(exc): raise
        matches = api.kernels_list(mine=True, search='RSNA Knee H46 Speedy Fixed', page_size=100)
        if matches:
            raise ValueError(f'Ambiguous existing own kernel: {[k.ref for k in matches]}') from exc
    return None


def kernel_status_name(response):
    raw=getattr(response,'status',None)
    return getattr(raw,'name',str(raw).rsplit('.',1)[-1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--asset-receipt',type=Path,required=True)
    parser.add_argument('--launch-receipt',type=Path,required=True)
    parser.add_argument('--previous-launch-receipt',type=Path)
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    asset=json.loads(args.asset_receipt.read_text())
    expected=json.dumps(build(SOURCE.read_bytes(),asset),ensure_ascii=False,indent=1)+'\n'
    actual=args.candidate.read_text()
    digest=hashlib.sha256(actual.encode()).hexdigest()
    if actual != expected or digest != EXPECTED_SHA:
        raise ValueError('Candidate/source/receipt drift; rebuild and re-audit')
    if subprocess.run(['git','diff','--quiet','--','scripts/launch_h46_candidate.py']).returncode:
        raise ValueError('Launch implementation has uncommitted changes')
    api=KaggleApi();api.authenticate()
    existing=own_slug_state(api)
    if existing is not None:
        if kernel_status_name(existing) != 'COMPLETE' or args.previous_launch_receipt is None:
            raise ValueError(f'H46 existing state is not an authorized completed-v2 repair: {existing}')
        previous=json.loads(args.previous_launch_receipt.read_text())
        if (previous.get('slug') != SLUG or previous.get('candidate_sha256') != PREVIOUS_V2_SHA
                or previous.get('version_number') != 2 or previous.get('kernel_id') != 136214178):
            raise ValueError('Previous launch receipt does not identify completed H46 v2')
    quota=api.quota_view();gpu=quota.gpu_quota
    remaining=seconds(gpu.total_time_allowed)-seconds(gpu.time_used)-seconds(gpu.time_reserved)
    if remaining < 2400:
        raise ValueError(f'Insufficient conservative GPU budget: {remaining:.1f}s')
    meta=json.loads(SOURCE.with_name('kernel-metadata.json').read_text())
    request=ApiSaveKernelRequest();request.slug=SLUG
    request.new_title='RSNA Knee H46 Speedy Fixed v1';request.text=actual
    request.language='python';request.kernel_type='notebook';request.is_private=True
    request.enable_gpu=True;request.enable_tpu=False;request.enable_internet=False
    request.session_timeout_seconds=1200;request.machine_shape='NvidiaTeslaT4'
    request.dataset_data_sources=meta['dataset_sources']
    request.kernel_data_sources=meta['kernel_sources']
    request.model_data_sources=meta['model_sources']
    request.competition_data_sources=meta['competition_sources']
    request.docker_image=meta['docker_image'];request.docker_image_pinning_type='original'
    record={'slug':SLUG,'utc':datetime.now(timezone.utc).isoformat(),
            'candidate_sha256':digest,'parent_sha256':PARENT_SHA,
            'asset_receipt_sha256':hashlib.sha256(args.asset_receipt.read_bytes()).hexdigest(),
            'gpu':True,'machine_shape':'NvidiaTeslaT4','internet':False,
            'timeout_seconds':1200,'remaining_gpu_seconds_before':remaining,
            'quota_refresh_time':str(quota.quota_refresh_time),
            'previous_version':2 if existing is not None else None,
            'state':'PREPARED_NOT_DISPATCHED'}
    if not args.launch:
        print(json.dumps(record,indent=2));return
    args.launch_receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.launch_receipt.open('x') as handle:json.dump(record,handle,indent=2)
    record['state']='DISPATCH_ATTEMPTED_RECONCILE_IF_UNKNOWN'
    try:
        with api.build_kaggle_client() as client:
            response=client.kernels.kernels_api_client.save_kernel(request)
        record['response']=str(response);record['kernel_id']=response.kernel_id
        record['version_number']=response.version_number
        record['state']='RESPONSE_RECEIVED_CHECK_STATUS'
    finally:
        with args.launch_receipt.open('w') as handle:json.dump(record,handle,indent=2)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
