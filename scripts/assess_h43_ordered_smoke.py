"""#RSNA #Kaggle #Pesquisa — contrato dos exemplos oficiais, nunca AUC ou envio."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scripts.h43_integrity import H43_TARGETS, h43_sha, h43_validate_rows
from scripts.prepare_h43_ordered_smoke import PURPOSE, SOURCE_SHA
from scripts.prepare_h43_ordered_fullstack import recipe, ORDERED_SHA
from scripts.assess_h43_deterministic_fullstack import validate_recipe
from scripts.assess_h43_coat_order import load_run, batch_contract
from scripts.assess_h43_stable_fullstack import RANK_SHA
from scripts.dino_rank_replay import aggregate


def read_rows(path, ids):
    with path.open(newline='') as f:
        reader = csv.DictReader(f); rows = list(reader)
        h43_validate_rows(rows,ids,reader.fieldnames)
    return rows


def check_receipts(receipt, smoke, ids):
    if (receipt.get('status') != 'PASSED_PARENT_INTEGRITY' or receipt.get('purpose') != PURPOSE
            or set(receipt.get('stages',[])) != {'dino','a5','rad','raptor','coat'}
            or receipt.get('studies') != len(ids)):
        raise ValueError('Incomplete official smoke integrity')
    if (smoke.get('status') != 'ORDERED_SMOKE_COMPLETE_NOT_SUBMITTED' or smoke.get('purpose') != PURPOSE
            or smoke.get('source_build_sha256') != SOURCE_SHA or smoke.get('studies') != len(ids)
            or smoke.get('submission_authorized') is not False):
        raise ValueError('Smoke provenance mismatch')
    if not str(smoke.get('official_root','')).startswith('/kaggle/input/'):
        raise ValueError('Smoke did not use official inputs')
    pre = receipt['preflight']
    if (pre['status'] != 'PASSED_ARTIFACT_PREFLIGHT' or pre['errors']
            or pre['artifact_lock_entries'] != 56 or pre['gpus'] != ['Tesla T4']*2):
        raise ValueError('Smoke preflight mismatch')


def assess(directory, sample, series_file):
    with sample.open(newline='') as f: ids = [r['StudyInstanceUID'] for r in csv.DictReader(f)]
    with series_file.open(newline='') as f: series = list(csv.DictReader(f))
    if len(ids) != 3 or len(set(ids)) != 3 or not {r['StudyInstanceUID'] for r in series} <= set(ids):
        raise ValueError('Expected only the three official visible examples')
    load = lambda name: json.loads((directory/name).read_text())
    receipt,smoke = load('h43_parent_integrity.json'),load('ordered_smoke_receipt.json')
    check_receipts(receipt,smoke,ids)
    for forbidden in ['submission.csv','benchmark_predictions.csv','h43_benchmark_selection.json']:
        if (directory/forbidden).exists(): raise ValueError('Unexpected submission/benchmark artifact')
    final = directory/'smoke_predictions.csv'; read_rows(final,ids)
    if h43_sha(final) != receipt['submission_sha256'] or h43_sha(final) != smoke['prediction_sha256']:
        raise ValueError('Smoke prediction hash mismatch')
    if h43_sha(final) != h43_sha(directory/'submission_0939_parent_exact.csv'):
        raise ValueError('Unexpected parent outer routing')
    for name in ['_raptor.csv','_coat_arm.csv','submission_public_0899.csv',
                 'submission_e10_v2.csv','submission_native_v38.csv','submission_legacy_fold_blend.csv']:
        read_rows(directory/name,ids)
    validate_recipe(directory,'prefetch')  # Protocol names identify the validated recipe origin.
    if load('h43_ordered_coat_recipe.json') != recipe() or h43_sha(directory/'h43_coat_ordered_runtime.py') != ORDERED_SHA:
        raise ValueError('Smoke CoAt recipe drift')
    if 'PACKED FALLBACK' in (directory/'h43_coat_worker.log').read_text(): raise ValueError('CoAt microbatch fallback')
    coat,_ = load_run(directory,ids,'_coat_arm.csv',receipt['preflight'],expected_series=len(series))
    for shard in coat['shards']: batch_contract(shard,'ordered',ids)
    dino = load('dino_stable_receipt.json')
    capture = directory/'dino_stable_inputs.npz'
    if (dino['members'],dino['studies'],dino['mode'],dino['helper_sha256']) != (20,len(ids),'prefetch',RANK_SHA):
        raise ValueError('DINO composition drift')
    if dino['input_sha256'] != h43_sha(capture): raise ValueError('DINO capture hash mismatch')
    with np.load(capture,allow_pickle=False) as arr:
        names,observed,pred = arr['member_ids'].tolist(),arr['study_ids'].tolist(),arr['predictions']
        if set(names) != set(receipt['preflight']['dino_ids']) or len(set(names)) != 20 or set(observed) != set(ids):
            raise ValueError('DINO member/study drift')
        if pred.shape != (20,len(ids),12) or not np.isfinite(pred).all(): raise ValueError('DINO raw coverage')
        expected = np.asarray(aggregate(pred.tolist(),list(range(20)),exact=True))
        rows = {r['StudyInstanceUID']:r for r in read_rows(directory/'submission_public_0899.csv',ids)}
        actual = np.asarray([[float(rows[u][t]) for t in H43_TARGETS] for u in observed])
        if not np.array_equal(actual,expected): raise ValueError('DINO independent replay failed')
    inputs = load('e03_fullstack_inputs.json')
    if [(r['recipe'],r['uid']) for r in inputs] != [(i,u) for i in range(3) for u in ids]:
        raise ValueError('Raptor input coverage/order')
    with np.load(directory/'e03_run0_raw.npz',allow_pickle=False) as arr:
        if arr['ids'].tolist() != ids: raise ValueError('Raptor ID drift')
        for key,shape in [('arm_probs',(4,len(ids),12)),('ranks',(len(ids),12))]:
            values = arr[key]
            if values.shape != shape or not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():
                raise ValueError('Raptor raw contract failed')
    timings = receipt['timings_seconds']
    if set(timings) != {'dino','a5','rad','raptor_coat_fusion'} or any(not math.isfinite(v) or v<=0 for v in timings.values()):
        raise ValueError('Smoke timings incomplete')
    return {'status':'PASSED_ORDERED_OFFICIAL_SMOKE','studies':len(ids),'series':len(series),
        'prediction_sha256':h43_sha(final),'timings_seconds':timings,'all_five_stages_present':True,
        'dino_independent_replay':True,'coat_independent_replay':True,'official_ids_exact':True,
        'automatic_submission_authorized':False,'auc_measured':False,
        'limitations':['Three visible examples do not measure AUC or hidden-test runtime.',
                       'Recipe-origin names retain benchmark identifiers; executed IDs are official test IDs.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--sample',type=Path,default=Path('data/raw/sample_submission.csv'))
    p.add_argument('--series',type=Path,default=Path('data/raw/test_series.csv'))
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); r=assess(args.directory,args.sample,args.series)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))


if __name__ == '__main__':main()
