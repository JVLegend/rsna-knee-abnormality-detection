"""#RSNA #Kaggle #Pesquisa — metadata-only audit, no inference or pixel certification."""
import argparse
from collections import Counter
import json
from pathlib import Path
from scripts.freeze_weak_validation import KEY, TARGETS, digest, group_hash, rows_by_id
from scripts.prepare_g04_preflight import V05, V05_SHA
from scripts.prepare_v04_confirmation import V03_BUILD, literal


def check_partitions(data, training, train_metadata, teacher):
    splits=data['splits']
    if set(splits)!={'train','development','confirmation','unused'} or splits['train']!=training:
        raise ValueError('Training/split inventory drift')
    if len(splits['development'])<300 or len(splits['confirmation'])<300:
        raise ValueError('Insufficient fresh split sizes')
    all_rows=sum(splits.values(),[]); ids=[r[KEY] for r in all_rows]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate split identity')
    owners={}
    for name, rows in splits.items():
        for row in rows:
            group=row['report_hash']
            if group in owners and owners[group]!=name: raise ValueError('Group overlap')
            owners[group]=name
            raw=train_metadata[row[KEY]]
            if group!=group_hash(raw['Report']): raise ValueError('Report group drift')
            values=[float(teacher[row[KEY]][t]) for t in TARGETS]
            if values!=row['labels'] or len(values)!=12 or any(not 0<=v<=1 for v in values): raise ValueError('Label drift')
    exposed={r['report_hash'] for r in data['known_exposure']}
    gold={group_hash(r['Report']) for r in train_metadata.values() if any(r[t].strip() for t in TARGETS) and r['Report'].strip()}
    for name in ['development','confirmation','unused']:
        if {r['report_hash'] for r in splits[name]} & (exposed|gold): raise ValueError('Fresh split exposed/gold')
    counts={}
    for name in ['development','confirmation']:
        counts[name]={}
        for j,t in enumerate(TARGETS):
            labels=[r['labels'][j] for r in splits[name]]
            counts[name][t]={'positive':sum(v>.5 for v in labels),'negative':sum(v<.5 for v in labels),'uncertain':sum(v==.5 for v in labels)}
    if counts!=data['class_counts'] or not data['class_gate_passed'] or data['class_gate_failures']:
        raise ValueError('Class coverage drift')
    if any(not c['positive'] or not c['negative'] for targets in counts.values() for c in targets.values()):
        raise ValueError('Class coverage failed')
    oof=data['oof']; rows=oof['assignments']; expected={r[KEY]:r['report_hash'] for r in splits['train']+splits['development']}
    if len(rows)!=len(expected) or {r[KEY]:r['report_hash'] for r in rows}!=expected:
        raise ValueError('OOF population drift')
    group_folds={}; counts_fold=Counter()
    for r in rows:
        f=r['fold']; group=r['report_hash']
        if type(f) is not int or not 0<=f<5: raise ValueError('Invalid OOF fold')
        if group in group_folds and group_folds[group]!=f: raise ValueError('OOF group overlap')
        group_folds[group]=f; counts_fold[f]+=1
    if oof['folds']!=5 or [counts_fold[f] for f in range(5)]!=oof['study_counts'] or not all(counts_fold[f] for f in range(5)):
        raise ValueError('OOF counts drift')
    if (oof['executed'] or data['confirmation_model_evaluations'] or data['development_model_evaluations']
        or data['pixel_integrity_checked'] or data['submission_eligible']): raise ValueError('Unexpected evaluation claim')
    return {'split_sizes':{k:len(v) for k,v in splits.items()},
            'split_groups':{k:len({r['report_hash'] for r in v}) for k,v in splits.items()},
            'oof_counts':[counts_fold[f] for f in range(5)]}


def assess(output):
    if output.exists(): raise FileExistsError(output)
    if digest(V05)!=V05_SHA: raise ValueError('Frozen V05 changed')
    data=json.loads(V05.read_text())
    for p,h in data['sources'].items():
        if digest(Path(p))!=h: raise ValueError('Source hash drift')
    v03=literal(V03_BUILD.read_text(),'V03')['manifest']
    expected_exposure=[{KEY:r[KEY],'report_hash':r['report_hash']} for r in v03['train']+v03['development']+v03['reserved']]
    if data['known_exposure']!=expected_exposure: raise ValueError('Exposure inventory drift')
    result=check_partitions(data,v03['train'],rows_by_id(Path('data/raw/train.csv')),rows_by_id(Path('data/external_labels/targetwise_teacher.csv')))
    result.update(status='PASSED_V05_METADATA_AUDIT_NOT_PIXEL_VALIDATION',manifest_sha256=V05_SHA,
                  pixels_checked=False,oof_executed=False,model_evaluations_performed=False,submission_eligible=False,
                  limitations=['Metadata-only checks; decoder, geometry, exact/approximate pixel duplicates remain pending.',
                               'Known own-model exposure excluded, not global historical virginity or verified patient identities.',
                               'No independent clinical labels; 0.5 weak uncertainty preserved.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    assess(p.parse_args().output)
