"""#RSNA #Kaggle #Pesquisa — public OOF structural audit, no score tuning.

Array names and provenance prose do not prove checkpoint exposure. This audit
never approves a public teacher without member-level train/selection receipts.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.freeze_weak_validation import KEY, TARGETS, group_hash, rows_by_id, digest

RAD_SHA = '43bc2cf07ce924e2d02e3240c31638130c3275882caaed77e4368d39dbf465ff'


def fold_summary(ids, folds, groups, excluded=()):
    if len(ids) != len(folds) or len(set(ids)) != len(ids) or not set(ids) <= set(groups):
        raise ValueError('Invalid OOF identity/fold coverage')
    excluded = set(excluded)
    by_group = defaultdict(set); group_ids = defaultdict(list)
    for uid, fold in zip(ids, folds):
        if uid in excluded: continue
        if int(fold) != fold or int(fold) not in range(5):
            raise ValueError('Invalid five-fold assignment')
        by_group[groups[uid]].add(int(fold)); group_ids[groups[uid]].append(uid)
    crossing = [g for g, fs in by_group.items() if len(fs) > 1]
    return {'studies': len(ids)-len(set(ids)&excluded), 'report_groups': len(by_group),
            'cross_fold_report_groups': len(crossing),
            'studies_in_crossing_groups': sum(len(group_ids[g]) for g in crossing),
            'fold_counts': dict(Counter(int(f) for u,f in zip(ids,folds) if u not in excluded)),
            'interpretation': 'Report hashes are a conservative grouping proxy, not proven patient identity.'}


def audit(root, train_path):
    train = rows_by_id(train_path)
    groups = {uid: group_hash(row['Report']) for uid,row in train.items()}
    gold = {uid for uid,row in train.items() if any(row[t].strip() for t in TARGETS)}
    dino = root/'dino-oof'; rad = root/'rad-oof'/'v52_oof.csv'
    arrays = {n: np.load(dino/(n+'.npy'), allow_pickle=False) for n in
              ['train_ids','train_fold','gold_mask','valid_mask','oof_equal_prob','targets']}
    ids = arrays['train_ids'].tolist(); n = len(ids)
    if set(ids) != set(train) or arrays['targets'].tolist() != TARGETS:
        raise ValueError('DINO IDs or targets do not match competition metadata')
    if any(arrays[k].shape != (n,) for k in ['train_fold','gold_mask','valid_mask']):
        raise ValueError('DINO mask/fold shape mismatch')
    if arrays['gold_mask'].dtype != bool or arrays['valid_mask'].dtype != bool:
        raise ValueError('DINO masks must be boolean')
    if set(np.asarray(ids)[arrays['gold_mask']]) != gold:
        raise ValueError('DINO gold identity mismatch')
    preds = arrays['oof_equal_prob']
    if preds.shape != (n,12) or not np.isfinite(preds).all() or np.any((preds<0)|(preds>1)):
        raise ValueError('Invalid DINO probabilities')
    if digest(rad) != RAD_SHA:
        raise ValueError('Rad receipt differs from author provenance digest')
    with rad.open(newline='') as handle:
        reader = csv.DictReader(handle); rad_rows = list(reader)
        if reader.fieldnames != [KEY]+TARGETS+['fold','is_gold']:
            raise ValueError('Rad schema changed')
    rad_ids = [r[KEY] for r in rad_rows]
    if set(rad_ids) != set(train) or len(rad_ids) != len(train):
        raise ValueError('Rad identity mismatch')
    if {r[KEY] for r in rad_rows if int(r['is_gold'])} != gold:
        raise ValueError('Rad gold mask mismatch')
    rp = np.array([[float(r[t]) for t in TARGETS] for r in rad_rows])
    if not np.isfinite(rp).all() or np.any((rp<0)|(rp>1)):
        raise ValueError('Invalid Rad probabilities')
    summaries = {'dino': fold_summary(ids, arrays['train_fold'].tolist(), groups, gold),
                 'rad': fold_summary(rad_ids, [int(r['fold']) for r in rad_rows], groups, gold)}
    files = [train_path, rad, root/'rad-oof'/'PROVENANCE.md'] + list(dino.glob('*.npy'))
    return {'status': 'STRUCTURE_AUDITED_PROVENANCE_NOT_CONFIRMED',
            'files_sha256': {str(p): digest(p) for p in files}, 'gold_excluded': len(gold),
            'dino_valid_mask_count': int(arrays['valid_mask'].sum()), 'folds': summaries,
            'can_select_fracture_blend': False, 'auc_computed': False,
            'blockers': ['No per-member train, checkpoint-selection and task-pretraining identity receipts audited',
                         'E13 receipt not yet audited; two sources do not reproduce full deployed graph'],
            'next': 'Obtain upstream member-level exposure before treating these predictions as independent OOF.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('reports/avance_av030_sources'))
    p.add_argument('--train',type=Path,default=Path('data/raw/train.csv'))
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result=audit(a.root,a.train); a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
