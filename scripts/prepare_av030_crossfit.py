"""#RSNA #Kaggle #Pesquisa — inner-selection identities for frozen V05 folds.

Metadata-only plan; does not read pixels, train teachers or create pseudo-labels.
The public checkpoints are NOT teachers in this own-model experiment.
"""
import argparse
import json
from pathlib import Path
from scripts.freeze_weak_validation import digest
from scripts.prepare_g04_preflight import V05, V05_SHA
from rsna_knee_baseline.crossfit_contracts import nested_partitions


def build():
    if digest(V05) != V05_SHA: raise ValueError('V05 changed')
    m=json.loads(V05.read_text())
    rows=m['splits']['train']+m['splits']['development']
    blocked=m['splits']['confirmation']
    partitions=nested_partitions(rows,m['oof']['assignments'],blocked)
    return {'format':'AV030_nested_teacher_plan_v1','parent_v05_sha256':V05_SHA,
            'folds':partitions,'training_executed':False,'pseudo_labels_created':False,
            'confirmation_evaluated':False,'pixel_integrity_checked':False,
            'generic_pretraining_only':True,
            'inner_selection_fraction_groups':.15,'seeds':[2026,42],
            'arms':['frozen_dino_control','dino_last2_blocks'],
            'student_warning':'For a student outer fold, ALL teachers producing student-training targets must also exclude that outer evaluation group. Reusing global OOF targets across outer folds can leak.',
            'gates':['Audit V05 pixels before training','Preregister loss/endpoints/epochs and cost before launching',
                     'Neither outer prediction fold nor gold58 may choose checkpoint',
                     'Do not combine channel ablation with fine-tuning in the same causal comparison']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=build();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'sha256':digest(a.output),'folds':[{'fold':p['fold'],**{k:len(p[k]) for k in ['train','selection','prediction']}} for p in result['folds']],
                      'training_executed':False,'confirmation_evaluated':False},indent=2))
