"""#RSNA #Kaggle #Pesquisa — L01 train-only diagnostic, no teacher replacement."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rsna_knee_baseline.lexicon import LEXICON, compile_patterns, score_report
from rsna_knee_baseline.scoped_mentions import analyze, RULE_VERSION, STATES
from scripts.freeze_weak_validation import group_hash, digest, freeze

MANIFEST_SHA='365566b0f830e398e78bd36bd9118a7365c3af340f588e774c8bcc351398b6df'


def validate_split(manifest):
    if manifest['format']!='rsna-weak-grouped-validation-v1':raise ValueError('Wrong manifest')
    splits=manifest['splits']
    if set(splits)!={'train','development','confirmation'}:raise ValueError('Missing split')
    seen_ids,seen_groups=set(),set()
    for name,rows in splits.items():
        ids=[r['StudyInstanceUID'] for r in rows];groups={r['report_hash'] for r in rows}
        if len(ids)!=len(set(ids)) or seen_ids.intersection(ids) or seen_groups.intersection(groups):
            raise ValueError('Duplicate IDs or report groups across splits')
        seen_ids.update(ids);seen_groups.update(groups)
    if not splits['train']:raise ValueError('Empty training split')
    return {r['StudyInstanceUID']:r for r in splits['train']}


def training_reports(path, selected):
    seen=set()
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        for row in reader:
            uid=row['StudyInstanceUID']
            if uid not in selected:continue  # No report analysis/label scoring outside train.
            if uid in seen:raise ValueError('Duplicate selected report')
            if group_hash(row['Report'])!=selected[uid]['report_hash']:raise ValueError('Training report hash drift')
            if any(row[t].strip() for t in LEXICON):raise ValueError('Gold label leaked into training audit')
            seen.add(uid)
            yield uid,row['Report']
    if seen!=set(selected):raise ValueError('Missing training report')


def audit(manifest_path, train_path):
    if digest(manifest_path)!=MANIFEST_SHA:raise ValueError('Frozen V01 hash drift')
    manifest=json.loads(manifest_path.read_text())
    if digest(train_path)!=manifest['sources']['train']['sha256']:raise ValueError('Report source drift')
    selected=validate_split(manifest)
    patterns={t:compile_patterns(terms) for t,terms in LEXICON.items()}
    records=[]; reports={}; counters={t:Counter() for t in LEXICON}; languages=Counter(); reasons=Counter()
    for uid,report in training_reports(train_path,selected):
        reports[uid]=report
        scoped=analyze(report)
        for target in LEXICON:
            legacy=score_report(report,patterns[target]);row=scoped[target];state=row['state']
            teacher=float(selected[uid]['labels'][target])
            direction='positive' if teacher>.5 else 'negative' if teacher<.5 else 'uncertain'
            assertion={'present':1,'absent':-1}.get(state)
            c=counters[target];c['total']+=1;c['state:'+state]+=1;c['legacy:'+str(legacy)]+=1
            comparable=assertion is not None and teacher!=.5
            disagreement=comparable and ((assertion==1)!=(teacher>.5))
            c['teacher_comparable']+=int(comparable);c['teacher_disagreements']+=int(disagreement)
            c['legacy_direction_changes']+=int(assertion is not None and assertion!=legacy)
            c['legacy_signal_now_abstains']+=int(legacy!=0 and assertion is None)
            c['excluded_evidence']+=sum(e['state']=='excluded' for e in row['evidence'])
            for e in row['evidence']:
                reasons[e['reason']]+=1
                languages.update(e['cue_languages'])
            records.append({'StudyInstanceUID':uid,'target':target,'report_hash':selected[uid]['report_hash'],
                'legacy_lexicon_score':legacy,'teacher_score':teacher,'teacher_direction':direction,
                'scoped_state':state,'disagrees_with_teacher':bool(disagreement),'evidence':row['evidence']})
    candidates=[r for r in records if r['disagrees_with_teacher'] or
        (r['legacy_lexicon_score'] and r['scoped_state'] not in ('present','absent')) or
        (r['scoped_state']=='absent' and r['legacy_lexicon_score']==1)]
    # Hash sampling avoids selecting examples by how persuasive they look.
    candidates.sort(key=lambda r:hashlib.sha256((r['StudyInstanceUID']+'|'+r['target']).encode()).hexdigest())
    review=[]; answer_key=[]
    for r in candidates[:60]:
        case=hashlib.sha256((r['StudyInstanceUID']+'|'+r['target']).encode()).hexdigest()[:16]
        review.append({'case_id':case,'target':r['target'],
            'full_report':reports[r['StudyInstanceUID']],
            'report_hash':r['report_hash'],'excerpts':[{'section':s,'text':c} for s,c in
                dict.fromkeys((e['section'],e['clause']) for e in r['evidence'])],
            'review_state':None,'review_note':None})
        answer_key.append({'case_id':case,'StudyInstanceUID':r['StudyInstanceUID'],
            'legacy_lexicon_score':r['legacy_lexicon_score'],'teacher_score':r['teacher_score'],'scoped_state':r['scoped_state']})
    totals=Counter()
    for counts in counters.values():totals.update(counts)
    summary={'status':'TRAIN_ONLY_DIAGNOSTIC_COMPLETE_NOT_PROMOTED','rule_version':RULE_VERSION,
        'manifest_sha256':MANIFEST_SHA,'studies':len(selected),'groups':len({r['report_hash'] for r in selected.values()}),
        'pairs':len(records),'per_target':{k:dict(v) for k,v in counters.items()},'totals':dict(totals),
        'evidence_reason_counts':dict(reasons),'cue_language_hits_not_report_languages':dict(languages),
        'review_cases':len(review),'development_reports_analyzed':0,'confirmation_reports_analyzed':0,
        'labels_overwritten':False,'visual_model_trained':False,'auc_measured':False,
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),
            ROOT/'src/rsna_knee_baseline/scoped_mentions.py',ROOT/'src/rsna_knee_baseline/lexicon.py']},
        'limitations':['Disagreement with teacher is not evidence that the teacher is wrong.',
            'Controlled string tests measure parser behavior, not clinical label accuracy.',
            'Legacy target vocabulary misses languages/synonyms; unknown is not negative.',
            'Multiple-target clauses and unsupported structural findings abstain.',
            'Review queue is selected for discordance, not representative prevalence.',
            'Full reports are included only in the private review queue; use them rather than excerpts alone.',
            'Keep validation labels fixed; visual comparison still requires V02.']}
    return summary,records,review,answer_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('data/processed/validation_weak_v1/manifest.json'))
    p.add_argument('--train',type=Path,default=Path('data/raw/train.csv'))
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if not a.output_dir.resolve().is_relative_to((ROOT/'reports').resolve()):
        raise ValueError('Restricted report extracts must stay in ignored reports/')
    summary,records,review,key=audit(a.manifest,a.train)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,payload in [('summary.json',summary),('records.json',records),('blind_review.json',review),('review_key.json',key)]:
        freeze(a.output_dir/name,payload)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['per_target','source_sha256']},indent=2))


if __name__=='__main__':main()
