"""#RSNA #Kaggle #Testes — controlled grammar tests, not clinical adjudication."""
import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rsna_knee_baseline.scoped_mentions import analyze, aggregate, STATES, normalized
from rsna_knee_baseline.lexicon import LEXICON, score_report, compile_patterns
from scripts.audit_scoped_training_mentions import training_reports, validate_split, audit
from scripts.freeze_weak_validation import group_hash

CASES = [
    ('ACL tear.','ACL','present'),
    ('ACL normal.','ACL','absent'),
    ('ACL is intact.','ACL','absent'),
    ('No ACL tear.','ACL','absent'),
    ('ACL tear is not excluded.','ACL','uncertain'),
    ('Cannot rule out ACL tear.','ACL','uncertain'),
    ('No definite ACL tear.','ACL','uncertain'),
    ('ACL tear?','ACL','uncertain'),
    ('ACL.','ACL','unresolved'),
    ('No fracture. Joint effusion present.','Effusion','present'),
    ('No fracture; joint effusion present.','Effusion','present'),
    ('No fracture but joint effusion present.','Effusion','present'),
    ('No fracture\nJoint effusion present.','Effusion','present'),
    ('No fracture and joint effusion present.','Effusion','unresolved'),
    ('Joint effusion absent.','Effusion','absent'),
    ('Possible joint effusion.','Effusion','uncertain'),
    ('Joint effusion. No effusion.','Effusion','conflict'),
    ('No effusion. Possible effusion.','Effusion','uncertain'),
    ('Indication: ACL tear.\nFindings: ACL normal.','ACL','absent'),
    ('Clinical history: ACL tear.\nImpression: No effusion.','ACL','excluded_only'),
    ('Indication:\nACL tear\nFindings:\nACL intact.','ACL','absent'),
    ('History: Fracture. Comparison: Fracture.','Fracture','excluded_only'),
    ('Technique: MRI knee. ACL normal. Conclusion: ACL normal.','ACL','absent'),
    ('Ligamento cruzado anterior íntegro.','ACL','absent'),
    ('Sem derrame articular.','Effusion','absent'),
    ('Não se pode excluir fratura.','Fracture','uncertain'),
    ('Menisco medial conservado.','Medial Meniscus','absent'),
    ('Sin derrame.','Effusion','absent'),
    ('Fractura no se puede excluir.','Fracture','uncertain'),
    ('Ligament croisé antérieur intact.','ACL','absent'),
    ('Sans épanchement.','Effusion','absent'),
    ('MCL intakt.','MCL','absent'),
    ('Geen effusion.','Effusion','absent'),
    ('Medial compartment.','Medial OA','unresolved'),
    ('Medial compartment cartilage loss.','Medial OA','present'),
    ('PF compartment imaging.','PF OA','not_mentioned'),
    ('Normal tendon.','Fracture','not_mentioned'),
    ('','ACL','not_mentioned'),
    ('No abnormality of another structure; ACL tear.','ACL','present'),
    ('Not only ACL tear.','ACL','unresolved'),
]


class ScopedMentionTests(unittest.TestCase):
    def test_controlled_assertion_cases(self):
        for text,target,state in CASES:
            with self.subTest(text=text,target=target):self.assertEqual(analyze(text)[target]['state'],state)

    def test_legacy_failure_reproductions(self):
        for text,target,old,new in [('ACL normal.','ACL',1,'absent'),
            ('No fracture. Joint effusion present.','Effusion',-1,'present')]:
            self.assertEqual(score_report(text,compile_patterns(LEXICON[target])),old)
            self.assertEqual(analyze(text)[target]['state'],new)

    def test_evidence_offsets_and_missing_not_negative(self):
        text='Indicação: fratura.\nAchados: Sem derrame articular.'
        result=analyze(text)
        self.assertEqual(set(result),set(LEXICON))
        for row in result.values():
            self.assertIn(row['state'],STATES)
            for e in row['evidence']:
                a,b=e['normalized_span'];self.assertEqual(normalized(text)[a:b],e['matched_term'])
        self.assertEqual(result['Fracture']['state'],'excluded_only')
        self.assertEqual(result['ACL']['state'],'not_mentioned')
        self.assertEqual(aggregate(['present','absent'],0),'conflict')

    def test_train_only_and_hash_gold_coverage_checks(self):
        text='ACL normal.';selected={'train-id':{'report_hash':group_hash(text)}}
        base={'StudyInstanceUID':'train-id','Report':text,**dict.fromkeys(LEXICON,'')}
        outside=dict(base,StudyInstanceUID='reserved-id',Report='Do not analyze',ACL='1')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'train.csv'
            def write(rows):
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(base));w.writeheader();w.writerows(rows)
            write([outside,base]);self.assertEqual(list(training_reports(p,selected)),[('train-id',text)])
            for rows in [[outside],[base,base],[dict(base,ACL='1')],[dict(base,Report='different')]]:
                write(rows)
                with self.assertRaises(ValueError):list(training_reports(p,selected))

    def test_split_overlap_and_manifest_drift(self):
        m={'format':'rsna-weak-grouped-validation-v1','splits':{
            'train':[{'StudyInstanceUID':'a','report_hash':'ga'}],
            'development':[{'StudyInstanceUID':'b','report_hash':'gb'}],
            'confirmation':[{'StudyInstanceUID':'c','report_hash':'gc'}]}}
        self.assertEqual(set(validate_split(m)),{'a'})
        for key,value in [('StudyInstanceUID','a'),('report_hash','ga')]:
            bad=copy.deepcopy(m);bad['splits']['confirmation'][0][key]=value
            with self.assertRaises(ValueError):validate_split(bad)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'V01 hash drift'):audit(p,p)


if __name__=='__main__':unittest.main()
