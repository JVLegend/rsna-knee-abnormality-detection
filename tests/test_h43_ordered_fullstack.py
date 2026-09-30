"""#RSNA #Kaggle #Testes — integração CoAt restrita à célula e receita auditadas."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from scripts.prepare_h43_ordered_fullstack import build, child_loader, recipe
from scripts.assess_h43_ordered_fullstack import validate_ordered_run

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT/'reports/avance_av012_build/h43_deterministic36_serial_v1.ipynb'


class OrderedFullstackTests(unittest.TestCase):
    def setUp(self):
        self.audit = dict(status='PASSED_ORDERED_REPEAT', eligible_for_fullstack_test=True,
                          inputs_exact=True, environments_exact=True)
        self.helper = (ROOT/'scripts/h43_coat_order.py').read_text()

    def test_gate_fails_closed(self):
        for audit in [{}, dict(self.audit, inputs_exact=False), dict(self.audit, environments_exact=False)]:
            with self.assertRaises(ValueError): build(b'{}',self.helper,'serial',audit)
        with self.assertRaisesRegex(ValueError,'build drift'): build(b'{}',self.helper,'serial',self.audit)

    @unittest.skipUnless(BUILD.exists(),'External benchmark unavailable')
    def test_only_coat_launch_cell_changes(self):
        for mode in ['serial','prefetch']:
            raw = (BUILD.parent/f'h43_deterministic36_{mode}_v1.ipynb').read_bytes()
            old = json.loads(raw); new = build(raw,self.helper,mode,self.audit)
            self.assertEqual([i for i,(a,b) in enumerate(zip(old['cells'],new['cells'])) if a!=b],[57])
            with self.assertRaisesRegex(ValueError,'helper drift'): build(raw,self.helper+'\n',mode,self.audit)
            s = new['cells'][57]['source']
            self.assertIn('PACKED FALLBACK',s)
            # Check the complete generated child string, not only the notebook AST.
            fn = next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='_coat_substitute')
            assignment = next(n for n in fn.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='child' for t in n.targets))
            child = eval(compile(ast.Expression(assignment.value),'<child>','eval'),
                         {'envd':'/fixture/env','art':'/fixture/art','out':'/fixture/out'})
            ast.parse(child)
            self.assertIn('h43_coat_ordered_runtime.py',child)
            self.assertLess(child.index('spec.loader.exec_module(rt)'),child.index('rt.run_submission'))

    def test_loader_keeps_exact_recipe(self):
        s = child_loader(self.helper)
        ast.parse(s)
        self.assertIn("patch_coat(Path(original_rt.__file__).read_text(), 'ordered')",s)
        self.assertIn(recipe()['runtime_sha256'],s)
        self.assertIn('sys.modules[spec.name] = rt',s)

    def test_auditor_rejects_wrong_recipe_or_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d/'h43_ordered_coat_recipe.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'recipe mismatch'): validate_ordered_run(d)
            (d/'h43_ordered_coat_recipe.json').write_text(json.dumps(recipe()))
            (d/'h43_coat_ordered_runtime.py').write_text('wrong')
            with self.assertRaisesRegex(ValueError,'runtime mismatch'): validate_ordered_run(d)


if __name__ == '__main__': unittest.main()
