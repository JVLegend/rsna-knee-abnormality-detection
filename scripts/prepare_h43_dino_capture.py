"""#RSNA #Kaggle #Pesquisa — captura DINO36 para replay, sem nova receita."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.prepare_h43_parent import replace_once
from scripts.prepare_h43_e03_fullstack import BENCH_SHA


def build(raw):
    if hashlib.sha256(raw).hexdigest() != BENCH_SHA:
        raise ValueError('Audited benchmark drift')
    n = json.loads(raw)
    end = [i for i, c in enumerate(n['cells']) if c['cell_type'] == 'code' and c['source'] == "main()\nlog('done')\n"]
    if len(end) != 1:
        raise ValueError('DINO stop boundary drift')
    n['cells'] = n['cells'][:end[0] + 1]
    found = 0
    for c in n['cells']:
        if c['cell_type'] == 'code' and 'def infer_from_package(' in c['source']:
            c['source'] = replace_once(c['source'],
                '    if len(public_frontier_members) == len(members):',
                "    h43_require(all(m['ids'] == public_frontier_members[0]['ids'] for m in public_frontier_members), 'capture IDs')\n"
                "    np.savez_compressed('/kaggle/working/dino_replay_inputs.npz',\n"
                "        member_ids=np.asarray([m['id'] for m in public_frontier_members]),\n"
                "        study_ids=np.asarray(public_frontier_members[0]['ids']),\n"
                "        predictions=np.stack([m['pred'] for m in public_frontier_members]))\n"
                '    if len(public_frontier_members) == len(members):')
            found += 1
    if found != 1:
        raise ValueError('Capture boundary drift')
    final = """h43_require(H43_RECEIPT['stages'] == ['dino'], 'DINO-only capture')
_p = Path('/kaggle/working/dino_replay_inputs.npz')
Path('/kaggle/working/dino_capture_receipt.json').write_text(json.dumps({
    'status': 'DINO_CAPTURED_NOT_SUBMISSION', 'sha256': h43_sha(_p),
    'purpose': 'replay_completion_orders_without_retraining',
    'members': 20, 'studies': 36}, indent=2))
print('DINO capture complete; no AUC, never submit this notebook')
"""
    n['cells'].append({'cell_type': 'code', 'source': final, 'metadata': {}, 'outputs': [], 'execution_count': None})
    for c in n['cells']:
        if c['cell_type'] == 'code': ast.parse(c['source'])
    n['metadata']['dino_capture'] = {'parent_benchmark_sha256': BENCH_SHA,
        'purpose': 'diagnostic_only_no_submission', 'no_prediction_recipe_change': True}
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    n = build(Path('reports/avance_av003_build/h43_benchmark36_timed_v2.ipynb').read_bytes())
    body = json.dumps(n, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body:
        raise ValueError('Refuse different overwrite')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f: f.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
