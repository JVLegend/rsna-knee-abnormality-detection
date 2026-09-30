"""#RSNA #Kaggle #Pesquisa — par serial/prefetch com a mesma regra DINO exata."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.prepare_h43_e03_fullstack import build as build_prefetch, BENCH_SHA
from scripts.prepare_h43_parent import replace_once


def build(raw, prefetch_helper, rank_helper, mode):
    if mode not in ('serial', 'prefetch'):
        raise ValueError('Unknown execution mode')
    n = build_prefetch(raw, prefetch_helper)  # Also checks pinned benchmark SHA.
    rank_sha = hashlib.sha256(rank_helper.encode()).hexdigest()
    changed = []
    for i, cell in enumerate(n['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = cell['source']
        if 'def infer_from_package(' in source:
            anchor = '        frontier_ids, frontier_acc = _combine(public_frontier_members)'
            replacement = """        frontier_ids, frontier_acc = h43_combine_public_exact(
            public_frontier_members, H43_RECEIPT['preflight']['dino_ids'], TARGETS)
        _stable_path = Path('/kaggle/working/dino_stable_inputs.npz')
        h43_require(all(m['ids'] == public_frontier_members[0]['ids'] for m in public_frontier_members), 'stable dump ID alignment')
        np.savez_compressed(_stable_path,
            member_ids=np.asarray([m['id'] for m in public_frontier_members]),
            study_ids=np.asarray(public_frontier_members[0]['ids']),
            predictions=np.stack([m['pred'] for m in public_frontier_members]))
        Path('/kaggle/working/dino_stable_receipt.json').write_text(json.dumps({
            'status': 'DINO_STABLE_AGGREGATED_NOT_SUBMISSION',
            'rule': 'public_uniform_doubled_integer_ranks_v1',
            'mode': H43_STABLE_MODE, 'helper_sha256': H43_STABLE_RANK_SHA,
            'input_sha256': h43_sha(_stable_path), 'members': len(public_frontier_members),
            'studies': len(frontier_ids), 'targets': list(TARGETS),
            'integer_sum_sha256': __import__('hashlib').sha256(frontier_acc.tobytes()).hexdigest()}, indent=2))"""
            source = replace_once(source, anchor, replacement)
            cell['source'] = (rank_helper + '\nH43_STABLE_MODE = ' + repr(mode)
                              + '\nH43_STABLE_RANK_SHA = ' + repr(rank_sha) + '\n' + source)
            changed.append(i)
        if "_KE_NS.update(E03_MODE='prefetch'" in source:
            cell['source'] = replace_once(source, "E03_MODE='prefetch'", f'E03_MODE={mode!r}')
    if len(changed) != 1:
        raise ValueError('Expected exactly one DINO injection')
    for cell in n['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    n['metadata']['e03_fullstack'].pop('expected_reference_csv_sha256')
    n['metadata']['e03_fullstack']['only_model_execution_change'] = (
        'public DINO exact integer aggregation plus selected Raptor execution mode')
    n['metadata']['h43_stable_fullstack'] = {
        'mode': mode, 'rank_helper_sha256': rank_sha, 'baseline_sha256': BENCH_SHA,
        'dino_cell': changed[0], 'purpose': 'training36_parity_only_never_submit',
        'legacy_public_recipe_changed': True, 'score_improvement_claimed': False}
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['serial', 'prefetch'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    helper = Path('scripts/ordered_prefetch.py').read_text() + '\n' + Path('scripts/h43_e03_runtime.py').read_text()
    n = build(Path('reports/avance_av003_build/h43_benchmark36_timed_v2.ipynb').read_bytes(),
              helper, Path('scripts/h43_stable_rank.py').read_text(), args.mode)
    body = json.dumps(n, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body:
        raise ValueError('Refuse different build overwrite')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f: f.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest(),
                      **n['metadata']['h43_stable_fullstack']}, indent=2))


if __name__ == '__main__': main()
