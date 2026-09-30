"""#RSNA #Kaggle #Pesquisa — repetição em outro worker, somente após ABBA aprovado."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from scripts.prepare_h43_parent import replace_once

SOURCE_SHA = 'a8fa2aad9a4b436b72fa06b5e777701dc04729a252b2c2d58c8f9efcfb75350a'
REPEAT = """E03_MODE, E03_INPUTS, E03_RUN_INDEX = 'prefetch', [], 0
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
_repeat_start = time.monotonic()
main()
torch.cuda.synchronize()
_repeat_seconds = time.monotonic() - _repeat_start
_repeat_path = Path('/kaggle/working/e03_run0_raw.npz')
with np.load(_repeat_path, allow_pickle=False) as _data:
    h43_require(_data['arm_probs'].shape == (4, 36, 12), 'repeat arm coverage')
    h43_require(np.isfinite(_data['arm_probs']).all(), 'repeat finite probabilities')
    _repeat_ids = _data['ids'].tolist()
h43_require(len(_repeat_ids) == 36 and len(E03_INPUTS) == 108, 'repeat study coverage')
Path('/kaggle/working/raptor_repeat_receipt.json').write_text(json.dumps({
    'status': 'RAPTOR_REPEAT_COMPLETE_NOT_SUBMITTED', 'mode': E03_MODE,
    'studies': 36, 'recipe_studies': len(E03_INPUTS), 'ids': _repeat_ids,
    'seconds': _repeat_seconds, 'raw_sha256': hashlib.sha256(_repeat_path.read_bytes()).hexdigest(),
    'peak_reserved_cuda_bytes': torch.cuda.max_memory_reserved(), 'inputs': E03_INPUTS,
    'purpose': 'second_worker_repeat_no_auc_no_submission'}, indent=2))
print('RAPTOR_REPEAT_COMPLETE', _repeat_seconds, flush=True)
"""


def build(raw, audit):
    if (audit.get('status') != 'VERIFIED_LOCAL_AUDIT' or audit.get('parity') is not True
            or audit.get('eligible_for_full_stack_test') is not True):
        raise ValueError('Successful, efficient deterministic ABBA audit required')
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Audited deterministic build drift')
    n = json.loads(raw)
    final = replace_once(n['cells'][-1]['source'], 'e03_benchmark()\n', REPEAT)
    final = replace_once(final, "'purpose': 'same_worker_36_study_abba_no_submission'",
                         "'purpose': 'second_worker_36_study_prefetch_no_submission'")
    n['cells'][-1]['source'] = final
    n['cells'][0]['source'] = ('# Raptor36 deterministic cross-worker repeat\n\n'
        'One prefetch pass on the same 36 training studies. Never submit.\n')
    n['metadata']['e03']['sequence'] = ['prefetch']
    n['metadata']['raptor_determinism'] = {
        'protocol': 'second_worker_prefetch36_v1', 'parent_abba_build_sha256': SOURCE_SHA,
        'purpose': 'diagnostic_not_submission', 'cause_confirmed': False}
    for c in n['cells']:
        if c['cell_type'] == 'code': ast.parse(c['source'])
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from scripts.assess_h43_raptor_determinism import assess
    audit = assess(args.reference)
    n = build(Path('reports/avance_av010_build/raptor36_deterministic_abba_v1.ipynb').read_bytes(), audit)
    body = json.dumps(n, indent=1, ensure_ascii=False) + '\n'
    if args.output.exists() and args.output.read_text() != body: raise ValueError('Refuse different overwrite')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as f: f.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest()}, indent=2))


if __name__ == '__main__': main()
