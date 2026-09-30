"""#RSNA #Kaggle #Pesquisa — auditoria ABBA36 com flags determinísticas fixas."""
import argparse
import json
from pathlib import Path
from scripts.assess_h43_e03 import assess as assess_abba

EXPECTED = {'cudnn_benchmark': False, 'cudnn_deterministic': True,
            'matmul_tf32': False, 'cudnn_tf32': False, 'deterministic_algorithms': True,
            'cublas_workspace_config': ':4096:8', 'seed': 2026,
            'purpose': 'same_worker_36_study_abba_no_submission', 'device': 'Tesla T4'}


def check_environment(env):
    if any(type(env.get(k)) is not type(v) or env.get(k) != v for k,v in EXPECTED.items()):
        raise ValueError('Deterministic environment receipt mismatch')
    if any(not env.get(k) for k in ['torch', 'cuda', 'cudnn']):
        raise ValueError('Missing runtime versions')


def assess(directory):
    env = json.loads((directory / 'raptor_determinism_environment.json').read_text())
    check_environment(env)
    result = assess_abba(directory, studies=36, series=205)
    expected = json.loads(Path('reports/avance_av009_serial_v1/h43_benchmark_selection.json').read_text())
    observed = json.loads((directory / 'h43_benchmark_selection.json').read_text())
    if expected != observed: raise ValueError('Not the same frozen 36-study sample')
    result.update(environment=env, cross_worker_determinism_demonstrated=False,
                  autotuner_cause_confirmed=False, no_fullstack_promotion=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = assess(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
