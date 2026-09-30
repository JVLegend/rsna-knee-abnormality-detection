#!/usr/bin/env python3
"""#RSNA #Kaggle #Pesquisa — aplica gates, preservando aritmética do parent.

Fonte de terceiro permanece em reports/, não é importada/executada pelo builder.
Só aceita a cópia auditada por SHA e patches que casem exatamente uma vez.
Não publica kernel nem submete à competição.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

SOURCE_SHA = '5b133e41d951aae8dc9efa9361d1975d5259a8f98f956cc1e84d78ea1aed80e7'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Source drift: expected one occurrence of {old[:100]!r}')
    return source.replace(old, new, 1)


def partial_names(source):
    # Não alterar sample_submission.csv nem nomes diagnósticos *_submission.csv.
    for prefix in ['', '/kaggle/working/']:
        for quote in ['"', "'"]:
            source = source.replace(quote + prefix + 'submission.csv' + quote,
                                    quote + prefix + 'h43_candidate.partial.csv' + quote)
    return source


def build_notebook(raw, runtime, artifact_receipt=None):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Source notebook SHA mismatch; audit new version first')
    notebook = copy.deepcopy(json.loads(raw))
    cells = notebook['cells']
    changed = []

    def patch(index, old, new):
        source = ''.join(cells[index]['source'])
        cells[index]['source'] = replace_once(source, old, new)
        changed.append(index)

    patch(3, 'PRESET = os.environ.get("RSNA_PRESET", "probe22")', 'PRESET = "parent"')
    patch(31, "    members = man['members']", "    members = man['members']\n"
          "    h43_check_members(H43_RECEIPT['preflight']['dino_ids'], [m['id'] for m in members])")
    patch(31, '    if not per_member:',
          "    h43_check_members([m['id'] for m in members], [m['id'] for m in per_member])\n"
          "    h43_check_members([m['id'] for m in members], [m['id'] for m in public_frontier_members])\n"
          "    H43_RECEIPT.setdefault('stages', []).append('dino')\n"
          '    if not per_member:')
    patch(35, '    sub[TARGETS] = sub[TARGETS].fillna(0.5)',
          "    h43_require(sub[TARGETS].notna().all().all(), 'DINO missing study prediction')")
    patch(35, '    write_benchmark_submission()\n', '')
    patch(35, '    if pkg is not None:',
          "    h43_require(pkg is not None, 'missing DINO package; training fallback forbidden')\n"
          '    if pkg is not None:')
    patch(35, '        except Exception as public_frontier_error:\n',
          '        except Exception as public_frontier_error:\n'
          "            raise RuntimeError('H43 public-frontier promotion failed') from public_frontier_error\n")
    cells[36]['source'] = "main()\nlog('done')\n"
    changed.append(36)
    patch(40, "    assert not [k for k in missing if not k.startswith('enc.')], f'missing {missing[:5]}'",
          "    h43_require(not missing, f'A5 missing state keys: {missing[:5]}')")
    patch(40, 'CFG = cfg',
          "h43_require(len(models) == 5, 'A5 requires five folds')\nCFG = cfg")
    patch(41, "_a5_ok = np.isfinite(preds).all(axis=(0, 2))",
          "h43_require(np.isfinite(preds).all(), 'A5 incomplete predictions')\n"
          "H43_RECEIPT.setdefault('stages', []).append('a5')\n"
          '_a5_ok = np.isfinite(preds).all(axis=(0, 2))')
    cells[44]['source'] = ''.join(cells[44]['source']) + (
        "\nh43_require(globals().get('V18_CALIBRATOR_APPLIED') is True, 'calibrator not applied')\n"
        "H43_RECEIPT.setdefault('stages', []).append('rad')\n")
    changed.append(44)
    # Intervenções dentro da string de código Raptor, somente literais AST.
    source = ''.join(cells[47]['source'])
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_KE_SRC' for t in n.targets))
    raptor = ast.literal_eval(node.value)
    raptor = replace_once(raptor, '\n            except Exception as error:\n',
                           '\n            except Exception as error:\n'
                           "                raise RuntimeError('H43 Raptor study decode failed') from error\n")
    raptor = replace_once(raptor, '\n                except Exception as error:\n',
                           '\n                except Exception as error:\n'
                           "                    raise RuntimeError('H43 Raptor arm failed') from error\n")
    raptor = replace_once(raptor, '                volume, mask = build_study(study_uid, series, tsdir, reader)',
                           '                volume, mask = build_study(study_uid, series, tsdir, reader)\n'
                           "                h43_require(bool(mask.any()), 'Raptor study has no decoded slot')")
    raptor = replace_once(raptor, '    weights = np.array([float(arm.get("w", 1.0)) for arm in ARMS], dtype=np.float64)',
                           "    h43_require(len(ARMS) == 4 and np.isfinite(np.stack(arm_probs)).all(), 'Raptor composition/predictions')\n"
                           "    H43_RECEIPT.setdefault('stages', []).append('raptor')\n"
                           '    weights = np.array([float(arm.get("w", 1.0)) for arm in ARMS], dtype=np.float64)')
    raptor = replace_once(raptor, '        ranks[~np.isfinite(ranks)] = 0.5',
                           "        raise RuntimeError('H43 nonfinite Raptor ranks')")
    ast.parse(raptor)
    lines = source.splitlines(keepends=True)
    cells[47]['source'] = ''.join(lines[:node.lineno - 1]) + '_KE_SRC = ' + repr(raptor) + '\n' + ''.join(lines[node.end_lineno:])
    changed.append(47)
    patch(48, "_KE_NS = {'__name__': '_ke_raptor', 'RUN': RUN}",
          "_KE_NS = {'__name__': '_ke_raptor', 'RUN': RUN, 'h43_require': h43_require, 'H43_RECEIPT': H43_RECEIPT}")
    patch(48, 'except Exception as _coat_err:\n',
          "except Exception as _coat_err:\n    raise RuntimeError('H43 required CoAt arm failed') from _coat_err\n")
    patch(48, '    _coat_n = _coat_substitute()',
          "    _coat_n = _coat_substitute()\n    H43_RECEIPT.setdefault('stages', []).append('coat')")
    # Mantém nomes/recibos do autor como diagnósticos, mas só publica após gate próprio.
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            cell['source'] = partial_names(''.join(cell['source']))
            cell['outputs'] = []
            cell['execution_count'] = None
            ast.parse(cell['source'])
    lock = []
    if artifact_receipt is not None:
        if artifact_receipt.get('status') != 'PASSED_ARTIFACT_PREFLIGHT' or artifact_receipt.get('source_sha256') != SOURCE_SHA:
            raise ValueError('Invalid artifact receipt')
        lock = sorted({(Path(record['path']).name, record['sha256'])
                       for record in artifact_receipt['files'].values()})
        if len(lock) < 50:
            raise ValueError('Incomplete artifact receipt')
    setup = runtime + '\nH43_ARTIFACT_LOCK = ' + repr(lock) + "\nh43_preflight(gpu=True)\n"
    first_code = next(i for i, c in enumerate(cells) if c['cell_type'] == 'code')
    cells.insert(first_code, {'cell_type': 'code', 'metadata': {}, 'source': setup,
                              'outputs': [], 'execution_count': None})
    final = """# Gate próprio: não é declaração de score reproduzido.
h43_publish('/kaggle/working/h43_candidate.partial.csv', '/kaggle/working/submission.csv',
            _ke_ours['StudyInstanceUID'].astype(str).tolist(), H43_RECEIPT, RUN)
print('H43_PARENT_INTEGRITY_PASSED: ready for runtime/coverage review, not auto-submitted')
"""
    # Compare IDs com test.csv, não com o próprio candidato.
    final = final.replace("_ke_ours['StudyInstanceUID'].astype(str).tolist()",
                          "pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()")
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': final,
                  'outputs': [], 'execution_count': None})
    notebook['metadata'].setdefault('kaggle', {}).update(
        isGpuEnabled=True, isInternetEnabled=False, accelerator='nvidiaTeslaT4')
    notebook['metadata']['h43_build'] = {'source_sha256': SOURCE_SHA, 'preset': 'parent',
                                       'modified_source_cells': sorted(set(changed)),
                                       'runtime_sha256': hashlib.sha256(runtime.encode()).hexdigest(),
                                       'artifact_lock_entries': len(lock),
                                       'no_independent_oof': True}
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('reports/avance_av001_sources/maverick/rsna-knee-0941-restructured.ipynb'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact-receipt', type=Path)
    args = parser.parse_args()
    runtime = Path(__file__).with_name('h43_integrity.py').read_text()
    receipt = json.loads(args.artifact_receipt.read_text()) if args.artifact_receipt else None
    result = build_notebook(args.source.read_bytes(), runtime, receipt)
    body = json.dumps(result, ensure_ascii=False, indent=1) + '\n'
    if args.output.exists() and args.output.read_text() != body:
        raise ValueError('Output exists with different content; use a new build filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open('x') as handle:
            handle.write(body)
    print(json.dumps({'path': str(args.output), 'sha256': hashlib.sha256(body.encode()).hexdigest(),
                      **result['metadata']['h43_build']}, indent=2))


if __name__ == '__main__':
    main()
