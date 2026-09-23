"""#RSNA #Kaggle #Pesquisa — instrumentação e ordenação de lotes CoAt, sem mudar pesos."""
import ast
import hashlib

COAT_SOURCE_SHA = 'b11e58f8d7cabe9e264ac70b01e811dc6a846aa01248ace1f0e6536d0d4094d9'
WAIT_ORIGINAL = '            finished, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)'
WAIT_ORDERED = '            finished, _ = wait((min(pending, key=pending.get),), return_when=FIRST_COMPLETED)'


def patch_coat(source, mode):
    if hashlib.sha256(source.encode()).hexdigest() != COAT_SOURCE_SHA:
        raise ValueError('CoAt runtime source drift')
    if mode not in ['completion', 'ordered']: raise ValueError('Unknown CoAt mode')
    def once(old, new):
        nonlocal source
        if source.count(old) != 1: raise ValueError('CoAt patch anchor drift')
        source = source.replace(old, new, 1)
    if mode == 'ordered': once(WAIT_ORIGINAL, WAIT_ORDERED)
    once('    prepared_studies = 0\n', '    prepared_studies = 0\n    input_contracts, batch_indices = [], []\n')
    once('        indices = [item[0] for item in batch]\n',
         '        indices = [item[0] for item in batch]\n        batch_indices.append(indices)\n')
    once('                    ready.append((local_index, bag))\n', '''                    import hashlib
                    input_contracts.append({
                        'index': int(local_index), 'uid': study_uids[local_index],
                        'arrays': {name: {'sha256': hashlib.sha256(getattr(bag, name).tobytes()).hexdigest(),
                            'shape': list(getattr(bag, name).shape), 'dtype': str(getattr(bag, name).dtype)}
                            for name in ['images', 'slots', 'metadata']}})
                    ready.append((local_index, bag))
''')
    once('        "prepared_studies": int(prepared_studies),\n', '''        "prepared_studies": int(prepared_studies),
        "order_probe": {'mode': ''' + repr(mode) + ''',
            'inputs': sorted(input_contracts, key=lambda row: row['index']),
            'batches': batch_indices, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
''')
    ast.parse(source)
    return source
