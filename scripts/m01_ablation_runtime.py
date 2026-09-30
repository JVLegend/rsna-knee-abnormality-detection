"""#RSNA #Kaggle #Pesquisa — builder supplies audited baseline functions."""


def check_bag(features, mask):
    if features.ndim != 3 or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Invalid bag/mask')
    if not mask.any(dim=1).all():
        raise ValueError('Empty bag')


class MeanPool(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.classifier = nn.Linear(dim, 12)

    def forward(self, features, mask):
        check_bag(features, mask)
        pooled = features.masked_fill(~mask[..., None], 0).sum(dim=1)
        return self.classifier(pooled / mask.sum(dim=1, keepdim=True))


class TargetAttention(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 12))
        # Standard Linear initialization; its rows classify separate target pools.
        self.classifier = nn.Linear(dim, 12)

    def forward(self, features, mask):
        check_bag(features, mask)
        weights = self.score(features).masked_fill(~mask[..., None], float('-inf')).softmax(dim=1)
        pooled = torch.einsum('bpc,bpd->bcd', weights, features)
        return (pooled * self.classifier.weight[None]).sum(dim=-1) + self.classifier.bias


def load_features(path):
    if sha(path) != M01['feature_sha256']:
        raise ValueError('Feature archive drift')
    rows = V02_BASELINE['splits']['train'] + V02_BASELINE['splits']['development']
    with np.load(path, allow_pickle=False) as a:
        features = a['features']
        if (features.shape != (549, 3, 384) or features.dtype != np.float32
                or not np.isfinite(features).all()
                or str(a['contract_hash']) != M01['baseline_contract']
                or a['ids'].tolist() != [r['StudyInstanceUID'] for r in rows]
                or a['image_hashes'].tolist() != [s['image_sha256'] for r in rows for s in r['series']]
                or hashlib.sha256(features.tobytes()).hexdigest() != M01['feature_fingerprint']):
            raise ValueError('Feature identity/content drift')
    return features


def main():
    global StudyAttention
    started = time.perf_counter()
    output = Path('/kaggle/working')
    spec = V02_BASELINE['spec']
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_name(i) != 'Tesla T4' for i in range(2)):
        raise ValueError('T4x2 required; only cuda:0 used')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    def guard():
        if time.perf_counter() - started > spec['timeout_seconds'] - 150:
            raise TimeoutError('M01 budget exhausted')
    paths = sorted(Path('/kaggle/input').rglob('v02_baseline_features.npz'))
    if len(paths) != 1:
        raise ValueError('Expected exactly one baseline feature attachment')
    features = load_features(paths[0])
    contract = hashlib.sha256(json.dumps(M01, sort_keys=True).encode()).hexdigest()
    train = V02_BASELINE['splits']['train']
    dev = V02_BASELINE['splits']['development']
    y = torch.tensor([r['labels'] for r in train], dtype=torch.float32, device='cuda')
    dev_y = torch.tensor([r['labels'] for r in dev], dtype=torch.float32, device='cuda')
    x = torch.from_numpy(features[:299]).cuda()
    dev_x = torch.from_numpy(features[299:]).cuda()
    factories = {'shared': SharedAttention, 'mean': MeanPool, 'target': TargetAttention}
    results = []
    for architecture in M01['architectures']:
        StudyAttention = factories[architecture]
        directory = output / architecture
        directory.mkdir(exist_ok=False)
        for seed in spec['seeds']:
            guard()
            fp = hashlib.sha256(f'{contract}:{architecture}:{seed}'.encode()).hexdigest()
            result = train_seed(x, y, dev_x, dev_y, seed, fp, directory, guard)
            result['architecture'] = architecture
            results.append(result)
        # Preserve finished architectures if a later run fails.
        (output / 'm01_progress.json').write_text(json.dumps(results, indent=2))
    receipt = {'status': 'COMPLETE_M01_NOT_SUBMISSION', 'contract_hash': contract,
               'spec': M01, 'seeds': results, 'gpu_names': [torch.cuda.get_device_name(i) for i in range(2)],
               'versions': {'torch': str(torch.__version__), 'numpy': np.__version__},
               'seconds': time.perf_counter() - started,
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'confirmation_evaluated': False, 'submission_eligible': False}
    (output / 'm01_receipt.json').write_text(json.dumps(receipt, indent=2))
    print('COMPLETE_M01_NOT_SUBMISSION', receipt['seconds'], flush=True)


if __name__ == '__main__':
    main()
