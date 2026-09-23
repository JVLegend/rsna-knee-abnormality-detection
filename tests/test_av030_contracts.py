"""#RSNA #Kaggle #Testes — synthetic integrity tests, not clinical validation."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from rsna_knee_baseline.crossfit_contracts import nested_partitions, validate_teacher, combine_teachers
from rsna_knee_baseline.input_contracts import triplet_channels, require_checkpoint_contract, configure_dino_tail


def rows():
    return [{'StudyInstanceUID': str(i), 'report_hash': f'g{i//2}'} for i in range(30)]


def partitions():
    r = rows()
    return nested_partitions(r, [dict(x, fold=int(x['StudyInstanceUID'])//2 % 5) for x in r], [])


def teacher():
    p = partitions()[0]
    return {'format': 'visual_teacher_exposure_v1', 'checkpoint_sha256': 'a'*64,
            'exposure_complete': True, 'task_pretraining': [], **p}


def test_nested_no_id_or_group_leaks_and_reproducible():
    ps = partitions()
    all_predictions = []
    for p in ps:
        sets = [{r['report_hash'] for r in p[k]} for k in ['train', 'selection', 'prediction']]
        assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i))
        all_predictions += [r['StudyInstanceUID'] for r in p['prediction']]
    assert sorted(all_predictions) == sorted(r['StudyInstanceUID'] for r in rows())
    assert ps == partitions()


@pytest.mark.parametrize('fault', ['duplicate', 'group', 'reserved', 'missing'])
def test_bad_folds_fail(fault):
    r = rows(); a = [dict(x, fold=int(x['StudyInstanceUID'])//2 % 5) for x in r]; blocked = []
    if fault == 'duplicate': r.append(r[0])
    if fault == 'group': a[0]['fold'] = 1
    if fault == 'reserved': blocked = [dict(r[0], StudyInstanceUID='other')]
    if fault == 'missing': a.pop()
    with pytest.raises(ValueError): nested_partitions(r, a, blocked)


@pytest.mark.parametrize('role', ['train', 'selection', 'task_pretraining'])
def test_teacher_exposure_checks_groups_not_just_ids(role):
    t = teacher(); assert validate_teacher(t, t['prediction'], [])
    t[role].append(dict(t['prediction'][0], StudyInstanceUID='different_uid'))
    with pytest.raises(ValueError): validate_teacher(t, t['prediction'], [])


def test_unknown_exposure_and_incomplete_coverage_fail():
    t = teacher(); t['exposure_complete'] = False
    with pytest.raises(ValueError): validate_teacher(t, t['prediction'], [])
    t = teacher()
    with pytest.raises(ValueError): validate_teacher(t, t['prediction'][:-1], [])


def test_student_outer_groups_cannot_leak_through_teacher_training():
    t = teacher(); forbidden = [dict(t['train'][0], StudyInstanceUID='student_outer')]
    with pytest.raises(ValueError): validate_teacher(t, t['prediction'], forbidden)


def test_soft_ensemble_aligns_ids_and_rejects_duplicate_and_nan():
    t = teacher(); targets = [f't{i}' for i in range(12)]
    first = {'provenance': t, 'targets': targets,
             'probabilities': {r['StudyInstanceUID']: [.2]*12 for r in t['prediction']}}
    second = copy.deepcopy(first); second['provenance']['checkpoint_sha256'] = 'b'*64
    second['probabilities'] = {k: [.8]*12 for k in reversed(list(first['probabilities']))}
    assert all(v == [.5]*12 for v in combine_teachers([first, second], t['prediction'], targets, []).values())
    with pytest.raises(ValueError): combine_teachers([first, first], t['prediction'], targets, [])
    second['probabilities'][t['prediction'][0]['StudyInstanceUID']][0] = float('nan')
    with pytest.raises(ValueError): combine_teachers([first, second], t['prediction'], targets, [])


def test_channels_are_nonmutating_and_match_center():
    x = np.arange(2*3*4*4).reshape(2, 3, 4, 4).astype('float32')
    y = triplet_channels(x, 'center_repeated')
    for c in range(3): np.testing.assert_array_equal(y[:, c], x[:, 1])
    np.testing.assert_array_equal(triplet_channels(x, 'physical_adjacent'), x)
    y[:] = 0; assert x.sum() > 0
    with pytest.raises(ValueError): triplet_channels(x[:, :2], 'center_repeated')
    with pytest.raises(ValueError): triplet_channels(x, 'guess')


@pytest.mark.parametrize('key', ['channels', 'resolution', 'normalization', 'geometry_sha256'])
def test_checkpoint_requires_same_input_contract(key):
    a = dict(channels='physical_adjacent', resolution=224, normalization='imagenet', geometry_sha256='a'*64)
    require_checkpoint_contract(a, a)
    b = dict(a); b[key] = 'different'
    with pytest.raises(ValueError): require_checkpoint_contract(a, b)


class ToyDino(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.layernorm = nn.LayerNorm(4)

    def forward(self, x):
        for layer in self.encoder.layer: x = torch.tanh(layer(x))
        return self.layernorm(x)


def test_finetune_really_updates_tail_not_frozen_prefix():
    torch.manual_seed(2026); model = ToyDino(); model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    params, receipt = configure_dino_tail(model, 2)
    assert 0 < receipt['trainable_parameters'] < receipt['total_parameters']
    assert not model.encoder.layer[0].training and model.encoder.layer[3].training
    optimizer = torch.optim.SGD(params, lr=.1)
    loss = (model(torch.randn(8, 4)) - torch.randn(8, 4)).square().mean()
    loss.backward(); optimizer.step()
    assert torch.equal(model.encoder.layer[0].weight, before['encoder.layer.0.weight'])
    assert not torch.equal(model.encoder.layer[3].weight, before['encoder.layer.3.weight'])
    assert model.encoder.layer[0].weight.grad is None
    params, r = configure_dino_tail(model, 0); assert params == [] and r['trainable_parameters'] == 0
    with pytest.raises(ValueError): configure_dino_tail(model, 5)
