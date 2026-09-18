"""#RSNA #Kaggle #Pesquisa — missing-mask equivalence and strict routing gates."""
import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.diagnose_r02_slots import PLANES, priority, stress
from scripts.assess_v02_baseline import loss, replay


def fixture():
    # Only the local production head's class, not any runtime top-level code.
    source = Path('scripts/v02_pilot_runtime.py').read_text()
    tree = ast.parse(source)
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StudyAttention')
    scope = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<local-head>', 'exec'), scope)
    torch.manual_seed(13)
    model = scope['StudyAttention']().double().eval()
    rng = np.random.default_rng(17)
    x = rng.normal(size=(5, 3, 384))
    y = rng.uniform(size=(5, 12))
    return model, x, y


def test_masked_slots_match_production_head_and_preserve_inputs():
    model, x, y = fixture()
    before = x.copy()
    intact, results = stress(x, y, model.state_dict(), list(map(str, range(12))))
    baseline = loss(intact, y).mean()
    for i, plane in enumerate(PLANES):
        mask = torch.ones((len(x), 3), dtype=torch.bool)
        mask[:, i] = False
        with torch.no_grad():
            logits = model(torch.tensor(x), mask).numpy()
        row = next(r for r in results if r['scenario'] == 'missing_masked' and r['plane'] == plane)
        assert row['mean_soft_bce'] == pytest.approx(loss(logits, y).mean(), abs=1e-12)
        assert row['delta_vs_intact'] == pytest.approx(loss(logits, y).mean()-baseline, abs=1e-12)
        changed = x.copy(); changed[:, i] = 0
        zero = next(r for r in results if r['scenario'] == 'zero_present' and r['plane'] == plane)
        assert zero['mean_soft_bce'] == pytest.approx(loss(replay(changed, model.state_dict()), y).mean())
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize('bad', ['nan_features', 'nan_labels', 'range', 'planes', 'targets', 'empty'])
def test_invalid_inputs_blocked(bad):
    model, x, y = fixture(); targets = list(map(str, range(12)))
    if bad == 'nan_features': x[0, 0, 0] = np.nan
    if bad == 'nan_labels': y[0, 0] = np.nan
    if bad == 'range': y[0, 0] = 2
    if bad == 'planes': x = x[:, :2]
    if bad == 'targets': targets[0] = targets[1]
    if bad == 'empty': x, y = x[:0], y[:0]
    with pytest.raises(ValueError, match='Invalid'): stress(x, y, model.state_dict(), targets)


def runs(delta):
    return [{'seed': seed, 'scenarios': [
        {'scenario': mode, 'plane': plane, 'delta_vs_intact': delta}
        for mode in ['missing_masked', 'zero_present'] for plane in PLANES]}
        for seed in [2026, 42]]


def test_operational_priority_requires_both_seeds_and_strict_threshold():
    assert not priority(runs(.01))['prioritize_slot_dropout_trial']
    r = runs(.02)
    r[0]['scenarios'][0]['delta_vs_intact'] = 0
    r[1]['scenarios'][1]['delta_vs_intact'] = 0
    assert priority(r)['planes_above_in_both_seeds'] == ['Axial']
    for run in r:
        for row in run['scenarios'][:3]: row['delta_vs_intact'] = -.01
    assert not priority(r)['prioritize_slot_dropout_trial']  # Zero-present never triggers it.


@pytest.mark.parametrize('bad', ['seed', 'missing', 'duplicate', 'nan'])
def test_incomplete_stress_matrix_blocked(bad):
    r = runs(.02)
    if bad == 'seed': r[1]['seed'] = 2026
    if bad == 'missing': r[0]['scenarios'].pop()
    if bad == 'duplicate': r[0]['scenarios'][0] = r[0]['scenarios'][1].copy()
    if bad == 'nan': r[0]['scenarios'][0]['delta_vs_intact'] = float('nan')
    with pytest.raises(ValueError): priority(r)
