"""#RSNA #Kaggle #Pesquisa — parse-only dispatch scope; no account/network calls."""
import base64
import copy
import json
import zlib

import pytest

from scripts.launch_g05_pixels import source_spec
from scripts.resolution_comparison import contract_hash


def build(spec):
    spec = copy.deepcopy(spec)
    spec['contract_hash'] = contract_hash(spec)
    packed = base64.b85encode(zlib.compress(json.dumps(spec).encode())).decode()
    return 'G05_SPEC = json.loads(zlib.decompress(base64.b85decode('+repr(packed)+')))'


def spec():
    return {'stage': 'pixels', 'rows': [{'split': s} for s, n in [('train', 1000), ('development', 300), ('confirmation', 300)] for _ in range(n)],
            'confirmation_evaluated': False, 'submission_eligible': False}


def test_packed_dispatch_scope_is_read_without_executing_source():
    value = source_spec(build(spec())+'\nraise RuntimeError("not executed")\n')
    assert value['stage'] == 'pixels' and len(value['rows']) == 1600


@pytest.mark.parametrize('fault', ['feature', 'confirmation', 'submission', 'split'])
def test_cpu_launcher_rejects_expanded_scope(fault):
    value = spec()
    if fault == 'feature': value['stage'] = 'features'
    if fault == 'confirmation': value['confirmation_evaluated'] = True
    if fault == 'submission': value['submission_eligible'] = True
    if fault == 'split': value['rows'][0]['split'] = 'development'
    with pytest.raises(ValueError): source_spec(build(value))
