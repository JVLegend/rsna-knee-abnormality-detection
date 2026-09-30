"""#RSNA #Kaggle #Testes — exact-source recovery launcher, no network."""
from datetime import timedelta
from types import SimpleNamespace

import pytest

from scripts.launch_h46_exact_source import (
    FAILED_SUBMISSION_REFS,
    MINIMUM_GPU_SECONDS,
    SESSION_TIMEOUT_SECONDS,
    SLUG,
    SOURCE_SHA256,
    quota_record,
    seconds,
    validate_source,
)


def test_exact_public_source_is_pinned_and_unmodified():
    text, metadata, digest = validate_source()
    assert digest == SOURCE_SHA256
    assert hashlib_sha256(text.encode()) == SOURCE_SHA256
    assert metadata['id'] == 'maverickss26/rsna-knee-restructured-version-3'
    assert SLUG == 'jvlegend/rsna-knee-h46-exact-public-source'
    assert SESSION_TIMEOUT_SECONDS == 43200
    assert FAILED_SUBMISSION_REFS == [56640374, 56652369, 56662611]


def hashlib_sha256(payload):
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def test_quota_gate_refuses_negative_or_short_budget():
    def api(allowed, used, reserved=0):
        quota = SimpleNamespace(
            gpu_quota=SimpleNamespace(total_time_allowed=f'{allowed}s',
                                      time_used=f'{used}s', time_reserved=f'{reserved}s'),
            quota_refresh_time='2026-10-03T00:00:00Z')
        return SimpleNamespace(quota_view=lambda: quota)
    assert quota_record(api(21600, 25650))['eligible'] is False
    assert quota_record(api(43200, 21601))['eligible'] is False
    assert quota_record(api(43200, 21600))['eligible'] is True
    assert MINIMUM_GPU_SECONDS == 21600


def test_duration_parser_matches_kaggle_sdk_rendering():
    assert seconds('3055.185440.0s') == pytest.approx(3055.18544)
    assert seconds('21600s') == 21600
    assert seconds(timedelta(hours=6)) == 21600
    with pytest.raises(ValueError):
        seconds('unknown')
