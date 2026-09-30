"""#RSNA #Kaggle #Testes — no network or launch in unit tests."""
import pytest
from datetime import timedelta
from scripts.launch_h46_candidate import (seconds, kernel_status_name, SLUG,
    EXPECTED_SHA, FAILED_V1_SHA, PREVIOUS_V2_SHA, PREVIOUS_V3_SHA,
    SESSION_TIMEOUT_SECONDS)
from enum import Enum


def test_duration_parser_handles_current_sdk_fraction_bug():
    assert seconds('3055.185440.0s') == pytest.approx(3055.18544)
    assert seconds('21600s') == 21600
    assert seconds('0s') == 0
    assert seconds(timedelta(days=1, seconds=21600)) == 108000
    with pytest.raises(ValueError):seconds('unknown')


def test_launch_identity_is_versioned_and_candidate_pinned():
    assert SLUG=='jvlegend/rsna-knee-h46-speedy-fixed-v1'
    assert len(EXPECTED_SHA)==64
    assert PREVIOUS_V3_SHA == EXPECTED_SHA
    assert len(PREVIOUS_V2_SHA)==64
    assert EXPECTED_SHA != PREVIOUS_V2_SHA
    assert EXPECTED_SHA != FAILED_V1_SHA
    assert SESSION_TIMEOUT_SECONDS == 43200


def test_sdk_status_enum_is_normalized():
    Status=Enum('Status',['ERROR'])
    assert kernel_status_name(type('Response',(),{'status':Status.ERROR})())=='ERROR'
