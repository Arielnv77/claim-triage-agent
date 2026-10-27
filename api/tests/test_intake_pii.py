"""Unit tests for the intake PII regexes. The model-backed flags are not exercised."""

import pytest

from app.intake import _has_pii

PII_TEXTS = [
    # Phone number
    "Water leak in the kitchen, call me at 612 345 678",
    # Email address
    "Fire damage in the garage, reach me at john.doe@example.com",
    # Payment card number
    "Storm damage, my card is 4111 1111 1111 1111",
]

CLEAN_TEXTS = [
    "storm broke 3 windows in 2 rooms",
    "A burst pipe under the kitchen sink soaked the floor and the lower cabinets.",
]


@pytest.mark.parametrize("text", PII_TEXTS)
def test_pii_is_detected(text):
    assert _has_pii(text) is True


@pytest.mark.parametrize("text", CLEAN_TEXTS)
def test_plain_damage_descriptions_are_not_flagged(text):
    assert _has_pii(text) is False
