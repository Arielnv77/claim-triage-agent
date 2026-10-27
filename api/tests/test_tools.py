"""Unit tests for the deterministic lookup tools. No LLM calls."""

import pytest

from app.tools import check_coverage, estimate_repair_cost


def test_two_areas_double_the_range():
    one = estimate_repair_cost("water_damage", "moderate", ["kitchen"])
    two = estimate_repair_cost("water_damage", "moderate", ["kitchen", "bathroom"])
    assert two["min"] == 2 * one["min"]
    assert two["max"] == 2 * one["max"]
    assert two["currency"] == "USD"


def test_duplicate_areas_count_once():
    one = estimate_repair_cost("fire_damage", "severe", ["garage"])
    duplicated = estimate_repair_cost("fire_damage", "severe", ["garage", "garage"])
    assert duplicated == one


def test_empty_area_list_counts_as_one():
    empty = estimate_repair_cost("storm_damage", "minor", [])
    one = estimate_repair_cost("storm_damage", "minor", ["roof"])
    assert empty == one


def test_unknown_severity_raises():
    with pytest.raises(KeyError):
        estimate_repair_cost("water_damage", "apocalyptic", ["kitchen"])


def test_basic_policy_does_not_cover_water_damage():
    result = check_coverage("basic_homeowner", "water_damage")
    assert result["covered"] is False
    # The payload still carries the policy terms so the agent can explain the denial
    assert result["deductible"] > 0
    assert result["max_payout"] > 0


def test_standard_policy_covers_water_damage():
    assert check_coverage("standard_homeowner", "water_damage")["covered"] is True
