"""Deterministic lookup tools: repair cost estimate and policy coverage."""

import json
from pathlib import Path

# Reference tables mounted into the container from ./tools
TOOLS_DIR = Path("/app/tools")


def _load_table(filename: str, key: str) -> dict:
    """Read a reference table and return the given top-level key."""
    with open(TOOLS_DIR / filename, encoding="utf-8") as f:
        return json.load(f)[key]


# Cost matrix from the JSON, renaming min_per_area/max_per_area to min/max
# so the shape stays {damage_type: {severity: {"min": ..., "max": ...}}}
COSTS = {
    damage_type: {
        severity: {"min": rates["min_per_area"], "max": rates["max_per_area"]}
        for severity, rates in severities.items()
    }
    for damage_type, severities in _load_table("cost-estimator.json", "cost_matrix").items()
}

def estimate_repair_cost(damage_type, severity, affected_areas):
    """Return the repair cost range for a damage type, severity and set of areas."""
    prices = COSTS[damage_type][severity]
    n_areas = max(1, len(set(affected_areas)))
    return {"min": prices["min"] * n_areas, "max": prices["max"] * n_areas, "currency": "USD"}
    

# Policy table from the JSON, already in the shape check_coverage expects
POLICIES = _load_table("coverage-table.json", "policies")

def check_coverage(policy_type, incident_category):
    """Return the policy terms and whether it covers the incident category."""
    policy = POLICIES[policy_type]
    covered = incident_category in policy["covered_categories"]
    return {"covered": covered, "deductible": policy["deductible"], "max_payout": policy["max_payout"], "notes": policy["notes"]}