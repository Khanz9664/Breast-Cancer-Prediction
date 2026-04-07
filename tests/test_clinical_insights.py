"""
test_clinical_insights.py – Tests for app/utils/clinical_insights.py

Covers:
  - High-severity radius_mean insight (> 18)
  - Medium-severity radius_mean insight (> 15 but <= 18)
  - High-severity concave_points_mean insight (> 0.1)
  - Medium-severity concave_points_mean insight (> 0.05)
  - High-severity area_worst insight (> 1000)
  - Medium-severity compactness_mean insight (> 0.2)
  - Default low-severity fallback when all metrics are normal
  - Return type is always a list of dicts
  - Each insight dict contains the required keys
  - Multiple simultaneous rules can fire at once
"""

import pandas as pd
from utils.clinical_insights import get_clinical_insights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(**overrides):
    """Build a single-row DataFrame with safe default values, overriding as needed."""
    defaults = {
        "radius_mean":           13.0,   # normal
        "concave_points_mean":   0.02,   # normal
        "area_worst":            500.0,  # normal
        "compactness_mean":      0.08,   # normal
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Return-Type Contracts
# ---------------------------------------------------------------------------

class TestReturnTypeContracts:

    def test_always_returns_a_list(self):
        inp = _make_input()
        result = get_clinical_insights(inp)
        assert isinstance(result, list)

    def test_list_is_never_empty(self):
        """At minimum the default 'General Morphology' insight is returned."""
        inp = _make_input()
        result = get_clinical_insights(inp)
        assert len(result) >= 1

    def test_each_insight_is_dict(self):
        inp = _make_input()
        result = get_clinical_insights(inp)
        for item in result:
            assert isinstance(item, dict)

    def test_each_insight_has_required_keys(self):
        inp = _make_input()
        result = get_clinical_insights(inp)
        required_keys = {"feature", "insight", "severity"}
        for item in result:
            assert required_keys.issubset(item.keys()), \
                f"Missing keys {required_keys - item.keys()} in {item}"

    def test_severity_values_are_valid(self):
        inp = _make_input()
        result = get_clinical_insights(inp)
        valid = {"high", "medium", "low"}
        for item in result:
            assert item["severity"] in valid


# ---------------------------------------------------------------------------
# Radius Mean Rules
# ---------------------------------------------------------------------------

class TestRadiusMeanRules:

    def test_radius_above_18_is_high_severity(self):
        inp = _make_input(radius_mean=19.0)
        result = get_clinical_insights(inp)
        radius_insights = [i for i in result if "Radius" in i["feature"]]
        assert len(radius_insights) >= 1
        assert radius_insights[0]["severity"] == "high"

    def test_radius_between_15_and_18_is_medium(self):
        inp = _make_input(radius_mean=16.0)
        result = get_clinical_insights(inp)
        radius_insights = [i for i in result if "Radius" in i["feature"]]
        assert len(radius_insights) >= 1
        assert radius_insights[0]["severity"] == "medium"

    def test_radius_below_15_generates_no_radius_insight(self):
        inp = _make_input(radius_mean=12.0)
        result = get_clinical_insights(inp)
        radius_insights = [i for i in result if "Radius" in i["feature"]]
        assert len(radius_insights) == 0

    def test_radius_exactly_18_is_not_high(self):
        """Boundary: > 18 only; exactly 18 should not be high."""
        inp = _make_input(radius_mean=18.0)
        result = get_clinical_insights(inp)
        high_radius = [i for i in result if "Radius" in i["feature"] and i["severity"] == "high"]
        assert len(high_radius) == 0


# ---------------------------------------------------------------------------
# Concave Points Mean Rules
# ---------------------------------------------------------------------------

class TestConcavePointsMeanRules:

    def test_concave_points_above_01_is_high(self):
        inp = _make_input(concave_points_mean=0.15)
        result = get_clinical_insights(inp)
        cp_insights = [i for i in result if "Concave" in i["feature"]]
        assert len(cp_insights) >= 1
        assert cp_insights[0]["severity"] == "high"

    def test_concave_points_between_005_and_01_is_medium(self):
        inp = _make_input(concave_points_mean=0.07)
        result = get_clinical_insights(inp)
        cp_insights = [i for i in result if "Concave" in i["feature"]]
        assert len(cp_insights) >= 1
        assert cp_insights[0]["severity"] == "medium"

    def test_concave_points_below_005_generates_no_insight(self):
        inp = _make_input(concave_points_mean=0.03)
        result = get_clinical_insights(inp)
        cp_insights = [i for i in result if "Concave" in i["feature"]]
        assert len(cp_insights) == 0


# ---------------------------------------------------------------------------
# Area Worst Rules
# ---------------------------------------------------------------------------

class TestAreaWorstRules:

    def test_area_worst_above_1000_is_high(self):
        inp = _make_input(area_worst=1500.0)
        result = get_clinical_insights(inp)
        area_insights = [i for i in result if "Area" in i["feature"]]
        assert len(area_insights) >= 1
        assert area_insights[0]["severity"] == "high"

    def test_area_worst_below_1000_generates_no_insight(self):
        inp = _make_input(area_worst=800.0)
        result = get_clinical_insights(inp)
        area_insights = [i for i in result if "Area" in i["feature"]]
        assert len(area_insights) == 0


# ---------------------------------------------------------------------------
# Compactness Mean Rules
# ---------------------------------------------------------------------------

class TestCompactnessMeanRules:

    def test_compactness_above_02_is_medium(self):
        inp = _make_input(compactness_mean=0.25)
        result = get_clinical_insights(inp)
        comp_insights = [i for i in result if "Compactness" in i["feature"]]
        assert len(comp_insights) >= 1
        assert comp_insights[0]["severity"] == "medium"

    def test_compactness_below_02_generates_no_insight(self):
        inp = _make_input(compactness_mean=0.10)
        result = get_clinical_insights(inp)
        comp_insights = [i for i in result if "Compactness" in i["feature"]]
        assert len(comp_insights) == 0


# ---------------------------------------------------------------------------
# Default Fallback
# ---------------------------------------------------------------------------

class TestDefaultFallback:

    def test_all_normal_returns_general_morphology(self):
        inp = _make_input()  # all default / safe values
        result = get_clinical_insights(inp)
        assert len(result) == 1
        assert result[0]["severity"] == "low"
        assert "General Morphology" in result[0]["feature"]

    def test_default_insight_contains_expected_text(self):
        inp = _make_input()
        result = get_clinical_insights(inp)
        assert "standard" in result[0]["insight"].lower() or "range" in result[0]["insight"].lower()


# ---------------------------------------------------------------------------
# Multiple-Rule Firing
# ---------------------------------------------------------------------------

class TestMultipleRules:

    def test_all_rules_can_fire_simultaneously(self):
        """When every threshold is exceeded, each rule generates its own insight."""
        inp = _make_input(
            radius_mean=20.0,
            concave_points_mean=0.15,
            area_worst=1200.0,
            compactness_mean=0.25,
        )
        result = get_clinical_insights(inp)
        # Expect 4 insights (radius, concave_points, area_worst, compactness)
        assert len(result) >= 4

    def test_no_default_returned_when_other_rules_fire(self):
        inp = _make_input(radius_mean=20.0)
        result = get_clinical_insights(inp)
        general = [i for i in result if "General Morphology" in i["feature"]]
        assert len(general) == 0, "Default fallback should NOT appear when other rules fire"
