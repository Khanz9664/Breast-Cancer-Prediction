"""
test_model_card.py – Tests for app/components/ui.py helper functions and
                    app/main.py utility functions that can be exercised without
                    a running Streamlit server.

Specifically tests:
  - _insight_severity_badge() returns valid HTML with correct badge class
  - _insight_severity_badge() handles unknown severity gracefully
  - _insight_block() returns a string containing the feature name
  - _insight_block() contains the insight text
  - _insight_block() wraps with the correct severity CSS class
  - _insight_block() handles edge-case severities without crash
"""

import importlib
import sys
import types
import pytest


# ---------------------------------------------------------------------------
# Isolated import of main.py without hitting st.set_page_config()
# We do this by patching the module before import if not already patched.
# The conftest.py already stubs streamlit, so we just import main directly.
# ---------------------------------------------------------------------------

# Re-import main freshly
if "main" in sys.modules:
    del sys.modules["main"]

import main as app_main


# ---------------------------------------------------------------------------
# _insight_severity_badge
# ---------------------------------------------------------------------------

class TestInsightSeverityBadge:

    def test_high_severity_contains_badge_error(self):
        html = app_main._insight_severity_badge("high")
        assert "badge-error" in html

    def test_high_severity_contains_High_text(self):
        html = app_main._insight_severity_badge("high")
        assert "High" in html

    def test_medium_severity_contains_badge_warning(self):
        html = app_main._insight_severity_badge("medium")
        assert "badge-warning" in html

    def test_medium_severity_contains_Moderate_text(self):
        html = app_main._insight_severity_badge("medium")
        assert "Moderate" in html

    def test_low_severity_contains_badge_success(self):
        html = app_main._insight_severity_badge("low")
        assert "badge-success" in html

    def test_low_severity_contains_Low_text(self):
        html = app_main._insight_severity_badge("low")
        assert "Low" in html

    def test_unknown_severity_returns_badge_neutral(self):
        html = app_main._insight_severity_badge("critical")
        assert "badge-neutral" in html

    def test_empty_string_severity_returns_badge_neutral(self):
        html = app_main._insight_severity_badge("")
        assert "badge-neutral" in html

    def test_return_type_is_string(self):
        for sev in ("high", "medium", "low", "unknown"):
            assert isinstance(app_main._insight_severity_badge(sev), str)

    def test_returns_html_span(self):
        html = app_main._insight_severity_badge("high")
        assert "<span" in html and "</span>" in html


# ---------------------------------------------------------------------------
# _insight_block
# ---------------------------------------------------------------------------

class TestInsightBlock:

    def _make_insight(self, feature="Radius Mean", insight="Test insight text.", severity="high"):
        return {"feature": feature, "insight": insight, "severity": severity}

    def test_returns_string(self):
        block = app_main._insight_block(self._make_insight())
        assert isinstance(block, str)

    def test_contains_feature_name(self):
        block = app_main._insight_block(self._make_insight(feature="Area Worst"))
        assert "Area Worst" in block

    def test_contains_insight_text(self):
        block = app_main._insight_block(self._make_insight(insight="Tumor is unusual."))
        assert "Tumor is unusual." in block

    def test_high_severity_uses_high_css_class(self):
        block = app_main._insight_block(self._make_insight(severity="high"))
        assert 'insight-row high' in block

    def test_medium_severity_uses_medium_css_class(self):
        block = app_main._insight_block(self._make_insight(severity="medium"))
        assert 'insight-row medium' in block

    def test_low_severity_uses_low_css_class(self):
        block = app_main._insight_block(self._make_insight(severity="low"))
        assert 'insight-row low' in block

    def test_unknown_severity_uses_low_as_default(self):
        insight = {"feature": "X", "insight": "text", "severity": "critical"}
        block = app_main._insight_block(insight)
        # Should default to 'low' CSS class
        assert 'insight-row low' in block

    def test_block_is_a_div(self):
        block = app_main._insight_block(self._make_insight())
        assert block.startswith("<div") and "</div>" in block

    def test_block_contains_badge(self):
        block = app_main._insight_block(self._make_insight(severity="high"))
        assert "badge" in block

    def test_missing_severity_defaults_gracefully(self):
        insight = {"feature": "Y", "insight": "something"}
        block = app_main._insight_block(insight)
        assert isinstance(block, str) and len(block) > 0
