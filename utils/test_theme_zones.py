"""Zone theme core tests — headless, no Streamlit runtime needed."""
import os
import pytest

from utils.theme import ZONES, theme_css, zone_plotly_layout, PLOTLY_LAYOUT, _CSS


ZONE_KEYS = ("data", "signals", "risk", "macro")


def test_zones_complete():
    assert set(ZONES) == set(ZONE_KEYS)
    for z in ZONES.values():
        for field in ("label", "accent", "bg_top", "bg_bot", "panel", "border", "glow"):
            assert field in z and z[field]


def test_theme_css_no_zone_is_legacy(monkeypatch):
    monkeypatch.setenv("ECOSYSTEM_UI_ENABLED", "true")
    assert theme_css(None) == _CSS


def test_theme_css_flag_off_is_legacy(monkeypatch):
    monkeypatch.setenv("ECOSYSTEM_UI_ENABLED", "false")
    assert theme_css("signals") == _CSS


def test_theme_css_flag_on_appends_zone(monkeypatch):
    monkeypatch.setenv("ECOSYSTEM_UI_ENABLED", "true")
    css = theme_css("signals")
    assert css.startswith(_CSS)
    assert ZONES["signals"]["accent"] in css


def test_zone_plotly_layout_flag_on(monkeypatch):
    monkeypatch.setenv("ECOSYSTEM_UI_ENABLED", "true")
    layout = zone_plotly_layout("risk")
    assert layout["paper_bgcolor"] == ZONES["risk"]["panel"]
    assert layout["plot_bgcolor"] == ZONES["risk"]["bg_top"]
    # deep copy: mutating must not leak into module default
    layout["xaxis"]["gridcolor"] = "SENTINEL"
    assert PLOTLY_LAYOUT["xaxis"]["gridcolor"] != "SENTINEL"


def test_zone_plotly_layout_flag_off(monkeypatch):
    monkeypatch.setenv("ECOSYSTEM_UI_ENABLED", "false")
    assert zone_plotly_layout("risk")["paper_bgcolor"] == PLOTLY_LAYOUT["paper_bgcolor"]
