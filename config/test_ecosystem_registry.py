"""Registry integrity tests — headless."""
from pathlib import Path

from config.ecosystem_registry import PAGES, GLOSSARY, DOCENT, safe_fact


def test_every_entry_valid():
    for key, p in PAGES.items():
        assert p["zone"] in ("data", "signals", "risk", "macro"), key
        assert Path(p["nav"]).exists(), f"{key}: nav path {p['nav']} missing"
        assert p["name"]


def test_edges_point_to_registry_keys():
    for key, p in PAGES.items():
        for edge in p.get("upstream", []) + p.get("downstream", []):
            assert edge["page"] in PAGES, f"{key} → {edge['page']} not registered"


def test_facts_never_raise(monkeypatch):
    import database.db as db

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "get_engine", boom)
    for key, p in PAGES.items():
        for edge in p.get("upstream", []) + p.get("downstream", []):
            if "fact" in edge:
                out = safe_fact(edge["fact"])
                assert isinstance(out, str)  # "—" or a real value, never an exception


def test_glossary_core_terms():
    for term in ("IC", "QAOA", "regime", "damping"):
        assert term in GLOSSARY
