import re
from pathlib import Path

from config.ecosystem_registry import DOCENT

_CALL = re.compile(r'docent\(\s*"([a-z0-9_]+)"\s*\)')


def _used_ids():
    ids = set()
    for f in [Path("app.py"), *Path("pages").glob("*.py")]:
        ids |= set(_CALL.findall(f.read_text()))
    return ids


def test_every_docent_call_has_content():
    used = _used_ids()
    assert used, "expected docent() calls on pilot pages"
    missing = used - set(DOCENT)
    assert not missing, f"docent ids without registry content: {missing}"


def test_docent_content_format():
    for pid, text in DOCENT.items():
        for tag in ("**What:**", "**Read it:**", "**Why:**"):
            assert tag in text, f"{pid} missing {tag}"
