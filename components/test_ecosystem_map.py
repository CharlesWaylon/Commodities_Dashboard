from components.ecosystem_map import build_map_bands
from config.ecosystem_registry import PAGES


def test_bands_cover_all_pages():
    bands, macro_col = build_map_bands()
    assert [b["zone"] for b in bands] == ["data", "signals", "risk"]
    placed = {k for b in bands for k in b["pages"]} | set(macro_col)
    assert placed == set(PAGES)


def test_band_pages_match_zone():
    bands, macro_col = build_map_bands()
    for band in bands:
        for key in band["pages"]:
            assert PAGES[key]["zone"] == band["zone"]
    for key in macro_col:
        assert PAGES[key]["zone"] == "macro"
