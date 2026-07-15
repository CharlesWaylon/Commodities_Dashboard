from types import SimpleNamespace

from utils.interactions import selected_labels, resolve_commodity_hint


def _event(points):
    return SimpleNamespace(selection=SimpleNamespace(points=points))


def test_selected_labels_happy_path():
    ev = _event([{"label": "Energy"}, {"label": "Wheat"}])
    assert selected_labels(ev) == ["Energy", "Wheat"]


def test_selected_labels_malformed():
    assert selected_labels(None) == []
    assert selected_labels(object()) == []
    assert selected_labels(_event([{}])) == []


def test_resolve_commodity_hint_query_param_wins():
    names = ["Crude Oil", "Wheat", "Gold"]
    assert resolve_commodity_hint("Wheat", "Gold", names) == 1
    assert resolve_commodity_hint(None, "Gold", names) == 2
    assert resolve_commodity_hint("Nope", None, names) == 0
