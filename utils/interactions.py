"""Pure helpers for chart selection events and deep links (spec §F1–F2)."""
from __future__ import annotations


def selected_labels(event) -> list[str]:
    """Labels from a st.plotly_chart on_select event. Never raises."""
    try:
        return [p["label"] for p in event.selection.points if p.get("label")]
    except Exception:
        return []


def resolve_commodity_hint(query_param: str | None, session_hint: str | None,
                           names: list[str]) -> int:
    """Selectbox index for a deep link: query param wins, then session hint, else 0."""
    for candidate in (query_param, session_hint):
        if candidate in names:
            return names.index(candidate)
    return 0
