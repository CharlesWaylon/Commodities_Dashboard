"""Shared display helpers used across pages."""

import streamlit as st


def color_delta(val: float) -> str:
    """Return green/red colored markdown string for a price change."""
    if val > 0:
        return f"🟢 +{val:.2f}%"
    elif val < 0:
        return f"🔴 {val:.2f}%"
    return f"⬜ {val:.2f}%"


def delta_color(val: float) -> str:
    """Streamlit metric delta_color string."""
    return "normal" if val >= 0 else "inverse"


def format_price(price: float, unit: str) -> str:
    """Format price with appropriate decimal places."""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 100:
        return f"${price:.2f}"
    elif price >= 10:
        return f"${price:.3f}"
    else:
        return f"${price:.4f}"


def sector_emoji(sector: str) -> str:
    mapping = {
        "Energy":      "⛽",
        "Metals":      "⚙️",
        "Agriculture": "🌾",
        "Livestock":   "🐄",
    }
    return mapping.get(sector, "📦")
