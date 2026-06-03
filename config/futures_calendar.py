"""
Futures contract calendar specifications — the data behind a *proper* stitched
constant-maturity M2 series.

WHY THIS EXISTS
───────────────
The first M2 implementation hard-coded a single dated contract per commodity
(e.g. ``CLQ26.NYM``).  That is wrong for history: ``CLQ26`` (Aug 2026) is the
genuine 2nd-nearby contract *today* (front = Jul), but a year ago it was ~13
months out — an M13, not an M2.  Computing ``basis = log(front / CLQ26)`` over a
full year therefore mislabels long-dated calendar spreads as near-term carry.

The fix is a contract calendar: for any date we resolve which *listed* contract
is genuinely M1 and which is M2, then stitch a constant-2nd-nearby series from
the real adjacent contracts.  This module holds the per-commodity inputs to that
resolver (``pipeline/contract_calendar.py``).

CONTRACT MONTH CYCLES — VERIFIED 2026-06-02
───────────────────────────────────────────
Cycles below were verified against CME Group / ICE contract specifications via
web search on 2026-06-02 (see MODEL_VERIFICATION_LOG.md).  Web-confirmed
directly: all COMEX metals, NYMEX platinum/palladium, every ICE soft, CME lean
hogs, CBOT soybean meal.  The remaining CBOT grains and CME live/feeder cattle
use the long-standing standard exchange cycles (textbook references); they match
the pattern of every web-confirmed sibling on the same exchange.

SIMPLIFICATIONS (documented, tunable)
─────────────────────────────────────
• Platinum & palladium technically list 3 serial months + a quarterly cycle.  We
  use only the *quarterly* cycle (the liquid contracts) — serial months are thin
  and poorly covered on Yahoo.  This matches the carry-signal intent (nearby
  *liquid* term structure).
• ``roll_offset_bdays`` is a per-group approximation of when the front contract
  rolls off (see pipeline/contract_calendar.approx_roll_date).  Exact roll timing
  matters far less than M1/M2 being genuinely adjacent listed contracts; the
  rolling z-score in build_term_structure_features absorbs the small boundary
  discontinuity.  Offsets are tunable here without touching code.

MAINTENANCE
───────────
Unlike the old fixed-ticker table, this file does NOT need semi-annual edits —
the resolver generates dated tickers on the fly from the cycle.  Only revisit if
an exchange changes a listed-month cycle (rare).
"""

from dataclasses import dataclass


# Yahoo Finance / standard futures month codes
MONTH_CODES: dict[int, str] = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}
CODE_TO_MONTH: dict[str, int] = {v: k for k, v in MONTH_CODES.items()}


@dataclass(frozen=True)
class ContractSpec:
    """
    Everything needed to enumerate a commodity's listed contracts and build its
    dated Yahoo tickers.

    Parameters
    ----------
    root : str
        Yahoo root symbol, e.g. "CL" for WTI, "GC" for gold.
    yf_ticker : str
        The commodity's *continuous* Yahoo ticker as stored in
        ``commodities.ticker`` (e.g. "CL=F").  This is the STABLE join key to the
        DB: the ``commodities.name`` column was seeded inconsistently (some bare,
        e.g. 'Copper'; some suffixed, e.g. 'Wheat (KC HRW)'), so name matching is
        fragile.  The ``ticker`` column is ``unique=True`` and reliable, so the
        ingest/stitch resolvers join on it (falling back to name only if needed).
    exchange : str
        Yahoo exchange suffix WITHOUT the dot, e.g. "NYM", "CMX", "CBT",
        "NYB", "CME".
    months : tuple[int, ...]
        Listed delivery months as ints (1=Jan … 12=Dec), sorted ascending.
    roll_offset_bdays : int
        Business-day offset applied to the *first calendar day of the delivery
        month* to approximate the contract's roll-off date:
          positive → roll BEFORE the delivery month starts (e.g. energy, whose
                     front expires ~20th of the prior month)
          negative → roll DURING the delivery month (e.g. grains expire mid-month,
                     metals/livestock near month end)
        See pipeline/contract_calendar.approx_roll_date.
    """
    root: str
    exchange: str
    months: tuple[int, ...]
    roll_offset_bdays: int = 0
    yf_ticker: str = ""   # continuous Yahoo ticker = commodities.ticker join key


# Roll-offset group defaults (business days relative to delivery-month start).
# Approximate; tunable. Sign convention per ContractSpec docstring.
_ENERGY_ROLL    = 7     # front expires ~20th of prior month → roll ~1 wk before month start
_METALS_ROLL    = -18   # COMEX/NYMEX metals expire near end of delivery month
_GRAINS_ROLL    = -10   # CBOT grains/oilseeds expire ~14th-15th of delivery month
_SOFTS_ROLL     = -7    # ICE softs vary; mid-ish delivery month
_LIVESTOCK_ROLL = -15   # CME livestock cash-settle late in / end of delivery month


# Display name → ContractSpec.  Keys are the MODELING_COMMODITIES display names,
# but the DB join is done on ``yf_ticker`` (commodities.ticker), NOT on these keys
# — the ``commodities.name`` column was seeded inconsistently and cannot be relied
# on.  Keep the keys aligned to MODELING_COMMODITIES for readability/logging.
FUTURES_CONTRACT_SPECS: dict[str, ContractSpec] = {
    # ── ENERGY — NYMEX (all 12 months) ─────────────────────────────────────────
    "WTI Crude Oil":     ContractSpec("CL", "NYM", tuple(range(1, 13)), _ENERGY_ROLL, yf_ticker="CL=F"),
    "Brent Crude Oil":   ContractSpec("BZ", "NYM", tuple(range(1, 13)), _ENERGY_ROLL, yf_ticker="BZ=F"),
    "Natural Gas":       ContractSpec("NG", "NYM", tuple(range(1, 13)), _ENERGY_ROLL, yf_ticker="NG=F"),
    "Gasoline (RBOB)":   ContractSpec("RB", "NYM", tuple(range(1, 13)), _ENERGY_ROLL, yf_ticker="RB=F"),
    "Heating Oil":       ContractSpec("HO", "NYM", tuple(range(1, 13)), _ENERGY_ROLL, yf_ticker="HO=F"),

    # ── METALS — COMEX / NYMEX ─────────────────────────────────────────────────
    "Gold (COMEX)":   ContractSpec("GC", "CMX", (2, 4, 6, 8, 10, 12), _METALS_ROLL, yf_ticker="GC=F"),
    "Silver (COMEX)": ContractSpec("SI", "CMX", (3, 5, 7, 9, 12), _METALS_ROLL, yf_ticker="SI=F"),
    "Copper (COMEX)": ContractSpec("HG", "CMX", (3, 5, 7, 9, 12), _METALS_ROLL, yf_ticker="HG=F"),
    # Platinum/Palladium trade on NYMEX (.NYM) — the .CMX guess 404'd on Yahoo.
    "Platinum":  ContractSpec("PL", "NYM", (1, 4, 7, 10), _METALS_ROLL, yf_ticker="PL=F"),
    "Palladium": ContractSpec("PA", "NYM", (3, 6, 9, 12), _METALS_ROLL, yf_ticker="PA=F"),

    # ── AGRICULTURE — CBOT ─────────────────────────────────────────────────────
    "Corn (CBOT)":       ContractSpec("ZC", "CBT", (3, 5, 7, 9, 12), _GRAINS_ROLL, yf_ticker="ZC=F"),
    "Wheat (CBOT SRW)":  ContractSpec("ZW", "CBT", (3, 5, 7, 9, 12), _GRAINS_ROLL, yf_ticker="ZW=F"),
    "Wheat (KC HRW)":    ContractSpec("KE", "CBT", (3, 5, 7, 9, 12), _GRAINS_ROLL, yf_ticker="KE=F"),
    "Soybeans (CBOT)":   ContractSpec("ZS", "CBT", (1, 3, 5, 7, 8, 9, 11), _GRAINS_ROLL, yf_ticker="ZS=F"),
    "Soybean Oil":       ContractSpec("ZL", "CBT", (1, 3, 5, 7, 8, 9, 10, 12), _GRAINS_ROLL, yf_ticker="ZL=F"),
    "Soybean Meal":      ContractSpec("ZM", "CBT", (1, 3, 5, 7, 8, 9, 10, 12), _GRAINS_ROLL, yf_ticker="ZM=F"),
    "Oats (CBOT)":       ContractSpec("ZO", "CBT", (3, 5, 7, 9, 12), _GRAINS_ROLL, yf_ticker="ZO=F"),
    "Rough Rice (CBOT)": ContractSpec("ZR", "CBT", (1, 3, 5, 7, 9, 11), _GRAINS_ROLL, yf_ticker="ZR=F"),

    # ── AGRICULTURE — ICE / NYBOT ──────────────────────────────────────────────
    "Coffee":              ContractSpec("KC", "NYB", (3, 5, 7, 9, 12), _SOFTS_ROLL, yf_ticker="KC=F"),
    "Cocoa":               ContractSpec("CC", "NYB", (3, 5, 7, 9, 12), _SOFTS_ROLL, yf_ticker="CC=F"),
    "Sugar":               ContractSpec("SB", "NYB", (3, 5, 7, 10), _SOFTS_ROLL, yf_ticker="SB=F"),
    "Cotton":              ContractSpec("CT", "NYB", (3, 5, 7, 10, 12), _SOFTS_ROLL, yf_ticker="CT=F"),
    "Orange Juice (FCOJ-A)": ContractSpec("OJ", "NYB", (1, 3, 5, 7, 9, 11), _SOFTS_ROLL, yf_ticker="OJ=F"),

    # ── LIVESTOCK — CME ────────────────────────────────────────────────────────
    "Live Cattle":   ContractSpec("LE", "CME", (2, 4, 6, 8, 10, 12), _LIVESTOCK_ROLL, yf_ticker="LE=F"),
    "Feeder Cattle": ContractSpec("GF", "CME", (1, 3, 4, 5, 8, 9, 10, 11), _LIVESTOCK_ROLL, yf_ticker="GF=F"),
    "Lean Hogs":     ContractSpec("HE", "CME", (2, 4, 5, 6, 7, 8, 10, 12), _LIVESTOCK_ROLL, yf_ticker="HE=F"),
}
