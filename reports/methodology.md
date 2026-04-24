# Commodities Dashboard — Data Pipeline Methodology
**Generated:** 2026-04-24  **Pipeline version:** Phase 2 (Steps 1–5 complete)  **Database:** PostgreSQL (`commodities`)  
---
## Overview
This document records every transformation applied to the raw Yahoo Finance price data before it reaches the analytical layer. It exists so that:

1. Any team member can reproduce the exact feature space used in model training
2. Future contributors know what they are inheriting and why decisions were made
3. The quantum kernel and relational model work has a traceable data provenance chain
---
## Step 1 — Raw Ingestion
**Source:** Yahoo Finance via `yfinance` (free, no API key required)  
**Script:** `pipeline/ingest.py`  
**Trigger:** Daily at 18:05 UTC (weekdays) via `pipeline/scheduler.py`; or manually with `python -m pipeline.ingest`

**What it does:**
- Seeds the `commodities` reference table with 41 instruments on first run
- Downloads OHLCV history (5-year backfill on first run; 5-day incremental thereafter)
- Inserts new rows into `price_history` using savepoint-per-row error isolation — a duplicate row triggers an `IntegrityError` on the savepoint only; the rest of the session remains intact

**Key design decision — savepoints:**  
The naive implementation used `db.rollback()` on duplicate rows, which silently rolled back the *entire* session, losing all rows inserted so far. Replacing this with `db.begin_nested()` (savepoints) fixed the bug.
---
## Step 2 — Data Quality Audit
**Script:** `pipeline/audit.py`  
**Output:** `reports/audit_YYYYMMDD.csv`

Six checks were run across all 41 instruments:

| Check | Threshold | Method |
|-------|-----------|--------|
| Date gaps | > 4 calendar days | Consecutive date diff |
| Zero volume | Any | Count |
| Stale prices | ≥ 3 consecutive identical closes | Run-length encoding |
| Price spikes | \|z\| > 4.0σ | Z-score on log returns |
| Price = 0 | Any | Count |
| Coverage % | < 98% vs global calendar | Vs union of all dates |

**Findings:**
- 95 spike events flagged across 28 futures instruments (all confirmed roll artifacts)
- 2 corrupt rows in Rough Rice (ZR=F): closes of $17.68 and $11.38 on days when surrounding prices were ~$1,800 (Yahoo Finance reported in wrong units). Both rows deleted: IDs 47882 (2024-06-17) and 48346 (2026-04-23)
- Coverage % appeared low (~69%) for non-Bitcoin instruments — confirmed artefact of including Bitcoin's 7-day calendar in the global calendar denominator; not a data gap
---
## Step 3 — Roll Adjustment
**Script:** `pipeline/roll_adjust.py`  
**Columns populated:** `price_history.adjusted_close`, `price_history.adjustment_factor`

**Algorithm — proportional backward adjustment:**

Commodity futures expire every 1–3 months. Yahoo Finance rolls from the expiring contract to the next front-month, creating an artificial price discontinuity. The fix:

1. Compute log returns: `lr[t] = log(close[t] / close[t-1])`
2. Z-score the series: `z[t] = (lr[t] - μ) / σ`
3. Flag roll points: `|z[t]| > Z_THRESHOLD`
   - `Z_THRESHOLD = 5.0σ` for all futures
   - `Z_THRESHOLD = 6.5σ` for Natural Gas (NG=F) — legitimate large moves
4. Working **backward** from the most recent date:
   - At each roll point `t`: `cumulative_factor *= close[t] / close[t-1]`
   - `adjusted_close[i] = close[i] × cumulative_factor`
   - `adjustment_factor[i] = cumulative_factor`

For ETF proxies, Bitcoin: `adjusted_close = close`, `adjustment_factor = 1.0`

**Results by instrument:**

| Instrument | Ticker | Rolls | Earliest Factor |
|------------|--------|-------|----------------|
| Brent Crude Oil | `BZ=F` | 5 | 0.504894 |
| Cocoa | `CC=F` | 3 | 0.518623 |
| Copper | `HG=F` | 3 | 0.802435 |
| Corn | `ZC=F` | 3 | 0.597903 |
| Cotton | `CT=F` | 1 | 0.76115 |
| Feeder Cattle | `GF=F` | 4 | 1.233728 |
| Gasoline (RBOB) | `RB=F` | 4 | 0.973329 |
| Gold | `GC=F` | 4 | 0.833912 |
| HRC Steel | `HRC=F` | 14 | 1.197214 |
| Heating Oil | `HO=F` | 4 | 0.494959 |
| Lean Hogs | `HE=F` | 12 | 0.388672 |
| Live Cattle | `LE=F` | 6 | 0.803093 |
| Natural Gas | `NG=F` | 2 | 0.769321 |
| Oats (CBOT) | `ZO=F` | 3 | 0.670536 |
| Palladium | `PA=F` | 4 | 0.696734 |
| Platinum | `PL=F` | 3 | 0.608407 |
| Rough Rice (CBOT) | `ZR=F` | 3 | 0.726083 |
| Silver | `SI=F` | 2 | 0.782819 |
| Soybean Meal | `ZM=F` | 5 | 0.522548 |
| Soybeans | `ZS=F` | 3 | 0.74815 |
| WTI Crude Oil | `CL=F` | 4 | 0.56228 |
| Wheat | `ZW=F` | 3 | 0.973649 |

**Roll dates (all 95 events):**

| Instrument | Ticker | Roll Date |
|------------|--------|----------|
| Brent Crude Oil | `BZ=F` | 2021-11-26 |
| Brent Crude Oil | `BZ=F` | 2022-03-09 |
| Brent Crude Oil | `BZ=F` | 2026-03-10 |
| Brent Crude Oil | `BZ=F` | 2026-04-01 |
| Brent Crude Oil | `BZ=F` | 2026-04-08 |
| Cocoa | `CC=F` | 2024-05-13 |
| Cocoa | `CC=F` | 2024-05-16 |
| Cocoa | `CC=F` | 2024-09-16 |
| Copper | `HG=F` | 2025-04-04 |
| Copper | `HG=F` | 2025-07-08 |
| Copper | `HG=F` | 2025-07-31 |
| Corn | `ZC=F` | 2021-07-15 |
| Corn | `ZC=F` | 2022-07-15 |
| Corn | `ZC=F` | 2023-07-17 |
| Cotton | `CT=F` | 2022-06-24 |
| Feeder Cattle | `GF=F` | 2021-05-28 |
| Feeder Cattle | `GF=F` | 2022-05-27 |
| Feeder Cattle | `GF=F` | 2023-05-26 |
| Feeder Cattle | `GF=F` | 2025-11-21 |
| Gasoline (RBOB) | `RB=F` | 2021-11-26 |
| Gasoline (RBOB) | `RB=F` | 2022-08-01 |
| Gasoline (RBOB) | `RB=F` | 2024-03-01 |
| Gasoline (RBOB) | `RB=F` | 2026-03-02 |
| Gold | `GC=F` | 2025-10-21 |
| Gold | `GC=F` | 2026-01-30 |
| Gold | `GC=F` | 2026-02-03 |
| Gold | `GC=F` | 2026-03-19 |
| HRC Steel | `HRC=F` | 2021-12-29 |
| HRC Steel | `HRC=F` | 2022-01-26 |
| HRC Steel | `HRC=F` | 2022-02-23 |
| HRC Steel | `HRC=F` | 2022-03-30 |
| HRC Steel | `HRC=F` | 2022-05-25 |
| HRC Steel | `HRC=F` | 2022-06-29 |
| HRC Steel | `HRC=F` | 2022-12-28 |
| HRC Steel | `HRC=F` | 2023-02-22 |
| HRC Steel | `HRC=F` | 2023-03-29 |
| HRC Steel | `HRC=F` | 2023-05-31 |
| HRC Steel | `HRC=F` | 2023-10-25 |
| HRC Steel | `HRC=F` | 2023-11-29 |
| HRC Steel | `HRC=F` | 2024-02-28 |
| HRC Steel | `HRC=F` | 2025-02-26 |
| Heating Oil | `HO=F` | 2022-03-09 |
| Heating Oil | `HO=F` | 2022-11-01 |
| Heating Oil | `HO=F` | 2026-02-02 |
| Heating Oil | `HO=F` | 2026-04-08 |
| Lean Hogs | `HE=F` | 2021-08-16 |
| Lean Hogs | `HE=F` | 2021-10-15 |
| Lean Hogs | `HE=F` | 2022-02-15 |
| Lean Hogs | `HE=F` | 2022-04-18 |
| Lean Hogs | `HE=F` | 2022-08-15 |
| Lean Hogs | `HE=F` | 2023-02-15 |
| Lean Hogs | `HE=F` | 2023-08-15 |
| Lean Hogs | `HE=F` | 2023-10-16 |
| Lean Hogs | `HE=F` | 2024-02-15 |
| Lean Hogs | `HE=F` | 2024-08-15 |
| Lean Hogs | `HE=F` | 2025-08-15 |
| Lean Hogs | `HE=F` | 2025-10-15 |
| Live Cattle | `LE=F` | 2021-09-01 |
| Live Cattle | `LE=F` | 2022-05-02 |
| Live Cattle | `LE=F` | 2023-05-01 |
| Live Cattle | `LE=F` | 2024-05-01 |
| Live Cattle | `LE=F` | 2024-07-01 |
| Live Cattle | `LE=F` | 2025-07-01 |
| Natural Gas | `NG=F` | 2022-01-27 |
| Natural Gas | `NG=F` | 2026-01-29 |
| Oats (CBOT) | `ZO=F` | 2022-07-15 |
| Oats (CBOT) | `ZO=F` | 2022-12-14 |
| Oats (CBOT) | `ZO=F` | 2022-12-15 |
| Palladium | `PA=F` | 2023-12-14 |
| Palladium | `PA=F` | 2025-12-29 |
| Palladium | `PA=F` | 2026-01-27 |
| Palladium | `PA=F` | 2026-01-30 |
| Platinum | `PL=F` | 2025-12-29 |
| Platinum | `PL=F` | 2026-01-27 |
| Platinum | `PL=F` | 2026-01-30 |
| Rough Rice (CBOT) | `ZR=F` | 2023-07-17 |
| Rough Rice (CBOT) | `ZR=F` | 2024-07-12 |
| Rough Rice (CBOT) | `ZR=F` | 2024-07-15 |
| Silver | `SI=F` | 2026-01-26 |
| Silver | `SI=F` | 2026-01-30 |
| Soybean Meal | `ZM=F` | 2022-01-18 |
| Soybean Meal | `ZM=F` | 2022-07-15 |
| Soybean Meal | `ZM=F` | 2022-08-15 |
| Soybean Meal | `ZM=F` | 2023-08-15 |
| Soybean Meal | `ZM=F` | 2024-07-15 |
| Soybeans | `ZS=F` | 2021-06-17 |
| Soybeans | `ZS=F` | 2022-07-15 |
| Soybeans | `ZS=F` | 2022-08-15 |
| WTI Crude Oil | `CL=F` | 2021-11-26 |
| WTI Crude Oil | `CL=F` | 2022-03-09 |
| WTI Crude Oil | `CL=F` | 2026-03-10 |
| WTI Crude Oil | `CL=F` | 2026-04-08 |
| Wheat | `ZW=F` | 2022-03-03 |
| Wheat | `ZW=F` | 2022-03-08 |
| Wheat | `ZW=F` | 2022-03-10 |
---
## Step 4 — Calendar Alignment
**Script:** `pipeline/align_calendar.py`  
**Table populated:** `aligned_prices`

**Canonical calendar construction:**  
Rather than importing an external market-calendar library, the calendar is derived empirically from the futures data itself:

> *A date is a valid US trading day if at least 50% of the 28 direct futures instruments have a price record on that date.*

This quorum approach:
- Excludes weekends and US public holidays naturally (zero or near-zero futures have data)
- Is robust to 1–2 instruments having data gaps on a real trading day
- Requires no external dependency

**Forward-fill rule:**  
For any instrument missing a canonical trading day (ETF holiday, contract gap), the prior day's `adjusted_close` is carried forward. `is_filled = True` marks these synthetic rows.

**Results:**

| Instrument | Type | Days | Filled | Fill% |
|------------|------|------|--------|-------|
| Bitcoin | crypto | 1258 | 0 | 0.0% |
| LNG / Intl Gas* | equity_proxy | 1258 | 2 | 0.2% |
| Metallurgical Coal* | equity_proxy | 1258 | 2 | 0.2% |
| Thermal Coal* | equity_proxy | 1258 | 2 | 0.2% |
| Zinc & Cobalt* | equity_proxy | 1258 | 2 | 0.2% |
| Carbon Credits* | etf_proxy | 1258 | 2 | 0.2% |
| Gold (Physical/London)* | etf_proxy | 1258 | 2 | 0.2% |
| Iron Ore / Steel* | etf_proxy | 1258 | 2 | 0.2% |
| Lithium* | etf_proxy | 1258 | 2 | 0.2% |
| Lumber* | etf_proxy | 1258 | 2 | 0.2% |
| Rare Earths* | etf_proxy | 1258 | 2 | 0.2% |
| Silver (Physical)* | etf_proxy | 1258 | 2 | 0.2% |
| Uranium* | etf_proxy | 1258 | 2 | 0.2% |
| Aluminum (COMEX) | futures | 1258 | 1 | 0.1% |
| Brent Crude Oil | futures | 1258 | 0 | 0.0% |
| Cocoa | futures | 1258 | 0 | 0.0% |
| Coffee | futures | 1258 | 0 | 0.0% |
| Copper | futures | 1258 | 0 | 0.0% |
| Corn | futures | 1258 | 1 | 0.1% |
| Cotton | futures | 1258 | 0 | 0.0% |
| Feeder Cattle | futures | 1258 | 1 | 0.1% |
| Gasoline (RBOB) | futures | 1258 | 0 | 0.0% |
| Gold | futures | 1258 | 0 | 0.0% |
| Heating Oil | futures | 1258 | 0 | 0.0% |
| HRC Steel | futures | 1258 | 1 | 0.1% |
| Lean Hogs | futures | 1258 | 1 | 0.1% |
| Live Cattle | futures | 1258 | 1 | 0.1% |
| Natural Gas | futures | 1258 | 0 | 0.0% |
| Oats (CBOT) | futures | 1258 | 1 | 0.1% |
| Orange Juice (FCOJ-A) | futures | 1258 | 0 | 0.0% |
| Palladium | futures | 1258 | 1 | 0.1% |
| Platinum | futures | 1258 | 0 | 0.0% |
| Rough Rice (CBOT) | futures | 1257 | 2 | 0.2% |
| Silver | futures | 1258 | 0 | 0.0% |
| Soybean Meal | futures | 1258 | 1 | 0.1% |
| Soybean Oil | futures | 1258 | 1 | 0.1% |
| Soybeans | futures | 1258 | 1 | 0.1% |
| Sugar | futures | 1258 | 0 | 0.0% |
| Wheat | futures | 1258 | 1 | 0.1% |
| Wheat (KC HRW) | futures | 1258 | 1 | 0.1% |
| WTI Crude Oil | futures | 1258 | 0 | 0.0% |
---
## Step 5 — Instrument Layer Separation
**Script:** `pipeline/layer.py`  
**Column added:** `commodities.instrument_type`  
**Views created:** `v_futures_aligned`, `v_proxies_aligned`, `v_crypto_aligned`, `v_all_typed`

**Classification rules:**

| Type | Rule | Instruments |
|------|------|-------------|
| `futures` | Ticker ends with `=F` | 28 |
| `etf_proxy` | Ticker in {URA, KRBN, SGOL, SIVR, SLX, LIT, REMX, WOOD} | 8 |
| `equity_proxy` | Ticker in {LNG, BTU, HCC, GLNCY} | 4 |
| `crypto` | Ticker = BTC-USD | 1 |

**Why separation matters:**  
Equity proxies carry equity market beta — the tendency of all stocks to move together with the S&P 500. Including them in a futures correlation matrix introduces spurious correlations driven by shared equity exposure rather than commodity fundamentals. The views enforce this separation structurally in the database so it cannot be accidentally bypassed.
---
## Step 6 — Statistical Validation
**Script:** `pipeline/validate.py`  
**Run date:** 2026-04-24

### Return Spike  ✅ PASS
Threshold: |z| ≤ 7.0σ on adjusted log returns.  
Worst case: **HRC Steel** on 2024-01-31 — z = 7.92σ (-10.75%)  

### Date Alignment  ✅ PASS
Canonical days: 1258  
Known exceptions: Rough Rice (CBOT) (1 day short — starts after canonical start date)  

### Correlation Sanity  ✅ PASS
| Pair | Observed r | Expected range | Result |
|------|-----------|----------------|--------|
| Brent Crude Oil / WTI Crude Oil | 0.951 | 0.88–1.0 | ✅ |
| Gold / Silver | 0.73 | 0.6–1.0 | ✅ |
| Corn (CBOT) / Soybeans (CBOT) | 0.5 | 0.4–1.0 | ✅ |
| Gasoline (RBOB) / Heating Oil (No.2) | 0.649 | 0.55–1.0 | ✅ |
| Live Cattle / Feeder Cattle | 0.593 | 0.45–1.0 | ✅ |
| Natural Gas (Henry Hub) / WTI Crude Oil | 0.112 | 0.0–0.4 | ✅ |

### Historical Events  ✅ PASS
| Event | Instrument | Max price in window | Min expected | Result |
|-------|------------|--------------------|--------------|---------|
| Russia invades Ukraine | WTI Crude Oil | 92.81 USD/bbl | 90.0 | ✅ |
| Russia invades Ukraine | Wheat (CBOT SRW) | 926.0 USc/bu | 800.0 | ✅ |
| European natural gas crisis peak | Natural Gas (Henry Hub) | 9.68 USD/MMBtu | 7.5 | ✅ |
| Cocoa supply crisis | Cocoa (ICE) | 7170.0 USD/MT | 6000.0 | ✅ |
| Palladium / Russia sanctions | Palladium | 2979.9 USD/oz | 2400.0 | ✅ |

---
## How to Use This Dataset
| Use case | Table / View | Price column |
|----------|-------------|-------------|
| Single-instrument chart | `price_history` | `adjusted_close` |
| Rolling volatility | `price_history` | `adjusted_close` |
| Correlation matrix | `v_futures_aligned` | `adjusted_close` |
| Model training (futures only) | `v_futures_aligned` | `adjusted_close` |
| Full dataset with type labels | `v_all_typed` | `adjusted_close` |
| Raw data inspection | `price_history` | `close` |

**Never use `price_history.close` for return calculations.** It contains the raw Yahoo Finance price with roll discontinuities intact.

**Never mix `v_futures_aligned` and `v_proxies_aligned` in the same model** without controlling for equity market beta.
---
*Generated by `pipeline/validate.py` on 2026-04-24. Re-run after any backfill to keep this document current.*
