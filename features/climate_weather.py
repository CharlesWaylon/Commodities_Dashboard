"""
Climate and weather feature signals via NOAA free APIs.

Three signals, each targeting a distinct commodity group:

1. Palmer Drought Severity Index (PDSI) — US Corn Belt
   PDSI measures moisture supply and demand on a monthly basis. Sustained
   negative PDSI (drought) in Iowa/Illinois/Indiana/Nebraska leads Corn and
   Soybean prices by 2-6 weeks, as traders anticipate yield losses before
   the crop condition reports confirm them. Data source: NOAA Climate Data
   Online (CDO) API, climate division dataset.

2. Heating/Cooling Degree Days (HDD/CDD) deviation
   HDD = max(65°F − avg_temp, 0). CDD = max(avg_temp − 65°F, 0).
   HDD drives residential natural gas heating demand (Nov-Mar). CDD drives
   electricity and air conditioning demand in summer — relevant for NG
   peaking plants and Heating Oil. "Deviation from 30-year normal" is the
   tradeable signal: a week 30% colder than normal tightens storage rapidly.
   Data source: NOAA CDO API, GHCND (daily summaries) for population-weighted
   US grid stations.

3. Multivariate ENSO Index v2 (MEI.v2)
   El Niño raises sea surface temperatures in the central Pacific, disrupting
   atmospheric circulation globally:
     Positive MEI (El Niño): dryer SE Asia → weaker Indonesian palm oil/cocoa/
       coffee; wetter US Gulf Coast → improved US soy; drier Australian wheat.
     Negative MEI (La Niña): wetter SE Asia; drier South America → Brazil
       soy stress; higher US natural gas demand via cooler winters.
   Lags vary: Cocoa 3-4 months, Coffee 3-6 months, Sugar 2-4 months.
   Data source: NOAA PSL, publicly available flat file (no API key required).

API keys
--------
NOAA CDO API requires a free token obtainable at:
  https://www.ncdc.noaa.gov/cdo-web/token
Store as environment variable NOAA_CDO_TOKEN or pass directly.
If the token is absent, PDSI and HDD/CDD features return NaN columns
with a logged warning. MEI (ENSO) requires no key.

Usage
-----
    from features.climate_weather import build_climate_features

    df = build_climate_features(noaa_token="YOUR_TOKEN")
    # Columns: pdsi_cornbelt, pdsi_zscore, hdd, cdd, hdd_dev, cdd_dev,
    #          mei, mei_lag3m, mei_lag6m, enso_phase
"""

import os
import warnings
import numpy as np
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, date, timedelta
from typing import Optional

# ── NOAA API configuration ─────────────────────────────────────────────────────
_CDO_BASE     = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
_MEI_URL      = "https://psl.noaa.gov/enso/mei/data/meiv2.data"
_REQUEST_TIMEOUT = 30   # seconds

# US Corn Belt climate divisions (NOAA CLIMDIV)
# Iowa (IA), Illinois (IL), Indiana (IN), Nebraska (NE), Minnesota (MN)
_CORNBELT_DIVISIONS = [
    "CLIMDIV-PDSIDR-1001", "CLIMDIV-PDSIDR-1004",  # Iowa (4 divisions)
    "CLIMDIV-PDSIDR-1101", "CLIMDIV-PDSIDR-1104",  # Illinois
    "CLIMDIV-PDSIDR-1201", "CLIMDIV-PDSIDR-1204",  # Indiana
    "CLIMDIV-PDSIDR-2501", "CLIMDIV-PDSIDR-2504",  # Nebraska
    "CLIMDIV-PDSIDR-2101", "CLIMDIV-PDSIDR-2104",  # Minnesota
]

# Population-weighted HDD/CDD stations (major Midwest/East cities)
_HDD_CDD_STATIONS = [
    "GHCND:USW00094847",   # Chicago O'Hare
    "GHCND:USW00014739",   # New York JFK
    "GHCND:USW00013739",   # Philadelphia Int'l
    "GHCND:USW00013960",   # Atlanta
    "GHCND:USW00094789",   # Detroit Metro
]

# 30-year HDD/CDD normals (approximate, for deviation computation)
# Units: degree-days per month. Source: NOAA 1991-2020 Normals.
_HDD_NORMALS = {   # (month → avg annual HDD for these stations)
    1: 985, 2: 840, 3: 602, 4: 265, 5: 72,  6: 4,
    7: 0,   8: 1,   9: 28,  10: 193, 11: 558, 12: 869,
}
_CDD_NORMALS = {
    1: 0,  2: 0,  3: 4,   4: 28,  5: 112, 6: 320,
    7: 480, 8: 440, 9: 192, 10: 38, 11: 2,   12: 0,
}

# MEI thresholds for ENSO phase classification
MEI_ELNINO  =  0.5    # above = El Niño
MEI_LANINA  = -0.5    # below = La Niña


def _cdo_headers(token: str) -> dict:
    return {"token": token}


# ── 1. PDSI — Corn Belt ───────────────────────────────────────────────────────

def fetch_pdsi_cornbelt(
    token: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch Palmer Drought Severity Index for the US Corn Belt via NOAA CDO API.

    PDSI is monthly. Values:
      < −4: extreme drought
      −4 to −3: severe drought
      −3 to −2: moderate drought (first tradeable signal for Corn/Soy)
      −1 to +1: near normal

    Returns
    -------
    pd.DataFrame
        Columns: pdsi_cornbelt (mean across divisions), pdsi_zscore (63-day).
        DatetimeIndex at monthly frequency, then resampled to daily.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    url = f"{_CDO_BASE}/data"
    params = {
        "datasetid":  "NORMAL_MLY",
        "datatypeid": "PDSI",
        "startdate":  start_date,
        "enddate":    end_date,
        "stationid":  _CORNBELT_DIVISIONS[:5],  # limit to 5 for API limit
        "limit":      1000,
        "units":      "standard",
    }

    try:
        resp = requests.get(
            url, params=params,
            headers=_cdo_headers(token),
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("results", [])
    except Exception as e:
        warnings.warn(f"PDSI fetch failed: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

    if not data:
        warnings.warn("NOAA CDO returned no PDSI results. Check token and date range.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.groupby("date")["value"].mean().rename("pdsi_cornbelt")

    # Resample monthly → daily (forward fill, business days)
    daily_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="B")
    daily = monthly.reindex(daily_idx).ffill()

    mu  = daily.rolling(63).mean()
    sig = daily.rolling(63).std().clip(lower=1e-9)
    zscore = ((daily - mu) / sig).rename("pdsi_zscore")

    result = pd.DataFrame({"pdsi_cornbelt": daily, "pdsi_zscore": zscore})
    result.index.name = "Date"
    return result


# ── 2. HDD/CDD — deviation from 30-year normals ───────────────────────────────

def fetch_hdd_cdd(
    token: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch daily average temperature for HDD/CDD computation via NOAA CDO.

    HDD deviation = (HDD_actual − HDD_30yr_normal) / HDD_30yr_normal.
    Positive deviation → colder than normal → higher NG/Heating Oil demand.

    Returns
    -------
    pd.DataFrame
        Columns: hdd, cdd, hdd_deviation_pct, cdd_deviation_pct,
                 hdd_21d_sum, cdd_21d_sum, hdd_dev_21d.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    url = f"{_CDO_BASE}/data"
    params = {
        "datasetid":  "GHCND",
        "datatypeid": "TAVG",
        "startdate":  start_date,
        "enddate":    end_date,
        "stationid":  _HDD_CDD_STATIONS[:3],
        "limit":      1000,
        "units":      "standard",
    }

    try:
        resp = requests.get(
            url, params=params,
            headers=_cdo_headers(token),
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("results", [])
    except Exception as e:
        warnings.warn(f"HDD/CDD fetch failed: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

    if not data:
        warnings.warn("NOAA CDO returned no temperature results.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    # TAVG in tenths of °C → convert to °F
    tavg_f = df.groupby("date")["value"].mean() / 10 * 9/5 + 32

    hdd = tavg_f.apply(lambda t: max(65 - t, 0)).rename("hdd")
    cdd = tavg_f.apply(lambda t: max(t - 65, 0)).rename("cdd")

    result = pd.DataFrame({"hdd": hdd, "cdd": cdd})
    result["month"] = result.index.month

    # Deviation from 30-year normals (normalised per-month)
    result["hdd_normal"] = result["month"].map(_HDD_NORMALS) / 30   # daily avg
    result["cdd_normal"] = result["month"].map(_CDD_NORMALS) / 30

    result["hdd_deviation_pct"] = (
        (result["hdd"] - result["hdd_normal"]) /
        result["hdd_normal"].clip(lower=0.1)
    ).clip(-5, 5)
    result["cdd_deviation_pct"] = (
        (result["cdd"] - result["cdd_normal"]) /
        result["cdd_normal"].clip(lower=0.1)
    ).clip(-5, 5)

    result["hdd_21d_sum"]  = result["hdd"].rolling(21).sum()
    result["cdd_21d_sum"]  = result["cdd"].rolling(21).sum()
    result["hdd_dev_21d"]  = result["hdd_deviation_pct"].rolling(21).mean()

    return result.drop(columns=["month", "hdd_normal", "cdd_normal"])


# ── 3. MEI — ENSO index ───────────────────────────────────────────────────────

def fetch_mei(lag_months: list = [3, 6]) -> pd.DataFrame:
    """
    Download the Multivariate ENSO Index v2 from NOAA PSL (no API key).

    MEI is bimonthly (2-month running means: DJFM, JFMA, FMAM, ...).
    We resample to monthly then daily.

    Parameters
    ----------
    lag_months : list[int]
        Lag the MEI series by these many months for commodity lead-lag signals.
        [3, 6] → leads Cocoa/Coffee by 3 and 6 months respectively.

    Returns
    -------
    pd.DataFrame
        Columns: mei, mei_lag3m, mei_lag6m (or per lag_months), enso_phase.
        DatetimeIndex at daily frequency.
    """
    try:
        resp = requests.get(_MEI_URL, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        warnings.warn(f"MEI download failed: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

    # Parse the flat file — format: year col1 col2 ... col12 (bimonthly periods)
    records = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        year = int(parts[0])
        for i, val in enumerate(parts[1:13]):
            try:
                mei_val = float(val)
                if abs(mei_val) >= 999:   # NOAA uses -999.00 as missing sentinel
                    continue
                # Bimonthly periods: DJ=1, JF=2, FM=3, MA=4, AM=5, MJ=6, ...
                # Centre each bimonthly period on its second month
                month = min(i + 2, 12)
                records.append({"year": year, "month": month, "mei": mei_val})
            except ValueError:
                continue

    if not records:
        warnings.warn("Could not parse MEI data.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    monthly = df.set_index("date")["mei"].sort_index()
    monthly = monthly[~monthly.index.duplicated(keep="last")]

    # Resample to daily
    daily_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="B")
    daily = monthly.reindex(daily_idx).ffill()

    result = pd.DataFrame({"mei": daily})

    for lag in lag_months:
        col = f"mei_lag{lag}m"
        result[col] = result["mei"].shift(lag * 21)   # ~21 trading days per month

    # ENSO phase: El Niño / Neutral / La Niña
    result["enso_phase"] = np.where(
        result["mei"] >= MEI_ELNINO, 1.0,
        np.where(result["mei"] <= MEI_LANINA, -1.0, 0.0)
    )

    result.index.name = "Date"
    return result


# ── Convenience runner ─────────────────────────────────────────────────────────

def build_climate_features(
    noaa_token: Optional[str] = None,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assemble all climate and weather signals into one aligned DataFrame.

    Parameters
    ----------
    noaa_token : str, optional
        NOAA CDO API token for PDSI and HDD/CDD. If None, falls back to
        the NOAA_CDO_TOKEN environment variable. PDSI/HDD features are
        NaN if no token is available; MEI requires no token.
    start_date, end_date : str
        ISO format date bounds for historical data retrieval.

    Returns
    -------
    pd.DataFrame
        All climate columns on a daily DatetimeIndex.
        Columns available without a token: mei, mei_lag3m, mei_lag6m, enso_phase.
        Columns requiring a token: pdsi_cornbelt, pdsi_zscore,
          hdd, cdd, hdd_deviation_pct, cdd_deviation_pct, hdd_dev_21d.
    """
    token = noaa_token or os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        warnings.warn(
            "No NOAA_CDO_TOKEN found. PDSI and HDD/CDD features will be empty. "
            "Get a free token at https://www.ncdc.noaa.gov/cdo-web/token"
        )

    parts = []

    # MEI — no key needed
    try:
        mei_df = fetch_mei(lag_months=[3, 6])
        if not mei_df.empty:
            parts.append(mei_df)
    except Exception as e:
        warnings.warn(f"MEI fetch failed: {e}")

    if token:
        # PDSI
        try:
            pdsi_df = fetch_pdsi_cornbelt(token, start_date, end_date)
            if not pdsi_df.empty:
                parts.append(pdsi_df)
        except Exception as e:
            warnings.warn(f"PDSI fetch failed: {e}")

        # HDD/CDD
        try:
            hdd_df = fetch_hdd_cdd(token, start_date, end_date)
            if not hdd_df.empty:
                parts.append(hdd_df)
        except Exception as e:
            warnings.warn(f"HDD/CDD fetch failed: {e}")

    if not parts:
        return pd.DataFrame()

    combined = parts[0]
    for df in parts[1:]:
        combined = combined.join(df, how="outer")

    return combined.ffill(limit=3).sort_index()
