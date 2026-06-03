"""
Data-quality gate — cheap, headless checks that catch the four ways a price feed
silently goes wrong BEFORE bad data reaches the signal layer:

  1. STALENESS      — a series stopped updating (feed died / ticker delisted).
  2. OUTLIER / SPIKE — an implausible one-day move (digit transposition, unit
                       error, stale-then-jump) that would poison returns.
  3. CALENDAR GAP   — interior holes (missing trading days) that break alignment.
  4. COVERAGE       — instruments in the universe with no recent data at all.

Everything here is a pure function over a wide close-price panel
(DatetimeIndex × ticker), so it runs in the OOS harness, in a scheduled job, or
behind the data-health page with identical results. No DB, no Streamlit, no
network. The richer per-row YF correction logic still lives in
pipeline/price_validator.py; this module is the fast portfolio-wide health read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

SEVERITIES = ("info", "warn", "error")


@dataclass(frozen=True)
class QualityIssue:
    kind: str          # staleness | outlier | calendar_gap | coverage
    severity: str      # info | warn | error
    instrument: str    # ticker, or "*" for universe-level
    detail: str        # human-readable explanation


@dataclass
class QualityConfig:
    max_stale_bdays: int = 3        # warn beyond this; error at 2x
    outlier_sigma: float = 8.0      # robust-z threshold on daily log returns
    outlier_hard_pct: float = 0.40  # absolute one-day move that is always flagged
    max_gap_bdays: int = 3          # interior calendar hole tolerated
    coverage_recent_bdays: int = 5  # "recent" window for coverage
    min_history: int = 30           # need this many points before judging outliers


@dataclass
class QualityReport:
    issues: List[QualityIssue] = field(default_factory=list)
    n_instruments: int = 0
    asof: Optional[date] = None

    @property
    def ok(self) -> bool:
        """True when no error-severity issue is present (warns are tolerated)."""
        return not any(i.severity == "error" for i in self.issues)

    def of_severity(self, severity: str) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == severity]

    def of_kind(self, kind: str) -> List[QualityIssue]:
        return [i for i in self.issues if i.kind == kind]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(i.kind, i.severity, i.instrument, i.detail) for i in self.issues],
            columns=["kind", "severity", "instrument", "detail"],
        )


def _bdays_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Business days from a to b (>=0), endpoints normalized to dates."""
    if a is None or b is None:
        return 0
    return int(np.busday_count(np.datetime64(a.date()), np.datetime64(b.date())))


def check_staleness(
    panel: pd.DataFrame, asof: pd.Timestamp, cfg: QualityConfig
) -> List[QualityIssue]:
    out: List[QualityIssue] = []
    for col in panel.columns:
        s = panel[col].dropna()
        if s.empty:
            continue  # handled by coverage
        last = pd.Timestamp(s.index[-1])
        stale = _bdays_between(last, asof)
        if stale > 2 * cfg.max_stale_bdays:
            out.append(QualityIssue("staleness", "error", str(col),
                                    f"no update for {stale} business days (last {last.date()})"))
        elif stale > cfg.max_stale_bdays:
            out.append(QualityIssue("staleness", "warn", str(col),
                                    f"stale {stale} business days (last {last.date()})"))
    return out


def check_outliers(panel: pd.DataFrame, cfg: QualityConfig) -> List[QualityIssue]:
    out: List[QualityIssue] = []
    for col in panel.columns:
        s = panel[col].dropna()
        if len(s) < cfg.min_history:
            continue
        ret = np.log(s / s.shift(1)).dropna()
        if ret.empty:
            continue
        med = ret.median()
        mad = (ret - med).abs().median()
        scale = mad * 1.4826 if mad > 0 else ret.std(ddof=0)
        if not scale or not np.isfinite(scale):
            continue
        robust_z = (ret - med) / scale
        flagged = ret[(robust_z.abs() >= cfg.outlier_sigma) |
                      (ret.abs() >= np.log1p(cfg.outlier_hard_pct))]
        for ts, r in flagged.items():
            pct = np.expm1(r) * 100.0
            out.append(QualityIssue("outlier", "warn", str(col),
                                    f"{pct:+.1f}% on {pd.Timestamp(ts).date()} "
                                    f"(robust z={float((r - med) / scale):+.1f})"))
    return out


def check_calendar_alignment(panel: pd.DataFrame, cfg: QualityConfig) -> List[QualityIssue]:
    out: List[QualityIssue] = []
    for col in panel.columns:
        s = panel[col].dropna()
        if len(s) < 2:
            continue
        idx = pd.DatetimeIndex(s.index)
        gaps = np.busday_count(idx[:-1].values.astype("datetime64[D]"),
                               idx[1:].values.astype("datetime64[D]"))
        worst = int(gaps.max()) if len(gaps) else 0
        if worst > cfg.max_gap_bdays:
            where = idx[int(np.argmax(gaps)) + 1].date()
            out.append(QualityIssue("calendar_gap", "warn", str(col),
                                    f"{worst}-business-day hole before {where}"))
    return out


def check_coverage(
    panel: pd.DataFrame,
    expected: Optional[Sequence[str]],
    asof: pd.Timestamp,
    cfg: QualityConfig,
) -> List[QualityIssue]:
    out: List[QualityIssue] = []
    expected = list(expected) if expected is not None else list(panel.columns)
    cutoff_n = cfg.coverage_recent_bdays
    for ticker in expected:
        if ticker not in panel.columns:
            out.append(QualityIssue("coverage", "error", str(ticker), "no column in panel"))
            continue
        s = panel[ticker].dropna()
        if s.empty:
            out.append(QualityIssue("coverage", "error", str(ticker), "no observations at all"))
            continue
        if _bdays_between(pd.Timestamp(s.index[-1]), asof) > cutoff_n:
            out.append(QualityIssue("coverage", "warn", str(ticker),
                                    f"no data in last {cutoff_n} business days"))
    return out


def run_quality_report(
    panel: pd.DataFrame,
    expected: Optional[Sequence[str]] = None,
    asof: Optional[date] = None,
    config: Optional[QualityConfig] = None,
) -> QualityReport:
    """Run all four checks over a wide close-price panel and aggregate."""
    cfg = config or QualityConfig()
    if panel is None or panel.empty:
        return QualityReport(
            issues=[QualityIssue("coverage", "error", "*", "empty panel")],
            n_instruments=0,
            asof=pd.Timestamp(asof).date() if asof is not None else None,
        )
    panel = panel.sort_index()
    ts = pd.Timestamp(asof) if asof is not None else pd.Timestamp(panel.index[-1])
    issues: List[QualityIssue] = []
    issues += check_staleness(panel, ts, cfg)
    issues += check_outliers(panel, cfg)
    issues += check_calendar_alignment(panel, cfg)
    issues += check_coverage(panel, expected, ts, cfg)
    return QualityReport(issues=issues, n_instruments=panel.shape[1], asof=ts.date())
