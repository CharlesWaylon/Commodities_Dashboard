"""
Tests for models/daily_retrain.py — Phase 5 Part 1.

Coverage
────────
 1. RetrainConfig — defaults are sensible
 2. RetrainSummary.pretty() — generates a non-empty string
 3. run_daily_retrain — succeeds with synthetic data
 4. run_daily_retrain — tier_distribution is populated
 5. run_daily_retrain — MetaPredictor is_trained after run
 6. run_daily_retrain — pkl is saved (non-dry-run)
 7. run_daily_retrain — dry_run does NOT write pkl
 8. run_daily_retrain — empty prices → failure (success=False, error non-empty)
 9. run_daily_retrain — too few pairs guard (< MIN_PAIRS)
10. _persist_training_log / recent_training_runs — roundtrip through SQLite
11. recent_training_runs — empty DB returns empty DataFrame
12. run_daily_retrain — accepts pre-loaded prices_df / macro_df (no DB hit)
13. run_daily_retrain — commodities filtered to those present in prices
14. RetrainSummary — success=False when error is set
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Project root on path ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.daily_retrain import (
    MIN_PAIRS,
    RetrainConfig,
    RetrainSummary,
    _persist_training_log,
    recent_training_runs,
    run_daily_retrain,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_prices(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic price matrix with 3 commodities.
    Large enough that BacktestHarness (min_train_rows=120) always passes.
    """
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    commodities = ["WTI Crude Oil", "Gold (COMEX)", "Corn (CBOT)"]
    data = {}
    for c in commodities:
        data[c] = 50 + np.cumsum(np.random.randn(n) * 0.8)
    return pd.DataFrame(data, index=dates)


def _make_macro(price_idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(price_idx)
    return pd.DataFrame({
        "vix":          [18.0] * n,
        "dxy_zscore63": [0.1] * n,
        "vix_risk_off":  [0.0] * n,
        "vix_crisis":    [0.0] * n,
        "dxy_stress":    [0.0] * n,
        "dxy_tailwind":  [0.0] * n,
        "tbill_yield":   [4.5] * n,
        "tlt_momentum":  [0.0] * n,
        "enso_index":    [0.0] * n,
        "precip_zscore": [0.0] * n,
        "temp_zscore":   [0.0] * n,
    }, index=price_idx)


def _base_config(tmp_path: Path, dry_run: bool = False) -> RetrainConfig:
    return RetrainConfig(
        commodities=["WTI Crude Oil", "Gold (COMEX)", "Corn (CBOT)"],
        max_depth=3,
        min_samples_leaf=1,
        save_path=tmp_path / "meta_predictor.pkl",
        db_path=tmp_path / "test.db",
        dry_run=dry_run,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

PASS = "PASS"


def _check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  FAIL  {label}{(' — ' + detail) if detail else ''}")
        raise AssertionError(f"Test failed: {label}. {detail}")


def run_tests() -> None:
    print("=" * 60)
    print("DAILY RETRAIN PIPELINE — TEST SUITE  (Phase 5 Part 1)")
    print("=" * 60)

    prices = _make_prices()
    macro  = _make_macro(prices.index)

    # ── 1. RetrainConfig defaults ──────────────────────────────────────────────
    cfg = RetrainConfig()
    _check(
        "RetrainConfig — defaults",
        cfg.max_depth == 5
        and cfg.min_samples_leaf == 10
        and cfg.prices_period == "3y"
        and cfg.dry_run is False,
        f"depth={cfg.max_depth} leaf={cfg.min_samples_leaf} period={cfg.prices_period}",
    )

    # ── 2. RetrainSummary.pretty() ─────────────────────────────────────────────
    s = RetrainSummary(retrained_at="2026-04-30T18:00:00+00:00", success=True,
                       n_training_pairs=50)
    pretty = s.pretty()
    _check(
        "RetrainSummary.pretty() — non-empty with status",
        "SUCCESS" in pretty and "50" in pretty,
        f"pretty={pretty[:60]}…",
    )

    # ── 3. run_daily_retrain — success with synthetic data ─────────────────────
    with tempfile.TemporaryDirectory() as td:
        cfg3 = _base_config(Path(td))
        result = run_daily_retrain(config=cfg3, prices_df=prices, macro_df=macro)
        _check(
            "run_daily_retrain — success=True",
            result.success,
            f"error={result.error!r}",
        )

    # ── 4. tier_distribution populated ────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        cfg4 = _base_config(Path(td))
        r4 = run_daily_retrain(config=cfg4, prices_df=prices, macro_df=macro)
        _check(
            "run_daily_retrain — tier_distribution non-empty",
            bool(r4.tier_distribution),
            f"dist={r4.tier_distribution}",
        )
        _check(
            "run_daily_retrain — tier_distribution values sum to n_training_pairs",
            sum(r4.tier_distribution.values()) == r4.n_training_pairs,
            f"sum={sum(r4.tier_distribution.values())} vs pairs={r4.n_training_pairs}",
        )

    # ── 5. MetaPredictor is_trained after run ──────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        cfg5 = _base_config(Path(td))
        r5 = run_daily_retrain(config=cfg5, prices_df=prices, macro_df=macro)
        from models.meta_predictor import MetaPredictor
        mp = MetaPredictor()
        mp.load(Path(td) / "meta_predictor.pkl")
        _check(
            "run_daily_retrain — loaded MetaPredictor is_trained",
            mp.is_trained,
            "pkl loaded but is_trained=False",
        )

    # ── 6. pkl saved (non-dry-run) ─────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        pkl_path = Path(td) / "meta_predictor.pkl"
        cfg6 = RetrainConfig(
            commodities=["WTI Crude Oil", "Gold (COMEX)", "Corn (CBOT)"],
            max_depth=3, min_samples_leaf=1,
            save_path=pkl_path,
            db_path=Path(td) / "test.db",
            dry_run=False,
        )
        run_daily_retrain(config=cfg6, prices_df=prices, macro_df=macro)
        _check(
            "run_daily_retrain — pkl written to disk",
            pkl_path.exists(),
            f"expected {pkl_path}",
        )

    # ── 7. dry_run does NOT write pkl ──────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        pkl_path = Path(td) / "meta_predictor.pkl"
        cfg7 = RetrainConfig(
            commodities=["WTI Crude Oil", "Gold (COMEX)", "Corn (CBOT)"],
            max_depth=3, min_samples_leaf=1,
            save_path=pkl_path,
            db_path=Path(td) / "test.db",
            dry_run=True,
        )
        r7 = run_daily_retrain(config=cfg7, prices_df=prices, macro_df=macro)
        _check(
            "run_daily_retrain — dry_run does not write pkl",
            not pkl_path.exists(),
            f"pkl should not exist but does",
        )
        _check(
            "run_daily_retrain — dry_run still success=True",
            r7.success,
            f"error={r7.error!r}",
        )

    # ── 8. empty prices → failure ─────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        cfg8 = _base_config(Path(td))
        r8 = run_daily_retrain(
            config=cfg8,
            prices_df=pd.DataFrame(),
            macro_df=pd.DataFrame(),
        )
        _check(
            "run_daily_retrain — empty prices → success=False",
            not r8.success,
            f"expected failure but success=True",
        )
        _check(
            "run_daily_retrain — empty prices → error is non-empty",
            bool(r8.error),
            f"error={r8.error!r}",
        )

    # ── 9. too few pairs guard ─────────────────────────────────────────────────
    # Use very short prices so BacktestHarness produces 0 records
    short_prices = _make_prices(n=50)
    short_macro  = _make_macro(short_prices.index)
    with tempfile.TemporaryDirectory() as td:
        pkl_path = Path(td) / "meta_predictor.pkl"
        cfg9 = RetrainConfig(
            commodities=["WTI Crude Oil"],
            max_depth=3, min_samples_leaf=1,
            save_path=pkl_path,
            db_path=Path(td) / "test.db",
        )
        r9 = run_daily_retrain(config=cfg9, prices_df=short_prices, macro_df=short_macro)
        _check(
            "run_daily_retrain — too few pairs → success=False, no pkl",
            not r9.success and not pkl_path.exists(),
            f"success={r9.success} pkl_exists={pkl_path.exists()} error={r9.error!r}",
        )

    # ── 10. _persist_training_log / recent_training_runs roundtrip ────────────
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        s10 = RetrainSummary(
            retrained_at="2026-04-30T18:00:00+00:00",
            n_commodities=3,
            n_training_pairs=60,
            tier_distribution={"statistical": 30, "ml": 30},
            tree_n_leaves=8,
            top_feature="vix",
            save_path=str(td) + "/meta_predictor.pkl",
            error="",
            config=RetrainConfig(),
            success=True,
        )
        _persist_training_log(s10, db_path=db)
        df10 = recent_training_runs(n=5, db_path=db)
        _check(
            "_persist_training_log / recent_training_runs — 1 row returned",
            len(df10) == 1,
            f"rows={len(df10)}",
        )
        _check(
            "recent_training_runs — n_training_pairs correct",
            int(df10.iloc[0]["n_training_pairs"]) == 60,
            f"got {df10.iloc[0]['n_training_pairs']}",
        )
        _check(
            "recent_training_runs — top_feature correct",
            df10.iloc[0]["top_feature"] == "vix",
            f"got {df10.iloc[0]['top_feature']}",
        )

    # ── 11. recent_training_runs — empty DB returns empty DataFrame ────────────
    with tempfile.TemporaryDirectory() as td:
        df11 = recent_training_runs(db_path=Path(td) / "empty.db")
        _check(
            "recent_training_runs — empty DB returns empty DataFrame",
            df11.empty,
            f"rows={len(df11)}",
        )

    # ── 12. accepts pre-loaded DataFrames (no DB hit) ─────────────────────────
    with tempfile.TemporaryDirectory() as td:
        cfg12 = _base_config(Path(td))
        r12 = run_daily_retrain(config=cfg12, prices_df=prices, macro_df=macro)
        _check(
            "run_daily_retrain — accepts prices_df/macro_df kwargs",
            r12.success and r12.n_training_pairs > 0,
            f"success={r12.success} pairs={r12.n_training_pairs}",
        )

    # ── 13. commodities filtered to those in prices ───────────────────────────
    with tempfile.TemporaryDirectory() as td:
        cfg13 = RetrainConfig(
            commodities=["WTI Crude Oil", "DOES NOT EXIST", "Gold (COMEX)"],
            max_depth=3, min_samples_leaf=1,
            save_path=Path(td) / "meta_predictor.pkl",
            db_path=Path(td) / "test.db",
        )
        r13 = run_daily_retrain(config=cfg13, prices_df=prices, macro_df=macro)
        _check(
            "run_daily_retrain — invalid commodity filtered out silently",
            r13.success,
            f"error={r13.error!r}",
        )
        _check(
            "run_daily_retrain — n_commodities ≤ 2 (filtered to valid)",
            r13.n_commodities <= 2,
            f"n_commodities={r13.n_commodities}",
        )

    # ── 14. RetrainSummary — failure state ────────────────────────────────────
    s14 = RetrainSummary(error="something went wrong", success=False)
    _check(
        "RetrainSummary — failure pretty() shows FAILED",
        "FAILED" in s14.pretty(),
        s14.pretty()[:80],
    )

    print()
    print("=" * 60)
    print("ALL 14 ASSERTIONS PASSED")
    print("Phase 5 Part 1 complete: daily retraining pipeline ready.")
    print("Run with:  python -m models.daily_retrain")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
