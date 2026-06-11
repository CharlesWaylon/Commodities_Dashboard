"""
Tests for models/daily_retrain.py — Phase 5 Part 1.

Updated 2026-06-11 for the SQLite → SQLAlchemy/Postgres migration (2026-05-11):
RetrainConfig no longer takes db_path; persistence goes through the shared
Postgres engine. Non-dry-run runs now also trigger post-retrain steps
(correlation snapshots, IC/forecast logging, macro-route refresh, cascade,
causal monitoring, training log) that write to the LIVE database and to
models/macro_routes.pkl — so this suite stubs every shared-state writer for
the duration of each run (see _isolated_side_effects), and exercises the real
Postgres roundtrip only via a sentinel row that is deleted afterwards.

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
10. _persist_training_log / recent_training_runs — roundtrip through Postgres
    (sentinel row, cleaned up afterwards)
11. recent_training_runs — DB error returns empty DataFrame
12. run_daily_retrain — accepts pre-loaded prices_df / macro_df (no DB hit)
13. run_daily_retrain — commodities filtered to those present in prices
14. RetrainSummary — success=False when error is set
"""

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
        dry_run=dry_run,
        skip_causal_monitoring=True,
    )


@contextmanager
def _isolated_side_effects():
    """
    Stub every shared-state writer that a non-dry-run retrain reaches, so the
    suite never touches the live Postgres tables (correlation_snapshots,
    ic_log, forecast_log, model_training_log, cascade_forecasts) or rebuilds
    models/macro_routes.pkl. daily_retrain imports these inside the function
    body at call time, so patching the source-module attributes is sufficient.
    """
    import models.cascade_orchestrator as cascade_orchestrator
    import models.cross_asset as cross_asset
    import models.daily_retrain as dr
    import models.ic_tracker as ic_tracker
    import models.macro_router as macro_router

    patches = [
        (cross_asset, "store_correlation_snapshot", lambda *a, **k: 0),
        (cross_asset, "store_covariance_snapshot",  lambda *a, **k: 0),
        (cross_asset, "log_forecasts",              lambda *a, **k: 0),
        (cross_asset, "realize_forecasts",          lambda *a, **k: 0),
        (cross_asset, "check_forecast_consistency", lambda *a, **k: []),
        (cross_asset, "load_correlation_matrix",    lambda *a, **k: None),
        (ic_tracker,  "log_ic_scores",              lambda *a, **k: 0),
        (macro_router, "_is_stale",                 lambda *a, **k: False),
        (cascade_orchestrator, "run_cascade",
         lambda *a, **k: SimpleNamespace(
             commodities=[], regime="test-stub", n_written=0,
             success=True, errors={},
         )),
        (dr, "_persist_training_log",               lambda summary: None),
    ]
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    try:
        for mod, name, stub in patches:
            setattr(mod, name, stub)
        yield
    finally:
        for mod, name, original in saved:
            setattr(mod, name, original)


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
    with tempfile.TemporaryDirectory() as td, _isolated_side_effects():
        cfg3 = _base_config(Path(td))
        result = run_daily_retrain(config=cfg3, prices_df=prices, macro_df=macro)
        _check(
            "run_daily_retrain — success=True",
            result.success,
            f"error={result.error!r}",
        )

    # ── 4. tier_distribution populated ────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td, _isolated_side_effects():
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
    with tempfile.TemporaryDirectory() as td, _isolated_side_effects():
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
    with tempfile.TemporaryDirectory() as td, _isolated_side_effects():
        pkl_path = Path(td) / "meta_predictor.pkl"
        cfg6 = RetrainConfig(
            commodities=["WTI Crude Oil", "Gold (COMEX)", "Corn (CBOT)"],
            max_depth=3, min_samples_leaf=1,
            save_path=pkl_path,
            dry_run=False,
            skip_causal_monitoring=True,
        )
        run_daily_retrain(config=cfg6, prices_df=prices, macro_df=macro)
        _check(
            "run_daily_retrain — pkl written to disk",
            pkl_path.exists(),
            f"expected {pkl_path}",
        )

    # ── 7. dry_run does NOT write pkl ──────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td, _isolated_side_effects():
        pkl_path = Path(td) / "meta_predictor.pkl"
        cfg7 = RetrainConfig(
            commodities=["WTI Crude Oil", "Gold (COMEX)", "Corn (CBOT)"],
            max_depth=3, min_samples_leaf=1,
            save_path=pkl_path,
            dry_run=True,
            skip_causal_monitoring=True,
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
    with tempfile.TemporaryDirectory() as td, _isolated_side_effects():
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
    with tempfile.TemporaryDirectory() as td, _isolated_side_effects():
        pkl_path = Path(td) / "meta_predictor.pkl"
        cfg9 = RetrainConfig(
            commodities=["WTI Crude Oil"],
            max_depth=3, min_samples_leaf=1,
            save_path=pkl_path,
            skip_causal_monitoring=True,
        )
        r9 = run_daily_retrain(config=cfg9, prices_df=short_prices, macro_df=short_macro)
        _check(
            "run_daily_retrain — too few pairs → success=False, no pkl",
            not r9.success and not pkl_path.exists(),
            f"success={r9.success} pkl_exists={pkl_path.exists()} error={r9.error!r}",
        )

    # ── 10. _persist_training_log / recent_training_runs roundtrip ────────────
    # Real Postgres roundtrip: write one sentinel row (far-future timestamp so
    # it sorts first in recent_training_runs), read it back, then delete it.
    from sqlalchemy import text as _sql_text
    from database.db import get_engine as _get_engine
    sentinel_ts = "2099-01-01T00:00:00+00:00"
    try:
        s10 = RetrainSummary(
            retrained_at=sentinel_ts,
            n_commodities=3,
            n_training_pairs=60,
            tier_distribution={"statistical": 30, "ml": 30},
            tree_n_leaves=8,
            top_feature="vix",
            save_path="/tmp/test_sentinel/meta_predictor.pkl",
            error="",
            config=RetrainConfig(),
            success=True,
        )
        _persist_training_log(s10)
        df10 = recent_training_runs(n=1)
        _check(
            "_persist_training_log / recent_training_runs — sentinel row returned",
            len(df10) == 1 and df10.iloc[0]["retrained_at"] == sentinel_ts,
            f"rows={len(df10)}",
        )
        _check(
            "recent_training_runs — n_training_pairs correct",
            int(df10.iloc[0]["n_training_pairs"]) == 60,
            f"got {df10.iloc[0]['n_training_pairs'] if len(df10) else 'no rows'}",
        )
        _check(
            "recent_training_runs — top_feature correct",
            df10.iloc[0]["top_feature"] == "vix",
            f"got {df10.iloc[0]['top_feature'] if len(df10) else 'no rows'}",
        )
    finally:
        with _get_engine().connect() as conn:
            conn.execute(
                _sql_text("DELETE FROM model_training_log WHERE retrained_at = :ts"),
                {"ts": sentinel_ts},
            )
            conn.commit()

    # ── 11. recent_training_runs — DB error returns empty DataFrame ────────────
    import models.daily_retrain as dr

    def _broken_engine():
        raise RuntimeError("simulated DB outage")

    _orig_engine = dr.get_engine
    try:
        dr.get_engine = _broken_engine
        df11 = recent_training_runs(n=5)
    finally:
        dr.get_engine = _orig_engine
    _check(
        "recent_training_runs — DB error returns empty DataFrame",
        df11.empty,
        f"rows={len(df11)}",
    )

    # ── 12. accepts pre-loaded DataFrames (no DB hit) ─────────────────────────
    with tempfile.TemporaryDirectory() as td, _isolated_side_effects():
        cfg12 = _base_config(Path(td))
        r12 = run_daily_retrain(config=cfg12, prices_df=prices, macro_df=macro)
        _check(
            "run_daily_retrain — accepts prices_df/macro_df kwargs",
            r12.success and r12.n_training_pairs > 0,
            f"success={r12.success} pairs={r12.n_training_pairs}",
        )

    # ── 13. commodities filtered to those in prices ───────────────────────────
    with tempfile.TemporaryDirectory() as td, _isolated_side_effects():
        cfg13 = RetrainConfig(
            commodities=["WTI Crude Oil", "DOES NOT EXIST", "Gold (COMEX)"],
            max_depth=3, min_samples_leaf=1,
            save_path=Path(td) / "meta_predictor.pkl",
            skip_causal_monitoring=True,
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
