"""
Database schema — SQLAlchemy ORM models.

HOW THIS WORKS:
  SQLAlchemy's ORM (Object-Relational Mapper) lets you define database tables
  as Python classes. Each class = one table. Each class attribute = one column.
  SQLAlchemy translates these into the correct SQL for whatever database you use
  (SQLite now, PostgreSQL later) — you never write CREATE TABLE by hand.

SCHEMA DESIGN:
  We use two tables in a classic "normalised" relational design:

  ┌────────────────────┐       ┌───────────────────────────────┐
  │   commodities      │       │        price_history           │
  │────────────────────│       │───────────────────────────────│
  │ id  (PK)           │◄──────│ commodity_id  (FK)            │
  │ name               │  1:N  │ date                          │
  │ ticker             │       │ open / high / low / close     │
  │ sector             │       │ volume / interval             │
  │ unit               │       │ adjusted_close                │
  │ instrument_type    │       │ adjustment_factor             │
  └────────────────────┘       └───────────────────────────────┘
           │
           │ 1:N   ┌───────────────────────────────┐
           └──────►│        aligned_prices          │
                   │───────────────────────────────│
                   │ commodity_id  (FK)             │
                   │ date  (canonical calendar)     │
                   │ adjusted_close  (ffilled)      │
                   │ is_filled                      │
                   └───────────────────────────────┘

  ┌────────────────────┐
  │   ingestion_log    │  ← one row per ticker per run; audit trail for failures
  │────────────────────│
  │ run_id  (UUID)     │
  │ started_at (UTC)   │
  │ ticker / name      │
  │ status             │  'ok' | 'empty' | 'error'
  │ rows_inserted      │
  │ rows_skipped       │
  │ error_msg          │
  │ duration_ms        │
  └────────────────────┘

  Database views (no storage cost — run on demand):
    v_futures_aligned  → aligned_prices WHERE instrument_type = 'futures'
    v_proxies_aligned  → aligned_prices WHERE instrument_type IN ('etf_proxy','equity_proxy')
    v_crypto_aligned   → aligned_prices WHERE instrument_type = 'crypto'
    v_all_typed        → aligned_prices with instrument_type column attached

  "Normalised" means we store each commodity's name/sector/unit ONCE in
  `commodities`, then reference it by ID in `price_history`. This avoids
  repeating "WTI Crude Oil" / "Energy" / "USD/bbl" in thousands of rows.
  One ID integer is far cheaper to store and join on.
"""

from sqlalchemy import (
    Column, Integer, String, Float, BigInteger, Boolean,
    Date, DateTime, UniqueConstraint, ForeignKey, Index, Text
)
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime, timezone


class Base(DeclarativeBase):
    """
    All ORM models inherit from Base.
    Base keeps a registry of every model so SQLAlchemy can create all
    tables in one call: Base.metadata.create_all(engine)
    """
    pass


class Commodity(Base):
    """
    Reference table — one row per tracked commodity.
    Think of this as a lookup / dimension table.
    """
    __tablename__ = "commodities"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    name            = Column(String(100), nullable=False, unique=True)   # "WTI Crude Oil"
    ticker          = Column(String(20),  nullable=False, unique=True)   # "CL=F"
    sector          = Column(String(50),  nullable=False)                # "Energy"
    unit            = Column(String(30),  nullable=False)                # "USD/bbl"
    # Instrument classification — populated by pipeline/layer.py.
    # Values: 'futures' | 'etf_proxy' | 'equity_proxy' | 'crypto'
    # Critical for model training: futures prices reflect commodity fundamentals;
    # equity/ETF proxies carry equity market beta that pollutes cross-asset analysis.
    instrument_type = Column(String(20),  nullable=True)

    # Relationship: one Commodity → many PriceHistory rows
    # `back_populates` creates a two-way link so you can do:
    #   commodity.prices      → list of all its price rows
    #   price_row.commodity   → the parent commodity object
    prices = relationship("PriceHistory", back_populates="commodity",
                          cascade="all, delete-orphan")
    aligned_prices = relationship("AlignedPrice", back_populates="commodity",
                                  cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Commodity {self.name} ({self.ticker})>"


class PriceHistory(Base):
    """
    Time-series OHLCV (Open / High / Low / Close / Volume) data.
    Each row is one day's candle for one commodity.

    OHLCV explained:
      Open   — price at market open
      High   — highest price during the day
      Low    — lowest price during the day
      Close  — price at market close  ← most commonly used
      Volume — number of contracts traded (0 for some commodities)
    """
    __tablename__ = "price_history"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    commodity_id = Column(Integer, ForeignKey("commodities.id"), nullable=False)
    date         = Column(Date,    nullable=False)
    open         = Column(Float)
    high         = Column(Float)
    low          = Column(Float)
    close        = Column(Float,   nullable=False)
    volume       = Column(BigInteger, default=0)
    interval     = Column(String(10), nullable=False, default="1d")  # "1d" or "1wk"
    ingested_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))  # when WE pulled it (UTC)

    # Roll-adjusted prices (populated by pipeline/roll_adjust.py).
    # `close` always holds the raw price Yahoo Finance gave us.
    # `adjusted_close` holds the proportionally back-adjusted price that
    # eliminates contract-roll discontinuities — use this for any return
    # calculation, model training, or backtesting.
    # `adjustment_factor` is the cumulative multiplier applied to reach
    # adjusted_close from close. Factor = 1.0 means no adjustment was made.
    adjusted_close    = Column(Float, nullable=True)
    adjustment_factor = Column(Float, nullable=True, default=1.0)

    commodity = relationship("Commodity", back_populates="prices")

    # UNIQUE constraint: one row per (commodity, date, interval).
    # This prevents duplicate rows if the ingestion script runs twice on the same day.
    # The database itself enforces this — no duplicates can ever sneak in.
    __table_args__ = (
        UniqueConstraint("commodity_id", "date", "interval",
                         name="uq_commodity_date_interval"),
        # Index speeds up the most common query: "give me all rows for ticker X
        # between date A and date B" — without an index, the DB scans every row.
        Index("ix_price_history_commodity_date", "commodity_id", "date"),
    )

    def __repr__(self):
        return f"<PriceHistory {self.commodity_id} {self.date} close={self.close}>"


class AlignedPrice(Base):
    """
    Time-aligned price series — every instrument on the same canonical calendar.

    WHY THIS TABLE EXISTS:
      price_history has mismatched date coverage across instruments:
        - Bitcoin trades 7 days/week  (~1,827 rows over 5 years)
        - Futures trade ~252 days/year (~1,260 rows)
        - ETFs follow US market hours
        - Some contracts have mid-week holidays others don't

      You cannot build a valid correlation matrix, or feed multiple instruments
      into a model, when their date indices don't line up. Pairing row 500 of
      Bitcoin with row 500 of WTI means pairing observations from different dates.

    HOW IT WORKS:
      1. The canonical calendar is derived empirically from the futures instruments
         themselves: any date where at least half of the 28 direct futures contracts
         have a record is treated as a US exchange trading day.
      2. Every instrument is reindexed to this calendar.
      3. Days with no observation (weekends for futures, ETF holidays, Bitcoin
         weekend days collapsed into Monday) are forward-filled — the last known
         price is carried forward.
      4. is_filled=True marks rows that were filled rather than observed, so
         downstream code can weight or exclude them if needed.

    WHAT TO USE IT FOR:
      - Correlation matrices (requires identical date index)
      - Multi-instrument model training (LSTM, XGBoost require aligned tensors)
      - Any cross-asset spread or ratio calculation
    """
    __tablename__ = "aligned_prices"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    commodity_id = Column(Integer, ForeignKey("commodities.id"), nullable=False)
    date         = Column(Date,    nullable=False)
    adjusted_close = Column(Float, nullable=False)   # forward-filled adjusted close
    is_filled    = Column(Boolean, default=False)    # True = no real observation; carried forward

    commodity = relationship("Commodity", back_populates="aligned_prices")

    __table_args__ = (
        UniqueConstraint("commodity_id", "date", name="uq_aligned_commodity_date"),
        Index("ix_aligned_commodity_date", "commodity_id", "date"),
    )


class CorrelationSnapshot(Base):
    """
    Daily rolling pairwise Pearson correlations between commodity pairs.

    Populated by models/cross_asset.py::store_correlation_snapshot().
    One row per (commodity_a, commodity_b, date).  The values come from
    a 21-day rolling window over aligned_prices so every instrument is on
    the same calendar before the correlation is computed.

    Used by:
      - Cross-asset meta-predictor: cluster-level prior propagation
      - Forecast consistency checker: flag divergent correlated pairs
      - Portfolio optimizer: covariance matrix construction
    """
    __tablename__ = "correlation_snapshots"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    date        = Column(Date,    nullable=False)
    commodity_a = Column(String(100), nullable=False)
    commodity_b = Column(String(100), nullable=False)
    correlation = Column(Float,   nullable=False)   # Pearson ∈ [−1, +1]
    window_days = Column(Integer, nullable=False, default=21)

    __table_args__ = (
        UniqueConstraint("date", "commodity_a", "commodity_b",
                         name="uq_corr_date_pair"),
        Index("ix_corr_date", "date"),
        Index("ix_corr_pair", "commodity_a", "commodity_b"),
    )

    def __repr__(self):
        return (
            f"<CorrelationSnapshot {self.date} "
            f"{self.commodity_a}↔{self.commodity_b} r={self.correlation:.3f}>"
        )


class ForecastLog(Base):
    """
    Immutable record of every forecast produced by each model.

    Written once per (forecast_date, commodity, model_name).  When the
    actual return is realised the following day, actual_return and error
    are back-filled via a separate UPDATE pass (see cross_asset.realize_forecasts).

    Why this table?
      ic_log aggregates IC across a backtest window; it cannot answer
      "what was Brent's XGBoost forecast on 2026-04-15, and how wrong was it
      in a Bull regime?"  ForecastLog stores every individual prediction so
      you can compute regime-conditional IC, rolling accuracy leaderboards,
      and relative-IC comparisons between correlated commodities.

    Used by:
      - Regime-conditional IC: segment accuracy by VIX regime
      - Relative IC: "Brent XGBoost IC 26% higher than WTI in Bull regime"
      - Meta-predictor training labels
      - Live accuracy leaderboard in dashboard
    """
    __tablename__ = "forecast_log"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    forecast_date   = Column(Date,    nullable=False)
    commodity       = Column(String(100), nullable=False)
    model_name      = Column(String(50),  nullable=False)   # "xgboost", "arima", "lstm" …
    tier            = Column(String(30),  nullable=False)   # "ml", "statistical", "deep", "quantum"
    regime          = Column(String(30),  nullable=True)    # "bull", "bear", "high_vol", "crisis"
    forecast_return = Column(Float, nullable=False)         # predicted next-day log-return
    actual_return   = Column(Float, nullable=True)          # realised return (filled next day)
    error           = Column(Float, nullable=True)          # abs(forecast − actual)
    confidence      = Column(Float, nullable=True)          # model confidence ∈ [0, 1]
    inserted_at     = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("forecast_date", "commodity", "model_name",
                         name="uq_forecast_date_commodity_model"),
        Index("ix_forecast_date",      "forecast_date"),
        Index("ix_forecast_commodity", "commodity", "forecast_date"),
        Index("ix_forecast_model",     "model_name", "forecast_date"),
        Index("ix_forecast_regime",    "regime",     "forecast_date"),
    )

    def __repr__(self):
        return (
            f"<ForecastLog {self.forecast_date} {self.commodity} "
            f"{self.model_name} fc={self.forecast_return:.4f}>"
        )


class PriceValidationLog(Base):
    """
    Audit trail for every price anomaly detected during ingestion.

    One row per anomalous price point. Written by pipeline/price_validator.py
    before any data enters price_history, so the raw incorrect value is always
    preserved here even after the corrected value is stored in price_history.

    reason_code values (from pipeline/price_validator.py):
      'scaling_error'              — systematic unit mismatch (e.g., cents vs dollars)
      'absolute_sanity_violation'  — price outside the hard floor/ceiling for this ticker
      'outlier_spike'              — extreme single-period return or rolling z-score breach
      'rollover_artifact'          — likely contract-roll discontinuity; kept for roll_adjust
      'missing_contract_continuity'— gap preventing proper back-adjustment

    action values:
      'rescaled'     — scale factor applied; row inserted with corrected_close
      'interpolated' — replaced with linear interpolation from neighbours
      'excluded'     — row dropped; NOT inserted into price_history
      'quarantined'  — logged but inserted as-is; roll_adjust.py may fix it
    """
    __tablename__ = "price_validation_log"

    id              = Column(Integer,    primary_key=True, autoincrement=True)
    run_id          = Column(String(36), nullable=False)
    logged_at       = Column(DateTime,   default=lambda: datetime.now(timezone.utc))
    ticker          = Column(String(20),  nullable=False)
    name            = Column(String(100), nullable=False)
    date            = Column(Date,        nullable=False)
    raw_close       = Column(Float,       nullable=False)
    corrected_close = Column(Float,       nullable=True)   # None if quarantined/excluded
    reason_code     = Column(String(50),  nullable=False)
    action          = Column(String(20),  nullable=False)
    details         = Column(String(500), nullable=True)

    __table_args__ = (
        Index("ix_pvlog_ticker_date",  "ticker", "date"),
        Index("ix_pvlog_run_id",       "run_id"),
        Index("ix_pvlog_reason_code",  "reason_code"),
        Index("ix_pvlog_logged_at",    "logged_at"),
    )

    def __repr__(self):
        return (
            f"<PriceValidationLog {self.ticker} {self.date} "
            f"reason={self.reason_code} action={self.action}>"
        )


class IngestionLog(Base):
    """
    One row per commodity per ingestion run.

    Gives a full audit trail: which tickers succeeded, which returned empty,
    and which raised exceptions — along with row counts and the error message.
    Without this table, a yfinance failure for a ticker leaves no trace; the
    ticker just silently disappears from new data until someone notices.

    run_id ties all rows from a single run_ingestion() call together so you can
    quickly see whether an entire run failed or just individual tickers.

    status values:
      'ok'    — rows were fetched and written (inserted > 0 or all duplicates)
      'empty' — yfinance returned an empty DataFrame for this ticker
      'error' — an exception was raised fetching or writing this ticker
    """
    __tablename__ = "ingestion_log"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    run_id        = Column(String(36), nullable=False)   # UUID4 shared across one run_ingestion() call
    started_at    = Column(DateTime, nullable=False)     # UTC — when this ticker's fetch began
    ticker        = Column(String(20),  nullable=False)
    name          = Column(String(100), nullable=False)
    status        = Column(String(20),  nullable=False)  # 'ok' | 'empty' | 'error'
    rows_inserted = Column(Integer, default=0)
    rows_skipped  = Column(Integer, default=0)
    error_msg     = Column(String(500), nullable=True)
    duration_ms   = Column(Integer, nullable=True)       # fetch + write time in milliseconds

    __table_args__ = (
        Index("ix_ingestion_log_run_id",    "run_id"),
        Index("ix_ingestion_log_started_at", "started_at"),
    )

    def __repr__(self):
        return f"<IngestionLog {self.ticker} {self.status} @ {self.started_at}>"


class TriggerEvent(Base):
    """
    One row per (family, trigger_date). Dedup key is UNIQUE(family, trigger_date).
    Re-firing the same trigger on the same day overwrites the prior row.
    Written by models/trigger_log.py.
    """
    __tablename__ = "trigger_events"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    detected_at          = Column(String(50),  nullable=False)   # ISO 8601 timestamp
    trigger_date         = Column(String(10),  nullable=False)   # YYYY-MM-DD
    family               = Column(String(100), nullable=False)   # trigger family name
    strength             = Column(Float,       nullable=False)   # [0, 1]
    rationale            = Column(Text,        nullable=True)
    affected_commodities = Column(Text,        nullable=True)    # JSON array
    trigger_metadata     = Column("metadata", Text, nullable=True)  # JSON object (col name: metadata)
    inserted_at          = Column(String(50),  nullable=False)   # UTC ISO 8601

    __table_args__ = (
        UniqueConstraint("family", "trigger_date", name="uq_trigger_family_date"),
        Index("ix_trigger_events_family_date", "family", "trigger_date"),
    )

    def __repr__(self):
        return f"<TriggerEvent {self.family} {self.trigger_date} strength={self.strength}>"


class ICLog(Base):
    """
    Spearman IC (Information Coefficient) per (commodity, tier) per run.
    UNIQUE(computed_at, commodity, tier) — one row per computation timestamp.
    Written by models/ic_tracker.py.
    """
    __tablename__ = "ic_log"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    computed_at  = Column(String(50),  nullable=False)   # UTC ISO timestamp
    commodity    = Column(String(100), nullable=False)
    tier         = Column(String(50),  nullable=False)   # "ml", "statistical", "deep", "quantum"
    ic_value     = Column(Float,       nullable=False)   # Spearman ∈ [−1, +1]
    n_obs        = Column(Integer,     nullable=False)
    window_start = Column(String(10),  nullable=True)    # YYYY-MM-DD
    window_end   = Column(String(10),  nullable=True)    # YYYY-MM-DD
    regime       = Column(String(30),  nullable=True)    # "bull", "bear", etc.
    inserted_at  = Column(String(50),  nullable=False)

    __table_args__ = (
        UniqueConstraint("computed_at", "commodity", "tier", name="uq_ic_log_computed_commodity_tier"),
        Index("ix_ic_log_tier_date",  "tier",      "computed_at"),
        Index("ix_ic_log_commodity",  "commodity", "computed_at"),
        Index("ix_ic_log_regime",     "regime",    "computed_at"),
    )

    def __repr__(self):
        return f"<ICLog {self.commodity} {self.tier} ic={self.ic_value:.4f} @ {self.computed_at[:10]}>"


class ModelTrainingLog(Base):
    """
    Append-only audit trail — one row per daily_retrain run.
    No UNIQUE constraint (every run appends a new row).
    Written by models/daily_retrain.py.
    """
    __tablename__ = "model_training_log"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    retrained_at      = Column(String(50),  nullable=False)
    n_commodities     = Column(Integer,     nullable=False)
    n_training_pairs  = Column(Integer,     nullable=False)
    tier_distribution = Column(Text,        nullable=True)   # JSON {tier: count}
    tree_n_leaves     = Column(Integer,     nullable=True)
    top_feature       = Column(String(100), nullable=True)
    save_path         = Column(String(500), nullable=True)
    error             = Column(Text,        nullable=True)
    config_json       = Column(Text,        nullable=True)
    inserted_at       = Column(String(50),  nullable=False)

    __table_args__ = (
        Index("ix_training_log_retrained_at", "retrained_at"),
    )

    def __repr__(self):
        return f"<ModelTrainingLog {self.retrained_at[:10]} n_pairs={self.n_training_pairs}>"


class ThresholdConfig(Base):
    """
    Append-only threshold tuning results — one row per (family, run).
    No UNIQUE constraint; load_optimal_thresholds() queries MAX(evaluated_at) per family.
    Written by models/threshold_tuner.py.
    """
    __tablename__ = "threshold_config"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    family                = Column(String(100), nullable=False)
    optimal_threshold     = Column(Float,       nullable=False)
    best_ic               = Column(Float,       nullable=True)
    continuous_ic         = Column(Float,       nullable=True)
    n_events_total        = Column(Integer,     nullable=True)
    n_events_at_threshold = Column(Integer,     nullable=True)
    lookback_days         = Column(Integer,     nullable=True)
    forward_days          = Column(Integer,     nullable=True)
    grid_results          = Column(Text,        nullable=True)   # JSON {threshold: ic}
    evaluated_at          = Column(String(50),  nullable=False)

    __table_args__ = (
        Index("ix_threshold_config_family", "family", "evaluated_at"),
    )

    def __repr__(self):
        return f"<ThresholdConfig {self.family} opt={self.optimal_threshold} @ {self.evaluated_at[:10]}>"


class CascadeValidationLog(Base):
    """
    Per-sector detail rows from cascade validation runs.
    UNIQUE(run_at, shock_date, shock_family, sector).
    Written by models/cascade_validator.py.
    """
    __tablename__ = "cascade_validation_log"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    run_at           = Column(String(50),  nullable=False)
    shock_date       = Column(String(10),  nullable=False)
    shock_family     = Column(String(100), nullable=False)
    shock_strength   = Column(Float,       nullable=False)
    sector           = Column(String(50),  nullable=False)
    cascade_fcast    = Column(Float,       nullable=True)
    baseline_fcast   = Column(Float,       nullable=True)
    actual_5d        = Column(Float,       nullable=True)
    actual_10d       = Column(Float,       nullable=True)
    cascade_correct  = Column(Integer,     nullable=True)   # 0 or 1
    baseline_correct = Column(Integer,     nullable=True)   # 0 or 1
    mae_cascade      = Column(Float,       nullable=True)
    mae_baseline     = Column(Float,       nullable=True)
    lift             = Column(Float,       nullable=True)
    lag_confirmed    = Column(Integer,     nullable=True)   # 0 or 1
    inserted_at      = Column(String(50),  nullable=False)

    __table_args__ = (
        UniqueConstraint("run_at", "shock_date", "shock_family", "sector",
                         name="uq_cvl_run_shock_sector"),
        Index("ix_cvl_run_at", "run_at"),
        Index("ix_cvl_shock",  "shock_date", "shock_family"),
        Index("ix_cvl_sector", "sector",     "run_at"),
    )

    def __repr__(self):
        return f"<CascadeValidationLog {self.sector} {self.shock_date} correct={self.cascade_correct}>"


class CascadeValidationSummary(Base):
    """
    One summary row per validation run (UNIQUE on run_at).
    Written by models/cascade_validator.py alongside CascadeValidationLog rows.
    """
    __tablename__ = "cascade_validation_summary"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    run_at               = Column(String(50),  nullable=False, unique=True)
    shock_window_days    = Column(Integer,     nullable=True)
    n_shock_events       = Column(Integer,     nullable=True)
    sector_accuracy_json = Column(Text,        nullable=True)   # JSON {sector: float}
    sector_lift_json     = Column(Text,        nullable=True)   # JSON {sector: float}
    lag_confirmed_pct    = Column(Float,       nullable=True)
    avg_lag1_corr        = Column(Float,       nullable=True)
    overall_status       = Column(String(50),  nullable=True)   # "healthy"|"degraded"|...
    flags_json           = Column(Text,        nullable=True)   # JSON list of strings
    inserted_at          = Column(String(50),  nullable=False)

    __table_args__ = (
        Index("ix_cvs_run_at", "run_at"),
    )

    def __repr__(self):
        return f"<CascadeValidationSummary {self.run_at[:10]} status={self.overall_status}>"


class CascadeForecast(Base):
    """
    Per-commodity cascade forecast rows.
    UNIQUE(forecast_date, commodity) — one row per commodity per day.
    Written by models/cascade_orchestrator.py.
    """
    __tablename__ = "cascade_forecasts"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    forecast_date       = Column(String(10),  nullable=False)
    commodity           = Column(String(100), nullable=False)
    sector              = Column(String(50),  nullable=False)
    sector_rank         = Column(Integer,     nullable=False)
    regime              = Column(String(30),  nullable=True)
    base_forecast       = Column(Float,       nullable=False)
    macro_adjustment    = Column(Float,       nullable=False, default=0.0)
    upstream_adjustment = Column(Float,       nullable=False, default=0.0)
    final_forecast      = Column(Float,       nullable=False)
    confidence          = Column(Float,       nullable=True)
    macro_detail        = Column(Text,        nullable=True)    # JSON
    upstream_detail     = Column(Text,        nullable=True)    # JSON
    inserted_at         = Column(String(50),  nullable=False)

    __table_args__ = (
        UniqueConstraint("forecast_date", "commodity", name="uq_cascade_date_commodity"),
        Index("ix_cascade_date",   "forecast_date"),
        Index("ix_cascade_sector", "sector", "forecast_date"),
    )

    def __repr__(self):
        return f"<CascadeForecast {self.commodity} {self.forecast_date} fc={self.final_forecast:.4f}>"


class CausalMonitoringLog(Base):
    """
    Append-only log of causal architecture health checks (Granger + route shifts
    + cascade validation). One row per daily_retrain run.
    No UNIQUE constraint — each run appends.
    Written by models/daily_retrain.py.
    """
    __tablename__ = "causal_monitoring_log"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    logged_at               = Column(String(50),  nullable=False)
    run_type                = Column(String(20),  nullable=False, default="daily")
    granger_refreshed       = Column(Integer,     nullable=False, default=0)   # 0 or 1
    granger_summary_json    = Column(Text,        nullable=True)
    route_coefficients_json = Column(Text,        nullable=True)
    route_shift_pct_json    = Column(Text,        nullable=True)
    sector_ic_json          = Column(Text,        nullable=True)
    cascade_status          = Column(String(50),  nullable=True)
    cascade_accuracy_json   = Column(Text,        nullable=True)
    cascade_n_events        = Column(Integer,     nullable=True, default=0)
    alerts_json             = Column(Text,        nullable=True)
    n_alerts                = Column(Integer,     nullable=False, default=0)

    __table_args__ = (
        Index("ix_causal_monitoring_logged_at", "logged_at"),
    )

    def __repr__(self):
        return f"<CausalMonitoringLog {self.logged_at[:10]} n_alerts={self.n_alerts}>"
