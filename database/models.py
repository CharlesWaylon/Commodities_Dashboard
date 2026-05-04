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
    Date, DateTime, UniqueConstraint, ForeignKey, Index
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
