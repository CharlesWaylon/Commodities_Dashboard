"""
Optuna-based sector hyperparameter tuner for XGBoost and Random Forest.

Groups the 41 commodities by sector (Energy / Metals / Agriculture / Livestock /
Digital) and runs one Optuna study per sector.  The objective maximises mean
Spearman IC across all commodities in the sector, averaged over the test split.

Results are persisted to  data/sector_params.json.  Both xgboost_shap.py and
random_forest.py automatically load that file on first use, falling back to the
domain-informed seeds in config.py when the file doesn't exist.

Usage
-----
    # Programmatic (call from a Jupyter cell or the dashboard)
    from models.ml.sector_tuner import SectorTuner
    from models.data_loader import load_price_matrix

    prices = load_price_matrix()
    tuner  = SectorTuner(prices)
    tuner.tune("xgb", n_trials=50)   # ~20-40 min depending on hardware
    tuner.tune("rf",  n_trials=30)
    tuner.save()                      # writes data/sector_params.json

    # CLI
    python -m models.ml.sector_tuner --model xgb --n-trials 50
    python -m models.ml.sector_tuner --model rf  --n-trials 30
    python -m models.ml.sector_tuner --model both --n-trials 40
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from models.config import (
    COMMODITY_SECTORS,
    MODELING_COMMODITIES,
    RANDOM_SEED,
    SECTOR_RF_DEFAULTS,
    SECTOR_XGB_DEFAULTS,
    TEST_FRACTION,
)
from models.features import build_feature_matrix, build_target

logger = logging.getLogger(__name__)

SECTOR_PARAMS_PATH = Path(__file__).resolve().parents[2] / "data" / "sector_params.json"

_EARLY_STOPPING = 30
_VAL_FRACTION   = 0.15
_N_CROSS        = 15        # keep same feature-selection logic as XGBoostForecaster


# ── helpers ────────────────────────────────────────────────────────────────────

def _sector_commodities(sector: str, available: set[str]) -> list[str]:
    return [
        name for name, s in COMMODITY_SECTORS.items()
        if s == sector and name in available
    ]


def _xgb_ic_for_commodity(
    params: dict,
    feat_df: pd.DataFrame,
    commodity: str,
    target: pd.Series,
) -> float:
    """Fit XGBoost with given params, return Spearman IC on the test split."""
    combined = pd.concat([feat_df, target], axis=1).dropna()
    X_df = combined.drop(columns=[target.name])
    y    = combined[target.name]

    # Replicate feature-selection logic from XGBoostForecaster.fit()
    own_prefix = f"{commodity}_"
    own_cols   = [c for c in X_df.columns if c.startswith(own_prefix)]
    cross_cols = [c for c in X_df.columns if not c.startswith(own_prefix)]
    train_cut  = int(len(X_df) * (1 - TEST_FRACTION))
    cross_corr = (
        X_df[cross_cols].iloc[:train_cut]
        .corrwith(y.iloc[:train_cut])
        .abs()
    )
    top_cross = cross_corr.nlargest(_N_CROSS).index.tolist()
    X_df = X_df[own_cols + top_cross]

    split = int(len(X_df) * (1 - TEST_FRACTION))
    X_tr, y_tr   = X_df.iloc[:split], y.iloc[:split]
    X_te, y_te   = X_df.iloc[split:], y.iloc[split:]

    val_cut = int(len(X_tr) * (1 - _VAL_FRACTION))
    X_fit, X_val = X_tr.iloc[:val_cut], X_tr.iloc[val_cut:]
    y_fit, y_val = y_tr.iloc[:val_cut], y_tr.iloc[val_cut:]

    scaler = StandardScaler()
    X_fit_sc = scaler.fit_transform(X_fit)
    X_val_sc = scaler.transform(X_val)
    X_te_sc  = scaler.transform(X_te)

    model = xgb.XGBRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.set_params(early_stopping_rounds=_EARLY_STOPPING)
        model.fit(X_fit_sc, y_fit, eval_set=[(X_val_sc, y_val)], verbose=False)

    preds = model.predict(X_te_sc)
    ic, _ = spearmanr(preds, y_te.values)
    return float(ic) if not np.isnan(ic) else 0.0


def _rf_ic_for_commodity(
    params: dict,
    feat_df: pd.DataFrame,
    target: pd.Series,
) -> float:
    """Fit Random Forest with given params, return Spearman IC on the test split."""
    combined = pd.concat([feat_df, target], axis=1).dropna()
    X = combined.drop(columns=[target.name]).values
    y = combined[target.name].values

    split = int(len(X) * (1 - TEST_FRACTION))
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    rf = RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr_sc, y_tr)

    preds = rf.predict(X_te_sc)
    ic, _ = spearmanr(preds, y_te)
    return float(ic) if not np.isnan(ic) else 0.0


# ── main class ─────────────────────────────────────────────────────────────────

class SectorTuner:
    """
    Optuna-based hyperparameter tuner that operates at the sector level.

    One Optuna study is run per sector.  The objective is the mean Spearman IC
    across all commodities in that sector that are present in `prices`.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price matrix (date × commodity display-name columns).
    """

    def __init__(self, prices: pd.DataFrame) -> None:
        if not HAS_OPTUNA:
            raise ImportError(
                "optuna is required for sector tuning. "
                "Install it with:  pip install optuna"
            )
        self.prices  = prices
        self.feat_df = build_feature_matrix(prices)
        self._available = set(prices.columns)
        self._best: dict[str, dict[str, dict]] = {"xgb": {}, "rf": {}}

    # ── XGBoost study ──────────────────────────────────────────────────────────

    def _xgb_objective(self, sector: str, commodities: list[str]):
        feat_df = self.feat_df
        prices  = self.prices

        def objective(trial: optuna.Trial) -> float:
            params = dict(
                n_estimators      = trial.suggest_int("n_estimators", 300, 1000),
                learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
                max_depth         = trial.suggest_int("max_depth", 3, 7),
                subsample         = trial.suggest_float("subsample", 0.60, 0.95),
                colsample_bytree  = trial.suggest_float("colsample_bytree", 0.60, 0.95),
                min_child_weight  = trial.suggest_int("min_child_weight", 5, 25),
                reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 0.5),
                reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 3.0),
            )
            ics = []
            for name in commodities:
                if name not in prices.columns:
                    continue
                try:
                    target = build_target(prices, name)
                    ic = _xgb_ic_for_commodity(params, feat_df, name, target)
                    ics.append(ic)
                except Exception:
                    pass
            return float(np.mean(ics)) if ics else 0.0

        return objective

    # ── RF study ───────────────────────────────────────────────────────────────

    def _rf_objective(self, sector: str, commodities: list[str]):
        feat_df = self.feat_df
        prices  = self.prices

        def objective(trial: optuna.Trial) -> float:
            params = dict(
                n_estimators     = trial.suggest_int("n_estimators", 100, 600),
                max_depth        = trial.suggest_int("max_depth", 3, 10),
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 30),
                max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            )
            ics = []
            for name in commodities:
                if name not in prices.columns:
                    continue
                try:
                    target = build_target(prices, name)
                    ic = _rf_ic_for_commodity(params, feat_df, target)
                    ics.append(ic)
                except Exception:
                    pass
            return float(np.mean(ics)) if ics else 0.0

        return objective

    # ── public API ─────────────────────────────────────────────────────────────

    def tune(
        self,
        model: Literal["xgb", "rf"] = "xgb",
        n_trials: int = 40,
        sectors: list[str] | None = None,
        seed: int = RANDOM_SEED,
    ) -> dict[str, dict]:
        """
        Run one Optuna study per sector for the given model type.

        Parameters
        ----------
        model    : "xgb" or "rf"
        n_trials : number of Optuna trials per sector study
        sectors  : subset of sectors to tune; defaults to all five
        seed     : reproducibility seed for the sampler

        Returns
        -------
        dict  sector → best hyperparameters dict
        """
        all_sectors = sectors or ["energy", "metals", "agriculture", "livestock", "digital"]
        sampler = optuna.samplers.TPESampler(seed=seed)

        for sector in all_sectors:
            commodities = _sector_commodities(sector, self._available)
            if not commodities:
                logger.warning("No data for sector '%s' — skipping.", sector)
                continue

            logger.info(
                "Tuning %s | sector=%s | commodities=%d | trials=%d",
                model.upper(), sector, len(commodities), n_trials,
            )
            print(
                f"  [{model.upper()}] {sector:12s} → "
                f"{len(commodities)} commodities, {n_trials} trials …",
                flush=True,
            )

            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                study_name=f"{model}_{sector}",
            )

            if model == "xgb":
                objective = self._xgb_objective(sector, commodities)
                seed_params = SECTOR_XGB_DEFAULTS[sector]
            else:
                objective = self._rf_objective(sector, commodities)
                seed_params = SECTOR_RF_DEFAULTS[sector]

            # Seed the study with the domain-default params so Optuna starts smart
            study.enqueue_trial(seed_params)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best = study.best_params
            print(
                f"    best IC={study.best_value:.4f}  params={best}",
                flush=True,
            )
            self._best[model][sector] = best

        return self._best[model]

    def save(self, path: Path = SECTOR_PARAMS_PATH) -> None:
        """Persist tuned params to JSON. Merges with any existing file."""
        existing: dict = {}
        if path.exists():
            with path.open() as fh:
                existing = json.load(fh)

        for model in ("xgb", "rf"):
            if model not in existing:
                existing[model] = {}
            existing[model].update(self._best.get(model, {}))

        existing["tuned_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(existing, indent=2))
        print(f"Saved sector params → {path}", flush=True)

    @staticmethod
    def load(path: Path = SECTOR_PARAMS_PATH) -> dict:
        """Load persisted params.  Returns {} if file doesn't exist."""
        if not path.exists():
            return {}
        with path.open() as fh:
            return json.load(fh)


# ── module-level convenience ───────────────────────────────────────────────────

def load_sector_params(path: Path = SECTOR_PARAMS_PATH) -> dict:
    """Return the persisted sector params dict, or {} if not yet tuned."""
    return SectorTuner.load(path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run Optuna sector-tuning for XGBoost and/or Random Forest."
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "rf", "both"],
        default="xgb",
        help="Which model to tune (default: xgb)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Optuna trials per sector study (default: 40)",
    )
    parser.add_argument(
        "--sectors",
        nargs="+",
        default=None,
        help="Subset of sectors to tune, e.g. --sectors energy metals",
    )
    args = parser.parse_args()

    # Import here so CLI startup stays fast when optuna is missing
    from models.data_loader import load_price_matrix  # type: ignore

    print("Loading price matrix …", flush=True)
    prices = load_price_matrix()

    tuner = SectorTuner(prices)

    models_to_run = ["xgb", "rf"] if args.model == "both" else [args.model]
    for m in models_to_run:
        print(f"\n=== Tuning {m.upper()} ({args.n_trials} trials/sector) ===", flush=True)
        tuner.tune(m, n_trials=args.n_trials, sectors=args.sectors)

    tuner.save()


if __name__ == "__main__":
    _cli()
