"""
Provider adapters — turn each model family's native output into a
ScenarioBand. Phase 1: ARIMA and VAR (both expose closed-form prediction
intervals via statsmodels, so no resampling is required).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from models.scenarios.band import ScenarioBand
from models.scenarios.conformal import calibrate_from_model


def _z(q: float) -> float:
    """Standard-normal inverse CDF — vectorisable scalar wrapper."""
    return float(norm.ppf(q))


# ── ARIMA ─────────────────────────────────────────────────────────────────────


def arima_band(
    returns: pd.Series,
    commodity: str,
    horizon: int = 5,
    bear_q: float = 0.10,
    bull_q: float = 0.90,
) -> ScenarioBand:
    """
    Fit an ARIMA forecaster on ``returns`` and emit a ScenarioBand at the
    requested quantiles.
    """
    from models.statistical.arima import ARIMAForecaster

    fc = ARIMAForecaster(commodity=commodity)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc.fit(returns.dropna())

    # Asymmetric quantiles: under the Gaussian assumption baked into the
    # state-space Kalman filter, the h-step forecast is N(mean[h], se[h]²).
    # We can derive a bound at ANY quantile by mean[h] + z(q) · se[h] —
    # no need for the symmetric two-sided CI.
    pred = fc._model_fit.get_forecast(steps=horizon)
    mean = np.asarray(pred.predicted_mean, dtype=float)
    se = np.asarray(pred.se_mean, dtype=float)
    bear = mean + _z(bear_q) * se
    bull = mean + _z(bull_q) * se

    return ScenarioBand(
        commodity=commodity,
        source_model="ARIMA",
        asof=pd.Timestamp(returns.index[-1]),
        horizon=horizon,
        bear_q=bear_q,
        bull_q=bull_q,
        mean=mean,
        bear=bear,
        bull=bull,
        diagnostics={
            "order": fc.best_order,
            "aic": fc.best_aic,
            "seasonal": fc._is_seasonal(),
        },
    )


# ── VAR ───────────────────────────────────────────────────────────────────────


def var_band(
    prices: pd.DataFrame,
    commodity: str,
    horizon: int = 5,
    group: str = "energy",
    bear_q: float = 0.10,
    bull_q: float = 0.90,
) -> Optional[ScenarioBand]:
    """
    Fit a VAR on ``group`` (or commodity-specific group) and extract a
    ScenarioBand for ``commodity``.

    Returns None if the commodity is not in the requested group (caller can
    fall back to ARIMA).
    """
    from models.statistical.var_vecm import CommodityVAR, COMMODITY_GROUPS

    if group not in COMMODITY_GROUPS:
        return None
    if commodity not in COMMODITY_GROUPS[group]:
        return None
    if commodity not in prices.columns:
        return None

    var = CommodityVAR(group=group, use_returns=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            var.fit(prices)
        except Exception:
            return None

    if var._var_fit is None:
        return None

    lag = var._lag_order
    y_input = var._data.values[-lag:]
    # Mean is symmetric; we derive asymmetric quantiles ourselves from the
    # per-step forecast error covariance (mse[h] is an n×n matrix whose
    # diagonal gives each series' h-step prediction variance under the VAR's
    # Gaussian-innovation assumption).
    mean_arr = var._var_fit.forecast(y_input, steps=horizon)   # (h, n)
    mse_seq = var._var_fit.mse(horizon)                         # (h, n, n)
    se_arr = np.sqrt(np.array([np.diag(m) for m in mse_seq]))   # (h, n)

    cols = var._cols
    idx = cols.index(commodity)
    mean = mean_arr[:, idx]
    se = se_arr[:, idx]
    bear = mean + _z(bear_q) * se
    bull = mean + _z(bull_q) * se

    return ScenarioBand(
        commodity=commodity,
        source_model=f"VAR[{group}]",
        asof=pd.Timestamp(var._data.index[-1]),
        horizon=horizon,
        bear_q=bear_q,
        bull_q=bull_q,
        mean=mean,
        bear=bear,
        bull=bull,
        diagnostics={
            "lag_order": lag,
            "n_series": len(cols),
            "group": group,
        },
    )


# ── Conformal-wrapped ML providers ────────────────────────────────────────────


def _ml_conformal_band(
    *,
    fit_predict_fn,
    prices: pd.DataFrame,
    commodity: str,
    horizon: int,
    bear_q: float,
    bull_q: float,
    model_label: str,
) -> Optional[ScenarioBand]:
    """
    Shared scaffolding: fit a 1-step ML forecaster, calibrate its residuals
    on its holdout split via split-conformal, and emit a ScenarioBand whose
    width widens by √h across the horizon.

    Parameters
    ----------
    fit_predict_fn : callable(prices, commodity) -> dict
        Returns
            {
                "point_preds_test": pd.Series of holdout predictions,
                "realized_test":    pd.Series of realized log returns,
                "next_forecast":    float — 1-step-ahead point prediction,
                "diagnostics":      dict,
            }
    """
    from models.features import build_feature_matrix, build_target
    from models.config import TEST_FRACTION

    if commodity not in prices.columns:
        return None

    try:
        bundle = fit_predict_fn(prices, commodity)
    except Exception as e:
        # Fail soft — caller decides whether to flag this
        return None

    if bundle is None:
        return None

    calib = calibrate_from_model(
        point_preds_test=bundle["point_preds_test"],
        realized_test=bundle["realized_test"],
        next_point_forecast=bundle["next_forecast"],
    )
    mean, bear, bull = calib.horizon_bands(horizon, bear_q, bull_q)

    return ScenarioBand(
        commodity=commodity,
        source_model=model_label,
        asof=pd.Timestamp(prices.index[-1]),
        horizon=horizon,
        bear_q=bear_q,
        bull_q=bull_q,
        mean=mean,
        bear=bear,
        bull=bull,
        diagnostics={
            "calibration_n": int(len(calib.residuals)),
            "next_forecast": calib.point_forecast,
            **bundle.get("diagnostics", {}),
        },
    )


def _elastic_net_bundle(prices: pd.DataFrame, commodity: str) -> Optional[dict]:
    from models.ml.elastic_net import ElasticNetFactorModel
    from models.features import build_feature_matrix, build_target
    from models.config import TEST_FRACTION

    feat = build_feature_matrix(prices)
    tgt = build_target(prices, commodity)
    en = ElasticNetFactorModel(commodity=commodity)
    en.fit(feat, tgt)

    combined = pd.concat([feat, tgt], axis=1).dropna()
    split = int(len(combined) * (1 - TEST_FRACTION))
    test_idx = combined.index[split:]
    preds = en.predict(feat).reindex(test_idx)
    realized = tgt.reindex(test_idx)
    next_fc = float(en.predict(feat).iloc[-1])
    return {
        "point_preds_test": preds,
        "realized_test": realized,
        "next_forecast": next_fc,
        "diagnostics": {"ic": en.ic_score(feat, tgt), **en.sparsity()},
    }


def elastic_net_band(
    prices: pd.DataFrame,
    commodity: str,
    horizon: int = 5,
    bear_q: float = 0.10,
    bull_q: float = 0.90,
) -> Optional[ScenarioBand]:
    """Conformal-wrapped Elastic Net 1-step forecaster, widened by √h."""
    return _ml_conformal_band(
        fit_predict_fn=_elastic_net_bundle,
        prices=prices, commodity=commodity, horizon=horizon,
        bear_q=bear_q, bull_q=bull_q,
        model_label="Conformal[ElasticNet]",
    )


def _random_forest_bundle(prices: pd.DataFrame, commodity: str) -> Optional[dict]:
    from models.ml.random_forest import CommodityRF
    from models.features import build_feature_matrix, build_target
    from models.config import TEST_FRACTION

    feat = build_feature_matrix(prices)
    tgt = build_target(prices, commodity)
    rf = CommodityRF(commodity=commodity)
    rf.fit(feat, tgt)

    # RF.predict expects the same column set it was fit on (incl. cross features)
    combined = pd.concat([feat[rf._feature_names], tgt], axis=1).dropna()
    split = int(len(combined) * (1 - TEST_FRACTION))
    test_idx = combined.index[split:]
    preds = rf.predict(feat).reindex(test_idx)
    realized = tgt.reindex(test_idx)
    next_fc = float(rf.predict(feat).iloc[-1])
    diag = {}
    try:
        diag["ic"] = rf.ic_score(feat, tgt)
    except Exception:
        pass
    return {
        "point_preds_test": preds,
        "realized_test": realized,
        "next_forecast": next_fc,
        "diagnostics": diag,
    }


def random_forest_band(
    prices: pd.DataFrame,
    commodity: str,
    horizon: int = 5,
    bear_q: float = 0.10,
    bull_q: float = 0.90,
) -> Optional[ScenarioBand]:
    """Conformal-wrapped Random Forest 1-step forecaster."""
    return _ml_conformal_band(
        fit_predict_fn=_random_forest_bundle,
        prices=prices, commodity=commodity, horizon=horizon,
        bear_q=bear_q, bull_q=bull_q,
        model_label="Conformal[RandomForest]",
    )


def _xgboost_bundle(prices: pd.DataFrame, commodity: str) -> Optional[dict]:
    from models.ml.xgboost_shap import XGBoostForecaster
    from models.features import build_feature_matrix, build_target
    from models.config import TEST_FRACTION

    feat = build_feature_matrix(prices)
    tgt = build_target(prices, commodity)
    xgb = XGBoostForecaster(commodity=commodity)
    xgb.fit(feat, tgt)

    combined = pd.concat([feat, tgt], axis=1).dropna()
    split = int(len(combined) * (1 - TEST_FRACTION))
    test_idx = combined.index[split:]
    preds = xgb.predict(feat).reindex(test_idx)
    realized = tgt.reindex(test_idx)
    next_fc = float(xgb.predict(feat).iloc[-1])
    diag = {}
    try:
        diag["ic"] = xgb.ic_score(feat, tgt)
    except Exception:
        pass
    return {
        "point_preds_test": preds,
        "realized_test": realized,
        "next_forecast": next_fc,
        "diagnostics": diag,
    }


def xgboost_band(
    prices: pd.DataFrame,
    commodity: str,
    horizon: int = 5,
    bear_q: float = 0.10,
    bull_q: float = 0.90,
) -> Optional[ScenarioBand]:
    """Conformal-wrapped XGBoost 1-step forecaster."""
    return _ml_conformal_band(
        fit_predict_fn=_xgboost_bundle,
        prices=prices, commodity=commodity, horizon=horizon,
        bear_q=bear_q, bull_q=bull_q,
        model_label="Conformal[XGBoost]",
    )


# ── Deep providers (LSTM via MC Dropout, TFT native quantiles, Prophet) ──────


def lstm_band(
    prices: pd.DataFrame,
    commodity: str,
    horizon: int = 5,
    bear_q: float = 0.10,
    bull_q: float = 0.90,
    n_mc_samples: int = 100,
    max_epochs: Optional[int] = None,
) -> Optional[ScenarioBand]:
    """
    MC-Dropout + conformal band for the project's bidirectional LSTM forecaster.

    Strategy
    --------
    1. Train ``LSTMForecaster`` on ``prices``.
    2. **Epistemic** uncertainty: run ``n_mc_samples`` MC-Dropout forward passes
       on the latest 30-day window. We keep the *mean* as the point forecast
       but discard the spread — vanilla MC Dropout captures only model
       uncertainty (e.g. Yarin Gal 2016), not aleatoric noise, and the spread
       is empirically far too narrow on financial returns.
    3. **Aleatoric** uncertainty: roll the trained model forward over the
       holdout period (``predict_sequence``), collect signed residuals
       ``y − ŷ``, and use their empirical quantiles as the band offsets.
       This is conformal calibration applied to the LSTM's actual error
       distribution — same logic as the ML providers.
    4. Widen by √h across the horizon.

    Returns None if torch is unavailable or the commodity is missing.
    """
    try:
        import torch  # noqa: F401
        from models.deep.lstm import LSTMForecaster, SEQ_LEN
    except ImportError:
        return None
    from models.features import build_feature_matrix
    from models.scenarios.mc_dropout import _enable_dropout_only

    if commodity not in prices.columns:
        return None

    forecaster = LSTMForecaster(commodities={commodity: ""})  # single-head saves time
    try:
        forecaster.fit(prices, verbose=False)
    except Exception:
        return None
    if forecaster._model is None:
        return None

    # Build the latest 30-day input sequence
    feat_df = build_feature_matrix(prices)
    X = feat_df[forecaster._feature_names].dropna().values
    if len(X) < SEQ_LEN:
        return None
    X_sc = forecaster._scaler.transform(X)
    seq = torch.tensor(X_sc[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(forecaster._device)

    # MC Dropout: dropout-only train mode, N forward passes
    _enable_dropout_only(forecaster._model)
    samples = []
    with torch.no_grad():
        for i in range(n_mc_samples):
            torch.manual_seed(i)  # vary the dropout mask
            out = forecaster._model(seq)
            samples.append(float(out[commodity].item()))
    forecaster._model.eval()
    samples = np.asarray(samples)

    point_fc = float(samples.mean())
    mc_std = float(samples.std())

    # ── Aleatoric calibration: holdout residuals from predict_sequence ──
    # predict_sequence rolls the LSTM forward across the test split, giving
    # us paired (prediction, realized) over a holdout. The signed residuals
    # are the empirical 1-step error distribution under the trained model.
    from models.features import log_returns
    try:
        preds_test = forecaster.predict_sequence(prices)
        if commodity not in preds_test.columns:
            return None
        realized = log_returns(prices)[commodity].reindex(preds_test.index)
        residuals = (realized - preds_test[commodity]).dropna().to_numpy()
    except Exception:
        residuals = np.array([])

    if len(residuals) >= 20:
        bear_off = float(np.quantile(residuals, bear_q))
        bull_off = float(np.quantile(residuals, bull_q))
        calibration_source = "conformal_holdout"
    else:
        # Fall back to MC samples if the holdout is too short (rare on this data)
        bear_off = float(np.quantile(samples, bear_q)) - point_fc
        bull_off = float(np.quantile(samples, bull_q)) - point_fc
        calibration_source = "mc_dropout_fallback"

    steps = np.arange(1, horizon + 1)
    sqrt_h = np.sqrt(steps)
    mean = np.full(horizon, point_fc)
    bear = point_fc + bear_off * sqrt_h
    bull = point_fc + bull_off * sqrt_h

    # Per-step Spearman IC on the holdout — same diagnostic as the ML providers
    diag_ic = None
    try:
        from scipy.stats import spearmanr
        if len(residuals) > 10:
            ic, _ = spearmanr(preds_test[commodity].dropna(),
                              realized.reindex(preds_test[commodity].dropna().index))
            diag_ic = round(float(ic), 4)
    except Exception:
        pass

    return ScenarioBand(
        commodity=commodity,
        source_model="MCDropout[LSTM]",
        asof=pd.Timestamp(prices.index[-1]),
        horizon=horizon,
        bear_q=bear_q,
        bull_q=bull_q,
        mean=mean,
        bear=bear,
        bull=bull,
        diagnostics={
            "n_mc_samples": n_mc_samples,
            "mc_sample_std": mc_std,
            "point_forecast": point_fc,
            "calibration": calibration_source,
            "calibration_n": int(len(residuals)),
            "residual_std": float(np.std(residuals)) if len(residuals) else None,
            "ic": diag_ic,
        },
    )


def tft_band(
    prices: pd.DataFrame,
    commodity: str,
    horizon: int = 5,
    bear_q: float = 0.10,
    bull_q: float = 0.90,
) -> Optional[ScenarioBand]:
    """
    Native-quantile band from the Temporal Fusion Transformer.

    TFT is trained with ``QuantileLoss`` over q10/q50/q90, so its output
    *is* a calibrated band — no MC Dropout required. We use its native
    quantiles when ``bear_q ≈ 0.10`` and ``bull_q ≈ 0.90``; for other
    quantiles we currently warn and fall back to the nearest-side quantile,
    since the trained network only emits the three configured quantiles.
    """
    try:
        from models.deep.tft import CommodityTFT
    except Exception:
        return None
    if commodity not in prices.columns:
        return None

    quantiles_native = (0.10, 0.50, 0.90)
    quantile_warning = None
    if abs(bear_q - quantiles_native[0]) > 0.05 or abs(bull_q - quantiles_native[2]) > 0.05:
        quantile_warning = (
            "TFT only emits its trained quantiles "
            f"{quantiles_native}; band reflects those rather than the requested "
            f"({bear_q}, {bull_q})."
        )

    # max_horizon = max(self.horizons) drives the prediction length, so include
    # the requested horizon explicitly.
    horizons = sorted(set([1, horizon, min(horizon, 20)]))
    tft = CommodityTFT(commodities={commodity: ""}, horizons=horizons)
    try:
        tft.fit(prices, max_epochs=15, verbose=False)
        fc = tft.forecast(prices)
    except Exception:
        return None
    if fc.empty:
        return None

    fc = fc[fc["commodity"] == commodity].sort_values("horizon")
    if fc.empty:
        return None

    # Build step-wise quantile arrays. TFT returns q-values at each horizon h
    # as the *cumulative* prediction over [0, h]. We treat those as the path of
    # cumulative log-returns and derive per-step deltas so they compose with
    # returns_to_price_paths the same way the other providers do.
    h_list = list(range(1, horizon + 1))
    fc_h = fc.set_index("horizon").reindex(h_list, method="nearest")

    cum_q10 = fc_h["q10"].to_numpy(dtype=float)
    cum_q50 = fc_h["q50"].to_numpy(dtype=float)
    cum_q90 = fc_h["q90"].to_numpy(dtype=float)

    bear = np.diff(np.concatenate([[0.0], cum_q10]))
    mean = np.diff(np.concatenate([[0.0], cum_q50]))
    bull = np.diff(np.concatenate([[0.0], cum_q90]))

    diag = {"native_quantiles": list(quantiles_native), "horizons_trained": horizons}
    if quantile_warning:
        diag["quantile_warning"] = quantile_warning

    return ScenarioBand(
        commodity=commodity,
        source_model="TFT[quantile]",
        asof=pd.Timestamp(prices.index[-1]),
        horizon=horizon,
        bear_q=bear_q,
        bull_q=bull_q,
        mean=mean,
        bear=bear,
        bull=bull,
        diagnostics=diag,
    )


def prophet_band(
    prices: pd.DataFrame,
    commodity: str,
    horizon: int = 5,
    bear_q: float = 0.10,
    bull_q: float = 0.90,
) -> Optional[ScenarioBand]:
    """
    Native-interval band from Prophet (log-price model).

    For symmetric quantiles (``bear_q + bull_q == 1``) Prophet's
    ``interval_width = bull_q - bear_q`` produces the desired band directly.
    For asymmetric quantiles we fit Prophet with a wide interval (99%),
    treat the resulting symmetric bounds as a Gaussian σ per step, then
    re-map to the requested asymmetric quantiles via mean ± z(q)·σ. This
    re-uses Prophet's uncertainty estimate without re-fitting twice.
    """
    try:
        from prophet import Prophet  # noqa: F401
    except ImportError:
        return None
    if commodity not in prices.columns:
        return None

    import warnings as _w
    from models.deep.prophet_decomp import _DEFAULT_CONFIG, _SEASONAL_CONFIG

    cfg = _SEASONAL_CONFIG.get(commodity, _DEFAULT_CONFIG)
    series = np.log(prices[commodity]).dropna()
    if len(series) < 100:
        return None
    df = pd.DataFrame({"ds": series.index, "y": series.values})

    # Decide native vs re-mapped: if the user asked for symmetric quantiles,
    # use Prophet's native interval directly. Otherwise, fit at 99 % and
    # re-map via Gaussian z-scores.
    symmetric = abs((bear_q + bull_q) - 1.0) < 1e-6
    width = (bull_q - bear_q) if symmetric else 0.99

    from prophet import Prophet
    m = Prophet(
        growth="linear",
        changepoint_prior_scale=cfg.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=cfg.get("seasonality_prior_scale", 10.0),
        weekly_seasonality=cfg.get("weekly", True),
        yearly_seasonality=False,
        daily_seasonality=False,
        interval_width=width,
    )
    if cfg.get("yearly", False):
        m.add_seasonality(
            name="yearly",
            period=365.25,
            fourier_order=cfg.get("yearly_order", 6),
            mode=cfg.get("mode", "additive"),
        )

    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            m.fit(df)
        future = m.make_future_dataframe(periods=horizon, freq="B")
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            fc = m.predict(future)
    except Exception:
        return None

    # Anchor on Prophet's *own* fitted value at the last historical date so
    # the per-step deltas don't absorb the in-sample residual (a 0.05 gap
    # between observed log-price and yhat at the anchor would otherwise be
    # injected into step 1 as a fake 5% return).
    fc_window = fc.tail(horizon + 1).reset_index(drop=True)
    yhat = fc_window["yhat"].to_numpy(dtype=float)
    ylow = fc_window["yhat_lower"].to_numpy(dtype=float)
    yup = fc_window["yhat_upper"].to_numpy(dtype=float)

    mean = np.diff(yhat)

    if symmetric:
        bear = np.diff(ylow)
        bull = np.diff(yup)
    else:
        # Per-step sigma from Prophet's 99% interval (z(0.995) ≈ 2.576).
        # sigma_path[h] is the cumulative log-price uncertainty at step h.
        # The per-step (return) variance is the *increment* of variance, not
        # the difference of sigmas:  var_step[h] = var_path[h] − var_path[h−1].
        z_99 = _z(0.995)
        sigma_path = (yup - ylow) / (2.0 * z_99)        # length horizon+1
        var_path = sigma_path ** 2
        var_step = np.diff(var_path)                    # length horizon
        var_step = np.maximum(var_step, 0.0)
        sigma_step = np.sqrt(var_step)
        bear = mean + _z(bear_q) * sigma_step
        bull = mean + _z(bull_q) * sigma_step

    return ScenarioBand(
        commodity=commodity,
        source_model="Prophet",
        asof=pd.Timestamp(series.index[-1]),
        horizon=horizon,
        bear_q=bear_q,
        bull_q=bull_q,
        mean=mean,
        bear=bear,
        bull=bull,
        diagnostics={
            "interval_width": width,
            "weekly": bool(cfg.get("weekly", True)),
            "yearly": bool(cfg.get("yearly", False)),
        },
    )


# ── Quantum provider (kernel ridge + split conformal) ────────────────────────


def quantum_kernel_band(
    prices: pd.DataFrame,
    commodity: str,
    horizon: int = 5,
    bear_q: float = 0.10,
    bull_q: float = 0.90,
    n_train: int = 80,
    n_calib: int = 40,
) -> Optional[ScenarioBand]:
    """
    Quantum-kernel ridge regressor wrapped in split-conformal.

    The fidelity-kernel ridge is a point predictor; we calibrate signed
    residuals on a held-out fold (size ``n_calib``) and widen by √h. Kernel
    construction is O((n_train+n_calib)²) quantum-circuit calls — keep
    sample sizes small (80 + 40 is ~2× faster than the published 100-sample
    benchmark and still produces enough calibration mass).
    """
    try:
        from models.quantum.hybrid import QuantumHybridRegressor
        from models.features import build_quantum_features
    except Exception:
        return None
    if commodity not in prices.columns:
        return None

    try:
        X_full, y_full = build_quantum_features(prices, target_name=commodity)
    except Exception:
        return None

    n = len(X_full)
    if n < n_train + n_calib + 1:
        return None

    # Use the most recent n_train + n_calib + 1 observations to keep training
    # contemporaneous with the inference point.
    X = X_full[-(n_train + n_calib + 1):]
    y = y_full[-(n_train + n_calib + 1):]
    X_train, y_train = X[:n_train], y[:n_train]
    X_calib, y_calib = X[n_train:n_train + n_calib], y[n_train:n_train + n_calib]
    X_inference = X[-1:]   # the most recent feature vector

    try:
        # The repo default ridge (1e-5) is fine for synthetic benchmarks but
        # under-regularises on noisy financial residuals — the model fits
        # noise on the train set and extrapolates wildly on held-out points.
        # ridge=0.1 yields predictions on the same order as σ_y in practice.
        qm = QuantumHybridRegressor(ridge_reg=0.1)
        qm.fit(X_train, y_train)
        preds_calib = qm.predict(X_calib)
        next_fc = float(qm.predict(X_inference)[0])
    except Exception:
        return None

    residuals = y_calib - preds_calib
    if residuals.std() < 1e-12:
        return None

    # Sanity gate: the QuantumHybridRegressor with tiny ridge can produce
    # wildly extrapolating predictions when the test point is far from train
    # in Hilbert space. A "next-day log return" of |·| > 0.5 implies a >50%
    # one-day move — never credible for liquid commodities. We reject the
    # band rather than display fabricated risk to investors.
    typical_sigma = float(np.std(y))
    if abs(next_fc) > max(0.5, 8 * typical_sigma) or residuals.std() > 10 * typical_sigma:
        return None

    bear_off = float(np.quantile(residuals, bear_q))
    bull_off = float(np.quantile(residuals, bull_q))
    steps = np.arange(1, horizon + 1)
    sqrt_h = np.sqrt(steps)
    mean = np.full(horizon, next_fc)
    bear = next_fc + bear_off * sqrt_h
    bull = next_fc + bull_off * sqrt_h

    return ScenarioBand(
        commodity=commodity,
        source_model="Quantum[KernelRidge]",
        asof=pd.Timestamp(prices.index[-1]),
        horizon=horizon,
        bear_q=bear_q,
        bull_q=bull_q,
        mean=mean,
        bear=bear,
        bull=bull,
        diagnostics={
            "n_train": n_train,
            "n_calib": n_calib,
            "calib_residual_std": float(residuals.std()),
            "next_forecast": next_fc,
            "n_qubits": int(X_train.shape[1]),
        },
    )


# ── Group-lookup helper ───────────────────────────────────────────────────────


def find_var_group(commodity: str) -> Optional[str]:
    """Return the first COMMODITY_GROUPS group that contains this commodity."""
    from models.statistical.var_vecm import COMMODITY_GROUPS

    for group, members in COMMODITY_GROUPS.items():
        if commodity in members:
            return group
    return None
