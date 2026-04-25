"""
Temporal Fusion Transformer (TFT) — multi-horizon forecasting across all futures.

The TFT (Lim et al. 2021, https://arxiv.org/abs/1912.09363) is a purpose-built
architecture for multi-step time series forecasting with interpretability.
Key components:

  Variable Selection Networks (VSN)
      Learned attention over input features — both static (commodity identity)
      and time-varying (returns, vol, momentum). The VSN weights directly
      answer "which features matter most?" without post-hoc SHAP computation.

  Multi-head Self-Attention over the lookback
      Produces attention weights across the 60-day encoder window, revealing
      which historical periods the model is drawing on for each prediction.
      A spike at day -5 means the model is anchoring on last week's pattern.

  Quantile output heads
      Simultaneously predicts the 10th, 50th, and 90th percentiles at horizons
      h ∈ {1, 5, 20} days. This gives calibrated uncertainty rather than
      a single point forecast — the 10/90 band is a model-derived risk range.

Multi-entity design
      All commodities are fed to a single model with the commodity name as a
      categorical static covariate. The model learns which behaviour is
      commodity-specific vs. which is driven by shared macro factors.

This is the most expressive model in the stack and has the longest training
time (~5-30 min on CPU for 2 years of 9-commodity data). Treat it as the
research-grade benchmark that the other models are evaluated against.

Usage
-----
    from models.deep.tft import CommodityTFT

    prices = load_price_matrix()

    tft = CommodityTFT(horizons=[1, 5, 20], lookback=60)
    tft.fit(prices, max_epochs=30, verbose=True)

    fc   = tft.forecast(prices)           # pd.DataFrame: quantile forecasts per commodity
    vi   = tft.variable_importance()      # pd.DataFrame: VSN weights
    attn = tft.attention_weights(prices)  # dict: lookback attention per commodity
    ic   = tft.ic_scores(prices)          # Spearman IC per commodity × horizon
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_forecasting.data import GroupNormalizer
    _PTF_AVAILABLE = True
except ImportError:
    _PTF_AVAILABLE = False

try:
    import lightning.pytorch as pl
    _PL_AVAILABLE = True
except ImportError:
    try:
        import pytorch_lightning as pl
        _PL_AVAILABLE = True
    except ImportError:
        _PL_AVAILABLE = False

from scipy.stats import spearmanr

from models.config import MODELING_COMMODITIES, TEST_FRACTION, RANDOM_SEED
from models.features import build_feature_matrix, log_returns


def _require_tft():
    missing = []
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if not _PTF_AVAILABLE:
        missing.append("pytorch-forecasting")
    if not _PL_AVAILABLE:
        missing.append("lightning")
    if missing:
        raise ImportError(
            f"CommodityTFT requires: {', '.join(missing)}.  "
            f"Install with: pip install torch pytorch-forecasting lightning"
        )


# ── Hyperparameters ────────────────────────────────────────────────────────────
QUANTILES       = [0.1, 0.5, 0.9]
HIDDEN_SIZE     = 32
ATTENTION_HEADS = 2
DROPOUT         = 0.1
HIDDEN_CONT     = 16
LR              = 3e-3
BATCH_SIZE      = 64
GRAD_CLIP       = 0.1


def _build_long_df(
    prices: pd.DataFrame,
    feat_df: pd.DataFrame,
    commodities: list,
) -> pd.DataFrame:
    """
    Convert wide price/feature DataFrames to the long format that
    pytorch-forecasting's TimeSeriesDataSet expects.

    Columns produced:
        time_idx      — integer index (0-based, monotone per commodity)
        commodity     — categorical group identifier
        log_return    — target: log return on this day
        <feature_cols> — all columns from feat_df (time-varying covariates)
    """
    ret_df = log_returns(prices)
    records = []

    # Align on dates present in both feature matrix and price data
    common_idx = feat_df.dropna().index.intersection(ret_df.dropna().index)

    for commodity in commodities:
        if commodity not in ret_df.columns:
            continue
        df = pd.DataFrame(index=common_idx)
        df["log_return"] = ret_df.loc[common_idx, commodity]
        df["commodity"]  = commodity
        df["time_idx"]   = np.arange(len(common_idx))
        for col in feat_df.columns:
            df[col] = feat_df.loc[common_idx, col].values
        records.append(df.reset_index(names=["date"]))

    long = pd.concat(records, ignore_index=True)
    long = long.dropna(subset=["log_return"])
    return long


class CommodityTFT:
    """
    Temporal Fusion Transformer trained jointly across all MODELING_COMMODITIES.

    Parameters
    ----------
    commodities : dict, optional
        {display_name: ticker}. Defaults to MODELING_COMMODITIES.
    horizons : list[int]
        Forecast horizons in trading days. [1, 5, 20] gives next-day,
        weekly, and monthly quantile forecasts simultaneously.
    lookback : int
        Encoder window (days). 60 days captures two trading months.
    """

    def __init__(
        self,
        commodities: Optional[dict] = None,
        horizons: Optional[list] = None,
        lookback: int = 60,
    ):
        _require_tft()
        self.commodities = commodities or MODELING_COMMODITIES
        self.horizons    = horizons or [1, 5, 20]
        self.lookback    = lookback
        self._model: Optional[TemporalFusionTransformer] = None
        self._training_ds: Optional[TimeSeriesDataSet] = None
        self._long_df: Optional[pd.DataFrame] = None
        self._feat_cols: Optional[list] = None
        self._training_cutoff: Optional[int] = None

    # ── private ────────────────────────────────────────────────────────────────

    def _build_datasets(
        self, prices: pd.DataFrame
    ) -> tuple["TimeSeriesDataSet", "TimeSeriesDataSet"]:
        feat_df = build_feature_matrix(prices)
        self._feat_cols = feat_df.columns.tolist()
        commodity_names = [c for c in self.commodities if c in prices.columns]

        long = _build_long_df(prices, feat_df, commodity_names)
        self._long_df = long

        max_time_idx = long["time_idx"].max()
        cutoff = int(max_time_idx * (1 - TEST_FRACTION))
        self._training_cutoff = cutoff

        max_horizon = max(self.horizons)

        training = TimeSeriesDataSet(
            long[long["time_idx"] <= cutoff],
            time_idx="time_idx",
            target="log_return",
            group_ids=["commodity"],
            min_encoder_length=self.lookback // 2,
            max_encoder_length=self.lookback,
            min_prediction_length=1,
            max_prediction_length=max_horizon,
            time_varying_unknown_reals=self._feat_cols + ["log_return"],
            target_normalizer=GroupNormalizer(groups=["commodity"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            long,
            predict=True,
            stop_randomization=True,
        )

        return training, validation

    # ── public interface ───────────────────────────────────────────────────────

    def fit(
        self,
        prices: pd.DataFrame,
        max_epochs: int = 30,
        verbose: bool = False,
    ) -> "CommodityTFT":
        """
        Train the TFT via pytorch-lightning.

        Parameters
        ----------
        prices : pd.DataFrame
            Closing prices from load_price_matrix().
        max_epochs : int
            Training epochs. 30 is sufficient for convergence on 2y data;
            50-100 for higher accuracy on longer histories.
        verbose : bool
            Show Lightning training progress.
        """
        import torch
        torch.manual_seed(RANDOM_SEED)

        training_ds, val_ds = self._build_datasets(prices)
        self._training_ds = training_ds

        train_loader = training_ds.to_dataloader(
            train=True, batch_size=BATCH_SIZE, num_workers=0
        )
        val_loader = val_ds.to_dataloader(
            train=False, batch_size=BATCH_SIZE, num_workers=0
        )

        tft = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=LR,
            hidden_size=HIDDEN_SIZE,
            attention_head_size=ATTENTION_HEADS,
            dropout=DROPOUT,
            hidden_continuous_size=HIDDEN_CONT,
            loss=QuantileLoss(quantiles=QUANTILES),
            log_interval=10 if verbose else -1,
            log_val_interval=1,
            reduce_on_plateau_patience=4,
        )

        log_level = "INFO" if verbose else "ERROR"
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=GRAD_CLIP,
            enable_progress_bar=verbose,
            enable_model_summary=verbose,
            logger=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(tft, train_loader, val_loader)

        self._model = tft
        return self

    def forecast(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate quantile forecasts for all commodities at all horizons.

        Returns
        -------
        pd.DataFrame
            Columns: commodity, horizon, q10, q50, q90, date_of_forecast.
            The forecast is for the period immediately after the last date
            in `prices`.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        feat_df = build_feature_matrix(prices)
        commodity_names = [c for c in self.commodities if c in prices.columns]
        long = _build_long_df(prices, feat_df, commodity_names)

        predict_ds = TimeSeriesDataSet.from_dataset(
            self._training_ds,
            long,
            predict=True,
            stop_randomization=True,
        )
        loader = predict_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._model.predict(
                loader, mode="quantiles", return_index=True, return_decoder_lengths=True
            )

        predictions, index = raw[0], raw[1]
        # predictions: (n_series, max_horizon, n_quantiles)

        rows = []
        max_horizon = max(self.horizons)
        for i, (_, row) in enumerate(index.iterrows()):
            commodity = row.get("commodity", commodity_names[i % len(commodity_names)])
            for h_idx, h in enumerate(range(1, max_horizon + 1)):
                if h not in self.horizons:
                    continue
                q_vals = predictions[i, h_idx, :]
                rows.append({
                    "commodity": commodity,
                    "horizon":   h,
                    "q10":       float(q_vals[0]),
                    "q50":       float(q_vals[1]),
                    "q90":       float(q_vals[2]),
                })

        return pd.DataFrame(rows)

    def variable_importance(self) -> pd.DataFrame:
        """
        Variable selection network (VSN) weights from the trained TFT.

        These are learned attention weights over the input feature set —
        the model's own answer to "which variables matter?"

        Returns
        -------
        pd.DataFrame
            Columns: variable, encoder_importance, decoder_importance.
            Sorted by encoder_importance descending.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        if self._long_df is None:
            raise RuntimeError("Internal state missing — refit the model.")

        predict_ds = TimeSeriesDataSet.from_dataset(
            self._training_ds,
            self._long_df,
            predict=True,
            stop_randomization=True,
        )
        loader = predict_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_predictions = self._model.predict(
                loader, mode="raw", return_index=True
            )
            interpretation = self._model.interpret_output(
                raw_predictions[0], reduction="mean"
            )

        enc_imp = interpretation.get("encoder_variables", {})
        dec_imp = interpretation.get("decoder_variables", {})

        all_vars = sorted(set(list(enc_imp.keys()) + list(dec_imp.keys())))
        rows = []
        for var in all_vars:
            rows.append({
                "variable":            var,
                "encoder_importance":  round(float(enc_imp.get(var, 0.0)), 5),
                "decoder_importance":  round(float(dec_imp.get(var, 0.0)), 5),
            })

        df = pd.DataFrame(rows)
        return df.sort_values("encoder_importance", ascending=False).reset_index(drop=True)

    def attention_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Mean attention weights over the lookback window, averaged across all
        commodities and test-set predictions.

        A spike at relative position -k means the model is drawing heavily on
        the data from k days before the forecast date.

        Returns
        -------
        pd.DataFrame
            Columns: relative_day (−lookback … −1), mean_attention.
            Index = relative_day integers.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        feat_df = build_feature_matrix(prices)
        commodity_names = [c for c in self.commodities if c in prices.columns]
        long = _build_long_df(prices, feat_df, commodity_names)

        predict_ds = TimeSeriesDataSet.from_dataset(
            self._training_ds,
            long,
            predict=True,
            stop_randomization=True,
        )
        loader = predict_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._model.predict(loader, mode="raw", return_index=True)
            interp = self._model.interpret_output(raw[0], reduction="mean")

        attn = interp.get("attention", None)
        if attn is None:
            return pd.DataFrame()

        # attn shape: (n_samples, n_heads, lookback_steps)
        mean_attn = attn.mean(dim=(0, 1)).cpu().numpy()
        n = len(mean_attn)
        relative_days = list(range(-n, 0))

        return pd.DataFrame({
            "relative_day":   relative_days,
            "mean_attention": mean_attn.round(5),
        }).set_index("relative_day")

    def ic_scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Spearman IC of the median forecast (q50) vs. actual returns,
        broken down by commodity and horizon.

        Returns
        -------
        pd.DataFrame
            Columns: commodity, horizon, IC, n_obs.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        ret_df  = log_returns(prices).dropna()
        fc_df   = self.forecast(prices)
        rows    = []

        for (commodity, horizon), sub in fc_df.groupby(["commodity", "horizon"]):
            if commodity not in ret_df.columns:
                continue
            actuals = ret_df[commodity].dropna()
            if len(sub) < 5 or len(actuals) < 5:
                continue
            n = min(len(sub), len(actuals))
            ic, _ = spearmanr(sub["q50"].values[:n], actuals.values[-n:])
            rows.append({
                "commodity": commodity,
                "horizon":   horizon,
                "IC":        round(float(ic), 4),
                "n_obs":     n,
            })

        return pd.DataFrame(rows).sort_values(
            ["horizon", "IC"], ascending=[True, False]
        ).reset_index(drop=True)
