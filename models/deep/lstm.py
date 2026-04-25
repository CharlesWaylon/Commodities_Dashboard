"""
2-layer Bidirectional LSTM with shared encoder and commodity-specific output heads.

Architecture
------------
    Input:   (batch, seq_len=30, n_features)
    Encoder: BiLSTM(hidden=64, layers=2, dropout=0.2)
             → (batch, seq_len, 128)   # 2 × hidden for bidirectional
             Take last timestep context: (batch, 128)
    Heads:   One per commodity — Linear(128→32) → ReLU → Linear(32→1)
    Output:  Dict[commodity → (batch,)]

Why bidirectional for financial data?
  The bidirectional component operates *within* the 30-day lookback window,
  not beyond the prediction horizon. All 30 days of the lookback are in the
  past relative to the forecast date, so both forward and backward passes
  over that window are information-leak-free. The backward pass lets the
  model align early-window signals with late-window outcomes, which helps
  identify multi-week supply cycle patterns.

Why shared encoder + separate heads?
  Commodities within the same complex (energy, grains) share macro drivers.
  The shared encoder learns cross-commodity dynamics. Separate heads allow
  each commodity to weight the shared representation differently — Gold and
  WTI respond to the same dollar strength signal but with different sign
  and magnitude.

Multi-task training loss is the mean MSE across all commodities. Each epoch,
the encoder is updated by gradients from every commodity simultaneously.

Usage
-----
    from models.deep.lstm import LSTMForecaster

    prices = load_price_matrix()
    forecaster = LSTMForecaster()
    history = forecaster.fit(prices, verbose=True)

    preds  = forecaster.predict_latest(prices)   # next-day return per commodity
    scores = forecaster.ic_scores(prices)         # Spearman IC per commodity
    hidden = forecaster.encode(prices)            # raw encoder output (n_days, 128)
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Optional
from scipy.stats import spearmanr

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from models.config import MODELING_COMMODITIES, TEST_FRACTION, RANDOM_SEED
from models.features import build_feature_matrix, log_returns

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN              = 30
HIDDEN_SIZE          = 64
N_LAYERS             = 2
DROPOUT              = 0.2
BATCH_SIZE           = 32
MAX_EPOCHS           = 200
LR                   = 1e-3
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 15
VALIDATION_FRACTION  = 0.15
GRAD_CLIP            = 1.0


# ── PyTorch modules (defined only when torch is available) ─────────────────────

def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for LSTMForecaster.  Install with: pip install torch"
        )


class _SequenceDataset(Dataset if _TORCH_AVAILABLE else object):
    """Sliding-window dataset: input=30-day feature sequence, target=next-day returns."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
        import torch
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, i):
        return self.X[i : i + self.seq_len], self.y[i + self.seq_len]


class _SharedBiLSTM(nn.Module if _TORCH_AVAILABLE else object):
    """Shared bidirectional LSTM encoder — output is the last-step hidden state."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = HIDDEN_SIZE,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * 2   # bidirectional doubles width

    def forward(self, x):
        out, _ = self.lstm(x)                # (batch, seq, hidden*2)
        return self.dropout(out[:, -1, :])   # last timestep context


class _CommodityHead(nn.Module if _TORCH_AVAILABLE else object):
    """Per-commodity output head: context → scalar next-day return."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _MultiCommodityLSTM(nn.Module if _TORCH_AVAILABLE else object):
    """Full model: shared BiLSTM encoder + dict of per-commodity heads."""

    def __init__(self, n_features: int, commodity_names: list):
        super().__init__()
        self.encoder = _SharedBiLSTM(n_features)
        # ModuleDict requires string keys with no special characters
        self._key = lambda name: name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        self.heads = nn.ModuleDict({
            self._key(n): _CommodityHead(self.encoder.output_dim)
            for n in commodity_names
        })
        self.commodity_names = commodity_names

    def forward(self, x):
        context = self.encoder(x)
        return {name: self.heads[self._key(name)](context) for name in self.commodity_names}

    def encode(self, x):
        return self.encoder(x)


# ── Training wrapper ───────────────────────────────────────────────────────────

def _best_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LSTMForecaster:
    """
    Training and inference wrapper for the multi-commodity BiLSTM.

    Parameters
    ----------
    commodities : dict, optional
        {display_name: ticker} subset to model. Defaults to MODELING_COMMODITIES.
    seq_len : int
        Lookback window in trading days.
    """

    def __init__(
        self,
        commodities: Optional[dict] = None,
        seq_len: int = SEQ_LEN,
    ):
        _require_torch()
        self.commodities = commodities or MODELING_COMMODITIES
        self.seq_len = seq_len
        self._model: Optional[_MultiCommodityLSTM] = None
        self._scaler = None
        self._feature_names: Optional[list] = None
        self._commodity_names: Optional[list] = None
        self._device = _best_device()
        self.train_history: list = []

    # ── private ───────────────────────────────────────────────────────────────

    def _build_arrays(self, prices: pd.DataFrame):
        """Return aligned (X, y, feat_names, commodity_names) arrays."""
        from sklearn.preprocessing import StandardScaler

        feat_df = build_feature_matrix(prices)
        ret_df  = log_returns(prices)

        target_cols = [c for c in self.commodities if c in ret_df.columns]
        combined = pd.concat([feat_df, ret_df[target_cols]], axis=1).dropna()

        feat_cols = feat_df.columns.tolist()
        X_raw = combined[feat_cols].values
        y_raw = combined[target_cols].values

        return X_raw, y_raw, feat_cols, target_cols

    # ── public interface ──────────────────────────────────────────────────────

    def fit(self, prices: pd.DataFrame, verbose: bool = False) -> list:
        """
        Train the model via Adam + ReduceLROnPlateau + early stopping.

        Parameters
        ----------
        prices : pd.DataFrame
            Closing prices from load_price_matrix().
        verbose : bool
            Print epoch loss every 10 epochs.

        Returns
        -------
        list
            Training history: list of dicts with epoch, train_loss, val_loss.
        """
        import torch
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        X_raw, y_raw, feat_names, commodity_names = self._build_arrays(prices)
        self._feature_names = feat_names
        self._commodity_names = commodity_names

        # Chronological split: train → val slice → test
        n = len(X_raw)
        test_cut  = int(n * (1 - TEST_FRACTION))
        val_cut   = int(test_cut * (1 - VALIDATION_FRACTION))

        X_train, y_train = X_raw[:val_cut], y_raw[:val_cut]
        X_val,   y_val   = X_raw[val_cut:test_cut], y_raw[val_cut:test_cut]

        # Scale features on training data only
        self._scaler = StandardScaler()
        X_train_sc = self._scaler.fit_transform(X_train)
        X_val_sc   = self._scaler.transform(X_val)

        train_ds = _SequenceDataset(X_train_sc, y_train, self.seq_len)
        val_ds   = _SequenceDataset(X_val_sc,   y_val,   self.seq_len)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        self._model = _MultiCommodityLSTM(
            n_features=len(feat_names),
            commodity_names=commodity_names,
        ).to(self._device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, min_lr=1e-5
        )
        loss_fn = nn.MSELoss()

        best_val  = float("inf")
        best_state = None
        patience  = 0
        history   = []

        for epoch in range(1, MAX_EPOCHS + 1):
            # ── train ──
            self._model.train()
            train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)   # (batch, n_commodities)
                optimizer.zero_grad()
                preds = self._model(X_b)
                # Stack per-commodity predictions → (batch, n_commodities)
                pred_tensor = torch.stack(
                    [preds[n] for n in commodity_names], dim=1
                )
                loss = loss_fn(pred_tensor, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), GRAD_CLIP)
                optimizer.step()
                train_loss += loss.item() * len(X_b)
            train_loss /= max(len(train_ds), 1)

            # ── validate ──
            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(self._device)
                    y_b = y_b.to(self._device)
                    preds = self._model(X_b)
                    pred_tensor = torch.stack(
                        [preds[n] for n in commodity_names], dim=1
                    )
                    val_loss += loss_fn(pred_tensor, y_b).item() * len(X_b)
            val_loss /= max(len(val_ds), 1)

            scheduler.step(val_loss)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val:
                best_val   = val_loss
                best_state = deepcopy(self._model.state_dict())
                patience   = 0
            else:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    if verbose:
                        print(f"Early stop at epoch {epoch}")
                    break

            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}")

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self.train_history = history
        return history

    def predict_latest(self, prices: pd.DataFrame) -> dict:
        """
        Predict next-day log-return for every commodity using the most recent
        `seq_len` days of features.

        Returns
        -------
        dict
            {commodity_name: float predicted return}
        """
        import torch
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        feat_df = build_feature_matrix(prices)
        X_raw = feat_df[self._feature_names].dropna().values
        X_sc  = self._scaler.transform(X_raw)
        seq   = torch.tensor(X_sc[-self.seq_len:], dtype=torch.float32)
        seq   = seq.unsqueeze(0).to(self._device)   # (1, seq_len, n_features)

        self._model.eval()
        with torch.no_grad():
            preds = self._model(seq)

        return {name: float(preds[name].item()) for name in self._commodity_names}

    def predict_sequence(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Roll the model across the full test set, predicting one step at a time.

        Returns
        -------
        pd.DataFrame
            Columns = commodity names. Index = DatetimeIndex (test period only).
        """
        import torch
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        X_raw, y_raw, _, commodity_names = self._build_arrays(prices)
        X_sc = self._scaler.transform(X_raw)
        n = len(X_sc)
        test_start = int(n * (1 - TEST_FRACTION))

        feat_df = build_feature_matrix(prices)
        idx = feat_df.dropna().index

        self._model.eval()
        records = []
        with torch.no_grad():
            for t in range(test_start, n):
                if t < self.seq_len:
                    continue
                seq = torch.tensor(
                    X_sc[t - self.seq_len : t], dtype=torch.float32
                ).unsqueeze(0).to(self._device)
                preds = self._model(seq)
                row = {name: float(preds[name].item()) for name in commodity_names}
                row["date"] = idx[t] if t < len(idx) else None
                records.append(row)

        df = pd.DataFrame(records).set_index("date")
        df.index = pd.to_datetime(df.index)
        return df

    def ic_scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Per-commodity Spearman IC on the test period.

        Returns
        -------
        pd.DataFrame
            Columns: commodity, IC, n_obs.
        """
        _, y_raw, _, commodity_names = self._build_arrays(prices)
        n = len(y_raw)
        test_start = int(n * (1 - TEST_FRACTION))

        preds_df = self.predict_sequence(prices)

        feat_df  = build_feature_matrix(prices)
        ret_df   = log_returns(prices)
        combined = pd.concat([feat_df, ret_df[commodity_names]], axis=1).dropna()
        actuals  = combined[commodity_names].iloc[test_start:]

        rows = []
        for name in commodity_names:
            if name not in preds_df.columns or name not in actuals.columns:
                continue
            p = preds_df[name]
            a = actuals[name].reindex(p.index).dropna()
            p = p.reindex(a.index)
            if len(p) < 10:
                continue
            ic, _ = spearmanr(p, a)
            rows.append({"commodity": name, "IC": round(float(ic), 4), "n_obs": len(p)})

        return pd.DataFrame(rows)

    def encode(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Run the shared encoder across the full dataset.
        Returns the 128-dim context vector for every valid date.
        Useful for visualising learned commodity state representations.

        Returns
        -------
        pd.DataFrame
            Shape (n_valid_days, 128). DatetimeIndex.
        """
        import torch
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        feat_df = build_feature_matrix(prices)
        X_raw = feat_df[self._feature_names].dropna()
        X_sc  = self._scaler.transform(X_raw.values)
        n = len(X_sc)

        self._model.eval()
        contexts = []
        with torch.no_grad():
            for t in range(self.seq_len, n + 1):
                seq = torch.tensor(
                    X_sc[t - self.seq_len : t], dtype=torch.float32
                ).unsqueeze(0).to(self._device)
                ctx = self._model.encode(seq).cpu().numpy()[0]
                contexts.append(ctx)

        idx = X_raw.index[self.seq_len:]
        return pd.DataFrame(contexts, index=idx)
