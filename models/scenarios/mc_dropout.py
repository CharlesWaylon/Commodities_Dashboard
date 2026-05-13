"""
Monte-Carlo Dropout — turn any dropout-bearing torch model into a
probabilistic predictor by running N forward passes with dropout *active
at inference time* and collecting the empirical distribution of outputs.

This is the cheapest way to extract calibrated uncertainty from an already
trained deep model: no retraining, no architectural changes, just toggle
dropout layers into ``train()`` mode for the forward pass.

We're careful to leave any non-dropout layers (BatchNorm, LayerNorm, etc.)
in eval mode so running stats aren't perturbed.
"""

from __future__ import annotations

from typing import Callable
import numpy as np


def _enable_dropout_only(module) -> None:
    """Put Dropout / Dropout1d / Dropout2d / Dropout3d in train mode; everything else in eval."""
    import torch.nn as nn

    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()
        else:
            m.eval()


def mc_dropout_samples(
    forward_fn: Callable[[], "np.ndarray | float"],
    model,
    n_samples: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """
    Run ``forward_fn`` ``n_samples`` times with dropout active.

    Parameters
    ----------
    forward_fn : callable
        Zero-arg function returning a float or a np.ndarray of model output.
        Must close over the input tensor; we don't pass arguments to it so
        the caller controls exactly what's evaluated.
    model : torch.nn.Module
        The model. We toggle its dropout layers; eval-mode otherwise.
    n_samples : int
        Number of stochastic forward passes (≈100 is standard).
    seed : int
        Seed for the per-sample RNG branch — keeps repeated calls reproducible.

    Returns
    -------
    np.ndarray
        Shape (n_samples, ...). First axis is the MC dimension.
    """
    import torch

    _enable_dropout_only(model)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    outputs = []
    with torch.no_grad():
        for i in range(n_samples):
            # Advance the generator so consecutive draws differ even though
            # the wrapped forward_fn does its own RNG handling internally.
            torch.manual_seed(int(rng.initial_seed()) + i)
            outputs.append(np.asarray(forward_fn()))

    # Restore eval mode so subsequent unrelated calls don't see dropout.
    model.eval()
    return np.stack(outputs, axis=0)


def quantile_band_from_samples(
    samples: np.ndarray,
    bear_q: float,
    bull_q: float,
) -> tuple[float, float, float]:
    """
    Reduce a 1-D array of MC-Dropout samples to a (bear, mean, bull) tuple.
    """
    samples = np.asarray(samples).ravel()
    return (
        float(np.quantile(samples, bear_q)),
        float(np.mean(samples)),
        float(np.quantile(samples, bull_q)),
    )
