"""
evaluation/ — the out-of-sample gate and look-ahead-correctness tooling.

This is the spine of the product: a signal is only promotable after a
walk-forward, purged/embargoed, cost-aware scorecard says so. This layer reads
the data and signal layers but never imports streamlit / pages / app (enforced
by ``.importlinter``).
"""

from evaluation.harness import (  # noqa: F401
    HarnessConfig,
    PassFailConfig,
    SignalScorecard,
    render,
    run_signal,
)

__all__ = [
    "run_signal",
    "render",
    "HarnessConfig",
    "PassFailConfig",
    "SignalScorecard",
]
