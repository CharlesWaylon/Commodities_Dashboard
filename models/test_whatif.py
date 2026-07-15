import ast
from pathlib import Path

from models.whatif import blended_prior, upstream_contribution, prior_table


def test_blended_prior_bounds():
    assert blended_prior(0.9, 0.0) == 1.0
    assert blended_prior(0.9, 1.0) == 0.9
    assert abs(blended_prior(0.5, 0.5) - 0.75) < 1e-12


def test_contribution_monotone_in_damping():
    lo = upstream_contribution(0.4, 0.02, 0.5, 0.9, 1.0)
    hi = upstream_contribution(0.4, 0.02, 1.0, 0.9, 1.0)
    assert hi > lo


def test_prior_table_covers_all_edges():
    from models.config import SECTOR_TRANSMISSION_PRIORS
    rows = prior_table(alpha=1.0, damping=1.0)
    n_edges = sum(len(v) for v in SECTOR_TRANSMISSION_PRIORS.values())
    assert len(rows) == n_edges
    assert {"src", "dst", "prior", "effective", "contribution"} <= set(rows[0])


def test_whatif_module_is_pure():
    """Guardrail (spec §F3): the sandbox layer may never touch the DB."""
    tree = ast.parse(Path("models/whatif.py").read_text())
    imported = {
        n.name if isinstance(node, ast.Import) else node.module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for n in (node.names if isinstance(node, ast.Import) else [node])
        if (n.name if isinstance(node, ast.Import) else node.module)
    }
    assert not any(m and m.startswith("database") for m in imported)
    assert not any(m and m.startswith("sqlalchemy") for m in imported)
