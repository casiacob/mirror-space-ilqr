"""Shared helpers for the experiment scripts: run a case's MS closed loops and score them
with the regression gate's metrics."""

from __future__ import annotations

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))

import regression  # noqa: E402
from cases import CASES  # noqa: E402

from msilqr import mirror_mpc  # noqa: E402


def run_ms(case_name: str, solver, mpc=mirror_mpc) -> dict:
    """Mirror-space closed loops of ``case_name`` with ``solver`` (one compile, Python loop)."""
    inp = CASES[case_name].inputs()
    res = [mpc(solver, x0, u0, inp.steps) for x0, u0 in zip(inp.x0, inp.u_init, strict=True)]
    out = {f"ms_{f}": np.stack([np.asarray(getattr(r, f)) for r in res]) for f in ("states", "controls", "iterations", "status")}
    out["x0"] = np.asarray(inp.x0)
    return out


def report(case_name: str, label: str, new: dict) -> None:
    """Print iterations, stalls, cost and goal error next to the golden values."""
    case = regression.CASES[case_name]
    golden = regression.load_legacy(case, regression.DEFAULT_GOLDEN)
    g, n = regression.metrics(case, golden, "ms"), regression.metrics(case, new, "ms")
    stalled = int(np.sum(new["ms_status"] != 0)) if "ms_status" in new else 0
    bitwise = np.array_equal(new["ms_states"], golden["ms_states"])
    print(
        f"{label:42s} iters {n['iters_total']:6d} (golden {g['iters_total']}) | max {n['iters_max']:4d} (golden {g['iters_max']}) "
        f"| not converged {stalled:3d} | cost {n['cost_mean']:10.4f} (golden {g['cost_mean']:.4f}) "
        f"| goal err {n['goal_error_mean']:.2e} | bitwise {bitwise}",
        flush=True,
    )
