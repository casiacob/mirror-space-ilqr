"""Regenerate the paper's Figures 2-4 from benchmark results.

Usage::

    python benchmarks/run.py --out results
    python benchmarks/figures.py --results results --out figures

Fig. 2: double integrator with polytopic constraints (case ``polytope``).
Fig. 3: cart-pole swing-up with box constraints (case ``cartpole``).
Fig. 4: iterations per MPC step, mirror-space (MS) vs interior-point (IP), averaged over
        the 25 control seeds of the compare cases. The curves are smoothed with a
        trailing moving average over 10 MPC steps.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DT = {"polytope": 0.02, "cartpole": 0.01}
WINDOW = 10


def moving_average(x: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Trailing mean over ``window`` samples (shorter windows at the start)."""
    c = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    ends = np.arange(1, len(x) + 1)
    starts = np.maximum(ends - window, 0)
    return (c[ends] - c[starts]) / (ends - starts)


def _load(results: str, case: str) -> dict:
    return dict(np.load(os.path.join(results, case + ".npz")))


def fig2(results: str):
    d = _load(results, "polytope")
    xs, us = d["ms_states"][0], d["ms_controls"][0]
    t = DT["polytope"] * np.arange(len(xs))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.6))
    a.plot(t, xs[:, 0], label="$p^1$")
    a.plot(t, xs[:, 1], "--", label="$p^2$")
    b.plot(t[:-1], us[:, 0], label="$u^1$")
    b.plot(t[:-1], us[:, 1], "--", label="$u^2$")
    for ax in (a, b):
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    return fig


def fig3(results: str):
    d = _load(results, "cartpole")
    xs, us = d["ms_states"][0], d["ms_controls"][0]
    t = DT["cartpole"] * np.arange(len(xs))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.6))
    a.plot(t, xs[:, 0], label="$p$")
    a.plot(t, xs[:, 1], "--", label=r"$\theta$")
    b.plot(t[:-1], us[:, 0], label="$u$")
    for bound in (-50.0, 50.0):
        b.axhline(bound, color="gray", lw=0.8, ls=":")
    for ax in (a, b):
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    return fig


def fig4(results: str):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    for ax, case, title in zip(axes, ("polytope-compare", "cartpole-compare"), ("Double Integrator", "Cart-Pole"), strict=True):
        d = _load(results, case)
        for method, style, label in (("ms", "-", "MS"), ("ip", "--", "IP")):
            curve = np.mean([moving_average(it) for it in d[f"{method}_iterations"]], axis=0)
            ax.plot(np.arange(len(curve)), curve, style, label=label)
        ax.set_title(title)
        ax.set_xlabel("Time step [t]")
        ax.grid(True)
        ax.legend()
    axes[0].set_ylabel("Avg. iterations")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--formats", nargs="*", default=["pdf", "png"])
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for name, make in (("fig2_double_integrator", fig2), ("fig3_cartpole", fig3), ("fig4_iterations", fig4)):
        fig = make(args.results)
        for ext in args.formats:
            fig.savefig(os.path.join(args.out, f"{name}.{ext}"), dpi=200)
        plt.close(fig)
        print(f"[figures] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
