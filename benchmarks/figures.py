"""Regenerate the paper's Figures 2-4 from benchmark results.

Usage::

    python benchmarks/run.py --out results
    python benchmarks/figures.py --results results --out figures

Writes the plotted data to ``<out>/data/*.csv`` and copies the pgfplots sources from
``benchmarks/tikz`` to ``<out>``. With ``pdflatex`` on the path, each figure is also
compiled to ``<out>/<figure>.pdf``.

Fig. 2: double integrator with polytopic constraints (case ``polytope``).
Fig. 3: cart-pole swing-up with box constraints (case ``cartpole``).
Fig. 4: iterations per MPC step, mirror-space (MS) vs interior-point (IP), averaged over
        the 25 control seeds of the compare cases. The curves are smoothed with a
        trailing moving average over 10 MPC steps.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess

import numpy as np

TIKZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tikz")
FIGURES = ("fig2_double_integrator", "fig3_cartpole", "fig4_iterations")
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


def trajectory(results: str, case: str, states: tuple[str, ...], controls: tuple[str, ...]) -> dict:
    """Closed loop of the first seed: time, the leading states and the controls."""
    d = _load(results, case)
    xs, us = d["ms_states"][0], d["ms_controls"][0]
    us = np.vstack([us, np.full((1, us.shape[1]), np.nan)])  # no control at the final state
    columns = {"t": DT[case] * np.arange(len(xs))}
    columns.update({name: xs[:, i] for i, name in enumerate(states)})
    columns.update({name: us[:, i] for i, name in enumerate(controls)})
    return columns


def iterations(results: str, case: str) -> dict:
    """Smoothed iterations per MPC step of both methods, averaged over the seeds."""
    d = _load(results, case)
    columns = {"step": np.arange(d["ms_iterations"].shape[1])}
    for method in ("ms", "ip"):
        columns[method] = np.mean([moving_average(it) for it in d[f"{method}_iterations"]], axis=0)
    return columns


def write_csv(path: str, columns: dict) -> None:
    np.savetxt(path, np.column_stack(list(columns.values())), delimiter=",", header=",".join(columns), comments="", fmt="%.8g")


def write_data(results: str, out: str) -> None:
    tables = {
        "fig2_double_integrator": trajectory(results, "polytope", ("p1", "p2"), ("u1", "u2")),
        "fig3_cartpole": trajectory(results, "cartpole", ("p", "theta"), ("u",)),
        "fig4_double_integrator": iterations(results, "polytope-compare"),
        "fig4_cartpole": iterations(results, "cartpole-compare"),
    }
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    for name, columns in tables.items():
        write_csv(os.path.join(out, "data", name + ".csv"), columns)
        print(f"[figures] data/{name}.csv")


def compile_figures(out: str) -> None:
    for source in sorted(os.listdir(TIKZ)):
        shutil.copy(os.path.join(TIKZ, source), out)
    if shutil.which("pdflatex") is None:
        print("[figures] pdflatex not found, wrote the sources only")
        return
    for name in FIGURES:
        command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={name}"]
        command.append(rf"\def\figurefile{{{name}}}\input{{standalone}}")
        if subprocess.run(command, cwd=out, stdout=subprocess.DEVNULL).returncode != 0:
            raise SystemExit(f"pdflatex failed, see {os.path.join(out, name + '.log')}")
        for ext in (".aux", ".log"):
            os.remove(os.path.join(out, name + ext))
        print(f"[figures] {name}.pdf")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    write_data(args.results, args.out)
    compile_figures(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
