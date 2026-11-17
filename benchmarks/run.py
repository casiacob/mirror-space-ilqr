"""Run the benchmark cases.

Usage (from the repository root)::

    python benchmarks/run.py --out results            # all cases
    python benchmarks/run.py --out results --cases cartpole polytope
    python benchmarks/regression.py --new results --bitwise   # check against golden

Writes ``<out>/<case>.npz`` and ``<out>/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import CASES  # noqa: E402

from msilqr import IPDDPOptions, Status, ipddp_mpc, mirror_mpc  # noqa: E402


def _run_method(fn, x0s, u_inits):
    """Run one closed loop per initial state with a single compiled function."""
    t0 = time.perf_counter()
    results = [fn(x0, u0) for x0, u0 in zip(x0s, u_inits, strict=True)]
    jax.block_until_ready(results)
    wall = time.perf_counter() - t0
    stack = {f: np.stack([np.asarray(getattr(r, f)) for r in results]) for f in ("states", "controls", "iterations", "status")}
    return stack, wall


def run_case(
    name: str,
    box_scale: float | None = None,
    ip_barrier_tol: float | None = None,
    baseline: str = "paper",
    ip_warm_start: bool = True,
    stop: str = "native",
) -> tuple[dict, dict]:
    """Run one case. ``box_scale`` (cart-pole cases only), ``ip_barrier_tol`` (paper
    baseline only) and ``baseline="fixed"`` select variants that differ from the original
    experiments; keep the defaults to reproduce the golden results. ``stop="kkt"`` stops
    MS on the natural residual and IPDDP on the KKT error ``E_0`` (needs ``baseline="fixed"``)."""
    case = CASES[name]
    inp = case.inputs()
    out = {"x0": np.asarray(inp.x0)}
    info = {"case": name, "description": case.description, "n_init": int(inp.x0.shape[0]), "steps": inp.steps}
    if box_scale is not None and name.startswith("cartpole"):
        info["box_scale"] = box_scale
    if case.ip_solver is not None:
        info["ip_baseline"] = baseline
        if baseline == "fixed":
            info["ip_warm_start"] = ip_warm_start
        elif ip_barrier_tol is not None:
            info["ip_barrier_tol"] = ip_barrier_tol

    ms = case.ms_solver(box_scale=box_scale) if "box_scale" in info else case.ms_solver()
    if stop == "kkt":
        if case.ip_solver is not None and baseline != "fixed":
            raise ValueError("--stop kkt needs --baseline fixed")
        info["stop"] = stop
        ms = replace(ms, options=replace(ms.options, stopping="natural"))
    res, wall = _run_method(lambda x0, u0: mirror_mpc(ms, x0, u0, inp.steps), inp.x0, inp.u_init)
    out.update({f"ms_{k}": v for k, v in res.items()})
    info["ms"] = {
        "wall_s": round(wall, 3),
        "iters_total": int(res["iterations"].sum()),
        "iters_max": int(res["iterations"].max()),
        "not_converged": int((res["status"] != 0).sum()),
    }
    if stop == "kkt":
        info["ms"]["stalled"] = int((res["status"] == Status.STALLED).sum())

    if case.ip_solver is not None:
        if baseline == "fixed":
            ip, warm = case.ip_solver(IPDDPOptions(stopping="kkt" if stop == "kkt" else "barrier")), ip_warm_start
        else:
            ip = case.ip_solver(IPDDPOptions.paper(barrier_tol=ip_barrier_tol)) if "ip_barrier_tol" in info else case.ip_solver()
            warm = False  # the paper's protocol: restart slacks, duals and mu at every step
        res, wall = _run_method(
            lambda x0, u0: ipddp_mpc(ip, x0, u0, inp.ip_slacks, inp.ip_duals, inp.steps, warm_start=warm), inp.x0, inp.u_init
        )
        out.update({f"ip_{k}": v for k, v in res.items()})
        info["ip"] = {
            "wall_s": round(wall, 3),
            "iters_total": int(res["iterations"].sum()),
            "iters_max": int(res["iterations"].max()),
            "not_converged": int((res["status"] != 0).sum()),
        }
    return out, info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the msilqr benchmark cases.")
    p.add_argument("--out", required=True)
    p.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    p.add_argument("--box-scale", type=float, help="variant: cart-pole box-map scale (original 0.5; paper Table I: 1)")
    p.add_argument("--ip-barrier-tol", type=float, help="variant: the paper's IPDDP also requires barrier parameter < this to stop")
    p.add_argument("--baseline", choices=["paper", "fixed"], default="paper", help="interior-point baseline (default: as in the paper)")
    p.add_argument("--ip-cold-start", action="store_true", help="fixed baseline: restart slacks, duals and mu every MPC step")
    p.add_argument("--stop", choices=["native", "kkt"], default="native", help="kkt: MS natural residual, IPDDP KKT error E_0")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    summary = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "cases": [],
    }
    for name in args.cases:
        arrays, info = run_case(
            name,
            box_scale=args.box_scale,
            ip_barrier_tol=args.ip_barrier_tol,
            baseline=args.baseline,
            ip_warm_start=not args.ip_cold_start,
            stop=args.stop,
        )
        np.savez(os.path.join(args.out, name + ".npz"), **arrays)
        summary["cases"].append(info)
        line = f"[run] {name:18s} MS iters {info['ms']['iters_total']:6d} (max {info['ms']['iters_max']:3d}, {info['ms']['wall_s']:.1f}s)"
        if "ip" in info:
            line += f" | IP iters {info['ip']['iters_total']:6d} (max {info['ip']['iters_max']:3d}, {info['ip']['wall_s']:.1f}s)"
        print(line, flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
