"""Run the benchmark cases.

Usage (from the repository root)::

    python benchmarks/run.py --out results            # all cases
    python benchmarks/run.py --out results --cases cartpole polytope

Writes ``<out>/<case>.npz`` and ``<out>/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import CASES  # noqa: E402

from msilqr import Status, ipddp_mpc, mirror_mpc  # noqa: E402


def _run_method(fn, x0s, u_inits):
    """Run one closed loop per initial state with a single compiled function."""
    t0 = time.perf_counter()
    results = [fn(x0, u0) for x0, u0 in zip(x0s, u_inits, strict=True)]
    jax.block_until_ready(results)
    wall = time.perf_counter() - t0
    stack = {f: np.stack([np.asarray(getattr(r, f)) for r in results]) for f in ("states", "controls", "iterations", "status")}
    return stack, wall


def run_case(name: str, warm_start: bool = False) -> tuple[dict, dict]:
    """Run one case. By default every MPC solve of both methods starts from the initial
    guess; ``warm_start=True`` starts it from the previous plan shifted by one step."""
    case = CASES[name]
    inp = case.inputs()
    out = {"x0": np.asarray(inp.x0)}
    info = {"case": name, "description": case.description, "n_init": int(inp.x0.shape[0]), "steps": inp.steps, "warm_start": warm_start}

    ms = case.ms_solver()
    res, wall = _run_method(lambda x0, u0: mirror_mpc(ms, x0, u0, inp.steps, warm_start=warm_start), inp.x0, inp.u_init)
    out.update({f"ms_{k}": v for k, v in res.items()})
    info["ms"] = _summary(res, wall)

    if case.ip_solver is not None:
        ip = case.ip_solver()
        res, wall = _run_method(
            lambda x0, u0: ipddp_mpc(ip, x0, u0, inp.ip_slacks, inp.ip_duals, inp.steps, warm_start=warm_start), inp.x0, inp.u_init
        )
        out.update({f"ip_{k}": v for k, v in res.items()})
        info["ip"] = _summary(res, wall)
    return out, info


def _summary(res: dict, wall: float) -> dict:
    return {
        "wall_s": round(wall, 3),
        "iters_total": int(res["iterations"].sum()),
        "iters_max": int(res["iterations"].max()),
        "not_converged": int((res["status"] != 0).sum()),
        "stalled": int((res["status"] == Status.STALLED).sum()),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the msilqr benchmark cases.")
    p.add_argument("--out", required=True)
    p.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    p.add_argument("--warm-start", action="store_true", help="start every MPC solve of both methods from the previous plan")
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
            warm_start=args.warm_start,
        )
        np.savez(os.path.join(args.out, name + ".npz"), **arrays)
        summary["cases"].append(info)
        line = "[run] " + f"{name:18s}"
        for label, key in (("MS", "ms"), ("IP", "ip")):
            if key in info:
                r = info[key]
                line += f" | {label} iters {r['iters_total']:6d} (max {r['iters_max']:4d}, not converged {r['not_converged']})"
        print(line, flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
