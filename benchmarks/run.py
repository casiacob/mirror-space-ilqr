"""Run the benchmark cases and write results in the standard format.

Usage (from the repository root)::

    python benchmarks/run.py --out results            # all cases
    python benchmarks/run.py --out results --cases cartpole polytope
    python benchmarks/regression.py --new results --bitwise   # check against golden

Writes ``<out>/<case>.npz`` (see ``benchmarks/regression.py`` for the format) and
``<out>/summary.json`` (iterations, statuses, timings, package versions).
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

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import CASES  # noqa: E402
from msilqr import ipddp_mpc, mirror_mpc  # noqa: E402


def _run_method(fn, x0s):
    """Run one closed loop per initial state with a single compiled function."""
    t0 = time.perf_counter()
    results = [fn(x0) for x0 in x0s]
    jax.block_until_ready(results)
    wall = time.perf_counter() - t0
    stack = {f: np.stack([np.asarray(getattr(r, f)) for r in results]) for f in ("states", "controls", "iterations", "status")}
    return stack, wall


def run_case(name: str) -> tuple[dict, dict]:
    case = CASES[name]
    inp = case.inputs()
    out = {"x0": np.asarray(inp.x0)}
    info = {"case": name, "description": case.description, "n_init": int(inp.x0.shape[0]), "steps": inp.steps}

    ms = case.ms_solver()
    res, wall = _run_method(lambda x0: mirror_mpc(ms, x0, inp.u_init, inp.steps), inp.x0)
    out.update({f"ms_{k}": v for k, v in res.items()})
    info["ms"] = {"wall_s": round(wall, 3), "iters_total": int(res["iterations"].sum()), "iters_max": int(res["iterations"].max()),
                  "not_converged": int((res["status"] != 0).sum())}

    if case.ip_solver is not None:
        ip = case.ip_solver()
        res, wall = _run_method(lambda x0: ipddp_mpc(ip, x0, inp.u_init, inp.ip_slacks, inp.ip_duals, inp.steps), inp.x0)
        out.update({f"ip_{k}": v for k, v in res.items()})
        info["ip"] = {"wall_s": round(wall, 3), "iters_total": int(res["iterations"].sum()), "iters_max": int(res["iterations"].max()),
                      "not_converged": int((res["status"] != 0).sum())}
    return out, info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the msilqr benchmark cases.")
    p.add_argument("--out", required=True)
    p.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    summary = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "cases": [],
    }
    for name in args.cases:
        arrays, info = run_case(name)
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
