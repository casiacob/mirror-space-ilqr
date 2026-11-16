"""Capture golden reference outputs from the ORIGINAL example scripts.

The revamped package must not regress on the experiments that shipped with the
original code. This tool runs the original, unmodified example scripts of a given
git revision and stores every array they produce (closed-loop states, controls and
per-MPC-step iteration counts of both solvers) so that the new implementation can be
compared against them (see ``benchmarks/regression.py``).

Usage (from the repository root)::

    python benchmarks/legacy/capture.py --ref effb14c --out benchmarks/legacy/golden

What it does:

1. ``git archive <ref>`` into a temporary directory (the working tree is untouched).
2. Runs each original example in its own Python process with that directory first on
   ``sys.path`` (so ``import msilqr`` resolves to the legacy package), with plotting
   disabled (Agg backend, ``plt.show`` is a no-op).
3. Saves all array-valued globals (and lists of equally shaped arrays, stacked) to
   ``<out>/<script>.npz`` plus timing and environment metadata to ``<out>/<script>.json``.

Compatibility shim: the original ``msilqr/ipddp.py`` calls
``jnp.clip(x, a_min=..., a_max=...)``. Current JAX releases removed these keyword
names in favour of ``min=`` / ``max=``. The shim below maps the old names to the new
ones (identical semantics) so the original algorithm runs unchanged. Nothing else in
the legacy code is altered.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

SCRIPTS = (
    "cartpole.py",
    "cartpole-mm-v-ip.py",
    "wiener-polytope.py",
    "wiener-polytope-mm-v-ip.py",
)

COMPAT_SHIMS = ["jnp.clip(a_min=, a_max=) -> jnp.clip(min=, max=)"]


def _install_compat_shims() -> None:
    import jax.numpy as jnp

    orig_clip = jnp.clip

    def clip_compat(x, *args, a_min=None, a_max=None, **kwargs):
        if a_min is not None:
            kwargs.setdefault("min", a_min)
        if a_max is not None:
            kwargs.setdefault("max", a_max)
        return orig_clip(x, *args, **kwargs)

    jnp.clip = clip_compat


def _run_one(legacy_root: str, script: str, out_prefix: str, ref: str) -> None:
    """Child-process entry point: execute one legacy script and save its arrays."""
    import runpy

    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, legacy_root)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda *a, **k: None

    import jax
    import numpy as np

    _install_compat_shims()

    t0 = time.perf_counter()
    namespace = runpy.run_path(os.path.join(legacy_root, "examples", script), run_name="__main__")
    wall = time.perf_counter() - t0

    arrays = {}
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        if isinstance(value, (jax.Array, np.ndarray)):
            arrays[name] = np.asarray(value)
        elif isinstance(value, list) and value and all(isinstance(v, (jax.Array, np.ndarray)) for v in value):
            if len({np.shape(v) for v in value}) == 1:
                arrays[name] = np.stack([np.asarray(v) for v in value])

    np.savez(out_prefix + ".npz", **arrays)

    meta = {
        "script": f"examples/{script}",
        "legacy_ref": ref,
        "wall_time_s_incl_compile": round(wall, 3),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "compat_shims": COMPAT_SHIMS,
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sorted(arrays.items())},
    }
    try:
        import optimistix

        meta["optimistix"] = optimistix.__version__
    except ImportError:  # pragma: no cover
        pass
    with open(out_prefix + ".json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", default="effb14c", help="git revision of the original code")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "golden"))
    parser.add_argument("--scripts", nargs="*", default=list(SCRIPTS))
    parser.add_argument("--run-one", nargs=3, metavar=("LEGACY_ROOT", "SCRIPT", "OUT_PREFIX"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one:
        _run_one(*args.run_one, ref=args.ref)
        return

    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="msilqr-legacy-") as legacy_root:
        archive = subprocess.run(["git", "-C", repo, "archive", args.ref], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", legacy_root], input=archive, check=True)
        for script in args.scripts:
            out_prefix = os.path.join(args.out, os.path.splitext(script)[0])
            print(f"[capture] {args.ref}:examples/{script}", flush=True)
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--ref", args.ref, "--run-one", legacy_root, script, out_prefix],
                check=True,
            )
    print(f"[capture] wrote {len(args.scripts)} reference files to {args.out}")


if __name__ == "__main__":
    main()
