"""Regression gate: compare new benchmark results against the golden legacy outputs.

This module is deliberately independent of the ``msilqr`` package. It re-implements
the original costs, dynamics and constraints of the four shipped experiments in plain
NumPy, so a bug in the new package cannot hide itself from the check.

Standard result format
----------------------
Every case is stored as ``<results_dir>/<case>.npz`` with these arrays
(``N`` initial conditions, ``T`` MPC steps)::

    x0                (N, nx)        initial states
    ms_states         (N, T+1, nx)   closed-loop states of the mirror-space solver, x0 first
    ms_controls       (N, T, nu)     applied controls
    ms_iterations     (N, T)         solver iterations per MPC step
    ms_status         (N, T)         solver status per MPC step (0 = converged; optional)
    ip_states / ip_controls / ip_iterations   same for the interior-point baseline (compare cases)

Cases: ``cartpole`` (examples/cartpole.py), ``cartpole-compare``
(examples/cartpole-mm-v-ip.py), ``polytope`` (examples/wiener-polytope.py) and
``polytope-compare`` (examples/wiener-polytope-mm-v-ip.py).

Gate (per case and method, aggregated over initial conditions)
--------------------------------------------------------------
* no NaN/inf in states or controls;
* closed-loop trajectories are consistent with the dynamics (|x_{t+1} - f(x_t,u_t)| small);
* every MPC solve converged (if statuses are reported);
* total and maximum iterations per MPC step do not increase;
* mean closed-loop cost does not increase beyond a relative float tolerance;
* constraint satisfaction does not get worse (mirror-space must stay feasible);
* distance to the goal at the end of the closed loop does not get worse beyond tolerance.

Trajectory deviations from the golden run are reported but do not fail the gate on
their own: an algorithmic improvement may legitimately change them.

Usage::

    python benchmarks/regression.py --new results/ [--golden benchmarks/legacy/golden]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLDEN = os.path.join(HERE, "legacy", "golden")

# ----------------------------------------------------------------------------------
# Independent re-implementation of the original experiments (NumPy, float64)
# ----------------------------------------------------------------------------------

POLY_A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]], float)
POLY_B = np.array([0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7])


def _cartpole_ode(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    g, l, mc, mp = 9.81, 0.5, 10.0, 1.0
    _, th, _, thd = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    s, c, a = np.sin(th), np.cos(th), u[..., 0]
    xdd = (a + mp * s * (l * thd**2 + g * c)) / (mc + mp * s**2)
    thdd = (-a * c - mp * l * thd**2 * c * s - (mc + mp) * g * s) / (l * mc + l * mp * s**2)
    return np.stack([x[..., 2], thd, xdd, thdd], axis=-1)


def _cartpole_step(x, u):
    return x + 0.01 * _cartpole_ode(x, u)


def _double_integrator_step(x, u):
    return x + 0.02 * np.concatenate([x[..., 2:], u], axis=-1)


def _quad(e, w):
    return 0.5 * np.einsum("...i,i,...i->...", e, w, e)


def _cartpole_cost(xs, us):
    def err(x):
        wrapped = np.concatenate([x[..., :1], np.mod(x[..., 1:2], 2 * np.pi), x[..., 2:]], axis=-1)
        return wrapped - np.array([0.0, np.pi, 0.0, 0.0])

    q = np.array([10.0, 10.0, 0.1, 0.1])
    stage = _quad(err(xs[..., :-1, :]), q) + _quad(us, np.array([1e-3]))
    return stage.sum(-1) + _quad(err(xs[..., -1, :]), q)


def _double_integrator_cost(xs, us):
    q, goal = np.array([10.0, 10.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.0, 0.0])
    stage = _quad(xs[..., :-1, :] - goal, q) + _quad(us, np.array([0.05, 0.05]))
    return stage.sum(-1) + _quad(xs[..., -1, :] - goal, q)


def _cartpole_goal_error(xs):
    x = xs[..., -1, :]
    ang = np.mod(x[..., 1] - np.pi + np.pi, 2 * np.pi) - np.pi  # angle error to pi, wrapped to [-pi, pi)
    return np.sqrt(x[..., 0] ** 2 + ang**2 + x[..., 2] ** 2 + x[..., 3] ** 2)


def _double_integrator_goal_error(xs):
    return np.linalg.norm(xs[..., -1, :] - np.array([1.0, 1.0, 0.0, 0.0]), axis=-1)


def _cartpole_applied_controls(xs):
    """Recover u_t from x_t, x_{t+1} (the Euler cart-pole step is affine in u)."""
    g, l, mc, mp, dt = 9.81, 0.5, 10.0, 1.0, 0.01
    x, xn = xs[..., :-1, :], xs[..., 1:, :]
    s, c, thd = np.sin(x[..., 1]), np.cos(x[..., 1]), x[..., 3]
    xdd = (xn[..., 2] - x[..., 2]) / dt
    return (xdd * (mc + mp * s**2) - mp * s * (l * thd**2 + g * c))[..., None]


def _double_integrator_applied_controls(xs):
    return (xs[..., 1:, 2:] - xs[..., :-1, 2:]) / 0.02


def _box_violation(us):  # max_t (|u| - 50): <= 0 means feasible
    return (np.abs(us) - 50.0).max(axis=(-1, -2))


def _polytope_violation(us):  # max_t max_j (A u - b)_j
    return (np.einsum("...ti,ji->...tj", us, POLY_A) - POLY_B).max(axis=(-1, -2))


@dataclass(frozen=True)
class Case:
    name: str
    legacy_file: str
    step: callable
    cost: callable
    goal_error: callable
    violation: callable
    applied_controls: callable = None
    methods: tuple = ("ms",)


CASES = {
    c.name: c
    for c in (
        Case("cartpole", "cartpole", _cartpole_step, _cartpole_cost, _cartpole_goal_error, _box_violation, _cartpole_applied_controls),
        Case(
            "cartpole-compare",
            "cartpole-mm-v-ip",
            _cartpole_step,
            _cartpole_cost,
            _cartpole_goal_error,
            _box_violation,
            _cartpole_applied_controls,
            ("ms", "ip"),
        ),
        Case(
            "polytope",
            "wiener-polytope",
            _double_integrator_step,
            _double_integrator_cost,
            _double_integrator_goal_error,
            _polytope_violation,
            _double_integrator_applied_controls,
        ),
        Case(
            "polytope-compare",
            "wiener-polytope-mm-v-ip",
            _double_integrator_step,
            _double_integrator_cost,
            _double_integrator_goal_error,
            _polytope_violation,
            _double_integrator_applied_controls,
            ("ms", "ip"),
        ),
    )
}

# The legacy interior-point MPC loops (examples/*-mm-v-ip.py, ip_mpc_body) apply
# control[0] but log control[0] of the *shifted* plan, i.e. the next planned control.
# The logged IP controls are therefore not the applied ones. Because both systems are
# affine in the control, the applied controls are recovered exactly from consecutive
# closed-loop states (validated against the correctly logged mirror-space controls).
LEGACY_IP_CONTROLS_SHIFTED = True
RECONSTRUCTION_ATOL = 1e-9

# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------


def load_legacy(case: Case, golden_dir: str) -> dict[str, np.ndarray]:
    """Map the legacy example globals to the standard result format."""
    d = np.load(os.path.join(golden_dir, case.legacy_file + ".npz"))
    out: dict[str, np.ndarray] = {}
    if case.name == "cartpole":
        out["x0"] = d["x0"][None]
        out["ms_states"] = d["mm_mpc_states"][None]
        out["ms_controls"] = d["mm_mpc_controls"][None]
        out["ms_iterations"] = d["mpc_iterations"][None]
    elif case.name == "polytope":
        out["x0"] = d["x0"][None]
        out["ms_states"] = d["mm_mpc_states"][None]
        out["ms_controls"] = d["mm_mpc_controls"][None]
        if "mpc_iterations_instrumented" in d.files:  # see benchmarks/legacy/capture.py
            out["ms_iterations"] = d["mpc_iterations_instrumented"][None]
    else:
        x0 = d["x0_samples"]
        out["x0"] = x0
        for m, legacy in (("ms", "mm"), ("ip", "ip")):
            xs = d[f"all_{legacy}_mpc_states"]
            if xs.shape[1] == d[f"all_{legacy}_mpc_controls"].shape[1]:  # x0 not included
                xs = np.concatenate([x0[:, None], xs], axis=1)
            out[f"{m}_states"] = xs
            out[f"{m}_controls"] = d[f"all_{legacy}_mpc_controls"]
            out[f"{m}_iterations"] = d[f"all_{legacy}_mpc_iterations"]
        if LEGACY_IP_CONTROLS_SHIFTED:
            # sanity: the reconstruction reproduces the correctly logged MS controls
            rec_ms = case.applied_controls(out["ms_states"])
            err = float(np.abs(rec_ms - out["ms_controls"]).max())
            if err > RECONSTRUCTION_ATOL:
                raise RuntimeError(f"{case.name}: control reconstruction check failed on MS data ({err:.1e})")
            out["ip_controls_logged"] = out["ip_controls"]
            out["ip_controls"] = case.applied_controls(out["ip_states"])
    return out


def load_new(case: Case, new_dir: str) -> dict[str, np.ndarray]:
    path = os.path.join(new_dir, case.name + ".npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing result file {path}")
    d = np.load(path)
    return {k: d[k] for k in d.files}


# ----------------------------------------------------------------------------------
# Metrics and gate
# ----------------------------------------------------------------------------------


def metrics(case: Case, res: dict[str, np.ndarray], method: str) -> dict:
    xs, us = res[f"{method}_states"], res[f"{method}_controls"]
    m = {
        "n_init": int(xs.shape[0]),
        "finite": bool(np.isfinite(xs).all() and np.isfinite(us).all()),
        "dynamics_residual": float(np.abs(xs[:, 1:] - case.step(xs[:, :-1], us)).max()),
        "x0_matches": bool(np.allclose(xs[:, 0], res["x0"], atol=0, rtol=0)),
        "cost_mean": float(case.cost(xs, us).mean()),
        "cost_per_run": case.cost(xs, us).tolist(),
        "violation_max": float(case.violation(us).max()),
        "goal_error_mean": float(case.goal_error(xs).mean()),
    }
    key = f"{method}_iterations"
    if key in res:
        it = np.asarray(res[key])
        m.update(iters_total=int(it.sum()), iters_max=int(it.max()), iters_mean=float(it.mean()))
    return m


@dataclass
class Verdict:
    case: str
    method: str
    checks: list = field(default_factory=list)  # (name, ok, detail)

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


COST_RTOL = 1e-9  # float noise allowance on the mean closed-loop cost
GOAL_ATOL = 1e-6  # absolute allowance on the final goal error
DYN_ATOL = 1e-9  # closed loop must follow the dynamics


def gate(case: Case, golden: dict, new: dict, method: str) -> Verdict:
    g, n = metrics(case, golden, method), metrics(case, new, method)
    v = Verdict(case.name, method)
    v.checks.append(("finite", n["finite"], ""))
    # Every golden solve converged by construction (the original loops had no iteration
    # cap and stopped only on the tolerance), so any non-converged new solve is a regression.
    status_key = f"{method}_status"
    if status_key not in new:
        v.checks.append(("all MPC solves converged", False, f"{status_key} missing"))
    else:
        n_bad = int(np.sum(np.asarray(new[status_key]) != 0))
        v.checks.append(("all MPC solves converged", n_bad == 0, f"{n_bad} not converged"))
    v.checks.append(("x0 identical", n["x0_matches"] and np.array_equal(new["x0"], golden["x0"]), ""))
    v.checks.append(("dynamics consistent", n["dynamics_residual"] <= DYN_ATOL, f"{n['dynamics_residual']:.1e}"))
    if "iters_total" in g:
        if "iters_total" not in n:
            v.checks.append(("iterations reported", False, "missing"))
        else:
            v.checks.append(("total iterations", n["iters_total"] <= g["iters_total"], f"{g['iters_total']} -> {n['iters_total']}"))
            v.checks.append(("max iterations/step", n["iters_max"] <= g["iters_max"], f"{g['iters_max']} -> {n['iters_max']}"))
    v.checks.append(
        ("mean closed-loop cost", n["cost_mean"] <= g["cost_mean"] * (1 + COST_RTOL), f"{g['cost_mean']:.6f} -> {n['cost_mean']:.6f}")
    )
    feas_limit = 0.0 if method == "ms" else max(g["violation_max"], 0.0)
    v.checks.append(("constraints", n["violation_max"] <= feas_limit, f"{g['violation_max']:+.2e} -> {n['violation_max']:+.2e}"))
    v.checks.append(
        (
            "goal error",
            n["goal_error_mean"] <= g["goal_error_mean"] + GOAL_ATOL,
            f"{g['goal_error_mean']:.2e} -> {n['goal_error_mean']:.2e}",
        )
    )
    dev = float(np.abs(new[f"{method}_states"] - golden[f"{method}_states"]).max())
    v.checks.append(("(info) max state deviation from golden", True, f"{dev:.2e}"))
    return v


def bitwise_checks(golden: dict, new: dict, method: str) -> list:
    """Exact equality with the golden run (states, iterations; controls exact for MS).

    The legacy IP controls are reconstructed from states (see LEGACY_IP_CONTROLS_SHIFTED),
    which is exact only up to ~1e-12, so IP controls are compared with a tolerance.
    """
    checks = []
    for field_ in ("states", "iterations"):
        key = f"{method}_{field_}"
        if key in golden:
            ok = key in new and new[key].shape == golden[key].shape and np.array_equal(new[key], golden[key])
            checks.append((f"bitwise {field_}", bool(ok), ""))
    key = f"{method}_controls"
    if method == "ip" and LEGACY_IP_CONTROLS_SHIFTED:
        err = float(np.abs(new[key] - golden[key]).max())
        checks.append(("controls (vs reconstructed)", err <= RECONSTRUCTION_ATOL, f"{err:.1e}"))
    else:
        checks.append(("bitwise controls", bool(np.array_equal(new[key], golden[key])), ""))
    return checks


def run(new_dir: str, golden_dir: str, cases: list[str], bitwise: bool = False) -> list[Verdict]:
    verdicts = []
    for name in cases:
        case = CASES[name]
        golden, new = load_legacy(case, golden_dir), load_new(case, new_dir)
        for method in case.methods:
            v = gate(case, golden, new, method)
            if bitwise:
                v.checks.extend(bitwise_checks(golden, new, method))
            verdicts.append(v)
    return verdicts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Regression gate against the golden legacy outputs.")
    p.add_argument("--new", required=True, help="directory with <case>.npz result files")
    p.add_argument("--golden", default=DEFAULT_GOLDEN)
    p.add_argument("--cases", nargs="*", default=list(CASES))
    p.add_argument("--json", help="write a machine-readable report here")
    p.add_argument("--bitwise", action="store_true", help="also require exact equality with the golden run")
    args = p.parse_args(argv)

    verdicts = run(args.new, args.golden, args.cases, bitwise=args.bitwise)
    for v in verdicts:
        print(f"[{'PASS' if v.ok else 'FAIL'}] {v.case} / {v.method.upper()}")
        for name, ok, detail in v.checks:
            mark = " " if name.startswith("(info)") else ("✓" if ok else "✗")
            print(f"    {mark} {name:40s} {detail}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{"case": v.case, "method": v.method, "ok": v.ok, "checks": v.checks} for v in verdicts], f, indent=2)
    n_fail = sum(not v.ok for v in verdicts)
    print(f"\n{len(verdicts) - n_fail}/{len(verdicts)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
