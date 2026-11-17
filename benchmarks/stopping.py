"""Stopping criteria of mirror-space iLQR on the double integrator with polytope constraints.

The iterates do not depend on the stopping test, so every instance is run once for a fixed
number of iterations with all stopping measures recorded, next to the distance to a
reference optimum (the problem is a convex QP). The criteria are then evaluated on these
traces: where each one stops, and how far the returned controls are from the optimum.

Instances (initial states at a random distance and direction from the goal):

* cold: the first solve, from small random controls;
* warm: MPC steps, from the shifted plan of the previous solve.

Usage (from the repository root)::

    python benchmarks/stopping.py --out results/stopping
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cases import polytope_map  # noqa: E402

from msilqr import DoubleIntegrator, MirrorILQR, SolverOptions, Status  # noqa: E402
from msilqr.common import gain_ratio_update  # noqa: E402

HORIZON = 15
# criterion -> tolerances. "natural-grad" scales the tolerance with max(1, |grad J(u_init)|_inf),
# "natural-start" with the natural residual of the initial controls.
CRITERIA = {
    "natural": (1e-4, 3e-5, 1e-5, 3e-6, 1e-6),
    "natural-grad": (1e-4, 1e-5, 1e-6),
    "natural-start": (1e-2, 1e-3, 1e-4),
}
WRONG = 1e-2  # a returned control this far from the optimum (1 % of the control range) is wrong


# ------------------------------------------------------------------------ instances ---


def initial_states(n: int, key) -> jax.Array:
    """``n`` states at distance ``r`` (log-uniform in [0.05, 2]) and a uniform direction from
    the goal position, with velocities ``0.1 N(0, I)``: from inactive to saturated constraints."""
    r_key, phi_key, v_key = jax.random.split(key, 3)
    r = jnp.exp(jax.random.uniform(r_key, (n,), minval=jnp.log(0.05), maxval=jnp.log(2.0)))
    phi = jax.random.uniform(phi_key, (n,), maxval=2 * jnp.pi)
    goal = jnp.array(DoubleIntegrator().goal)
    position = goal[:2] + r[:, None] * jnp.stack((jnp.cos(phi), jnp.sin(phi)), axis=1)
    return jnp.concatenate((position, 0.1 * jax.random.normal(v_key, (n, 2))), axis=1)


def cold_instances(x0s, seeds: int):
    """Every initial state with ``seeds`` draws of the controls ``0.1 N(0, I)``."""
    xs, us = [], []
    for i, x0 in enumerate(x0s):
        for seed in range(seeds):
            xs.append(x0)
            us.append(0.1 * jax.random.normal(jax.random.PRNGKey(1000 * seed + i), (HORIZON, 2)))
    return jnp.stack(xs), jnp.stack(us)


def warm_instances(solver: MirrorILQR, x0s, u_init, steps):
    """MPC closed loops with ``solver``; returns the state and the shifted plan (the warm
    start) at the MPC steps ``steps``."""
    xs, us = [], []
    for x0 in x0s:
        state, plan = x0, u_init
        for step in range(max(steps) + 1):
            if step in steps:
                xs.append(state)
                us.append(plan)
            sol = solver.solve(state, plan, validate=False)
            state, plan = sol.states[1], jnp.vstack((sol.controls[1:], sol.controls[-1]))
    return jnp.stack(xs), jnp.stack(us)


# ------------------------------------------------------------------------ reference ---


def reference_optimum(problem, polytope, x0s, iters: int = 4000):
    """Optimal controls of every instance by accelerated projected gradient on the condensed
    QP (strongly convex, so the optimum is unique). Returns ``(u_star, J_star, residual)``."""

    def cost(x0, flat):
        controls = flat.reshape(HORIZON, 2)
        return problem.total_cost(problem.rollout(x0, controls), controls)

    eigenvalues = jnp.linalg.eigvalsh(jax.hessian(cost, 1)(x0s[0], jnp.zeros(2 * HORIZON)))  # constant Hessian
    lipschitz, kappa = eigenvalues[-1], eigenvalues[-1] / eigenvalues[0]
    momentum = (jnp.sqrt(kappa) - 1) / (jnp.sqrt(kappa) + 1)
    project = lambda flat: jax.vmap(polytope.project)(flat.reshape(HORIZON, 2)).reshape(-1)  # noqa: E731

    def solve(x0):
        gradient = jax.grad(cost, 1)

        def body(carry, _):
            u, v = carry
            u_new = project(v - gradient(x0, v) / lipschitz)
            return (u_new, u_new + momentum * (u_new - u)), None

        (u, _), _ = jax.lax.scan(body, (jnp.zeros(2 * HORIZON), jnp.zeros(2 * HORIZON)), None, length=iters)
        residual = jnp.max(jnp.abs(u - project(u - gradient(x0, u))))
        return u.reshape(HORIZON, 2), cost(x0, u), residual

    return jax.jit(jax.vmap(solve))(x0s)


def polytope_vertices(G, h) -> jax.Array:
    """Vertices of ``{u : G u <= h}`` in the plane (intersections of two facets that are feasible)."""
    vertices = []
    for i, j in itertools.combinations(range(len(h)), 2):
        A = np.array([G[i], G[j]])
        if abs(np.linalg.det(A)) > 1e-12:
            v = np.linalg.solve(A, np.array([h[i], h[j]]))
            if np.all(np.asarray(G) @ v <= np.asarray(h) + 1e-12):
                vertices.append(v)
    return jnp.asarray(np.unique(np.round(vertices, 12), axis=0))


# ---------------------------------------------------------------------------- traces ---


def make_trace(problem, polytope, iters: int):
    """``trace(x0, u_init, u_star)``: ``iters`` iterations of :class:`MirrorILQR` (the loop
    of ``MirrorILQR._solve`` without a stopping test) with the stopping measures of every
    iteration. Measures are taken at the iterate before the step, as in the solver; the
    ``*_after`` entries describe the iterate after it, which the solver would return."""
    solver = MirrorILQR(problem, polytope)
    opt = solver.options
    vertices = polytope_vertices(np.asarray(polytope.G), np.asarray(polytope.h))

    def open_loop_cost(x0, controls):
        return problem.total_cost(problem.rollout(x0, controls), controls)

    def trace(x0, u_init, u_star):
        states = problem.rollout(x0, u_init)

        def iteration(carry, _):
            states, controls, reg, reg_factor, cost = carry
            K, k, predicted, positive_definite, natural_error, heading = solver.backward_pass(states, controls, reg)
            gradient = jax.grad(open_loop_cost, 1)(x0, controls)
            # Frank-Wolfe gap sum_t max_{v in U} <g_t, u_t - v>: an upper bound of J - J* (J convex)
            gap = jnp.sum(jnp.max(jnp.sum(gradient * controls, axis=1)[:, None] - gradient @ vertices.T, axis=1))

            new_states, new_controls = solver.forward_pass(states, controls, K, k, heading)
            new_cost = problem.total_cost(new_states, new_controls)
            actual = new_cost - cost
            accept = jnp.logical_and(jnp.logical_and(actual < 0.0, predicted < 0.0), positive_definite)
            stalled = jnp.logical_and(
                jnp.logical_and(~accept, positive_definite), jnp.abs(predicted) <= 10.0 * jnp.finfo(cost.dtype).eps * jnp.abs(cost)
            )
            new_reg, reg_factor = gain_ratio_update(reg, reg_factor, actual / predicted, accept, opt)

            states = jnp.where(accept, new_states, states)
            step = jnp.where(accept, jnp.max(jnp.abs(new_controls - controls)), 0.0)
            controls = jnp.where(accept, new_controls, controls)
            new_cost = jnp.where(accept, new_cost, cost)
            record = {
                "natural": natural_error,
                "gap": gap,
                "predicted": predicted,
                "accept": accept,
                "stalled": stalled,
                "reg": reg,
                "cost": cost,
                "step": step,
                "error_after": jnp.max(jnp.abs(controls - u_star)),
                "cost_after": new_cost,
            }
            return (states, controls, new_reg, reg_factor, new_cost), record

        init = (states, u_init, jnp.array(opt.reg_init), jnp.array(opt.reg_factor_init), problem.total_cost(states, u_init))
        _, records = jax.lax.scan(iteration, init, None, length=iters)
        return records

    return jax.jit(jax.vmap(trace))


# ------------------------------------------------------------------------ evaluation ---


def first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or the number of columns if there is none."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1), mask.shape[1])


def evaluate(trace: dict, criterion: str, tol: float) -> dict:
    """Where ``criterion`` stops every instance (as ``MirrorILQR.solve`` would) and the
    distance of the returned controls to the optimum."""
    iters = trace["natural"].shape[1]
    measure = trace["natural"]
    if criterion == "natural-grad":
        measure = measure / np.maximum(1.0, trace["gradient0"])[:, None]
    elif criterion == "natural-start":
        measure = measure / measure[:, :1]
    converged = first_true(measure < tol)
    stalled = first_true(trace["stalled"])
    stop = np.minimum(np.minimum(converged, stalled), iters - 1)  # index of the last iteration run
    status = np.where(
        converged <= np.minimum(stalled, iters - 1), Status.CONVERGED, np.where(stalled < iters, Status.STALLED, Status.MAX_ITER)
    )
    return {"iterations": stop + 1, "status": status, "error": np.take_along_axis(trace["error_after"], stop[:, None], axis=1)[:, 0]}


def summarize(trace: dict, criterion: str, tol: float) -> dict:
    result = evaluate(trace, criterion, tol)
    status, error, n = result["status"], result["error"], len(result["status"])
    # iterations until the returned controls are within tol of the optimum (absolute tolerances only)
    oracle = first_true(trace["error_after"] <= tol) + 1 if "-" not in criterion else np.array([np.nan])
    return {
        "criterion": criterion,
        "tol": tol,
        "converged_pct": 100 * float(np.mean(status == Status.CONVERGED)),
        "stalled_pct": 100 * float(np.mean(status == Status.STALLED)),
        "max_iter_pct": 100 * float(np.mean(status == Status.MAX_ITER)),
        "iterations_median": float(np.median(result["iterations"])),
        "iterations_p95": float(np.percentile(result["iterations"], 95)),
        "oracle_iterations_median": float(np.median(oracle)),
        "error_median": float(np.median(error)),
        "error_p95": float(np.percentile(error, 95)),
        "error_max": float(error.max()),
        "wrong_pct": 100 * float(np.mean(error > WRONG)),
        "converged_but_wrong": int(np.sum((status == Status.CONVERGED) & (error > WRONG))),
        "n": n,
    }


def check_against_solver(problem, polytope, x0s, u_inits, trace: dict, n: int = 20) -> None:
    """The traces must stop where ``MirrorILQR.solve`` stops, with the same status."""
    solver = MirrorILQR(problem, polytope, SolverOptions(max_iter=trace["natural"].shape[1]))
    result = evaluate({key: value[:n] for key, value in trace.items()}, "natural", solver.options.tol)
    for i in range(min(n, len(x0s))):
        sol = solver.solve(x0s[i], u_inits[i], validate=False)
        expected = (int(result["iterations"][i]), int(result["status"][i]))
        if (int(sol.iterations), int(sol.status)) != expected:
            raise AssertionError(f"instance {i}: solver {int(sol.iterations), int(sol.status)}, trace {expected}")


def print_table(name: str, rows: list[dict]) -> None:
    print(f"\n{name} ({rows[0]['n']} instances)")
    print(
        f"{'criterion':13s} {'tol':>6s} | {'conv':>5s} {'stall':>5s} {'cap':>5s} | {'its med':>7s} {'p95':>5s} {'oracle':>6s} | "
        f"{'err med':>8s} {'p95':>8s} {'max':>8s} | {'wrong':>6s} {'conv+wrong':>10s}"
    )
    for r in rows:
        print(
            f"{r['criterion']:13s} {r['tol']:6.0e} | {r['converged_pct']:4.0f}% {r['stalled_pct']:4.0f}% {r['max_iter_pct']:4.0f}% | "
            f"{r['iterations_median']:7.0f} {r['iterations_p95']:5.0f} {r['oracle_iterations_median']:6.0f} | "
            f"{r['error_median']:8.1e} {r['error_p95']:8.1e} {r['error_max']:8.1e} | {r['wrong_pct']:5.1f}% {r['converged_but_wrong']:10d}"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stopping criteria of mirror-space iLQR on the polytope problem.")
    p.add_argument("--out", required=True)
    p.add_argument("--n-x0", type=int, default=100, help="number of initial states")
    p.add_argument("--seeds", type=int, default=2, help="cold starts per initial state")
    p.add_argument("--warm-steps", type=int, nargs="*", default=[1, 5, 20, 60], help="MPC steps whose warm starts are used")
    p.add_argument("--iters", type=int, default=1000, help="iterations per instance (the solver's default cap)")
    args = p.parse_args(argv)

    problem, polytope = DoubleIntegrator().problem(), polytope_map()
    x0s = initial_states(args.n_x0, jax.random.PRNGKey(0))
    sets = {"cold": cold_instances(x0s, args.seeds)}
    t0 = time.perf_counter()
    first_plan = 0.1 * jax.random.normal(jax.random.PRNGKey(12), (HORIZON, 2))
    sets["warm"] = warm_instances(MirrorILQR(problem, polytope), x0s, first_plan, args.warm_steps)
    print(f"[stopping] warm starts from {args.n_x0} closed loops: {time.perf_counter() - t0:.0f} s")

    os.makedirs(args.out, exist_ok=True)
    trace_fn = make_trace(problem, polytope, args.iters)
    gradient_fn = jax.jit(jax.vmap(jax.grad(lambda x0, u: problem.total_cost(problem.rollout(x0, u), u), 1)))
    summary = {"iters": args.iters, "criteria": CRITERIA, "wrong_threshold": WRONG, "sets": {}}
    for name, (xs, us) in sets.items():
        u_star, cost_star, residual = reference_optimum(problem, polytope, xs)
        t0 = time.perf_counter()
        trace = {key: np.asarray(value) for key, value in trace_fn(xs, us, u_star).items()}
        trace["gradient0"] = np.abs(np.asarray(gradient_fn(xs, us))).max(axis=(1, 2))  # |grad J(u_init)|_inf
        active = np.sum(np.asarray(polytope.h) - np.einsum("jk,ntk->ntj", np.asarray(polytope.G), np.asarray(u_star)) < 1e-9, axis=(1, 2))
        print(
            f"[stopping] {name}: {len(xs)} instances traced in {time.perf_counter() - t0:.0f} s; reference residual "
            f"<= {float(residual.max()):.1e}; {int(np.sum(active > 0))} instances have active constraints at the optimum"
        )
        check_against_solver(problem, polytope, xs, us, trace)
        rows = [summarize(trace, criterion, tol) for criterion, tolerances in CRITERIA.items() for tol in tolerances]
        print_table(name, rows)
        np.savez_compressed(
            os.path.join(args.out, f"{name}.npz"),
            x0=np.asarray(xs),
            u_init=np.asarray(us),
            u_star=np.asarray(u_star),
            cost_star=np.asarray(cost_star),
            **trace,
        )
        summary["sets"][name] = {"n": len(xs), "active_at_optimum": int(np.sum(active > 0)), "rows": rows}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
