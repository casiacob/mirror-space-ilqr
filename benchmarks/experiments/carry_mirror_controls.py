"""Carrying the mirror controls y instead of recomputing ybar = grad psi(ubar).

The paper (Sec. III-C) and the code define the nominal mirror controls as
``ybar = grad psi(ubar)``. In float64 this round trip truncates ``|ybar|`` of nearly
saturated controls (to about 18 for the half-scale cart-pole map). This script runs the
alternative that keeps the mirror controls ``y`` produced by the forward pass (also across
MPC warm starts) and computes ``Psi^{-1}`` from ``u = grad psi*(y)``, on ``cartpole`` and
``cartpole-compare`` (README, note 8).

Usage: ``python benchmarks/experiments/carry_mirror_controls.py`` (~30 s).
"""

from __future__ import annotations

from dataclasses import dataclass

from common import report  # sets up sys.path and float64; must come first

import jax
import jax.numpy as jnp
import numpy as np

from cases import CASES, cartpole_ms_solver
from msilqr import MirrorILQR, Solution
from msilqr._common import termination_status
from msilqr.problem import rollout


@dataclass(frozen=True, eq=False)
class CarryILQR(MirrorILQR):
    def forward_pass_y(self, states, ys, K, k):
        p, mmap = self.problem, self.mirror_map

        def step(x_prev, inp):
            x_nom, y_nom, k_t, K_t = inp
            y_new = y_nom + k_t + K_t @ (x_prev - x_nom)
            u_new = mmap.inverse(y_new, None)
            x_new = p.dynamics(x_prev, u_new)
            return x_new, (x_new, u_new, y_new)

        _, (xs, us, ys_new) = jax.lax.scan(step, states[0], (states[:-1], ys, k, K))
        return jnp.vstack((states[0], xs)), us, ys_new

    def solve_y(self, x0, ys):
        opt, total_cost = self.options, self.problem.total_cost
        controls = jax.vmap(self.mirror_map.inverse, in_axes=(0, None))(ys, None)
        states = rollout(self.problem.dynamics, x0, controls)

        def cond(c):
            return jnp.logical_and(jnp.logical_not(c[-1] < opt.tol), c[-2] < opt.max_iter)

        def body(c):
            states, controls, ys, reg, reg_factor, cost, it, _ = c
            K, k, predicted, feasible, err = self.backward_pass(states, controls, reg)
            nxs, nus, nys = self.forward_pass_y(states, ys, K, k)
            ncost = total_cost(nxs, nus)
            actual = ncost - cost
            ratio = actual / predicted
            accept = jnp.logical_and(jnp.logical_and(actual < 0.0, predicted < 0.0), feasible)
            reg_new = jnp.where(accept, reg * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3), reg * reg_factor)
            reg_new = jnp.clip(reg_new, opt.reg_min, opt.reg_max)
            sel = lambda a, b: jnp.where(accept, a, b)  # noqa: E731
            reg_factor_new = jnp.where(accept, 2.0, 2.0 * reg_factor)
            return sel(nxs, states), sel(nus, controls), sel(nys, ys), reg_new, reg_factor_new, sel(ncost, cost), it + 1, err

        init = (states, controls, ys, jnp.array(opt.reg_init), jnp.array(opt.reg_factor_init), total_cost(states, controls), 0, 1e10)
        states, controls, ys, reg, _, cost, it, err = jax.lax.while_loop(cond, body, init)
        return Solution(states, controls, cost, it, reg, err, termination_status(err < opt.tol, err, it)), ys


def carry_mpc(solver, x0, u_init, num_steps):
    ys0 = jax.vmap(solver.mirror_map.grad)(u_init)

    @jax.jit
    def run(x0, ys):
        def body(carry, _):
            state, ys = carry
            sol, ys_opt = solver.solve_y(state, ys)
            return (sol.states[1], jnp.vstack((ys_opt[1:], ys_opt[-1]))), (sol.states[1], sol.controls[0], sol.iterations, sol.status)

        _, (xs, us, its, status) = jax.lax.scan(body, (x0, ys), None, length=num_steps)
        return jnp.vstack((x0, xs)), us, its, status

    return run(x0, ys0)


def main():
    base = cartpole_ms_solver()
    solver = CarryILQR(base.problem, base.mirror_map, base.options)
    for case_name in ("cartpole", "cartpole-compare"):
        inp = CASES[case_name].inputs()
        res = [carry_mpc(solver, x0, inp.u_init, inp.steps) for x0 in inp.x0]
        new = {f"ms_{f}": np.stack([np.asarray(r[i]) for r in res]) for i, f in enumerate(("states", "controls", "iterations", "status"))}
        new["x0"] = np.asarray(inp.x0)
        report(case_name, f"{case_name}: carry y", new)


if __name__ == "__main__":
    main()
