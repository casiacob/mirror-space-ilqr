"""Receding-horizon (MPC) closed loops for both solvers.

Each MPC step solves the finite-horizon problem from the current state, applies the
first control, and warm-starts the next solve with the plan shifted by one step (the
last control repeated). The model is exact, so the next state equals the first
predicted state of the plan.

Both loops are jit-compiled once per solver and number of steps (the compiled function
is cached on the solver object and released with it), and record the APPLIED control
of every step (the original interior-point scripts recorded the next planned control
instead).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from msilqr.mirror_ilqr import _as_float64

__all__ = ["MPCResult", "mirror_mpc", "ipddp_mpc"]


def _cached(solver, key, build):
    cache = solver.__dict__.setdefault("_mpc_cache", {})  # frozen dataclass: mutate the dict, not attributes
    if key not in cache:
        cache[key] = build()
    return cache[key]


class MPCResult(NamedTuple):
    states: jax.Array  #: ``(S+1, n)`` closed-loop states, ``x0`` first
    controls: jax.Array  #: ``(S, m)`` applied controls
    iterations: jax.Array  #: ``(S,)`` solver iterations per MPC step
    status: jax.Array  #: ``(S,)`` solver status per MPC step (0 = converged)
    plan_costs: jax.Array  #: ``(S,)`` cost of each optimized plan (``nan`` if not tracked)


def _shift(a):
    return jnp.vstack((a[1:], a[-1]))


def _mirror_mpc_fn(solver, num_steps: int):
    def run(x0, u_init):
        def body(carry, _):
            state, controls = carry
            sol = solver.solve(state, controls)
            out = (sol.states[1], sol.controls[0], sol.iterations, sol.status, sol.cost)
            return (sol.states[1], _shift(sol.controls)), out

        _, (xs, us, its, status, costs) = jax.lax.scan(body, (x0, u_init), None, length=num_steps)
        return MPCResult(jnp.vstack((x0, xs)), us, its, status, costs)

    return jax.jit(run)


def mirror_mpc(solver, x0: jax.Array, u_init: jax.Array, num_steps: int) -> MPCResult:
    """Closed loop of :class:`~msilqr.mirror_ilqr.MirrorILQR`.

    Args:
        solver: the mirror-space solver (its problem defines the model).
        x0: initial state ``(n,)``.
        u_init: initial control sequence ``(T, m)``, strictly feasible (checked).
        num_steps: number of MPC steps.
    """
    x0, u_init = _as_float64(x0), _as_float64(u_init)
    solver.validate_controls(u_init)
    fn = _cached(solver, ("mirror", int(num_steps)), lambda: _mirror_mpc_fn(solver, int(num_steps)))
    return fn(x0, u_init)


def _ipddp_mpc_fn(solver, num_steps: int):
    dynamics = solver.problem.dynamics

    def run(x0, u_init, slacks, duals):
        def body(carry, _):
            state, controls = carry
            sol = solver.solve(state, controls, slacks, duals)
            applied = sol.controls[0]
            next_state = dynamics(state, applied)
            out = (next_state, applied, sol.iterations, sol.status)
            return (next_state, _shift(sol.controls)), out

        _, (xs, us, its, status) = jax.lax.scan(body, (x0, u_init), None, length=num_steps)
        return MPCResult(jnp.vstack((x0, xs)), us, its, status, jnp.full(num_steps, jnp.nan))

    return jax.jit(run)


def ipddp_mpc(solver, x0: jax.Array, u_init: jax.Array, slacks: jax.Array, duals: jax.Array, num_steps: int) -> MPCResult:
    """Closed loop of :class:`~msilqr.ipddp.InteriorPointDDP`.

    As in the paper's experiments, the controls are warm-started (shifted plan) but every
    MPC step restarts from the given ``slacks`` and ``duals``, and the barrier parameter
    and regularization are re-initialized. Warm-starting only the slacks and duals is not
    an improvement: with the barrier parameter reset to its large initial value the
    complementarity residual is inconsistent and every second solve stalls at the
    iteration cap (observed on the double integrator).
    """
    x0, u_init, slacks, duals = (_as_float64(a) for a in (x0, u_init, slacks, duals))
    fn = _cached(solver, ("ipddp", int(num_steps)), lambda: _ipddp_mpc_fn(solver, int(num_steps)))
    return fn(x0, u_init, slacks, duals)
