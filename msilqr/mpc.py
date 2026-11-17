"""Receding-horizon (MPC) closed loops for both solvers.

Each step solves the finite-horizon problem from the current state and applies the first
control. With ``warm_start=True`` (the default) the next solve starts from the previous
plan shifted by one step; with ``warm_start=False`` every solve starts from the initial
guess passed to the loop.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from msilqr.common import as_float64 as _as_float64
from msilqr.common import check_trajectory_shapes

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


def _mirror_mpc_fn(solver, num_steps: int, warm_start: bool):
    def run(x0, u_init):
        def body(carry, _):
            state, controls = carry
            sol = solver.solve(state, controls, validate=False)  # shifted plans may touch the bound
            out = (sol.states[1], sol.controls[0], sol.iterations, sol.status, sol.cost)
            return (sol.states[1], _shift(sol.controls) if warm_start else u_init), out

        _, (xs, us, its, status, costs) = jax.lax.scan(body, (x0, u_init), None, length=num_steps)
        return MPCResult(jnp.vstack((x0, xs)), us, its, status, costs)

    return jax.jit(run)


def mirror_mpc(solver, x0: jax.Array, u_init: jax.Array, num_steps: int, *, warm_start: bool = True) -> MPCResult:
    """Closed loop of :class:`~msilqr.msilqr.MirrorILQR`.

    Args:
        solver: the mirror-space solver (its problem defines the model).
        x0: initial state ``(n,)``.
        u_init: initial control sequence ``(T, m)``, strictly feasible (checked).
        num_steps: number of MPC steps.
        warm_start: start every solve after the first from the previous plan shifted by one
            step (the last control repeated). ``False``: start every solve from ``u_init``.
    """
    check_trajectory_shapes(x0, u_init)
    x0, u_init = _as_float64(x0), _as_float64(u_init)
    solver.validate_controls(u_init)
    key = ("mirror", int(num_steps), bool(warm_start))
    fn = _cached(solver, key, lambda: _mirror_mpc_fn(solver, int(num_steps), bool(warm_start)))
    return fn(x0, u_init)


def _ipddp_mpc_fn(solver, num_steps: int, warm_start: bool):
    dynamics = solver.problem.dynamics
    mu_floor = solver._mu_floor()

    def run(x0, u_init, slacks, duals):
        if warm_start and num_steps > 0:

            def solve_step(state, controls, s, z, barrier):
                sol = solver.solve(state, controls, s, z, barrier=barrier)
                applied = sol.controls[0]
                next_state = dynamics(state, applied)
                out = (next_state, applied, sol.iterations, sol.status)
                return (next_state, _shift(sol.controls), _shift(sol.slacks), _shift(sol.duals)), out

            def body(carry, _):
                state, controls, s, z = carry
                return solve_step(state, controls, s, z, jnp.maximum(jnp.mean(s * z), mu_floor))  # centred at the warm start

            # the given slacks and duals are not a warm start: the first step uses the initial-barrier rule
            carry, first = solve_step(x0, u_init, slacks, duals, None)
            _, rest = jax.lax.scan(body, carry, None, length=num_steps - 1)
            xs, us, its, status = (jnp.concatenate((f[None], r)) for f, r in zip(first, rest, strict=True))
        else:

            def body(state, _):
                sol = solver.solve(state, u_init, slacks, duals)
                applied = sol.controls[0]
                next_state = dynamics(state, applied)
                return next_state, (next_state, applied, sol.iterations, sol.status)

            _, (xs, us, its, status) = jax.lax.scan(body, x0, None, length=num_steps)
        return MPCResult(jnp.vstack((x0, xs)), us, its, status, jnp.full(num_steps, jnp.nan))

    return jax.jit(run)


def ipddp_mpc(
    solver, x0: jax.Array, u_init: jax.Array, slacks: jax.Array, duals: jax.Array, num_steps: int, *, warm_start: bool = True
) -> MPCResult:
    """Closed loop of :class:`~msilqr.ipddp.InteriorPointDDP`.

    With ``warm_start=True`` every MPC step after the first starts from the shifted
    controls, slacks and duals of the previous plan, with the barrier parameter set to their
    average complementarity ``mean(s z)`` (at least the solver's smallest ``mu``), so the warm start
    lies near the central path. The first step starts from the given slacks and duals with
    the initial barrier rule. With ``warm_start=False`` every step starts from the given
    ``u_init``, ``slacks`` and ``duals`` with the initial barrier rule.
    """
    check_trajectory_shapes(x0, u_init, slacks, duals)
    x0, u_init, slacks, duals = (_as_float64(a) for a in (x0, u_init, slacks, duals))
    solver.validate_slacks_duals(slacks, duals)
    key = ("ipddp", int(num_steps), bool(warm_start))
    fn = _cached(solver, key, lambda: _ipddp_mpc_fn(solver, int(num_steps), bool(warm_start)))
    return fn(x0, u_init, slacks, duals)
