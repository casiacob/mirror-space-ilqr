"""Receding-horizon (MPC) closed loops for both solvers.

Each MPC step solves the finite-horizon problem from the current state, applies the
first control, and warm-starts the next solve with the plan shifted by one step (the
last control repeated). The model is exact, so the next state equals the first
predicted state of the plan.

Both loops are jit-compiled once per solver and number of steps, and record the
APPLIED control of every step (the original interior-point scripts recorded the next
planned control instead).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from msilqr.problem import rollout

__all__ = ["MPCResult", "mirror_mpc", "ipddp_mpc"]


class MPCResult(NamedTuple):
    states: jax.Array  #: ``(S+1, n)`` closed-loop states, ``x0`` first
    controls: jax.Array  #: ``(S, m)`` applied controls
    iterations: jax.Array  #: ``(S,)`` solver iterations per MPC step
    status: jax.Array  #: ``(S,)`` solver status per MPC step (0 = converged)
    plan_costs: jax.Array  #: ``(S,)`` cost of each optimized plan (``nan`` if not tracked)


def _shift(a):
    return jnp.vstack((a[1:], a[-1]))


@functools.cache
def _mirror_mpc_fn(solver, num_steps: int):
    dynamics = solver.problem.dynamics

    def run(x0, u_init):
        def body(carry, _):
            nominal_states, nominal_controls = carry
            sol = solver.solve(nominal_states, nominal_controls)
            new_controls = _shift(sol.controls)
            new_states = rollout(dynamics, sol.states[1], new_controls)
            out = (sol.states[1], sol.controls[0], sol.iterations, sol.status, sol.cost)
            return (new_states, new_controls), out

        init = (rollout(dynamics, x0, u_init), u_init)
        _, (xs, us, its, status, costs) = jax.lax.scan(body, init, None, length=num_steps)
        return MPCResult(jnp.vstack((x0, xs)), us, its, status, costs)

    return jax.jit(run)


def mirror_mpc(solver, x0: jax.Array, u_init: jax.Array, num_steps: int) -> MPCResult:
    """Closed loop of :class:`~msilqr.mirror_ilqr.MirrorILQR`.

    Args:
        solver: the mirror-space solver (its problem defines the model).
        x0: initial state ``(n,)``.
        u_init: initial control sequence ``(T, m)``, strictly feasible.
        num_steps: number of MPC steps.
    """
    return _mirror_mpc_fn(solver, int(num_steps))(x0, u_init)


@functools.cache
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
    return _ipddp_mpc_fn(solver, int(num_steps))(x0, u_init, slacks, duals)
