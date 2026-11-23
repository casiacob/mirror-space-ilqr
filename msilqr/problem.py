"""Optimal control problem definition and trajectory helpers (Sec. II of the paper).

``minimize  J = c_T(x_T) + sum_{t=0}^{T-1} c_t(x_t, u_t)``
``subject to x_{t+1} = f(x_t, u_t),  x_0 = x_s,  u_t in U``

The control constraint ``u_t in U`` is not part of :class:`OptimalControlProblem`: the
mirror-space solver encodes it in a :class:`~msilqr.maps.MirrorMap`, the
interior-point baseline in an explicit inequality ``g(x, u) <= 0``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = ["OptimalControlProblem", "rollout", "euler"]

Dynamics = Callable[[jax.Array, jax.Array], jax.Array]


@dataclass(frozen=True, eq=False)
class OptimalControlProblem:
    """Discrete-time finite-horizon problem with time-invariant costs.

    Attributes:
        dynamics:   ``f(x, u) -> x_next`` (discrete time).
        stage_cost: ``c(x, u) -> scalar``, twice differentiable.
        final_cost: ``c_T(x) -> scalar``, twice differentiable.
    """

    dynamics: Dynamics
    stage_cost: Callable[[jax.Array, jax.Array], jax.Array]
    final_cost: Callable[[jax.Array], jax.Array]

    def total_cost(self, states: jax.Array, controls: jax.Array) -> jax.Array:
        """``J`` for ``states`` of shape ``(T+1, n)`` and ``controls`` of shape ``(T, m)``."""
        stage = jax.vmap(self.stage_cost)(states[:-1], controls)
        return self.final_cost(states[-1]) + jnp.sum(stage)

    def rollout(self, x0: jax.Array, controls: jax.Array) -> jax.Array:
        """Simulate the dynamics; returns states of shape ``(T+1, n)`` with ``x0`` first."""
        return rollout(self.dynamics, x0, controls)


def rollout(dynamics: Dynamics, x0: jax.Array, controls: jax.Array) -> jax.Array:
    """Open-loop simulation ``x_{t+1} = f(x_t, u_t)``; returns ``(T+1, n)`` states."""

    def body(x, u):
        x_next = dynamics(x, u)
        return x_next, x_next

    _, states = jax.lax.scan(body, x0, controls)
    return jnp.vstack((x0, states))


def euler(ode: Dynamics, dt: float) -> Dynamics:
    """Explicit Euler discretization ``x_{t+1} = x_t + dt * ode(x_t, u_t)``."""

    def dynamics(x, u):
        return x + dt * ode(x, u)

    return dynamics
