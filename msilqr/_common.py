"""Pieces shared by the mirror-space solver and the interior-point baseline.

Both solvers are Levenberg-Marquardt-regularized DDP methods: they expand the stage
cost (or Lagrangian) and dynamics to second order along a nominal trajectory, adapt the
regularization with the gain-ratio rule of Madsen, Nielsen and Tingleff ([37], Eq. 2.21),
and stop on a tolerance or an iteration cap.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import grad, hessian, jacrev

__all__ = [
    "Status",
    "RegularizationOptions",
    "StageExpansion",
    "expand_stage",
    "gain_ratio_update",
    "termination_status",
    "as_float64",
]


class Status(enum.IntEnum):
    """Termination status of a solve."""

    CONVERGED = 0  #: stopping measure below ``tol``
    MAX_ITER = 1  #: iteration cap reached with a finite stopping measure
    NONFINITE = 2  #: stopping measure is NaN or inf (the iterate itself is the last accepted one)


@dataclass(frozen=True)
class RegularizationOptions:
    """Stopping and Levenberg-Marquardt options common to both solvers.

    Attributes:
        tol: stop when the solver's stopping measure falls below ``tol``.
        max_iter: iteration cap (the original code had none and could hang).
        reg_init: initial regularization ``lambda``.
        reg_factor_init: initial increase factor ``nu`` of the gain-ratio rule.
        reg_min, reg_max: ``lambda`` is clipped to this interval after every update.
    """

    tol: float = 1e-6
    max_iter: int = 1000
    reg_init: float = 1e-2
    reg_factor_init: float = 2.0
    reg_min: float = 1e-16
    reg_max: float = 1e16


def gain_ratio_update(reg, reg_factor, gain_ratio, accept, options: RegularizationOptions):
    """Gain-ratio rule ([37], Eq. 2.21): after an accepted step with gain ratio ``rho``,
    ``lambda <- lambda * max(1/3, 1 - (2 rho - 1)^3)`` and ``nu <- 2``; after a rejected
    step ``lambda <- lambda * nu`` and ``nu <- 2 nu``. ``lambda`` is then clipped."""
    reg_new = jnp.where(accept, reg * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3), reg * reg_factor)
    reg_factor_new = jnp.where(accept, 2.0, 2.0 * reg_factor)
    return jnp.clip(reg_new, options.reg_min, options.reg_max), reg_factor_new


class StageExpansion(NamedTuple):
    """First and second derivatives of a stage cost ``l(x, u, ...)`` and the dynamics."""

    lx: jax.Array
    lu: jax.Array
    lxx: jax.Array
    luu: jax.Array
    lxu: jax.Array
    fx: jax.Array
    fu: jax.Array


def expand_stage(dynamics: Callable, cost: Callable, x, u, *cost_args) -> StageExpansion:
    """Derivatives of ``cost(x, u, *cost_args)`` and ``dynamics(x, u)`` at ``(x, u)``."""
    return StageExpansion(
        lx=grad(cost, 0)(x, u, *cost_args),
        lu=grad(cost, 1)(x, u, *cost_args),
        lxx=hessian(cost, 0)(x, u, *cost_args),
        luu=hessian(cost, 1)(x, u, *cost_args),
        lxu=jacrev(jacrev(cost, 0), 1)(x, u, *cost_args),
        fx=jacrev(dynamics, 0)(x, u),
        fu=jacrev(dynamics, 1)(x, u),
    )


def termination_status(converged, err, it):
    """``CONVERGED`` if ``converged``; ``NONFINITE`` if at least one iteration ran and the
    stopping measure is NaN/inf; ``MAX_ITER`` otherwise."""
    nonfinite = jnp.logical_and(it > 0, jnp.logical_not(jnp.isfinite(err)))
    return jnp.where(converged, Status.CONVERGED, jnp.where(nonfinite, Status.NONFINITE, Status.MAX_ITER))


def as_float64(a):
    """Cast concrete inputs to float64 (traced values are left untouched)."""
    if isinstance(a, jax.core.Tracer):
        return a
    return jnp.asarray(a, dtype=jnp.float64)
