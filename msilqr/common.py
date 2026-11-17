"""Pieces shared by the mirror-space solver and the interior-point baseline.

Both solvers are Levenberg-Marquardt-regularized DDP methods: they expand the stage
cost (or Lagrangian) and dynamics to second order along a nominal trajectory, adapt the
regularization with the gain-ratio rule of Madsen, Nielsen and Tingleff ([37], Eq. 2.21),
and stop on a tolerance or an iteration cap.
"""

from __future__ import annotations

import enum
import operator
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
    "check_trajectory_shapes",
    "require_x64",
    "check_slacks_duals",
    "natural_residual",
]


class Status(enum.IntEnum):
    """Termination status of a solve."""

    CONVERGED = 0  #: stopping measure below ``tol``
    MAX_ITER = 1  #: iteration cap reached (stopping measure finite, or ``inf`` if no iteration ran)
    NONFINITE = 2  #: stopping measure is NaN or inf (the iterate itself is the last accepted one)
    STALLED = 3  #: no further step can register a cost decrease in float64 (mirror-space solver)


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

    tol: float = 1e-4
    max_iter: int = 1000
    reg_init: float = 1e-2
    reg_factor_init: float = 2.0
    reg_min: float = 1e-16
    reg_max: float = 1e16

    def __post_init__(self):
        try:
            max_iter_ok = operator.index(self.max_iter) >= 0
        except TypeError:
            max_iter_ok = False
        checks = [
            (self.tol >= 0, "tol must be >= 0"),
            (max_iter_ok, "max_iter must be a non-negative integer"),
            (self.reg_init > 0, "reg_init must be > 0"),
            (self.reg_factor_init > 1, "reg_factor_init must be > 1 (a rejected step must increase lambda)"),
            (0 <= self.reg_min <= self.reg_max, "need 0 <= reg_min <= reg_max"),
        ]
        for ok, message in checks:  # written as `ok` so NaN fails every check
            if not ok:
                raise ValueError(f"{message}, got {self}")


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


def termination_status(converged, err, it, stalled=False):
    """``CONVERGED`` if ``converged``; ``NONFINITE`` if at least one iteration ran and the
    stopping measure is NaN/inf; ``STALLED`` if ``stalled``; ``MAX_ITER`` otherwise."""
    nonfinite = jnp.logical_and(it > 0, jnp.logical_not(jnp.isfinite(err)))
    otherwise = jnp.where(stalled, Status.STALLED, Status.MAX_ITER)
    return jnp.where(converged, Status.CONVERGED, jnp.where(nonfinite, Status.NONFINITE, otherwise))


def check_trajectory_shapes(x0, controls, *per_stage) -> None:
    """Raise ``ValueError`` unless ``x0`` is ``(n,)``, ``controls`` is ``(T, m)`` and every
    array in ``per_stage`` is ``(T, k)`` (shapes are static, so this also works under jit)."""
    if jnp.ndim(x0) != 1:
        raise ValueError(f"x0 must be a single state of shape (n,), got shape {jnp.shape(x0)}")
    if jnp.ndim(controls) != 2:
        raise ValueError(f"controls must have shape (T, m), got shape {jnp.shape(controls)}")
    horizon = jnp.shape(controls)[0]
    for a in per_stage:
        if jnp.ndim(a) != 2 or jnp.shape(a)[0] != horizon:
            raise ValueError(f"per-stage arrays must have shape ({horizon}, k), got shape {jnp.shape(a)}")


def check_slacks_duals(slacks, duals) -> None:
    """Raise ``ValueError`` unless concrete slacks and duals are strictly positive and
    finite (no-op for traced values; concrete values are checked even inside ``jit``)."""
    if isinstance(slacks, jax.core.Tracer) or isinstance(duals, jax.core.Tracer):
        return
    with jax.ensure_compile_time_eval():
        ok = all(bool(jnp.all((a > 0) & jnp.isfinite(a))) for a in (slacks, duals))
    if not ok:
        raise ValueError("initial slacks and duals must be strictly positive and finite")


def require_x64() -> None:
    """Raise unless ``jax_enable_x64`` is on: in float32 the solvers may never reach ``tol``."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError('msilqr requires double precision. Enable it before creating arrays: jax.config.update("jax_enable_x64", True)')


def as_float64(a):
    """Cast concrete inputs to float64 (traced values are left untouched)."""
    if isinstance(a, jax.core.Tracer):
        return a
    return jnp.asarray(a, dtype=jnp.float64)


def natural_residual(controls, gradient, project: Callable[[jax.Array], jax.Array]):
    """``max_t ||u_t - project(u_t - g_t)||_inf`` for the gradient ``g_t = dJ/du_t``.

    ``project`` is the Euclidean projection onto the convex control set ``U``. The residual
    is zero exactly at the first-order (KKT) points of ``min J`` subject to ``u_t in U``. For
    a box it is ``median(u - b, g, u - a)`` per component: ``|g|``, or the distance to the
    bound when the step ``u - g`` crosses it.
    """
    return jnp.max(jnp.abs(controls - jax.vmap(project)(controls - gradient)))
