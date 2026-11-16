"""Mirror-space iLQR (Sec. III-C of the paper).

The constrained problem (1) is reparameterized with the mirror map ``u = grad psi*(y)``
(problem (2)) and solved with iLQR in the unconstrained mirror controls ``y``:

* derivatives under the mirror map, Lemma 3 / Prop. 4 with the Gauss-Newton
  approximation (12): ``Q_y = Psi^{-1} Q_u``, ``Q_xy = Q_xu Psi^{-1}``,
  ``Q_yy = Psi^{-1} Q_uu Psi^{-1}`` (only derivatives of ``psi``, never of
  ``grad psi*``);
* Levenberg-Marquardt regularization ``Qhat_yy = Q_yy + lambda I`` adapted with the
  gain-ratio rule of Madsen, Nielsen and Tingleff ([37], Eq. 2.21);
* value recursion (13) using ``Qhat_yy`` throughout; the summed ``Delta V`` is the
  predicted cost change in the gain ratio;
* forward rollout (14): ``y_t = ybar_t + k_t + K_t dx_t``, ``u_t = grad psi*(y_t)`` with
  ``ybar_t = grad psi(ubar_t)``, so every iterate is strictly feasible.

The solver stops when ``max_t |Q_y,t|`` (the mirror-space Hamiltonian gradient, evaluated
in the backward pass of the current iterate) falls below ``tol``, or after ``max_iter``
iterations.

Numerical reproducibility: the floating-point operations follow the original
implementation (commit effb14c) one to one, which the regression benchmarks check
bitwise. MPC closed loops on the cart-pole are chaotically sensitive to rounding, so
even mathematically identical rewrites (Cholesky instead of LU, derivatives vmapped
over the horizon) change iteration counts; see the README before "optimizing" this
module.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax import grad, hessian, jacrev

from msilqr._x64 import require_x64
from msilqr.mirror_maps import MirrorMap
from msilqr.problem import OptimalControlProblem

__all__ = ["MirrorILQR", "SolverOptions", "Solution", "Status"]


class Status(enum.IntEnum):
    """Termination status of a solve."""

    CONVERGED = 0  #: stopping measure below ``tol``
    MAX_ITER = 1  #: iteration cap reached with a finite stopping measure
    NONFINITE = 2  #: stopping measure is NaN or inf (the iterate itself is the last accepted one)


@dataclass(frozen=True)
class SolverOptions:
    """Options of :class:`MirrorILQR` (defaults reproduce the paper's experiments).

    Attributes:
        tol: stop when ``max_t |Q_y,t| < tol``.
        max_iter: iteration cap (the original code had none and could hang).
        reg_init: initial Levenberg-Marquardt parameter ``lambda``.
        reg_factor_init: initial increase factor ``nu`` of the gain-ratio rule.
        reg_min, reg_max: ``lambda`` is clipped to this interval after every update.
    """

    tol: float = 1e-6
    max_iter: int = 1000
    reg_init: float = 1e-2
    reg_factor_init: float = 2.0
    reg_min: float = 1e-16
    reg_max: float = 1e16


class Solution(NamedTuple):
    """Result of :meth:`MirrorILQR.solve` (all fields are JAX arrays)."""

    states: jax.Array  #: ``(T+1, n)`` optimal states
    controls: jax.Array  #: ``(T, m)`` optimal controls, strictly feasible
    cost: jax.Array  #: total cost ``J`` of the returned trajectory
    iterations: jax.Array  #: number of iterations (backward + forward pass), incl. rejected ones
    reg: jax.Array  #: final Levenberg-Marquardt parameter
    error: jax.Array  #: last stopping measure ``max_t |Q_y,t|``
    status: jax.Array  #: :class:`Status` code


class _Derivatives(NamedTuple):
    cx: jax.Array
    cy: jax.Array
    fx: jax.Array
    fy: jax.Array
    cxx: jax.Array
    cyy: jax.Array
    cxy: jax.Array


@dataclass(frozen=True, eq=False)
class MirrorILQR:
    """Mirror-space iLQR for ``problem`` with control constraints encoded by ``mirror_map``.

    Example::

        solver = MirrorILQR(problem, BoxEntropy(-50.0, 50.0))
        sol = solver.solve(problem.rollout(x0, u_init), u_init)

    ``solve`` is jit-compiled once per solver instance and input shape.
    """

    problem: OptimalControlProblem
    mirror_map: MirrorMap
    options: SolverOptions = field(default_factory=SolverOptions)

    def __post_init__(self):
        object.__setattr__(self, "_solve_jit", jax.jit(self._solve))

    # ---------------------------------------------------------------- derivatives ---

    def _derivatives(self, x, u) -> _Derivatives:
        """Stage derivatives in mirror coordinates at ``(x, u)``, Gauss-Newton (12)."""
        p, mmap = self.problem, self.mirror_map
        Psi_inv = mmap.inv_hessian(u)

        cx = grad(p.stage_cost, 0)(x, u)
        cu = grad(p.stage_cost, 1)(x, u)
        cxx = hessian(p.stage_cost, 0)(x, u)
        cuu = hessian(p.stage_cost, 1)(x, u)
        cxu = jacrev(jacrev(p.stage_cost, 0), 1)(x, u)
        fx = jacrev(p.dynamics, 0)(x, u)
        fu = jacrev(p.dynamics, 1)(x, u)

        cy = Psi_inv @ cu  # Eq. (9) / (7)
        fy = fu @ Psi_inv  # chain rule through u = grad psi*(y)
        cxy = cxu @ Psi_inv  # Eq. (10)
        cyy = Psi_inv @ cuu @ Psi_inv  # Eq. (12), third-order term dropped
        return _Derivatives(cx, cy, fx, fy, cxx, cyy, cxy)

    # -------------------------------------------------------------- backward pass ---

    def backward_pass(self, states, controls, reg):
        """Value recursion (13) with ``Qhat_yy = Q_yy + reg * I``.

        Returns ``(K, k, dV, positive_definite, error)``: feedback and feedforward gains
        in mirror coordinates, the predicted cost change ``sum_t Delta V_t``, whether
        every ``Qhat_yy,t`` was positive definite, and ``max_t |Q_y,t|``.
        """
        p = self.problem
        x_final = states[-1]
        Vx_final = grad(p.final_cost, 0)(x_final)
        Vxx_final = hessian(p.final_cost, 0)(x_final)

        def step(carry, inp):
            Vx, Vxx = carry
            x, u = inp
            d = self._derivatives(x, u)

            Qx = d.cx + d.fx.T @ Vx
            Qy = d.cy + d.fy.T @ Vx

            Qxx = d.cxx + d.fx.T @ Vxx @ d.fx
            Qyy = d.cyy + d.fy.T @ Vxx @ d.fy + reg * jnp.eye(u.shape[0])
            Qxy = d.cxy + d.fx.T @ Vxx @ d.fy
            Qyx = Qxy.T

            positive_definite = jnp.all(jnp.linalg.eigvalsh(Qyy) > 0)

            k = -jsc.linalg.solve(Qyy, Qy)
            K = -jsc.linalg.solve(Qyy, Qyx)

            Vx = Qx + Qxy @ k + K.T @ Qy + K.T @ Qyy @ k
            Vxx = Qxx + K.T @ Qyy @ K + K.T @ Qyx + Qxy @ K
            Vxx = 0.5 * (Vxx + Vxx.T)
            dV = Qy @ k + 0.5 * k.T @ Qyy @ k
            return (Vx, Vxx), (K, k, dV, positive_definite, Qy)

        _, (K, k, dV, positive_definite, Qy) = jax.lax.scan(
            step, (Vx_final, Vxx_final), (states[:-1], controls), reverse=True
        )
        return K, k, jnp.sum(dV), jnp.all(positive_definite), jnp.max(jnp.abs(Qy))

    # --------------------------------------------------------------- forward pass ---

    def forward_pass(self, states, controls, K, k):
        """Rollout (14): ``y = grad psi(ubar) + k + K (x - xbar)``, ``u = grad psi*(y)``."""
        p, mmap = self.problem, self.mirror_map
        mirror_controls = jax.vmap(mmap.grad)(controls)

        def step(x_prev, inp):
            x_nom, y_nom, u_nom, k_t, K_t = inp
            dx = x_prev - x_nom
            y_new = y_nom + k_t + K_t @ dx
            u_new = mmap.inverse(y_new, u_nom)
            x_new = p.dynamics(x_prev, u_new)
            return x_new, (x_new, u_new)

        _, (new_states, new_controls) = jax.lax.scan(
            step, states[0], (states[:-1], mirror_controls, controls, k, K)
        )
        return jnp.vstack((states[0], new_states)), new_controls

    # ----------------------------------------------------------------- main loop ---

    def solve(self, states: jax.Array, controls: jax.Array) -> Solution:
        """Solve from an initial trajectory (``states = problem.rollout(x0, controls)``).

        ``controls`` must be strictly feasible (in ``int(U)``).
        """
        require_x64()
        return self._solve_jit(states, controls)

    def _solve(self, states, controls) -> Solution:
        opt = self.options
        total_cost = self.problem.total_cost

        def cond(carry):
            *_, it, err = carry
            not_converged = jnp.logical_not(err < opt.tol)
            return jnp.logical_and(not_converged, it < opt.max_iter)

        def body(carry):
            states, controls, reg, reg_factor, cost, it, _ = carry

            K, k, predicted, feasible, err = self.backward_pass(states, controls, reg)
            new_states, new_controls = self.forward_pass(states, controls, K, k)
            new_cost = total_cost(new_states, new_controls)

            actual = new_cost - cost
            gain_ratio = actual / predicted
            accept = jnp.logical_and(jnp.logical_and(actual < 0.0, predicted < 0.0), feasible)

            # Madsen-Nielsen-Tingleff gain-ratio rule ([37], Eq. 2.21)
            reg_new = jnp.where(
                accept,
                reg * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
                reg * reg_factor,
            )
            reg_factor_new = jnp.where(accept, 2.0, 2.0 * reg_factor)
            reg_new = jnp.clip(reg_new, opt.reg_min, opt.reg_max)

            states = jnp.where(accept, new_states, states)
            controls = jnp.where(accept, new_controls, controls)
            cost = jnp.where(accept, new_cost, cost)
            return states, controls, reg_new, reg_factor_new, cost, it + 1, err

        init = (
            states,
            controls,
            jnp.array(opt.reg_init),
            jnp.array(opt.reg_factor_init),
            total_cost(states, controls),
            0,
            1e10,
        )
        states, controls, reg, _, cost, it, err = jax.lax.while_loop(cond, body, init)

        status = jnp.where(
            err < opt.tol,
            Status.CONVERGED,
            jnp.where(jnp.isfinite(err), Status.MAX_ITER, Status.NONFINITE),
        )
        return Solution(states, controls, cost, it, reg, err, status)
