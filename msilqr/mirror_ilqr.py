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
  ``ybar_t = grad psi(ubar_t)``, so every iterate is feasible.

The solver stops when ``max_t |Q_y,t|`` (the mirror-space Hamiltonian gradient, evaluated
in the backward pass of the current iterate) falls below ``tol``, or after ``max_iter``
iterations.

Numerical reproducibility: the floating-point operations follow the original
implementation (commit effb14c) one to one, which the regression benchmarks check
bitwise. MPC closed loops on the cart-pole are chaotically sensitive to rounding, so
even mathematically identical rewrites (Cholesky instead of LU, derivatives vmapped
over the horizon) change iteration counts; see ``docs/REGRESSION.md`` before
"optimizing" this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax import grad, hessian

from msilqr._common import (
    RegularizationOptions,
    Status,
    as_float64,
    check_trajectory_shapes,
    expand_stage,
    gain_ratio_update,
    termination_status,
)
from msilqr._x64 import require_x64
from msilqr.mirror_maps import MirrorMap
from msilqr.problem import OptimalControlProblem, rollout

__all__ = ["MirrorILQR", "SolverOptions", "Solution", "Status"]


@dataclass(frozen=True)
class SolverOptions(RegularizationOptions):
    """Options of :class:`MirrorILQR` (defaults reproduce the paper's experiments).

    The stopping measure is ``max_t |Q_y,t|``; see
    :class:`~msilqr.RegularizationOptions` for the fields.
    """


class Solution(NamedTuple):
    """Result of :meth:`MirrorILQR.solve` (all fields are JAX arrays)."""

    states: jax.Array  #: ``(T+1, n)`` optimal states
    controls: jax.Array  #: ``(T, m)`` optimal controls (in ``U``)
    cost: jax.Array  #: total cost ``J`` of the returned trajectory
    iterations: jax.Array  #: number of iterations (backward + forward pass), incl. rejected ones
    reg: jax.Array  #: final Levenberg-Marquardt parameter
    error: jax.Array  #: last stopping measure ``max_t |Q_y,t|``
    status: jax.Array  #: :class:`Status` code


class MirrorDerivatives(NamedTuple):
    """Stage derivatives in mirror coordinates (Lemma 3, Gauss-Newton (12))."""

    cx: jax.Array
    cy: jax.Array
    fx: jax.Array
    fy: jax.Array
    cxx: jax.Array
    cyy: jax.Array
    cxy: jax.Array


def mirror_derivatives(problem: OptimalControlProblem, mirror_map: MirrorMap, x, u) -> MirrorDerivatives:
    """Stage derivatives with respect to ``(x, y)`` at ``(x, u)``, ``u = grad psi*(y)``.

    Only ``Psi^{-1}(u)`` is needed: ``c_y = Psi^{-1} c_u`` (Eq. (7)), ``f_y = f_u Psi^{-1}``,
    ``c_xy = c_xu Psi^{-1}`` (Eq. (10)), ``c_yy = Psi^{-1} c_uu Psi^{-1}`` (Eq. (12)).
    """
    Psi_inv = mirror_map.inv_hessian(u)
    e = expand_stage(problem.dynamics, problem.stage_cost, x, u)
    return MirrorDerivatives(
        cx=e.lx,
        cy=Psi_inv @ e.lu,
        fx=e.fx,
        fy=e.fu @ Psi_inv,
        cxx=e.lxx,
        cyy=Psi_inv @ e.luu @ Psi_inv,
        cxy=e.lxu @ Psi_inv,
    )


def q_expansion(d: MirrorDerivatives, Vx, Vxx, reg):
    """Q-function expansion in mirror coordinates (Prop. 4 with (12)) and the
    Levenberg-Marquardt damped ``Qhat_yy = Q_yy + reg I``."""
    Qx = d.cx + d.fx.T @ Vx
    Qy = d.cy + d.fy.T @ Vx
    Qxx = d.cxx + d.fx.T @ Vxx @ d.fx
    Qyy_hat = d.cyy + d.fy.T @ Vxx @ d.fy + reg * jnp.eye(d.cy.shape[0])
    Qxy = d.cxy + d.fx.T @ Vxx @ d.fy
    return Qx, Qy, Qxx, Qyy_hat, Qxy


def feedback_gains(Qy, Qyy_hat, Qxy):
    """``k = -Qhat_yy^{-1} Q_y``, ``K = -Qhat_yy^{-1} Q_xy^T``, and whether ``Qhat_yy > 0``."""
    positive_definite = jnp.all(jnp.linalg.eigvalsh(Qyy_hat) > 0)
    k = -jsc.linalg.solve(Qyy_hat, Qy)
    K = -jsc.linalg.solve(Qyy_hat, Qxy.T)
    return k, K, positive_definite


def value_update(Qx, Qy, Qxx, Qyy_hat, Qxy, k, K):
    """Value recursion (13) with ``Qhat_yy`` throughout; returns ``V_x, V_xx, Delta V``."""
    Qyx = Qxy.T
    Vx = Qx + Qxy @ k + K.T @ Qy + K.T @ Qyy_hat @ k
    Vxx = Qxx + K.T @ Qyy_hat @ K + K.T @ Qyx + Qxy @ K
    Vxx = 0.5 * (Vxx + Vxx.T)
    dV = Qy @ k + 0.5 * k.T @ Qyy_hat @ k
    return Vx, Vxx, dV


@dataclass(frozen=True, eq=False)
class MirrorILQR:
    """Mirror-space iLQR for ``problem`` with control constraints encoded by ``mirror_map``.

    Example::

        solver = MirrorILQR(problem, BoxEntropy(-50.0, 50.0))
        sol = solver.solve(x0, u_init)

    ``solve`` is jit-compiled once per solver instance and input shape.
    """

    problem: OptimalControlProblem
    mirror_map: MirrorMap
    options: SolverOptions = field(default_factory=SolverOptions)

    def __post_init__(self):
        object.__setattr__(self, "_solve_jit", jax.jit(self._solve))

    # -------------------------------------------------------------- backward pass ---

    def backward_pass(self, states, controls, reg):
        """Backward pass in mirror coordinates with ``Qhat_yy = Q_yy + reg * I``.

        Returns ``(K, k, dV, positive_definite, error)``: feedback and feedforward gains
        in mirror coordinates, the predicted cost change ``sum_t Delta V_t``, whether
        every ``Qhat_yy,t`` was positive definite, and ``max_t |Q_y,t|``.
        """
        p = self.problem
        Vx_final = grad(p.final_cost, 0)(states[-1])
        Vxx_final = hessian(p.final_cost, 0)(states[-1])

        def step(carry, inp):
            Vx, Vxx = carry
            d = mirror_derivatives(p, self.mirror_map, *inp)
            Qx, Qy, Qxx, Qyy_hat, Qxy = q_expansion(d, Vx, Vxx, reg)
            k, K, positive_definite = feedback_gains(Qy, Qyy_hat, Qxy)
            Vx, Vxx, dV = value_update(Qx, Qy, Qxx, Qyy_hat, Qxy, k, K)
            return (Vx, Vxx), (K, k, dV, positive_definite, Qy)

        _, (K, k, dV, positive_definite, Qy) = jax.lax.scan(step, (Vx_final, Vxx_final), (states[:-1], controls), reverse=True)
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

        _, (new_states, new_controls) = jax.lax.scan(step, states[0], (states[:-1], mirror_controls, controls, k, K))
        return jnp.vstack((states[0], new_states)), new_controls

    # ----------------------------------------------------------------- main loop ---

    def solve(self, x0: jax.Array, controls: jax.Array, *, validate: bool = True) -> Solution:
        """Solve from the initial state ``x0`` (``(n,)``) and initial controls ``(T, m)``.

        The initial controls must be strictly feasible (in ``int(U)``): at a boundary
        point ``Psi^{-1} = 0``, so the mirror-space gradient vanishes and the solver would
        stop at once. With ``validate=True`` concrete controls are checked and a
        ``ValueError`` is raised otherwise (traced controls cannot be checked).
        ``validate=False`` skips the check; the MPC loops use it because their warm starts
        (shifted plans) may legitimately contain controls that rounded onto the bound.
        """
        require_x64()
        check_trajectory_shapes(x0, controls)
        x0, controls = as_float64(x0), as_float64(controls)
        if validate:
            self.validate_controls(controls)
        return self._solve_jit(x0, controls)

    def validate_controls(self, controls) -> None:
        """Raise ``ValueError`` unless every concrete control is in ``int(U)`` (no-op when traced)."""
        if isinstance(controls, jax.core.Tracer):
            return
        inside = jax.vmap(self.mirror_map.contains)(controls)
        if not bool(jnp.all(inside)):
            raise ValueError(
                f"{int(jnp.sum(~inside))} of {controls.shape[0]} initial controls are not strictly inside "
                "the constraint set; mirror-space iLQR needs a strictly feasible initial guess"
            )

    def _solve(self, x0, controls) -> Solution:
        opt = self.options
        total_cost = self.problem.total_cost
        states = rollout(self.problem.dynamics, x0, controls)

        def keep_going(carry):
            *_, it, err = carry
            return jnp.logical_and(jnp.logical_not(err < opt.tol), it < opt.max_iter)

        def iteration(carry):
            states, controls, reg, reg_factor, cost, it, _ = carry
            K, k, predicted, positive_definite, err = self.backward_pass(states, controls, reg)
            new_states, new_controls = self.forward_pass(states, controls, K, k)
            new_cost = total_cost(new_states, new_controls)

            actual = new_cost - cost
            gain_ratio = actual / predicted
            accept = jnp.logical_and(jnp.logical_and(actual < 0.0, predicted < 0.0), positive_definite)
            reg, reg_factor = gain_ratio_update(reg, reg_factor, gain_ratio, accept, opt)

            states = jnp.where(accept, new_states, states)
            controls = jnp.where(accept, new_controls, controls)
            cost = jnp.where(accept, new_cost, cost)
            return states, controls, reg, reg_factor, cost, it + 1, err

        init = (states, controls, jnp.array(opt.reg_init), jnp.array(opt.reg_factor_init), total_cost(states, controls), 0, jnp.inf)
        states, controls, reg, _, cost, it, err = jax.lax.while_loop(keep_going, iteration, init)
        return Solution(states, controls, cost, it, reg, err, termination_status(err < opt.tol, err, it))
