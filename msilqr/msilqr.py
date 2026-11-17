"""Mirror-space iLQR (Sec. III-C of the paper).

iLQR in the unconstrained mirror controls ``y = grad psi(u)``, so every iterate is
feasible. The derivatives follow Lemma 3 and Prop. 4 with the Gauss-Newton
approximation (12), steps are damped with ``Q_yy + lambda I``, and the rollout (14) maps
the new mirror controls back with ``u = grad psi*(y)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax import grad, hessian

from msilqr.common import (
    RegularizationOptions,
    Status,
    as_float64,
    check_trajectory_shapes,
    expand_stage,
    gain_ratio_update,
    natural_residual,
    require_x64,
    termination_status,
)
from msilqr.maps import MirrorMap
from msilqr.problem import OptimalControlProblem, rollout

__all__ = ["MirrorILQR", "SolverOptions", "Solution", "Status"]


@dataclass(frozen=True)
class SolverOptions(RegularizationOptions):
    """Options of :class:`MirrorILQR` (defaults reproduce the paper's experiments).

    Attributes:
        stopping: ``"mirror"`` (the paper's test): stop once ``max_t |Q_y,t| < tol``.
            ``Q_y = Psi^{-1} Q_u`` vanishes at every saturated control, whichever way the
            gradient points. ``"natural"``: stop once the natural residual
            ``max_t ||u_t - Pi_U(u_t - dJ/du_t)||_inf < tol``, which is zero exactly at the KKT
            points of the constrained problem (``dJ/du_t`` from an adjoint sweep in the
            backward pass, ``Pi_U`` is ``mirror_map.project``). The solve also ends, with
            ``Status.STALLED``, when a step is rejected whose predicted cost decrease is below
            ``10 eps |J|``: a larger ``lambda`` only shrinks it, so no later step at this
            iterate can register a decrease in float64.

    See :class:`~msilqr.RegularizationOptions` for the other fields.
    """

    stopping: str = "mirror"

    def __post_init__(self):
        super().__post_init__()
        if self.stopping not in ("mirror", "natural"):
            raise ValueError(f'stopping must be "mirror" or "natural", got {self.stopping!r}')


class Solution(NamedTuple):
    """Result of :meth:`MirrorILQR.solve` (all fields are JAX arrays)."""

    states: jax.Array  #: ``(T+1, n)`` optimal states
    controls: jax.Array  #: ``(T, m)`` optimal controls (in ``U``)
    cost: jax.Array  #: total cost ``J`` of the returned trajectory
    iterations: jax.Array  #: number of iterations (backward + forward pass), incl. rejected ones
    reg: jax.Array  #: final Levenberg-Marquardt parameter
    error: jax.Array  #: last stopping measure: ``max_t |Q_y,t|`` or the natural residual
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
    cu: jax.Array  #: ``c_u`` (control space, for the natural residual)
    fu: jax.Array  #: ``f_u`` (control space, for the natural residual)


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
        cu=e.lu,
        fu=e.fu,
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
        every ``Qhat_yy,t`` was positive definite, and the stopping measure (``max_t |Q_y,t|``
        or the natural residual, see :class:`SolverOptions`).
        """
        p = self.problem
        natural = self.options.stopping == "natural"
        Vx_final = grad(p.final_cost, 0)(states[-1])
        Vxx_final = hessian(p.final_cost, 0)(states[-1])

        def step(carry, inp):
            Vx, Vxx = carry[:2]
            d = mirror_derivatives(p, self.mirror_map, *inp)
            Qx, Qy, Qxx, Qyy_hat, Qxy = q_expansion(d, Vx, Vxx, reg)
            k, K, positive_definite = feedback_gains(Qy, Qyy_hat, Qxy)
            Vx, Vxx, dV = value_update(Qx, Qy, Qxx, Qyy_hat, Qxy, k, K)
            if natural:  # open-loop adjoint: l_t = c_x + f_x^T l_{t+1}, dJ/du_t = c_u + f_u^T l_{t+1}
                adjoint = carry[2]
                return (Vx, Vxx, d.cx + d.fx.T @ adjoint), (K, k, dV, positive_definite, d.cu + d.fu.T @ adjoint)
            return (Vx, Vxx), (K, k, dV, positive_definite, Qy)

        init = (Vx_final, Vxx_final, Vx_final) if natural else (Vx_final, Vxx_final)
        _, (K, k, dV, positive_definite, g) = jax.lax.scan(step, init, (states[:-1], controls), reverse=True)
        error = natural_residual(controls, g, self.mirror_map.project) if natural else jnp.max(jnp.abs(g))
        return K, k, jnp.sum(dV), jnp.all(positive_definite), error

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
        """Raise ``ValueError`` unless every concrete control is in ``int(U)`` (no-op for
        traced controls; concrete ones are checked even inside ``jit``)."""
        if isinstance(controls, jax.core.Tracer):
            return
        with jax.ensure_compile_time_eval():
            inside = jax.vmap(self.mirror_map.contains)(controls)
            n_out = int(jnp.sum(~inside))
        if n_out:
            raise ValueError(
                f"{n_out} of {controls.shape[0]} initial controls are not strictly inside "
                "the constraint set; mirror-space iLQR needs a strictly feasible initial guess"
            )

    def _solve(self, x0, controls) -> Solution:
        opt = self.options
        total_cost = self.problem.total_cost
        states = rollout(self.problem.dynamics, x0, controls)

        def keep_going(carry):
            *_, it, err, stalled = carry
            return jnp.logical_and(jnp.logical_and(jnp.logical_not(err < opt.tol), it < opt.max_iter), jnp.logical_not(stalled))

        def iteration(carry):
            states, controls, reg, reg_factor, cost, it, _, stalled = carry
            K, k, predicted, positive_definite, err = self.backward_pass(states, controls, reg)
            new_states, new_controls = self.forward_pass(states, controls, K, k)
            new_cost = total_cost(new_states, new_controls)

            actual = new_cost - cost
            gain_ratio = actual / predicted
            accept = jnp.logical_and(jnp.logical_and(actual < 0.0, predicted < 0.0), positive_definite)
            reg, reg_factor = gain_ratio_update(reg, reg_factor, gain_ratio, accept, opt)
            if opt.stopping == "natural":  # a rejected step too small to register in float64
                floor = 10.0 * jnp.finfo(cost.dtype).eps * jnp.abs(cost)
                stalled = jnp.logical_and(jnp.logical_and(~accept, positive_definite), jnp.abs(predicted) <= floor)

            states = jnp.where(accept, new_states, states)
            controls = jnp.where(accept, new_controls, controls)
            cost = jnp.where(accept, new_cost, cost)
            return states, controls, reg, reg_factor, cost, it + 1, err, stalled

        init = (
            states,
            controls,
            jnp.array(opt.reg_init),
            jnp.array(opt.reg_factor_init),
            total_cost(states, controls),
            0,
            jnp.inf,
            jnp.array(False),
        )
        states, controls, reg, _, cost, it, err, stalled = jax.lax.while_loop(keep_going, iteration, init)
        return Solution(states, controls, cost, it, reg, err, termination_status(err < opt.tol, err, it, stalled))
