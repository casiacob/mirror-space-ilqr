"""Infeasible interior-point DDP: the baseline of the paper's comparison (Sec. V, [20]).

Pavlov, Shames and Manzie, "Interior point differential dynamic programming", IEEE
TCST 29(6), 2021 -- the "infeasible" variant: inequality constraints
``g(x, u) <= 0`` become ``g(x, u) + s = 0`` with slacks ``s > 0`` and duals ``z > 0``;
each backward pass solves the perturbed KKT system of the barrier subproblem with
barrier parameter ``mu``, and ``mu`` is decreased when the subproblem residual is small.
Iterates may violate the constraints until convergence.

This module reproduces the original baseline (commit effb14c) operation for operation,
so the regression benchmarks match it bitwise. Changes relative to the original:

* ``jnp.clip(a_min=, a_max=)`` -> ``jnp.clip(min=, max=)`` (the original crashes on
  current JAX);
* the feedback gain of the state constraint term uses ``Sigma = S^{-1} Z`` instead of
  ``S^{-1}`` (a bug; it only matters for state-dependent constraints, the benchmarks
  constrain controls only);
* the definiteness test of the symmetrized ``Q_uu`` uses ``eigvalsh`` instead of the
  general complex ``eigvals`` (CPU-only in JAX; ``>`` on complex numbers is
  lexicographic). Bitwise identical on all benchmarks;
* an iteration cap and a status;
* option ``barrier_tol``: also require ``mu < barrier_tol`` before stopping. The original
  (default ``None``) stops on the barrier-subproblem residual alone, so it can return
  points of the central path with ``mu`` far above ``tol``.

Known limitations kept for comparability with the paper (documented in the README):
the step acceptance ignores primal infeasibility, the fraction-to-boundary rule is
applied per stage (each stage takes its own dual/slack step size), and the stopping
measure (max of ``|Qhat_u|``, ``|r_p|``, ``|r_d|``, where ``Qhat_u = Q_u + c_u^T S^{-1} r``
is the gradient of the barrier-augmented Q-function) differs from the mirror-space
solver's (``max |Q_y|``), so "the same tolerance" does not mean the same accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.lax
import jax.numpy as jnp
import jax.scipy as jcp
from jax import grad, hessian, jacrev

from msilqr._x64 import require_x64
from msilqr.mirror_ilqr import _as_float64, termination_status
from msilqr.problem import OptimalControlProblem, rollout

__all__ = ["InteriorPointDDP", "IPDDPOptions", "IPDDPSolution"]


@dataclass(frozen=True)
class IPDDPOptions:
    """Options of :class:`InteriorPointDDP` (defaults reproduce the paper's experiments).

    Attributes:
        tol: stop when the stopping measure is below ``tol``.
        max_iter: iteration cap.
        reg_init, reg_factor_init, reg_min, reg_max: Levenberg-Marquardt regularization,
            as in :class:`~msilqr.mirror_ilqr.SolverOptions`.
        barrier_tol: if set, additionally require ``mu < barrier_tol`` to stop.
        barrier_init: initial barrier parameter. ``None`` uses the original rule
            ``mu_0 = J(u_0) / (T n_c)`` (Pavlov et al.).
        barrier_init_fallback: ``mu_0`` used when the original rule gives ``mu_0 <= 0``
            (non-positive costs; the original code then stalled). Whenever the rule gives
            a positive value it is used unchanged, so results with positive costs are
            bitwise identical to the original.
    """

    tol: float = 1e-6
    max_iter: int = 1000
    reg_init: float = 1e-2
    reg_factor_init: float = 2.0
    reg_min: float = 1e-16
    reg_max: float = 1e16
    barrier_tol: float | None = None
    barrier_init: float | None = None
    barrier_init_fallback: float = 1.0


class IPDDPSolution(NamedTuple):
    states: jax.Array
    controls: jax.Array
    slacks: jax.Array
    duals: jax.Array
    iterations: jax.Array
    barrier: jax.Array  #: final barrier parameter ``mu``
    error: jax.Array  #: last stopping measure
    status: jax.Array  #: :class:`~msilqr.mirror_ilqr.Status` code


@dataclass(frozen=True, eq=False)
class InteriorPointDDP:
    """Infeasible IPDDP for ``problem`` subject to ``constraints(x, u) <= 0``."""

    problem: OptimalControlProblem
    constraints: Callable[[jax.Array, jax.Array], jax.Array]
    options: IPDDPOptions = field(default_factory=IPDDPOptions)

    def __post_init__(self):
        object.__setattr__(self, "_solve_jit", jax.jit(self._solve))

    # ------------------------------------------------------------------ helpers ---

    def _barrier_stage_cost(self, state, control, slack, barrier_param):
        return self.problem.stage_cost(state, control) - barrier_param * jnp.sum(jnp.log(slack))

    def _barrier_total_cost(self, states, controls, slack, barrier_param):
        J = jax.vmap(self._barrier_stage_cost, in_axes=(0, 0, 0, None))(states[:-1], controls, slack, barrier_param)
        return jnp.sum(J) + self.problem.final_cost(states[-1])

    def _stage_lagrangian(self, state, control, dual):
        return self.problem.stage_cost(state, control) + dual.T @ self.constraints(state, control)

    # ------------------------------------------------------------ backward pass ---

    def backward_pass(self, states, controls, slacks, duals, barrier_param, reg_param):
        p, constraints, lag = self.problem, self.constraints, self._stage_lagrangian

        def bwd_step(carry, inp):
            Vx, Vxx, pos_def = carry
            state, control, slack, dual = inp

            lx = grad(lag, 0)(state, control, dual)
            fx = jacrev(p.dynamics, 0)(state, control)
            lu = grad(lag, 1)(state, control, dual)
            fu = jacrev(p.dynamics, 1)(state, control)
            lxx = hessian(lag, 0)(state, control, dual)
            luu = hessian(lag, 1)(state, control, dual)
            lxu = jacrev(jacrev(lag, 0), 1)(state, control, dual)
            cx = jacrev(constraints, 0)(state, control)
            cu = jacrev(constraints, 1)(state, control)

            # Q-function expansion
            Qx = lx + fx.T @ Vx
            Qu = lu + fu.T @ Vx
            Qxx = lxx + fx.T @ Vxx @ fx
            Quu = luu + fu.T @ Vxx @ fu
            Quu += reg_param * jnp.eye(control.shape[0])
            Qxu = lxu + fx.T @ Vxx @ fu
            Quu = (Quu + Quu.T) / 2.0
            pos_def = jnp.logical_and(jnp.all(jnp.linalg.eigvalsh(Quu) > 0), pos_def)

            # perturbed KKT system
            Slack_inv = jnp.diag(1 / slack)
            Sigma = Slack_inv @ jnp.diag(dual)
            rp = constraints(state, control) + slack
            rd = dual * slack - barrier_param
            r = dual * rp - rd
            A = Quu + cu.T @ Sigma @ cu

            def convex():
                chol_and_lower = jcp.linalg.cho_factor(A)
                alpha = jcp.linalg.cho_solve(chol_and_lower, -cu.T @ Slack_inv @ r - Qu)
                beta = jcp.linalg.cho_solve(chol_and_lower, -cu.T @ Sigma @ cx - Qxu.T)
                eta = Sigma @ cu @ alpha + Slack_inv @ r
                theta = Sigma @ (cu @ beta + cx)
                chi = -rp - cu @ alpha
                zeta = -(cu @ beta + cx)
                return alpha, beta, eta, theta, chi, zeta

            def indef():
                return (
                    jnp.zeros_like(control),
                    jnp.zeros((control.shape[0], state.shape[0])),
                    jnp.zeros_like(dual),
                    jnp.zeros((dual.shape[0], state.shape[0])),
                    jnp.zeros_like(slack),
                    jnp.zeros((slack.shape[0], state.shape[0])),
                )

            k_control, K_control, k_dual, K_dual, k_slack, K_slack = jax.lax.cond(pos_def, convex, indef)

            # augmented Q-function
            Qx = Qx + cx.T @ Slack_inv @ r
            Qu = Qu + cu.T @ Slack_inv @ r
            Qxx = Qxx + cx.T @ Sigma @ cx
            Qxu = Qxu + cx.T @ Sigma @ cu
            Quu = Quu + cu.T @ Sigma @ cu

            Vx = Qx + Qxu @ k_control
            Vxx = Qxx + Qxu @ K_control
            dV = k_control.T @ Qu + 0.5 * k_control.T @ Quu @ k_control

            error = jnp.hstack((Qu, rp, rd))
            return (Vx, Vxx, pos_def), (k_control, k_dual, k_slack, K_control, K_dual, K_slack, dV, error)

        xN = states[-1]
        VxN = grad(p.final_cost, 0)(xN)
        VxxN = hessian(p.final_cost, 0)(xN)
        feasible = jnp.bool_(1.0)

        carry_out, out = jax.lax.scan(bwd_step, (VxN, VxxN, feasible), (states[:-1], controls, slacks, duals), reverse=True)
        k_u, k_z, k_s, K_u, K_z, K_s, dV, err = out
        _, _, feasible = carry_out
        return k_u, k_z, k_s, K_u, K_z, K_s, jnp.sum(dV), jnp.max(jnp.abs(err)), feasible

    # ------------------------------------------------------------- forward pass ---

    def forward_pass(self, states, controls, slacks, duals, k_u, k_s, k_z, K_u, K_s, K_z):
        dynamics = self.problem.dynamics

        def fwd_step(prev_state, inp):
            state, control, slack, dual, kc, ks, kz, Kc, Ks, Kz = inp

            control = control + kc + Kc @ (prev_state - state)

            dual_direction = kz + Kz @ (prev_state - state)
            max_dual_step = jnp.where(-0.995 * dual / dual_direction > 0.0, -0.995 * dual / dual_direction, 0)
            max_dual_step = jnp.min(max_dual_step, initial=1.0, where=max_dual_step > 0)
            dual = dual + max_dual_step * dual_direction

            slack_direction = ks + Ks @ (prev_state - state)
            max_slack_step = jnp.where(-0.995 * slack / slack_direction > 0.0, -0.995 * slack / slack_direction, 0)
            max_slack_step = jnp.min(max_slack_step, initial=1.0, where=max_slack_step > 0)
            slack = slack + max_slack_step * slack_direction

            next_state = dynamics(prev_state, control)
            feasible = jnp.logical_and(jnp.all(dual > 0), jnp.all(slack > 0))
            return next_state, (next_state, control, slack, dual, feasible)

        _, out = jax.lax.scan(fwd_step, states[0], (states[:-1], controls, slacks, duals, k_u, k_s, k_z, K_u, K_s, K_z))
        new_states, new_controls, new_slacks, new_duals, feasible = out
        return jnp.vstack((states[0], new_states)), new_controls, new_slacks, new_duals, jnp.all(feasible)

    # --------------------------------------------------------------- main loop ---

    def solve(self, x0, controls, slacks, duals) -> IPDDPSolution:
        """Solve from ``x0`` with initial ``controls (T, m)``, ``slacks`` and ``duals`` ``(T, n_c)``.

        The controls need not be feasible (infeasible variant), but slacks and duals must be
        strictly positive; concrete inputs are checked.
        """
        require_x64()
        x0, controls, slacks, duals = (_as_float64(a) for a in (x0, controls, slacks, duals))
        if not isinstance(slacks, jax.core.Tracer) and not isinstance(duals, jax.core.Tracer):
            if not (bool(jnp.all(slacks > 0)) and bool(jnp.all(duals > 0))):
                raise ValueError("initial slacks and duals must be strictly positive")
        return self._solve_jit(x0, controls, slacks, duals)

    def _solve(self, x0, initial_controls, initial_slacks, initial_duals) -> IPDDPSolution:
        opt = self.options

        def iteration(val):
            states, controls, slacks, duals, reg_param, reg_factor, barrier_param, _, it = val
            cost = self._barrier_total_cost(states, controls, slacks, barrier_param)

            k_u, k_z, k_s, K_u, K_z, K_s, diff_cost, err, feasible_bp = self.backward_pass(
                states, controls, slacks, duals, barrier_param, reg_param
            )
            new_states, new_controls, new_slacks, new_duals, _ = self.forward_pass(
                states, controls, slacks, duals, k_u, k_s, k_z, K_u, K_s, K_z
            )
            new_cost = self._barrier_total_cost(new_states, new_controls, new_slacks, barrier_param)

            val_change = cost - new_cost
            gain_ratio = val_change / (-diff_cost)

            def accept_step():
                return (
                    reg_param * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
                    2.0,
                    new_states,
                    new_controls,
                    new_slacks,
                    new_duals,
                )

            def reject_step():
                return reg_param * reg_factor, reg_factor * 2.0, states, controls, slacks, duals

            reg_param, reg_factor, states, controls, slacks, duals = jax.lax.cond(
                jnp.logical_and(gain_ratio > 0.0, feasible_bp), accept_step, reject_step
            )
            reg_param = jnp.clip(reg_param, min=opt.reg_min, max=opt.reg_max)

            barrier_param = jnp.where(
                jnp.logical_and(err < 0.2 * barrier_param, feasible_bp),
                jnp.min(jnp.array([0.2 * barrier_param, barrier_param**1.2])),
                barrier_param,
            )
            return states, controls, slacks, duals, reg_param, reg_factor, barrier_param, err, it + 1

        def keep_going(val):
            *_, barrier_param, err, it = val
            done = err < opt.tol
            if opt.barrier_tol is not None:
                done = jnp.logical_and(done, barrier_param < opt.barrier_tol)
            return jnp.logical_and(jnp.logical_not(done), it < opt.max_iter)

        initial_states = rollout(self.problem.dynamics, x0, initial_controls)
        horizon = initial_controls.shape[0]
        n_c = self.constraints(initial_states[0], initial_controls[0]).shape[0]
        if opt.barrier_init is not None:
            initial_barrier = jnp.asarray(opt.barrier_init, dtype=jnp.float64)
        else:  # original rule; fall back only where it is non-positive (bitwise-neutral otherwise)
            rule = self._barrier_total_cost(initial_states, initial_controls, initial_slacks, 0.0) / (horizon * n_c)
            initial_barrier = jnp.where(rule > 0, rule, opt.barrier_init_fallback)

        init = (
            initial_states,
            initial_controls,
            initial_slacks,
            initial_duals,
            opt.reg_init,
            opt.reg_factor_init,
            initial_barrier,
            jnp.inf,
            0,
        )
        states, controls, slacks, duals, _, _, barrier, err, it = jax.lax.while_loop(keep_going, iteration, init)
        converged = err < opt.tol
        if opt.barrier_tol is not None:
            converged = jnp.logical_and(converged, barrier < opt.barrier_tol)
        return IPDDPSolution(states, controls, slacks, duals, it, barrier, err, termination_status(converged, err, it))
