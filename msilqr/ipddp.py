"""Infeasible interior-point DDP: the baseline of the paper's comparison (Sec. V, [20]).

Pavlov, Shames and Manzie, "Interior point differential dynamic programming", IEEE
TCST 29(6), 2021 -- the "infeasible" variant: inequality constraints
``g(x, u) <= 0`` become ``g(x, u) + s = 0`` with slacks ``s > 0`` and duals ``z > 0``;
each backward pass solves the perturbed KKT system of the barrier subproblem with
barrier parameter ``mu``, and ``mu`` is decreased when the subproblem residual is small.
Iterates may violate the constraints until convergence.

Changes relative to the original baseline (commit effb14c), which produced the paper's
Fig. 4. ``IPDDPOptions.paper()`` reproduces that baseline operation for operation (the
regression benchmarks check it bitwise); it includes the fixes in this first list:

* ``jnp.clip(a_min=, a_max=)`` -> ``jnp.clip(min=, max=)`` (the original crashes on
  current JAX);
* the feedback gain of the state constraint term uses ``Sigma = S^{-1} Z`` instead of
  ``S^{-1}`` (a bug; it only matters for state-dependent constraints, the benchmarks
  constrain controls only);
* the definiteness test of the symmetrized ``Q_uu`` uses ``eigvalsh`` instead of the
  general complex ``eigvals`` (CPU-only in JAX; ``>`` on complex numbers is
  lexicographic). Bitwise identical on all benchmarks;
* an iteration cap and a status; slacks and duals are checked to be positive;
* option ``barrier_init``, and a fallback when the original rule ``mu_0 = J(u_0)/(T n_c)``
  gives ``mu_0 <= 0`` (non-positive costs);
* option ``barrier_tol``: also require ``mu < barrier_tol`` before stopping.

The default options also resolve the original's departures from Pavlov et al.
(``docs/PAPER_NOTES.md``, notes 2 and 4):

* it stops only once ``mu < barrier_tol`` (default ``1e-6``); the original stopped on the
  barrier-subproblem residual alone and returned central-path points with ``mu`` far
  above ``tol``. ``mu`` is decreased down to ``barrier_tol / 10``, also once the residual
  is below ``tol`` (otherwise a ``mu`` just above ``barrier_tol`` can stall the solve);
* controls, slacks and duals take one step size ``alpha`` at every stage, halved until the
  fraction-to-boundary rule holds (the original took the full control step and separate
  per-stage factors for slacks and duals);
* a step is accepted by a filter on (barrier cost, ``||g + s||_1``), so primal
  infeasibility counts (the original accepted any decrease of the barrier cost). The
  filter restarts whenever ``mu`` decreases. The Levenberg-Marquardt update is kept, with
  the model decrease scaled to ``alpha``;
* :func:`~msilqr.ipddp_mpc` warm-starts slacks, duals and ``mu``.

The stopping measure (max of ``|Qhat_u|``, ``|r_p|``, ``|r_d|``, where
``Qhat_u = Q_u + c_u^T S^{-1} r`` is the gradient of the barrier-augmented Q-function) still
differs from the mirror-space solver's (``max |Q_y|``), so "the same tolerance" does not
mean the same accuracy (note 3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jcp
from jax import grad, hessian

from msilqr._common import (
    RegularizationOptions,
    as_float64,
    check_slacks_duals,
    check_trajectory_shapes,
    expand_stage,
    gain_ratio_update,
    require_x64,
    termination_status,
)
from msilqr.problem import OptimalControlProblem, rollout

__all__ = ["InteriorPointDDP", "IPDDPOptions", "IPDDPSolution"]


@dataclass(frozen=True)
class IPDDPOptions(RegularizationOptions):
    """Options of :class:`InteriorPointDDP`; :meth:`paper` gives the baseline of the paper's Fig. 4.

    The stopping measure is ``max(|Qhat_u|, |r_p|, |r_d|)``; the stopping and
    Levenberg-Marquardt fields are those of :class:`~msilqr.RegularizationOptions`.

    Attributes:
        barrier_tol: if set, additionally require ``mu < barrier_tol`` to stop (``None``:
            stop on the barrier-subproblem residual alone, as the paper's baseline did).
        line_search: one step size for all stages, slacks and duals, accepted by a filter
            on (barrier cost, ``||g + s||_1``). ``False``: the paper's full step with
            per-stage fraction-to-boundary factors, accepted when the barrier cost
            decreases.
        max_backtracks: the line search tries ``alpha = 1, 1/2, ..., 2^-max_backtracks``.
        barrier_init: initial barrier parameter. ``None`` uses the original rule
            ``mu_0 = J(u_0) / (T n_c)`` (Pavlov et al.).
        barrier_init_fallback: ``mu_0`` used when the original rule gives ``mu_0 <= 0``
            (non-positive costs; the original code then stalled). Whenever the rule gives
            a positive value it is used unchanged, so results with positive costs are
            bitwise identical to the original.
    """

    barrier_tol: float | None = 1e-6
    barrier_init: float | None = None
    barrier_init_fallback: float = 1.0
    line_search: bool = True
    max_backtracks: int = 10

    def __post_init__(self):
        super().__post_init__()
        for name in ("barrier_tol", "barrier_init", "barrier_init_fallback"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if not (isinstance(self.max_backtracks, int) and self.max_backtracks >= 0):
            raise ValueError(f"max_backtracks must be a non-negative integer, got {self.max_backtracks}")

    @classmethod
    def paper(cls, **overrides) -> IPDDPOptions:
        """The baseline of the paper's Fig. 4: no line search or filter, and it stops on the
        barrier-subproblem residual alone (reproduced bitwise by the benchmarks)."""
        return cls(**{"barrier_tol": None, "line_search": False, **overrides})


class IPDDPSolution(NamedTuple):
    states: jax.Array
    controls: jax.Array
    slacks: jax.Array
    duals: jax.Array
    iterations: jax.Array
    barrier: jax.Array  #: final barrier parameter ``mu``
    error: jax.Array  #: last stopping measure
    status: jax.Array  #: :class:`~msilqr.Status` code


# ------------------------------------------------------------- stage computations ---


class PerturbedKKT(NamedTuple):
    """Quantities of the perturbed KKT system at one stage (slack ``s``, dual ``z``)."""

    S_inv: jax.Array  #: ``diag(1 / s)``
    Sigma: jax.Array  #: ``diag(z / s)``
    rp: jax.Array  #: primal residual ``g(x, u) + s``
    rd: jax.Array  #: complementarity residual ``z s - mu``
    r: jax.Array  #: ``z r_p - r_d``


def perturbed_kkt(constraint_value, slack, dual, barrier) -> PerturbedKKT:
    S_inv = jnp.diag(1 / slack)
    Sigma = S_inv @ jnp.diag(dual)
    rp = constraint_value + slack
    rd = dual * slack - barrier
    r = dual * rp - rd
    return PerturbedKKT(S_inv, Sigma, rp, rd, r)


def newton_gains(A, Qu, Qxu, cx, cu, kkt: PerturbedKKT):
    """Feedforward/feedback gains of controls, duals and slacks from the reduced KKT
    system ``A = Qhat_uu + c_u^T Sigma c_u`` (eliminating the slack and dual steps)."""
    chol_and_lower = jcp.linalg.cho_factor(A)
    alpha = jcp.linalg.cho_solve(chol_and_lower, -cu.T @ kkt.S_inv @ kkt.r - Qu)
    beta = jcp.linalg.cho_solve(chol_and_lower, -cu.T @ kkt.Sigma @ cx - Qxu.T)
    eta = kkt.Sigma @ cu @ alpha + kkt.S_inv @ kkt.r
    theta = kkt.Sigma @ (cu @ beta + cx)
    chi = -kkt.rp - cu @ alpha
    zeta = -(cu @ beta + cx)
    return alpha, beta, eta, theta, chi, zeta


def zero_gains(n_x, n_u, n_c):
    return (
        jnp.zeros(n_u),
        jnp.zeros((n_u, n_x)),
        jnp.zeros(n_c),
        jnp.zeros((n_c, n_x)),
        jnp.zeros(n_c),
        jnp.zeros((n_c, n_x)),
    )


def augmented_q(Qx, Qu, Qxx, Qxu, Quu, cx, cu, kkt: PerturbedKKT):
    """Q-function of the barrier subproblem after eliminating slacks and duals."""
    return (
        Qx + cx.T @ kkt.S_inv @ kkt.r,
        Qu + cu.T @ kkt.S_inv @ kkt.r,
        Qxx + cx.T @ kkt.Sigma @ cx,
        Qxu + cx.T @ kkt.Sigma @ cu,
        Quu + cu.T @ kkt.Sigma @ cu,
    )


def fraction_to_boundary(value, direction, tau=0.995):
    """Largest step in ``(0, 1]`` that keeps ``value + step * direction`` above
    ``(1 - tau) * value`` (applied per stage, as in the original)."""
    step = jnp.where(-tau * value / direction > 0.0, -tau * value / direction, 0)
    return jnp.min(step, initial=1.0, where=step > 0)


_TAU = 0.995  # fraction-to-boundary parameter of the line search (as in the paper's baseline)


class _Filter(NamedTuple):
    """Filter points (barrier cost, infeasibility ``||g + s||_1``) in fixed-size arrays."""

    cost: jax.Array
    infeasibility: jax.Array
    valid: jax.Array


def _filter_start(cost, infeasibility, size) -> _Filter:
    return _Filter(
        jnp.full(size, jnp.inf).at[0].set(cost),
        jnp.full(size, jnp.inf).at[0].set(infeasibility),
        jnp.zeros(size, bool).at[0].set(True),
    )


def _filter_rejects(f: _Filter, cost, infeasibility):
    """Whether some filter point is at least as good in both measures."""
    return jnp.any(f.valid & (cost >= f.cost) & (infeasibility >= f.infeasibility))


def _filter_add(f: _Filter, cost, infeasibility) -> _Filter:
    """Add a point and drop the points it dominates."""
    valid = f.valid & ~((cost <= f.cost) & (infeasibility <= f.infeasibility))
    slot = jnp.argmin(valid)  # first free slot
    return _Filter(f.cost.at[slot].set(cost), f.infeasibility.at[slot].set(infeasibility), valid.at[slot].set(True))


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
        """Returns ``(k_u, k_z, k_s, K_u, K_z, K_s, dV, error, positive_definite, dV1)``, where
        ``dV1`` is the linear part of the model decrease ``dV`` (for step sizes below 1)."""
        p, constraints = self.problem, self.constraints

        def step(carry, inp):
            Vx, Vxx, pos_def = carry
            state, control, slack, dual = inp

            e = expand_stage(p.dynamics, self._stage_lagrangian, state, control, dual)
            cx = jax.jacrev(constraints, 0)(state, control)
            cu = jax.jacrev(constraints, 1)(state, control)

            # Q-function expansion of the Lagrangian, regularized and symmetrized
            Qx = e.lx + e.fx.T @ Vx
            Qu = e.lu + e.fu.T @ Vx
            Qxx = e.lxx + e.fx.T @ Vxx @ e.fx
            Quu = e.luu + e.fu.T @ Vxx @ e.fu
            Quu += reg_param * jnp.eye(control.shape[0])
            Qxu = e.lxu + e.fx.T @ Vxx @ e.fu
            Quu = (Quu + Quu.T) / 2.0
            pos_def = jnp.logical_and(jnp.all(jnp.linalg.eigvalsh(Quu) > 0), pos_def)

            kkt = perturbed_kkt(constraints(state, control), slack, dual, barrier_param)
            A = Quu + cu.T @ kkt.Sigma @ cu
            k_u, K_u, k_z, K_z, k_s, K_s = jax.lax.cond(
                pos_def,
                lambda: newton_gains(A, Qu, Qxu, cx, cu, kkt),
                lambda: zero_gains(state.shape[0], control.shape[0], dual.shape[0]),
            )

            Qx, Qu, Qxx, Qxu, Quu = augmented_q(Qx, Qu, Qxx, Qxu, Quu, cx, cu, kkt)
            Vx = Qx + Qxu @ k_u
            Vxx = Qxx + Qxu @ K_u
            dV = k_u.T @ Qu + 0.5 * k_u.T @ Quu @ k_u

            error = jnp.hstack((Qu, kkt.rp, kkt.rd))
            return (Vx, Vxx, pos_def), (k_u, k_z, k_s, K_u, K_z, K_s, dV, error, k_u.T @ Qu)

        x_final = states[-1]
        carry0 = (grad(p.final_cost, 0)(x_final), hessian(p.final_cost, 0)(x_final), jnp.bool_(1.0))
        (_, _, positive_definite), out = jax.lax.scan(step, carry0, (states[:-1], controls, slacks, duals), reverse=True)
        k_u, k_z, k_s, K_u, K_z, K_s, dV, err, dV1 = out
        return k_u, k_z, k_s, K_u, K_z, K_s, jnp.sum(dV), jnp.max(jnp.abs(err)), positive_definite, jnp.sum(dV1)

    # ------------------------------------------------------------- forward pass ---

    def forward_pass(self, states, controls, slacks, duals, k_u, k_s, k_z, K_u, K_s, K_z):
        dynamics = self.problem.dynamics

        def step(prev_state, inp):
            state, control, slack, dual, kc, ks, kz, Kc, Ks, Kz = inp
            dx = prev_state - state
            control = control + kc + Kc @ dx

            dual_direction = kz + Kz @ dx
            dual = dual + fraction_to_boundary(dual, dual_direction) * dual_direction

            slack_direction = ks + Ks @ dx
            slack = slack + fraction_to_boundary(slack, slack_direction) * slack_direction

            next_state = dynamics(prev_state, control)
            feasible = jnp.logical_and(jnp.all(dual > 0), jnp.all(slack > 0))
            return next_state, (next_state, control, slack, dual, feasible)

        _, out = jax.lax.scan(step, states[0], (states[:-1], controls, slacks, duals, k_u, k_s, k_z, K_u, K_s, K_z))
        new_states, new_controls, new_slacks, new_duals, feasible = out
        return jnp.vstack((states[0], new_states)), new_controls, new_slacks, new_duals, jnp.all(feasible)

    def trial_step(self, states, controls, slacks, duals, k_u, k_s, k_z, K_u, K_s, K_z, alpha):
        """Closed-loop rollout with one step size ``alpha`` for controls, slacks and duals at
        every stage; also returns whether a slack or dual leaves the fraction-to-boundary
        region ``new >= (1 - tau) old``."""
        dynamics = self.problem.dynamics

        def step(prev_state, inp):
            state, control, slack, dual, kc, ks, kz, Kc, Ks, Kz = inp
            dx = prev_state - state
            control = control + alpha * kc + Kc @ dx
            new_slack = slack + alpha * ks + Ks @ dx
            new_dual = dual + alpha * kz + Kz @ dx
            violated = jnp.logical_or(jnp.any(new_slack < (1 - _TAU) * slack), jnp.any(new_dual < (1 - _TAU) * dual))
            next_state = dynamics(prev_state, control)
            return next_state, (next_state, control, new_slack, new_dual, violated)

        _, out = jax.lax.scan(step, states[0], (states[:-1], controls, slacks, duals, k_u, k_s, k_z, K_u, K_s, K_z))
        new_states, new_controls, new_slacks, new_duals, violated = out
        return jnp.vstack((states[0], new_states)), new_controls, new_slacks, new_duals, jnp.any(violated)

    def _infeasibility(self, states, controls, slacks):
        """``||g(x, u) + s||_1`` over the trajectory."""
        return jnp.sum(jnp.abs(jax.vmap(self.constraints)(states[:-1], controls) + slacks))

    # --------------------------------------------------------------- main loop ---

    def solve(self, x0, controls, slacks, duals, *, barrier=None) -> IPDDPSolution:
        """Solve from ``x0`` with initial ``controls (T, m)``, ``slacks`` and ``duals`` ``(T, n_c)``.

        The controls need not be feasible (infeasible variant), but slacks and duals must be
        strictly positive; concrete inputs are checked. ``barrier`` overrides the initial
        ``mu`` (for warm starts); by default ``options.barrier_init`` or the original rule.
        """
        require_x64()
        check_trajectory_shapes(x0, controls, slacks, duals)
        x0, controls, slacks, duals = (as_float64(a) for a in (x0, controls, slacks, duals))
        self.validate_slacks_duals(slacks, duals)
        if barrier is not None:
            if not isinstance(barrier, jax.core.Tracer) and not float(barrier) > 0:
                raise ValueError(f"barrier must be > 0, got {barrier}")
            barrier = as_float64(barrier)
        return self._solve_jit(x0, controls, slacks, duals, barrier)

    validate_slacks_duals = staticmethod(check_slacks_duals)

    def _initial_barrier(self, states, controls, slacks):
        opt = self.options
        if opt.barrier_init is not None:
            return jnp.asarray(opt.barrier_init, dtype=jnp.float64)
        horizon, n_c = controls.shape[0], self.constraints(states[0], controls[0]).shape[0]
        rule = self._barrier_total_cost(states, controls, slacks, 0.0) / (horizon * n_c)
        return jnp.where(rule > 0, rule, opt.barrier_init_fallback)  # bitwise-neutral when rule > 0

    def _converged(self, err, barrier_param):
        done = err < self.options.tol
        if self.options.barrier_tol is not None:
            done = jnp.logical_and(done, barrier_param < self.options.barrier_tol)
        return done

    def _solve(self, x0, initial_controls, initial_slacks, initial_duals, barrier) -> IPDDPSolution:
        if self.options.line_search:
            return self._solve_line_search(x0, initial_controls, initial_slacks, initial_duals, barrier)
        return self._solve_paper(x0, initial_controls, initial_slacks, initial_duals, barrier)

    def _solve_paper(self, x0, initial_controls, initial_slacks, initial_duals, barrier) -> IPDDPSolution:
        """The paper's baseline: full step, per-stage fraction-to-boundary, gain-ratio test."""
        opt = self.options

        def iteration(val):
            states, controls, slacks, duals, reg_param, reg_factor, barrier_param, _, it = val
            cost = self._barrier_total_cost(states, controls, slacks, barrier_param)
            k_u, k_z, k_s, K_u, K_z, K_s, diff_cost, err, positive_definite, _ = self.backward_pass(
                states, controls, slacks, duals, barrier_param, reg_param
            )
            new = self.forward_pass(states, controls, slacks, duals, k_u, k_s, k_z, K_u, K_s, K_z)
            new_states, new_controls, new_slacks, new_duals, _ = new
            new_cost = self._barrier_total_cost(new_states, new_controls, new_slacks, barrier_param)

            gain_ratio = (cost - new_cost) / (-diff_cost)
            accept = jnp.logical_and(gain_ratio > 0.0, positive_definite)
            reg_param, reg_factor = gain_ratio_update(reg_param, reg_factor, gain_ratio, accept, opt)
            states, controls, slacks, duals = (
                jnp.where(accept, a, b)
                for a, b in zip((new_states, new_controls, new_slacks, new_duals), (states, controls, slacks, duals), strict=True)
            )

            # decrease the barrier parameter once the subproblem residual is small
            barrier_param = jnp.where(
                jnp.logical_and(err < 0.2 * barrier_param, positive_definite),
                jnp.min(jnp.array([0.2 * barrier_param, barrier_param**1.2])),
                barrier_param,
            )
            return states, controls, slacks, duals, reg_param, reg_factor, barrier_param, err, it + 1

        def keep_going(val):
            *_, barrier_param, err, it = val
            return jnp.logical_and(jnp.logical_not(self._converged(err, barrier_param)), it < opt.max_iter)

        states = rollout(self.problem.dynamics, x0, initial_controls)
        init = (
            states,
            initial_controls,
            initial_slacks,
            initial_duals,
            opt.reg_init,
            opt.reg_factor_init,
            self._initial_barrier(states, initial_controls, initial_slacks) if barrier is None else barrier,
            jnp.inf,
            0,
        )
        states, controls, slacks, duals, _, _, barrier, err, it = jax.lax.while_loop(keep_going, iteration, init)
        return IPDDPSolution(states, controls, slacks, duals, it, barrier, err, termination_status(self._converged(err, barrier), err, it))

    def _solve_line_search(self, x0, initial_controls, initial_slacks, initial_duals, barrier) -> IPDDPSolution:
        """One step size for all stages, slacks and duals, accepted by a filter; stops only
        once ``mu < barrier_tol`` (if set)."""
        opt = self.options
        mu_floor = (opt.barrier_tol if opt.barrier_tol is not None else opt.tol) / 10.0
        size = opt.max_iter + 1  # the filter gains at most one point per iteration

        def start_filter(states, controls, slacks, mu):
            infeasibility = self._infeasibility(states, controls, slacks)
            infeasibility = jnp.where(infeasibility < opt.tol, 0.0, infeasibility)
            return _filter_start(self._barrier_total_cost(states, controls, slacks, mu), infeasibility, size)

        def line_search(states, controls, slacks, duals, gains, mu, filt, descent):
            def trial(k):
                alpha = 0.5**k
                xs, us, ss, zs, violated = self.trial_step(states, controls, slacks, duals, *gains, alpha)
                cost = self._barrier_total_cost(xs, us, ss, mu)
                infeasibility = jnp.maximum(opt.tol, self._infeasibility(xs, us, ss))
                finite = jnp.isfinite(cost) & jnp.all(jnp.isfinite(xs)) & jnp.all(jnp.isfinite(zs))
                ok = ~violated & finite & ~_filter_rejects(filt, cost, infeasibility)
                return ok, alpha, (xs, us, ss, zs, cost, infeasibility)

            def keep_going(carry):
                k, ok, _, _ = carry
                return jnp.logical_and(~ok, k <= opt.max_backtracks)

            def body(carry):
                ok, alpha, point = trial(carry[0])
                return carry[0] + 1, ok, alpha, point

            k0 = jnp.where(descent, 0, opt.max_backtracks + 1)  # no search without a valid step
            init = (k0, jnp.array(False), jnp.array(1.0), (states, controls, slacks, duals, jnp.array(jnp.inf), jnp.array(jnp.inf)))
            _, ok, alpha, point = jax.lax.while_loop(keep_going, body, init)
            return ok, alpha, point

        def iteration(val):
            states, controls, slacks, duals, reg, reg_factor, mu, _, it, filt = val
            cost = self._barrier_total_cost(states, controls, slacks, mu)
            k_u, k_z, k_s, K_u, K_z, K_s, dV, err, positive_definite, dV1 = self.backward_pass(states, controls, slacks, duals, mu, reg)
            gains = (k_u, k_s, k_z, K_u, K_s, K_z)
            accept, alpha, (xs, us, ss, zs, new_cost, infeasibility) = line_search(
                states, controls, slacks, duals, gains, mu, filt, positive_definite
            )
            accept = jnp.logical_and(accept, positive_definite)

            # Levenberg-Marquardt update with the model decrease of the step actually taken
            predicted = alpha * dV1 + alpha**2 * (dV - dV1)
            reg, reg_factor = gain_ratio_update(reg, reg_factor, (cost - new_cost) / (-predicted), accept, opt)
            filt = jax.tree.map(lambda a, b: jnp.where(accept, a, b), _filter_add(filt, new_cost, infeasibility), filt)
            states, controls, slacks, duals = (
                jnp.where(accept, a, b) for a, b in zip((xs, us, ss, zs), (states, controls, slacks, duals), strict=True)
            )

            # decrease mu once the barrier subproblem is solved (or its residual is already below
            # tol, so a mu just above barrier_tol cannot stall), and restart the filter
            decrease = jnp.logical_and(jnp.logical_or(err < 0.2 * mu, err < opt.tol), positive_definite)
            mu = jnp.where(decrease, jnp.maximum(mu_floor, jnp.minimum(0.2 * mu, mu**1.2)), mu)
            filt = jax.tree.map(lambda a, b: jnp.where(decrease, a, b), start_filter(states, controls, slacks, mu), filt)
            return states, controls, slacks, duals, reg, reg_factor, mu, err, it + 1, filt

        def keep_going(val):
            mu, err, it = val[6], val[7], val[8]
            return jnp.logical_and(jnp.logical_not(self._converged(err, mu)), it < opt.max_iter)

        states = rollout(self.problem.dynamics, x0, initial_controls)
        mu0 = self._initial_barrier(states, initial_controls, initial_slacks) if barrier is None else barrier
        filt = start_filter(states, initial_controls, initial_slacks, mu0)
        init = (states, initial_controls, initial_slacks, initial_duals, opt.reg_init, opt.reg_factor_init, mu0, jnp.inf, 0, filt)
        states, controls, slacks, duals, _, _, mu, err, it, _ = jax.lax.while_loop(keep_going, iteration, init)
        return IPDDPSolution(states, controls, slacks, duals, it, mu, err, termination_status(self._converged(err, mu), err, it))
