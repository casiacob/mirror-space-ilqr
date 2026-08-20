import jax.numpy as jnp
import jax.lax
from jax import jacrev, grad, hessian
import jax.scipy as jcp
from objects import IPOCP
from utils import rollout_dynamics
from typing import Callable


def bwd_pass(
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
    nominal_slacks: jnp.ndarray,
    nominal_duals: jnp.ndarray,
    barrier_param: float,
    reg_param: float,
    stage_Lagrangian: Callable,
    ocp: IPOCP,
):
    def bwd_step(carry, inp):
        # unpack carry
        Vx, Vxx, pos_def = carry

        # state, controls, dual
        state, control, slack, dual = inp

        # compute derivatives at nominal_values
        lx = grad(stage_Lagrangian, 0)(state, control, dual)
        fx = jacrev(ocp.dynamics, 0)(state, control)
        lu = grad(stage_Lagrangian, 1)(state, control, dual)
        fu = jacrev(ocp.dynamics, 1)(state, control)
        lxx = hessian(stage_Lagrangian, 0)(state, control, dual)
        luu = hessian(stage_Lagrangian, 1)(state, control, dual)
        lxu = jacrev(jacrev(stage_Lagrangian, 0), 1)(state, control, dual)
        cx = jacrev(ocp.constraints, 0)(state, control)
        cu = jacrev(ocp.constraints, 1)(state, control)

        # Q function expansion coeffs
        Qx = lx + fx.T @ Vx
        Qu = lu + fu.T @ Vx
        Qxx = lxx + fx.T @ Vxx @ fx
        Quu = luu + fu.T @ Vxx @ fu
        Quu += reg_param * jnp.eye(control.shape[0])
        Qxu = lxu + fx.T @ Vxx @ fu
        Quu = (Quu + Quu.T) / 2.0
        pos_def = jnp.logical_and(jnp.all(jnp.linalg.eigvals(Quu) > 0), pos_def)

        # perturbed KKT
        Slack_inv = jnp.diag(1 / slack)
        Sigma = Slack_inv @ jnp.diag(dual)
        rp = ocp.constraints(state, control) + slack
        rd = dual * slack - barrier_param
        r = dual * rp - rd
        A = Quu + cu.T @ Sigma @ cu

        def convex():
            chol_and_lower = jcp.linalg.cho_factor(A)
            alpha = jcp.linalg.cho_solve(chol_and_lower, -cu.T @ Slack_inv @ r - Qu)
            beta = jcp.linalg.cho_solve(chol_and_lower, -cu.T @ Slack_inv @ cx - Qxu.T)
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

        # return petrubed KKT system solution if the minimization problem is convex
        k_control, K_control, k_dual, K_dual, k_slack, K_slack = jax.lax.cond(
            pos_def, convex, indef
        )

        # update Q function expansion parameters
        Qx = Qx + cx.T @ Slack_inv @ r
        Qu = Qu + cu.T @ Slack_inv @ r
        Qxx = Qxx + cx.T @ Sigma @ cx
        Qxu = Qxu + cx.T @ Sigma @ cu
        Quu = Quu + cu.T @ Sigma @ cu

        # Value function expansion parameters and diff cost
        Vx = Qx + Qxu @ k_control
        Vxx = Qxx + Qxu @ K_control
        dV = k_control.T @ Qu + 0.5 * k_control.T @ Quu @ k_control

        error = jnp.hstack((Qu, rp, rd))
        return (Vx, Vxx, pos_def), (
            k_control,
            k_dual,
            k_slack,
            K_control,
            K_dual,
            K_slack,
            dV,
            error,
        )

    xN = nominal_states[-1]
    VxN = grad(ocp.final_cost, 0)(xN)
    VxxN = hessian(ocp.final_cost, 0)(xN)

    feasible = jnp.bool_(1.0)

    carry_out, bwd_pass_out = jax.lax.scan(
        bwd_step,
        (VxN, VxxN, feasible),
        (nominal_states[:-1], nominal_controls, nominal_slacks, nominal_duals),
        reverse=True,
    )
    (
        control_ff_gain,
        dual_ff_gain,
        slack_ff_gain,
        control_gain,
        dual_gain,
        slack_gain,
        diff_cost,
        optimality_error,
    ) = bwd_pass_out

    diff_cost = jnp.sum(diff_cost)
    _, _, feasible = carry_out

    return (
        control_ff_gain,
        dual_ff_gain,
        slack_ff_gain,
        control_gain,
        dual_gain,
        slack_gain,
        diff_cost,
        jnp.max(jnp.abs(optimality_error)),
        feasible,
    )


def fwd_pass(
    nominal_states: jnp.ndarray,
    nominal_controls: jnp.ndarray,
    nominal_slacks: jnp.ndarray,
    nominal_duals: jnp.ndarray,
    control_ff_gain: jnp.ndarray,
    slack_ff_gain,
    dual_ff_gain,
    control_gain: jnp.ndarray,
    slack_gain: jnp.ndarray,
    dual_gain: jnp.ndarray,
    dynamics,
):
    def fwd_step(prev_state, inp):
        (
            state,
            control,
            slack,
            dual,
            k_control,
            k_slack,
            k_dual,
            K_control,
            K_slack,
            K_dual,
        ) = inp

        control = control + k_control + K_control @ (prev_state - state)

        dual_direction = k_dual + K_dual @ (prev_state - state)
        max_dual_step = jnp.where(
            -0.995 * dual / dual_direction > 0.0, -0.995 * dual / dual_direction, 0
        )
        max_dual_step = jnp.min(max_dual_step, initial=1.0, where=max_dual_step > 0)
        dual = dual + max_dual_step * dual_direction

        slack_direction = k_slack + K_slack @ (prev_state - state)
        max_slack_step = jnp.where(
            -0.995 * slack / slack_direction > 0.0, -0.995 * slack / slack_direction, 0
        )
        max_slack_step = jnp.min(max_slack_step, initial=1.0, where=max_slack_step > 0)
        slack = slack + max_slack_step * slack_direction

        next_state = dynamics(prev_state, control)

        feasible = jnp.logical_and(jnp.all(dual > 0), jnp.all(slack > 0))

        return next_state, (next_state, control, slack, dual, feasible)

    _, fwd_pass_out = jax.lax.scan(
        fwd_step,
        nominal_states[0],
        (
            nominal_states[:-1],
            nominal_controls,
            nominal_slacks,
            nominal_duals,
            control_ff_gain,
            slack_ff_gain,
            dual_ff_gain,
            control_gain,
            slack_gain,
            dual_gain,
        ),
    )
    (
        new_states,
        new_controls,
        new_slacks,
        new_duals,
        feasible_fp,
    ) = fwd_pass_out

    new_states = jnp.vstack((nominal_states[0], new_states))

    return (
        new_states,
        new_controls,
        new_slacks,
        new_duals,
        jnp.all(feasible_fp),
    )


def infeasible_ipddp(
    initial_state0: jnp.ndarray,
    initial_controls: jnp.ndarray,
    initial_slacks: jnp.ndarray,
    initial_duals: jnp.ndarray,
    ocp: IPOCP,
    tol: float = 1e-6,
):
    def barrier_stage_cost(
        state: jnp.ndarray,
        control: jnp.ndarray,
        slack: jnp.ndarray,
        barrier_param: float,
    ):
        return ocp.stage_cost(
            state,
            control,
        ) - barrier_param * jnp.sum(jnp.log(slack))

    def barrier_total_cost(
        states: jnp.ndarray,
        controls: jnp.ndarray,
        slack: jnp.ndarray,
        barrier_param: float,
    ):
        J = jax.vmap(barrier_stage_cost, in_axes=(0, 0, 0, None))(
            states[:-1], controls, slack, barrier_param
        )
        return jnp.sum(J) + ocp.final_cost(states[-1])

    def stage_Lagrangian(state: jnp.ndarray, control: jnp.ndarray, dual: jnp.ndarray):
        return ocp.stage_cost(state, control) + dual.T @ ocp.constraints(state, control)

    def ipddp_iteration(val):
        # unpack iteration values
        (
            states,
            controls,
            slacks,
            duals,
            reg_param,
            reg_param_mult_fact,
            barrier_param,
            opt_err_bp,
            loop_counter,
        ) = val

        # compute barrier problem cost at current values
        cost = barrier_total_cost(states, controls, slacks, barrier_param)

        # run backward pass
        (
            control_ff_gain,
            dual_ff_gain,
            slack_ff_gain,
            control_gain,
            dual_gain,
            slack_gain,
            diff_cost,
            opt_err_bp_new,
            feasible_bp,
        ) = bwd_pass(
            states,
            controls,
            slacks,
            duals,
            barrier_param,
            reg_param,
            stage_Lagrangian,
            ocp,
        )

        # run forward pass
        new_states, new_controls, new_slacks, new_duals, feasible_fp = fwd_pass(
            states,
            controls,
            slacks,
            duals,
            control_ff_gain,
            slack_ff_gain,
            dual_ff_gain,
            control_gain,
            slack_gain,
            dual_gain,
            ocp.dynamics,
        )

        # compute cost at new values
        new_cost = barrier_total_cost(
            new_states, new_controls, new_slacks, barrier_param
        )

        # compute gain ratio: actual reduction/predicted reduction
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
            return (
                reg_param * reg_param_mult_fact,
                reg_param_mult_fact * 2.0,
                states,
                controls,
                slacks,
                duals,
            )

        # accept or reject step and update the regularization parameter
        (
            reg_param,
            reg_param_mult_fact,
            states,
            controls,
            slacks,
            duals,
        ) = jax.lax.cond(
            jnp.logical_and(gain_ratio > 0.0, feasible_bp),
            accept_step,
            reject_step,
        )

        # CLIPPING: Prevent reg_param from hitting jnp.inf to avoid NaN blowups
        reg_param = jnp.clip(reg_param, a_min=1e-16, a_max=1e16)

        # check the optimality error and update the barrier parameter
        # Note: opt_err_bp_new is still needed here for barrier updating logic
        barrier_param = jnp.where(
            jnp.logical_and(opt_err_bp_new < 0.2 * barrier_param, feasible_bp),
            jnp.min(jnp.array([0.2 * barrier_param, barrier_param**1.2])),
            barrier_param,
        )

        loop_counter += 1

        return (
            states,
            controls,
            slacks,
            duals,
            reg_param,
            reg_param_mult_fact,
            barrier_param,
            opt_err_bp_new,
            loop_counter,
        )

    def ipddp_convergence(val):
        # We no longer extract opt_err to use it for convergence logic
        _, _, _, _, _, _, _, opt_err_bp_new, loop_counter = val
        exit_condition = opt_err_bp_new < tol
        return jnp.logical_not(exit_condition)

    # initialize regularization parameter and multiplication factor
    initial_reg_param = 1e-2
    initial_reg_param_mult_fact = 2.0

    # run the dynamics on the initial controls and initial state for initial states
    initial_states = rollout_dynamics(ocp.dynamics, initial_controls, initial_state0)

    # initialize barrier parameter
    horizon = initial_controls.shape[0]
    n_c = ocp.constraints(initial_states[0], initial_controls[0]).shape[0]
    initial_barrier_param = barrier_total_cost(
        initial_states, initial_controls, initial_slacks, 0.0
    ) / (horizon * n_c)

    # run infeasible ipddp
    (
        optimal_states,
        optimal_controls,
        optimal_slacks,
        optimal_duals,
        _,
        _,
        _,
        _,
        iterations,
    ) = jax.lax.while_loop(
        ipddp_convergence,
        ipddp_iteration,
        (
            initial_states,
            initial_controls,
            initial_slacks,
            initial_duals,
            initial_reg_param,
            initial_reg_param_mult_fact,
            initial_barrier_param,
            jnp.inf,  # initial opt_err (still needed in state tuple)
            0,  # initial loop_counter
        ),
    )
    return (optimal_states, optimal_controls, optimal_slacks, optimal_duals, iterations)
