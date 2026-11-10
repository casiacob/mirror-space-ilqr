import jax.numpy as jnp
import jax.scipy as jsc
from msilqr.objects import DualOCP, Derivatives
from jax import grad, jacrev, hessian
import jax


def compute_derivatives(
    state: jnp.ndarray, control: jnp.ndarray, ocp: DualOCP
):
    Psi_inv = ocp.control_dgf_inv_hessian(control)

    cx = grad(ocp.stage_cost, 0)(state, control)
    cu = grad(ocp.stage_cost, 1)(state, control)
    cxx = hessian(ocp.stage_cost, 0)(state, control)
    cuu = hessian(ocp.stage_cost, 1)(state, control)
    cxu = jacrev(jacrev(ocp.stage_cost, 0), 1)(state, control)
    fx = jacrev(ocp.dynamics, 0)(state, control)
    fu = jacrev(ocp.dynamics, 1)(state, control)


    cy = Psi_inv @ cu
    fy = fu @ Psi_inv
    cxy = cxu @ Psi_inv

    cyy = Psi_inv @ cuu @ Psi_inv
    return Derivatives(cx, cy, fx, fy, cxx, cyy, cxy)


def backward_pass(
    state_trajectory: jnp.ndarray,
    control_sequence: jnp.ndarray,
    reg_param: float,
    ocp: DualOCP,
):
    final_state = state_trajectory[-1]

    cx_final = grad(ocp.final_cost, 0)(final_state)
    cxx_final = hessian(ocp.final_cost, 0)(final_state)
    Vx_final = cx_final
    Vxx_final = cxx_final

    def body(carry, inp):
        Vx, Vxx = carry
        x, u = inp
        d = compute_derivatives(x, u, ocp)

        Qx = d.cx + d.fx.T @ Vx
        Qy = d.cy + d.fy.T @ Vx

        Qxx = d.cxx + d.fx.T @ Vxx @ d.fx
        Qyy = d.cyy + d.fy.T @ Vxx @ d.fy + reg_param * jnp.eye(u.shape[0])
        Qxy = d.cxy + d.fx.T @ Vxx @ d.fy
        Qyx = Qxy.T

        eigvals = jnp.linalg.eigvalsh(Qyy)
        is_pos_def = jnp.all(eigvals > 0)

        k = -jsc.linalg.solve(Qyy, Qy)
        K = -jsc.linalg.solve(Qyy, Qyx)

        Vx = Qx + Qxy @ k + K.T @ Qy + K.T @ Qyy @ k
        Vxx = Qxx + K.T @ Qyy @ K + K.T @ Qyx + Qxy @ K
        Vxx = 0.5 * (Vxx + Vxx.T)
        dV = Qy @ k + 0.5 * k.T @ Qyy @ k

        return (Vx, Vxx), (K, k, dV, is_pos_def, Qy)

    _, (gain, gain_ff, val_fun_stage_diff, pos_def_flags, Q_grad) = jax.lax.scan(
        body,
        (Vx_final, Vxx_final),
        (state_trajectory[:-1], control_sequence),
        reverse=True,
    )

    val_fun_diff = jnp.sum(val_fun_stage_diff)
    all_pos_def = jnp.all(pos_def_flags)
    return gain, gain_ff, val_fun_diff, all_pos_def, jnp.max(jnp.abs(Q_grad))


def forward_pass(
    states: jnp.ndarray,
    controls: jnp.ndarray,
    gain: jnp.ndarray,
    gain_ff: jnp.ndarray,
    ocp: DualOCP,
):
    mirror_controls = jax.vmap(ocp.control_mirror_map)(controls)

    def body(carry, inp):
        prev_x = carry
        x, y, prev_u, k, K = inp
        dx = prev_x - x
        y_new = y + k + K @ dx
        u_new = ocp.control_inverse_map(y_new, prev_u)
        x_new = ocp.dynamics(prev_x, u_new)
        return x_new, (x_new, u_new)

    _, (new_states, new_controls) = jax.lax.scan(
        body,
        states[0],
        (states[:-1], mirror_controls, controls, gain_ff, gain),
    )

    new_states = jnp.vstack((states[0], new_states))
    return new_states, new_controls


def solve_mirror_ilqr(
    initial_states: jnp.ndarray,
    initial_controls: jnp.ndarray,
    ocp: DualOCP,
    reg_param_init: float = 1e-2,
    r_inc_init: float = 2.0,
    tol: float = 1e-6,
):
    def cond_fun(carry):
        _, _, _, _, _, it, opt_err = carry
        exit_condition = opt_err < tol
        return jnp.logical_not(exit_condition)

    def body_fun(carry):
        states, controls, rp, r_inc, cost, it, _ = carry

        K, k, predicted_reduction, bwd_pass_feasible, opt_err = backward_pass(
            states, controls, rp, ocp
        )

        new_states, new_controls = forward_pass(states, controls, K, k, ocp)
        new_cost = ocp.total_cost(new_states, new_controls)

        actual_reduction = new_cost - cost

        gain_ratio = actual_reduction / predicted_reduction

        succesful_minimization = jnp.logical_and(
            jnp.logical_and(actual_reduction < 0.0, predicted_reduction < 0.0),
            bwd_pass_feasible,
        )

        rp_new = jnp.where(
            succesful_minimization,
            rp * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3),
            rp * r_inc,
        )
        r_inc_new = jnp.where(succesful_minimization, 2.0, 2.0 * r_inc)
        rp_new = jnp.clip(rp_new, 1e-16, 1e16)

        states = jnp.where(succesful_minimization, new_states, states)
        controls = jnp.where(succesful_minimization, new_controls, controls)
        cost = jnp.where(succesful_minimization, new_cost, cost)
        it_new = it + 1
        return states, controls, rp_new, r_inc_new, cost, it_new, opt_err

    init_cost = ocp.total_cost(initial_states, initial_controls)

    (
        optimal_states,
        optimal_controls,
        final_rp,
        final_r_inc,
        final_cost,
        total_it,
        _,
    ) = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (
            initial_states,
            initial_controls,
            jnp.array(reg_param_init),
            jnp.array(r_inc_init),
            init_cost,
            0,
            1e10,
        ),
    )
    return optimal_states, optimal_controls, final_cost, final_rp, total_it
