import jax.numpy as jnp
import jax.scipy as jsc
from objects import DualOCP, Derivatives
from jax import grad, jacrev, hessian
import jax


def compute_derivatives(
    state: jnp.ndarray, control: jnp.ndarray, Phi_x_tp1: jnp.ndarray, ocp: DualOCP
):
    Phi_inv = ocp.state_dgf_inv_hessian(state)
    Psi_inv = ocp.control_dgf_inv_hessian(control)

    cx = grad(ocp.stage_cost, 0)(state, control)
    cu = grad(ocp.stage_cost, 1)(state, control)
    cxx = hessian(ocp.stage_cost, 0)(state, control)
    cuu = hessian(ocp.stage_cost, 1)(state, control)
    cxu = jacrev(jacrev(ocp.stage_cost, 0), 1)(state, control)
    fx = jacrev(ocp.dynamics, 0)(state, control)
    fu = jacrev(ocp.dynamics, 1)(state, control)

    cz = Phi_inv @ cx
    cy = Psi_inv @ cu
    fz = Phi_x_tp1 @ fx @ Phi_inv
    fy = Phi_x_tp1 @ fu @ Psi_inv
    czy = Phi_inv @ cxu @ Psi_inv

    # T_phi = jacrev(ocp.state_dgf_hessian, 0)(state)
    # T_psi = jacrev(ocp.control_dgf_hessian, 0)(control)
    #
    # T_dot_cz = jnp.tensordot(T_phi, cz, axes=(-1, 0))
    # T_dot_cy = jnp.tensordot(T_psi, cy, axes=(-1, 0))

    czz = Phi_inv @ cxx @ Phi_inv
    cyy = Psi_inv @ cuu @ Psi_inv
    return Derivatives(cz, cy, fz, fy, czz, cyy, czy)


def backward_pass(
    state_trajectory: jnp.ndarray,
    control_sequence: jnp.ndarray,
    reg_param: float,
    ocp: DualOCP,
):
    final_state = state_trajectory[-1]

    Phi_final = ocp.state_dgf_hessian(final_state)
    Phi_inv_final = ocp.state_dgf_inv_hessian(final_state)
    cx_final = grad(ocp.final_cost, 0)(final_state)
    cxx_final = hessian(ocp.final_cost, 0)(final_state)
    Vz_final = Phi_inv_final @ cx_final
    T_phi_final = jacrev(ocp.state_dgf_hessian, 0)(final_state)
    T_dot_Vz = jnp.tensordot(T_phi_final, Vz_final, axes=(-1, 0))
    Vzz_final = Phi_inv_final @ (cxx_final - T_dot_Vz) @ Phi_inv_final

    def body(carry, inp):
        Vz, Vzz, Phi_tp1 = carry
        x, u = inp
        d = compute_derivatives(x, u, Phi_tp1, ocp)

        Qz = d.cz + d.fz.T @ Vz
        Qy = d.cy + d.fy.T @ Vz

        Qzz = d.czz + d.fz.T @ Vzz @ d.fz
        Qyy = d.cyy + d.fy.T @ Vzz @ d.fy + reg_param * jnp.eye(u.shape[0])
        Qzy = d.czy + d.fz.T @ Vzz @ d.fy
        Qyz = Qzy.T

        eigvals = jnp.linalg.eigvalsh(Qyy)
        is_pos_def = jnp.all(eigvals > 0)

        k = -jsc.linalg.solve(Qyy, Qy)
        K = -jsc.linalg.solve(Qyy, Qyz)

        Vz = Qz + Qzy @ k + K.T @ Qy + K.T @ Qyy @ k
        Vzz = Qzz + K.T @ Qyy @ K + K.T @ Qyz + Qzy @ K
        Vzz = 0.5 * (Vzz + Vzz.T)
        dV = Qy @ k + 0.5 * k.T @ Qyy @ k
        Phi_tp1 = ocp.state_dgf_hessian(x)

        return (Vz, Vzz, Phi_tp1), (K, k, dV, is_pos_def, Qy)

    _, (gain, gain_ff, val_fun_stage_diff, pos_def_flags, Q_grad) = jax.lax.scan(
        body,
        (Vz_final, Vzz_final, Phi_final),
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
    dual_states = jax.vmap(ocp.state_mirror_map)(states)
    dual_controls = jax.vmap(ocp.control_mirror_map)(controls)

    def body(carry, inp):
        prev_z, prev_x = carry
        z, y, prev_u, k, K = inp
        dz = prev_z - z
        y_new = y + k + K @ dz
        u_new = ocp.control_inverse_map(y_new, prev_u)
        x_new = ocp.dynamics(prev_x, u_new)
        z_new = ocp.state_mirror_map(x_new)
        return (z_new, x_new), (x_new, u_new)

    _, (new_states, new_controls) = jax.lax.scan(
        body,
        (dual_states[0], states[0]),
        (dual_states[:-1], dual_controls, controls, gain_ff, gain),
    )

    new_states = jnp.vstack((states[0], new_states))
    return new_states, new_controls


def solve_dual_ilqr(
    initial_states: jnp.ndarray,
    initial_controls: jnp.ndarray,
    ocp: DualOCP,
    reg_param_init: float = 1e-2,
    r_inc_init: float = 2.0,
    tol: float = 1e-6,
):
    def cond_fun(carry):
        _, _, _, _, _, it, opt_err = carry

        # exit_condition = jnp.logical_or(rel_dcost < tol, it > max_iters)
        exit_condition = opt_err < tol
        return jnp.logical_not(exit_condition)

    def body_fun(carry):
        states, controls, rp, r_inc, cost, it, _ = carry

        K, k, predicted_reduction, bwd_pass_feasible, opt_err = backward_pass(
            states, controls, rp, ocp
        )

        new_states, new_controls = forward_pass(states, controls, K, k, ocp)
        new_cost = ocp.total_cost(new_states, new_controls)

        # Positive means decrease in cost
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
