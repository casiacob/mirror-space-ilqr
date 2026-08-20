import jax
import jax.numpy as jnp
import jax.scipy as jsc
import optimistix as optx

from mmilqr.utils import rollout_dynamics, discretize_dynamics
from mmilqr.objects import DualOCP
from jax import config
from mmilqr import solve_dual_ilqr
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


def wiener_model(state: jnp.ndarray, control: jnp.ndarray):
    A = jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    B = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    return A @ state + B @ control


def final_cost(state: jnp.ndarray):
    goal_state = jnp.array([1.0, 1.0, 0.0, 0.0])
    final_state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e0, 1e0]))

    err = state - goal_state
    c = 0.5 * err.T @ final_state_cost @ err
    return c


def transient_cost(state: jnp.ndarray, control: jnp.ndarray):
    goal_state = jnp.array([1.0, 1.0, 0.0, 0.0])
    state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e0, 1e0]))
    control_cost = jnp.diag(jnp.array([5e-2, 5e-2]))

    err = state - goal_state
    c = 0.5 * err.T @ state_cost @ err
    c += 0.5 * control.T @ control_cost @ control
    return c


def total_cost(states: jnp.ndarray, controls: jnp.ndarray):
    ct = jax.vmap(transient_cost)(states[:-1], controls)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


# polytope constraints A u < b
A_control = jnp.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
    ]
)

b_control = jnp.array(
    [
        0.5,
        0.5,
        0.5,
        0.5,
        0.7,
        0.7,
        0.7,
        0.7,
    ]
)


def control_slack(control: jnp.ndarray):
    return b_control - A_control @ control


def control_dgf(control: jnp.ndarray):
    s = control_slack(control)
    return 0.5 * control.T @ control - jnp.sum(jnp.log(s))


def control_mirror_map(control: jnp.ndarray):
    s = control_slack(control)
    return control + A_control.T @ (1.0 / s)


def control_dgf_hessian(control: jnp.ndarray):
    s = control_slack(control)
    W = 1.0 / s**2
    n_u = control.shape[0]
    return jnp.eye(n_u) + A_control.T @ (W[:, None] * A_control)


def control_inv_dgf_hessian(control: jnp.ndarray):
    H = control_dgf_hessian(control)
    return jsc.linalg.solve(H, jnp.eye(control.shape[0]))


def softplus_inv(s: jnp.ndarray):
    return s + jnp.log1p(-jnp.exp(-s))


def control_inverse_map(dual_control: jnp.ndarray, prev_controls: jnp.ndarray):
    n_u = dual_control.shape[0]

    def residual(q, dual_control):
        u = q[:n_u]
        r = q[n_u:]

        s = jax.nn.softplus(r)

        mirror_residual = u + A_control.T @ (1.0 / s) - dual_control

        slack_residual = A_control @ u + s - b_control

        return jnp.concatenate((mirror_residual, slack_residual))

    solver = optx.Newton(
        rtol=1e-11,
        atol=1e-11,
        norm=optx.rms_norm,
    )

    u0 = prev_controls
    s0 = b_control - A_control @ u0
    r0 = softplus_inv(s0)
    q0 = jnp.concatenate((u0, r0))

    sol = optx.root_find(
        residual,
        solver,
        q0,
        args=dual_control,
        max_steps=128,
        throw=False,
    )
    return sol.value[:n_u]


def state_dgf_hessian(state: jnp.ndarray):
    return jnp.eye(state.shape[0])


def state_dgf_inv_hessian(state: jnp.ndarray):
    return jnp.eye(state.shape[0])


def state_mirror_map(state: jnp.ndarray):
    return state


dt = 0.02
discrete_wiener = discretize_dynamics(wiener_model, dt)

horizon = 15
key = jax.random.PRNGKey(12)

x0 = jnp.array([0.2, -0.1, -0.1, 0.1])

u_init = jax.random.normal(key, shape=(horizon, 2)) * 0.1


x_init = rollout_dynamics(discrete_wiener, u_init, x0)

wiener_ocp = DualOCP(
    discrete_wiener,
    transient_cost,
    final_cost,
    total_cost,
    state_mirror_map,
    state_dgf_hessian,
    state_dgf_inv_hessian,
    control_mirror_map,
    control_dgf_hessian,
    control_inv_dgf_hessian,
    control_inverse_map,
)


def mm_mpc_body(carry, inp):
    nominal_states, nominal_controls = carry

    optimal_states, optimal_controls, _, _, iterations = solve_dual_ilqr(
        nominal_states,
        nominal_controls,
        wiener_ocp,
    )

    new_nominal_controls = jnp.vstack(
        (
            optimal_controls[1:],
            optimal_controls[-1],
        )
    )

    new_nominal_states = rollout_dynamics(
        wiener_ocp.dynamics,
        new_nominal_controls,
        optimal_states[1],
    )

    return (new_nominal_states, new_nominal_controls), (
        optimal_states[1],
        optimal_controls[0],
        optimal_states,
    )


_, (mm_mpc_states, mm_mpc_controls, all_states) = jax.lax.scan(
    mm_mpc_body,
    (x_init, u_init),
    xs=None,
    length=300,
)

mm_mpc_states = jnp.vstack((x0, mm_mpc_states))

plt.plot(mm_mpc_states[:, 0])
plt.plot(mm_mpc_states[:, 1])
plt.show()

plt.plot(mm_mpc_controls)
plt.show()
