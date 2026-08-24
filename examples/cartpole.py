import jax
import jax.numpy as jnp
from mmilqr.utils import wrap_angle, rollout_dynamics, discretize_dynamics
from mmilqr.objects import DualOCP
from jax import config
from mmilqr.ilqr import solve_dual_ilqr
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


def cartpole(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:

    # https://underactuated.mit.edu/acrobot.html#cart_pole

    gravity = 9.81
    pole_length = 0.5
    cart_mass = 10.0
    pole_mass = 1.0
    total_mass = cart_mass + pole_mass

    cart_position, pole_position, cart_velocity, pole_velocity = state

    sth = jnp.sin(pole_position)
    cth = jnp.cos(pole_position)

    cart_acceleration = (
        action + pole_mass * sth * (pole_length * pole_velocity**2 + gravity * cth)
    ) / (cart_mass + pole_mass * sth**2)

    pole_acceleration = (
        -action * cth
        - pole_mass * pole_length * pole_velocity**2 * cth * sth
        - total_mass * gravity * sth
    ) / (pole_length * cart_mass + pole_length * pole_mass * sth**2)

    return jnp.hstack(
        (cart_velocity, pole_velocity, cart_acceleration, pole_acceleration)
    )


def final_cost(state: jnp.ndarray):
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    final_state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e-1, 1e-1]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ final_state_cost @ (_wrapped - goal_state)
    return c


def transient_cost(state: jnp.ndarray, action: jnp.ndarray) -> float:
    goal_state = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    state_cost = jnp.diag(jnp.array([1e1, 1e1, 1e-1, 1e-1]))
    action_cost = jnp.diag(jnp.array([1e-3]))

    _wrapped = jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))
    c = 0.5 * (_wrapped - goal_state).T @ state_cost @ (_wrapped - goal_state)
    c += 0.5 * action.T @ action_cost @ action
    return c


def total_cost(states: jnp.ndarray, controls: jnp.ndarray):
    ct = jax.vmap(transient_cost)(states[:-1], controls)
    cT = final_cost(states[-1])
    return cT + jnp.sum(ct)


def control_mirror_map(control):
    u_min = -50.0
    u_max = 50.0
    return 0.5 * jnp.log((control - u_min) / (u_max - control))


def control_inverse_map(dual_control, prev_control):
    u_min = -50.0
    u_max = 50.0
    R = 0.5 * (u_max - u_min)
    return R * jnp.tanh(dual_control)


def control_dgf_hessian(control):
    u_min = -50.0
    u_max = 50.0
    numerator = u_max - u_min
    denominator = 2.0 * (control - u_min) * (u_max - control)
    denominator += 1e-16
    h_diag = numerator / denominator
    return jnp.diag(jnp.atleast_1d(h_diag))


def control_inv_dgf_hessian(control):
    u_min = -50.0
    u_max = 50.0
    h_inv_diag = 2.0 * (control - u_min) * (u_max - control) / (u_max - u_min)
    return jnp.diag(jnp.atleast_1d(h_inv_diag))


def state_dgf_hessian(state):
    return jnp.eye(state.shape[0])


def state_dgf_inv_hessian(state):
    return jnp.eye(state.shape[0])


def state_mirror_map(state):
    return state


discrete_cartpole = discretize_dynamics(cartpole, 0.01)
horizon = 100
key = jax.random.PRNGKey(12)
x0 = jnp.array([-0.01, 0.01, -0.01, 0.01])


u_init = jax.random.normal(key, shape=(horizon, 1)) * 0.1
x_init = rollout_dynamics(discrete_cartpole, u_init, x0)

cartpole_ocp = DualOCP(
    discrete_cartpole,
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
        nominal_states, nominal_controls, cartpole_ocp
    )
    cost = total_cost(optimal_states, optimal_controls)
    new_nominal_controls = jnp.vstack((optimal_controls[1:], optimal_controls[-1]))

    new_nominal_states = rollout_dynamics(
        cartpole_ocp.dynamics, new_nominal_controls, optimal_states[1]
    )
    return (new_nominal_states, new_nominal_controls), (
        optimal_states[1],
        optimal_controls[0],
        cost,
        iterations,
    )


_, (mm_mpc_states, mm_mpc_controls, mpc_traj_costs, mpc_iterations) = jax.lax.scan(
    mm_mpc_body, (x_init, u_init), xs=None, length=400
)

mm_mpc_states = jnp.vstack((x0, mm_mpc_states))
plt.plot(mm_mpc_states[:, 0])
plt.plot(mm_mpc_states[:, 1])
plt.show()

plt.plot(mm_mpc_controls)
plt.show()
