import jax
import jax.numpy as jnp

from msilqr.ipddp import infeasible_ipddp
from msilqr.utils import wrap_angle, rollout_dynamics, discretize_dynamics
from msilqr.objects import DualOCP, IPOCP
from jax import config
from msilqr.ilqr import solve_dual_ilqr
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

num_initial_states = 10

x0_mean = jnp.array([-0.01, 0.01, -0.01, 0.01])
x0_std = jnp.array([0.0001, 0.0001, 0.05, 0.05])

key, x0_key, u_key = jax.random.split(key, 3)

x0_samples = x0_mean + x0_std * jax.random.normal(x0_key, shape=(num_initial_states, 4))

u_init = jax.random.normal(u_key, shape=(horizon, 1)) * 0.1


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


def constraints(state: jnp.ndarray, control: jnp.ndarray):
    control_ub = 50.0
    control_lb = -50.0
    c0 = control - control_ub
    c1 = -control + control_lb
    return jnp.hstack((c0, c1))


z = 0.01 * jnp.ones((horizon, 2))
s = 0.1 * jnp.ones((horizon, 2))

cartpole_problem = IPOCP(discrete_cartpole, transient_cost, final_cost, constraints)


def ip_mpc_body(carry, inp):
    prev_state, prev_control = carry
    states, control, _, _, iterations = infeasible_ipddp(
        prev_state, prev_control, s, z, cartpole_problem
    )
    next_state = discrete_cartpole(prev_state, control[0])
    control = jnp.vstack((control[1:], control[-1]))
    return (next_state, control), (next_state, control[0], iterations)


all_mm_mpc_iterations = []
all_ip_mpc_iterations = []

all_mm_mpc_states = []
all_ip_mpc_states = []

all_mm_mpc_controls = []
all_ip_mpc_controls = []

for i in range(num_initial_states):
    x0 = x0_samples[i]
    x_init = rollout_dynamics(discrete_cartpole, u_init, x0)

    _, (
        mm_mpc_states,
        mm_mpc_controls,
        mpc_traj_costs,
        mm_mpc_iterations,
    ) = jax.lax.scan(mm_mpc_body, (x_init, u_init), xs=None, length=400)

    _, (
        ip_mpc_states,
        ip_mpc_controls,
        ip_mpc_iterations,
    ) = jax.lax.scan(ip_mpc_body, (x0, u_init), None, length=400)

    all_mm_mpc_iterations.append(mm_mpc_iterations)
    all_ip_mpc_iterations.append(ip_mpc_iterations)

    all_mm_mpc_states.append(mm_mpc_states)
    all_ip_mpc_states.append(ip_mpc_states)

    all_mm_mpc_controls.append(mm_mpc_controls)
    all_ip_mpc_controls.append(ip_mpc_controls)

all_mm_mpc_iterations = jnp.stack(all_mm_mpc_iterations)
all_ip_mpc_iterations = jnp.stack(all_ip_mpc_iterations)


window = 10


def moving_average(x):
    cumsum = jnp.cumsum(x)
    cumsum_padded = jnp.concatenate((jnp.array([0.0]), cumsum))

    starts = jnp.maximum(jnp.arange(x.shape[0]) - window + 1, 0)
    ends = jnp.arange(x.shape[0]) + 1

    sums = cumsum_padded[ends] - cumsum_padded[starts]
    counts = ends - starts

    return sums / counts


all_mm_mpc_iterations_ma = jax.vmap(moving_average)(all_mm_mpc_iterations)
all_ip_mpc_iterations_ma = jax.vmap(moving_average)(all_ip_mpc_iterations)

mean_mm_mpc_iterations_ma = jnp.mean(all_mm_mpc_iterations_ma, axis=0)
mean_ip_mpc_iterations_ma = jnp.mean(all_ip_mpc_iterations_ma, axis=0)

time_ma = jnp.arange(all_mm_mpc_iterations.shape[1])

plt.plot(time_ma, mean_mm_mpc_iterations_ma, label="MM iterations moving average")
plt.plot(time_ma, mean_ip_mpc_iterations_ma, label="IP iterations moving average")
plt.legend()
plt.show()
