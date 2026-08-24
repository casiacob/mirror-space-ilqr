import jax.numpy as jnp
import jax


def rollout_dynamics(dynamics, control_sequence, initial_state):
    def body(xt, ut):
        return dynamics(xt, ut), dynamics(xt, ut)

    _, states = jax.lax.scan(body, initial_state, control_sequence)
    states = jnp.vstack((initial_state, states))
    return states


def discretize_dynamics(ode, simulation_step):
    def dynamics(state, control):
        return state + simulation_step * ode(state, control)

    return dynamics


def wrap_angle(x: float) -> float:
    return x % (2.0 * jnp.pi)
