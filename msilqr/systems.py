"""The two systems of the paper's experiments (Sec. V)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from msilqr.problem import OptimalControlProblem, euler

__all__ = [
    "wrap_angle",
    "CartPole",
    "DoubleIntegrator",
    "POLYTOPE_G",
    "POLYTOPE_H",
]


def wrap_angle(x):
    """Map an angle to ``[0, 2 pi)``.

    Used by the cart-pole cost so that both ``+pi`` and ``-pi`` count as upright. The
    resulting cost has a kink at ``theta = 0`` (the hanging position), i.e. it is not
    twice differentiable there.
    """
    return x % (2.0 * jnp.pi)


# --------------------------------------------------------------------------------------
# Cart-pole (Sec. V-b), swing-up with box-constrained force
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CartPole:
    """Cart-pole (Spong [39]; https://underactuated.mit.edu/acrobot.html#cart_pole).

    State ``x = [p, theta, v, omega]`` (``theta = 0`` hanging, ``pi`` upright), control
    ``u = [force]``. Explicit Euler with step ``dt``.
    """

    gravity: float = 9.81
    pole_length: float = 0.5
    cart_mass: float = 10.0
    pole_mass: float = 1.0
    dt: float = 0.01
    goal: tuple = (0.0, jnp.pi, 0.0, 0.0)
    state_weights: tuple = (1e1, 1e1, 1e-1, 1e-1)
    final_weights: tuple = (1e1, 1e1, 1e-1, 1e-1)
    control_weights: tuple = (1e-3,)

    def ode(self, state, action):
        gravity, pole_length = self.gravity, self.pole_length
        cart_mass, pole_mass = self.cart_mass, self.pole_mass
        total_mass = cart_mass + pole_mass

        cart_position, pole_position, cart_velocity, pole_velocity = state

        sth = jnp.sin(pole_position)
        cth = jnp.cos(pole_position)

        cart_acceleration = (action + pole_mass * sth * (pole_length * pole_velocity**2 + gravity * cth)) / (cart_mass + pole_mass * sth**2)

        pole_acceleration = (-action * cth - pole_mass * pole_length * pole_velocity**2 * cth * sth - total_mass * gravity * sth) / (
            pole_length * cart_mass + pole_length * pole_mass * sth**2
        )

        return jnp.hstack((cart_velocity, pole_velocity, cart_acceleration, pole_acceleration))

    def _wrapped(self, state):
        return jnp.hstack((state[0], wrap_angle(state[1]), state[2], state[3]))

    def final_cost(self, state):
        goal_state = jnp.array(self.goal)
        final_state_cost = jnp.diag(jnp.array(self.final_weights))
        err = self._wrapped(state) - goal_state
        return 0.5 * err.T @ final_state_cost @ err

    def stage_cost(self, state, action):
        goal_state = jnp.array(self.goal)
        state_cost = jnp.diag(jnp.array(self.state_weights))
        action_cost = jnp.diag(jnp.array(self.control_weights))
        err = self._wrapped(state) - goal_state
        c = 0.5 * err.T @ state_cost @ err
        c += 0.5 * action.T @ action_cost @ action
        return c

    def problem(self) -> OptimalControlProblem:
        return OptimalControlProblem(euler(self.ode, self.dt), self.stage_cost, self.final_cost)


# --------------------------------------------------------------------------------------
# Double integrator with polytopic control constraints (Sec. V-a)
# --------------------------------------------------------------------------------------

#: ``|u1| <= 0.5, |u2| <= 0.5, |u1 + u2| <= 0.7, |u1 - u2| <= 0.7`` as ``G u <= h``.
POLYTOPE_G = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0))
POLYTOPE_H = (0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7)


@dataclass(frozen=True, eq=False)
class DoubleIntegrator:
    """Double integrator in two dimensions ([38, Ex. 4.2], the Wiener velocity model).

    State ``x = [p1, p2, v1, v2]``, control ``u = [a1, a2]``. Explicit Euler with step
    ``dt``.
    """

    dt: float = 0.02
    goal: tuple = (1.0, 1.0, 0.0, 0.0)
    state_weights: tuple = (1e1, 1e1, 1e0, 1e0)
    final_weights: tuple = (1e1, 1e1, 1e0, 1e0)
    control_weights: tuple = (5e-2, 5e-2)

    def ode(self, state, control):
        F = jnp.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        L = jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        return F @ state + L @ control

    def final_cost(self, state):
        goal_state = jnp.array(self.goal)
        final_state_cost = jnp.diag(jnp.array(self.final_weights))
        err = state - goal_state
        return 0.5 * err.T @ final_state_cost @ err

    def stage_cost(self, state, control):
        goal_state = jnp.array(self.goal)
        state_cost = jnp.diag(jnp.array(self.state_weights))
        control_cost = jnp.diag(jnp.array(self.control_weights))
        err = state - goal_state
        c = 0.5 * err.T @ state_cost @ err
        c += 0.5 * control.T @ control_cost @ control
        return c

    def problem(self) -> OptimalControlProblem:
        return OptimalControlProblem(euler(self.ode, self.dt), self.stage_cost, self.final_cost)
