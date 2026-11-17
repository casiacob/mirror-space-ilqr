"""Benchmark cases: the experiments of the paper.

* ``polytope``: double integrator in a polytope (Fig. 2);
* ``cartpole``: cart-pole swing-up (Fig. 3);
* ``polytope-compare``, ``cartpole-compare``: comparison with interior-point DDP (Fig. 4).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from msilqr import (
    BoxEntropy,
    CartPole,
    DoubleIntegrator,
    InteriorPointDDP,
    IPDDPOptions,
    MirrorILQR,
    PolytopeLogBarrier,
)
from msilqr.systems import POLYTOPE_G, POLYTOPE_H

CARTPOLE_BOX = (-50.0, 50.0)
CARTPOLE_BOX_SCALE = 0.5  # the original scripts' map; the paper's Table I map is scale=1


@dataclass(frozen=True)
class Inputs:
    x0: jax.Array  # (N, nx)
    u_init: jax.Array  # (N, T, nu), one per initial state
    steps: int
    ip_slacks: jax.Array | None = None  # (T, nc)
    ip_duals: jax.Array | None = None  # (T, nc)


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    inputs: Callable[[], Inputs]
    ms_solver: Callable[..., MirrorILQR]  # cart-pole cases accept box_scale=
    ip_solver: Callable[..., InteriorPointDDP] | None = None  # options default to IPDDPOptions.paper()


def sample_runs(n, x0_mean, x0_std, horizon, nu):
    """``n`` runs from ``PRNGKey(12)``: each run's key draws its initial state
    ``x0_mean + x0_std N(0, I)`` and its initial controls ``0.1 N(0, I)``."""

    def draw(key):
        x0_key, u_key = jax.random.split(key)
        x0 = x0_mean + x0_std * jax.random.normal(x0_key, shape=x0_mean.shape)
        return x0, 0.1 * jax.random.normal(u_key, shape=(horizon, nu))

    return jax.vmap(draw)(jax.random.split(jax.random.PRNGKey(12), n))


# ------------------------------------------------------------------- cart-pole ---


def cartpole_ms_solver(box_scale: float = CARTPOLE_BOX_SCALE) -> MirrorILQR:
    return MirrorILQR(CartPole().problem(), BoxEntropy(*CARTPOLE_BOX, scale=box_scale))


def cartpole_box_constraints(state, control):
    control_ub, control_lb = CARTPOLE_BOX[1], CARTPOLE_BOX[0]
    c0 = control - control_ub
    c1 = -control + control_lb
    return jnp.hstack((c0, c1))


def cartpole_ip_solver(options: IPDDPOptions | None = None) -> InteriorPointDDP:
    return InteriorPointDDP(CartPole().problem(), cartpole_box_constraints, options or IPDDPOptions.paper())


def cartpole_inputs() -> Inputs:
    key = jax.random.PRNGKey(12)
    x0 = jnp.array([-0.01, 0.01, -0.01, 0.01])
    u_init = jax.random.normal(key, shape=(100, 1)) * 0.1
    return Inputs(x0[None], u_init[None], steps=400)


def cartpole_compare_inputs() -> Inputs:
    horizon = 100
    x0, u_init = sample_runs(10, jnp.array([-0.01, 0.01, -0.01, 0.01]), jnp.array([0.0001, 0.0001, 0.05, 0.05]), horizon, 1)
    return Inputs(x0, u_init, steps=400, ip_slacks=0.1 * jnp.ones((horizon, 2)), ip_duals=0.01 * jnp.ones((horizon, 2)))


# ------------------------------------------------------------ double integrator ---


def polytope_map() -> PolytopeLogBarrier:
    return PolytopeLogBarrier(G=jnp.array(POLYTOPE_G), h=jnp.array(POLYTOPE_H))


def polytope_ms_solver() -> MirrorILQR:
    return MirrorILQR(DoubleIntegrator().problem(), polytope_map())


def polytope_constraints(state, control):
    return jnp.array(POLYTOPE_G) @ control - jnp.array(POLYTOPE_H)


def polytope_ip_solver(options: IPDDPOptions | None = None) -> InteriorPointDDP:
    return InteriorPointDDP(DoubleIntegrator().problem(), polytope_constraints, options or IPDDPOptions.paper())


def polytope_inputs() -> Inputs:
    key = jax.random.PRNGKey(12)
    x0 = jnp.array([0.2, -0.1, -0.1, 0.1])
    u_init = jax.random.normal(key, shape=(15, 2)) * 0.1
    return Inputs(x0[None], u_init[None], steps=300)


def polytope_compare_inputs() -> Inputs:
    horizon, nc = 15, len(POLYTOPE_H)
    x0, u_init = sample_runs(10, jnp.zeros(4), jnp.array([0.1, 0.1, 0.005, 0.005]), horizon, 2)
    return Inputs(x0, u_init, steps=300, ip_slacks=0.1 * jnp.ones((horizon, nc)), ip_duals=0.01 * jnp.ones((horizon, nc)))


CASES = {
    c.name: c
    for c in (
        Case("cartpole", "cart-pole swing-up, box |u| <= 50 (Fig. 3)", cartpole_inputs, cartpole_ms_solver),
        Case(
            "cartpole-compare",
            "cart-pole, MS vs IP over 10 initial states (Fig. 4 right)",
            cartpole_compare_inputs,
            cartpole_ms_solver,
            cartpole_ip_solver,
        ),
        Case("polytope", "double integrator, 8-sided polytope (Fig. 2)", polytope_inputs, polytope_ms_solver),
        Case(
            "polytope-compare",
            "double integrator, MS vs IP over 10 initial states (Fig. 4 left)",
            polytope_compare_inputs,
            polytope_ms_solver,
            polytope_ip_solver,
        ),
    )
}
