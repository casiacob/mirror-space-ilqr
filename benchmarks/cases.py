"""The four experiments shipped with the original code, rebuilt on the new API.

Each case replays the exact inputs of one original script (``examples/*.py`` at commit
effb14c), including its JAX random-number calls, so the results can be compared
bitwise with ``benchmarks/legacy/golden``:

==================  =========================================  =========  ====  =====
case                original script                            x0         T     steps
==================  =========================================  =========  ====  =====
cartpole            cartpole.py                                1 (fixed)  100   400
cartpole-compare    cartpole-mm-v-ip.py (MS vs IP)              10 random  100   400
polytope            wiener-polytope.py                          1 (fixed)  15    300
polytope-compare    wiener-polytope-mm-v-ip.py (MS vs IP)       10 random  15    300
==================  =========================================  =========  ====  =====

Settings that differ from the paper's text are kept on purpose (they are what the
golden results were produced with) and are listed in the README: the cart-pole box map
uses half of Table I's entropy DGF (``scale=0.5``), and the double-integrator
comparison samples its initial states around 0 instead of the paper's ``x0``.
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
    u_init: jax.Array  # (T, nu), shared by all initial states
    steps: int
    ip_slacks: jax.Array | None = None  # (T, nc)
    ip_duals: jax.Array | None = None  # (T, nc)


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    inputs: Callable[[], Inputs]
    ms_solver: Callable[..., MirrorILQR]  # cart-pole cases accept box_scale=
    ip_solver: Callable[..., InteriorPointDDP] | None = None  # accepts options=IPDDPOptions(...)


# ------------------------------------------------------------------- cart-pole ---


def cartpole_ms_solver(box_scale: float = CARTPOLE_BOX_SCALE) -> MirrorILQR:
    return MirrorILQR(CartPole().problem(), BoxEntropy(*CARTPOLE_BOX, scale=box_scale))


def cartpole_box_constraints(state, control):
    control_ub, control_lb = CARTPOLE_BOX[1], CARTPOLE_BOX[0]
    c0 = control - control_ub
    c1 = -control + control_lb
    return jnp.hstack((c0, c1))


def cartpole_ip_solver(options: IPDDPOptions | None = None) -> InteriorPointDDP:
    return InteriorPointDDP(CartPole().problem(), cartpole_box_constraints, options or IPDDPOptions())


def cartpole_inputs() -> Inputs:
    key = jax.random.PRNGKey(12)
    x0 = jnp.array([-0.01, 0.01, -0.01, 0.01])
    u_init = jax.random.normal(key, shape=(100, 1)) * 0.1
    return Inputs(x0[None], u_init, steps=400)


def cartpole_compare_inputs() -> Inputs:
    horizon, n = 100, 10
    key = jax.random.PRNGKey(12)
    key, x0_key, u_key = jax.random.split(key, 3)
    x0_mean = jnp.array([-0.01, 0.01, -0.01, 0.01])
    x0_std = jnp.array([0.0001, 0.0001, 0.05, 0.05])
    x0 = x0_mean + x0_std * jax.random.normal(x0_key, shape=(n, 4))
    u_init = jax.random.normal(u_key, shape=(horizon, 1)) * 0.1
    return Inputs(x0, u_init, steps=400, ip_slacks=0.1 * jnp.ones((horizon, 2)), ip_duals=0.01 * jnp.ones((horizon, 2)))


# ------------------------------------------------------------ double integrator ---


def polytope_map() -> PolytopeLogBarrier:
    return PolytopeLogBarrier(G=jnp.array(POLYTOPE_G), h=jnp.array(POLYTOPE_H))


def polytope_ms_solver() -> MirrorILQR:
    return MirrorILQR(DoubleIntegrator().problem(), polytope_map())


def polytope_constraints(state, control):
    return jnp.array(POLYTOPE_G) @ control - jnp.array(POLYTOPE_H)


def polytope_ip_solver(options: IPDDPOptions | None = None) -> InteriorPointDDP:
    return InteriorPointDDP(DoubleIntegrator().problem(), polytope_constraints, options or IPDDPOptions())


def polytope_inputs() -> Inputs:
    key = jax.random.PRNGKey(12)
    x0 = jnp.array([0.2, -0.1, -0.1, 0.1])
    u_init = jax.random.normal(key, shape=(15, 2)) * 0.1
    return Inputs(x0[None], u_init, steps=300)


def polytope_compare_inputs() -> Inputs:
    horizon, n, nc = 15, 10, len(POLYTOPE_H)
    key = jax.random.PRNGKey(12)
    key, x0_key, u_key = jax.random.split(key, 3)
    x0_mean = jnp.array([0.0, 0.0, 0.0, 0.0])
    x0_std = jnp.array([0.1, 0.1, 0.005, 0.005])
    x0 = x0_mean + x0_std * jax.random.normal(x0_key, shape=(n, 4))
    u_init = jax.random.normal(u_key, shape=(horizon, 2)) * 0.1
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
