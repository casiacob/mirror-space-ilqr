"""Closed loops record applied controls and follow the model."""

import jax
import jax.numpy as jnp
import numpy as np

from msilqr import DoubleIntegrator, InteriorPointDDP, MirrorILQR, PolytopeLogBarrier, ipddp_mpc, mirror_mpc
from msilqr.systems import POLYTOPE_G, POLYTOPE_H

G, H = jnp.array(POLYTOPE_G), jnp.array(POLYTOPE_H)


def test_closed_loops_follow_dynamics_and_constraints():
    system = DoubleIntegrator()
    problem = system.problem()
    x0 = jnp.array([0.2, -0.1, -0.1, 0.1])
    u_init = 0.1 * jax.random.normal(jax.random.PRNGKey(12), (15, 2))
    steps = 25

    ms = mirror_mpc(MirrorILQR(problem, PolytopeLogBarrier(G, H)), x0, u_init, steps)
    ip = ipddp_mpc(
        InteriorPointDDP(problem, lambda x, u: G @ u - H),
        x0,
        u_init,
        0.1 * jnp.ones((15, 8)),
        0.01 * jnp.ones((15, 8)),
        steps,
    )
    for res in (ms, ip):
        assert res.states.shape == (steps + 1, 4) and res.controls.shape == (steps, 2)
        np.testing.assert_array_equal(res.states[0], x0)
        predicted = jax.vmap(problem.dynamics)(res.states[:-1], res.controls)
        np.testing.assert_allclose(res.states[1:], predicted, rtol=0, atol=1e-14)
        assert jnp.all(res.status == 0)
    # mirror-space iterates are strictly feasible; the infeasible IPDDP only approximately
    assert jnp.all(ms.controls @ G.T - H < 0)
    assert jnp.all(ip.controls @ G.T - H <= 1e-6)
