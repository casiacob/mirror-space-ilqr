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


def test_cold_start_restarts_every_solve_from_the_initial_guess():
    problem = DoubleIntegrator().problem()
    x0 = jnp.array([0.2, -0.1, -0.1, 0.1])
    u_init = 0.1 * jax.random.normal(jax.random.PRNGKey(12), (15, 2))
    slacks, duals = 0.1 * jnp.ones((15, 8)), 0.01 * jnp.ones((15, 8))
    ms = MirrorILQR(problem, PolytopeLogBarrier(G, H))
    ip = InteriorPointDDP(problem, lambda x, u: G @ u - H)
    runs = {
        "ms": [mirror_mpc(ms, x0, u_init, 10, warm_start=warm) for warm in (True, False)],
        "ip": [ipddp_mpc(ip, x0, u_init, slacks, duals, 10, warm_start=warm) for warm in (True, False)],
    }
    for warm, cold in runs.values():
        assert jnp.all(warm.status == 0) and jnp.all(cold.status == 0)
        np.testing.assert_array_equal(warm.states[1], cold.states[1])  # the first solve is the same
        assert int(cold.iterations.sum()) > int(warm.iterations.sum())
    # a cold step is the plain solve from the initial guess at that state
    state = runs["ms"][1].states[3]
    np.testing.assert_array_equal(runs["ms"][1].controls[3], ms.solve(state, u_init).controls[0])
    np.testing.assert_array_equal(runs["ip"][1].controls[3], ip.solve(runs["ip"][1].states[3], u_init, slacks, duals).controls[0])
    # cold is the default
    np.testing.assert_array_equal(mirror_mpc(ms, x0, u_init, 10).controls, runs["ms"][1].controls)
    np.testing.assert_array_equal(ipddp_mpc(ip, x0, u_init, slacks, duals, 10).controls, runs["ip"][1].controls)
