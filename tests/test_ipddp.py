"""Interior-point DDP baseline."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from msilqr import DoubleIntegrator, InteriorPointDDP, IPDDPOptions, OptimalControlProblem, Status, ipddp_mpc
from msilqr.systems import POLYTOPE_G, POLYTOPE_H

G, H = jnp.array(POLYTOPE_G), jnp.array(POLYTOPE_H)


def nonlinear_problem():
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.1]])
    return OptimalControlProblem(
        lambda x, u: A @ x + B @ u + 0.05 * jnp.array([jnp.sin(x[1]), x[0] * u[0]]),
        lambda x, u: 0.5 * x @ jnp.diag(jnp.array([2.0, 1.0])) @ x + 0.5 * 0.3 * u @ u + 0.1 * x[0] * u[0],
        lambda x: 0.5 * x @ jnp.diag(jnp.array([5.0, 3.0])) @ x,
    )


def state_constraints(x, u):
    return jnp.array([u[0] - 1.0 + 0.3 * x[0] + 0.2 * x[1] ** 2, -u[0] - 1.0 + 0.5 * x[1], 0.4 * x[0] - x[1] + 0.1 * u[0] ** 2 - 2.0])


def test_backward_pass_gains_solve_the_perturbed_kkt_system():
    """Gains of one stage against a direct solve of the linearized perturbed KKT system
    (state-dependent constraints, so the Sigma = S^{-1} Z feedback term matters)."""
    problem = nonlinear_problem()
    ip = InteriorPointDDP(problem, state_constraints)
    x0, u = jnp.array([0.3, -0.2]), jnp.array([[0.1]])
    states = problem.rollout(x0, u)
    s, z = jnp.array([[0.4, 0.7, 1.3]]), jnp.array([[0.2, 0.05, 0.6]])
    mu, reg = 0.1, 1e-2
    k_u, k_z, k_s, K_u, K_z, K_s, *_ = ip.backward_pass(states, u, s, z, mu, reg)

    x, uu, ss, zz = states[0], u[0], s[0], z[0]
    lag = lambda x_, u_: problem.stage_cost(x_, u_) + zz @ state_constraints(x_, u_)  # noqa: E731
    Vx, Vxx = jax.grad(problem.final_cost)(states[1]), jax.hessian(problem.final_cost)(states[1])
    fx, fu = jax.jacrev(problem.dynamics, 0)(x, uu), jax.jacrev(problem.dynamics, 1)(x, uu)
    Qu = jax.grad(lag, 1)(x, uu) + fu.T @ Vx
    Quu = jax.hessian(lag, 1)(x, uu) + fu.T @ Vxx @ fu + reg * jnp.eye(1)
    Qux = jax.jacrev(jax.grad(lag, 1), 0)(x, uu) + fu.T @ Vxx @ fx
    cx, cu = jax.jacrev(state_constraints, 0)(x, uu), jax.jacrev(state_constraints, 1)(x, uu)
    rp, rd = state_constraints(x, uu) + ss, zz * ss - mu
    m, nc, n = 1, 3, 2
    kkt = jnp.block(
        [
            [Quu, jnp.zeros((m, nc)), cu.T],
            [cu, jnp.eye(nc), jnp.zeros((nc, nc))],
            [jnp.zeros((nc, m)), jnp.diag(zz), jnp.diag(ss)],
        ]
    )
    ff = jnp.linalg.solve(kkt, -jnp.concatenate([Qu, rp, rd]))
    fb = jnp.linalg.solve(kkt, -jnp.concatenate([Qux, cx, jnp.zeros((nc, n))]))
    np.testing.assert_allclose(k_u[0], ff[:m], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(K_u[0], fb[:m], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(k_s[0], ff[m : m + nc], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(K_s[0], fb[m : m + nc], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(k_z[0], ff[m + nc :], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(K_z[0], fb[m + nc :], rtol=1e-10, atol=1e-14)

    # the original formula (S^{-1} instead of Sigma in the c_x term) gives a different gain
    S_inv, Sigma = jnp.diag(1 / ss), jnp.diag(zz / ss)
    K_original = jnp.linalg.solve(Quu + cu.T @ Sigma @ cu, -cu.T @ S_inv @ cx - Qux)
    assert not jnp.allclose(K_original, fb[:m], rtol=1e-3)


def polytope_solver(**options):
    return InteriorPointDDP(DoubleIntegrator().problem(), lambda x, u: G @ u - H, IPDDPOptions(**options))


def polytope_inputs():
    return jnp.array([0.2, -0.1, -0.1, 0.1]), 0.1 * jnp.ones((15, 2)), 0.1 * jnp.ones((15, 8)), 0.01 * jnp.ones((15, 8))


def test_status_cap_and_barrier_tol():
    x0, u0, s0, z0 = polytope_inputs()
    sol = polytope_solver().solve(x0, u0, s0, z0)
    assert sol.status == Status.CONVERGED and sol.barrier > 1e-6  # the original stops before mu reaches tol
    strict = polytope_solver(barrier_tol=1e-6).solve(x0, u0, s0, z0)
    assert strict.status == Status.CONVERGED and strict.barrier < 1e-6 and strict.iterations > sol.iterations
    capped = polytope_solver(max_iter=3).solve(x0, u0, s0, z0)
    assert int(capped.iterations) == 3 and capped.status == Status.MAX_ITER
    none = polytope_solver(max_iter=0).solve(x0, u0, s0, z0)
    assert int(none.iterations) == 0 and none.status == Status.MAX_ITER
    assert jnp.all(G @ strict.controls.T - H[:, None] <= 1e-6)


def test_barrier_init_and_nonpositive_costs():
    x0, u0, s0, z0 = polytope_inputs()
    system = DoubleIntegrator()
    shifted = OptimalControlProblem(system.problem().dynamics, lambda x, u: system.stage_cost(x, u) - 100.0, system.final_cost)
    reference = polytope_solver(barrier_tol=1e-8).solve(x0, u0, s0, z0)
    for options in (dict(barrier_init=1.0), dict()):  # explicit, and the fallback of the original rule
        ip = InteriorPointDDP(shifted, lambda x, u: G @ u - H, IPDDPOptions(barrier_tol=1e-8, **options))
        sol = ip.solve(x0, u0, s0, z0)
        assert sol.status == Status.CONVERGED
        # same solution up to the interior-point accuracy at this tolerance (with mu_0 < 0
        # the original rule stalled at controls ~0.01 instead of ~0.2-0.5)
        np.testing.assert_allclose(sol.controls, reference.controls, atol=1e-4)


def test_rejects_nonpositive_slacks_or_duals():
    x0, u0, s0, z0 = polytope_inputs()
    with pytest.raises(ValueError, match="strictly positive"):
        polytope_solver().solve(x0, u0, s0.at[0, 0].set(0.0), z0)
    with pytest.raises(ValueError, match="strictly positive"):
        polytope_solver().solve(x0, u0, s0, -z0)
    with pytest.raises(ValueError, match="strictly positive"):
        ipddp_mpc(polytope_solver(), x0, u0, s0, z0.at[3, 1].set(0.0), 2)
    with pytest.raises(ValueError, match="shape"):
        polytope_solver().solve(x0, u0, s0[:-1], z0)


@pytest.mark.parametrize("bad", [{"barrier_init": 0.0}, {"barrier_tol": -1e-6}, {"barrier_init_fallback": 0.0}, {"reg_init": -1.0}])
def test_options_are_validated(bad):
    with pytest.raises(ValueError):
        IPDDPOptions(**bad)
