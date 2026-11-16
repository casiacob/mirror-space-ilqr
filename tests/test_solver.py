"""Mirror-space iLQR on small problems with known answers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from msilqr import BoxEntropy, MirrorILQR, OptimalControlProblem, SolverOptions, Status

A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
B = jnp.array([[0.005], [0.1]])
LOW, HIGH = -1.0, 1.0
T = 30


def lq_problem():
    def dynamics(x, u):
        return A @ x + B @ u

    def stage_cost(x, u):
        return 0.5 * x @ jnp.diag(jnp.array([10.0, 1.0])) @ x + 0.5 * 0.1 * u @ u

    def final_cost(x):
        return 0.5 * x @ jnp.diag(jnp.array([100.0, 10.0])) @ x

    return OptimalControlProblem(dynamics, stage_cost, final_cost)


def reference_box_qp(problem, x0, iters=20000):
    """Solve the condensed box-constrained QP to high accuracy (accelerated projected gradient)."""
    J = lambda U: problem.total_cost(problem.rollout(x0, U.reshape(T, 1)), U.reshape(T, 1))  # noqa: E731
    U0 = jnp.zeros(T)
    H = jax.hessian(J)(U0)
    step = 1.0 / jnp.linalg.eigvalsh(H).max()
    g = jax.grad(J)

    def body(i, carry):
        U, V, t = carry
        U_new = jnp.clip(V - step * g(V), LOW, HIGH)
        t_new = 0.5 * (1 + jnp.sqrt(1 + 4 * t**2))
        V_new = U_new + ((t - 1) / t_new) * (U_new - U)
        return U_new, V_new, t_new

    U, _, _ = jax.lax.fori_loop(0, iters, body, (U0, U0, 1.0))
    return U.reshape(T, 1), J(U)


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_box_lq_matches_reference_qp(scale):
    problem = lq_problem()
    x0 = jnp.array([3.0, 0.0])  # far enough that the bounds are active for a while
    u_ref, J_ref = reference_box_qp(problem, x0)
    assert (jnp.abs(u_ref) > 1 - 1e-9).sum() >= 3, "test problem should have active constraints"

    solver = MirrorILQR(problem, BoxEntropy(LOW, HIGH, scale=scale), SolverOptions(tol=1e-9))
    u0 = jnp.zeros((T, 1))
    sol = solver.solve(problem.rollout(x0, u0), u0)

    assert sol.status == Status.CONVERGED
    assert jnp.all(sol.controls >= LOW) and jnp.all(sol.controls <= HIGH)
    np.testing.assert_allclose(sol.cost, J_ref, rtol=1e-6)
    np.testing.assert_allclose(sol.controls, u_ref, atol=1e-3)
    np.testing.assert_allclose(sol.cost, problem.total_cost(sol.states, sol.controls), rtol=1e-12)


def test_iteration_cap_and_status():
    problem = lq_problem()
    solver = MirrorILQR(problem, BoxEntropy(LOW, HIGH), SolverOptions(tol=0.0, max_iter=4))
    x0 = jnp.array([3.0, 0.0])
    u0 = jnp.zeros((T, 1))
    sol = solver.solve(problem.rollout(x0, u0), u0)
    assert int(sol.iterations) == 4
    assert sol.status == Status.MAX_ITER


def test_proposition_5_mirror_gains_are_scaled_control_gains():
    """k^y = Psi k^u and K^y = Psi K^u with Qhat_uu = Q_uu + lambda Psi^2 (Prop. 5)."""
    problem = lq_problem()
    mmap = BoxEntropy(LOW, HIGH)
    solver = MirrorILQR(problem, mmap)
    x0 = jnp.array([1.0, -0.5])
    u = jnp.array([[0.4]])  # horizon 1: one stage
    states = problem.rollout(x0, u)
    lam = 0.3
    K_y, k_y, _, _, _ = solver.backward_pass(states, u, lam)

    Vx = jax.grad(problem.final_cost)(states[1])
    Vxx = jax.hessian(problem.final_cost)(states[1])
    x, uu = states[0], u[0]
    Qu = jax.grad(problem.stage_cost, 1)(x, uu) + B.T @ Vx
    Quu = jax.hessian(problem.stage_cost, 1)(x, uu) + B.T @ Vxx @ B
    Qux = B.T @ Vxx @ A
    Psi = mmap.hessian(uu)
    Quu_hat = Quu + lam * Psi @ Psi
    k_u = -jnp.linalg.solve(Quu_hat, Qu)
    K_u = -jnp.linalg.solve(Quu_hat, Qux)

    np.testing.assert_allclose(k_y[0], Psi @ k_u, rtol=1e-10)
    np.testing.assert_allclose(K_y[0], Psi @ K_u, rtol=1e-10)
