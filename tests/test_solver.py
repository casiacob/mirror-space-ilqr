"""Mirror-space iLQR on small problems with known answers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from msilqr import (
    BoxEntropy,
    BoxLogBarrier,
    EuclideanBall,
    MirrorILQR,
    OptimalControlProblem,
    Positivity,
    SolverOptions,
    Status,
    mirror_mpc,
)

T = 30


def lq_1d():
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.005], [0.1]])
    return OptimalControlProblem(
        lambda x, u: A @ x + B @ u,
        lambda x, u: 0.5 * x @ jnp.diag(jnp.array([10.0, 1.0])) @ x + 0.5 * 0.1 * u @ u,
        lambda x: 0.5 * x @ jnp.diag(jnp.array([100.0, 10.0])) @ x,
    )


def lq_2d():
    """Planar double integrator with a 2-D control (for the Euclidean ball)."""
    A = jnp.eye(4) + 0.1 * jnp.eye(4, k=2)
    B = jnp.vstack((0.005 * jnp.eye(2), 0.1 * jnp.eye(2)))
    return OptimalControlProblem(
        lambda x, u: A @ x + B @ u,
        lambda x, u: 0.5 * x @ jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0])) @ x + 0.5 * 0.1 * u @ u,
        lambda x: 0.5 * x @ jnp.diag(jnp.array([100.0, 100.0, 10.0, 10.0])) @ x,
    )


def reference_qp(problem, x0, m, project, iters=30000):
    """Constrained optimum by accelerated projected gradient on the condensed problem."""
    J = lambda U: problem.total_cost(problem.rollout(x0, U.reshape(T, m)), U.reshape(T, m))  # noqa: E731
    U0 = jnp.zeros(T * m)
    step = 1.0 / jnp.linalg.eigvalsh(jax.hessian(J)(U0)).max()
    g = jax.grad(J)
    proj = lambda U: jax.vmap(project)(U.reshape(T, m)).reshape(-1)  # noqa: E731

    def body(i, carry):
        U, V, t = carry
        U_new = proj(V - step * g(V))
        t_new = 0.5 * (1 + jnp.sqrt(1 + 4 * t**2))
        return U_new, U_new + ((t - 1) / t_new) * (U_new - U), t_new

    U, _, _ = jax.lax.fori_loop(0, iters, body, (U0, U0, 1.0))
    return U.reshape(T, m), J(U)


def unit_ball_projection(u):
    n = jnp.linalg.norm(u)
    return jnp.where(n > 1.0, u / n, u)


def box(lo, hi):
    return lambda u: jnp.clip(u, lo, hi)


CASES = [
    # name, problem, mirror map, x0, initial control, projection, cost rtol, converges
    ("box-entropy", lq_1d, BoxEntropy(-1.0, 1.0), [3.0, 0.0], 0.0, box(-1.0, 1.0), 1e-6, True),
    ("box-entropy-half", lq_1d, BoxEntropy(-1.0, 1.0, scale=0.5), [3.0, 0.0], 0.0, box(-1.0, 1.0), 1e-6, True),
    ("box-entropy-asym-low", lq_1d, BoxEntropy(-1.0, 0.1), [-3.0, 0.0], -0.45, box(-1.0, 0.1), 1e-6, True),
    ("box-entropy-asym-high", lq_1d, BoxEntropy(0.1, 0.7), [3.0, 0.0], 0.4, box(0.1, 0.7), 1e-6, True),
    ("box-log-barrier", lq_1d, BoxLogBarrier(-1.0, 1.0), [3.0, 0.0], 0.0, box(-1.0, 1.0), 1e-6, True),
    ("positivity", lq_1d, Positivity(), [3.0, 0.0], 0.5, lambda u: jnp.maximum(u, 0.0), 1e-6, True),
    ("euclidean-ball-axis", lq_2d, EuclideanBall(), [2.0, 0.0, 0.0, 0.0], 0.0, unit_ball_projection, 1e-6, True),
    # Known limitation (README): when the optimal controls slide along the curved boundary
    # of the ball, the lambda*I damping in mirror coordinates makes progress roughly like
    # 1/k (error 5.5e-4, 9e-5, 1.1e-5 after 1k, 5k, 20k iterations). Feasible and close,
    # but not converged within the default cap.
    ("euclidean-ball-oblique", lq_2d, EuclideanBall(), [3.0, -2.0, 0.0, 0.0], 0.0, unit_ball_projection, 1e-4, False),
]


@pytest.mark.parametrize("name,make,mmap,x0,u_init,project,rtol,converges", CASES, ids=[c[0] for c in CASES])
def test_matches_reference_constrained_optimum(name, make, mmap, x0, u_init, project, rtol, converges):
    problem = make()
    x0 = jnp.array(x0)
    m = 2 if make is lq_2d else 1
    u_ref, J_ref = reference_qp(problem, x0, m, project)
    u_free, _ = reference_qp(problem, x0, m, lambda u: u)
    assert float(jnp.abs(u_ref - u_free).max()) > 1e-2, "the constraint should be active"

    solver = MirrorILQR(problem, mmap, SolverOptions(tol=1e-9))
    sol = solver.solve(x0, jnp.full((T, m), u_init))

    assert sol.status == (Status.CONVERGED if converges else Status.MAX_ITER)
    np.testing.assert_array_equal(jax.vmap(project)(sol.controls), sol.controls)  # feasible (closed set)
    np.testing.assert_allclose(sol.cost, J_ref, rtol=rtol)
    assert sol.cost >= J_ref * (1 - 1e-12)  # never better than the true optimum
    np.testing.assert_allclose(sol.cost, problem.total_cost(sol.states, sol.controls), rtol=1e-12)
    np.testing.assert_array_equal(sol.states, problem.rollout(x0, sol.controls))


def test_iteration_cap_and_status():
    problem = lq_1d()
    x0, u0 = jnp.array([3.0, 0.0]), jnp.zeros((T, 1))
    sol = MirrorILQR(problem, BoxEntropy(-1.0, 1.0), SolverOptions(tol=0.0, max_iter=4)).solve(x0, u0)
    assert int(sol.iterations) == 4 and sol.status == Status.MAX_ITER
    sol = MirrorILQR(problem, BoxEntropy(-1.0, 1.0), SolverOptions(max_iter=0)).solve(x0, u0)
    assert int(sol.iterations) == 0 and sol.status == Status.MAX_ITER


@pytest.mark.parametrize(
    "mmap,u_bad",
    [(Positivity(), 0.0), (BoxEntropy(0.0, 1.0), 0.0), (BoxEntropy(0.0, 1.0), 1.0), (BoxEntropy(-1.0, 1.0), 1.5)],
)
def test_rejects_initial_guess_outside_interior(mmap, u_bad):
    """At a boundary point Psi^{-1} = 0, so the solver would 'converge' at once."""
    problem = lq_1d()
    u0 = jnp.full((T, 1), u_bad)
    with pytest.raises(ValueError, match="strictly inside"):
        MirrorILQR(problem, mmap).solve(jnp.array([-3.0, 0.0]), u0)
    with pytest.raises(ValueError, match="strictly inside"):
        mirror_mpc(MirrorILQR(problem, mmap), jnp.array([-3.0, 0.0]), u0, 5)


def test_float32_inputs_are_promoted():
    problem = lq_1d()
    solver = MirrorILQR(problem, BoxEntropy(-1.0, 1.0))
    sol32 = solver.solve(jnp.array([3.0, 0.0], dtype=jnp.float32), jnp.zeros((T, 1), dtype=jnp.float32))
    sol64 = solver.solve(jnp.array([3.0, 0.0]), jnp.zeros((T, 1)))
    assert sol32.controls.dtype == jnp.float64
    np.testing.assert_array_equal(sol32.controls, sol64.controls)


def test_requires_x64():
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="double precision"):
            MirrorILQR(lq_1d(), BoxEntropy(-1.0, 1.0)).solve(jnp.zeros(2), jnp.zeros((T, 1)))
    finally:
        jax.config.update("jax_enable_x64", True)


def test_proposition_5_on_a_full_horizon():
    """k^y = Psi k^u, K^y = Psi K^u at every stage, where (k^u, K^u) come from control-space
    iLQR with Qhat_uu = Q_uu + lambda Psi^2 and the value recursion (13) in control space."""
    problem, mmap, lam, H = lq_1d(), BoxEntropy(-1.0, 1.0), 0.3, 6
    x0 = jnp.array([1.0, -0.5])
    u = 0.6 * jnp.sin(jnp.arange(H, dtype=float))[:, None]  # interior, non-trivial
    states = problem.rollout(x0, u)
    K_y, k_y, _, _, _ = MirrorILQR(problem, mmap).backward_pass(states, u, lam)

    A, B = jax.jacrev(problem.dynamics, 0)(x0, u[0]), jax.jacrev(problem.dynamics, 1)(x0, u[0])
    Vx, Vxx = jax.grad(problem.final_cost)(states[-1]), jax.hessian(problem.final_cost)(states[-1])
    for t in reversed(range(H)):
        x, ut = states[t], u[t]
        cx, cu = jax.grad(problem.stage_cost, 0)(x, ut), jax.grad(problem.stage_cost, 1)(x, ut)
        cxx, cuu = jax.hessian(problem.stage_cost, 0)(x, ut), jax.hessian(problem.stage_cost, 1)(x, ut)
        Qx, Qu = cx + A.T @ Vx, cu + B.T @ Vx
        Qxx, Quu, Qux = cxx + A.T @ Vxx @ A, cuu + B.T @ Vxx @ B, B.T @ Vxx @ A
        Psi = mmap.hessian(ut)
        Quu_hat = Quu + lam * Psi @ Psi
        k_u, K_u = -jnp.linalg.solve(Quu_hat, Qu), -jnp.linalg.solve(Quu_hat, Qux)
        np.testing.assert_allclose(k_y[t], Psi @ k_u, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(K_y[t], Psi @ K_u, rtol=1e-9, atol=1e-12)
        Vx = Qx + Qux.T @ k_u + K_u.T @ Qu + K_u.T @ Quu_hat @ k_u
        Vxx = Qxx + K_u.T @ Quu_hat @ K_u + K_u.T @ Qux + Qux.T @ K_u
        Vxx = 0.5 * (Vxx + Vxx.T)
