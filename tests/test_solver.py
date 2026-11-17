"""Mirror-space iLQR on small problems with known answers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from msilqr import (
    BoxEntropy,
    BoxLogBarrier,
    DoubleIntegrator,
    EuclideanBall,
    MirrorILQR,
    OptimalControlProblem,
    PolytopeLogBarrier,
    Positivity,
    SolverOptions,
    Status,
    mirror_mpc,
)
from msilqr.systems import POLYTOPE_G, POLYTOPE_H

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
    ("box-entropy-asym-low", lq_1d, BoxEntropy(-1.0, 0.1), [-3.0, 0.0], -0.45, box(-1.0, 0.1), 1e-6, True),
    ("box-entropy-asym-high", lq_1d, BoxEntropy(0.1, 0.7), [3.0, 0.0], 0.4, box(0.1, 0.7), 1e-6, True),
    ("box-log-barrier", lq_1d, BoxLogBarrier(-1.0, 1.0), [3.0, 0.0], 0.0, box(-1.0, 1.0), 1e-6, True),
    ("positivity", lq_1d, Positivity(), [3.0, 0.0], 0.5, lambda u: jnp.maximum(u, 0.0), 1e-6, True),
    ("euclidean-ball-axis", lq_2d, EuclideanBall(), [2.0, 0.0, 0.0, 0.0], 0.0, unit_ball_projection, 1e-6, True),
    # Known limitation: when the optimal controls slide along
    # the curved boundary of the ball, the lambda*I damping in mirror coordinates makes
    # progress roughly like 1/k (error 3.7e-4, 9.0e-5, 1.3e-5 after 1k, 5k, 20k iterations).
    # Feasible and close, but not converged within the default cap.
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

    solver = MirrorILQR(problem, mmap, SolverOptions(tol=1e-6))
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


def test_validate_flag_and_eager_mpc_with_controls_on_the_bound():
    """With x0 = [-30, 0] every planned control rounds onto the bound u = 1. Such shifted
    plans are legitimate MPC warm starts, so the loop must not reject them, also when it
    runs eagerly (where the controls are concrete and could be checked)."""
    problem, x0 = lq_1d(), jnp.array([-30.0, 0.0])
    solver = MirrorILQR(problem, BoxEntropy(-1.0, 1.0))
    plan = solver.solve(x0, jnp.zeros((8, 1))).controls
    assert bool(jnp.all(plan == 1.0))
    with pytest.raises(ValueError, match="strictly inside"):
        solver.solve(x0, plan)
    assert solver.solve(x0, plan, validate=False).status == Status.CONVERGED

    jitted = mirror_mpc(solver, x0, jnp.zeros((8, 1)), 3, warm_start=True)
    with jax.disable_jit():
        eager = mirror_mpc(MirrorILQR(problem, BoxEntropy(-1.0, 1.0)), x0, jnp.zeros((8, 1)), 3, warm_start=True)
    assert jnp.all(eager.status == 0)
    np.testing.assert_allclose(eager.states, jitted.states, rtol=1e-12)


def test_rejects_bad_shapes():
    solver = MirrorILQR(lq_1d(), BoxEntropy(-1.0, 1.0))
    with pytest.raises(ValueError, match="x0"):
        solver.solve(jnp.zeros((T + 1, 2)), jnp.zeros((T, 1)))  # old call style: states instead of x0
    with pytest.raises(ValueError, match="controls"):
        solver.solve(jnp.zeros(2), jnp.zeros(T))


@pytest.mark.parametrize(
    "bad",
    [
        {"tol": -1.0},
        {"tol": float("nan")},
        {"max_iter": -1},
        {"max_iter": 10.5},
        {"reg_init": 0.0},
        {"reg_factor_init": 1.0},
        {"reg_min": 1.0, "reg_max": 0.1},
    ],
)
def test_options_are_validated(bad):
    with pytest.raises(ValueError):
        SolverOptions(**bad)


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
    K_y, k_y, *_ = MirrorILQR(problem, mmap, SolverOptions(one_sided=False)).backward_pass(states, u, lam)  # the paper's map

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


# ------------------------------------------------------------ natural stopping test ---

NATURAL = SolverOptions(tol=1e-6)


def test_natural_residual_matches_autodiff():
    """The backward pass gets ``dJ/du`` from an adjoint sweep; compare with ``jax.grad``."""
    problem, x0 = lq_1d(), jnp.array([2.0, 0.0])
    controls = 0.9 * jnp.sin(jnp.arange(T, dtype=float))[:, None]
    gradient = jax.grad(lambda u: problem.total_cost(problem.rollout(x0, u), u))(controls)
    expected = jnp.max(jnp.abs(controls - jnp.clip(controls - gradient, -1.0, 1.0)))
    *_, error, _ = MirrorILQR(problem, BoxEntropy(-1.0, 1.0), NATURAL).backward_pass(problem.rollout(x0, controls), controls, 1e-2)
    assert expected > 1e-2
    np.testing.assert_allclose(error, expected, rtol=1e-12)


def test_natural_stopping_reaches_the_constrained_optimum():
    problem, x0 = lq_1d(), jnp.array([2.0, 0.0])  # 26 of the 30 optimal controls on the bound
    u_ref, _ = reference_qp(problem, x0, 1, box(-1.0, 1.0))
    sol = MirrorILQR(problem, BoxEntropy(-1.0, 1.0), NATURAL).solve(x0, jnp.zeros((T, 1)))
    assert sol.status == Status.CONVERGED and sol.error < 1e-6
    np.testing.assert_allclose(sol.controls, u_ref, atol=1e-5)


def test_natural_stopping_reports_frozen_controls():
    """With the full map, controls exactly on the bound have ``Psi^{-1} = 0``, so no step can
    move them. The natural residual sees that the gradient points into the box: the solve ends ``STALLED``."""
    problem, x0 = lq_1d(), jnp.array([2.0, 0.0])
    plan = jnp.ones((T, 1))  # on the upper bound; the optimum pushes them down
    sol = MirrorILQR(problem, BoxEntropy(-1.0, 1.0), SolverOptions(one_sided=False)).solve(x0, plan, validate=False)
    assert sol.status == Status.STALLED and sol.error > 1e-2 and int(sol.iterations) == 1
    assert SolverOptions().tol == 1e-4


# ---------------------------------------------------------------- one-sided box map ---

FULL_MAP = SolverOptions(tol=1e-6, one_sided=False)


def test_one_sided_map_identities():
    box = BoxEntropy(jnp.array([-1.0, 0.5]), jnp.array([2.0, 3.0]))
    u, up = jnp.array([1.3, 0.9]), jnp.array([True, False])
    y = box.one_sided_grad(u, up)
    np.testing.assert_allclose(box.one_sided_inverse(y, up), u, rtol=1e-13)
    jacobian = jax.jacfwd(lambda v: box.one_sided_grad(v, up))(u)  # Psi of the one-sided map
    np.testing.assert_allclose(box.one_sided_inv_hessian(u, up) @ jacobian, jnp.eye(2), atol=1e-13)
    # on a bound: frozen when heading into it, free when heading away from it
    on_bound = jnp.array([2.0, 0.5])
    np.testing.assert_allclose(jnp.diag(box.one_sided_inv_hessian(on_bound, jnp.array([True, False]))), 0.0)
    assert bool(jnp.all(jnp.diag(box.one_sided_inv_hessian(on_bound, jnp.array([False, True]))) > 1.0))
    np.testing.assert_allclose(box.one_sided_inverse(box.one_sided_grad(on_bound, up), up), on_bound)


def test_one_sided_releases_frozen_controls():
    """Controls on the upper bound while the optimum is at the lower one: the two-sided map
    cannot move them (``STALLED``); the one-sided map reaches the constrained optimum."""
    problem, x0 = lq_1d(), jnp.array([2.0, 0.0])
    u_ref, _ = reference_qp(problem, x0, 1, box(-1.0, 1.0))
    plan = jnp.ones((T, 1))
    assert MirrorILQR(problem, BoxEntropy(-1.0, 1.0), FULL_MAP).solve(x0, plan, validate=False).status == Status.STALLED
    for start in (plan, jnp.zeros((T, 1))):
        sol = MirrorILQR(problem, BoxEntropy(-1.0, 1.0), NATURAL).solve(x0, start, validate=False)
        assert sol.status == Status.CONVERGED and sol.error < 1e-6
        np.testing.assert_allclose(sol.controls, u_ref, atol=1e-5)
    assert SolverOptions().one_sided and not BoxLogBarrier(-1.0, 1.0).has_one_sided  # such maps keep the full map


def test_one_sided_polytope_map_identities():
    polytope = PolytopeLogBarrier(jnp.array(POLYTOPE_G), jnp.array(POLYTOPE_H))
    u, direction = jnp.array([0.3, -0.25]), jnp.array([1.0, 0.2])
    ahead = polytope.heading(u, direction)
    np.testing.assert_array_equal(ahead, jnp.array(POLYTOPE_G) @ direction > 0)
    y = polytope.one_sided_grad(u, ahead)
    np.testing.assert_allclose(polytope.one_sided_inverse(y, ahead, u), u, atol=1e-10)
    jacobian = jax.jacfwd(lambda v: polytope.one_sided_grad(v, ahead))(u)
    np.testing.assert_allclose(polytope.one_sided_inv_hessian(u, ahead) @ jacobian, jnp.eye(2), atol=1e-12)
    # with every facet ahead it is the full map; with none it is the identity
    everything, nothing = jnp.ones(8, bool), jnp.zeros(8, bool)
    np.testing.assert_allclose(polytope.one_sided_grad(u, everything), polytope.grad(u), rtol=1e-14)
    np.testing.assert_allclose(polytope.one_sided_grad(u, nothing), u)
    # a step across a facet that is not ahead is shortened and stays strictly inside
    outside = polytope.one_sided_inverse(jnp.array([-2.0, 0.0]), nothing, u)
    assert bool(polytope.contains(outside)) and float(jnp.min(polytope.slack(outside))) > 0


def test_one_sided_polytope_releases_controls_next_to_a_facet():
    """A plan 1e-7 from the facet ``u1 <= 0.5`` while the optimum pushes ``u1`` down: the full
    map cannot move it away within the iteration cap; the one-sided map reaches the optimum."""
    problem, polytope = DoubleIntegrator().problem(), PolytopeLogBarrier(jnp.array(POLYTOPE_G), jnp.array(POLYTOPE_H))
    x0, plan = jnp.array([1.6, 1.0, 0.0, 0.0]), jnp.tile(jnp.array([0.5 - 1e-7, 0.0]), (T, 1))
    u_ref, _ = reference_qp(problem, x0, 2, polytope.project)
    assert float(u_ref[0, 0]) < -0.4
    full = MirrorILQR(problem, polytope, SolverOptions(max_iter=200, one_sided=False)).solve(x0, plan)
    assert full.status != Status.CONVERGED
    sol = MirrorILQR(problem, polytope).solve(x0, plan)
    assert sol.status == Status.CONVERGED and int(sol.iterations) < 100
    np.testing.assert_allclose(sol.controls, u_ref, atol=2e-3)
    assert bool(jnp.all(jax.vmap(polytope.contains)(sol.controls)))
