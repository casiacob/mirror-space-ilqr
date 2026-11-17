"""Mirror-map identities (Sec. III-A, Proposition 2, Table I)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from msilqr import BoxEntropy, BoxLogBarrier, EuclideanBall, PolytopeLogBarrier, Positivity
from msilqr.systems import POLYTOPE_G, POLYTOPE_H

KEY = jax.random.PRNGKey(0)


def _box_points(low, high, n=7):
    t = jnp.linspace(0.02, 0.98, n)
    return low + jnp.stack([t, t[::-1]], axis=1) * (high - low)  # (n, 2), interior points


CLOSED_FORM = [
    ("box-entropy", BoxEntropy(-50.0, 50.0), _box_points(jnp.array([-50.0, -50.0]), jnp.array([50.0, 50.0]))),
    (
        "box-entropy-asym",
        BoxEntropy(jnp.array([-10.0, 0.0]), jnp.array([50.0, 1.0])),
        _box_points(jnp.array([-10.0, 0.0]), jnp.array([50.0, 1.0])),
    ),
    (
        "box-log-barrier",
        BoxLogBarrier(jnp.array([-2.0, 1.0]), jnp.array([3.0, 4.0])),
        _box_points(jnp.array([-2.0, 1.0]), jnp.array([3.0, 4.0])),
    ),
    ("positivity", Positivity(), jnp.array([[0.1, 2.0], [1.0, 1.0], [5.0, 0.01]])),
    ("euclidean-ball", EuclideanBall(), jnp.array([[0.0, 0.0], [0.3, -0.4], [-0.7, 0.6], [0.0, 0.999]])),
]


@pytest.mark.parametrize("name,mmap,points", CLOSED_FORM, ids=[c[0] for c in CLOSED_FORM])
def test_round_trip_and_hessians(name, mmap, points):
    for u in points:
        y = mmap.grad(u)
        np.testing.assert_allclose(mmap.inverse(y, u), u, rtol=1e-10, atol=1e-10)
        # Psi = Jacobian of grad psi, Psi^{-1} = Jacobian of grad psi* (Proposition 2)
        np.testing.assert_allclose(jax.jacfwd(mmap.grad)(u), mmap.hessian(u), rtol=1e-8, atol=1e-8)
        inverse_jacobian = jax.jacfwd(lambda y_, u_=u: mmap.inverse(y_, u_))(y)
        np.testing.assert_allclose(inverse_jacobian, mmap.inv_hessian(u), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(mmap.inv_hessian(u) @ mmap.hessian(u), np.eye(2), atol=1e-9)


@pytest.mark.parametrize("name,mmap,points", CLOSED_FORM, ids=[c[0] for c in CLOSED_FORM])
def test_inverse_lands_in_interior(name, mmap, points):
    ys = 3.0 * jax.random.normal(KEY, (50, 2))
    for y in ys:
        u = mmap.inverse(y, points[0])
        np.testing.assert_allclose(mmap.grad(u), y, rtol=1e-8, atol=1e-8)


def test_box_entropy_is_table_one():
    a, b = -10.0, 50.0
    mmap = BoxEntropy(a, b)
    mu, rho = (a + b) / 2, (b - a) / 2
    for u in jnp.linspace(a + 1e-3, b - 1e-3, 11):
        np.testing.assert_allclose(mmap.grad(u), jnp.log((u - a) / (b - u)), rtol=1e-12)
        np.testing.assert_allclose(mmap.hessian(u)[0, 0], 2 * rho / ((u - a) * (b - u)), rtol=1e-12)
        y = mmap.grad(u)
        np.testing.assert_allclose(mmap.inverse(y), mu + rho * jnp.tanh(y / 2), rtol=1e-12)


def test_box_entropy_saturation_is_finite():
    mmap = BoxEntropy(-50.0, 50.0)
    y_at_bound = mmap.grad(jnp.array(50.0))
    assert jnp.isfinite(y_at_bound) and y_at_bound == mmap.y_clip
    assert mmap.grad(jnp.array(-50.0)) == -mmap.y_clip
    # the clip value maps exactly onto the bound
    assert mmap.inverse(mmap.y_clip) == 50.0 and mmap.inverse(-mmap.y_clip) == -50.0
    assert mmap.inv_hessian(jnp.array(50.0))[0, 0] == 0.0


@pytest.mark.parametrize("cls", [BoxEntropy, BoxLogBarrier])
@pytest.mark.parametrize("low,high", [(1.0, 1.0), (2.0, 1.0), (-np.inf, 1.0), (0.0, np.nan)])
def test_box_rejects_invalid_bounds(cls, low, high):
    with pytest.raises(ValueError):
        cls(low, high)


@pytest.mark.parametrize("cls,y_big", [(BoxEntropy, 1e3), (BoxLogBarrier, 1e300)])
def test_box_saturation_stays_in_closed_box_for_asymmetric_bounds(cls, y_big):
    """mu +- rho can differ from the bound by one rounding error; the inverse must clip."""
    rng = np.random.default_rng(0)
    lows = rng.uniform(-100, 100, 2000)
    highs = lows + rng.uniform(1e-3, 100, 2000)
    for lo, hi in zip(lows, highs, strict=True):
        mmap = cls(float(lo), float(hi))
        for y in (y_big, -y_big):
            u = mmap.inverse(jnp.array(y))
            assert lo <= u <= hi
            assert jnp.isfinite(mmap.grad(u))
            assert mmap.inv_hessian(u)[0, 0] >= 0
    # asymmetric boxes: the saturated value is exactly the bound
    assert BoxEntropy(0.1, 0.7).inverse(jnp.array(-40.0)) == 0.1
    assert BoxEntropy(-1.0, 0.1).inverse(jnp.array(40.0)) == 0.1


def test_positivity_and_ball_extreme_inputs():
    pos = Positivity()
    for y in (-1e4, -800.0, 800.0, 1e4):
        u = pos.inverse(jnp.array([y]))
        assert pos.contains(u) and jnp.all(jnp.isfinite(u)) and jnp.all(jnp.isfinite(pos.grad(u)))
    ball = EuclideanBall()
    for y in ([1e8, 0.0], [1e9, 1e9], [1e200, -1e200], [1e-300, 0.0], [0.0, 0.0]):
        u = ball.inverse(jnp.array(y))
        assert ball.contains(u) and jnp.all(jnp.isfinite(ball.grad(u))) and jnp.all(jnp.isfinite(ball.inv_hessian(u)))
    np.testing.assert_allclose(ball.inverse(jnp.array([3.0, 4.0])), jnp.array([3.0, 4.0]) / jnp.sqrt(26.0), rtol=1e-15)


@pytest.mark.parametrize("dim", [2, 3, 10, 100])
def test_ball_inverse_stays_inside_for_large_random_y(dim):
    """Rounding in the capped branch must never put u on or outside the sphere."""
    rng = np.random.default_rng(dim)
    directions = rng.normal(size=(4000, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.exp(rng.uniform(np.log(1e6), np.log(1e300), size=(4000, 1)))
    ball = EuclideanBall()
    u = jax.vmap(ball.inverse)(jnp.asarray(directions * radii))
    assert bool(jnp.all(jax.vmap(ball.contains)(u)))
    assert bool(jnp.all(jnp.isfinite(jax.vmap(ball.grad)(u))))


@pytest.mark.parametrize("y", [[0.0, 0.0], [1e-300, 0.0], [1e-160, -1e-160], [3.0, -4.0], [2e7, 1e7], [1e200, -3e199]])
def test_ball_inverse_reverse_mode_derivatives_are_finite(y):
    """Both branches of the inverse are evaluated; the unused one must not leak NaN."""
    ball, y = EuclideanBall(), jnp.array(y)
    J_rev, J_fwd = jax.jacrev(ball.inverse)(y), jax.jacfwd(ball.inverse)(y)
    assert bool(jnp.all(jnp.isfinite(J_rev)))
    np.testing.assert_allclose(J_rev, J_fwd, rtol=1e-12, atol=1e-15)
    if float(jnp.linalg.norm(y)) < 1e-100:
        np.testing.assert_array_equal(J_rev, jnp.eye(2))


def test_ball_grad_is_finite_on_the_sphere():
    ball, u = EuclideanBall(), jnp.array([0.6, 0.8])
    assert bool(jnp.all(jnp.isfinite(ball.grad(u)))) and bool(jnp.all(jnp.isfinite(ball.inv_hessian(u))))


def test_box_accepts_list_bounds():
    for cls in (BoxEntropy, BoxLogBarrier):
        mmap = cls([-1.0, 0.0], [1.0, 2.0])
        u = mmap.inverse(jnp.array([0.3, -0.2]))
        assert mmap.contains(u)
        np.testing.assert_allclose(mmap.grad(u), jnp.array([0.3, -0.2]), rtol=1e-12)


def test_contains():
    assert BoxEntropy(-1.0, 1.0).contains(jnp.array([0.5])) and not BoxEntropy(-1.0, 1.0).contains(jnp.array([1.0]))
    assert BoxLogBarrier(0.0, 2.0).contains(jnp.array([1.0])) and not BoxLogBarrier(0.0, 2.0).contains(jnp.array([0.0]))
    assert Positivity().contains(jnp.array([1e-300])) and not Positivity().contains(jnp.array([0.0, 1.0]))
    assert not Positivity().contains(jnp.array([jnp.inf])) and not Positivity().contains(jnp.array([jnp.nan]))
    assert EuclideanBall().contains(jnp.array([0.6, 0.7])) and not EuclideanBall().contains(jnp.array([0.6, 0.8]))
    poly = PolytopeLogBarrier(G=jnp.array(POLYTOPE_G), h=jnp.array(POLYTOPE_H))
    assert poly.contains(jnp.array([0.1, 0.1])) and not poly.contains(jnp.array([0.5, 0.0]))


def test_polytope_map():
    mmap = PolytopeLogBarrier(G=jnp.array(POLYTOPE_G), h=jnp.array(POLYTOPE_H))
    u = jnp.array([0.1, -0.2])
    np.testing.assert_allclose(jax.jacfwd(mmap.grad)(u), mmap.hessian(u), rtol=1e-10)
    # Proposition 2: the Jacobian of the (numerical) inverse map is Psi^{-1}
    y = mmap.grad(u)
    inverse_jacobian = jax.jacrev(lambda y_: mmap.inverse(y_, u))(y)
    np.testing.assert_allclose(inverse_jacobian, mmap.inv_hessian(u), rtol=1e-7, atol=1e-10)
    # Newton inverse from a nearby warm start
    for du in 0.05 * jax.random.normal(KEY, (20, 2)):
        target = u + du
        y = mmap.grad(target)
        u_inv = mmap.inverse(y, u)
        residual, min_slack = mmap.inverse_residual(y, u_inv)
        assert residual < 1e-9 and min_slack > 0
        np.testing.assert_allclose(u_inv, target, atol=1e-9)
