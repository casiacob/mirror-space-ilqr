"""Lemma 3 / Proposition 4: derivatives under the mirror map, against autodiff."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from msilqr import BoxEntropy, BoxLogBarrier, EuclideanBall

MAPS = [
    ("box-entropy", BoxEntropy(jnp.array([-1.0, -2.0]), jnp.array([3.0, 1.0])), jnp.array([0.5, -0.3])),
    ("box-log-barrier", BoxLogBarrier(jnp.array([-1.0, -2.0]), jnp.array([3.0, 1.0])), jnp.array([0.5, -0.3])),
    ("euclidean-ball", EuclideanBall(), jnp.array([0.2, -0.5])),
]


def g(u):
    return jnp.sin(u[0]) * u[1] + 0.3 * u[0] ** 2 + jnp.exp(0.5 * u[1])


@pytest.mark.parametrize("name,mmap,u", MAPS, ids=[m[0] for m in MAPS])
def test_lemma3_gradient_and_hessian(name, mmap, u):
    y = mmap.grad(u)
    h = lambda y_: g(mmap.inverse(y_, u))  # noqa: E731
    W = mmap.inv_hessian(u)

    # Eq. (7): grad h = Psi^{-1} grad g
    np.testing.assert_allclose(jax.grad(h)(y), W @ jax.grad(g)(u), rtol=1e-9, atol=1e-12)

    # Eq. (8): hess h = Psi^{-1} [hess g - T_psi . grad h] Psi^{-1}
    T = jax.jacfwd(mmap.hessian)(u)  # T[i, j, k] = d^3 psi / du_i du_j du_k
    grad_h = jax.grad(h)(y)
    contraction = jnp.einsum("ijk,k->ij", T, grad_h)
    expected = W @ (jax.hessian(g)(u) - contraction) @ W
    np.testing.assert_allclose(jax.hessian(h)(y), expected, rtol=1e-8, atol=1e-10)
