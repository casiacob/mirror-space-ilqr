"""Mirror maps for convex control constraints (Sec. III-A and Table I of the paper).

A mirror map is the gradient of a Legendre-type distance-generating function (DGF)
``psi`` whose domain interior is the admissible set ``int(U)`` (Definition 1). It maps
constrained controls ``u`` to unconstrained mirror controls ``y = grad psi(u)``; its
inverse is ``grad psi*(y)`` (Proposition 2). The solver only needs:

* ``grad(u)``         -- the mirror map ``y = grad psi(u)``,
* ``inverse(y, u_ref)`` -- the inverse map ``u = grad psi*(y)`` (``u_ref`` is a warm start
  for maps whose inverse has no closed form),
* ``inv_hessian(u)``  -- ``Psi(u)^{-1} = (grad^2 psi(u))^{-1}``, the Jacobian ``du/dy``.

``hessian(u)`` (``Psi(u)``) is provided for completeness and for tests, and
``contains(u)`` tests ``u in int(U)``.

All maps act on a single control vector of shape ``(m,)`` and are pure JAX functions.

Floating point: in exact arithmetic ``grad psi*`` maps every ``y`` into ``int(U)``. In
float64 the closed-form inverses saturate (``tanh`` rounds to exactly 1, ``exp``
underflows), so the returned control can land exactly on the boundary of ``U``. Every
map here guarantees that ``inverse`` returns a point of the closed set ``U`` and that
``grad`` stays finite at boundary points, so no ``inf``/``NaN`` enters an iteration.
At such a point ``Psi^{-1}`` is 0 (the control is frozen until a later warm start moves
it); see the README.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsc
import numpy as np

__all__ = [
    "MirrorMap",
    "BoxEntropy",
    "BoxLogBarrier",
    "Positivity",
    "EuclideanBall",
    "PolytopeLogBarrier",
]


class MirrorMap(abc.ABC):
    """Interface of a mirror map ``grad psi`` for a Legendre-type DGF ``psi``."""

    @abc.abstractmethod
    def grad(self, u: jax.Array) -> jax.Array:
        """Mirror map ``y = grad psi(u)`` for ``u`` in ``int(U)``."""

    @abc.abstractmethod
    def inverse(self, y: jax.Array, u_ref: jax.Array) -> jax.Array:
        """Inverse map ``u = grad psi*(y)``; ``u_ref`` may warm-start numerical inverses."""

    @abc.abstractmethod
    def inv_hessian(self, u: jax.Array) -> jax.Array:
        """``Psi(u)^{-1}``, the inverse Hessian of ``psi`` (an ``(m, m)`` matrix)."""

    @abc.abstractmethod
    def hessian(self, u: jax.Array) -> jax.Array:
        """``Psi(u) = grad^2 psi(u)`` (an ``(m, m)`` matrix)."""

    @abc.abstractmethod
    def contains(self, u: jax.Array) -> jax.Array:
        """Whether ``u`` lies in the interior ``int(U)`` (a boolean scalar)."""


def _check_box(low, high) -> None:
    """Validate concrete box bounds (skipped for traced values)."""
    if isinstance(low, jax.core.Tracer) or isinstance(high, jax.core.Tracer):
        return
    lo, hi = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("box bounds must be finite")
    if not np.all(lo < hi):
        raise ValueError(f"box bounds must satisfy low < high, got low={low}, high={high}")


# --------------------------------------------------------------------------------------
# Box (entropy), Table I row 1
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BoxEntropy(MirrorMap):
    r"""Entropy DGF of the box ``low < u < high`` (Table I, row "Box (entropy)").

    ``psi(u) = scale * sum_i [(b_i - u_i) log(b_i - u_i) + (u_i - a_i) log(u_i - a_i)]``

    * mirror map   ``y = scale * log((u - a) / (b - u))``
    * inverse      ``u = mu + rho * tanh(y / (2 scale))`` with ``mu = (a + b) / 2``,
      ``rho = (b - a) / 2``
    * ``Psi^{-1}(u) = diag((u - a)(b - u) / (2 rho scale))``

    ``scale = 1`` is exactly the paper's Table I map. Any ``scale > 0`` gives a valid
    Legendre function with the same constraint set but a different geometry: with the
    paper's ``lambda * I`` damping in mirror coordinates (Sec. III-C) the scale changes
    the iterates. The original cart-pole experiments used ``scale = 0.5``
    (``y = 0.5 log(...)``, ``u = rho tanh(y)``); see the README.

    Floating point: once ``|y| / (2 scale)`` exceeds about 19.06, ``tanh`` rounds to
    exactly 1 in float64. Then ``mu + rho * tanh(.)`` evaluates to ``mu +- rho``, which for
    an asymmetric box can differ from the bound by one rounding error, i.e. lie just
    OUTSIDE the box. ``inverse`` therefore clips its result to ``[low, high]`` (a no-op
    for symmetric boxes), and ``grad`` clips ``y`` to ``+-40 * scale`` instead of
    returning ``+-inf`` at the bound (``tanh(20)`` is also exactly 1, and
    ``Psi^{-1} = 0`` at the bound makes the mirror-space step zero there). Both guards
    leave the results of the original symmetric-box experiments bitwise unchanged.
    """

    low: float | jax.Array
    high: float | jax.Array
    scale: float = 1.0

    def __post_init__(self):
        if not self.scale > 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        _check_box(self.low, self.high)

    @property
    def mu(self):
        return 0.5 * (self.low + self.high)

    @property
    def rho(self):
        return 0.5 * (self.high - self.low)

    @property
    def y_clip(self) -> float:
        return 40.0 * self.scale

    def grad(self, u):
        y = self.scale * jnp.log((u - self.low) / (self.high - u))
        return jnp.clip(y, -self.y_clip, self.y_clip)

    def inverse(self, y, u_ref=None):
        u = self.mu + self.rho * jnp.tanh(y / (2 * self.scale))
        return jnp.clip(u, self.low, self.high)

    def inv_hessian(self, u):
        diag = (u - self.low) * (self.high - u) / (2 * self.rho * self.scale)
        return jnp.diag(jnp.atleast_1d(diag))

    def hessian(self, u):
        diag = 2 * self.rho * self.scale / ((u - self.low) * (self.high - u))
        return jnp.diag(jnp.atleast_1d(diag))

    def contains(self, u):
        return jnp.all((u > self.low) & (u < self.high))


# --------------------------------------------------------------------------------------
# Box (log-barrier), Positivity, Euclidean ball: Table I rows 2-4
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BoxLogBarrier(MirrorMap):
    r"""Log-barrier DGF of the box ``low < u < high`` (Table I, row "Box (log-barrier)").

    ``psi(u) = -sum_i [log(b_i - u_i) + log(u_i - a_i)]``;
    ``y = 1/(b - u) - 1/(u - a)``; ``u = mu + rho tanh(arcsinh(rho y) / 2)``.

    As for :class:`BoxEntropy`, ``inverse`` clips to ``[low, high]`` and ``grad`` returns
    ``+-1e300`` instead of ``+-inf`` at the bound.
    """

    low: float | jax.Array
    high: float | jax.Array

    #: finite stand-in for ``+-inf``; ``inverse(+-Y_CLIP)`` is exactly the bound
    Y_CLIP = 1e300

    def __post_init__(self):
        _check_box(self.low, self.high)

    @property
    def mu(self):
        return 0.5 * (self.low + self.high)

    @property
    def rho(self):
        return 0.5 * (self.high - self.low)

    def grad(self, u):
        y = 1.0 / (self.high - u) - 1.0 / (u - self.low)
        return jnp.clip(y, -self.Y_CLIP, self.Y_CLIP)

    def inverse(self, y, u_ref=None):
        u = self.mu + self.rho * jnp.tanh(0.5 * jnp.arcsinh(self.rho * y))
        return jnp.clip(u, self.low, self.high)

    def hessian(self, u):
        return jnp.diag(jnp.atleast_1d(1.0 / (self.high - u) ** 2 + 1.0 / (u - self.low) ** 2))

    def inv_hessian(self, u):
        d = 1.0 / (self.high - u) ** 2 + 1.0 / (u - self.low) ** 2
        return jnp.diag(jnp.atleast_1d(1.0 / d))

    def contains(self, u):
        return jnp.all((u > self.low) & (u < self.high))


@dataclass(frozen=True, eq=False)
class Positivity(MirrorMap):
    r"""Entropy DGF of the positive orthant ``u > 0`` (Table I, row "Positivity").

    ``psi(u) = sum_i (u_i log u_i - u_i)``; ``y = log u``; ``u = exp(y)``.

    ``inverse`` clips ``y`` to ``[log(tiny), log(max)]`` of float64 so that ``u`` is a
    positive, finite, normal number (``exp`` would underflow to 0 or overflow to inf).
    """

    _Y_MIN = float(np.log(np.finfo(np.float64).tiny))
    _Y_MAX = float(np.log(np.finfo(np.float64).max))

    def grad(self, u):
        return jnp.log(u)

    def inverse(self, y, u_ref=None):
        return jnp.exp(jnp.clip(y, self._Y_MIN, self._Y_MAX))

    def hessian(self, u):
        return jnp.diag(jnp.atleast_1d(1.0 / u))

    def inv_hessian(self, u):
        return jnp.diag(jnp.atleast_1d(u))

    def contains(self, u):
        return jnp.all(u > 0)


@dataclass(frozen=True, eq=False)
class EuclideanBall(MirrorMap):
    r"""DGF of the open unit ball ``||u||_2 < 1`` (Table I, row "Euclidean ball").

    ``psi(u) = -sqrt(1 - ||u||^2)``; ``y = u / sqrt(1 - ||u||^2)``;
    ``u = y / sqrt(1 + ||y||^2)``; ``Psi^{-1}(u) = s (I - u u^T)`` with
    ``s = sqrt(1 - ||u||^2)`` (Sherman-Morrison).

    ``inverse`` uses the closed form for ``||y|| < 1e7``. Beyond that it computes ``||y||``
    without overflow and caps the radius of ``u`` at ``1 - 2^-52``: in float64 the closed
    form would round onto the sphere (``||u|| = 1``, where ``grad`` is infinite) and
    ``y @ y`` overflows for ``||y|| > 1e154``. So ``u`` stays strictly inside the ball for
    every finite ``y``.
    """

    _R_MAX = 1.0 - 2.0**-52
    _N_DIRECT = 1e7

    def grad(self, u):
        return u / jnp.sqrt(1.0 - u @ u)

    def inverse(self, y, u_ref=None):
        m = jnp.max(jnp.abs(y))
        safe_m = jnp.where(m > 0, m, 1.0)
        n = safe_m * jnp.sqrt(jnp.sum((y / safe_m) ** 2))  # ||y|| without overflow
        large = n >= self._N_DIRECT
        y_small = jnp.where(large, jnp.zeros_like(y), y)  # keeps the unused branch finite
        direct = y_small / jnp.sqrt(1.0 + y_small @ y_small)
        r = jnp.minimum(n / jnp.hypot(1.0, n), self._R_MAX)  # ||u|| = n / sqrt(1 + n^2)
        capped = y * (r / jnp.where(large, n, 1.0))
        return jnp.where(large, capped, direct)

    def hessian(self, u):
        s = jnp.sqrt(1.0 - u @ u)
        return jnp.eye(u.shape[0]) / s + jnp.outer(u, u) / s**3

    def inv_hessian(self, u):
        s = jnp.sqrt(1.0 - u @ u)
        return s * (jnp.eye(u.shape[0]) - jnp.outer(u, u))

    def contains(self, u):
        return u @ u < 1.0


# --------------------------------------------------------------------------------------
# Polytope (Sec. V-a): quadratic plus log-barrier DGF, numerical inverse
# --------------------------------------------------------------------------------------


def _softplus_inv(s):
    return s + jnp.log1p(-jnp.exp(-s))


@dataclass(frozen=True, eq=False)
class PolytopeLogBarrier(MirrorMap):
    r"""DGF of the polytope ``G u < h`` used in the paper's double-integrator example.

    ``psi(u) = 0.5 u^T u - sum_j log s_j(u)`` with slack ``s(u) = h - G u`` (Sec. V-a):

    * mirror map ``y = u + G^T (1 / s(u))``
    * ``Psi(u) = I + G^T diag(1 / s^2) G``
    * inverse ``grad psi*(y)``: no closed form; solved with Newton's root-finding method
      (Sec. V-a, [22]) from the warm start ``u_ref``.

    The Newton solve reproduces the original implementation exactly: Newton on the
    lifted system ``(u, r)`` with ``s = softplus(r)``, residuals
    ``u + G^T / s - y`` and ``G u + s - h``, ``rtol = atol = 1e-11`` (RMS norm), at most
    128 steps. It does not raise on non-convergence (as in the original); for large
    mirror-space steps it can return NaN, which makes the solver reject that step. Use
    :meth:`inverse_residual` to check a result. The regression benchmarks depend on this
    exact solver: a mathematically cleaner damped Newton changes the golden results.
    """

    G: jax.Array
    h: jax.Array
    newton_tol: float = 1e-11
    newton_max_steps: int = 128

    def slack(self, u):
        return self.h - self.G @ u

    def psi(self, u):
        s = self.slack(u)
        return 0.5 * u.T @ u - jnp.sum(jnp.log(s))

    def grad(self, u):
        s = self.slack(u)
        return u + self.G.T @ (1.0 / s)

    def hessian(self, u):
        s = self.slack(u)
        W = 1.0 / s**2
        return jnp.eye(u.shape[0]) + self.G.T @ (W[:, None] * self.G)

    def inv_hessian(self, u):
        H = self.hessian(u)
        return jsc.linalg.solve(H, jnp.eye(u.shape[0]))

    def inverse(self, y, u_ref):
        import optimistix as optx

        n_u = y.shape[0]
        G, h = self.G, self.h

        def residual(q, y):
            u, r = q[:n_u], q[n_u:]
            s = jax.nn.softplus(r)
            mirror_residual = u + G.T @ (1.0 / s) - y
            slack_residual = G @ u + s - h
            return jnp.concatenate((mirror_residual, slack_residual))

        solver = optx.Newton(rtol=self.newton_tol, atol=self.newton_tol, norm=optx.rms_norm)
        q0 = jnp.concatenate((u_ref, _softplus_inv(h - G @ u_ref)))
        sol = optx.root_find(residual, solver, q0, args=y, max_steps=self.newton_max_steps, throw=False)
        return sol.value[:n_u]

    def inverse_residual(self, y, u):
        """``max |grad psi(u) - y|`` and ``min s(u)``: a returned inverse is valid only if
        the first is small and the second is positive."""
        return jnp.max(jnp.abs(self.grad(u) - y)), jnp.min(self.slack(u))

    def contains(self, u):
        return jnp.all(self.slack(u) > 0)
