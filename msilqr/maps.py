"""Mirror maps for convex control constraints (Sec. III-A and Table I of the paper).

A mirror map ``y = grad psi(u)`` is the gradient of a Legendre function ``psi`` whose
domain interior is ``int(U)``; its inverse is ``u = grad psi*(y)``. Maps act on a single
control of shape ``(m,)``.
"""

from __future__ import annotations

import abc
import itertools
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

    def project(self, u: jax.Array) -> jax.Array:
        """Euclidean projection of ``u`` onto the closed set ``U`` (used by the stopping
        test of :class:`~msilqr.MirrorILQR`)."""
        raise NotImplementedError(f"{type(self).__name__} has no projection")

    #: whether the map has the one-sided variant that :class:`~msilqr.MirrorILQR` uses by default:
    #: ``heading(u, direction)`` says which part of the boundary a control moving along
    #: ``direction`` is heading towards, and ``one_sided_grad(u, heading)``,
    #: ``one_sided_inverse(y, heading, u_ref)`` and ``one_sided_inv_hessian(u, heading)`` are
    #: the map of that part alone.
    has_one_sided = False


def _as_bound(value):
    """Turn list/tuple bounds into float64 arrays (scalars and arrays pass through unchanged)."""
    return np.asarray(value, dtype=np.float64) if isinstance(value, list | tuple) else value


def _init_box(box) -> None:
    object.__setattr__(box, "low", _as_bound(box.low))
    object.__setattr__(box, "high", _as_bound(box.high))
    _check_box(box.low, box.high)


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

    ``psi(u) = sum_i [(b_i - u_i) log(b_i - u_i) + (u_i - a_i) log(u_i - a_i)]``

    * mirror map   ``y = log((u - a) / (b - u))``
    * inverse      ``u = mu + rho * tanh(y / 2)`` with ``mu = (a + b) / 2``,
      ``rho = (b - a) / 2``
    * ``Psi^{-1}(u) = diag((u - a)(b - u) / (2 rho))``

    Floating point: once ``|y| / 2`` exceeds about 19.06, ``tanh`` rounds to exactly 1 in
    float64. Then ``mu + rho * tanh(.)`` evaluates to ``mu +- rho``, which for an asymmetric
    box can differ from the bound by one rounding error, i.e. lie just OUTSIDE the box.
    ``inverse`` therefore clips its result to ``[low, high]`` (a no-op for symmetric boxes),
    and ``grad`` clips ``y`` to ``+-40`` instead of returning ``+-inf`` at the bound
    (``tanh(20)`` is also exactly 1, and ``Psi^{-1} = 0`` at the bound makes the mirror-space
    step zero there).

    One-sided variant (the solver's default): a control heading up (``up``)
    uses the entropy map of the upper bound alone, ``y = -log(b - u)`` with
    ``Psi^{-1} = b - u``; one heading down uses ``y = log(u - a)`` with ``Psi^{-1} = u - a``.
    Near the bound a control is heading to, this agrees with the two-sided map. A control on
    a bound that heads back into the box is not frozen.
    """

    low: float | jax.Array
    high: float | jax.Array
    has_one_sided = True
    y_clip = 40.0

    def __post_init__(self):
        _init_box(self)

    @property
    def mu(self):
        return 0.5 * (self.low + self.high)

    @property
    def rho(self):
        return 0.5 * (self.high - self.low)

    def grad(self, u):
        y = jnp.log((u - self.low) / (self.high - u))
        return jnp.clip(y, -self.y_clip, self.y_clip)

    def inverse(self, y, u_ref=None):
        u = self.mu + self.rho * jnp.tanh(y / 2)
        return jnp.clip(u, self.low, self.high)

    def inv_hessian(self, u):
        diag = (u - self.low) * (self.high - u) / (2 * self.rho)
        return jnp.diag(jnp.atleast_1d(diag))

    def hessian(self, u):
        diag = 2 * self.rho / ((u - self.low) * (self.high - u))
        return jnp.diag(jnp.atleast_1d(diag))

    def contains(self, u):
        return jnp.all((u > self.low) & (u < self.high))

    def project(self, u):
        return jnp.clip(u, self.low, self.high)

    def heading(self, u, direction):
        """Per coordinate: whether a control moving along ``direction`` heads to the upper bound."""
        return direction >= 0.0

    def one_sided_grad(self, u, up):
        y = jnp.where(up, -jnp.log(self.high - u), jnp.log(u - self.low))
        return jnp.clip(y, -700.0, 700.0)  # exp(-700) > 0 in float64

    def one_sided_inverse(self, y, up, u_ref=None):
        u = jnp.where(up, self.high - jnp.exp(-y), self.low + jnp.exp(y))
        return jnp.clip(u, self.low, self.high)  # a step against the heading may reach the other bound

    def one_sided_inv_hessian(self, u, up):
        return jnp.diag(jnp.atleast_1d(jnp.where(up, self.high - u, u - self.low)))


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
        _init_box(self)

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

    def project(self, u):
        return jnp.clip(u, self.low, self.high)


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
        return jnp.all((u > 0) & jnp.isfinite(u))

    def project(self, u):
        return jnp.maximum(u, 0.0)


@dataclass(frozen=True, eq=False)
class EuclideanBall(MirrorMap):
    r"""DGF of the open unit ball ``||u||_2 < 1`` (Table I, row "Euclidean ball").

    ``psi(u) = -sqrt(1 - ||u||^2)``; ``y = u / sqrt(1 - ||u||^2)``;
    ``u = y / sqrt(1 + ||y||^2)``; ``Psi^{-1}(u) = s (I - u u^T)`` with
    ``s = sqrt(1 - ||u||^2)`` (Sherman-Morrison).

    Floating point:

    * ``inverse`` uses the closed form for ``||y|| < 1e7``. Beyond that the closed form
      would round onto the sphere (where ``grad`` is infinite), and ``y @ y`` overflows
      for ``||y|| > 1e154``. So for large ``y`` it computes ``||y||`` without overflow and
      caps the radius of ``u`` at ``1 - 2^-49``. A tighter cap is not safe: the rounding
      in ``u = y * (r / ||y||)`` and in ``u @ u`` can exceed a margin of a few ulps.
    * Both branches of ``inverse`` are evaluated, so each is fed an input that keeps it
      finite, with finite derivatives, when it is not selected. Otherwise reverse-mode
      derivatives would be ``0 * inf = NaN`` at ``y = 0``.
    * ``grad`` and ``inv_hessian`` floor ``1 - ||u||^2`` at the smallest normal float64,
      so they stay finite even if ``u`` reaches the sphere.
    """

    _R_MAX = 1.0 - 2.0**-49
    _N_DIRECT = 1e7
    _S2_MIN = float(np.finfo(np.float64).tiny)

    @staticmethod
    def _norm(y):
        """``||y||`` without overflow; finite derivatives for ``y != 0``."""
        m = jnp.max(jnp.abs(y))
        safe_m = jnp.where(m > 0, m, 1.0)
        return safe_m * jnp.sqrt(jnp.sum((y / safe_m) ** 2))

    def _s(self, u):
        return jnp.sqrt(jnp.maximum(1.0 - u @ u, self._S2_MIN))

    def grad(self, u):
        return u / self._s(u)

    def inverse(self, y, u_ref=None):
        large = self._norm(jax.lax.stop_gradient(y)) >= self._N_DIRECT
        y_small = jnp.where(large, jnp.zeros_like(y), y)
        direct = y_small / jnp.sqrt(1.0 + y_small @ y_small)
        y_large = jnp.where(large, y, jnp.full_like(y, self._N_DIRECT))
        n = self._norm(y_large)
        r = jnp.minimum(n / jnp.hypot(1.0, n), self._R_MAX)  # ||u|| = n / sqrt(1 + n^2)
        return jnp.where(large, y_large * (r / n), direct)

    def hessian(self, u):
        s = jnp.sqrt(1.0 - u @ u)
        return jnp.eye(u.shape[0]) / s + jnp.outer(u, u) / s**3

    def inv_hessian(self, u):
        return self._s(u) * (jnp.eye(u.shape[0]) - jnp.outer(u, u))

    def contains(self, u):
        return u @ u < 1.0

    def project(self, u):
        return u / jnp.maximum(1.0, jnp.linalg.norm(u))


# --------------------------------------------------------------------------------------
# Polytope (Sec. V-a): quadratic plus log-barrier DGF, numerical inverse
# --------------------------------------------------------------------------------------


_TAU = 0.995  # largest fraction of a slack that one step may use up (one-sided polytope map)


@dataclass(frozen=True, eq=False)
class PolytopeLogBarrier(MirrorMap):
    r"""DGF of the polytope ``G u < h``: a quadratic plus a weighted log barrier (Sec. V-a).

    ``psi(u) = 0.5 u^T u - weight * sum_j log s_j(u)`` with slack ``s(u) = h - G u``:

    * mirror map ``y = u + weight * G^T (1 / s(u))``
    * ``Psi(u) = I + weight * G^T diag(1 / s^2) G``
    * inverse ``grad psi*(y)``: no closed form; it is the minimizer of the strictly convex
      ``phi(u) = psi(u) - y^T u``, found with a damped Newton method in ``u`` from the warm
      start ``u_ref``.

    A small ``weight`` makes the map close to the identity in the interior, where the solver
    then takes nearly undamped steps, and confines the barrier to within about
    ``sqrt(weight)`` of a facet. The solver's damping ``lambda`` then has to fall far less
    before a control can reach a facet.

    Newton's direction is ``d = -Psi(u)^{-1} (grad psi(u) - y)``, and the step is
    ``u + d / (1 + rho)`` with ``rho = ||diag(1 / s) G d||``, the length of ``d`` in the
    barrier's local norm. Since ``rho / (1 + rho) < 1`` the step stays strictly inside the
    polytope (the Dikin ellipsoid), and because the barrier is self-concordant ``phi``
    cannot increase along it. Far from the facets ``rho`` is small and the step is the full
    Newton step; convergence is quadratic. The iteration stops once the Newton decrement
    ``sqrt(d^T Psi(u) d) <= newton_tol`` or after ``newton_max_steps`` steps. Every iterate
    is strictly feasible, also if the iteration stops early. Use :meth:`inverse_residual`
    to check a result.

    One-sided variant (the solver's default): the barrier keeps only the facets
    a control is heading towards (``G d > 0`` for its direction ``d``),
    ``psi(u) = 0.5 u^T u - weight * sum_{j ahead} log s_j(u)``. A control next to a facet that heads
    back into the polytope is then not frozen. The facets left out are not guarded by the
    map, so ``one_sided_inverse`` shortens a step that would use up more than 99.5 % of the
    slack of one of them.
    """

    G: jax.Array
    h: jax.Array
    weight: float = 1.0
    newton_tol: float = 1e-11
    newton_max_steps: int = 128
    has_one_sided = True

    def __post_init__(self):
        if not self.weight > 0:
            raise ValueError(f"weight must be positive, got {self.weight}")

    def slack(self, u):
        return self.h - self.G @ u

    def psi(self, u):
        s = self.slack(u)
        return 0.5 * u.T @ u - self.weight * jnp.sum(jnp.log(s))

    def grad(self, u):
        s = self.slack(u)
        return u + self.G.T @ (self.weight / s)

    def hessian(self, u):
        s = self.slack(u)
        W = self.weight / s**2
        return jnp.eye(u.shape[0]) + self.G.T @ (W[:, None] * self.G)

    def inv_hessian(self, u):
        H = self.hessian(u)
        return jsc.linalg.solve(H, jnp.eye(u.shape[0]))

    def inverse(self, y, u_ref):
        return self._newton_inverse(y, u_ref, self.weight * jnp.ones_like(self.h))

    def _newton_inverse(self, y, u_ref, weights):
        """Minimizer of ``0.5 u^T u - sum_j weights_j log s_j(u) - y^T u`` by the damped Newton
        method of the class docstring (differentiable in ``y`` through the root condition)."""
        G, h, eye = self.G, self.h, jnp.eye(u_ref.shape[0])

        def residual(u):
            return u + G.T @ (weights / (h - G @ u)) - y

        def direction(u):
            s = h - G @ u
            hessian = eye + G.T @ ((weights / s**2)[:, None] * G)
            gradient = u + G.T @ (weights / s) - y
            d = -jsc.linalg.solve(hessian, gradient, assume_a="pos")
            rho = jnp.linalg.norm(jnp.where(weights > 0, G @ d / s, 0.0))  # length of d in the barrier's norm
            return d, rho, jnp.sqrt(jnp.maximum(-gradient @ d, 0.0))

        def solve(_, u0):
            def keep_going(carry):
                _, _, _, decrement, step = carry
                return jnp.logical_and(decrement > self.newton_tol, step < self.newton_max_steps)

            def newton_step(carry):
                u, d, rho, _, step = carry
                u = u + d / (1.0 + rho)
                return (u, *direction(u), step + 1)

            return jax.lax.while_loop(keep_going, newton_step, (u0, *direction(u0), 0))[0]

        def tangent_solve(linear, b):
            return jnp.linalg.solve(jax.jacobian(linear)(b), b)

        return jax.lax.custom_root(residual, u_ref, solve, tangent_solve)

    def heading(self, u, direction):
        """Per facet: whether a control moving along ``direction`` heads towards it."""
        return self.G @ direction > 0.0

    def one_sided_grad(self, u, ahead):
        return u + self.G.T @ jnp.where(ahead, self.weight / self.slack(u), 0.0)

    def one_sided_inv_hessian(self, u, ahead):
        W = jnp.where(ahead, self.weight / self.slack(u) ** 2, 0.0)
        H = jnp.eye(u.shape[0]) + self.G.T @ (W[:, None] * self.G)
        return jsc.linalg.solve(H, jnp.eye(u.shape[0]))

    def one_sided_inverse(self, y, ahead, u_ref):
        slack = self.slack(u_ref)
        u = self._newton_inverse(y, u_ref, jnp.where(ahead, self.weight, 0.0))
        # fraction to the boundary of the facets the map does not guard
        used = slack - self.slack(u)
        limit = jnp.where(~ahead & (used > _TAU * slack), _TAU * slack / jnp.where(used > 0, used, 1.0), 1.0)
        return u_ref + jnp.min(limit) * (u - u_ref)

    def inverse_residual(self, y, u):
        """``max |grad psi(u) - y|`` and ``min s(u)``: a returned inverse is valid only if
        the first is small and the second is positive."""
        return jnp.max(jnp.abs(self.grad(u) - y)), jnp.min(self.slack(u))

    def contains(self, u):
        return jnp.all(self.slack(u) > 0)

    def project(self, u):
        """Closest feasible point among the projections of ``u`` onto the affine hulls of
        the faces (every set of at most ``m`` facets). Exact, but the number of sets grows
        combinatorially, so it is meant for small control dimensions."""
        G, h = self.G, self.h
        candidates = [u[None]]
        for k in range(1, min(G.shape) + 1):
            sets = jnp.array(list(itertools.combinations(range(G.shape[0]), k)))

            def onto(rows):
                A, b = G[rows], h[rows]
                return u - A.T @ (jnp.linalg.pinv(A @ A.T) @ (A @ u - b))

            candidates.append(jax.vmap(onto)(sets))
        candidates = jnp.concatenate(candidates)
        feasible = jnp.all(candidates @ G.T <= h + 1e-10 * (1.0 + jnp.abs(h)), axis=1)
        distance = jnp.where(feasible, jnp.sum((candidates - u) ** 2, axis=1), jnp.inf)
        return candidates[jnp.argmin(distance)]
