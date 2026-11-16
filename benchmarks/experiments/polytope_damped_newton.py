"""A damped Newton method for the polytope inverse map instead of the original solver.

``grad psi*(y) = argmin_u psi(u) - y^T u`` is a smooth, strictly convex problem, so a
damped Newton method with a feasibility-preserving backtracking line search is the
textbook choice. This script swaps it in for the original optimistix root finding on
``polytope-compare`` (README, "Performance follow-ups").

Usage: ``python benchmarks/experiments/polytope_damped_newton.py`` (~20 s).
"""

from __future__ import annotations

from dataclasses import dataclass

from common import report, run_ms  # sets up sys.path and float64; must come first

import jax
import jax.numpy as jnp

from cases import polytope_map, polytope_ms_solver
from msilqr import MirrorILQR, PolytopeLogBarrier


@dataclass(frozen=True, eq=False)
class DampedNewtonPolytope(PolytopeLogBarrier):
    iterations: int = 50

    def inverse(self, y, u_ref):
        G, h = self.G, self.h

        def objective(u):
            return self.psi(u) - y @ u

        def newton_step(_, u):
            s = h - G @ u
            g = u + G.T @ (1.0 / s) - y
            H = jnp.eye(u.shape[0]) + G.T @ ((1.0 / s**2)[:, None] * G)
            d = -jnp.linalg.solve(H, g)
            Gd = G @ d
            t_max = jnp.min(jnp.where(Gd > 0, s / Gd, jnp.inf))
            f0, slope = objective(u), g @ d

            def reject(t):
                un = u + t * d
                return (jnp.min(h - G @ un) <= 0) | (objective(un) > f0 + 1e-4 * t * slope)

            t = jax.lax.while_loop(lambda t: reject(t) & (t > 1e-12), lambda t: 0.5 * t, jnp.minimum(1.0, 0.99 * t_max))
            return u + t * d

        return jax.lax.fori_loop(0, self.iterations, newton_step, u_ref)


def main():
    report("polytope-compare", "original optimistix Newton", run_ms("polytope-compare", polytope_ms_solver()))
    base = polytope_map()
    solver = MirrorILQR(polytope_ms_solver().problem, DampedNewtonPolytope(base.G, base.h))
    report("polytope-compare", "damped Newton (50 iterations)", run_ms("polytope-compare", solver))


if __name__ == "__main__":
    main()
