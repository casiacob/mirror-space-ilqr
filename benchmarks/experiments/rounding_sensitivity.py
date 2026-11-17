"""Mathematically identical rewrites of the backward pass on ``cartpole-compare``.

* stage derivatives computed with ``jax.vmap`` before the backward scan;
* one Cholesky factorization of ``Qhat_yy`` per stage instead of two LU solves;
* the 10 closed loops batched with ``jax.vmap``.

Usage: ``python benchmarks/experiments/rounding_sensitivity.py`` (~1 min).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from common import report, run_ms  # sets up sys.path and float64; must come first

import jax
import jax.numpy as jnp
import jax.scipy as jsc
import numpy as np
from jax import grad, hessian

from cases import CASES, cartpole_ms_solver
from msilqr import MirrorILQR
from msilqr.msilqr import mirror_derivatives
from msilqr.mpc import _mirror_mpc_fn


@dataclass(frozen=True, eq=False)
class VariantILQR(MirrorILQR):
    vmap_derivatives: bool = False
    cholesky: bool = False

    def backward_pass(self, states, controls, reg):
        p = self.problem
        Vx_final = grad(p.final_cost, 0)(states[-1])
        Vxx_final = hessian(p.final_cost, 0)(states[-1])
        expand = lambda x, u: mirror_derivatives(p, self.mirror_map, x, u)  # noqa: E731
        derivs = jax.vmap(expand)(states[:-1], controls) if self.vmap_derivatives else None

        def step(carry, inp):
            Vx, Vxx = carry
            d = inp if self.vmap_derivatives else expand(*inp)
            m = d.cy.shape[0]
            Qx = d.cx + d.fx.T @ Vx
            Qy = d.cy + d.fy.T @ Vx
            Qxx = d.cxx + d.fx.T @ Vxx @ d.fx
            Qyy = d.cyy + d.fy.T @ Vxx @ d.fy + reg * jnp.eye(m)
            Qxy = d.cxy + d.fx.T @ Vxx @ d.fy
            Qyx = Qxy.T
            positive_definite = jnp.all(jnp.linalg.eigvalsh(Qyy) > 0)
            if self.cholesky:
                factor = jsc.linalg.cho_factor(Qyy)
                k, K = -jsc.linalg.cho_solve(factor, Qy), -jsc.linalg.cho_solve(factor, Qyx)
            else:
                k, K = -jsc.linalg.solve(Qyy, Qy), -jsc.linalg.solve(Qyy, Qyx)
            Vx = Qx + Qxy @ k + K.T @ Qy + K.T @ Qyy @ k
            Vxx = Qxx + K.T @ Qyy @ K + K.T @ Qyx + Qxy @ K
            Vxx = 0.5 * (Vxx + Vxx.T)
            dV = Qy @ k + 0.5 * k.T @ Qyy @ k
            return (Vx, Vxx), (K, k, dV, positive_definite, Qy)

        xs = derivs if self.vmap_derivatives else (states[:-1], controls)
        _, (K, k, dV, positive_definite, Qy) = jax.lax.scan(step, (Vx_final, Vxx_final), xs, reverse=True)
        return K, k, jnp.sum(dV), jnp.all(positive_definite), jnp.max(jnp.abs(Qy))


def main():
    base = cartpole_ms_solver()
    report("cartpole-compare", "reference (MirrorILQR)", run_ms("cartpole-compare", base))
    for label, kw in (
        ("derivatives vmapped over the horizon", dict(vmap_derivatives=True)),
        ("Cholesky instead of two LU solves", dict(cholesky=True)),
        ("both", dict(vmap_derivatives=True, cholesky=True)),
    ):
        solver = VariantILQR(base.problem, base.mirror_map, base.options, **kw)
        report("cartpole-compare", label, run_ms("cartpole-compare", solver))

    # batch the 10 closed loops with vmap instead of a Python loop
    inp = CASES["cartpole-compare"].inputs()
    fn = _mirror_mpc_fn(base, inp.steps)
    loop = lambda: [fn(x0, u0) for x0, u0 in zip(inp.x0, inp.u_init, strict=True)]  # noqa: E731
    batched = jax.jit(jax.vmap(fn))
    jax.block_until_ready(loop()), jax.block_until_ready(batched(inp.x0, inp.u_init))  # compile both
    t = time.perf_counter()
    jax.block_until_ready(loop())
    t_loop = time.perf_counter() - t
    t = time.perf_counter()
    res = batched(inp.x0, inp.u_init)
    jax.block_until_ready(res)
    t_batch = time.perf_counter() - t
    new = {f"ms_{f}": np.asarray(getattr(res, f)) for f in ("states", "controls", "iterations", "status")}
    new["x0"] = np.asarray(inp.x0)
    report("cartpole-compare", "10 closed loops batched with vmap", new)
    print(f"{'':42s} warm wall time: Python loop {t_loop:.2f} s, vmap {t_batch:.2f} s")


if __name__ == "__main__":
    main()
