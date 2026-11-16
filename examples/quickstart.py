"""Quickstart: box-constrained cart-pole swing-up with mirror-space iLQR.

Run from the repository root after ``pip install -e .``::

    python examples/quickstart.py
"""

import jax

jax.config.update("jax_enable_x64", True)  # msilqr requires double precision

import jax.numpy as jnp  # noqa: E402

from msilqr import BoxEntropy, CartPole, MirrorILQR, SolverOptions, Status, mirror_mpc  # noqa: E402

# Problem: swing the pole up (theta = pi) with |force| <= 50 N.
system = CartPole()
problem = system.problem()

# Constraint set as a mirror map: the entropy DGF of the box (Table I of the paper).
solver = MirrorILQR(problem, BoxEntropy(-50.0, 50.0), SolverOptions(tol=1e-6, max_iter=500))

x0 = jnp.array([-0.01, 0.01, -0.01, 0.01])
u0 = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (100, 1))  # strictly feasible guess

# One open-loop solve over a 1 s horizon.
sol = solver.solve(x0, u0)
print(f"open loop: {Status(int(sol.status)).name} after {int(sol.iterations)} iterations, cost {float(sol.cost):.3f}")
print(f"           max |u| = {float(jnp.abs(sol.controls).max()):.3f} (bound 50)")

# Receding-horizon control for 4 s.
res = mirror_mpc(solver, x0, u0, num_steps=400)
angle_error = float(jnp.abs((res.states[-1, 1] % (2 * jnp.pi)) - jnp.pi))
print(f"MPC: {int(res.iterations.sum())} iterations over 400 steps, final angle error {angle_error:.2e} rad")
