# Mirror-space iLQR

JAX implementation of mirror-space iLQR, from the paper *Mirror-Space Constrained Optimal
Control* by Casian Iacob, Hany Abdulsamad and Simo Särkkä (in preparation).

The method handles convex control constraints by running iLQR in mirror coordinates. The
constraint set is built into this change of variables, so every iterate is feasible
without barriers, penalties or projections.

## Installation

Requires Python ≥ 3.12.

```bash
git clone https://github.com/casiacob/mirror-space-ilqr
cd mirror-space-ilqr
pip install -e .
```

## Quickstart

```python
import jax

jax.config.update("jax_enable_x64", True)  # msilqr requires double precision
import jax.numpy as jnp
from msilqr import BoxEntropy, CartPole, MirrorILQR, mirror_mpc

problem = CartPole().problem()
solver = MirrorILQR(problem, BoxEntropy(-50.0, 50.0))  # |u| <= 50

x0 = jnp.array([-0.01, 0.01, -0.01, 0.01])
u0 = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (100, 1))
sol = solver.solve(x0, u0)  # one finite-horizon solve
res = mirror_mpc(solver, x0, u0, num_steps=400)  # closed loop
```

For your own system, define `OptimalControlProblem(dynamics, stage_cost, final_cost)` and
pick a constraint set: `BoxEntropy`, `BoxLogBarrier`, `Positivity`, `EuclideanBall` or
`PolytopeLogBarrier`. The initial controls must lie strictly inside it.

## License

[MIT](LICENSE) © 2026 Casian Iacob and Hany Abdulsamad.

## Citation

```bibtex
@article{iacob2026mirror,
  title  = {Mirror-Space Constrained Optimal Control},
  author = {Iacob, Casian and Abdulsamad, Hany and S{\"a}rkk{\"a}, Simo},
  year   = {2026},
  note   = {In preparation}
}
```
