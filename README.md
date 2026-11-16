# Mirror-space iLQR

JAX implementation of **mirror-space iLQR**, from the paper *Mirror-Space Constrained
Optimal Control* by Casian Iacob, Hany Abdulsamad and Simo Särkkä (in preparation).

Mirror-space iLQR solves nonlinear optimal control problems with **convex control
constraints** `u ∈ U`. It takes a Legendre-type function `ψ` whose domain is `U` and maps
the constrained controls to unconstrained *mirror controls* `y = ∇ψ(u)`, then runs iLQR in
those coordinates:

1. The backward pass expands the Q-function in `y`. All derivatives come from `ψ` itself
   (`Q_y = Ψ⁻¹Q_u`, `Q_yy ≈ Ψ⁻¹Q_uuΨ⁻¹` with `Ψ = ∇²ψ`), never from the inverse map
   (Lemma 3, Prop. 4).
2. Levenberg–Marquardt damping `Q_yy + λI` is adapted with a gain-ratio rule.
3. The forward pass steps in `y` and maps back with `u = ∇ψ*(y)`, so **every iterate is
   feasible by construction**: no barrier parameter, penalty weights or projection.

For box constraints and linear-quadratic problems the iteration is a Levenberg–Marquardt
method with a global convergence guarantee (Sec. IV of the paper).

**What is in the repository**

* `msilqr`, a small library: problem definition, the mirror maps of the paper's Table I
  plus a polytope map, the mirror-space solver, an interior-point DDP baseline (Pavlov,
  Shames and Manzie, 2021), and model-predictive-control (MPC) loops.
* The paper's experiments (a double integrator with polytopic constraints and a
  cart-pole swing-up with a force limit), with a script that regenerates Figs. 2-4.
* A regression harness that checks the results bit for bit against the original
  implementation.

## Installation

Requires Python ≥ 3.12 and JAX ≥ 0.12; a CPU is enough.

```bash
git clone https://github.com/casiacob/mirror-space-ilqr
cd mirror-space-ilqr
pip install -e ".[benchmarks]"      # library + optimistix (polytope map) + matplotlib
pip install -e ".[dev]"             # everything above + pytest and ruff
```

The reference results were produced with Python 3.14 and JAX 0.12.0.1. To reproduce them
bit for bit, install that pinned environment:

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps
```

## Quickstart

```python
import jax

jax.config.update("jax_enable_x64", True)  # msilqr requires double precision
import jax.numpy as jnp
from msilqr import BoxEntropy, CartPole, MirrorILQR, mirror_mpc

problem = CartPole().problem()  # dynamics + costs
solver = MirrorILQR(problem, BoxEntropy(-50.0, 50.0))  # |u| <= 50 via Table I's entropy map

x0 = jnp.array([-0.01, 0.01, -0.01, 0.01])
u0 = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (100, 1))
sol = solver.solve(x0, u0)  # one finite-horizon solve
print(sol.status, sol.iterations, sol.cost)

res = mirror_mpc(solver, x0, u0, num_steps=400)  # receding-horizon closed loop
```

See also [`examples/quickstart.py`](examples/quickstart.py).

### Your own problem

A problem is a discrete-time model plus a stage and a final cost; the constraint set is
a mirror map. For example, a planar double integrator whose accelerations must stay in the
diamond `|u₁| + |u₂| ≤ 1`:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from msilqr import MirrorILQR, OptimalControlProblem, PolytopeLogBarrier, euler


def ode(x, u):  # x = [position (2), velocity (2)], u = acceleration (2)
    return jnp.concatenate([x[2:], u])


problem = OptimalControlProblem(
    dynamics=euler(ode, dt=0.05),
    stage_cost=lambda x, u: x @ x + 0.1 * u @ u,
    final_cost=lambda x: 10.0 * x @ x,
)
G = jnp.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])  # G u < h
diamond = PolytopeLogBarrier(G, h=jnp.ones(4))

sol = MirrorILQR(problem, diamond).solve(jnp.array([1.0, -1.0, 0.0, 0.0]), jnp.zeros((40, 2)))
```

The initial controls must lie strictly inside the constraint set; `solve` checks this
(`validate=False` skips the check, as the MPC loops do for their shifted warm starts).

## Mirror maps

| Class | Set `int(U)` | ψ(u) | Map ∇ψ(u) | Inverse ∇ψ*(y) |
|---|---|---|---|---|
| `BoxEntropy(a, b, scale=1)` | a < u < b | c·[(b−u)log(b−u) + (u−a)log(u−a)] | c·log((u−a)/(b−u)) | μ + ρ tanh(y / 2c) |
| `BoxLogBarrier(a, b)` | a < u < b | −log(b−u) − log(u−a) | 1/(b−u) − 1/(u−a) | μ + ρ tanh(½ arcsinh(ρy)) |
| `Positivity()` | u > 0 | Σ (uⁱ log uⁱ − uⁱ) | log u | eʸ |
| `EuclideanBall()` | ‖u‖₂ < 1 | −√(1−‖u‖²) | u / √(1−‖u‖²) | y / √(1+‖y‖²) |
| `PolytopeLogBarrier(G, h)` | Gu < h | ½uᵀu − Σ log(h−Gu)ⱼ | u + Gᵀ(1/s) | Newton root finding |

Here μ = (a+b)/2, ρ = (b−a)/2 and s = h − Gu. The first four rows are Table I of the paper
(`scale = c = 1`); the polytope map is the one of Sec. V-a. Bounds `a`, `b` can be scalars
or vectors.

New constraint sets subclass `MirrorMap`. A subclass implements five methods: `grad(u)`
(∇ψ), `inverse(y, u_ref)` (∇ψ*), `hessian(u)` (Ψ), `inv_hessian(u)` (Ψ⁻¹) and
`contains(u)` (the test u ∈ int(U)).

In float64 the closed-form inverses saturate (`tanh` rounds to 1, `exp` underflows), so a
control can land exactly on the boundary of `U`. The box, positivity and ball maps still
return a point of `U` with a finite `grad` there; such a control stays frozen until an MPC
warm start moves it. The polytope map's Newton inverse can fail; it then returns NaN and
the solver rejects that step.

## API

| Object | Purpose |
|---|---|
| `OptimalControlProblem(dynamics, stage_cost, final_cost)` | The problem without its control constraint; provides `.total_cost` and `.rollout` |
| `MirrorILQR(problem, mirror_map, SolverOptions())` | The paper's method: `.solve(x0, controls) -> Solution` |
| `SolverOptions` | `tol=1e-6`, `max_iter=1000`, `reg_init=1e-2` (λ₀), `reg_factor_init=2`, `reg_min=1e-16`, `reg_max=1e16` |
| `Solution` | Fields `states, controls, cost, iterations, reg, error, status` |
| `Status` | `CONVERGED`, `MAX_ITER`, `NONFINITE` |
| `InteriorPointDDP(problem, constraints, IPDDPOptions())` | Baseline for `g(x, u) ≤ 0`: `.solve(x0, controls, slacks, duals)` |
| `mirror_mpc`, `ipddp_mpc` | Receding-horizon closed loops; return `MPCResult(states, controls, iterations, status, plan_costs)` with the applied controls |
| `CartPole`, `DoubleIntegrator`, `euler`, `rollout` | The paper's systems and helpers |

Each solver compiles once per instance and input shape.

### Code map to the paper

| Step | Paper | `msilqr/mirror_ilqr.py` |
|---|---|---|
| Stage derivatives in mirror coordinates | Lemma 3, Eqs. (7)–(10) | `mirror_derivatives` |
| Q-function expansion, Gauss–Newton `Q_yy` | Prop. 4, Eq. (12) | `q_expansion` |
| Damping `Q̂_yy = Q_yy + λI`, gains | Sec. III-C | `q_expansion`, `feedback_gains` |
| Value recursion and predicted decrease | Eq. (13) | `value_update` |
| λ update (gain ratio) | [37, Eq. 2.21] | `msilqr/_common.py: gain_ratio_update` |
| Forward rollout | Eq. (14) | `MirrorILQR.forward_pass` |
| Stopping test `max_t ‖Q_y,t‖ < tol` | Sec. V | `MirrorILQR._solve` |

## Reproducing the paper

```bash
python benchmarks/run.py --out results                         # the four experiments (~35 s on a CPU)
python benchmarks/figures.py --results results --out figures   # Figs. 2-4 as PDF and PNG
python benchmarks/regression.py --new results --bitwise        # compare with the original implementation
```

| Case | Paper | MPC steps × initial states | Mirror-space iterations: total (per step, max) | Interior-point iterations: total (per step, max) |
|---|---|---|---|---|
| `polytope` | Fig. 2 | 300 × 1 | 2,746 (9.2, 34) | – |
| `cartpole` | Fig. 3 | 400 × 1 | 1,978 (4.9, 39) | – |
| `polytope-compare` | Fig. 4, left | 300 × 10 | 27,642 (9.2, 36) | 52,745 (17.6, 38) |
| `cartpole-compare` | Fig. 4, right | 400 × 10 | 23,381 (5.8, 102) | 74,083 (18.5, 43) |

Per MPC step, the mirror-space method needs 1.9× (double integrator) and 3.2× (cart-pole)
fewer iterations than the interior-point baseline.

These numbers are identical to the original implementation. Where the experiments differ
from the paper's text, and how the comparison with the baseline should be read, is
described in [`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md). Examples: the cart-pole box map
uses half of Table I's scaling, and the baseline exits before its barrier parameter
reaches the tolerance.

## Testing

```bash
pytest -m "not slow"     # unit tests (~30 s)
pytest -m slow           # bitwise regression against the original implementation (~1-2 min)
ruff check . && ruff format --check .
```

The unit tests check:
* the Table I identities and Lemma 3 against automatic differentiation;
* every mirror map on constrained linear-quadratic problems against independently
  computed optima;
* Proposition 5;
* the interior-point gains against a direct KKT solve.

The slow test regenerates reference outputs from the original commit and requires the new
code to match them exactly. Bit-for-bit equality is required because the MPC closed
loops are chaotically sensitive to rounding. [`docs/REGRESSION.md`](docs/REGRESSION.md)
explains why, describes the harness, and lists performance changes deferred for that
reason.

## Repository layout

```
msilqr/
  problem.py        OptimalControlProblem, rollout, Euler discretization
  mirror_maps.py    MirrorMap interface, Table I maps, polytope map
  mirror_ilqr.py    mirror-space iLQR (the paper's method)
  ipddp.py          interior-point DDP baseline
  mpc.py            receding-horizon closed loops
  systems.py        cart-pole and double integrator of Sec. V
  _common.py        options, gain-ratio rule and stage expansion shared by both solvers
benchmarks/
  cases.py, run.py  the paper's experiments
  figures.py        Figs. 2-4
  regression.py     regression gate against the original implementation
  legacy/           capture tool and reference outputs of the original code
  experiments/      scripts behind the numbers in docs/
examples/           quickstart
tests/              unit, solver and regression tests
docs/               paper notes, regression workflow, change log, audit
```

## License

[MIT](LICENSE) © 2026 Casian Iacob and Hany Abdulsamad.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{iacob2026mirror,
  title  = {Mirror-Space Constrained Optimal Control},
  author = {Iacob, Casian and Abdulsamad, Hany and S{\"a}rkk{\"a}, Simo},
  year   = {2026},
  note   = {In preparation}
}
```
