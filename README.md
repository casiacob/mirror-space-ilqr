# Mirror-space iLQR

Code for **"Mirror-Space Constrained Optimal Control"** (Casian Iacob, Hany Abdulsamad,
Simo Särkkä; IEEE letter, in preparation).

Mirror-space iLQR solves nonlinear optimal control problems with convex control
constraints `u ∈ U`. The gradient of a Legendre-type function `ψ` whose domain is `U` maps
the constrained controls to unconstrained *mirror controls* `y = ∇ψ(u)`. iLQR runs in
these coordinates, and every iterate `u = ∇ψ*(y)` is strictly feasible by construction,
without barrier parameters or penalty schedules. All derivatives the backward pass needs
come from `ψ` itself (Lemma 3), never from the inverse map.

The repository also contains the interior-point DDP baseline used in the paper's
comparison (Pavlov, Shames and Manzie, 2021), the paper's two experiments, and a
regression harness that pins the results to the original implementation.

## Installation

Requires Python ≥ 3.11 and JAX (CPU is enough).

```bash
pip install -e ".[benchmarks]"      # library + optimistix (polytope map) + matplotlib
pip install -e ".[dev]"             # + pytest and ruff
```

To reproduce the reference results bit for bit, use the pinned environment the golden
outputs were captured with:

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
sol = solver.solve(problem.rollout(x0, u0), u0)  # one finite-horizon solve
print(sol.status, sol.iterations, sol.cost)

res = mirror_mpc(solver, x0, u0, num_steps=400)  # receding-horizon closed loop
```

See [`examples/quickstart.py`](examples/quickstart.py).

## API

| Object | What it is |
|---|---|
| `OptimalControlProblem(dynamics, stage_cost, final_cost)` | Problem (1) without the control constraint; `.total_cost`, `.rollout` |
| `MirrorMap` | Interface: `grad(u)` = ∇ψ, `inverse(y, u_ref)` = ∇ψ*, `inv_hessian(u)` = Ψ⁻¹, `hessian(u)` = Ψ |
| `MirrorILQR(problem, mirror_map, SolverOptions())` | The paper's method; `.solve(states, controls) -> Solution` |
| `SolverOptions` | `tol=1e-6`, `max_iter=1000`, `reg_init=1e-2` (λ₀), `reg_factor_init=2`, λ clip `[1e-16, 1e16]` |
| `Solution` | `states, controls, cost, iterations, reg, error, status` |
| `Status` | `CONVERGED`, `MAX_ITER`, `NONFINITE` |
| `InteriorPointDDP(problem, constraints, IPDDPOptions())` | Baseline for `g(x, u) ≤ 0`; `.solve(x0, controls, slacks, duals)` |
| `mirror_mpc`, `ipddp_mpc` | Jitted receding-horizon loops; return `MPCResult(states, controls, iterations, status, plan_costs)` with the **applied** controls |
| `CartPole`, `DoubleIntegrator` | The systems of Sec. V (`.problem()`) |

### Mirror maps (Table I and Sec. V)

| Class | Set `int(U)` | ψ(u) | Map ∇ψ(u) | Inverse ∇ψ*(y) |
|---|---|---|---|---|
| `BoxEntropy(a, b, scale=1)` | a < u < b | c·[(b−u)log(b−u) + (u−a)log(u−a)] | c·log((u−a)/(b−u)) | μ + ρ tanh(y / 2c) |
| `BoxLogBarrier(a, b)` | a < u < b | −log(b−u) − log(u−a) | 1/(b−u) − 1/(u−a) | μ + ρ tanh(½ arcsinh(ρy)) |
| `Positivity()` | u > 0 | Σ (uⁱ log uⁱ − uⁱ) | log u | eʸ |
| `EuclideanBall()` | ‖u‖₂ < 1 | −√(1−‖u‖²) | u / √(1−‖u‖²) | y / √(1+‖y‖²) |
| `PolytopeLogBarrier(G, h)` | Gu < h | ½uᵀu − Σ log(h−Gu)ⱼ | u + Gᵀ(1/s) | Newton root finding |

μ = (a+b)/2, ρ = (b−a)/2, s = h − Gu. `scale = c = 1` is exactly Table I.

### The algorithm, equation by equation

| Step | Paper | Code (`msilqr/mirror_ilqr.py`) |
|---|---|---|
| Derivatives under the mirror map | Lemma 3, Eqs. (7)–(10) | `_derivatives`: `Q_y = Ψ⁻¹Q_u`, `Q_xy = Q_xuΨ⁻¹` |
| Gauss–Newton mirror Hessian | Eq. (12) | `Q_yy = Ψ⁻¹Q_uuΨ⁻¹` (the third-order term of (11) is dropped) |
| Levenberg–Marquardt damping | Sec. III-C, [37, Eq. 2.21] | `Q̂_yy = Q_yy + λI`; λ updated with the gain ratio |
| Gains and value recursion | Eq. (13) | `backward_pass` (uses `Q̂_yy` throughout, as in the paper) |
| Forward rollout | Eq. (14) | `forward_pass`: `y = ∇ψ(ū) + k + KΔx`, `u = ∇ψ*(y)` |
| Stopping criterion | Sec. V | `max_t |Q_y,t| < tol` (or `max_iter`) |

## Reproducing the paper

```bash
python benchmarks/run.py --out results                    # all four experiments (~35 s on CPU)
python benchmarks/figures.py --results results --out figures   # Figs. 2-4 as PDF/PNG
python benchmarks/regression.py --new results --bitwise   # compare with the original code
```

| Case | Paper | Original script | Steps × runs |
|---|---|---|---|
| `polytope` | Fig. 2 | `examples/wiener-polytope.py` | 300 × 1 |
| `cartpole` | Fig. 3 | `examples/cartpole.py` | 400 × 1 |
| `polytope-compare` | Fig. 4 (left) | `examples/wiener-polytope-mm-v-ip.py` | 300 × 10 |
| `cartpole-compare` | Fig. 4 (right) | `examples/cartpole-mm-v-ip.py` | 400 × 10 |

Results (identical to the original implementation, commit `effb14c`):

| Case | MS iterations: total (mean/step, max) | IP iterations: total (mean/step, max) | Closed-loop cost MS / IP |
|---|---|---|---|
| `cartpole` | 1,978 (4.9, 39) | – | 5985.5 / – |
| `cartpole-compare` | 23,381 (5.8, 102) | 74,083 (18.5, 43) | 6656.8 / 6094.8 |
| `polytope` | 2,746 (9.2, 34) | – | 606.5 / – |
| `polytope-compare` | 27,642 (9.2, 36) | 52,745 (17.6, 38) | 701.0 / 698.2 |

The mirror-space method needs 2–3× fewer iterations per MPC step. Its closed-loop cost is
higher than the interior-point baseline's (+9.2 % on the cart-pole, +0.4 % on the double
integrator).

## Regression workflow

The revamp kept the solvers' arithmetic identical to the original code and changed
everything around it. Three pieces enforce that:

* `benchmarks/legacy/capture.py` exports the original commit with `git archive`, runs its
  four example scripts unmodified and stores every array they produce in
  `benchmarks/legacy/golden/`.
* `benchmarks/regression.py` re-implements the experiments' dynamics, costs and constraints
  in plain NumPy (independent of `msilqr`). Per case and method it checks closed-loop
  consistency with the dynamics, total and maximum iterations per MPC step, closed-loop
  cost, constraint satisfaction and goal error against the golden run; `--bitwise` also
  requires exact equality.
* `tests/test_regression.py` runs both (`pytest -m slow`).

**Why bitwise?** The MPC closed loops, the cart-pole especially, are chaotically
sensitive to rounding. Mathematically identical rewrites (a Cholesky solve instead of LU,
derivatives vmapped over the horizon) change iteration counts by hundreds, and one even
leaves a solve stalled. Only exact equality shows that a refactor changed nothing.

XLA compiles for the host CPU, so golden outputs captured on another machine can differ in
the last bits. Regenerate them locally with `MSILQR_RECAPTURE_GOLDEN=1 pytest -m slow`, or
`python benchmarks/legacy/capture.py --out <dir>` followed by
`regression.py --golden <dir>`. CI does this on every run.

## Notes for the authors: paper vs. code

These points come from an audit of the original code against the paper
([`docs/AUDIT.md`](docs/AUDIT.md)). Numbers are from `benchmarks/run.py` with the variant
flags shown.

1. **The cart-pole uses half of Table I's box map.** Sec. V-b says the cart-pole uses
   "the entropy DGF for box constraints from Table I". The scripts use
   `y = ½ log((u−a)/(b−u))`, `u = ρ tanh(y)` (Table I with `scale=0.5`). Under the
   `λI` damping in mirror coordinates the scale changes the iterates. With the exact
   Table I map (`--box-scale 1`):
   * `cartpole` (Fig. 3): 3,559 iterations (+80 %), cost 7576.9 (+27 %), and a final goal
     error of 2.0 instead of 0.005: the swing-up does not settle within 4 s.
   * `cartpole-compare`: 24,402 iterations (+4.4 %), but max 76 per step (was 102) and
     cost −3.2 %.

   The benchmarks keep `scale=0.5`. Either the paper text or the experiment should change.
2. **The interior-point baseline exits early.** IPDDP stops when the barrier
   subproblem's residual is below `tol`, without driving the barrier parameter μ to `tol`.
   It returns central-path points, and its iteration counts in Fig. 4 are understated.
   With `--ip-barrier-tol 1e-6`: `cartpole-compare` IP 88,565 iterations (+19.5 %),
   `polytope-compare` IP 60,641 (+15 %, with one solve stalled at the 1,000-iteration cap).
3. **"The same tolerance" is not the same accuracy.** MS stops on `max|Q_y|` in mirror
   coordinates. That quantity is `Ψ⁻¹Q_u`, so it vanishes at saturated controls whatever the
   sign of the control-space gradient. IP stops on `max(|Q_u|, |r_p|, |r_d|)`.
4. **The baseline is cold-started.** IP restarts slacks (0.1), duals (0.01), μ and λ at
   every MPC step, while MS warm-starts its controls. Warm-starting only slacks and duals
   makes IPDDP worse (every second solve stalls), because μ is reset. A fair warm start
   has to carry μ as well. The step acceptance also ignores primal infeasibility, and the
   fraction-to-boundary rule is applied per stage.
5. **The double-integrator comparison does not start from the paper's x₀.** The 10 initial
   states of `polytope-compare` are sampled around 0 (std 0.1 / 0.005), not around the
   paper's `x0 = [0.2, −0.1, −0.1, 0.1]`. Fig. 4 is a trailing 10-step moving average,
   which the caption does not mention. Other settings the paper leaves out: Euler steps of
   0.02 s and 0.01 s, horizons of 15 and 100, λ₀ = 10⁻², `tol` = 10⁻⁶.
6. **The original IP scripts logged the wrong controls.** They recorded the *next planned*
   control instead of the applied one (off by up to 23 N on the cart-pole). Any plot or
   cost computed from those traces is wrong; the new MPC loops record applied controls.
7. **The original IPDDP crashes on current JAX** (`jnp.clip(a_min=...)`). Fig. 4 cannot be
   reproduced from `effb14c` with a fresh install.
8. **"Strictly feasible" holds in exact arithmetic only.** In float64, `tanh` rounds to 1
   for `|y|/2c ≳ 19`, so controls land exactly on the bound: 290 of the 4,000 applied
   controls in `cartpole-compare`. There `Ψ⁻¹ = 0` and the control is frozen until the MPC
   shift moves it. The obvious fix, carrying `y` instead of recomputing `ȳ = ∇ψ(ū)`, is
   much **worse** with Euclidean damping: +40 % iterations and +28 % cost on the
   cart-pole. Recomputing `ȳ` implicitly caps `|ȳ|` at about 18, which keeps saturated
   controls close to the interior in `y`. The library keeps the paper's `ȳ = ∇ψ(ū)` and
   only replaces `±∞` by `±40c`, which is bitwise neutral.
9. **The cart-pole cost is not C² at θ = 0** (the hanging start), because the angle is
   wrapped with `θ mod 2π`. Sec. II assumes twice-differentiable costs.

## Changes relative to the original code

* New typed API (`OptimalControlProblem`, `MirrorMap`, `MirrorILQR`, `InteriorPointDDP`,
  MPC loops) with docstrings mapped to the paper; the four duplicated scripts are replaced
  by `msilqr/systems.py` and `benchmarks/cases.py`.
* Both solvers have an iteration cap and a convergence status; the original loops could
  run forever.
* A float64 guard: in float32 the original cart-pole solve never converged.
* All Table I maps plus the polytope map. The box map includes μ (the original was wrong
  for asymmetric boxes) and never returns `±∞`.
* IPDDP: works on current JAX; `eigvalsh` instead of the complex `eigvals`; the feedback
  gain of state constraints uses Σ = S⁻¹Z instead of S⁻¹ (a bug that does not affect the
  benchmarks, which constrain controls only); optional `barrier_tol`.
* MPC loops record the applied controls.
* `pyproject.toml` with extras, a pinned lock file, 28 tests (map identities, Lemma 3
  against autodiff, LQ problem against a reference QP solution, Proposition 5, the
  regression), CI and figure generation.

Every change is bitwise-neutral on the four benchmarks.

## Performance follow-ups (deliberately not done)

Correctness came first. These would speed things up but change the floating-point
results, so each needs a reviewed re-baseline:

| Change | Measured effect on the benchmarks |
|---|---|
| Vectorize stage derivatives over the horizon | `cartpole-compare` +89 iterations, max 111 (was 102) |
| One Cholesky factorization per stage instead of two LU solves | one cart-pole solve stalls at the iteration cap |
| `vmap` the 10 MPC runs | not bitwise, one stalled solve, and slower (a batched `while_loop` waits for the slowest run) |
| Damped Newton for the polytope inverse map (instead of optimistix) | `polytope-compare` +1.6 % iterations, max 45 (was 36) |

The runner already compiles each closed loop once and reuses it for all initial states.

## Development

```bash
pytest -m "not slow"     # unit tests (~10 s)
pytest -m slow           # bitwise regression against the original code (~40 s)
ruff check . && ruff format --check .
```

## Citation

```bibtex
@article{iacob2026mirror,
  title  = {Mirror-Space Constrained Optimal Control},
  author = {Iacob, Casian and Abdulsamad, Hany and S{\"a}rkk{\"a}, Simo},
  year   = {2026},
  note   = {In preparation}
}
```

## License

No license has been chosen yet; the authors need to pick one before a public release.
