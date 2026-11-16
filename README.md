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
sol = solver.solve(x0, u0)  # one finite-horizon solve
print(sol.status, sol.iterations, sol.cost)

res = mirror_mpc(solver, x0, u0, num_steps=400)  # receding-horizon closed loop
```

See [`examples/quickstart.py`](examples/quickstart.py).

## API

| Object | What it is |
|---|---|
| `OptimalControlProblem(dynamics, stage_cost, final_cost)` | Problem (1) without the control constraint; `.total_cost`, `.rollout` |
| `MirrorMap` | Interface: `grad(u)` = ∇ψ, `inverse(y, u_ref)` = ∇ψ*, `inv_hessian(u)` = Ψ⁻¹, `hessian(u)` = Ψ, `contains(u)` = u ∈ int(U) |
| `MirrorILQR(problem, mirror_map, SolverOptions())` | The paper's method; `.solve(x0, controls) -> Solution`. The initial controls must be strictly inside `U` (checked) |
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

In exact arithmetic every inverse lands in `int(U)`. In float64 the closed forms saturate
(`tanh` rounds to 1, `exp` underflows), so every map guarantees that `inverse` returns a
point of the closed set `U` and that `grad` stays finite there. Without that guard, an
asymmetric box such as (−50, 49.9) produced controls one rounding error outside the box.
The ball and positivity inverses cannot leave `int(U)` for any finite `y`.

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

The mirror-space method needs 1.9× (double integrator) and 3.2× (cart-pole) fewer
iterations per MPC step. Its closed-loop cost is
higher than the interior-point baseline's (+9.2 % on the cart-pole, +0.4 % on the double
integrator).

## Regression workflow

The revamp kept the solvers' arithmetic identical to the original code and changed
everything around it. Three pieces enforce that:

* `benchmarks/legacy/capture.py` exports the original commit with `git archive`, runs its
  four example scripts unmodified and stores every array they produce in
  `benchmarks/legacy/golden/`. `wiener-polytope.py` does not keep its iteration counts, so
  the tool re-runs that script's MPC loop with the original functions to record them. The
  re-run is first checked to reproduce the script's closed loop bitwise.
* `benchmarks/regression.py` re-implements the experiments' dynamics, costs and constraints
  in plain NumPy (independent of `msilqr`). Per case and method it checks that every MPC
  solve converged, closed-loop consistency with the dynamics, total and maximum iterations
  per MPC step, closed-loop cost, constraint satisfaction and goal error against the
  golden run. `--bitwise` also requires exact equality.
* `tests/test_regression.py` runs both (`pytest -m slow`); `tests/test_gate.py` checks
  that the gate catches damaged results.

**Why bitwise?** The MPC closed loops, the cart-pole especially, are chaotically
sensitive to rounding. Mathematically identical rewrites (a Cholesky solve instead of LU,
derivatives vmapped over the horizon) change `cartpole-compare` from 23,381 to 23,470
or 24,422 iterations, and the Cholesky variant leaves one solve stalled at the iteration
cap. Only exact equality shows that a refactor changed nothing.

XLA compiles for the host CPU, so outputs are only comparable bit for bit on the same
machine. `pytest -m slow` therefore regenerates the golden outputs from `effb14c` on the
current machine before comparing (it needs the git history; CI does the same).
`MSILQR_USE_COMMITTED_GOLDEN=1` compares with the committed files instead, which is faster
and valid on the machine that captured them. By hand:
`python benchmarks/legacy/capture.py --out <dir>`, then `regression.py --golden <dir>`.

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
   `polytope-compare` IP 60,641 (+15 %, with two solves stalled at the 1,000-iteration cap).
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
   shift moves it. The obvious fix is to carry `y` instead of recomputing
   `ȳ = ∇ψ(ū)` (`benchmarks/experiments/carry_mirror_controls.py`). It gives the same
   iterations and cost on `cartpole`, but on `cartpole-compare` it needs 24,458 iterations
   (+4.6 %) and one solve stalls at the iteration cap. Recomputing `ȳ` implicitly caps
   `|ȳ|` at about 18, which keeps saturated controls close to the interior in `y`. The
   library keeps the paper's `ȳ = ∇ψ(ū)` and only replaces `±∞` by `±40c`, which is
   bitwise neutral.
9. **The cart-pole cost is not even C¹ at θ = 0** (the hanging start): the angle is
   wrapped with `θ mod 2π`, so the gradient jumps by ±10π there. Sec. II assumes
   twice-differentiable costs.
10. **Euclidean-ball constraints can converge slowly.** On a small 2-D LQ problem whose
    optimal controls slide along the boundary of the unit ball (`tests/test_solver.py`),
    the solver's stopping measure falls roughly like 1/k: 5.5·10⁻⁴, 9·10⁻⁵ and 1.1·10⁻⁵
    after 1,000, 5,000 and 20,000 iterations. When the active controls point in a fixed
    direction it converges in 36 iterations. The paper's convergence analysis covers
    boxes only; under the `λI` damping, steps along the curved boundary shrink with the
    radial `Ψ⁻¹`.

## Changes relative to the original code

* New typed API (`OptimalControlProblem`, `MirrorMap`, `MirrorILQR`, `InteriorPointDDP`,
  MPC loops) with docstrings mapped to the paper; the four duplicated scripts are replaced
  by `msilqr/systems.py` and `benchmarks/cases.py`.
* Both solvers have an iteration cap and a convergence status; the original loops could
  run forever.
* A float64 guard: in float32 the original cart-pole solve never converged.
* All Table I maps plus the polytope map, each with `contains(u)`. The box map includes μ
  (the original was wrong for asymmetric boxes). Every inverse returns a point of `U` even
  when the closed form saturates in float64, and `grad` never returns `±∞`.
* `solve(x0, controls)` rejects initial controls outside `int(U)`. On the boundary
  `Ψ⁻¹ = 0` and the solver would otherwise report convergence after one iteration.
* IPDDP: works on current JAX; `eigvalsh` instead of the complex `eigvals`; the feedback
  gain of state constraints uses Σ = S⁻¹Z instead of S⁻¹ (a bug that does not affect the
  benchmarks, which constrain controls only); optional `barrier_tol`, `barrier_init`, and
  a fallback when the original initial-barrier rule gives μ₀ ≤ 0 (non-positive costs).
* MPC loops record the applied controls.
* The regression gate also requires every MPC solve to converge and checks the polytope
  iteration counts.
* `pyproject.toml` with extras, a pinned lock file, CI, figure generation, and tests:
  * map identities;
  * Lemma 3 against autodiff;
  * LQ problems for every map against reference solutions of the constrained problem;
  * Proposition 5 over a full horizon;
  * IPDDP gains against a direct KKT solve;
  * the gate's own failure detection;
  * the bitwise regression.

Every change is bitwise-neutral on the four benchmarks.

## Performance follow-ups (deliberately not done)

Correctness came first. These would speed things up but change the floating-point
results, so each needs a reviewed re-baseline. The scripts in `benchmarks/experiments/`
reproduce every row:

| Change | Script | Measured effect |
|---|---|---|
| Vectorize stage derivatives over the horizon | `rounding_sensitivity.py` | `cartpole-compare` 23,470 iterations (+89), max 111 (was 102) |
| One Cholesky factorization per stage instead of two LU solves | `rounding_sensitivity.py` | 24,422 iterations; one solve stalls at the iteration cap |
| `vmap` the 10 MPC runs | `rounding_sensitivity.py` | 24,622 iterations, one stalled solve, and slower (4.8 s vs 3.9 s; a batched `while_loop` waits for the slowest run) |
| Damped Newton for the polytope inverse map (instead of optimistix) | `polytope_damped_newton.py` | `polytope-compare` 28,074 iterations (+1.6 %), max 45 (was 36) |

The runner already compiles each closed loop once and reuses it for all initial states.

## Development

```bash
pytest -m "not slow"     # unit tests (~30 s)
pytest -m slow           # bitwise regression against the original code (~1-2 min)
ruff check . && ruff format --check .
python benchmarks/experiments/rounding_sensitivity.py    # the README's experiment numbers
python benchmarks/experiments/carry_mirror_controls.py
python benchmarks/experiments/polytope_damped_newton.py
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
