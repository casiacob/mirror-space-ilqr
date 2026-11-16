# Paper vs. code

The paper's figures were produced by the original scripts (commit `effb14c`), and this
repository reproduces them bit for bit. These notes list where the paper's text differs
from those experiments (notes 1-7) and properties of the method found while testing
(notes 8-11). They come from an audit of the original code ([`AUDIT.md`](AUDIT.md)).
Numbers are from `benchmarks/run.py` with the variant flags shown, or from the scripts in
`benchmarks/experiments/`.

1. **The cart-pole uses half of Table I's box map.** Sec. V-b says the cart-pole uses
   "the entropy DGF for box constraints from Table I". The experiments use
   `y = ½ log((u−a)/(b−u))`, `u = ρ tanh(y)` (Table I with `scale=0.5`). Under the
   `λI` damping in mirror coordinates the scale changes the iterates. With the exact
   Table I map (`--box-scale 1`):
   * `cartpole` (Fig. 3): 3,559 iterations (+80 %), cost 7576.9 (+27 %), and a final goal
     error of 2.0 instead of 0.005: the swing-up does not settle within 4 s.
   * `cartpole-compare`: 24,402 iterations (+4.4 %), but max 76 per step (was 102) and
     cost −3.2 %.

   The benchmarks keep `scale=0.5`, the setting behind Figs. 3 and 4.
2. **The interior-point baseline stops before its barrier parameter reaches `tol`.** IPDDP
   stops when the barrier subproblem's residual is below `tol`, so it returns
   central-path points, and its Fig. 4 counts exclude the iterations needed to drive μ to
   `tol`. With `--ip-barrier-tol 1e-6`: `cartpole-compare` IP 88,565 iterations
   (+19.5 %), `polytope-compare` IP 60,641 (+15 %, with two solves stalled at the
   1,000-iteration cap).
3. **The two methods use different stopping tests.** MS stops on `max|Q_y|` in mirror
   coordinates. That quantity is `Ψ⁻¹Q_u`, so it vanishes at saturated controls whatever the
   sign of the control-space gradient. IP stops on `max(|Q_u|, |r_p|, |r_d|)`.
4. **The baseline is cold-started.** IP restarts slacks (0.1), duals (0.01), μ and λ at
   every MPC step, while MS warm-starts its controls. Warm-starting only slacks and duals
   makes IPDDP worse (every second solve stalls), because μ is reset; a warm start would
   have to carry μ as well. The step acceptance also ignores primal infeasibility, and the
   fraction-to-boundary rule is applied per stage.
5. **Settings of the comparison.** The 10 initial states of `polytope-compare` are sampled
   around 0 (std 0.1 / 0.005), not around Fig. 2's `x0 = [0.2, −0.1, −0.1, 0.1]`. Fig. 4
   shows a trailing 10-step moving average. Settings not stated in the paper: Euler steps
   of 0.02 s and 0.01 s, horizons of 15 and 100, λ₀ = 10⁻², `tol` = 10⁻⁶.
6. **The original IP scripts logged the wrong controls.** They recorded the *next planned*
   control instead of the applied one (off by up to 23 N on the cart-pole). The paper's
   figures do not plot them (Fig. 4 shows iteration counts); the MPC loops here record
   the applied controls.
7. **The original IPDDP crashes on current JAX** (`jnp.clip(a_min=...)`), so Fig. 4 cannot
   be reproduced from `effb14c` with a fresh install. The code here runs on JAX 0.12.
8. **"Strictly feasible" holds in exact arithmetic only.** In float64, `tanh` rounds to 1
   for `|y|/2c ≳ 19`, so controls land exactly on the bound: 290 of the 4,000 applied
   controls in `cartpole-compare`. There `Ψ⁻¹ = 0` and the control is frozen until the MPC
   shift moves it. Carrying `y` instead of recomputing `ȳ = ∇ψ(ū)`
   (`benchmarks/experiments/carry_mirror_controls.py`) gives the same iterations and cost
   on `cartpole`. On `cartpole-compare` it needs 24,458 iterations instead of 23,381, one
   solve stalls at the iteration cap, and the mean closed-loop cost is slightly lower
   (6655.00 vs 6656.78). Mathematically identical rewrites of the solver move this case by
   as much (24,422 and 24,622 iterations, one stall each; `docs/REGRESSION.md`), so the
   experiment shows no measurable effect either way. The library keeps the paper's
   `ȳ = ∇ψ(ū)` and only replaces `±∞` by `±40c`, which is bitwise neutral.
9. **The cart-pole cost is not even C¹ at θ = 0** (the hanging start): the angle is
   wrapped with `θ mod 2π`, so the one-sided derivatives are ±10π and the gradient jumps
   by 20π there. Sec. II assumes twice-differentiable costs.
10. **Euclidean-ball constraints can converge slowly.** On a small 2-D LQ problem whose
    optimal controls slide along the boundary of the unit ball (`tests/test_solver.py`),
    the solver's stopping measure falls roughly like 1/k: about 4·10⁻⁴ after 1,000
    iterations (it oscillates between 3.7·10⁻⁴ and 6.6·10⁻⁴ around there), 9.0·10⁻⁵ after
    5,000 and 1.3·10⁻⁵ after 20,000. When the active controls point in a fixed direction
    it converges in 36 iterations (tol 10⁻⁶; 50 at 10⁻⁹). The paper's convergence
    analysis covers boxes only; under the `λI` damping, steps along the curved boundary
    shrink with the radial `Ψ⁻¹`.
11. **The box log-barrier map can stall at a float64 resolution limit.** In the cart-pole
    MPC with `BoxLogBarrier(-50, 50)`, 15 of the 400 solves end at the iteration cap, even
    with a cap of 5,000; the closed loop itself is fine. In such a solve the stopping
    measure is just above the tolerance (1.7·10⁻⁶), but the best model step would lower a
    cost of about 10³ by only 4·10⁻¹⁴, below float64 resolution. No step registers as a
    strict decrease, every step is rejected, and λ climbs to its cap. The original code
    would loop forever here; the iteration cap ends it with `MAX_ITER`.
