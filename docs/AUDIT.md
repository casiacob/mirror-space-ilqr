# Audit of the original implementation (commit effb14c)

Five reviewers audited the original code against the paper, each from one angle: theory, numerics, the IPDDP baseline, design, and reproducibility.
Duplicates were merged into 29 findings. Each finding was then checked by an independent adversarial verifier,
and critical and major findings by up to two more. All 29 were upheld. The table records what the revamp did about each one.
A final review of the revamped branch found further problems: asymmetric-box saturation, boundary initial guesses, and gaps in the gate. Their fixes are folded into the dispositions below.
A second review of the restructured branch is recorded [at the end](#second-review-of-the-revamped-branch).

| ID | Severity | Finding | Disposition | Details |
|---|---|---|---|---|
| F01 | critical | IPDDP baseline crashes on the repo's JAX 0.12: jnp.clip(a_min=, a_max=) no longer exists, so the IP half of Fig. 4 cannot run | Fixed | `jnp.clip(min=, max=)`; the capture tool shims the old keywords to run the original. |
| F02 | major | Neither solver's while_loop has an iteration cap or returns a convergence status: stalled, non-finite, warm-started, float32 or tight-tolerance... | Fixed | `max_iter` and a `Status` (`CONVERGED`, `MAX_ITER`, `NONFINITE`) in both solvers, consistent for `max_iter=0`; bitwise-neutral; tested. |
| F03 | major | Box controls that round onto the bound stay frozen there: ybar = grad psi(ubar) and Psi^{-1}(ubar) are recomputed from the rounded control, giving... | Partly fixed, rest documented | `±inf` replaced by `±40·scale`, and box inverses clipped to the closed box (both bitwise-neutral). The suggested fix (carry `y`, `benchmarks/experiments/carry_mirror_controls.py`) gives the same `cartpole` result; its changes on `cartpole-compare` (+4.6 % iterations, one stalled solve, 0.03 % lower cost) are within the spread of mathematically identical rewrites, so it shows no measurable benefit and the paper's `ȳ = ∇ψ(ū)` is kept. PAPER_NOTES note 8. |
| F04 | major | MS and IP stop on different quantities. MS tests max\|Q_y\| = max\|Psi^{-1} Q_u\| in mirror coordinates, which vanishes near the bounds whatever... | Documented | Changing either stopping rule changes the paper's numbers. PAPER_NOTES note 3, `ipddp.py` docstring. |
| F05 | major | IPDDP exits on the barrier-subproblem residual alone and never drives mu to tol, so it returns central-path points and its Fig. 4 iteration counts... | Fixed | The default options stop only once μ < 1e-6, and μ is also decreased once the residual is below tol (without this, two solves stalled). `IPDDPOptions.paper()` keeps the early exit for Fig. 4. PAPER_NOTES notes 2 and 4; tested. |
| F09 | major | Polytope inverse map: an undamped optimistix Newton solve on a lifted (u, softplus-slack) system with throw=False silently returns NaN for mirror... | Diagnostic added, documented | `PolytopeLogBarrier.inverse_residual`. A damped Newton changes the golden results (+1.6 % iterations, `benchmarks/experiments/polytope_damped_newton.py`), so the original solver is kept; a NaN inverse makes the solver reject that step. |
| F10 | major | IPDDP feedback gain uses Y^{-1} instead of Sigma = Y^{-1}S for the c_x term, which gives a wrong K (and a non-symmetric Vxx) for state-dependent... | Fixed | Σ instead of S⁻¹ in the feedback gain of the state-constraint term; bitwise-neutral (benchmarks constrain controls only); tested against a direct KKT solve. |
| F12 | major | The library silently requires float64: it never enables or checks jax_enable_x64, and in JAX's default float32 the first cart-pole solve never... | Fixed | `require_x64()` raises an informative error instead of hanging; float32 inputs are promoted; tested. |
| F13 | major | Benchmark inputs come from a JAX PRNG stream that depends on the JAX version and its config; flipping one flag changes the cart-pole benchmark... | Mitigated | `requirements-lock.txt` pins the capture environment; the slow test regenerates golden outputs on the current machine. |
| F06 | minor | IP cold-starts slacks, duals, mu and the regularisation at every MPC step, while MS warm-starts all of its variables | Fixed | `ipddp_mpc` warm-starts slacks, duals and μ (their mean product) after the first step: 6.8 / 8.3 iterations per step instead of 20.4 / 25.3 cold. PAPER_NOTES note 4. |
| F07 | minor | IP acceptance test ignores primal infeasibility (no filter or penalty on c + y), and the forward-pass feasibility flag is computed but never used | Fixed | The default options accept a step by a filter on (barrier cost, ‖g + s‖₁); `IPDDPOptions.paper()` keeps the original test. PAPER_NOTES note 4. |
| F08 | minor | IP MPC loops log the next planned control instead of the control actually applied, so the MS and IP control traces are offset by one step | Fixed | MPC loops record applied controls; the regression gate reconstructs the applied controls of the legacy data; tested. |
| F11 | minor | Box inverse map drops the centre mu (it returns rho*tanh(y)), and the bounds are hard-coded separately in four functions per script, so any... | Fixed | `BoxEntropy(a, b, scale)` with μ, bound validation, and an inverse clipped to `[a, b]` (μ ± ρ can miss the bound by one rounding error); tested on random asymmetric boxes and asymmetric LQ problems. |
| F14 | minor | Paper Figs. 2-4 cannot be regenerated as files: the scripts only call plt.show(), save no data and plot against the step index, and Fig. 4 applies... | Fixed | `benchmarks/figures.py` saves Figs. 2-4 (time axes in seconds for Figs. 2-3). |
| F15 | minor | setup.py has no Python or dependency constraints, forces example-only dependencies on library users, has no build configuration, and ships an... | Fixed | `pyproject.toml` with extras (`polytope`, `benchmarks`, `dev`) replaces `setup.py`. |
| F16 | minor | No LICENSE, no CITATION and no docstrings, and a README that documents neither the method, the paper, the results nor how to reproduce them,... | Fixed | README with a citation block, docstrings, and an MIT license. |
| F17 | minor | No tests and no CI; a prototype test suite run against effb14c shows which of the paper's properties the code satisfies and which it violates | Fixed | Unit, solver, IPDDP, gate and regression tests; GitHub Actions CI. |
| F18 | minor | The public solvers cannot be jitted as written, so every eager call retraces and recompiles (about 0.5-1.0 s per call, against 1.6 ms jitted) | Fixed | Solvers compile once per instance; MPC loops are jitted once per solver and step count (cached on the solver). |
| F19 | minor | Problem and solver interfaces are untyped positional tuples. DualOCP is a bag of 8 callables (redundant total_cost, a never-used... | Fixed | `OptimalControlProblem`, `MirrorMap`, `SolverOptions`, `Solution`, `MPCResult`; both solvers take `x0`. |
| F20 | minor | The box mirror map implements half of Table I's entropy DGF (y = 0.5 log((u-a)/(b-u)), Psi_code = Psi/2), which silently rescales the LM damping... | Library follows the paper; benchmarks keep the original | `BoxEntropy` defaults to Table I (`scale=1`); the cart-pole benchmarks use `scale=0.5`. Effect quantified in PAPER_NOTES note 1. |
| F21 | minor | The LM gain ratio uses the decrease of the penalized model (the sum of dV with the damped Qyy_hat) as its denominator, not the undamped-model... | Kept (follows the paper) | Eq. (13) and the text below it use `Q̂_yy` for ΔV, which is what the code does. |
| F22 | minor | Section V protocol differs from the paper: the double-integrator comparison samples its 10 initial states around the origin instead of the paper's... | Documented | PAPER_NOTES note 5. |
| F23 | minor | The IP forward pass truncates the dual and slack steps with separate per-stage fraction-to-boundary factors while always taking the full control... | Fixed | The default options take one step size for controls, slacks and duals at all stages, halved until the fraction-to-boundary rule holds; `IPDDPOptions.paper()` keeps the per-stage factors. PAPER_NOTES note 4. |
| F25 | minor | The IPDDP definiteness test runs the general complex eigensolver (jnp.linalg.eigvals: lexicographic '>' on complex values, CPU-only) on Quu... | Fixed | `eigvalsh` on the symmetrized matrix; bitwise identical on all benchmarks. |
| F26 | minor | Problem definitions and mirror maps (the core of the method) live in four duplicated, non-importable example scripts that run everything at module... | Fixed | `msilqr/systems.py` and `benchmarks/cases.py`. |
| F27 | minor | Each backward-pass stage computes eigvalsh only for a definiteness flag and then does two LU solves with the same Qyy; one Cholesky would make the... | Deferred (performance) | A Cholesky solve changes the golden results (one stalled cart-pole solve; `benchmarks/experiments/rounding_sensitivity.py`). |
| F24 | nit | wrap_angle (theta mod 2*pi) gives the cart-pole cost a concave kink at theta = 0, right at the hanging start: the cost is not C^2 there, and the... | Documented | `systems.wrap_angle` docstring, PAPER_NOTES note 9 (not even C¹: the gradient jumps by 20π). |
| F28 | nit | Computing the derivatives inside the backward scan is not a significant cost (hoisting them with vmap gains only 1.04-1.11x), and hoisting still... | Agreed, no change | Vectorizing the derivatives also changes the golden results (+89 iterations). |
| F29 | nit | The stopping value comes from the backward pass at the pre-update iterate, and from the DDP-propagated Vx instead of the costate, so the returned... | Documented | Kept for reproducibility: the stopping value is the backward-pass `Q_y` of the current iterate, as in the original. |

## Second review of the revamped branch

An independent review of the commits "Address the final review" and "Restructure solver
internals" confirmed that the restructuring is operation-for-operation identical to the
code before it, and found the following. Every fix keeps the four benchmarks bitwise
identical.

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| R1 | major | JAX 0.12 requires Python ≥ 3.12, so the Python 3.11 CI job could not install. | Fixed: `requires-python = ">=3.12"`, CI matrix 3.12–3.14. |
| R2 | major | `EuclideanBall.inverse` capped the radius at `1 − 2⁻⁵²`; rounding put about 5 % of large random `y` on or outside the sphere, where `grad` is NaN. | Fixed: cap `1 − 2⁻⁴⁹`, `grad`/`inv_hessian` floored, randomized test in 2–100 dimensions. |
| R3 | major | Reverse-mode derivatives of `EuclideanBall.inverse` were NaN at `y = 0` (the unused branch leaked `0 · ∞`). | Fixed: each branch gets a safe input ("double where"); `jacrev` = `jacfwd` tested. |
| R4 | minor | The claim that every map returns a finite `grad` was false for the ball and the polytope map. | Fixed: claim limited to the closed-form maps (module docstring, CHANGES). |
| R5 | minor | An eager (`disable_jit`) MPC loop raised on shifted plans whose controls rounded onto the bound. | Fixed: `solve(..., validate=False)`, used by the MPC loops; tested eagerly. |
| R6 | minor | `ipddp_mpc` did not check slacks and duals; solver options were not validated. | Fixed, tested. |
| R7 | minor | The CI regression test could skip (and pass) without the original commit; a result without status skipped the convergence check; ruff was unpinned. | Fixed: `MSILQR_REQUIRE_REGRESSION=1` in CI, missing status fails, separate lint job with ruff 0.17.5. |
| R8 | minor | The carry-`y` note claimed harm; the measured change is within the spread of mathematically identical rewrites. | Fixed: PAPER_NOTES note 8 and F03 reworded. |

Nits fixed as well: the size of the gradient jump in note 9 (20π), the measured numbers in
note 10, a timing, docstrings that named the private `msilqr._common`, `Positivity.contains`
accepting `inf`, list bounds for the box maps, a shape check that catches the old
`solve(states, controls)` call style, a consistent `inf` initial stopping measure in both
solvers, and a misnamed gate test.
