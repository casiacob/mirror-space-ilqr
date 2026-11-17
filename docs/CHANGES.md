# Changes relative to the original code (commit effb14c)

* New typed API (`OptimalControlProblem`, `MirrorMap`, `MirrorILQR`, `InteriorPointDDP`,
  MPC loops) with docstrings mapped to the paper; the four duplicated scripts are replaced
  by `msilqr/systems.py` and `benchmarks/cases.py`.
* Both solvers have an iteration cap and a convergence status; the original loops could
  run forever.
* Solver internals restructured without changing their arithmetic (verified bitwise):
  * `msilqr/common.py` holds what both solvers share (status, stopping and
    regularization options, the gain-ratio update, the stage expansion);
  * the mirror-space backward pass is split into `mirror_derivatives`, `q_expansion`,
    `feedback_gains` and `value_update`, mapped to Lemma 3, Prop. 4, Eq. (12) and Eq. (13);
  * the IPDDP backward pass is split into `perturbed_kkt`, `newton_gains` and
    `augmented_q`, and its forward pass uses `fraction_to_boundary`.
* A float64 guard: in float32 the original cart-pole solve never converged.
* Requires Python ≥ 3.12 (as JAX 0.12 does).
* All Table I maps plus the polytope map, behind an abstract `MirrorMap` class: a custom
  map implements `grad`, `inverse`, `hessian`, `inv_hessian` and `contains`. The box map
  includes μ (the original was wrong for asymmetric boxes). The closed-form inverses (box,
  positivity, ball) return a point of `U` for every finite `y`, even when they saturate in
  float64, with finite `grad` and `inv_hessian` there and finite reverse-mode
  derivatives. The polytope map's Newton inverse can return NaN; the solver rejects such
  a step.
* `solve(x0, controls)` rejects initial controls outside `int(U)`. On the boundary
  `Ψ⁻¹ = 0` and the solver would otherwise report convergence after one iteration.
  `validate=False` skips the check; the MPC loops use it because their shifted warm
  starts may touch the bound. Input shapes and solver options are checked as well, and
  `ipddp_mpc` checks slacks and duals like `InteriorPointDDP.solve`. Concrete inputs are
  checked even inside `jit` (the check used to raise a tracer error there).
* IPDDP: works on current JAX; `eigvalsh` instead of the complex `eigvals`; the feedback
  gain of state constraints uses Σ = S⁻¹Z instead of S⁻¹ (a bug that does not affect the
  benchmarks, which constrain controls only); `barrier_init`, and a fallback when the
  original initial-barrier rule gives μ₀ ≤ 0 (non-positive costs).
* The default IPDDP options resolve the baseline points of `docs/PAPER_NOTES.md` (notes 2
  and 4): it stops only once μ < `barrier_tol` (1e-6); controls, slacks and duals take one
  step size, halved until the fraction-to-boundary rule holds; a filter on (barrier cost,
  `‖g + s‖₁`) accepts steps; `ipddp_mpc` warm-starts slacks, duals and μ.
  `IPDDPOptions.paper()` reproduces the paper's baseline bitwise, and
  `benchmarks/run.py --baseline fixed` runs the comparison with the fixed one. Tested
  against a projected-gradient QP solution and SciPy's SLSQP.
* Optional stopping tests on the KKT conditions of the constrained problem, next to the
  paper's (the defaults are unchanged):
  * `SolverOptions(stopping="natural")`: mirror-space iLQR stops on the natural residual
    `max |u − Π_U(u − ∇J)|` (`∇J` from an adjoint sweep in the backward pass; every
    constraint set has a `project` method). It ends with `Status.STALLED` once a rejected
    step predicts a decrease below `10ε|J|`, which float64 cannot register;
  * `IPDDPOptions(stopping="kkt")`: IPDDP stops on the unperturbed KKT error
    `max(|∇_u L| / c, |g + s|, |s z| / c)` with its own duals and IPOPT's scaling `c`.

  `benchmarks/run.py --baseline fixed --stop kkt` runs the comparison with both.
* MPC loops record the applied controls.
* The regression gate also requires every MPC solve to converge (a result without status
  fails) and checks the polytope iteration counts. In CI the regression test fails,
  instead of skipping, when the original commit is missing, and lint runs with a pinned
  ruff.
* `pyproject.toml` with extras, a pinned lock file, CI, figure generation, and tests:
  * map identities;
  * Lemma 3 against autodiff;
  * LQ problems for every map against reference solutions of the constrained problem;
  * Proposition 5 over a full horizon;
  * IPDDP gains against a direct KKT solve;
  * the gate's own failure detection;
  * the bitwise regression.

* MIT license (Casian Iacob and Hany Abdulsamad), a citation block and a user-facing README;
  background material lives in `docs/` (this file, `PAPER_NOTES.md`, `REGRESSION.md`,
  `AUDIT.md`).

* The comparison cases draw each run's initial state and initial controls from its own
  key (split from `PRNGKey(12)`); the original scripts shared one control sequence.

Every other change is bitwise-neutral on the benchmarks; the regression test checks
`cartpole` and `polytope`, which keep the original inputs.
