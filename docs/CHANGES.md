# Changes relative to the original code (commit effb14c)

* New typed API (`OptimalControlProblem`, `MirrorMap`, `MirrorILQR`, `InteriorPointDDP`,
  MPC loops) with docstrings mapped to the paper; the four duplicated scripts are replaced
  by `msilqr/systems.py` and `benchmarks/cases.py`.
* Both solvers have an iteration cap and a convergence status; the original loops could
  run forever.
* Solver internals restructured without changing their arithmetic (verified bitwise):
  * `msilqr/_common.py` holds what both solvers share (status, stopping and
    regularization options, the gain-ratio update, the stage expansion);
  * the mirror-space backward pass is split into `mirror_derivatives`, `q_expansion`,
    `feedback_gains` and `value_update`, mapped to Lemma 3, Prop. 4, Eq. (12) and Eq. (13);
  * the IPDDP backward pass is split into `perturbed_kkt`, `newton_gains` and
    `augmented_q`, and its forward pass uses `fraction_to_boundary`.
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

* MIT license (Casian Iacob and Hany Abdulsamad), a citation block and a user-facing README;
  author-facing material lives in `docs/` (this file, `PAPER_NOTES.md`, `REGRESSION.md`,
  `AUDIT.md`).

Every change is bitwise-neutral on the four benchmarks.
