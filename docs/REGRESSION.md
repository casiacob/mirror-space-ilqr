# Regression workflow

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
