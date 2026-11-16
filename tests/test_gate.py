"""The regression gate itself (fast: uses the committed golden outputs, no solver runs)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

import regression as R  # noqa: E402


def golden_as_results(tmp_path, mutate=None):
    out = tmp_path / "results"
    out.mkdir()
    for name, case in R.CASES.items():
        res = R.load_legacy(case, R.DEFAULT_GOLDEN)
        res = {k: v.copy() for k, v in res.items() if not k.endswith("_logged")}
        for m in case.methods:
            res[f"{m}_status"] = np.zeros(res[f"{m}_states"].shape[:2][0:1] + (res[f"{m}_controls"].shape[1],), dtype=int)
        if mutate:
            mutate(name, res)
        np.savez(out / f"{name}.npz", **res)
    return str(out)


def verdicts(path, bitwise=False):
    return {(v.case, v.method): v.ok for v in R.run(path, R.DEFAULT_GOLDEN, list(R.CASES), bitwise=bitwise)}


def test_golden_passes_its_own_gate(tmp_path):
    assert all(verdicts(golden_as_results(tmp_path), bitwise=True).values())


def test_polytope_iterations_are_checked():
    golden = R.load_legacy(R.CASES["polytope"], R.DEFAULT_GOLDEN)
    assert golden["ms_iterations"].sum() == 2746 and golden["ms_iterations"].max() == 34


@pytest.mark.parametrize(
    "case,method,mutation",
    [
        ("cartpole", "ms", lambda r: r["ms_iterations"].__setitem__((0, 5), r["ms_iterations"][0, 5] + 1)),
        ("polytope", "ms", lambda r: r["ms_iterations"].__setitem__((0, 0), r["ms_iterations"][0, 0] + 1)),
        ("polytope", "ms", lambda r: r["ms_status"].__setitem__((0, 7), 1)),
        ("cartpole-compare", "ip", lambda r: r["ip_status"].__setitem__((2, 3), 2)),
        ("polytope-compare", "ms", lambda r: r["ms_controls"].__setitem__((0, 0), r["ms_controls"][0, 0] + 1e-6)),
    ],
    ids=["extra-iteration", "polytope-extra-iteration", "not-converged", "nonfinite-status", "inconsistent-control"],
)
def test_gate_catches_regressions(tmp_path, case, method, mutation):
    def mutate(name, res):
        if name == case:
            mutation(res)

    result = verdicts(golden_as_results(tmp_path, mutate))
    assert not result[(case, method)]
    assert all(ok for key, ok in result.items() if key != (case, method))
