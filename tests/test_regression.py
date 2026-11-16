"""End-to-end regression against the original implementation (slow, ~1 min).

By default the new results are compared bitwise with ``benchmarks/legacy/golden``.
XLA compiles for the host CPU, so golden files captured on another machine may differ
in the last bits; set ``MSILQR_RECAPTURE_GOLDEN=1`` to regenerate the golden outputs from
the original commit on this machine first (needs the git history).
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))


@pytest.mark.slow
def test_benchmarks_match_original_bitwise(tmp_path):
    import regression
    import run

    golden = regression.DEFAULT_GOLDEN
    if os.environ.get("MSILQR_RECAPTURE_GOLDEN"):
        golden = str(tmp_path / "golden")
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "benchmarks", "legacy", "capture.py"), "--ref", "effb14c", "--out", golden],
            check=True,
            cwd=ROOT,
        )
    results = str(tmp_path / "results")
    assert run.main(["--out", results]) == 0
    assert regression.main(["--new", results, "--golden", golden, "--bitwise"]) == 0
