"""End-to-end regression against the original implementation (slow, ~1-2 min).

XLA compiles for the host CPU, so outputs are only comparable bit for bit on the same
machine. By default this test therefore regenerates the golden outputs from the original
commit (effb14c, needs the git history) on this machine and compares the new
implementation with them. Set ``MSILQR_USE_COMMITTED_GOLDEN=1`` to compare with
``benchmarks/legacy/golden`` instead (valid on the machine that captured it; faster).

Without the git history the test is skipped, unless ``MSILQR_REQUIRE_REGRESSION=1`` (set
in CI), which turns the skip into a failure so the gate can never pass vacuously.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
LEGACY_REF = "effb14c"


def _has_legacy_commit() -> bool:
    try:
        subprocess.run(["git", "-C", ROOT, "cat-file", "-e", f"{LEGACY_REF}^{{commit}}"], check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


@pytest.mark.slow
def test_benchmarks_match_original_bitwise(tmp_path):
    import regression
    import run

    if os.environ.get("MSILQR_USE_COMMITTED_GOLDEN"):
        golden = regression.DEFAULT_GOLDEN
    else:
        if not _has_legacy_commit():
            reason = f"git history with {LEGACY_REF} not available (shallow clone?); set MSILQR_USE_COMMITTED_GOLDEN=1"
            if os.environ.get("MSILQR_REQUIRE_REGRESSION"):
                pytest.fail(reason)
            pytest.skip(reason)
        golden = str(tmp_path / "golden")
        capture = os.path.join(ROOT, "benchmarks", "legacy", "capture.py")
        subprocess.run([sys.executable, capture, "--ref", LEGACY_REF, "--out", golden], check=True, cwd=ROOT)
    results = str(tmp_path / "results")
    assert run.main(["--out", results]) == 0
    assert regression.main(["--new", results, "--golden", golden, "--bitwise"]) == 0
