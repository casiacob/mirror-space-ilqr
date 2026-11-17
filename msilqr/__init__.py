"""Mirror-space iLQR: optimal control with convex control constraints.

The solver runs iLQR in mirror coordinates, so every iterate satisfies the constraints.
"""

from msilqr.common import RegularizationOptions, Status
from msilqr.ipddp import InteriorPointDDP, IPDDPOptions, IPDDPSolution
from msilqr.mirror_maps import (
    BoxEntropy,
    BoxLogBarrier,
    EuclideanBall,
    MirrorMap,
    PolytopeLogBarrier,
    Positivity,
)
from msilqr.mpc import MPCResult, ipddp_mpc, mirror_mpc
from msilqr.msilqr import MirrorILQR, Solution, SolverOptions
from msilqr.problem import OptimalControlProblem, euler, rollout
from msilqr.systems import CartPole, DoubleIntegrator

__version__ = "1.0.0.dev0"

__all__ = [
    "BoxEntropy",
    "BoxLogBarrier",
    "CartPole",
    "DoubleIntegrator",
    "EuclideanBall",
    "InteriorPointDDP",
    "IPDDPOptions",
    "IPDDPSolution",
    "MirrorILQR",
    "MirrorMap",
    "MPCResult",
    "OptimalControlProblem",
    "PolytopeLogBarrier",
    "Positivity",
    "RegularizationOptions",
    "Solution",
    "SolverOptions",
    "Status",
    "euler",
    "ipddp_mpc",
    "mirror_mpc",
    "rollout",
]
