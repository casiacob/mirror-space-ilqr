"""Mirror-space constrained optimal control (Iacob, Abdulsamad, Sarkka).

Mirror-space iLQR handles convex control constraints by solving iLQR in the mirror
coordinates ``y = grad psi(u)`` of a Legendre-type function ``psi`` whose domain is the
constraint set, so every iterate is feasible (strictly feasible in exact arithmetic; in
float64 a saturated control can round onto the boundary).
"""

from msilqr._common import RegularizationOptions, Status
from msilqr.ipddp import InteriorPointDDP, IPDDPOptions, IPDDPSolution
from msilqr.mirror_ilqr import MirrorILQR, Solution, SolverOptions
from msilqr.mirror_maps import (
    BoxEntropy,
    BoxLogBarrier,
    EuclideanBall,
    MirrorMap,
    PolytopeLogBarrier,
    Positivity,
)
from msilqr.mpc import MPCResult, ipddp_mpc, mirror_mpc
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
