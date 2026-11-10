from typing import NamedTuple, Callable
import jax.numpy as jnp


class DualOCP(NamedTuple):
    dynamics: Callable
    stage_cost: Callable
    final_cost: Callable
    total_cost: Callable
    control_mirror_map: Callable
    control_dgf_hessian: Callable
    control_dgf_inv_hessian: Callable
    control_inverse_map: Callable


class Derivatives(NamedTuple):
    cx: jnp.ndarray
    cy: jnp.ndarray
    fx: jnp.ndarray
    fy: jnp.ndarray
    cxx: jnp.ndarray
    cyy: jnp.ndarray
    cxy: jnp.ndarray


class IPOCP(NamedTuple):
    dynamics: Callable
    stage_cost: Callable
    final_cost: Callable
    constraints: Callable
