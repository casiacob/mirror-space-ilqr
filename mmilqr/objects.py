from typing import NamedTuple, Callable
import jax.numpy as jnp


class DualOCP(NamedTuple):
    dynamics: Callable
    stage_cost: Callable
    final_cost: Callable
    total_cost: Callable
    state_mirror_map: Callable
    state_dgf_hessian: Callable
    state_dgf_inv_hessian: Callable
    control_mirror_map: Callable
    control_dgf_hessian: Callable
    control_dgf_inv_hessian: Callable
    control_inverse_map: Callable


class Derivatives(NamedTuple):
    cz: jnp.ndarray
    cy: jnp.ndarray
    fz: jnp.ndarray
    fy: jnp.ndarray
    czz: jnp.ndarray
    cyy: jnp.ndarray
    czy: jnp.ndarray


class IPOCP(NamedTuple):
    dynamics: Callable
    stage_cost: Callable
    final_cost: Callable
    constraints: Callable
