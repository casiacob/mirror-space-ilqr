"""float64 guard.

The solvers assume double precision: with JAX's default float32 the cart-pole
experiments never reach the stopping tolerance (the original code then looped forever).
"""

import jax


def require_x64() -> None:
    """Raise an informative error unless ``jax_enable_x64`` is on."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError('msilqr requires double precision. Enable it before creating arrays: jax.config.update("jax_enable_x64", True)')
