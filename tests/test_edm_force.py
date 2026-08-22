"""Verify the compact EDM force-difference formula used by Multiphase.apply_force and
MultiphaseMRT.apply_force is an exact algebraic substitute for feq(rho, u + du) - feq(rho, u).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_lab.core.lattice import LatticeD2Q9, LatticeD3Q19

SEED = 0


def _compact_delta_feq(rho, u, du, w, c):
    cu = 3.0 * jnp.dot(u, c)
    dcu = 3.0 * jnp.dot(du, c)
    delta_usqr = 1.5 * (2.0 * jnp.sum(u * du, axis=-1, keepdims=True) + jnp.sum(jnp.square(du), axis=-1, keepdims=True))
    return rho * w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr)


def _two_equilibrium_delta_feq(rho, u, du, w, c):
    def feq(u_):
        cu = 3.0 * jnp.dot(u_, c)
        usqr = 1.5 * jnp.sum(jnp.square(u_), axis=-1, keepdims=True)
        return rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

    return feq(u + du) - feq(u)


def _check(lattice_class, dim):
    lattice = lattice_class("f64/f64")
    c = jnp.array(lattice.c, dtype=jnp.float64)
    w = jnp.array(lattice.w, dtype=jnp.float64)

    rng = np.random.default_rng(SEED)
    shape = (8, 8, 8, 1) if dim == 3 else (8, 8, 1)
    rho = jnp.asarray(rng.uniform(0.5, 5.0, size=shape))
    u = jnp.asarray(rng.uniform(-0.05, 0.05, size=(*shape[:-1], dim)))
    du = jnp.asarray(rng.uniform(-0.01, 0.01, size=(*shape[:-1], dim)))

    compact = _compact_delta_feq(rho, u, du, w, c)
    reference = _two_equilibrium_delta_feq(rho, u, du, w, c)

    assert np.max(np.abs(np.asarray(compact) - np.asarray(reference))) < 1e-12


def test_compact_edm_matches_two_equilibrium_form_2d():
    _check(LatticeD2Q9, dim=2)


def test_compact_edm_matches_two_equilibrium_form_3d():
    _check(LatticeD3Q19, dim=3)


if __name__ == "__main__":
    test_compact_edm_matches_two_equilibrium_form_2d()
    test_compact_edm_matches_two_equilibrium_form_3d()
    print("compact EDM difference matches the two-equilibrium reference")
