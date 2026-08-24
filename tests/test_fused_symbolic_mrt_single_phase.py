"""Verify fused single-phase MRT collision against an independent unfused reference with nonzero force."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.models import MRTSim

DOMAIN = (10, 10, 10)
PRECISION = "f64/f64"
OMEGA = 1.0
FORCE = jnp.array([1e-4, -2e-4, 5e-5], dtype=jnp.float64)
SEED = 0


def _mrt_matrix(lattice):
    e = np.asarray(lattice.c).T
    en = np.linalg.norm(e, axis=1)
    x, y, z = e.T
    matrix = np.zeros((19, 19))
    matrix[0, :] = en**0
    matrix[1, :] = 19 * en**2 - 30
    matrix[2, :] = (21 * en**4 - 53 * en**2 + 24) / 2
    matrix[3, :] = x
    matrix[4, :] = (5 * en**2 - 9) * x
    matrix[5, :] = y
    matrix[6, :] = (5 * en**2 - 9) * y
    matrix[7, :] = z
    matrix[8, :] = (5 * en**2 - 9) * z
    matrix[9, :] = 3 * x**2 - en**2
    matrix[10, :] = (3 * en**2 - 5) * (3 * x**2 - en**2)
    matrix[11, :] = y**2 - z**2
    matrix[12, :] = (3 * en**2 - 5) * (y**2 - z**2)
    matrix[13, :] = x * y
    matrix[14, :] = y * z
    matrix[15, :] = x * z
    matrix[16, :] = (y**2 - z**2) * x
    matrix[17, :] = (z**2 - x**2) * y
    matrix[18, :] = (x**2 - y**2) * z
    return matrix


class ForcedMRT(MRTSim):
    def get_force(self):
        return FORCE


def _build_sim():
    nx, ny, nz = DOMAIN
    lattice = LatticeD3Q19(PRECISION)
    s = OMEGA
    return ForcedMRT(
        lattice=lattice,
        nx=nx,
        ny=ny,
        nz=nz,
        precision=PRECISION,
        M=_mrt_matrix(lattice),
        s_rho=s,
        s_e=s,
        s_eta=s,
        s_j=s,
        s_q=s,
        s_v=s,
        s_pi=s,
        s_m=s,
    )


def _random_f(sim):
    f = sim.assign_fields_sharded()
    rng = np.random.default_rng(SEED)
    return f + jnp.asarray(rng.uniform(-0.01, 0.01, size=f.shape))


def _reference_collision(sim, f):
    """Independent unfused reference: m = f @ M; mout = -(m - meq) @ S + delta_meq; fout = f + mout @ M_inv."""
    f = sim.precision_policy.cast_to_compute(f)
    m = jnp.dot(f, sim.M)
    rho, u = sim.update_macroscopic(f)
    feq = sim.equilibrium(rho, u)
    meq = jnp.dot(feq, sim.M)
    mout = -jnp.dot(m - meq, sim.S)
    mout = sim.apply_force(mout, meq, rho, u)
    return sim.precision_policy.cast_to_output(f + jnp.dot(mout, sim.M_inv))


def test_fused_symbolic_collision_matches_reference():
    sim = _build_sim()
    f = _random_f(sim)

    actual = sim.collision(f)
    expected = _reference_collision(sim, f)

    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) < 1e-9
