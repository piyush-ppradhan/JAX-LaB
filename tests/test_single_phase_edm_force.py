"""Verify the compact EDM force formula is applied correctly for single-phase BGK (population space) and MRT
(moment space, via LBMBase.apply_force / MRTSim.apply_force), against independent two-equilibrium references
computed directly in this test (not by calling the library's own formula)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.models import BGKSim, MRTSim
from jax_lab.core.thermal import Thermal

DOMAIN = (8, 8, 0)
PRECISION = "f64/f64"
OMEGA = 1.0
FORCE = jnp.array([0.001, -0.0005], dtype=jnp.float64)
SEED = 0


def _mrt_matrix_d2q9(lattice):
    e = np.asarray(lattice.c).T
    en = np.linalg.norm(e, axis=1)
    matrix = np.zeros((9, 9))
    matrix[0, :] = en**0
    matrix[1, :] = -4 * en**0 + 3 * en**2
    matrix[2, :] = 4 * en**0 - 10.5 * en**2 + 4.5 * en**4
    matrix[3, :] = e[:, 0]
    matrix[4, :] = (-5 * en**0 + 3 * en**2) * e[:, 0]
    matrix[5, :] = e[:, 1]
    matrix[6, :] = (-5 * en**0 + 3 * en**2) * e[:, 1]
    matrix[7, :] = e[:, 0] ** 2 - e[:, 1] ** 2
    matrix[8, :] = e[:, 0] * e[:, 1]
    return matrix


class ForcedBGK(BGKSim):
    def get_force(self):
        return FORCE


class ForcedMRT(MRTSim):
    def get_force(self):
        return FORCE


def _random_fields():
    nx, ny = DOMAIN[0], DOMAIN[1]
    rng = np.random.default_rng(SEED)
    rho = jnp.asarray(rng.uniform(0.5, 2.0, size=(nx, ny, 1)))
    u = jnp.asarray(rng.uniform(-0.02, 0.02, size=(nx, ny, 2)))
    return rho, u


def test_bgk_apply_force_matches_population_space_reference():
    sim = ForcedBGK(lattice=LatticeD2Q9(PRECISION), omega=OMEGA, nx=DOMAIN[0], ny=DOMAIN[1], nz=DOMAIN[2], precision=PRECISION)
    rho, u = _random_fields()
    feq = sim.equilibrium(rho, u, cast_output=False)
    rng = np.random.default_rng(SEED + 1)
    fout = feq + jnp.asarray(rng.uniform(-0.01, 0.01, size=feq.shape))

    actual = sim.apply_force(fout, feq, rho, u)

    du = sim.get_force()
    feq_force = sim.equilibrium(rho, u + du, cast_output=False)
    expected = fout + feq_force - feq

    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) < 1e-12


def test_mrt_apply_force_matches_moment_space_reference():
    lattice = LatticeD2Q9(PRECISION)
    s = OMEGA
    sim = ForcedMRT(
        lattice=lattice,
        nx=DOMAIN[0],
        ny=DOMAIN[1],
        nz=DOMAIN[2],
        precision=PRECISION,
        M=_mrt_matrix_d2q9(lattice),
        s_rho=s,
        s_e=s,
        s_eta=s,
        s_j=s,
        s_q=s,
        s_v=s,
    )
    rho, u = _random_fields()
    feq = sim.equilibrium(rho, u, cast_output=False)
    meq = jnp.dot(feq, sim.M)
    rng = np.random.default_rng(SEED + 2)
    m = meq + jnp.asarray(rng.uniform(-0.01, 0.01, size=meq.shape))

    actual = sim.apply_force(m, meq, rho, u)

    du = sim.get_force()
    feq_force = sim.equilibrium(rho, u + du, cast_output=False)
    # Correctly space-matched reference: transform both equilibria into moment space before subtracting,
    # instead of subtracting a moment-space meq from a population-space feq_force.
    expected = m + jnp.dot(feq_force, sim.M) - jnp.dot(feq, sim.M)

    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) < 1e-10


def test_thermal_buoyancy_force_matches_population_space_reference():
    fluid = BGKSim(lattice=LatticeD2Q9(PRECISION), omega=OMEGA, nx=DOMAIN[0], ny=DOMAIN[1], nz=DOMAIN[2], precision=PRECISION)
    thermal = Thermal(fluid_solver=fluid, specific_heat=1.0, thermal_conductivity=0.05, apply_buoyancy=True, gravity=[0.0, -1e-4])

    rho, u = _random_fields()
    feq = fluid.equilibrium(rho, u, cast_output=False)
    rng = np.random.default_rng(SEED + 3)
    fout = feq + jnp.asarray(rng.uniform(-0.01, 0.01, size=feq.shape))

    # apply_buoyancy=True monkeypatches fluid.apply_force to Thermal.apply_force_thermal.
    actual = fluid.apply_force(fout, feq, rho, u)

    du = thermal.gravity * (rho - rho.mean()) + fluid.force
    feq_force = fluid.equilibrium(rho, u + du, cast_output=False)
    expected = fout + feq_force - feq

    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) < 1e-12


if __name__ == "__main__":
    test_bgk_apply_force_matches_population_space_reference()
    test_mrt_apply_force_matches_moment_space_reference()
    test_thermal_buoyancy_force_matches_population_space_reference()
    print("single-phase BGK, MRT and thermal-buoyancy apply_force match their independent references")
