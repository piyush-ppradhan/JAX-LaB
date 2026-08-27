"""Verify thermal buoyancy forcing in population- and moment-space solvers."""

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
GRAVITY = [0.0, -1e-4]
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


def _random_fields():
    nx, ny = DOMAIN[0], DOMAIN[1]
    rng = np.random.default_rng(SEED)
    rho = jnp.asarray(rng.uniform(0.5, 2.0, size=(nx, ny, 1)))
    u = jnp.asarray(rng.uniform(-0.02, 0.02, size=(nx, ny, 2)))
    return rho, u


def _manual_buoyancy_delta_feq(thermal, rho, u):
    rho_average = rho.mean()
    buoyancy = jnp.repeat(rho - rho_average, repeats=thermal.dim, axis=-1) * thermal.gravity
    du = buoyancy + thermal.fluid_solver.force
    c = jnp.array(thermal.lattice.c, dtype=thermal.precision_policy.compute_dtype)
    cu = 3.0 * jnp.dot(u, c)
    dcu = 3.0 * jnp.dot(du, c)
    delta_usqr = 1.5 * (2.0 * jnp.sum(u * du, axis=-1, keepdims=True) + jnp.sum(jnp.square(du), axis=-1, keepdims=True))
    return rho * thermal.lattice.w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr)


def test_bgk_buoyancy_stays_population_space():
    fluid = BGKSim(lattice=LatticeD2Q9(PRECISION), omega=OMEGA, nx=DOMAIN[0], ny=DOMAIN[1], nz=DOMAIN[2], precision=PRECISION)
    thermal = Thermal(fluid_solver=fluid, specific_heat=1.0, thermal_conductivity=0.05, apply_buoyancy=True, gravity=GRAVITY)

    rho, u = _random_fields()
    feq = fluid.equilibrium(rho, u, cast_output=False)
    rng = np.random.default_rng(SEED + 1)
    fout = feq + jnp.asarray(rng.uniform(-0.01, 0.01, size=feq.shape))

    actual = fluid.apply_force(fout, feq, rho, u)  # monkeypatched to apply_force_thermal
    expected = fout + _manual_buoyancy_delta_feq(thermal, rho, u)

    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) < 1e-12


def test_mrt_buoyancy_is_transformed_to_moment_space():
    lattice = LatticeD2Q9(PRECISION)
    s = OMEGA
    fluid = MRTSim(
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
    thermal = Thermal(fluid_solver=fluid, specific_heat=1.0, thermal_conductivity=0.05, apply_buoyancy=True, gravity=GRAVITY)

    rho, u = _random_fields()
    feq = fluid.equilibrium(rho, u, cast_output=False)
    meq = jnp.dot(feq, fluid.M)
    rng = np.random.default_rng(SEED + 1)
    m = meq + jnp.asarray(rng.uniform(-0.01, 0.01, size=meq.shape))

    actual = fluid.apply_force(m, meq, rho, u)  # monkeypatched to apply_force_thermal
    expected = m + jnp.dot(_manual_buoyancy_delta_feq(thermal, rho, u), fluid.M)

    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) < 1e-9

    # The bug this guards against: adding the buoyancy delta_feq directly in population space (ignoring the M
    # transform) gives a different, wrong result when m/meq are moments - not just float64-noise different.
    wrong = m + _manual_buoyancy_delta_feq(thermal, rho, u)
    assert np.max(np.abs(np.asarray(actual) - np.asarray(wrong))) > 1e-5
