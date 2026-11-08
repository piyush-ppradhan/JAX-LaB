"""Verify thermal buoyancy forcing in population- and moment-space solvers."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.base import LBMBase
from jax_lab.core.models import BGKSim, KBCSim, MRTSim
from jax_lab.core.thermal import Thermal

DOMAIN = (8, 8, 0)
PRECISION = "f64/f64"
OMEGA = 1.0
GRAVITY = [0.0, -1e-4]
SEED = 0


@pytest.fixture(autouse=True)
def _enable_x64():
    """These tests use f64 tolerances: enable x64 for each test (other test modules may have switched it off since import)."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


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

    actual = fluid.apply_force(fout, feq, rho, u)  # buoyancy via LBMBase.get_force_field
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

    actual = fluid.apply_force(m, meq, rho, u)  # buoyancy via LBMBase.get_force_field
    expected = m + jnp.dot(_manual_buoyancy_delta_feq(thermal, rho, u), fluid.M)

    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) < 1e-9

    # The bug this guards against: adding the buoyancy delta_feq directly in population space (ignoring the M
    # transform) gives a different, wrong result when m/meq are moments - not just float64-noise different.
    wrong = m + _manual_buoyancy_delta_feq(thermal, rho, u)
    assert np.max(np.abs(np.asarray(actual) - np.asarray(wrong))) > 1e-5


def _mrt_kwargs(lattice):
    return dict(M=_mrt_matrix_d2q9(lattice), s_rho=OMEGA, s_e=OMEGA, s_eta=OMEGA, s_j=OMEGA, s_q=OMEGA, s_v=OMEGA)


def _array_path(cls):
    """A subclass whose update_macroscopic is a distinct (identical) function, so collision() takes the array path."""

    def update_macroscopic(self, f):
        return LBMBase.update_macroscopic.__wrapped__(self, f)

    return type(f"{cls.__name__}ArrayPath", (cls,), {"update_macroscopic": update_macroscopic})


def test_fused_collision_applies_buoyancy():
    """The fused per-direction collisions must apply the buoyancy force exactly like the array path."""
    for cls, extra in ((BGKSim, {"omega": OMEGA}), (KBCSim, {"omega": OMEGA}), (MRTSim, None)):
        lattice = LatticeD2Q9(PRECISION)
        kwargs = dict(lattice=lattice, nx=DOMAIN[0], ny=DOMAIN[1], nz=DOMAIN[2], precision=PRECISION)
        kwargs.update(extra if extra is not None else _mrt_kwargs(lattice))
        results = []
        for solver_cls in (cls, _array_path(cls)):
            fluid = solver_cls(**kwargs)
            Thermal(fluid_solver=fluid, specific_heat=1.0, thermal_conductivity=0.05, apply_buoyancy=True, gravity=GRAVITY)
            rho, u = _random_fields()
            rng = np.random.default_rng(SEED + 2)
            f = fluid.equilibrium(rho, u, cast_output=False) + jnp.asarray(rng.uniform(-0.01, 0.01, size=(*rho.shape[:-1], lattice.q)))
            results.append(np.asarray(fluid.collision(f)))
            # Without buoyancy the result must differ (the test would otherwise pass with the force dropped)
            if solver_cls is cls:
                fluid.buoyancy_gravity = None
                fluid.force = None
                no_buoyancy = np.asarray(type(fluid).collision.__wrapped__(fluid, f))
        assert np.max(np.abs(results[0] - results[1])) < 1e-12, cls.__name__
        assert np.max(np.abs(results[0] - no_buoyancy)) > 1e-7, cls.__name__
