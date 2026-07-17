"""Taylor-Green vortex regression tests for BGK and MRT collision models."""

import os
import shutil
from pathlib import Path

# os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jax_lab.eos import VanderWaals
from jax_lab.lattice import LatticeD2Q9, LatticeD3Q19
from jax_lab.models import BGKSim, MRTSim
from jax_lab.multiphase import MultiphaseBGK, MultiphaseMRT


DOMAIN_CASES = (
    pytest.param(LatticeD2Q9, (16, 16, 0), 0.08, id="2d"),
    pytest.param(LatticeD3Q19, (10, 10, 10), 0.12, id="3d"),
)
SINGLE_PHASE_MODELS = (
    pytest.param(BGKSim, id="bgk"),
    pytest.param(MRTSim, id="mrt"),
)
MULTIPHASE_MODELS = (
    pytest.param(MultiphaseBGK, id="bgk"),
    pytest.param(MultiphaseMRT, id="mrt"),
)

PRECISION = "f32/f32"
OMEGA = 1.0
VISCOSITY = (1.0 / OMEGA - 0.5) / 3.0
REFERENCE_VELOCITY = 0.02
NUMBER_OF_UPDATES = 10


@pytest.fixture(autouse=True)
def isolated_output_directory(tmp_path, monkeypatch):
    """Run each test in a temporary directory and remove all generated files."""
    original_directory = Path.cwd()
    output_directory = tmp_path / "collision-output"
    output_directory.mkdir()
    monkeypatch.chdir(output_directory)
    try:
        yield
    finally:
        os.chdir(original_directory)
        shutil.rmtree(output_directory, ignore_errors=True)


def _coordinates(domain):
    spatial_shape = domain[:2] if domain[2] == 0 else domain
    return jnp.meshgrid(
        *[2.0 * jnp.pi * jnp.arange(size, dtype=jnp.float32) / size for size in spatial_shape],
        indexing="ij",
    )


def _taylor_green_fields(domain, time):
    coordinates = _coordinates(domain)
    spatial_shape = domain[:2] if domain[2] == 0 else domain
    decay_rate = VISCOSITY * sum((2.0 * jnp.pi / size) ** 2 for size in spatial_shape)
    decay = jnp.exp(-decay_rate * time)

    x, y = coordinates[:2]
    z_factor = jnp.cos(coordinates[2]) if len(coordinates) == 3 else 1.0
    velocity_x = REFERENCE_VELOCITY * jnp.sin(x) * jnp.cos(y) * z_factor * decay
    velocity_y = -REFERENCE_VELOCITY * jnp.cos(x) * jnp.sin(y) * z_factor * decay
    velocity_z = jnp.zeros_like(velocity_x)
    velocity = jnp.stack((velocity_x, velocity_y), axis=-1) if len(coordinates) == 2 else jnp.stack((velocity_x, velocity_y, velocity_z), axis=-1)

    density_decay = jnp.exp(-2.0 * decay_rate * time)
    density = 1.0 - (REFERENCE_VELOCITY**2 / 12.0) * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y)) * density_decay
    return density[..., None], velocity


class TaylorGreenInitialFields:
    """Provide periodic Taylor-Green initial conditions to a solver."""

    def initialize_macroscopic_fields(self):
        domain = (self.nx, self.ny, self.nz)
        density, velocity = _taylor_green_fields(domain, time=0.0)
        return (
            self.precisionPolicy.cast_to_output(density),
            self.precisionPolicy.cast_to_output(velocity),
        )


class MultiphaseTaylorGreenInitialFields:
    """Provide one-component Taylor-Green fields and zero force."""

    def initialize_macroscopic_fields(self):
        domain = (self.nx, self.ny, self.nz)
        density, velocity = _taylor_green_fields(domain, time=0.0)
        return [self.precisionPolicy.cast_to_output(density)], [self.precisionPolicy.cast_to_output(velocity)]

    def compute_force(self, rho_tree, T=None):
        return [jnp.zeros((*density.shape[:-1], self.dim), dtype=self.precisionPolicy.compute_dtype) for density in rho_tree]

    def adjust_surface_tension(self, psi_tree):
        return [jnp.zeros((*psi.shape[:-1], self.q), dtype=self.precisionPolicy.compute_dtype) for psi in psi_tree]


def _mrt_matrix(lattice):
    e = np.asarray(lattice.c).T
    en = np.linalg.norm(e, axis=1)

    if lattice.d == 2:
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

    matrix = np.zeros((19, 19))
    x, y, z = e.T
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


def _base_parameters(lattice_class, domain):
    lattice = lattice_class(PRECISION)
    nx, ny, nz = domain
    return {
        "lattice": lattice,
        "omega": OMEGA,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
    }


def _mrt_relaxation_parameters(lattice, multiphase):
    value = [OMEGA] if multiphase else OMEGA
    parameters = {
        "M": [_mrt_matrix(lattice)] if multiphase else _mrt_matrix(lattice),
        "s_rho": value,
        "s_e": value,
        "s_eta": value,
        "s_j": value,
        "s_q": value,
        "s_v": value,
    }
    if lattice.d == 3:
        parameters |= {"s_pi": value, "s_m": value}
    if multiphase:
        parameters["kappa"] = [0.0]
    return parameters


def _relative_velocity_error(actual, expected):
    numerator = jnp.sum(jnp.square(actual - expected))
    denominator = jnp.sum(jnp.square(expected))
    return float(jnp.sqrt(numerator / denominator))


@pytest.mark.parametrize("model_class", SINGLE_PHASE_MODELS)
@pytest.mark.parametrize("lattice_class, domain, maximum_error", DOMAIN_CASES)
def test_single_phase_collision_against_taylor_green(model_class, lattice_class, domain, maximum_error):
    """Verify single-phase BGK and equivalent-rate MRT against TGV decay."""
    parameters = _base_parameters(lattice_class, domain)
    if model_class is MRTSim:
        parameters |= _mrt_relaxation_parameters(parameters["lattice"], multiphase=False)
    solver_class = type("SinglePhaseTaylorGreen", (TaylorGreenInitialFields, model_class), {})
    solver = solver_class(**parameters)

    populations = solver.run(NUMBER_OF_UPDATES - 1)
    _, velocity = solver.update_macroscopic(populations)
    _, analytical_velocity = _taylor_green_fields(domain, time=NUMBER_OF_UPDATES)

    assert _relative_velocity_error(velocity, analytical_velocity) < maximum_error


@pytest.mark.parametrize("model_class", MULTIPHASE_MODELS)
@pytest.mark.parametrize("lattice_class, domain, maximum_error", DOMAIN_CASES)
def test_multiphase_collision_against_taylor_green(model_class, lattice_class, domain, maximum_error):
    """Verify multiphase BGK and equivalent-rate MRT against TGV decay."""
    parameters = _base_parameters(lattice_class, domain)
    parameters |= {
        "n_components": 1,
        "omega": [OMEGA],
        "g_kkprime": -np.ones((1, 1)),
        "k": [1.0],
        "A": np.zeros((1, 1)),
        "EOS": VanderWaals(a=[9.0 / 49.0], b=[2.0 / 21.0], R=[1.0], T=0.8 * 0.5714285714),
    }
    if model_class is MultiphaseMRT:
        parameters |= _mrt_relaxation_parameters(parameters["lattice"], multiphase=True)
    solver_class = type("MultiphaseTaylorGreen", (MultiphaseTaylorGreenInitialFields, model_class), {})
    solver = solver_class(**parameters)

    populations = solver.run(NUMBER_OF_UPDATES - 1)
    _, velocity_tree = solver.update_macroscopic(populations)
    _, analytical_velocity = _taylor_green_fields(domain, time=NUMBER_OF_UPDATES)

    assert _relative_velocity_error(velocity_tree[0], analytical_velocity) < maximum_error
