"""End-to-end tests for single- and multiphase checkpoint restart."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jax_lab.core.lattice import LatticeD2Q9, LatticeD3Q19
from jax_lab.core.models import BGKSim
from jax_lab.core.multiphase import MultiphaseBGK


SINGLE_PHASE_PRECISION_CASES = (
    pytest.param("f16/f16", 2e-3, id="f16-f16"),
    pytest.param("f32/f32", 1e-6, id="f32-f32"),
    pytest.param("f64/f64", 1e-12, id="f64-f64"),
)

pytestmark = pytest.mark.slow


def _sinusoidal_phase(spatial_shape, dtype, phase=0.0):
    coordinates = jnp.meshgrid(
        *[2.0 * jnp.pi * jnp.arange(size, dtype=dtype) / size for size in spatial_shape],
        indexing="ij",
    )
    return sum(jnp.sin(coordinate + phase) for coordinate in coordinates) / len(spatial_shape)


class SinusoidalBGK(BGKSim):
    """Single-phase BGK solver with sinusoidal initial density and velocity."""

    def initialize_macroscopic_fields(self):
        spatial_shape = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        phase = _sinusoidal_phase(spatial_shape, self.precision_policy.compute_dtype)
        density = (1.0 + 0.01 * phase)[..., None]
        velocity = jnp.zeros((*spatial_shape, self.dim), dtype=self.precision_policy.compute_dtype)
        for axis in range(self.dim):
            velocity = velocity.at[..., axis].set(0.005 * jnp.roll(phase, shift=axis, axis=axis))
        return density, velocity


class SinusoidalMultiphaseBGK(MultiphaseBGK):
    """Two-component BGK solver with sinusoidal fields and no interactions."""

    def initialize_macroscopic_fields(self):
        spatial_shape = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        first_phase = _sinusoidal_phase(spatial_shape, self.precision_policy.compute_dtype)
        second_phase = _sinusoidal_phase(spatial_shape, self.precision_policy.compute_dtype, phase=jnp.pi / 3.0)
        density_tree = [
            (1.0 + 0.01 * first_phase)[..., None],
            (0.8 + 0.01 * second_phase)[..., None],
        ]

        velocity_tree = []
        for phase, direction in ((first_phase, 1.0), (second_phase, -1.0)):
            velocity = jnp.zeros(
                (*spatial_shape, self.dim),
                dtype=self.precision_policy.compute_dtype,
            )
            for axis in range(self.dim):
                velocity = velocity.at[..., axis].set(direction * 0.005 * jnp.roll(phase, shift=axis, axis=axis))
            velocity_tree.append(velocity)
        return density_tree, velocity_tree

    def compute_force(self, rho_tree, T=None):
        return [
            jnp.zeros(
                (*density.shape[:-1], self.dim),
                dtype=self.precision_policy.compute_dtype,
            )
            for density in rho_tree
        ]


def _solver_parameters(lattice_class, domain, precision, checkpoint_dir, checkpoint_rate):
    nx, ny, nz = domain
    return {
        "lattice": lattice_class(precision),
        "omega": 1.0,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": precision,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": checkpoint_rate,
        "checkpoint_dir": str(checkpoint_dir),
        "restore_checkpoint": False,
    }


def _run_with_checkpoint_restart(solver_class, parameters):
    checkpoint_solver = solver_class(**parameters)
    checkpoint_solver.run(10)
    checkpoint_solver.mngr.wait_until_finished()
    assert checkpoint_solver.mngr.latest_step() == 10
    checkpoint_solver.mngr.close()

    restart_solver = solver_class(**(parameters | {"restore_checkpoint": True}))
    restarted_output = restart_solver.run(20)
    restart_solver.mngr.wait_until_finished()
    restart_solver.mngr.close()
    return restarted_output


@pytest.mark.parametrize("precision, tolerance", SINGLE_PHASE_PRECISION_CASES)
def test_single_phase_bgk_checkpoint_restart(tmp_path, precision, tolerance):
    """Verify restart equivalence for representative precision policies."""
    parameters = _solver_parameters(LatticeD2Q9, (8, 7, 0), precision, tmp_path / "single", checkpoint_rate=10)
    restarted_output = _run_with_checkpoint_restart(SinusoidalBGK, parameters)

    continuous_solver = SinusoidalBGK(**(parameters | {"checkpoint_rate": 0}))
    continuous_output = continuous_solver.run(20)

    np.testing.assert_allclose(
        np.asarray(restarted_output),
        np.asarray(continuous_output),
        rtol=0.0,
        atol=tolerance,
    )
    assert restarted_output.dtype == continuous_output.dtype


def test_multiphase_bgk_checkpoint_restart(tmp_path):
    """Verify restart equivalence for a 3D multiphase pytree."""
    parameters = _solver_parameters(
        LatticeD3Q19,
        (6, 5, 4),
        "f32/f32",
        tmp_path / "multiphase",
        checkpoint_rate=10,
    )
    parameters |= {
        "n_components": 2,
        "omega": [1.0, 1.0],
        "g_kkprime": np.zeros((2, 2)),
        "k": [0.0, 0.0],
        "A": np.zeros((2, 2)),
    }
    restarted_output = _run_with_checkpoint_restart(SinusoidalMultiphaseBGK, parameters)

    continuous_solver = SinusoidalMultiphaseBGK(**(parameters | {"checkpoint_rate": 0}))
    continuous_output = continuous_solver.run(20)

    assert len(restarted_output) == len(continuous_output) == 2
    for restarted_component, continuous_component in zip(restarted_output, continuous_output):
        np.testing.assert_allclose(
            np.asarray(restarted_component),
            np.asarray(continuous_component),
            rtol=0.0,
            atol=1e-6,
        )
        assert restarted_component.dtype == continuous_component.dtype
