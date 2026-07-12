"""
Tests for the hybrid thermal LBM solver: lattice difference stencils
(grad_x, laplacian_x), Runge-Kutta diffusion against the analytic decay of a
sinusoid, and the Dirichlet/Neumann thermal boundary conditions.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_lab.lattice import LatticeD2Q9, LatticeD3Q19
from jax_lab.models import BGKSim
from jax_lab.thermal import DirichletTemperature, NeumannTemperature, Thermal

PRECISION = "f64/f64"
NX = NY = NZ = 32


@pytest.fixture(autouse=True)
def enable_x64():
    import jax

    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def make_thermal(dim, K=0.05, c_v=1.0):
    lattice = LatticeD2Q9(PRECISION) if dim == 2 else LatticeD3Q19(PRECISION)
    fluid = BGKSim(
        lattice=lattice,
        omega=1.0,
        nx=NX,
        ny=NY,
        nz=0 if dim == 2 else NZ,
        precision=PRECISION,
        io_rate=0,
        print_info_rate=0,
    )
    return Thermal(fluid_solver=fluid, specific_heat=c_v, thermal_conductivity=K)


def sinusoid(dim):
    k = 2.0 * np.pi / NX
    x = np.arange(NX)
    if dim == 2:
        field = np.sin(k * x)[:, None] * np.ones((1, NY))
        return jnp.array(field[..., None]), k
    field = np.sin(k * x)[:, None, None] * np.ones((1, NY, NZ))
    return jnp.array(field[..., None]), k


@pytest.mark.parametrize("dim", [2, 3])
def test_grad_x_of_sinusoid(dim):
    solver = make_thermal(dim)
    T, k = sinusoid(dim)
    grad = solver.grad_x(T)

    x = np.arange(NX)
    expected_x = k * np.cos(k * x)
    # The lattice stencil applied to sin(kx) yields sin(k) / k * expected (second order accurate)
    scale = np.sin(k) / k
    got_x = np.array(grad[..., 0]).mean(axis=tuple(range(1, dim)))
    assert np.allclose(got_x, scale * expected_x, atol=1e-8)
    # No variation along the other axes
    for axis in range(1, dim):
        assert np.allclose(np.array(grad[..., axis]), 0.0, atol=1e-8)


@pytest.mark.parametrize("dim", [2, 3])
def test_laplacian_x_of_sinusoid(dim):
    solver = make_thermal(dim)
    T, k = sinusoid(dim)
    lap = solver.laplacian_x(T)

    # The discrete laplacian of sin(kx) is -k_eff^2 sin(kx) with
    # k_eff^2 = 2(1 - cos k) (central difference symbol)
    k_eff2 = 2.0 * (1.0 - np.cos(k))
    assert np.allclose(np.array(lap), -k_eff2 * np.array(T), atol=1e-8)


def test_rk4_diffusion_decay_2d():
    K = 0.05
    solver = make_thermal(2, K=K)
    T0, k = sinusoid(2)
    T_mean = 1.0
    T = T_mean + T0

    rho = jnp.ones((NX, NY, 1))
    u = jnp.zeros((NX, NY, 2))

    n_steps = 200
    for _ in range(n_steps):
        k1 = solver.RHS(T, rho, u)
        k2 = solver.RHS(T + 0.5 * k1, rho, u)
        k3 = solver.RHS(T + 0.5 * k2, rho, u)
        k4 = solver.RHS(T + k3, rho, u)
        T = T + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    # Analytic decay with the discrete wavenumber of the stencil
    k_eff2 = 2.0 * (1.0 - np.cos(k))
    expected = T_mean + np.exp(-K * k_eff2 * n_steps) * np.array(T0)
    assert np.allclose(np.array(T), expected, rtol=1e-4, atol=1e-6)


def test_step_conserves_uniform_temperature():
    solver = make_thermal(2)
    T = solver.assign_fields_sharded()
    f = solver.fluid_solver.assign_fields_sharded()
    T_new, f, _ = solver.step(T, f, 0)
    assert np.allclose(np.array(T_new), 1.0, atol=1e-12)


def test_dirichlet_and_neumann_bc():
    solver = make_thermal(2)
    bottom = solver.fluid_solver.boundingBoxIndices["bottom"]
    left = solver.fluid_solver.boundingBoxIndices["left"]

    T = jnp.array(np.random.default_rng(0).uniform(size=(NX, NY, 1)))

    dirichlet = DirichletTemperature(tuple(bottom.T), prescribed=2.5)
    T_d = dirichlet.apply(T, 0)
    assert np.allclose(np.array(T_d[:, 0, 0]), 2.5)

    neumann = NeumannTemperature(tuple(left.T), normal=(-1, 0), prescribed=0.0)
    T_n = neumann.apply(T, 0)
    # Adiabatic: boundary nodes copy their interior neighbor
    assert np.allclose(np.array(T_n[0, :, 0]), np.array(T[1, :, 0]))

    neumann_flux = NeumannTemperature(tuple(left.T), normal=(-1, 0), prescribed=0.1)
    T_q = neumann_flux.apply(T, 0)
    assert np.allclose(np.array(T_q[0, :, 0]), np.array(T[1, :, 0]) + 0.1)
