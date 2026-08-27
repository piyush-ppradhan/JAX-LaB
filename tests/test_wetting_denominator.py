"""Verify wetting-density averaging against a periodic NumPy reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.eos import VanderWaals
from jax_lab.core.lattice import LatticeD2Q9, LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseBGK, MultiphaseMRT

DOMAIN_3D = (20, 20, 20)
DOMAIN_2D = (24, 24, 0)
PRECISION = "f32/f32"
SEED = 0


def _random_solid_mask(domain, fraction=0.08, seed=SEED):
    rng = np.random.default_rng(seed)
    spatial_shape = domain[:2] if domain[2] == 0 else domain
    mask = rng.random(spatial_shape) < fraction
    return mask


def _reference_average_density(rho, mask, lattice):
    """Compute the weighted fluid-neighbor average with periodic rolls."""
    c = np.asarray(lattice.c).T
    weights = np.zeros(c.shape[0])
    norms = np.linalg.norm(c, axis=1)
    is_d2q9 = c.shape[1] == 2
    g1, g2 = (1 / 3, 1 / 12) if is_d2q9 else (1 / 6, 1 / 12)
    weights[np.isclose(norms, 1.0)] = g1
    weights[np.isclose(norms, np.sqrt(2.0))] = g2

    fluid = (1.0 - mask.astype(np.float64))[..., None]
    rho = rho[..., None] if rho.ndim == mask.ndim else rho
    numerator = np.zeros_like(fluid[..., 0])
    denominator = np.zeros_like(fluid[..., 0])
    for direction, weight in zip(c, weights):
        if weight == 0.0:
            continue
        axis = tuple(range(len(direction)))
        shifted_rho = np.roll(rho[..., 0], direction, axis=axis)
        shifted_fluid = np.roll(fluid[..., 0], direction, axis=axis)
        numerator += weight * shifted_rho * shifted_fluid
        denominator += weight * shifted_fluid
    return numerator / denominator


class RandomSolidWetting(MultiphaseMRT):
    def __init__(self, mask, **kwargs):
        self._mask = mask
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        ind = np.array(np.where(self._mask)).T
        self.BCs[0].append(BounceBack(tuple(ind.T), self.grid_info, self.precision_policy, np.pi / 3, 1.1, 0.0))


class RandomSolidNoWetting(RandomSolidWetting):
    def set_boundary_conditions(self):
        ind = np.array(np.where(self._mask)).T
        self.BCs[0].append(BounceBack(tuple(ind.T), self.grid_info, self.precision_policy))


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


def _simulation_kwargs(domain, lattice_class):
    nx, ny, nz = domain
    lattice = lattice_class(PRECISION)
    omega = 0.8
    s = [omega]
    kwargs = {
        "lattice": lattice,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
        "n_components": 1,
        "omega": [omega],
        "g_kkprime": -np.ones((1, 1)),
        "k": [1.0],
        "A": np.zeros((1, 1)),
        "EOS": VanderWaals(a=[9.0 / 49.0], b=[2.0 / 21.0], R=[1.0], T=0.8 * 0.5714285714),
        "M": [_mrt_matrix(lattice)],
        "s_rho": s,
        "s_e": s,
        "s_eta": s,
        "s_j": s,
        "s_q": s,
        "s_v": s,
        "kappa": [0.0],
    }
    if lattice.d == 3:
        kwargs |= {"s_pi": s, "s_m": s}
    return kwargs


def _build_sim(domain, lattice_class, mask):
    kwargs = _simulation_kwargs(domain, lattice_class)
    kwargs["wetting_formulation"] = "improved_virtual_density"
    return RandomSolidWetting(mask, **kwargs)


def _check(domain, lattice_class):
    mask = _random_solid_mask(domain)
    sim = _build_sim(domain, lattice_class, mask)

    rng = np.random.default_rng(SEED + 1)
    spatial_shape = domain[:2] if domain[2] == 0 else domain
    rho_host = rng.uniform(0.5, 5.0, size=(*spatial_shape, 1)).astype(np.float32)
    rho = sim.distributed_array_init(rho_host.shape, sim.precision_policy.compute_dtype, init_val=rho_host)

    actual = np.asarray(sim.compute_average_density([rho])[0])[..., 0]
    expected = _reference_average_density(rho_host[..., 0], mask, sim.lattice)

    assert np.max(np.abs(actual - expected)) < 1e-5


def test_compute_average_density_matches_reference_3d():
    _check(DOMAIN_3D, LatticeD3Q19)


def test_compute_average_density_matches_reference_2d():
    _check(DOMAIN_2D, LatticeD2Q9)


def test_no_wetting_does_not_build_average_density_data():
    mask = _random_solid_mask(DOMAIN_2D)
    sim = RandomSolidNoWetting(mask, **_simulation_kwargs(DOMAIN_2D, LatticeD2Q9))

    assert sim.scalar_neighbor_sum is None
    assert sim.average_density_denominator is None
    assert sim.local_geometric_wetting is None
    assert sim.geometric_wetting_data is None
    assert sim.geometric_fluid_mask is None


def test_wetting_boundary_requires_explicit_formulation():
    mask = _random_solid_mask(DOMAIN_2D)
    with pytest.raises(ValueError, match="wetting_formulation must be selected"):
        RandomSolidWetting(mask, **_simulation_kwargs(DOMAIN_2D, LatticeD2Q9))


@pytest.mark.skipif(jax.device_count() < 2, reason="Multiple devices required for local wetting-index coverage")
def test_local_improved_wetting_matches_reference_for_multiple_components():
    nx, ny = 24, 18
    mask_0 = np.zeros((nx, ny), dtype=bool)
    mask_1 = np.zeros((nx, ny), dtype=bool)
    mask_0[1::4, 2::3] = True
    mask_1[2::5, 1::4] = True
    indices_0 = np.asarray(np.where(mask_0)).T
    indices_1 = np.asarray(np.where(mask_1)).T

    class TwoComponentWetting(MultiphaseBGK):
        def set_boundary_conditions(self):
            self.BCs[0].append(BounceBack(tuple(indices_0.T), self.grid_info, self.precision_policy, np.pi / 3, 1.1, 0.0))
            theta_1 = np.full((nx, ny, 1), 2.0 * np.pi / 3, dtype=np.float32)
            self.BCs[1].append(BounceBack(tuple(indices_1.T), self.grid_info, self.precision_policy, theta_1, 0.9, 0.2))

    sim = TwoComponentWetting(
        lattice=LatticeD2Q9(PRECISION),
        omega=[1.0, 1.0],
        nx=nx,
        ny=ny,
        nz=0,
        precision=PRECISION,
        io_rate=0,
        print_info_rate=0,
        checkpoint_rate=0,
        n_components=2,
        g_kkprime=np.array([[-1.0, -0.25], [-0.25, -1.0]]),
        k=[1.0, 1.0],
        A=np.zeros((2, 2)),
        EOS=VanderWaals(a=[9.0 / 49.0] * 2, b=[2.0 / 21.0] * 2, R=[1.0, 1.0], T=0.8 * 0.5714285714),
        wetting_formulation="improved_virtual_density",
    )
    rng = np.random.default_rng(SEED + 3)
    rho_tree = [
        sim.distributed_array_init((nx, ny, 1), jnp.float32, init_val=rng.uniform(0.5, 4.0, size=(nx, ny, 1)).astype(np.float32)) for _ in range(2)
    ]
    rho_ave_tree = sim.compute_average_density(rho_tree)
    expected = [np.asarray(rho).copy() for rho in rho_tree]
    expected[0][tuple(indices_0.T)] = 1.1 * np.asarray(rho_ave_tree[0])[tuple(indices_0.T)]
    expected[1][tuple(indices_1.T)] = np.asarray(rho_ave_tree[1])[tuple(indices_1.T)] - 0.2
    expected = [np.clip(value, np.min(np.asarray(rho)), np.max(np.asarray(rho))) for value, rho in zip(expected, rho_tree, strict=True)]

    actual = sim.apply_contact_angle(rho_tree)
    assert sim.local_geometric_wetting is None
    assert sim.geometric_wetting_data is None
    assert sim.geometric_fluid_mask is None
    assert all(data is not None for data in (sim.local_improved_wetting_data[0][0], sim.local_improved_wetting_data[1][0]))
    for actual_component, expected_component in zip(actual, expected, strict=True):
        assert np.max(np.abs(np.asarray(actual_component) - expected_component)) < 1e-6


def _reference_geometric_wetting(rho, component_data, fluid_mask):
    rho = rho.copy()
    rho_min = np.min(rho[fluid_mask])
    rho_max = np.max(rho[fluid_mask])

    def interpolate(point_data):
        if rho.ndim == 3:
            x0, y0, x1, y1, w00, w10, w01, w11 = point_data
            return w00[:, None] * rho[x0, y0] + w10[:, None] * rho[x1, y0] + w01[:, None] * rho[x0, y1] + w11[:, None] * rho[x1, y1]
        x0, y0, z0, x1, y1, z1, w000, w100, w010, w110, w001, w101, w011, w111 = point_data
        return (
            w000[:, None] * rho[x0, y0, z0]
            + w100[:, None] * rho[x1, y0, z0]
            + w010[:, None] * rho[x0, y1, z0]
            + w110[:, None] * rho[x1, y1, z0]
            + w001[:, None] * rho[x0, y0, z1]
            + w101[:, None] * rho[x1, y0, z1]
            + w011[:, None] * rho[x0, y1, z1]
            + w111[:, None] * rho[x1, y1, z1]
        )

    for data in component_data:
        point_data = data["points"] if rho.ndim == 4 else (data["point_1"], data["point_2"])
        samples = np.stack([interpolate(point) for point in point_data], axis=-2)
        wall_density = np.where(data["theta"] <= np.pi / 2, np.max(samples, axis=-2), np.min(samples, axis=-2))
        rho[data["indices"]] = np.clip(wall_density, rho_min, rho_max)
    return rho


def _assert_geometric_samples_use_fluid_nodes(component_data, fluid_mask):
    for data in component_data:
        point_data = data["points"] if fluid_mask.ndim == 3 else (data["point_1"], data["point_2"])
        for point in point_data:
            if fluid_mask.ndim == 2:
                x0, y0, x1, y1, w00, w10, w01, w11 = point
                samples = ((x0, y0, w00), (x1, y0, w10), (x0, y1, w01), (x1, y1, w11))
            else:
                x0, y0, z0, x1, y1, z1, w000, w100, w010, w110, w001, w101, w011, w111 = point
                samples = (
                    (x0, y0, z0, w000),
                    (x1, y0, z0, w100),
                    (x0, y1, z0, w010),
                    (x1, y1, z0, w110),
                    (x0, y0, z1, w001),
                    (x1, y0, z1, w101),
                    (x0, y1, z1, w011),
                    (x1, y1, z1, w111),
                )
            for *sample_indices, weights in samples:
                assert np.all(fluid_mask[tuple(sample_indices)] | (weights <= 1e-12))


def test_geometric_3d_rejects_rays_crossing_solid_and_underresolved_force_nodes():
    domain = (16, 15, 15)
    solid = np.ones(domain, dtype=bool)
    wall = np.array([[7, 7, 3]], dtype=np.int32)
    solid[7, 7:9, 4] = False  # One local fluid-side interpolation stencil.
    solid[:, :, 10:] = False  # Remote pore reached only after crossing solid.
    sim = RandomSolidNoWetting(solid, **_simulation_kwargs(domain, LatticeD3Q19))
    sim.geometric_preprocessing_backend = "cpu"

    points = sim._build_geometric_3d_characteristic_data(
        wall,
        normals=np.array([[0.0, 0.0, 1.0]]),
        theta=np.array([np.pi / 3]),
        solid_mask=solid,
    )
    for point_data in points:
        x0, y0, z0, x1, y1, z1, w000, w100, w010, w110, w001, w101, w011, w111 = point_data
        weights = np.stack((w000, w100, w010, w110, w001, w101, w011, w111), axis=1)
        corners = np.stack(
            (
                np.column_stack((x0, y0, z0)),
                np.column_stack((x1, y0, z0)),
                np.column_stack((x0, y1, z0)),
                np.column_stack((x1, y1, z0)),
                np.column_stack((x0, y0, z1)),
                np.column_stack((x1, y0, z1)),
                np.column_stack((x0, y1, z1)),
                np.column_stack((x1, y1, z1)),
            ),
            axis=1,
        )
        sample = np.sum(weights[..., None] * corners, axis=1)
        np.testing.assert_allclose(sample, np.array([[7.0, 7.0, 4.0]]), rtol=0.0, atol=1e-6)

    fluid = np.zeros((5, 5, 5), dtype=bool)
    fluid[2, 2, 2] = True
    fluid[3, 2, 2] = True
    fluid[2, 3, 2] = True
    assert not sim._create_geometric_force_mask(fluid)[2, 2, 2]
    fluid[1, 2, 2] = True
    assert sim._create_geometric_force_mask(fluid)[2, 2, 2]


@pytest.mark.skipif(jax.device_count() < 2, reason="Multiple devices required for local geometric-wetting coverage")
@pytest.mark.parametrize(
    "domain, lattice_class, n_components",
    (
        pytest.param((8, 7, 0), LatticeD2Q9, 2, id="2d-multicomponent"),
        pytest.param((4, 4, 4), LatticeD3Q19, 1, id="3d"),
    ),
)
def test_local_geometric_wetting_matches_global_reference(domain, lattice_class, n_components):
    class GeometricWetting(MultiphaseBGK):
        def set_boundary_conditions(self):
            cases = (
                ((0, "bottom", np.pi / 3), (0, "left", 2.0 * np.pi / 3)) if self.dim == 3 else ((0, "bottom", np.pi / 3), (1, "top", 2.0 * np.pi / 3))
            )
            for component, face, theta in cases:
                indices = self.bounding_box_indices[face]
                self.BCs[component].append(BounceBack(tuple(indices.T), self.grid_info, self.precision_policy, theta))

    nx, ny, nz = domain
    sim = GeometricWetting(
        lattice=lattice_class(PRECISION),
        omega=[1.0] * n_components,
        nx=nx,
        ny=ny,
        nz=nz,
        precision=PRECISION,
        io_rate=0,
        print_info_rate=0,
        checkpoint_rate=0,
        n_components=n_components,
        g_kkprime=-np.eye(n_components),
        k=[1.0] * n_components,
        A=np.zeros((n_components, n_components)),
        EOS=VanderWaals(a=[9.0 / 49.0] * n_components, b=[2.0 / 21.0] * n_components, R=[1.0] * n_components, T=0.8 * 0.5714285714),
        wetting_formulation="geometric",
        geometric_preprocessing_backend="gpu" if nz != 0 else "cpu",
    )
    if nz != 0:
        sim.geometric_preprocessing_backend = "cpu"
        global_data, global_masks = sim._create_geometric_wetting_data(localize=False)
        sim.geometric_preprocessing_backend = "gpu"
        gpu_data, _ = sim._create_geometric_wetting_data(localize=False)
        for gpu_leaf, cpu_leaf in zip(jax.tree.leaves(gpu_data), jax.tree.leaves(global_data), strict=True):
            np.testing.assert_array_equal(gpu_leaf, cpu_leaf)
    else:
        global_data, global_masks = sim._create_geometric_wetting_data(localize=False)
    for component_data, fluid_mask in zip(global_data, global_masks, strict=True):
        _assert_geometric_samples_use_fluid_nodes(component_data, np.asarray(fluid_mask)[..., 0])
    spatial_shape = domain[:2] if nz == 0 else domain
    rng = np.random.default_rng(SEED + 4)
    rho_host_tree = [rng.uniform(0.5, 4.0, size=(*spatial_shape, 1)).astype(np.float32) for _ in range(n_components)]
    rho_tree = [sim.distributed_array_init(rho.shape, jnp.float32, init_val=rho) for rho in rho_host_tree]

    actual_tree = sim.apply_contact_angle(rho_tree)
    expected_tree = [
        _reference_geometric_wetting(rho, data, np.asarray(mask)[..., 0])
        for rho, data, mask in zip(rho_host_tree, global_data, global_masks, strict=True)
    ]

    for component_data in sim.geometric_wetting_data:
        for data in component_data:
            assert data["local_indices"].dtype == jnp.int32
            assert data["local_select_max"].dtype == jnp.bool_
            assert data["request_indices"].dtype == jnp.int32
            if "request_slots" in data:
                assert data["request_slots"].dtype == jnp.int32
                template = data.get("reduce_scatter_template", data.get("sample_template"))
                assert template.dtype == jnp.bool_
                assert data["request_indices"].shape[1] < np.prod(template.shape[1:])
            else:
                assert data["local_sample_indices"].dtype == jnp.int32
    for actual, expected in zip(actual_tree, expected_tree, strict=True):
        assert np.max(np.abs(np.asarray(actual) - expected)) < 1e-6
