"""Verify Multiphase.compute_average_density (scalar mask + cached denominator) against an independent
G_ff-weighted neighbor-average reference computed with plain (periodic) numpy rolls.
"""

import numpy as np

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.eos import VanderWaals
from jax_lab.core.lattice import LatticeD2Q9, LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseMRT

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
    """Independent G_ff-weighted neighbor average, using plain periodic numpy rolls (matches scalar_neighbor_sum_m
    exactly when n_devices == 1, since the x-halo ppermute exchange degenerates to a periodic self-roll)."""
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


def _build_sim(domain, lattice_class, mask):
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


if __name__ == "__main__":
    test_compute_average_density_matches_reference_3d()
    test_compute_average_density_matches_reference_2d()
    print("compute_average_density matches the independent reference")
