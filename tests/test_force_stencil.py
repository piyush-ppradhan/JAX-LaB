"""Verify multiphase force stencils against periodic NumPy references."""

import numpy as np

from jax_lab.core.eos import VanderWaals
from jax_lab.core.lattice import LatticeD2Q9, LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseBGK

DOMAIN_3D = (12, 12, 12)
DOMAIN_2D = (16, 16, 0)
PRECISION = "f32/f32"
SEED = 0


def _reference_force_stencil(field, weights, dim):
    """Compute the weighted directional neighbor sum with periodic rolls."""
    total = np.zeros((*field.shape[:-1], dim))
    for direction, w in zip(*weights):
        if np.all(w == 0.0):
            continue
        shifted = np.roll(field[..., 0], direction, axis=tuple(range(dim)))
        total += shifted[..., None] * w
    return total


def _reference_compute_fluid_fluid_force(psi_arrays, U_arrays, A, g_kkprime, G_ff, c, dim):
    directions = c.T  # (q, dim)
    weights = (directions, G_ff[:, None] * c.T)  # paired for _reference_force_stencil's zip
    psi_stencil = [_reference_force_stencil(psi, weights, dim) for psi in psi_arrays]
    U_stencil = [_reference_force_stencil(U, weights, dim) for U in U_arrays]

    n = len(psi_arrays)
    forces = []
    for k in range(n):
        ffk_1 = sum((1 - A[k, j]) * g_kkprime[k, j] * psi_stencil[j] for j in range(n))
        ffk_2 = sum(A[k, j] * U_stencil[j] for j in range(n))
        forces.append(psi_arrays[k] * ffk_1 + ffk_2)
    return forces


def _check(domain, lattice_class):
    dim = 2 if domain[2] == 0 else 3
    nx, ny, nz = domain
    lattice = lattice_class(PRECISION)

    rng = np.random.default_rng(SEED)
    A = rng.uniform(0.0, 0.3, size=(2, 2))
    A = (A + A.T) / 2  # A must combine with a symmetric-looking interaction; off-diagonal exercised either way
    g_kkprime = np.array([[-1.0, -0.5], [-0.5, -1.0]])

    kwargs = {
        "lattice": lattice,
        "omega": [1.0, 1.0],
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
        "n_components": 2,
        "g_kkprime": g_kkprime,
        "k": [1.0, 1.0],
        "A": A,
        "EOS": VanderWaals(a=[9.0 / 49.0, 9.0 / 49.0], b=[2.0 / 21.0, 2.0 / 21.0], R=[1.0, 1.0], T=0.8 * 0.5714285714),
    }
    sim = MultiphaseBGK(**kwargs)

    spatial_shape = domain[:2] if dim == 2 else domain
    rng = np.random.default_rng(SEED + 1)
    psi_host = [rng.uniform(0.1, 1.0, size=(*spatial_shape, 1)).astype(np.float32) for _ in range(2)]
    U_host = [rng.uniform(-0.1, 0.1, size=(*spatial_shape, 1)).astype(np.float32) for _ in range(2)]
    psi_tree = [sim.distributed_array_init(p.shape, sim.precision_policy.compute_dtype, init_val=p) for p in psi_host]
    U_tree = [sim.distributed_array_init(u.shape, sim.precision_policy.compute_dtype, init_val=u) for u in U_host]

    actual = [np.asarray(f) for f in sim.compute_fluid_fluid_force(psi_tree, U_tree)]

    c = np.asarray(lattice.c)
    G_ff = np.asarray(sim.G_ff)
    expected = _reference_compute_fluid_fluid_force(psi_host, U_host, A, g_kkprime, G_ff, c, dim)

    for a, e in zip(actual, expected, strict=True):
        assert np.max(np.abs(a - e)) < 1e-5


def test_compute_fluid_fluid_force_matches_reference_3d():
    _check(DOMAIN_3D, LatticeD3Q19)


def test_compute_fluid_fluid_force_matches_reference_2d():
    _check(DOMAIN_2D, LatticeD2Q9)
