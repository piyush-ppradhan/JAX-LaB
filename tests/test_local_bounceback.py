"""Verify the local (per-shard, int32) BounceBack path matches global bounce-back exactly."""

import jax.numpy as jnp
import numpy as np

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.eos import VanderWaals
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.models import BGKSim
from jax_lab.core.multiphase import MultiphaseBGK

DOMAIN = (32, 16, 16)
PRECISION = "f32/f32"
OMEGA = 1.0
SEED = 0


def _random_solid_indices(domain, fraction=0.1, seed=SEED):
    """Random solid-node coordinates on the interior of the domain (fixed numpy seed)."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = domain
    count = int(fraction * nx * ny * nz)
    x = rng.integers(1, nx - 1, size=count)
    y = rng.integers(1, ny - 1, size=count)
    z = rng.integers(1, nz - 1, size=count)
    return np.unique(np.stack([x, y, z], axis=-1), axis=0)


class RandomSolidSinglePhase(BGKSim):
    def set_boundary_conditions(self):
        indices = _random_solid_indices(DOMAIN)
        self.BCs.append(BounceBack(tuple(indices.T), self.grid_info, self.precision_policy))


class RandomSolidMultiphase(MultiphaseBGK):
    def set_boundary_conditions(self):
        indices = _random_solid_indices(DOMAIN)
        self.BCs[0].append(BounceBack(tuple(indices.T), self.grid_info, self.precision_policy))


def _base_kwargs():
    nx, ny, nz = DOMAIN
    return {
        "lattice": LatticeD3Q19(PRECISION),
        "omega": OMEGA,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
    }


def _expected_bounceback(fin, indices, opp_indices):
    """Ground-truth global bounce-back: reflect fin at the solid indices."""
    return fin[tuple(indices.T)][..., opp_indices]


def test_local_bounceback_matches_global_single_phase():
    sim = RandomSolidSinglePhase(**_base_kwargs())
    fin = sim.assign_fields_sharded()
    rng = np.random.default_rng(SEED + 1)
    fin = fin + jnp.asarray(rng.uniform(-0.05, 0.05, size=fin.shape), dtype=fin.dtype)

    fout = sim.apply_bc(jnp.array(fin), fin, 0, "PostCollision")

    indices = np.asarray(sim.BCs[0].indices).T
    expected = _expected_bounceback(np.asarray(fin), indices, np.asarray(sim.lattice.opp_indices))
    actual = np.asarray(fout)[tuple(indices.T)]

    assert np.max(np.abs(actual - expected)) == 0.0


def test_local_bounceback_matches_global_multiphase():
    kwargs = _base_kwargs()
    kwargs |= {
        "n_components": 1,
        "omega": [OMEGA],
        "g_kkprime": -np.ones((1, 1)),
        "k": [1.0],
        "A": np.zeros((1, 1)),
        "EOS": VanderWaals(a=[9.0 / 49.0], b=[2.0 / 21.0], R=[1.0], T=0.8 * 0.5714285714),
    }
    sim = RandomSolidMultiphase(**kwargs)
    fin_tree = sim.assign_fields_sharded()
    rng = np.random.default_rng(SEED + 1)
    fin_tree = [f + jnp.asarray(rng.uniform(-0.05, 0.05, size=f.shape), dtype=f.dtype) for f in fin_tree]

    fout_tree = sim.apply_bc([jnp.array(f) for f in fin_tree], fin_tree, 0, "PostCollision")

    indices = np.asarray(sim.BCs[0][0].indices).T
    expected = _expected_bounceback(np.asarray(fin_tree[0]), indices, np.asarray(sim.lattice.opp_indices))
    actual = np.asarray(fout_tree[0])[tuple(indices.T)]

    assert np.max(np.abs(actual - expected)) == 0.0
