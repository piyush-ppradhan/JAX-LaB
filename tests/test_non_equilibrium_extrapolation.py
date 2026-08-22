"""Verify NonEquilibriumExtrapolation and ExactNonEquilibriumExtrapolation (which evaluate density/velocity only
at the boundary and neighbor nodes) produce finite results and, for the exact variant, match the prescribed
boundary density.
"""

import jax.numpy as jnp
import numpy as np

from jax_lab.core.boundary_conditions import BounceBack, ExactNonEquilibriumExtrapolation, NonEquilibriumExtrapolation
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.models import BGKSim

DOMAIN = (16, 16, 16)
PRECISION = "f32/f32"
OMEGA = 1.0
SEED = 0


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


def _make_sim(bc_class):
    class Sim(BGKSim):
        def set_boundary_conditions(self):
            bottom = self.bounding_box_indices["bottom"]
            self.BCs.append(BounceBack(tuple(bottom.T), self.grid_info, self.precision_policy))
            top = self.bounding_box_indices["top"]
            prescribed_rho = 0.9 * jnp.ones((top.shape[0], 1))
            if bc_class is ExactNonEquilibriumExtrapolation:
                self.BCs.append(bc_class(tuple(top.T), self.grid_info, self.precision_policy, prescribed_rho, "density"))
            else:
                self.BCs.append(bc_class(tuple(top.T), self.grid_info, self.precision_policy, prescribed_rho))

    return Sim(**_base_kwargs())


def _random_fin(sim):
    fin = sim.assign_fields_sharded()
    rng = np.random.default_rng(SEED)
    return fin + jnp.asarray(rng.uniform(-0.05, 0.05, size=fin.shape), dtype=fin.dtype)


def test_non_equilibrium_extrapolation_produces_finite_output():
    sim = _make_sim(NonEquilibriumExtrapolation)
    fin = _random_fin(sim)

    fout = sim.apply_bc(jnp.array(fin), fin, 0, "PostStreaming")

    assert bool(jnp.all(jnp.isfinite(fout)))


def test_exact_non_equilibrium_extrapolation_matches_prescribed_density():
    sim = _make_sim(ExactNonEquilibriumExtrapolation)
    fin = _random_fin(sim)

    fout = sim.apply_bc(jnp.array(fin), fin, 0, "PostStreaming")

    top = sim.bounding_box_indices["top"]
    boundary_rho = np.asarray(jnp.sum(fout, axis=-1, keepdims=True))[tuple(top.T)]
    assert np.allclose(boundary_rho, 0.9, atol=1e-5)


if __name__ == "__main__":
    test_non_equilibrium_extrapolation_produces_finite_output()
    test_exact_non_equilibrium_extrapolation_matches_prescribed_density()
    print("NonEquilibriumExtrapolation and ExactNonEquilibriumExtrapolation behave correctly")
