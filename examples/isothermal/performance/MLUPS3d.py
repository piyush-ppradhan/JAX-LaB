"""
This script computes the MLUPS (Million Lattice Updates per Second) in 3D by simulating fluid flow inside a 2D cavity.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from time import time

# config.update('jax_disable_jit', True)
# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
# config.update("jax_enable_x64", True)
from jax_lab.core.utils import save_fields_vtk
from jax_lab.core.boundary_conditions import BounceBack, EquilibriumBC
from jax_lab.core.models import BGKSim
from jax_lab.core.lattice import LatticeD3Q19

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Cavity(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((
            self.bounding_box_indices["left"],
            self.bounding_box_indices["right"],
            self.bounding_box_indices["bottom"],
            self.bounding_box_indices["front"],
            self.bounding_box_indices["back"],
        ))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBack(tuple(walls.T), self.grid_info, self.precision_policy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.bounding_box_indices["top"]

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precision_policy.compute_dtype)
        vel_wall[:, 0] = u_wall
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.grid_info, self.precision_policy, rho_wall, vel_wall))


if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD3Q19(precision)
    # Create a parser that will read the command line arguments
    parser = argparse.ArgumentParser("Calculate MLUPS for a 3D cavity flow simulation")
    parser.add_argument("N", help="The total number of voxels all directions. The final dimension will be N*NxN", default=100, type=int)
    parser.add_argument("N_ITERS", help="Number of timesteps", default=10000, type=int)

    args = parser.parse_args()
    n = args.N
    n_iters = args.N_ITERS

    # Store the Reynolds number in the variable Re
    Re = 100.0
    # Store the velocity of the lid in the variable u_wall
    u_wall = 0.1
    # Store the length of the cavity in the variable clength
    clength = n - 1

    # Compute the viscosity from the Reynolds number, the lid velocity, and the length of the cavity
    visc = u_wall * clength / Re
    # Compute the relaxation parameter from the viscosity
    omega = 1.0 / (3.0 * visc + 0.5)

    kwargs = {"lattice": lattice, "omega": omega, "nx": n, "ny": n, "nz": n, "precision": precision, "compute_MLUPS": True}

    sim = Cavity(**kwargs)
    # Run the simulation
    sim.run(n_iters)
