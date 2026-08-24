"""
This script computes the MLUPS (Million Lattice Updates per Second) in 2D by simulating fluid flow inside a 2D cavity.
"""

import subprocess
import argparse
import jax.numpy as jnp
import numpy as np
from jax import config
from time import time

from jax_lab.core.utils import save_fields_vtk
from jax_lab.core.boundary_conditions import BounceBack, EquilibriumBC
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.models import BGKSim

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
    lattice = LatticeD2Q9(precision)

    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument("N", help="The total number of voxels will be NxN", type=int)
    parser.add_argument("timestep", help="Number of timesteps", type=int)
    args = parser.parse_args()

    n = args.N
    max_iter = args.timestep
    Re = 100.0
    u_wall = 0.1
    clength = n - 1

    visc = u_wall * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)
    logger.info(f"omega = {omega}")

    kwargs = {
        "lattice": lattice,
        "omega": omega,
        "nx": n,
        "ny": n,
        "nz": 0,
        "precision": precision,
        "compute_MLUPS": True,
    }

    subprocess.run("rm -rf ./*.vtk && rm -rf ./*.png", shell=True, check=True)
    sim = Cavity(**kwargs)
    sim.run(max_iter)
