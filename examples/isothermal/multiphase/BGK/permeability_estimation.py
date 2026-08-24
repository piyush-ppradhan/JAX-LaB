"""
Single phase absolute permeability estimation for a porous media. A pressure gradient is specified along the flow direction and the mean steady state flow velocity
is used for computing permeability. The simulations is run for 50,000 lattice time steps to ensure steady state is reached. An alternative way could be to have a convergence
check in the output_data function, similar to how pressure difference and error % for predicted density is measured in the droplet_2d example.
"""

import os
import subprocess
import operator
import numpy as np

from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.utils import save_fields_vtk
from jax_lab.core.multiphase import MultiphaseBGK
from jax_lab.core.boundary_conditions import BounceBack, Regularized

from functools import partial
from jax import jit, vmap, config
from jax.tree import reduce
from jax.tree import map as tree_map
import jax.numpy as jnp
import h5py
from urllib.request import urlretrieve

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


# Geometry downloaded from Digital Rocks Portal
# https://www.digitalrocksportal.org/projects/372

urlretrieve(
    "https://www.digitalrocksportal.org/projects/372/origin_data/2165/",
    "374_03_09_256.mat",
)
subprocess.run(["mv", "374_03_09_256.mat", "./assets"], check=True)


# config.update("jax_default_matmul_precision", "float32")


class PorousMedia(MultiphaseBGK):
    def initialize_macroscopic_fields(self):
        rho_tree = []

        rho = np.ones((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # apply inlet equilibrium boundary condition at the left
        inlet = self.bounding_box_indices["left"]
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        self.BCs[0].append(Regularized(tuple(inlet.T), self.grid_info, self.precision_policy, "pressure", rho_inlet))

        # Same at the outlet
        outlet = self.bounding_box_indices["right"]
        rho_outlet = 0.97 * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        self.BCs[0].append(Regularized(tuple(outlet.T), self.grid_info, self.precision_policy, "pressure", rho_outlet))

        # Wall boundary condition
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        ind = np.where(binary == 1.0)
        idx = np.zeros((len(ind[0]), 3), dtype=int)
        idx[:, 0] = ind[0]
        idx[:, 1] = ind[1]
        idx[:, 2] = ind[2]
        wall = np.concatenate((
            idx,
            self.bounding_box_indices["top"],
            self.bounding_box_indices["bottom"],
            self.bounding_box_indices["front"],
            self.bounding_box_indices["back"],
        ))
        self.BCs[0].append(
            BounceBack(tuple(wall.T), self.grid_info, self.precision_policy, theta[tuple(wall.T)], phi[tuple(wall.T)], delta_rho[tuple(wall.T)])
        )

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        U_tree = tree_map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree):
        def f(g_kk):
            return reduce(operator.add, tree_map(lambda _gkk, psi: _gkk * psi, list(g_kk), psi_tree))

        return tree_map(lambda rho, psi, nt: rho / 3 + 1.5 * psi * nt, rho_tree, psi_tree, list(vmap(f, in_axes=(0,))(self.g_kkprime)))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs.get("rho_tree")[0][0, ...])
        p = np.array(kwargs["p_tree"][0])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        # HDF5/XDMF output option:
        # from jax_lab.core.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, "output", "data")
        save_fields_vtk(timestep, fields, "output", "data")


class PorousMediaGeometric(PorousMedia):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        # apply inlet equilibrium boundary condition at the left
        inlet = self.bounding_box_indices["left"]
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        self.BCs[0].append(Regularized(tuple(inlet.T), self.grid_info, self.precision_policy, "pressure", rho_inlet))

        # Same at the outlet
        outlet = self.bounding_box_indices["right"]
        rho_outlet = 0.97 * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        self.BCs[0].append(Regularized(tuple(outlet.T), self.grid_info, self.precision_policy, "pressure", rho_outlet))

        # Wall boundary condition
        ind = np.where(binary == 1.0)
        idx = np.zeros((len(ind[0]), 3), dtype=int)
        idx[:, 0] = ind[0]
        idx[:, 1] = ind[1]
        idx[:, 2] = ind[2]
        wall = np.concatenate((
            idx,
            self.bounding_box_indices["top"],
            self.bounding_box_indices["bottom"],
            self.bounding_box_indices["front"],
            self.bounding_box_indices["back"],
        ))
        wall = tuple(wall.T)
        self.BCs[0].append(BounceBack(wall, self.grid_info, self.precision_policy, theta[wall]))


if __name__ == "__main__":
    precision = "f32/f32"
    g_kkprime = 0 * np.ones((1, 1))
    nx = 256
    ny = 256
    nz = 256
    geometry = h5py.File("./assets/374_09_03_256.mat", "r")
    binary = np.array(geometry["bin"], dtype=int)

    theta = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    phi = np.ones((nx, ny, nz, 1))
    delta_rho = np.zeros((nx, ny, nz, 1))

    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": g_kkprime,
        "omega": [1.0],
        "wetting_formulation": "improved_virtual_density",
        "precision": precision,
        "k": [0],
        "A": np.zeros((1, 1)),
        "io_rate": 1000,
        "compute_MLUPS": False,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    # os.system("rm -rf output*/ *.vtk")
    sim = PorousMedia(**kwargs)
    sim.run(50000)
