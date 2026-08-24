import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_ENABLE_X64"] = "1"

import operator
import numpy as np

from jax_lab.core.lattice import LatticeD3Q19

from jax_lab.core.multiphase import MultiphaseBGK
from jax_lab.core.boundary_conditions import BounceBack, Regularized

from functools import partial
from jax import jit, vmap, config
from jax.tree import reduce
from jax.tree import map as tree_map

import jax.numpy as jnp
import h5py

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


class PorousMedia(MultiphaseBGK):
    def initialize_macroscopic_fields(self):
        rho_tree = []

        rho = np.ones((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init(
            (self.nx, self.ny, self.nz, 1),
            self.precision_policy.compute_dtype,
            init_val=rho,
        )
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init(
            (self.nx, self.ny, self.nz, 3),
            self.precision_policy.compute_dtype,
            init_val=u,
        )
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # apply inlet equilibrium boundary condition at the left
        inlet = self.bounding_box_indices["left"]
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        self.BCs[0].append(
            Regularized(
                tuple(inlet.T),
                self.grid_info,
                self.precision_policy,
                "pressure",
                rho_inlet,
            )
        )

        # Same at the outlet
        outlet = self.bounding_box_indices["right"]
        rho_outlet = 0.97 * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        self.BCs[0].append(
            Regularized(
                tuple(outlet.T),
                self.grid_info,
                self.precision_policy,
                "pressure",
                rho_outlet,
            )
        )

        # Wall boundary condition
        wall = np.concatenate((
            idx,
            self.bounding_box_indices["top"],
            self.bounding_box_indices["bottom"],
            self.bounding_box_indices["front"],
            self.bounding_box_indices["back"],
        ))
        self.BCs[0].append(
            BounceBack(
                tuple(wall.T),
                self.grid_info,
                self.precision_policy,
                theta[tuple(wall.T)],
                phi[tuple(wall.T)],
                delta_rho[tuple(wall.T)],
            )
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


if __name__ == "__main__":
    g_kkprime = 0 * np.ones((1, 1))
    nx = 480
    ny = 300
    nz = 300
    geometry = h5py.File("./assets/574_05_480.mat", "r")
    binary = np.array(geometry["bin"], dtype=int)

    # Obtained from https://github.com/cageo/Krzikalla-2012/
    # nx = 512
    # ny = 512
    # nz = 512
    # geometry = h5py.File("./assets/segmented-kongju.mat", "r")
    # binary = np.array(geometry["berea"], dtype=int)

    binary = binary[:, 0:ny, 0:nz]
    ind = np.where(binary == 1.0)

    theta = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    phi = np.ones((nx, ny, nz, 1))
    delta_rho = np.zeros((nx, ny, nz, 1))

    idx = np.zeros((len(ind[0]), 3), dtype=int)
    idx[:, 0] = ind[0]
    idx[:, 1] = ind[1]
    idx[:, 2] = ind[2]

    Precision = ["f32/f16", "f32/f32", "f64/f16", "f64/f32", "f64/f64"]
    for precision in Precision:
        if precision in ["f64/f16", "f64/f32", "f64/f64"]:
            config.update("jax_default_matmul_precision", "highest")
        else:
            config.update("jax_default_matmul_precision", "float32")
        logger.info(f"Precision: {precision}")
        kwargs = {
            "n_components": 1,
            "lattice": LatticeD3Q19(precision),
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "g_kkprime": g_kkprime,
            "omega": [1.0],
            "precision": precision,
            "body_force": [0.0, 0.0, 0.0],
            "k": [0],
            "A": np.zeros((1, 1)),
            "io_rate": 1000,
            "compute_MLUPS": True,
            "print_info_rate": 1000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }
        # os.system("rm -rf output*/ *.vtk")
        sim = PorousMedia(**kwargs)
        sim.run(20000)
