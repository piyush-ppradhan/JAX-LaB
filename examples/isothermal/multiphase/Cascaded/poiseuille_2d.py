"""
Single component 2D Poiseuille flow driven by body force.

The collision matrix is based on:
1. Fei, L. & Luo, K. H. Consistent forcing scheme in the cascaded lattice Boltzmann method. Phys. Rev. E 96, 053307 (2017).
"""

import os
import subprocess

import numpy as np

from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.utils import save_fields_vtk
from jax_lab.core.multiphase import MultiphaseCascade
from jax_lab.core.boundary_conditions import BounceBack

import operator
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit, config
from jax.tree import reduce
from jax.tree import map as tree_map

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# config.update("jax_default_matmul_precision", "float32")


class Poiseuille2D(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        rho_tree = []

        rho = np.ones((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        walls = np.concatenate((self.bounding_box_indices["top"], self.bounding_box_indices["bottom"]))
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBack(walls, self.grid_info, self.precision_policy, theta=theta[walls], phi=phi[walls], delta_rho=delta_rho[walls]))

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        psi_tree = tree_map(lambda rho: jnp.exp(-1 / rho), rho_tree)
        U_tree = tree_map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return psi_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree):
        def f(g_kk):
            return reduce(operator.add, tree_map(lambda _gkk, psi: _gkk * psi, list(g_kk), psi_tree))

        return tree_map(lambda rho, psi, nt: rho / 3 + 1.5 * psi * nt, rho_tree, psi_tree, list(vmap(f, in_axes=(0,))(self.g_kkprime)))

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        save_fields_vtk(timestep, fields, "output", "data")


if __name__ == "__main__":
    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)

    # ex = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1])
    # ey = np.array([0, 0, 0, 1, -1, 1, -1, -1, 1])
    # _e = np.zeros((9, 2))
    # _e[:, 0] = ex
    # _e[:, 1] = ey
    # order = np.array([e.tolist().index((_e[i]).tolist()) for i in range(9)])
    #
    # lattice = LatticeD2Q9()
    # lattice.e = _e.T

    # Cascaded LBM collision matrix
    M = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 1, 1, 1, 1, 2, 2, 2, 2],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 1, 1, -1, -1],
        [0, 0, 0, 0, 0, 1, -1, -1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
    ])

    nx = 100
    ny = 50

    visc = 0.5
    s2 = 1 / (3 * visc + 0.5)
    s2 = 1.754

    s_0 = [s2]  # Mass conservation
    s_1 = [s2]  # Fixed
    s_2 = [s2]
    s_b = [s2]
    s_3 = [(16 - 8 * s2) / (8 - s2)]  # No slip
    s_4 = [s2]  # No slip

    theta = (np.pi / 2) * np.ones((nx, ny, 1))
    phi = np.ones((nx, ny, 1))
    delta_rho = np.zeros((nx, ny, 1))

    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "g_kkprime": np.zeros((1, 1)),
        "body_force": [1e-6, 0.0],
        "k": [0.0],
        "A": np.zeros((1, 1)),
        "M": [M],
        "s_0": s_0,
        "s_1": s_1,
        "s_b": s_b,
        "s_2": s_2,
        "s_3": s_3,
        "s_4": s_4,
        "sigma": [0.0],
        "precision": precision,
        "io_rate": 1000,
        "compute_MLUPS": False,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    subprocess.run("rm -rf output*/ *.vtk", shell=True, check=True)
    sim = Poiseuille2D(**kwargs)
    sim.run(10000)
