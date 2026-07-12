"""
Single component 2D Taylor Green vortex using Cascaded LBM.

The collision matrix is based on:
1. Fei, L. & Luo, K. H. Consistent forcing scheme in the cascaded lattice Boltzmann method. Phys. Rev. E 96, 053307 (2017).
"""

import os
import subprocess

import numpy as np

from jax_lab.lattice import LatticeD2Q9
from jax_lab.utils import save_fields_vtk
from jax_lab.multiphase import MultiphaseCascade
from jax_lab.boundary_conditions import BounceBack

import operator
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit, config
from jax.tree import reduce
from jax.tree import map as tree_map


# config.update("jax_default_matmul_precision", "float32")


class TaylorGreen2D(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        rho_tree = []

        rho = np.zeros((self.nx, self.ny, 1))
        rho[..., 0] = 3.0 * (1.0 - u_0**2 / 4.0 * (np.cos(2.0 * phi * x) + np.cos(2.0 * phi * y)))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u[..., 0] = u_0 * np.sin(phi * x) * np.cos(phi * y)
        u[..., 1] = -u_0 * np.cos(phi * x) * np.sin(phi * y)
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        psi_tree = tree_map(lambda rho: jnp.exp(-1 / rho), rho_tree)
        U_tree = tree_map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return psi_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree):
        def f(g_kk):
            return reduce(operator.add, tree_map(lambda _gkk, psi: _gkk * psi, list(g_kk), psi_tree))

        return tree_map(
            lambda rho, psi, nt: rho / 3 + 1.5 * psi * nt,
            rho_tree,
            psi_tree,
            list(vmap(f, in_axes=(0,))(self.g_kkprime)),
        )

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "u_analytical_y": u_analytical[..., 0],
            "u_analytical_x": u_analytical[..., 1],
        }
        save_fields_vtk(
            timestep,
            fields,
            "output",
            "data",
        )


if __name__ == "__main__":
    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)

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

    N = 80
    nx = N
    ny = N

    u_0 = 0.05
    phi = 2 * np.pi / N

    x = np.linspace(0, N - 1, N)
    y = np.linspace(0, N - 1, N)
    x, y = np.meshgrid(x, y)

    Re = 150
    visc = u_0 * np.pi / Re

    F = np.zeros((nx, ny, 2))
    F[..., 0] = 2 * visc * u_0 * phi**2 * np.cos(phi * x) * np.cos(phi * y)
    F[..., 1] = 2 * visc * u_0 * phi**2 * np.sin(phi * x) * np.sin(phi * y)

    u_analytical = np.zeros_like(F)
    u_analytical[..., 0] = u_0 * np.sin(phi * x) * np.sin(phi * y)
    u_analytical[..., 1] = u_0 * np.cos(phi * x) * np.cos(phi * y)

    s2 = 1 / (3 * visc + 0.5)
    s_0 = [1.0]  # Mass conservation
    s_1 = [1.0]  # Fixed
    s_2 = [s2]
    s_b = [s2]
    s_3 = [1.0]  # [(16 - 8 * s2) / (8 - s2)]  # No slip
    s_4 = [1.0]  # No slip

    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "g_kkprime": np.zeros((1, 1)),
        "body_force": F,
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
        "io_rate": 40000,
        "compute_MLUPS": False,
        "print_info_rate": 40000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    subprocess.run("rm -rf output*/ *.vtk", shell=True, check=True)
    sim = TaylorGreen2D(**kwargs)
    sim.run(400000)
