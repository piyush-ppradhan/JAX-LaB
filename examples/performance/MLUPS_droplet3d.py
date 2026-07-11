"""
Single component 3D droplet example where liquid droplet is suspended in its vapor. The density of each region is computed using Maxwell's Construction. The density profile
is initialized with smooth profile with specified interface width. Boundary conditions are periodic everywhere. Useful for tuning the various coefficients.

The collision matrix is based on:
1. McCracken, M. E. & Abraham, J. Multiple-relaxation-time lattice-Boltzmann model for multiphase flow. Phys. Rev. E 71, 036701 (2005).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np

from src.lattice import LatticeD3Q19
from src.multiphase import MultiphaseBGK
from src.eos import VanderWaal

from jax import config
import operator
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit
from jax.tree import reduce
from jax.tree import map as tree_map


class Droplet3D(MultiphaseBGK):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        z = np.linspace(0, self.nz - 1, self.nz, dtype=int)
        x, y, z = np.meshgrid(x, y, z)

        rho_tree = []

        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - self.nz / 2) ** 2)

        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)

        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init(
            (self.nx, self.ny, self.nz, 1),
            self.precisionPolicy.compute_dtype,
            init_val=rho,
        )
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init(
            (self.nx, self.ny, self.nz, 3),
            self.precisionPolicy.compute_dtype,
            init_val=u,
        )
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

        return tree_map(lambda rho, psi, nt: rho / 3 + 1.5 * psi * nt, rho_tree, psi_tree, list(vmap(f, in_axes=(0,))(self.g_kkprime)))


if __name__ == "__main__":
    r = 30
    width = 4

    rho_l = 2.783
    rho_g = 0.3675

    N = [480, 320, 300, 280, 270]
    Precision = ["f32/f16", "f32/f32", "f64/f16", "f64/f32", "f64/f64"]
    for i in range(5):
        precision = Precision[i]
        n = N[i]
        if precision in ["f64/f16", "f64/f32", "f64/f64"]:
            config.update("jax_default_matmul_precision", "highest")
        else:
            config.update("jax_default_matmul_precision", "float32")
        print(f"Precision: {precision}")
        kwargs = {
            "n_components": 1,
            "lattice": LatticeD3Q19(precision),
            "omega": [1.0],
            "nx": n,
            "ny": n,
            "nz": n,
            "g_kkprime": -1.0 * np.ones((1, 1)),
            # "EOS": eos,
            "body_force": [0.0, 0.0, 0.0],
            "k": [0.27],
            "A": 0.0 * np.ones((1, 1)),
            # "s_rho": s_rho,
            # "s_e": s_e,
            # "s_eta": s_eta,
            # "s_j": s_j,
            # "s_q": s_q,
            # "s_pi": s_pi,
            # "s_m": s_m,
            # "s_v": [1.0],
            # "M": [M],
            "kappa": [0.0],
            "precision": precision,
            "io_rate": 20000,
            "compute_MLUPS": True,
            "print_info_rate": 20000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }

        # os.system("rm -rf output*/ *.vtk")
        sim = Droplet3D(**kwargs)
        sim.run(20000)
