"""
Single component 3D droplet example where liquid droplet is suspended in its vapor. The density of each region is computed using Maxwell's Construction. The density profile
is initialized with smooth profile with specified interface width. Boundary conditions are periodic everywhere. Useful for tuning the various coefficients.

The collision matrix is based on:
1. Fei, L., Luo, K. H. & Li, Q. Three-dimensional cascaded lattice Boltzmann method: Improved implementation and consistent forcing scheme. Phys. Rev. E 97, 053309 (2018).
"""

import os
import subprocess

import numpy as np

from jax_lab.core.lattice import LatticeD3Q19, LatticeD3Q27
from jax_lab.core.utils import save_fields_vtk
from jax_lab.core.multiphase import MultiphaseCascade

import operator
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit, config
from jax.tree import reduce
from jax.tree import map as tree_map


# config.update("jax_default_matmul_precision", "float32")


class Droplet3D(MultiphaseCascade):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        z = np.linspace(0, self.nz - 1, self.nz, dtype=int)
        x, y, z = np.meshgrid(x, y, z)

        rho_tree = []

        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - self.nz / 2) ** 2)

        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)

        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
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

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1], "uz": u[..., 2]}
        offset = 45
        rho_north = rho[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        rho_south = rho[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        rho_west = rho[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        rho_east = rho[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        rho_front = rho[self.nx // 2, self.ny // 2, self.nz // 2 + offset, 0]
        rho_back = rho[self.nx // 2, self.ny // 2, self.nz // 2 - offset, 0]
        rho_g_pred = (rho_north + rho_south + rho_west + rho_east + rho_front + rho_back) / 6
        rho_l_pred = rho[self.nx // 2, self.ny // 2, self.nz // 2, 0]
        print(f"%Error Min: {(rho_g_pred - rho_g) * 100 / rho_g} Max: {(rho_l_pred - rho_l) * 100 / rho_l}")
        print(f"Density: Min: {rho_g_pred} Max: {rho_l_pred}")
        print(f"Maxwell construction: Min: {rho_g} Max: {rho_l}")
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        p_south = p[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        p_west = p[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        p_east = p[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        p_front = p[self.nx // 2 - offset, self.ny // 2, self.nz // 2 + offset, 0]
        p_back = p[self.nx // 2 + offset, self.ny // 2, self.nz // 2 - offset, 0]
        pressure_difference = p[self.nx // 2, self.ny // 2, self.nz // 2, 0] - (p_north + p_south + p_west + p_east + p_front + p_back) / 6
        print(f"Pressure difference: {pressure_difference}")
        save_fields_vtk(timestep, fields, "output", "data")


if __name__ == "__main__":
    # Cascaded LBM collision matrix
    # D3Q19
    e = LatticeD3Q19().c.T
    ex = e[:, 0]
    ey = e[:, 1]
    ez = e[:, 2]
    M = np.zeros((19, 19))
    M[0, :] = ex**0
    M[1, :] = ex
    M[2, :] = ey
    M[3, :] = ez
    M[4, :] = ex * ey
    M[5, :] = ex * ez
    M[6, :] = ey * ez
    M[7, :] = ex * ex
    M[8, :] = ey * ey
    M[9, :] = ez * ez
    M[10, :] = ex * ey * ey
    M[11, :] = ex * ez * ez
    M[12, :] = ey * ex * ex
    M[13, :] = ez * ex * ex
    M[14, :] = ey * ez * ez
    M[15, :] = ez * ey * ey
    M[16, :] = ex * ex * ey * ey
    M[17, :] = ex * ex * ez * ez
    M[18, :] = ey * ey * ez * ez

    # D3Q27
    # e = LatticeD3Q27().c.T
    # ex = e[:, 0]
    # ey = e[:, 1]
    # ez = e[:, 2]
    # M = np.zeros((27, 27))
    # M[0, :] = ex**0
    # M[1, :] = ex
    # M[2, :] = ey
    # M[3, :] = ez
    # M[4, :] = ex * ey
    # M[5, :] = ex * ez
    # M[6, :] = ey * ez
    # M[7, :] = ex * ex
    # M[8, :] = ey * ey
    # M[9, :] = ez * ez
    # M[10, :] = ex * ey * ey
    # M[11, :] = ex * ez * ez
    # M[12, :] = ey * ex * ex
    # M[13, :] = ez * ex * ex
    # M[14, :] = ey * ez * ez
    # M[15, :] = ez * ey * ey
    # M[16, :] = ex * ey * ez
    # M[17, :] = ex * ex * ey * ey
    # M[18, :] = ex * ex * ez * ez
    # M[19, :] = ey * ey * ez * ez
    # M[20, :] = ex * ex * ey * ez
    # M[21, :] = ex * ey * ey * ez
    # M[22, :] = ex * ey * ez * ez
    # M[23, :] = ex * ey * ey * ez * ez
    # M[24, :] = ex * ex * ey * ez * ez
    # M[25, :] = ex * ex * ey * ey * ez
    # M[26, :] = ex * ex * ey * ey * ez * ez

    rho_l = 2.783
    rho_g = 0.3675

    r = 30
    nx = 100
    ny = 100
    nz = 100

    width = 3

    s2 = 1.4
    s_0 = [1.0]  # Conserved
    s_1 = [1.0]  # Conserved
    s_2 = [s2]
    s_b = [s2]
    s_3 = [1.0]
    s_4 = [1.0]
    s_3b = [1.0]
    s_4b = [1.0]
    s_5 = [1.0]
    s_6 = [1.0]

    G = -10 / 3

    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(precision),
        # "lattice": LatticeD3Q27(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": G * np.ones((1, 1)),
        "body_force": [0.0, 0.0, 0.0],
        "k": [0.0],
        "A": np.zeros((1, 1)),
        "M": [M],
        "s_0": s_0,
        "s_1": s_1,
        "s_b": s_b,
        "s_2": s_2,
        "s_3": s_3,
        "s_4": s_4,
        "s_3b": s_3b,
        "s_4b": s_4b,
        "s_5": s_5,
        "s_6": s_6,
        "sigma": [0.0],
        "precision": precision,
        "io_rate": 10000,
        "compute_MLUPS": False,
        "print_info_rate": 10000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    subprocess.run("rm -rf output*/ *.vtk", shell=True, check=True)
    sim = Droplet3D(**kwargs)
    sim.run(30000)
