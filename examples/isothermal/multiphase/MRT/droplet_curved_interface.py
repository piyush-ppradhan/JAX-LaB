"""
Single component 2D droplet between curved surface. where liquid droplet is suspended in its vapor. The density of each region is computed using Maxwell's Construction. The density profile
is initialized with smooth profile with specified interface width. Droplet is initialized away from curved surface and it drops on the
curved surface because of a small weak body force in the vertical direction. Boundary conditions are periodic everywhere.

The collision matrix is based on:
1. McCracken, M. E. & Abraham, J. Multiple-relaxation-time lattice-Boltzmann model for multiphase flow. Phys. Rev. E 71, 036701 (2005).
"""

import os
import subprocess
from jax import config
import numpy as np

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.eos import VanderWaals
from jax_lab.core.utils import save_fields_vtk
from jax_lab.core.multiphase import MultiphaseMRT

# config.update("jax_default_matmul_precision", "float32")


class DropletOnCurvedSurface2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        rho_tree = []

        dist = np.sqrt((x - self.nx / 2 - 50) ** 2 + (y - self.ny / 2) ** 2)

        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)

        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        sphere_1 = (x - self.nx // 2) ** 2 + (y - self.ny // 2 + spacing // 2) ** 2 - R_grain**2
        sphere_2 = (x - self.nx // 2) ** 2 + (y - self.ny // 2 - spacing // 2) ** 2 - R_grain**2
        ind_1 = np.array(np.where(sphere_1 <= 0)).T
        ind_2 = np.array(np.where(sphere_2 <= 0)).T
        ind = np.concatenate((ind_1, ind_2))
        ind = tuple(ind.T)
        self.BCs[0].append(BounceBack(ind, self.grid_info, self.precision_policy, theta[ind], phi[ind], delta_rho[ind]))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        offset = 90
        rho_north = rho[self.nx // 2, self.ny // 2 - offset, 0]
        rho_south = rho[self.nx // 2, self.ny // 2 + offset, 0]
        rho_west = rho[self.nx // 2 - offset, self.ny // 2, 0]
        rho_east = rho[self.nx // 2 + offset, self.ny // 2, 0]
        rho_g_pred = 0.25 * (rho_north + rho_south + rho_west + rho_east)
        rho_l_pred = rho[self.nx // 2, self.ny // 2, 0]
        print(f"%Error Min: {(rho_g_pred - rho_g) * 100 / rho_g} Max: {(rho_l_pred - rho_l) * 100 / rho_l}")
        print(f"Density: Min: {rho_g_pred} Max: {rho_l_pred}")
        print(f"Maxwell construction: Min: {rho_g} Max: {rho_l}")
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p[self.nx // 2, self.ny // 2 - offset, 0]
        p_south = p[self.nx // 2, self.ny // 2 + offset, 0]
        p_west = p[self.nx // 2 - offset, self.ny // 2, 0]
        p_east = p[self.nx // 2 + offset, self.ny // 2, 0]
        pressure_difference = p[self.nx // 2, self.ny // 2, 0] - 0.25 * (p_north + p_south + p_west + p_east)
        print(f"Pressure difference: {pressure_difference}")
        # HDF5/XDMF output option:
        from jax_lab.core.utils import save_fields_hdf5_xdmf

        save_fields_hdf5_xdmf(timestep, fields, "output", "data")
        # save_fields_vtk(timestep, fields, "output", "data")


class DropletOnCurvedSurface2DGeometric(DropletOnCurvedSurface2D):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        sphere_1 = (x - self.nx // 2) ** 2 + (y - self.ny // 2 + spacing // 2) ** 2 - R_grain**2
        sphere_2 = (x - self.nx // 2) ** 2 + (y - self.ny // 2 - spacing // 2) ** 2 - R_grain**2
        ind_1 = np.array(np.where(sphere_1 <= 0)).T
        ind_2 = np.array(np.where(sphere_2 <= 0)).T
        ind = np.concatenate((ind_1, ind_2))
        ind = tuple(ind.T)
        self.BCs[0].append(BounceBack(ind, self.grid_info, self.precision_policy, theta[ind]))


if __name__ == "__main__":
    r = 18
    R_grain = 30
    spacing = 80
    nx = 200
    ny = 200

    width = 2

    a = 9 / 49
    b = 2 / 21
    R = 1.0

    rho_l = 6.764470400
    rho_g = 0.838834226
    Tc = 0.5714285714
    T = 0.8 * Tc

    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((9, 9))
    M[0, :] = en**0
    M[1, :] = -4 * en**0 + 3 * en**2
    M[2, :] = 4 * en**0 - (21 / 2) * en**2 + (9 / 2) * en**4
    M[3, :] = e[:, 0]
    M[4, :] = (-5 * en**0 + 3 * en**2) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (-5 * en**0 + 3 * en**2) * e[:, 1]
    M[7, :] = e[:, 0] ** 2 - e[:, 1] ** 2
    M[8, :] = e[:, 0] * e[:, 1]

    s_rho = [0.0]
    s_e = [1.0]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1.0]

    kwargs = {"a": a, "b": b, "R": R, "T": T}
    eos = VanderWaals(**kwargs)

    x = np.linspace(0, nx - 1, nx, dtype=int)
    y = np.linspace(0, ny - 1, ny, dtype=int)
    x, y = np.meshgrid(x, y)

    theta = (np.pi / 6) * np.ones((nx, ny, 1))
    phi = 1.2 * np.ones((nx, ny, 1))
    delta_rho = np.zeros((nx, ny, 1))

    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": eos,
        "body_force": [0.0, -3.02e-6],
        "k": [0.16],
        "A": -0.032 * np.ones((1, 1)),
        "M": [M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "kappa": [1.0],
        "precision": precision,
        "io_rate": 10000,
        "compute_MLUPS": False,
        "print_info_rate": 200000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    subprocess.run("rm -rf output*/ *.vtk", shell=True, check=True)
    sim = DropletOnCurvedSurface2D(**kwargs)
    sim.run(200000)
