"""
Single component 3D droplet example where liquid droplet is suspended in its vapor. The density of each region is computed using Maxwell's Construction. The density profile
is initialized with smooth profile with specified interface width. Boundary conditions are periodic everywhere. Useful for tuning the various coefficients.

The collision matrix is based on:
1. Coveney, P. V. et al. Multiple–relaxation–time lattice Boltzmann models in three dimensions. Philosophical Transactions of the Royal Society of London.
Series A: Mathematical, Physical and Engineering Sciences 360, 437–451 (2002).
"""

import os
import numpy as np
from jax import config

from src.lattice import LatticeD3Q19
from src.multiphase import MultiphaseMRT
from src.eos import VanderWaal
from src.utils import save_fields_vtk

# config.update("jax_default_matmul_precision", "float32")


class Droplet3D(MultiphaseMRT):
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

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho_tree"][0][0, 1:-1, 1:-1, 1:-1, :])
        p = np.array(kwargs["p_tree"][0][1:-1, 1:-1, 1:-1, :])
        u = np.array(kwargs["u_tree"][0][0, 1:-1, 1:-1, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1], "uz": u[..., 2]}
        offset = 95
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
        # HDF5/XDMF output option:
        # from src.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, "output", "data")
        save_fields_vtk(timestep, fields, "output", "data")


if __name__ == "__main__":
    s_rho = [0.0]
    s_e = [1.0]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_m = [1.0]
    s_pi = [1.0]
    s_v = [1.0]

    e = LatticeD3Q19().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((19, 19))
    M[0, :] = en**0
    M[1, :] = 19 * en**2 - 30
    M[2, :] = (21 * en**4 - 53 * en**2 + 24) / 2
    M[3, :] = e[:, 0]
    M[4, :] = (5 * en**2 - 9) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (5 * en**2 - 9) * e[:, 1]
    M[7, :] = e[:, 2]
    M[8, :] = (5 * en**2 - 9) * e[:, 2]
    M[9, :] = 3 * e[:, 0] ** 2 - en**2
    M[10, :] = (3 * en**2 - 5) * (3 * e[:, 0] ** 2 - en**2)
    M[11, :] = e[:, 1] ** 2 - e[:, 2] ** 2
    M[12, :] = (3 * en**2 - 5) * (e[:, 1] ** 2 - e[:, 2] ** 2)
    M[13, :] = e[:, 0] * e[:, 1]
    M[14, :] = e[:, 1] * e[:, 2]
    M[15, :] = e[:, 0] * e[:, 2]
    M[16, :] = (e[:, 1] ** 2 - e[:, 2] ** 2) * e[:, 0]
    M[17, :] = (e[:, 2] ** 2 - e[:, 0] ** 2) * e[:, 1]
    M[18, :] = (e[:, 0] ** 2 - e[:, 1] ** 2) * e[:, 2]

    r = 30
    width = 4
    nx = 200
    ny = 200
    nz = 200

    a = 9 / 49
    b = 2 / 21
    R = 1.0

    rho_l = 6.764470400
    rho_g = 0.838834226
    Tc = 0.5714285714
    T = 0.8 * Tc

    theta = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    phi = np.ones((nx, ny, nz, 1))
    delta_rho = np.zeros((nx, ny, nz, 1))

    kwargs = {"a": a, "b": b, "R": R, "T": T}
    eos = VanderWaal(**kwargs)

    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": eos,
        "body_force": [0.0, 0.0, 0.0],
        "k": [0.27],
        "A": 0.01 * np.ones((1, 1)),
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_pi": s_pi,
        "s_m": s_m,
        "s_v": [1.0],
        "M": [M],
        "kappa": [0.0],
        "precision": precision,
        "io_rate": 10000,
        "compute_MLUPS": False,
        "print_info_rate": 10000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    os.system("rm -rf output*/ *.vtk")
    sim = Droplet3D(**kwargs)
    sim.run(30000)
