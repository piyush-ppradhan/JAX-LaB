"""
Single component 3D droplet example where liquid droplet is suspended in its vapor. The density of each region is computed using Maxwell's Construction. The density profile
is initialized with smooth profile with specified interface width. Boundary conditions are periodic everywhere. Useful for tuning the various coefficients.
"""

import os
import subprocess

import jax.numpy as jnp
import numpy as np
from jax.tree import map as tree_map

from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseMRT
from jax_lab.core.eos import VanderWaals
from jax_lab.core.utils import save_fields_vtk

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# config.update("jax_default_matmul_precision", "float32")


class Droplet3D(MultiphaseMRT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def relaxation_matrix(mass, momentum, shear, plus, minus, third_order, fourth_order):
            relaxation = np.diag([
                mass,
                momentum,
                momentum,
                momentum,
                shear,
                shear,
                shear,
                plus,
                plus,
                plus,
                third_order,
                third_order,
                third_order,
                third_order,
                third_order,
                third_order,
                fourth_order,
                fourth_order,
                fourth_order,
            ])
            relaxation[7, 8] = minus
            relaxation[7, 9] = minus
            relaxation[8, 7] = minus
            relaxation[8, 9] = minus
            relaxation[9, 7] = minus
            relaxation[9, 8] = minus
            return jnp.array(relaxation, dtype=self.precision_policy.compute_dtype)

        self.S = [relaxation_matrix(s_0, s_1, s_2, s_plus, s_minus, s_3, s_4)]
        self.collision_matrix = tree_map(lambda M, S, M_inv: jnp.dot(jnp.dot(M, S), M_inv), self.M, self.S, self.M_inv)
        self.collision_terms = []
        for collision_matrix in self.collision_matrix:
            matrix = np.asarray(collision_matrix)
            columns = []
            for output_direction in range(self.lattice.q):
                columns.append(
                    tuple(
                        (input_direction, np.float32(matrix[input_direction, output_direction]))
                        for input_direction in range(self.lattice.q)
                        if not np.isclose(matrix[input_direction, output_direction], 0.0, atol=1e-7)
                    )
                )
            self.collision_terms.append(tuple(columns))

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
        logger.info(f"%Error Min: {(rho_g_pred - rho_g) * 100 / rho_g} Max: {(rho_l_pred - rho_l) * 100 / rho_l}")
        logger.info(f"Density: Min: {rho_g_pred} Max: {rho_l_pred}")
        logger.info(f"Maxwell construction: Min: {rho_g} Max: {rho_l}")
        logger.info(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        p_south = p[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        p_west = p[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        p_east = p[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        p_front = p[self.nx // 2 - offset, self.ny // 2, self.nz // 2 + offset, 0]
        p_back = p[self.nx // 2 + offset, self.ny // 2, self.nz // 2 - offset, 0]
        pressure_difference = p[self.nx // 2, self.ny // 2, self.nz // 2, 0] - (p_north + p_south + p_west + p_east + p_front + p_back) / 6
        logger.info(f"Pressure difference: {pressure_difference}")
        # HDF5/XDMF output option:
        # from jax_lab.core.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, "output", "data")
        save_fields_vtk(timestep, fields, "output", "data")


if __name__ == "__main__":
    s_0 = 0.0  # Conserved density mode
    s_1 = 0.0  # Conserved momentum modes
    s_2 = 0.8  # Shear modes; nu = (1 / s_2 - 0.5) / 3
    s_b = 0.8  # Bulk/diagonal stress modes
    s_3 = (16 - 8 * s_2) / (8 - s_2)  # Third-order modes
    s_4 = 1.0  # Fourth-order modes
    s_plus = (s_b + 2 * s_2) / 3
    s_minus = (s_b - s_2) / 3

    e = LatticeD3Q19().c.T
    ex = e[:, 0]
    ey = e[:, 1]
    ez = e[:, 2]

    # Raw-moment basis used by porous_media_evaporation_3D.py.
    M = np.zeros((19, 19))
    M[0, :] = ex**0
    M[1, :] = ex
    M[2, :] = ey
    M[3, :] = ez
    M[4, :] = ex * ey
    M[5, :] = ex * ez
    M[6, :] = ey * ez
    M[7, :] = ex**2
    M[8, :] = ey**2
    M[9, :] = ez**2
    M[10, :] = ex * ey**2
    M[11, :] = ex * ez**2
    M[12, :] = ey * ex**2
    M[13, :] = ez * ex**2
    M[14, :] = ey * ez**2
    M[15, :] = ez * ey**2
    M[16, :] = ex**2 * ey**2
    M[17, :] = ex**2 * ez**2
    M[18, :] = ey**2 * ez**2

    r = 40
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

    kwargs = {"a": a, "b": b, "R": R, "T": T}
    eos = VanderWaals(**kwargs)

    precision = "f32/f16"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": eos,
        "k": [0.25],
        "A": np.zeros((1, 1)),
        "s_rho": [s_0],
        "s_e": [s_b],
        "s_eta": [s_b],
        "s_j": [s_1],
        "s_q": [s_3],
        "s_pi": [s_3],
        "s_m": [s_4],
        "s_v": [s_2],
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

    subprocess.run("rm -rf output*/", shell=True, check=True)
    sim = Droplet3D(**kwargs)
    sim.run(30000)
