"""
Multi-component droplet example, can be used for computing surface tension coefficient, tuning various parameters.

The collision matrix is based on:
1. McCracken, M. E. & Abraham, J. Multiple-relaxation-time lattice-Boltzmann model for multiphase flow. Phys. Rev. E 71, 036701 (2005).
"""

import os
import operator
import numpy as np

from jax_lab.lattice import LatticeD2Q9
from jax_lab.utils import save_fields_vtk
from jax_lab.multiphase import MultiphaseMRT

from functools import partial
from jax import jit, vmap, config
from jax.tree import reduce
from jax.tree import map as tree_map
import jax.numpy as jnp

# config.update("jax_default_matmul_precision", "float32")


class Droplet2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []
        dist = (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 - r**2

        # Injected fluid
        rho_inside = (1 - fraction) * rho_t
        rho_outside = fraction * rho_t
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((nx, ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        # Displaced fluid
        rho_inside = fraction * rho_t
        rho_outside = (1 - fraction) * rho_t
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((nx, ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

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
        rho = np.array(kwargs.get("rho_total")[0, 1:-1, 1:-1, :])
        rho_1 = np.array(kwargs["rho_tree"][0][0, 1:-1, 1:-1, :])
        rho_2 = np.array(kwargs["rho_tree"][1][0, 1:-1, 1:-1, :])
        p_1 = np.array(kwargs["p_tree"][0][1:-1, 1:-1])
        p_2 = np.array(kwargs["p_tree"][1][1:-1, 1:-1])
        p = np.array(kwargs["p"][0, 1:-1, 1:-1])
        u_1 = np.array(kwargs["u_tree"][0][0, 1:-1, 1:-1, :])
        u_2 = np.array(kwargs["u_tree"][1][0, 1:-1, 1:-1, :])
        u = np.array(kwargs["u_total"][0, 1:-1, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "p_invading": p_1[..., 0],
            "p_displaced": p_2[..., 0],
            "rho": rho[..., 0],
            "rho_invading": rho_1[..., 0],
            "rho_displaced": rho_2[..., 0],
            "ux_displaced": u_2[..., 0],
            "uy_displaced": u_2[..., 1],
            "ux_invading": u_1[..., 0],
            "uy_invading": u_1[..., 1],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        offset = 90
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p_1[self.nx // 2, self.ny // 2 - offset, 0]
        p_south = p_1[self.nx // 2, self.ny // 2 + offset, 0]
        p_west = p_1[self.nx // 2 - offset, self.ny // 2, 0]
        p_east = p_1[self.nx // 2 + offset, self.ny // 2, 0]
        pressure_difference = p_2[self.nx // 2, self.ny // 2, 0] - 0.25 * (p_north + p_south + p_west + p_east)
        print(f"Pressure difference for radius = {r}: {pressure_difference}")
        # HDF5/XDMF output option:
        # from jax_lab.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, f"output_{r}", "data")
        save_fields_vtk(timestep, fields, f"output_{r}", "data")


if __name__ == "__main__":
    nx = 200
    ny = 200

    tau_2 = 1.9
    v_2 = (tau_2 - 0.5) / 3

    M = 10.0  # set
    v_1 = v_2 / M
    tau_1 = 3 * v_1 + 0.5

    rho_2 = 1.0
    rho_1 = 1.0
    width = 5

    rho_t = rho_1 + rho_2

    # Fraction of total density in the bulk of one component
    # It needs to be non-zero value to prevent NaN values.
    fraction = 0.97

    precision = "f32/f32"
    g_kkprime = -0.027 * np.ones((2, 2))
    # for g in np.linspace(0, 1.0, 100):
    g = 0.57
    g_kkprime[0, 1] = g
    g_kkprime[1, 0] = g

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

    s_rho = [0.0, 0.0]  # Mass
    s_e = [0.4, 0.4]
    s_eta = [1.0, 1.0]
    s_j = [0.0, 0.0]  # Momentum
    s_q = [1.0, 1.0]
    s_v = [1 / tau_1, 1 / tau_2]

    for r in [25, 30, 35, 40, 45, 50]:
        kwargs = {
            "n_components": 2,
            "lattice": LatticeD2Q9(precision),
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "body_force": [0.0, 0.0],
            "g_kkprime": g_kkprime,
            "precision": precision,
            "M": [M, M],
            "s_rho": s_rho,
            "s_e": s_e,
            "s_eta": s_eta,
            "s_j": s_j,
            "s_q": s_q,
            "s_v": s_v,
            "kappa": [0, 0],
            "k": [0, 0],
            "A": np.zeros((2, 2)),
            "io_rate": 30000,
            "compute_MLUPS": False,
            "print_info_rate": 10000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }

        # os.system("rm -rf output*/ *.vtk")
        sim = Droplet2D(**kwargs)
        sim.run(30000)
