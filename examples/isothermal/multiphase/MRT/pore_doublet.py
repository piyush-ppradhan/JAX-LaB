"""
Pore doublet example. The obstacle and branching is elliptical shaped. The simulations are performed using MRT collision model for the case of
imbibition (wetting fluid injected). This is an example of multicomponent simulation using JAX-LaB

The collision matrix is based on:
1. McCracken, M. E. & Abraham, J. Multiple-relaxation-time lattice-Boltzmann model for multiphase flow. Phys. Rev. E 71, 036701 (2005).
"""

import os
import operator
import numpy as np

from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.utils import save_fields_vtk
from jax_lab.core.multiphase import MultiphaseMRT
from jax_lab.core.boundary_conditions import BounceBack, EquilibriumBC

from functools import partial
from jax import jit, vmap, config
from jax.tree import reduce
from jax.tree import map as tree_map
import jax.numpy as jnp

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# config.update("jax_default_matmul_precision", "float32")


class PoreDoublet(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        rho_tree = []

        # Invading fluid
        rho = (1 - fraction) * rho_t * np.ones((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        # Displaced fluid
        rho = fraction * rho_t * np.ones((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        _, yy = coord[:, 0], coord[:, 1]
        poiseuille_profile = lambda x, x0, d, umax: np.maximum(0.0, 4.0 * umax / (d**2) * ((x - x0) * d - (x - x0) ** 2))

        # apply bounce back boundary condition to the walls
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(BounceBack(walls, self.grid_info, self.precision_policy, theta_i[walls], phi_i[walls], delta_rho_i[walls]))
        self.BCs[1].append(BounceBack(walls, self.grid_info, self.precision_policy, theta_d[walls], phi_d[walls], delta_rho_d[walls]))

        # apply inlet equilibrium boundary condition at the left
        yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        rho_inlet = fraction * rho_t * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precision_policy.compute_dtype)
        vel_inlet[:, 0] = poiseuille_profile(
            yy_inlet,
            yy_inlet.min(),
            yy_inlet.max() - yy_inlet.min(),
            3.0 / 2.0 * prescribed_vel,
        )
        self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel_inlet))
        rho_inlet = (1 - fraction) * rho_t * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precision_policy.compute_dtype)
        vel_inlet[:, 0] = poiseuille_profile(yy_inlet, yy_inlet.min(), yy_inlet.max() - yy_inlet.min(), 3.0 / 2.0 * prescribed_vel)
        self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel_inlet))

        # Same at the outlet
        yy_outlet = yy.reshape(self.nx, self.ny)[tuple(outlet.T)]
        rho_outlet = fraction * rho_t * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_outlet = np.zeros(outlet.shape, dtype=self.precision_policy.compute_dtype)
        vel_outlet[:, 0] = poiseuille_profile(yy_outlet, yy_outlet.min(), yy_outlet.max() - yy_outlet.min(), 3.0 / 2.0 * prescribed_vel)
        self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel_outlet))
        rho_outlet = (1 - fraction) * rho_t * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_outlet = np.zeros(outlet.shape, dtype=self.precision_policy.compute_dtype)
        vel_outlet[:, 0] = poiseuille_profile(yy_outlet, yy_outlet.min(), yy_outlet.max() - yy_outlet.min(), 3.0 / 2.0 * prescribed_vel)
        self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel_outlet))

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        U_tree = tree_map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

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
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs.get("rho_total")[0, 1:-1, 1:-1, :])
        rho_d = np.array(kwargs["rho_tree"][0][0, 1:-1, 1:-1, :])
        rho_i = np.array(kwargs["rho_tree"][1][0, 1:-1, 1:-1, :])
        p_d = np.array(kwargs["p_tree"][0][1:-1, 1:-1])
        p_i = np.array(kwargs["p_tree"][1][1:-1, 1:-1])
        p = np.array(kwargs["p"][0, 1:-1, 1:-1])
        u_d = np.array(kwargs["u_tree"][0][0, 1:-1, 1:-1, :])
        u_i = np.array(kwargs["u_tree"][1][0, 1:-1, 1:-1, :])
        u = np.array(kwargs["u_total"][0, 1:-1, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "p_d": p_d[..., 0],
            "p_i": p_i[..., 0],
            "rho": rho[..., 0],
            "rho_displaced": rho_d[..., 0],
            "rho_invading": rho_i[..., 0],
            "ux_displaced": u_d[..., 0],
            "uy_displaced": u_d[..., 1],
            "ux_invading": u_i[..., 0],
            "uy_invading": u_i[..., 1],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        # HDF5/XDMF output option:
        # from jax_lab.core.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, "output_pore_doublet", "data")
        save_fields_vtk(timestep, fields, "output_pore_doublet", "data")


class PoreDoubletGeometric(PoreDoublet):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        _, yy = coord[:, 0], coord[:, 1]
        poiseuille_profile = lambda x, x0, d, umax: np.maximum(0.0, 4.0 * umax / (d**2) * ((x - x0) * d - (x - x0) ** 2))

        # apply bounce back boundary condition to the walls
        self.BCs[0].append(BounceBack(walls, self.grid_info, self.precision_policy, theta_i[walls]))
        self.BCs[1].append(BounceBack(walls, self.grid_info, self.precision_policy, theta_d[walls]))

        # apply inlet equilibrium boundary condition at the left
        yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        rho_inlet = fraction * rho_t * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precision_policy.compute_dtype)
        vel_inlet[:, 0] = poiseuille_profile(
            yy_inlet,
            yy_inlet.min(),
            yy_inlet.max() - yy_inlet.min(),
            3.0 / 2.0 * prescribed_vel,
        )
        self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel_inlet))
        rho_inlet = (1 - fraction) * rho_t * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precision_policy.compute_dtype)
        vel_inlet[:, 0] = poiseuille_profile(yy_inlet, yy_inlet.min(), yy_inlet.max() - yy_inlet.min(), 3.0 / 2.0 * prescribed_vel)
        self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel_inlet))

        # Same at the outlet
        yy_outlet = yy.reshape(self.nx, self.ny)[tuple(outlet.T)]
        rho_outlet = fraction * rho_t * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_outlet = np.zeros(outlet.shape, dtype=self.precision_policy.compute_dtype)
        vel_outlet[:, 0] = poiseuille_profile(yy_outlet, yy_outlet.min(), yy_outlet.max() - yy_outlet.min(), 3.0 / 2.0 * prescribed_vel)
        self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel_outlet))
        rho_outlet = (1 - fraction) * rho_t * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_outlet = np.zeros(outlet.shape, dtype=self.precision_policy.compute_dtype)
        vel_outlet[:, 0] = poiseuille_profile(yy_outlet, yy_outlet.min(), yy_outlet.max() - yy_outlet.min(), 3.0 / 2.0 * prescribed_vel)
        self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel_outlet))


if __name__ == "__main__":
    precision = "f32/f32"
    nx = 200
    ny = 90

    tau_2 = 1.9
    v_2 = (tau_2 - 0.5) / 3

    visc_ratio = 10.0  # viscosity ratio
    v_1 = v_2 / visc_ratio
    tau_1 = 3 * v_1 + 0.5

    rho_2 = 1.0  # Displaced fluid
    rho_1 = 1.0  # Invading fluid

    rho_t = rho_1 + rho_2

    fraction = 0.97

    # Channel dimensions
    width = 12  # Width of the inlet and outlet channel

    inlet = np.array(
        [[0, y] for y in np.arange(ny // 2 - width / 2, ny // 2 + width // 2)],
        dtype=int,
    )

    outlet = np.array(
        [[nx - 1, y] for y in np.arange(ny // 2 - width / 2, ny // 2 + width // 2)],
        dtype=int,
    )

    # Elliptical dimensions
    a = 45  # Major axis length for the obstacle
    b = 25  # Minor axis length for the obstacle
    offset = 4  # Offset for obstacle to define different channels with different widths

    x = np.linspace(0, nx - 1, nx, dtype=int)
    y = np.linspace(0, ny - 1, ny, dtype=int)
    x, y = np.meshgrid(x, y)
    x = x.T
    y = y.T
    obstacle = ((x - nx // 2) / a) ** 2 + (((y - ny // 2) - offset) / b) ** 2 - 1
    elliptical_channel = ((x - nx // 2) / (a + width)) ** 2 + ((y - ny // 2) / (b + width)) ** 2 - 1

    mask = np.ones((nx, ny), dtype=int)
    mask[:, ny // 2 - width : ny // 2 + width // 2 + 1] = 0
    mask[elliptical_channel <= 0] = 0
    mask[obstacle <= 0] = 1

    walls = np.where(mask == 1)
    prescribed_vel = 0.05
    g_kkprime = -0.027 * np.ones((2, 2))
    g = 0.57
    g_kkprime[0, 1] = g
    g_kkprime[1, 0] = g
    Lx = nx // 2
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
    s_e = [1.0, 1.0]
    s_eta = [1.0, 1.0]
    s_j = [0.0, 0.0]  # Momentum
    s_q = [1.0, 1.0]
    s_v = [1 / tau_1, 1 / tau_2]

    # Store contact angle, only the values at solid nodes are important and are stored during boundary condition definition
    theta_d = 0.5 * np.pi * np.ones((nx, ny, 1))
    theta_i = (np.pi / 6) * np.ones((nx, ny, 1))
    phi_d = 1.4 * np.ones((nx, ny, 1))
    phi_i = 1.0 * np.ones((nx, ny, 1))
    delta_rho_d = np.zeros((nx, ny, 1))
    delta_rho_i = np.zeros((nx, ny, 1))

    kwargs = {
        "n_components": 2,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "g_kkprime": g_kkprime,
        "body_force": [0.0, 0.0],
        "precision": precision,
        "M": [M, M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "kappa": [0.0, 0.0],
        "k": [0, 0],
        "A": np.zeros((2, 2)),
        "io_rate": 100,
        "compute_MLUPS": False,
        "print_info_rate": 30000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    # os.system("rm -rf output*/ *.vtk")
    sim = PoreDoublet(**kwargs)
    sim.run(30000)
