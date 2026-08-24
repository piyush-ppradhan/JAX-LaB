"""
Single component 3D evaporation from droplet placed on a surface due to the prescribed vapor concentration gradient.
The boundary conditions are: open (top, left, right, front and back), no slip-wall (bottom).

The collision matrix for is based on:
1. Fei, L., Derome, D. & Carmeliet, J. Pore-scale study on the effect of heterogeneity on evaporation in porous media.
Journal of Fluid Mechanics 983, A6 (2024).
"""

import os
import subprocess
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import config, jit
from jax.tree import map as tree_map

from jax_lab.core.boundary_conditions import BounceBack, ExtrapolationOutflowMultiphase
from jax_lab.core.eos import PengRobinson
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseCascade
from jax_lab.core.utils import save_fields_hdf5_xdmf

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

config.update("jax_default_matmul_precision", "float32")


class DropletOnWall(MultiphaseCascade):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value=None):
        self._k = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value=None):
        self._A = value

    def initialize_macroscopic_fields(self):
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z) ** 2)
        rho = np.ones((self.nx, self.ny, self.nz, 1))
        rho[..., 0] = 0.5 * (rho_w_l + rho_w_g) - 0.5 * (rho_w_l - rho_w_g) * np.tanh(2 * (dist - r) / width)
        rho[..., 0, 0] = 1.0
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree = [rho]

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # open_idx = np.concatenate((
        #     self.bounding_box_indices["top"],
        #     self.bounding_box_indices["left"],
        #     self.bounding_box_indices["right"],
        #     self.bounding_box_indices["front"],
        #     self.bounding_box_indices["back"],
        # ))

        # top = self.bounding_box_indices["top"]
        # left = np.array([[0, y, z] for y in range(self.ny) for z in range(1, self.nz)], dtype=int)
        # right = np.array([[self.nx - 1, y, z] for y in range(self.ny) for z in range(1, self.nz)], dtype=int)
        # front = np.array([[x, 0, z] for x in range(self.ny) for z in range(1, self.nz)], dtype=int)
        # back = np.array([[x, self.ny - 1, z] for x in range(self.nx) for z in range(1, self.nz)], dtype=int)
        # open_idx = np.concatenate((top, left, right, front, back))

        open_idx = self.bounding_box_indices["top"]
        self.BCs[0].append(ExtrapolationOutflowMultiphase(tuple(open_idx.T), self.grid_info, self.precision_policy))

        wall = self.bounding_box_indices["bottom"]
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        # wall = np.array([[x, y, buffer] for x in range(self.nx) for y in range(self.ny)], dtype=int)
        self.BCs[0].append(
            BounceBack(tuple(wall.T), self.grid_info, self.precision_policy, theta_w[tuple(wall.T)], phi_w[tuple(wall.T)], delta_rho_w[tuple(wall.T)])
        )

    @partial(jit, static_argnums=(0,))
    def compute_fluid_fluid_force(self, psi_tree, U_tree):
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype).T
        psi_s_tree = tree_map(lambda psi: self.streaming(jnp.repeat(psi, axis=-1, repeats=self.q)), psi_tree)
        # U_s_tree = tree_map(lambda U: self.streaming(jnp.repeat(U, axis=-1, repeats=self.q)), U_tree)

        def ffk_1():
            return tree_map(lambda G, psi_s: jnp.dot(G * self.G_ff * psi_s, c), self.g_kkprime.diagonal().tolist(), psi_s_tree)

        return tree_map(lambda psi, nt_1: psi * nt_1, psi_tree, ffk_1())

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = tree_map(lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree)
        p_tree = self.compute_pressure(rho_tree)
        # Shan-Chen potential using modified pressure
        psi_tree = tree_map(lambda p, rho, G: jnp.sqrt(2 * (p - self.lattice.cs2 * rho) / G), p_tree, rho_tree, self.g_kkprime.diagonal().tolist())
        # Zhang-Chen potential is not used here
        U_tree = tree_map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return psi_tree, U_tree

    @partial(jit, static_argnums=(0, 2))
    def compute_force(self, rho_tree, inter_component_interaction=True):
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree, U_tree = self.compute_potential(rho_tree)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree, U_tree)
        if self.body_force is not None:
            return tree_map(lambda ff, rho: ff + self.body_force * rho, fluid_fluid_force, rho_tree)
        else:
            return fluid_fluid_force

    @partial(jit, static_argnums=(0,))
    def compute_force_central_moments(self, F_tree, F_intra_tree, psi_tree):
        def f(F, F_intra, sigma, psi, s_b):
            C = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype)
            Fx = F_intra[..., 0]
            Fy = F_intra[..., 1]
            Fz = F_intra[..., 2]
            eta = 2 * sigma * (Fx**2 + Fy**2 + Fz**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))  # For mechanical stability
            Fx = F[..., 0]
            Fy = F[..., 1]
            Fz = F[..., 2]
            C = C.at[..., 1].set(Fx)
            C = C.at[..., 2].set(Fy)
            C = C.at[..., 3].set(Fz)
            C = C.at[..., 7].set(eta)
            C = C.at[..., 8].set(eta)
            C = C.at[..., 9].set(eta)
            C = C.at[..., 10].set(Fx * self.lattice.cs2)
            C = C.at[..., 11].set(Fx * self.lattice.cs2)
            C = C.at[..., 12].set(Fy * self.lattice.cs2)
            C = C.at[..., 13].set(Fz * self.lattice.cs2)
            C = C.at[..., 14].set(Fy * self.lattice.cs2)
            C = C.at[..., 15].set(Fz * self.lattice.cs2)
            return C

        return tree_map(lambda F, F_intra, sigma, psi, s_b: f(F, F_intra, sigma, psi, s_b), F_tree, F_intra_tree, self.sigma, psi_tree, self.s_b)

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, Tdash_tree, rho_tree, u_tree):
        F_tree = self.compute_force(rho_tree)
        F_intra_tree = self.compute_force(rho_tree, inter_component_interaction=False)
        psi_tree, _ = self.compute_potential(rho_tree)
        C_tree = self.compute_force_central_moments(F_tree, F_intra_tree, psi_tree)
        Tf_tree = tree_map(lambda S, C: jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * S), self.S, C_tree)
        return tree_map(lambda Tdash, Tf: Tdash + Tf, Tdash_tree, Tf_tree)

    def output_data(self, **kwargs):
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        mask = np.array(self.solid_mask_streamed[0][..., 0])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho_water": rho_water[..., 0], "ux": u[..., 0], "uy": u[..., 1], "uz": u[..., 2]}
        save_fields_hdf5_xdmf(timestep, fields, "output", "data", static_fields={"mask": mask})

        # Measure radius from the area enclosed by droplet just next to the wall
        volume = np.sum(rho_water[:, :, :, 0] > 0.5 * (rho_w_l + rho_w_g))
        radius = (1.5 * volume / np.pi) ** (1 / 3)

        if radius <= 0:
            exit()

        if timestep == 3600:
            self.io_rate = 10
            self.print_info_rate = 10

        file.write(f"{timestep},{radius}\n")
        file.flush()


if __name__ == "__main__":
    # Cascaded LBM collision matrix
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

    precision = "f32/f32"

    # Water
    rho_w_l = 6.499210784
    rho_w_g = 0.379598891

    # initial liquid region width
    s2 = 0.8  # This sets the kinematic viscosity
    s_0 = [1.0]  # Mass conservation
    s_1 = [1.0]  # Fixed
    s_2 = [s2]
    s_b = [0.8]
    s_3 = [(16 - 8 * s2) / (8 - s2)]  # No slip
    s_4 = [1.0]  # [1.0]  # No slip

    g_kkprime = -1 * np.ones((1, 1))

    a = [3 / 49]
    b = [2 / 21]
    R = [1.0]
    Tc = 0.1093785558
    T = 0.86 * Tc
    kwargs = {"a": a, "b": b, "pr_omega": [0.344], "R": R, "T": T}
    eos = PengRobinson(**kwargs)

    # Initial semi circular droplet specification
    r = 40
    nx = 256
    ny = 256
    nz = 256
    width = 1

    # Circular wall
    R = 60
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    z = np.linspace(0, nz - 1, nz)
    x, y, z = np.meshgrid(x, y, z)
    sphere = (x - nx / 2) ** 2 + (y - ny / 2) ** 2 + (z - nz / 2 + 40) ** 2 - R**2
    ind = np.array(np.where(sphere <= 0), dtype=int).T

    theta_w = (np.pi / 3) * np.ones((nx, ny, nz, 1))
    phi_w = 1.8 * np.ones((nx, ny, nz, 1))
    delta_rho_w = 0.0 * np.ones((nx, ny, nz, 1))

    file = open("droplet_radius_3D.csv", "w", buffering=1)
    file.write("Timestep,Radius\n")
    file.flush()
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": g_kkprime,
        "EOS": eos,
        "M": [M],
        "s_0": s_0,
        "s_1": s_1,
        "s_b": s_b,
        "s_2": s_2,
        "s_3": s_3,
        "s_4": s_4,
        "sigma": [0.102],
        "precision": precision,
        "io_rate": 100,
        "print_info_rate": 100,
        "compute_MLUPS": False,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    subprocess.run(["rm", "-rf", "output/"], check=True)
    sim = DropletOnWall(**kwargs)
    try:
        sim.run(4000)
    finally:
        file.close()
