"""
Single component droplet impact on wall simulation, where liquid droplet is suspended in its vapor. The density of each region is computed using Maxwell's
Construction. The density profile is initialized with smooth profile with specified interface width. Boundary conditions is BounceBack at the top and
bottom, periodic everywhere else. This example is very similar to droplet_impact_on_wall example in the same folder.

The collision matrix is based on:
1. Coveney, P. V. et al. Multiple–relaxation–time lattice Boltzmann models in three dimensions. Philosophical Transactions of the Royal Society of London.
Series A: Mathematical, Physical and Engineering Sciences 360, 437–451 (2002).
"""

import os
import subprocess

import numpy as np

from jax_lab.core.boundary_conditions import BounceBackHalfway
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseMRT
from jax_lab.core.eos import PengRobinson
from jax_lab.core.utils import save_fields_vtk

from jax_lab.render import Light, Scene, SurfaceRendering

import jax
import jax.numpy as jnp

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

jax.config.update("jax_default_matmul_precision", "float32")


class DropletOnLiquid3D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        z = np.linspace(0, self.nz - 1, self.nz, dtype=int)
        x, y, z = np.meshgrid(x, y, z)
        x = np.transpose(x, axes=(1, 0, 2))
        y = np.transpose(y, axes=(1, 0, 2))
        z = np.transpose(z, axes=(1, 0, 2))

        rho_tree = []

        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 5) ** 2 + (z - self.nz / 2) ** 2)
        sphere = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        rho_profile = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (y - L) / width)

        rho = np.zeros((self.nx, self.ny, self.nz, 1))
        rho[:, 0 : self.ny, :, 0] = rho_profile[::-1].reshape((-1, 1))
        rho[..., 0] = rho[..., 0] + sphere - rho_g

        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = jnp.zeros((self.nx, self.ny, self.nz, 3), dtype=self.precision_policy.compute_dtype)
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        walls = np.array(
            [[i, j, k] for i in range(self.nx) for j in [0, 1, 2, self.ny - 3, self.ny - 2, self.ny - 1] for k in range(self.nz)], dtype=int
        )
        walls = tuple(walls.T)
        self.BCs[0].append(
            BounceBackHalfway(walls, self.grid_info, self.precision_policy, theta=theta[walls], phi=phi[walls], delta_rho=delta_rho[walls])
        )

    def output_data(self, **kwargs):
        rho = jnp.array(kwargs["rho_tree"][0][0, ..., 0], dtype=self.precision_policy.compute_dtype)
        dx, dy, dz = (0.01, 0.01, 0.01)
        focal_point = (self.nx * dx / 2, self.ny * dy / 2, self.nz * dz)
        camera_position = (self.nx * dx / 2, self.ny * dy / 2, -1.2 * self.nz * dz)
        scene = Scene(
            {
                "density": SurfaceRendering(
                    value_range=(7.6, rho_l),
                    color=(1.0, 0.0, 0.0),
                    metallic=0.0,
                    roughness=0.35,
                    spacing=(dx, dy, dz),
                )
            },
            position=camera_position,
            target=focal_point,
            up=(0.0, -1.0, 0.0),
            resolution=(1920, 1080),
            background_color=(1.0, 1.0, 1.0),
            lights=(Light(position=(0.0, -1.0, -1.0), intensity=5.0),),
            output_dir=".",
        )
        scene.render(
            {"density": rho},
            timestep=kwargs["timestep"],
            filename=f"droplet_impact_liquid{kwargs['timestep']:07d}.png",
        )

        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p_tree"][0][...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "uz": u[..., 2],
        }
        # HDF5/XDMF output option:
        # from jax_lab.core.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(
        #     timestep,
        #     fields,
        #     "output",
        #     "data",
        # )
        save_fields_vtk(
            timestep,
            fields,
            "output",
            "data",
        )


class DropletOnLiquid3DGeometric(DropletOnLiquid3D):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        walls = np.array(
            [[i, j, k] for i in range(self.nx) for j in [0, 1, 2, self.ny - 3, self.ny - 2, self.ny - 1] for k in range(self.nz)], dtype=int
        )
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBackHalfway(walls, self.grid_info, self.precision_policy, theta=theta[walls]))


if __name__ == "__main__":
    s_rho = [1.0]
    s_e = [0.5]
    s_eta = [1.0]
    s_j = [1.0]
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
    nx = 400
    ny = 200
    nz = 200

    a = 3 / 49
    b = 2 / 21
    R = 1.0
    pr_omega = 0.344

    rho_l = 8.425313934
    rho_g = 0.025644451
    Tc = 0.1093785558
    T = 0.7 * Tc

    kwargs = {"a": a, "b": b, "R": R, "pr_omega": pr_omega, "T": T}
    eos = PengRobinson(**kwargs)

    theta = (12 * np.pi / 180) * np.ones((nx, ny, nz, 1))
    phi = 1.05 * np.ones((nx, ny, nz, 1))
    delta_rho = np.zeros((nx, ny, nz, 1))

    # Thickness of liquid layer on which the droplet will impact
    L = 40

    precision = "f32/f32"
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": eos,
        "body_force": [0.0, 1e-5, 0.0],
        "k": [0.05],
        "A": -0.028 * np.ones((1, 1)),
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
        "io_rate": 100,
        "compute_MLUPS": False,
        "print_info_rate": 100,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }

    subprocess.run("rm -rf output*/ *.vtk droplet*", shell=True, check=True)
    sim = DropletOnLiquid3D(**kwargs)
    sim.run(10000)
