"""
Single component droplet impact on wall simulation, where liquid droplet is suspended in its vapor. The density of each region is computed using Maxwell's
Construction. The density profile is initialized with smooth profile with specified interface width. Boundary conditions is BounceBack at the top and
bottom, periodic everywhere else. This example demonstrates how to do in-situ GPU rendering using PhantomGaze

The collision matrix is based on:
1. Coveney, P. V. et al. Multiple–relaxation–time lattice Boltzmann models in three dimensions. Philosophical Transactions of the Royal Society of London.
Series A: Mathematical, Physical and Engineering Sciences 360, 437–451 (2002).
"""

import os
import subprocess

import numpy as np

from jax_lab.boundary_conditions import BounceBackHalfway
from jax_lab.lattice import LatticeD3Q19
from jax_lab.multiphase import MultiphaseMRT
from jax_lab.eos import Peng_Robinson

import matplotlib.pyplot as plt
import phantomgaze as pg

from jax import config
import jax.numpy as jnp

# config.update("jax_default_matmul_precision", "float32")


class DropletOnWall3D(MultiphaseMRT):
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

        rho = jnp.zeros((self.nx, self.ny, self.nz, 1), dtype=self.precisionPolicy.compute_dtype)
        rho = rho.at[..., 0].set(0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width))

        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = jnp.zeros((self.nx, self.ny, self.nz, 3), dtype=self.precisionPolicy.compute_dtype)
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        walls = np.concatenate((self.boundingBoxIndices["front"], self.boundingBoxIndices["back"]))
        walls = tuple(walls.T)
        self.BCs[0].append(
            BounceBackHalfway(walls, self.gridInfo, self.precisionPolicy, vel=None, theta=theta[walls], phi=phi[walls], delta_rho=delta_rho[walls])
        )

        self.visualization_bc = jnp.zeros((self.nx, self.ny, self.nz), dtype=jnp.float32)
        self.visualization_bc = self.visualization_bc.at[tuple(self.boundingBoxIndices["back"].T)].set(1.0)

    def output_data(self, **kwargs):
        rho = jnp.array(kwargs["rho_tree"][0][0, ..., 0], dtype=self.precisionPolicy.compute_dtype)

        red = pg.SolidColor(color=(1.0, 0.0, 0.0), opacity=1.0)
        grey = pg.SolidColor(color=(0.9607, 0.8941, 0.8313), opacity=1.0)

        dx, dy, dz = (0.01, 0.01, 0.01)
        origin = (0.0, 0.0, 0.0)

        rho_volume = pg.objects.Volume(rho, spacing=(dx, dy, dz), origin=origin)
        boundary_volume = pg.objects.Volume(self.visualization_bc, spacing=(dx, dy, dz), origin=origin)

        # Get camera parameters
        focal_point = (self.nx * dx / 2, self.ny * dy / 2, self.nz * dz)
        camera_position = (self.nx * dx / 2, self.ny * dy / 2, -1.2 * self.nz * dz)

        # Rotate camera
        camera = pg.Camera(
            position=camera_position,
            focal_point=focal_point,
            view_up=(0.0, -1.0, 0.0),
            height=2160,
            width=3840,
            background=pg.SolidBackground(color=(1.0, 1.0, 1.0)),
        )

        screen_buffer = pg.render.contour(rho_volume, threshold=7.6, colormap=red, camera=camera)
        screen_buffer = pg.render.contour(boundary_volume, camera, threshold=0.95, colormap=grey, screen_buffer=screen_buffer)

        # HDF5/XDMF output option:
        # from jax_lab.utils import save_fields_hdf5_xdmf
        # fields = {"rho": np.array(rho)}
        # static_fields = {"flag": np.array(self.visualization_bc)}
        # save_fields_hdf5_xdmf(kwargs["timestep"], fields, "output", "data", static_fields=static_fields)

        plt.imsave("droplet_impact" + str(kwargs["timestep"]).zfill(7) + ".png", np.minimum(screen_buffer.image.get(), 1.0))


class DropletOnWall3DGeometric(DropletOnWall3D):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        walls = np.concatenate((self.boundingBoxIndices["front"], self.boundingBoxIndices["back"]))
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBackHalfway(walls, self.gridInfo, self.precisionPolicy, vel=None, theta=theta[walls]))

        self.visualization_bc = jnp.zeros((self.nx, self.ny, self.nz), dtype=jnp.float32)
        self.visualization_bc = self.visualization_bc.at[tuple(self.boundingBoxIndices["back"].T)].set(1.0)


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
    eos = Peng_Robinson(**kwargs)

    theta = (12 * np.pi / 180) * np.ones((nx, ny, nz, 1))
    phi = 1.05 * np.ones((nx, ny, nz, 1))
    delta_rho = np.zeros((nx, ny, nz, 1))

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
    sim = DropletOnWall3D(**kwargs)
    sim.run(10000)
