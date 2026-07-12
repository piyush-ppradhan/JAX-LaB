"""
Multi-component droplet on wall example. Used for parameter tuning (phi and delta_rho) for setting contact angle. Parameters set here are used in channe_rel_perm example.
Unlike Free-Energy method, contact angle in Shan-Chen method is dependent on the interaction coefficient between wall and the fluid and it changes with lattice size. A good approach to
set interaction strengths can be use equivalent size domain and then compute contact angle v/s inital volume (radius) if it convergences after some volume sizes, that intetraction value is
good enough.

The collision matrix is based on:
1. McCracken, M. E. & Abraham, J. Multiple-relaxation-time lattice-Boltzmann model for multiphase flow. Phys. Rev. E 71, 036701 (2005).
"""

import os
import subprocess
import numpy as np
from jax import config

from jax_lab.lattice import LatticeD2Q9
from jax_lab.multiphase import MultiphaseMRT
from jax_lab.eos import Peng_Robinson
from jax_lab.boundary_conditions import BounceBack
from jax_lab.utils import save_fields_vtk

# config.update("jax_default_matmul_precision", "float32")


class DropletOnWall2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2 - 100) ** 2)
        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)
        # rho[ind[:, 0], ind[:, 1]] = 1.0
        rho = rho.reshape((nx, ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = [rho]

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(
            BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta[tuple(ind.T)], phi[tuple(ind.T)], delta_rho[tuple(ind.T)])
        )

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1], "flag": np.array(self.solid_mask_streamed[0][..., 0])}
        u_sp = np.sqrt(np.sum(np.square(u), axis=-1))
        print(f"Max spurious velocity: {np.max(u_sp)}")
        # HDF5/XDMF output option:
        # from jax_lab.utils import save_fields_hdf5_xdmf
        # dynamic_fields = {key: value for key, value in fields.items() if key != "flag"}
        # static_fields = {"flag": fields["flag"]}
        # save_fields_hdf5_xdmf(timestep, dynamic_fields, "output", "data", static_fields=static_fields)
        save_fields_vtk(timestep, fields, "output", "data")


class DropletOnWall2DGeometric(DropletOnWall2D):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta[tuple(ind.T)]))


if __name__ == "__main__":
    precision = "f32/f32"

    # Initial semi circular droplet specification
    r = 50
    nx = 300
    ny = 350
    width = 4

    # Circular wall
    R = 70
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    x, y = np.meshgrid(x, y)
    x = x.T
    y = y.T
    circ = (x - nx / 2) ** 2 + (y - ny / 2 + 20) ** 2 - R**2
    ind = np.array(np.where(circ <= 0), dtype=int).T

    visc = 0.15
    tau = 3 * visc + 0.5

    # Peng-Robinson droplet specification
    a = 3 / 49
    b = 2 / 21
    Tc = 0.1093785558
    T = 0.86 * Tc
    kwargs = {"a": a, "b": b, "pr_omega": 0.344, "R": 1.0, "T": T}
    eos = Peng_Robinson(**kwargs)
    rho_g = 0.379598891
    rho_l = 6.499210784

    g_kkprime = -1 * np.ones((1, 1))

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

    s_rho = [0.0]  # Mass
    s_e = [1.0]
    s_eta = [1.0]
    s_j = [0.0]  # Momentum
    s_q = [1.0]
    s_v = [1 / tau]

    theta = 5 * np.pi / 180 * np.ones((nx, ny, 1))
    phi = 0.0 * np.ones((nx, ny, 1))
    delta_rho = 0.0 * np.ones((nx, ny, 1))
    kwargs = {
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "n_components": 1,
        "body_force": [0.0, 0.0],
        "g_kkprime": g_kkprime,
        "M": [M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "EOS": eos,
        "kappa": [0],
        "k": [1.0],
        "A": 0.0 * np.zeros((1, 1)),
        "precision": precision,
        "io_rate": 1000,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,  # Disable checkpointing
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
        "wetting_formulation": "geometric",  # Remove and use DropletOnWall2D for improved virtual density scheme
    }
    subprocess.run("rm -rf output*/", shell=True, check=True)
    sim = DropletOnWall2DGeometric(**kwargs)
    sim.run(20000)
