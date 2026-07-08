"""
Multi-component droplet on wall example. Used for parameter tuning (phi and delta_rho) for setting contact angle for improved virtual density scheme.

The collision matrix is based on:
1. Coveney, P. V. et al. Multiple-relaxation-time lattice Boltzmann models in three dimensions.
Philosophical Transactions of the Royal Society of London.
Series A: Mathematical, Physical and Engineering Sciences 360, 437–451 (2002).
"""

import os
import numpy as np
from jax import config

from src.lattice import LatticeD3Q19
from src.multiphase import MultiphaseMRT
from src.eos import Peng_Robinson
from src.boundary_conditions import BounceBack
from src.utils import save_fields_vtk


# config.update("jax_default_matmul_precision", "float32")


class DropletOnWall3D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - self.nz / 2 - 70) ** 2)
        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)
        # rho[ind[:, 0], ind[:, 1]] = 1.0
        rho = rho.reshape((nx, ny, nz, 1))
        rho = self.distributed_array_init(
            (self.nx, self.ny, self.nz, 1),
            self.precisionPolicy.compute_dtype,
            init_val=rho,
        )
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = [rho]

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        self.BCs[0].append(BounceBack(ind, self.gridInfo, self.precisionPolicy, theta[ind], phi[ind], delta_rho[ind]))

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1], "uz": u[..., 2], "flag": np.array(self.solid_mask_streamed[0][..., 0])}
        u_sp = np.sqrt(np.sum(np.square(u), axis=-1))
        fluid_mask = np.ones(rho.shape[:-1], dtype=bool)
        fluid_mask[ind] = False
        rho_scalar = rho[..., 0]
        f = np.array(kwargs["f_poststreaming_tree"][0])
        f_mass = np.sum(f, dtype=np.float64)
        total_mass = np.sum(rho_scalar, dtype=np.float64)
        fluid_mass = np.sum(rho_scalar[fluid_mask], dtype=np.float64)
        solid_mass = np.sum(rho_scalar[~fluid_mask], dtype=np.float64)
        print(f"Max spurious velocity: {np.max(u_sp)}")
        print(f"Distribution mass: {f_mass}")
        print(f"Total mass: {total_mass}")
        print(f"Fluid mass: {fluid_mass}")
        print(f"Solid mass: {solid_mass}")
        # HDF5/XDMF output option:
        # from src.utils import save_fields_hdf5_xdmf
        # dynamic_fields = {key: value for key, value in fields.items() if key != "flag"}
        # static_fields = {"flag": fields["flag"]}
        # save_fields_hdf5_xdmf(timestep, dynamic_fields, "output", "data", static_fields=static_fields)
        save_fields_vtk(timestep, fields, "output", "data")


class DropletOnWall3DGeometric(DropletOnWall3D):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(BounceBack(ind, self.gridInfo, self.precisionPolicy, theta[ind]))


if __name__ == "__main__":
    precision = "f32/f32"

    # Initial semi circular droplet specification
    r = 40
    nx = 300
    ny = 300
    nz = 350
    width = 4

    # Spherical wall
    R = 70
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    z = np.linspace(0, nz - 1, nz)
    x, y, z = np.meshgrid(x, y, z)
    sphere = (x - nx / 2) ** 2 + (y - ny / 2) ** 2 + (z - nz / 2 + 40) ** 2 - R**2
    ind = np.array(np.where(sphere <= 0), dtype=int)
    ind = tuple(ind)

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

    s_rho = [0.0]  # Mass
    s_e = [1.0]
    s_eta = [1.0]
    s_j = [0.0]  # Momentum
    s_q = [1.0]
    s_v = [1 / tau]
    s_m = [1.0]
    s_pi = [1.0]
    s_v = [1.0]

    theta = 30 * np.pi / 180 * np.ones((nx, ny, nz, 1))
    phi = 0.0 * np.ones((nx, ny, nz, 1))
    delta_rho = 1.0 * np.ones((nx, ny, nz, 1))

    kwargs = {
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_components": 1,
        "body_force": [0.0, 0.0, 0.0],
        "g_kkprime": g_kkprime,
        "M": [M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_pi": s_pi,
        "s_m": s_m,
        "s_v": [1.0],
        "EOS": eos,
        "kappa": [0],
        # values not used, can be set as anything
        "k": [1.0],
        "A": 0.0 * np.zeros((1, 1)),
        "precision": precision,
        "io_rate": 1000,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,  # Disable checkpointing
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
        "wetting_formulation": "geometric",
    }
    os.system("rm -rf output*/")
    sim = DropletOnWall3DGeometric(**kwargs)
    sim.run(20000)
