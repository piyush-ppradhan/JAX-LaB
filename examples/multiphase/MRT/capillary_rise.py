"""
Single component 2D capillary rise example where a channel is initially submerged in liquid. The density of each region is computed using Maxwell's Construction. The density profile is initialized with smooth profile with specified
interface thickness.
"""

import os

import numpy as np

from src.lattice import LatticeD2Q9
from src.multiphase import MultiphaseMRT
from src.boundary_conditions import BounceBack
from src.utils import save_fields_vtk
from src.eos import VanderWaal

from jax import config

# config.update("jax_default_matmul_precision", "float32")


# Estimate surface tension
class Droplet2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)
        x = x.T
        y = y.T
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2)
        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = []
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        offset_x = 95
        offset_y = 95
        rho_north = rho[self.nx // 2, self.ny // 2 - offset_y, 0]
        rho_south = rho[self.nx // 2, self.ny // 2 + offset_y, 0]
        rho_west = rho[self.nx // 2 - offset_x, self.ny // 2, 0]
        rho_east = rho[self.nx // 2 + offset_x, self.ny // 2, 0]
        rho_g_pred = 0.25 * (rho_north + rho_south + rho_west + rho_east)
        rho_l_pred = rho[self.nx // 2, self.ny // 2, 0]
        print(f"%Error Min: {(rho_g_pred - rho_g) * 100 / rho_g} Max: {(rho_l_pred - rho_l) * 100 / rho_l}")
        print(f"Density: Min: {rho_g_pred} Max: {rho_l_pred}")
        print(f"Maxwell construction: Min: {rho_g} Max: {rho_l}")
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p[self.nx // 2, self.ny // 2 - offset_y, 0]
        p_south = p[self.nx // 2, self.ny // 2 + offset_y, 0]
        p_west = p[self.nx // 2 - offset_x, self.ny // 2, 0]
        p_east = p[self.nx // 2 + offset_x, self.ny // 2, 0]
        pressure_difference = p[self.nx // 2, self.ny // 2, 0] - 0.25 * (p_north + p_south + p_west + p_east)
        print(f"Pressure difference: {pressure_difference}")
        if timestep == 60000:
            file.write(f"{1 / r},{pressure_difference}\n")
        # HDF5/XDMF output option:
        # from src.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, f"output_{r}", "data")
        save_fields_vtk(timestep, fields, f"output_{r}", "data")


# Estimate contact angle
class DropletOnSurface2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)
        x = x.T
        y = y.T
        dist = np.sqrt((x - self.nx / 2) ** 2 + (y + disp) ** 2)
        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = []
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta[walls], phi[walls], delta_rho[walls]))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        p = np.array(kwargs["p"][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"p": p[..., 0], "rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        # HDF5/XDMF output option:
        # from src.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, f"output_{disp}", "data")
        save_fields_vtk(timestep, fields, f"output_{disp}", "data")


class DropletOnSurface2DGeometric(DropletOnSurface2D):
    def set_boundary_conditions(self):
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta[walls]))


class CapillaryRise2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        rho_profile = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (x - L) / width)
        rho = rho_g * np.ones((self.nx, self.ny, 1))
        rho[:, :, 0] = rho_profile.reshape((self.nx, 1))

        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree = [rho]

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]

        return rho_tree, u_tree

    def set_boundary_conditions(self):
        top_wall = np.array(
            [[x, y] for x in range(150, 451) for y in range(29)],
            dtype=np.int32,
        )
        bottom_wall = np.array(
            [[x, y] for x in range(150, 451) for y in range(self.ny - 28, self.ny)],
            dtype=np.int32,
        )
        walls = np.concatenate((top_wall, bottom_wall))
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta[walls], phi[walls], delta_rho[walls]))

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_prev_tree"][0][0, ...])
        p = np.array(kwargs["p_tree"][0][...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        fields = {"flag": self.solid_mask_streamed[0][..., 0], "p": p[..., 0], "rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        # HDF5/XDMF output option:
        # from src.utils import save_fields_hdf5_xdmf
        # dynamic_fields = {key: value for key, value in fields.items() if key != "flag"}
        # static_fields = {"flag": fields["flag"]}
        # save_fields_hdf5_xdmf(timestep, dynamic_fields, "output_", "data", static_fields=static_fields)
        save_fields_vtk(timestep, fields, "output_", "data")
        ind_mid = np.argmin(p[150:451, self.ny // 2, 0])
        ind_side = np.argmin(p[150:451, self.ny - 30, 0])
        meniscus_position = ind_mid
        meniscus_height = ind_side - ind_mid
        file.write(f"{timestep},{meniscus_height},{meniscus_position}\n")


class CapillaryRise2DGeometric(CapillaryRise2D):
    def set_boundary_conditions(self):
        top_wall = np.array(
            [[x, y] for x in range(150, 451) for y in range(29)],
            dtype=np.int32,
        )
        bottom_wall = np.array(
            [[x, y] for x in range(150, 451) for y in range(self.ny - 28, self.ny)],
            dtype=np.int32,
        )
        walls = np.concatenate((top_wall, bottom_wall))
        walls = tuple(walls.T)
        self.BCs[0].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta[walls]))


if __name__ == "__main__":
    nx = 600
    ny = 100
    width = 1  # Initial Liquid-vapor interface thickness

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
    s_e = [0.5]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1.0]

    a = 9 / 49
    b = 2 / 21
    R = 1.0

    rho_l = 7.491548920
    rho_g = 0.448078056
    Tc = 0.5714285714
    T = 0.7 * Tc

    kwargs = {"a": a, "b": b, "R": R, "T": T}
    eos = VanderWaal(**kwargs)

    precision = "f32/f32"

    # Estimate surface tension
    # os.system("rm -rf output* surface_tension.txt")
    file = open("surface_tension.txt", "w")
    file.write("Inverse Radius,Pressure Difference\n")
    file.write("0,0\n")
    for r in [20, 25, 30, 35]:
        kwargs = {
            "n_components": 1,
            "lattice": LatticeD2Q9(precision),
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "body_force": [0.0, 0.0],
            "g_kkprime": -1 * np.ones((1, 1)),
            "k": [0.15],
            "A": -0.115 * np.ones((1, 1)),
            "M": [M],
            "s_rho": s_rho,
            "s_e": s_e,
            "s_eta": s_eta,
            "s_j": s_j,
            "s_q": s_q,
            "s_v": s_v,
            "kappa": [0.5],
            "precision": precision,
            "EOS": eos,
            "io_rate": 60000,
            "print_info_rate": 60000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }
        sim = Droplet2D(**kwargs)
        sim.run(60000)
    file.close()

    theta = 19.1 * (np.pi / 180) * np.ones((nx, ny, 1))
    phi = 1.4 * np.ones((nx, ny, 1))
    delta_rho = np.zeros((nx, ny, 1))

    # Estimate contact angle
    os.system("rm -rf output* *.vtk")
    r = 50
    Disp = [0, 10, 20, 30]
    for disp in Disp:
        kwargs = {
            "n_components": 1,
            "lattice": LatticeD2Q9(precision),
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "body_force": [0.0, 0.0],
            "g_kkprime": -1 * np.ones((1, 1)),
            "k": [0.15],
            "A": -0.115 * np.ones((1, 1)),
            "M": [M],
            "s_rho": s_rho,
            "s_e": s_e,
            "s_eta": s_eta,
            "s_j": s_j,
            "s_q": s_q,
            "s_v": s_v,
            "kappa": [0.5],
            "precision": precision,
            "io_rate": 40000,
            "EOS": eos,
            "print_info_rate": 40000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }
        sim = DropletOnSurface2D(**kwargs)
        sim.run(400000)

    L = 150
    theta = 19.1 * (np.pi / 180) * np.ones((nx, ny, 1))
    phi = 1.4 * np.ones((nx, ny, 1))
    delta_rho = np.zeros((nx, ny, 1))

    file = open("lucas_washburn.txt", "w")
    file.write("Time,Menicus Height,Position\n")
    os.system("rm -rf output* *.vtk")
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "body_force": [0.0, 0.0],
        "g_kkprime": -1 * np.ones((1, 1)),
        "k": [0.15],
        "A": -0.115 * np.ones((1, 1)),
        "M": [M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "kappa": [0.5],
        "EOS": eos,
        "precision": precision,
        "io_rate": 1000,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    sim = CapillaryRise2D(**kwargs)
    sim.run(50000)
    file.close()
