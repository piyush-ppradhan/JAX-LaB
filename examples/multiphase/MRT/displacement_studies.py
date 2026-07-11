"""
Drainage simulation for a Berea sandstone. First the 2 component droplet case and 2 component droplet on wall case is
performed for identifying the values of various parameters.

The Berea sandstone geometry used here is taken from Digital Rocks Portal:
1. E. Santos, Javier, Chang, Bernard, Kang, Qinjun, Viswanathan, Hari, Lubbers, Nicholas, Gigliotti, Alex, Prodanovic, Masa.
"3D Dataset of Simulations." Digital Rocks Portal,  Digital Rocks Portal, https://www.doi.org/10.17612/93pd-y471.

The collision matrix is based on:
1. Coveney, P. V. et al. Multiple–relaxation–time lattice Boltzmann models in three dimensions. Philosophical Transactions of the Royal Society of
London. Series A: Mathematical, Physical and Engineering Sciences 360, 437–451 (2002).
"""

import os
import subprocess
import numpy as np

from jax_lab.lattice import LatticeD3Q19
from jax_lab.eos import Peng_Robinson
from jax_lab.multiphase import MultiphaseMRT
from jax_lab.boundary_conditions import BounceBack
from jax_lab.utils import save_fields_vtk

import h5py

# config.update("jax_default_matmul_precision", "float32")


# Multi-component droplet simulation to tune fluid-fluid interaction parameters and surface tension
class Droplet3D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        dist = (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - self.nz / 2) ** 2 - r**2

        # Water
        rho_inside = rho_w_l
        rho_outside = rho_w_g
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        # CO2
        rho_inside = rho_c_g
        rho_outside = rho_c_l
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs.get("rho_total")[0, ...])
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        rho_CO2 = np.array(kwargs["rho_tree"][1][0, ...])
        p_water = np.array(kwargs["p_tree"][0][...])
        p_CO2 = np.array(kwargs["p_tree"][1][...])
        p = np.array(kwargs["p"][0, ...])
        u_water = np.array(kwargs["u_tree"][0][0, ...])
        u_CO2 = np.array(kwargs["u_tree"][1][0, ...])
        u = np.array(kwargs["u_total"][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "p_water": p_water[..., 0],
            "p_CO2": p_CO2[..., 0],
            "rho": rho[..., 0],
            "rho_water": rho_water[..., 0],
            "rho_CO2": rho_CO2[..., 0],
            "ux_CO2": u_CO2[..., 0],
            "uy_CO2": u_CO2[..., 1],
            "ux_water": u_water[..., 0],
            "uy_water": u_water[..., 1],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        offset = 65
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        p_south = p[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        p_west = p[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        p_east = p[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        p_back = p[self.nx // 2, self.ny // 2, self.nz // 2 - offset, 0]
        p_front = p[self.nx // 2, self.ny // 2, self.nz // 2 + offset, 0]
        pressure_difference = p[self.nx // 2, self.ny // 2, self.nz // 2, 0] - (p_north + p_south + p_west + p_east + p_front + p_back) / 6
        print(f"Pressure difference for radius = {r}: {pressure_difference}")

        rho_north = rho_CO2[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        rho_south = rho_CO2[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        rho_west = rho_CO2[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        rho_east = rho_CO2[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        rho_back = rho_CO2[self.nx // 2 + offset, self.ny // 2, self.nz // 2 - offset, 0]
        rho_front = rho_CO2[self.nx // 2 + offset, self.ny // 2, self.nz // 2 + offset, 0]
        rho_l_pred = (rho_north + rho_south + rho_west + rho_east + rho_front + rho_back) / 6
        rho_g_pred = rho_CO2[self.nx // 2, self.ny // 2, self.nz // 2, 0]
        print(f"%Error CO2 Min: {(rho_g_pred - rho_c_g) * 100 / rho_c_g} Max: {(rho_l_pred - rho_c_l) * 100 / rho_c_l}")

        rho_north = rho_water[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        rho_south = rho_water[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        rho_west = rho_water[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        rho_east = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        rho_back = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2 - offset, 0]
        rho_front = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2 + offset, 0]
        rho_g_pred = (rho_north + rho_south + rho_west + rho_east + rho_front + rho_back) / 6
        rho_l_pred = rho_water[self.nx // 2, self.ny // 2, self.nz // 2, 0]
        print(f"%Error Water Min: {(rho_g_pred - rho_w_g) * 100 / rho_w_g} Max: {(rho_l_pred - rho_w_l) * 100 / rho_w_l}")

        # HDF5/XDMF output option:
        # from jax_lab.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, f"output_{r}", "data")
        save_fields_vtk(timestep, fields, f"output_{r}", "data")


# Multi-component droplet on wall example to tune contact angle
class DropletOnWall3D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        dist = (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - self.nz / 2) ** 2 - r**2
        rho_tree = []

        # Water
        rho_inside = rho_w_l
        rho_outside = rho_w_g
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        # CO2
        rho_inside = rho_c_g
        rho_outside = rho_c_l
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        self.BCs[0].append(
            BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta_w[tuple(ind.T)], phi_w[tuple(ind.T)], delta_rho_w[tuple(ind.T)])
        )
        self.BCs[1].append(
            BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta_c[tuple(ind.T)], phi_c[tuple(ind.T)], delta_rho_c[tuple(ind.T)])
        )

    def output_data(self, **kwargs):
        rho = np.array(kwargs.get("rho_total")[0, ...])
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        rho_CO2 = np.array(kwargs["rho_tree"][1][0, ...])
        p_water = np.array(kwargs["p_tree"][0][...])
        p_CO2 = np.array(kwargs["p_tree"][1][...])
        u_water = np.array(kwargs["u_tree"][0][0, ...])
        u_CO2 = np.array(kwargs["u_tree"][1][0, ...])
        u = np.array(kwargs["u_total"][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p_water": p_water[..., 0],
            "p_CO2": p_CO2[..., 0],
            "rho": rho[..., 0],
            "rho_water": rho_water[..., 0],
            "rho_CO2": rho_CO2[..., 0],
            "ux_CO2": u_CO2[..., 0],
            "uy_CO2": u_CO2[..., 1],
            "ux_water": u_water[..., 0],
            "uy_water": u_water[..., 1],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        offset = 65
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")

        rho_north = rho_CO2[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        rho_south = rho_CO2[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        rho_west = rho_CO2[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        rho_east = rho_CO2[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        rho_back = rho_CO2[self.nx // 2 + offset, self.ny // 2, self.nz // 2 - offset, 0]
        rho_front = rho_CO2[self.nx // 2 + offset, self.ny // 2, self.nz // 2 + offset, 0]
        rho_l_pred = (rho_north + rho_south + rho_west + rho_east + rho_front + rho_back) / 6
        rho_g_pred = rho_CO2[self.nx // 2, self.ny // 2, self.nz // 2, 0]
        print(f"%Error CO2 Min: {(rho_g_pred - rho_c_g) * 100 / rho_c_g} Max: {(rho_l_pred - rho_c_l) * 100 / rho_c_l}")

        rho_north = rho_water[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        rho_south = rho_water[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        rho_west = rho_water[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        rho_east = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        rho_back = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2 - offset, 0]
        rho_front = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2 + offset, 0]
        rho_g_pred = (rho_north + rho_south + rho_west + rho_east + rho_front + rho_back) / 6
        rho_l_pred = rho_water[self.nx // 2, self.ny // 2, self.nz // 2, 0]
        print(f"%Error Water Min: {(rho_g_pred - rho_w_g) * 100 / rho_w_g} Max: {(rho_l_pred - rho_w_l) * 100 / rho_w_l}")

        # HDF5/XDMF output option:
        # from jax_lab.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, f"output_{r}", "data")
        save_fields_vtk(timestep, fields, f"output_{r}", "data")


class DropletOnWall3DGeometric(DropletOnWall3D):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta_w[tuple(ind.T)]))
        self.BCs[1].append(BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta_c[tuple(ind.T)]))


class PorousMedia(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        rho_tree = []

        # Water
        # Initialize the density field smoothly to prevent large Shan-Chen forces initially
        # x = np.linspace(0, self.nx - 1, self.nx)
        # y = np.linspace(0, self.ny - 1, self.ny)
        # z = np.linspace(0, self.nz - 1, self.nz)
        # x, _, _ = np.meshgrid(x, y, z)
        # x = np.transpose(x, (1, 0, 2))
        # rho_1 = fraction * rho_w
        # rho_2 = (1 - fraction) * rho_w
        # rho_ = 0.5 * (rho_w_g + rho_w_l) - 0.5 * (rho_w_g - rho_w_l) * np.tanh(
        #     2 * (x - (buffer - 10)) / width
        # )
        rho = rho_w_l * np.ones((self.nx, self.ny, self.nz, 1))
        # rho[tuple(idx.T)] = 1.0
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        # CO2
        rho = rho_c_g * np.ones((self.nx, self.ny, self.nz, 1))
        rho[0:buffer, ...] = rho_c_l
        # rho[tuple(idx.T)] = 1.0
        # rho_1 = (1 - fraction) * rho_c
        # rho_2 = fraction * rho_c
        # rho_ = 0.5 * (rho_c_l + rho_c_g) - 0.5 * (rho_c_l - rho_c_g) * np.tanh(
        #     2 * (x - buffer) / width
        # )
        # rho[..., 0] = rho_
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # apply inlet equilibrium boundary condition at the left
        # inlet = self.boundingBoxIndices["left"]
        # rho_inlet = rho_w_l * np.ones(
        #     (inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype
        # )
        # vel_inlet = np.zeros(
        #     (inlet.shape[0], 3), dtype=self.precisionPolicy.compute_dtype
        # )
        # vel_inlet[:, 0] = prescribed_vel
        # self.BCs[0].append(
        #     EquilibriumBC(
        #         tuple(inlet.T),
        #         self.gridInfo,
        #         self.precisionPolicy,
        #         rho_inlet,
        #         vel_inlet,
        #     )
        # )
        # rho_inlet = rho_c_l * np.ones(
        #     (inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype
        # )
        # vel_inlet = np.zeros(
        #     (inlet.shape[0], 3), dtype=self.precisionPolicy.compute_dtype
        # )
        # vel_inlet[:, 0] = prescribed_vel
        # self.BCs[1].append(
        #     EquilibriumBC(
        #         tuple(inlet.T),
        #         self.gridInfo,
        #         self.precisionPolicy,
        #         rho_inlet,
        #         vel_inlet,
        #     )
        # )
        #
        # Same at the outlet
        # outlet = self.boundingBoxIndices["right"]
        # rho_outlet = rho_w_l * np.ones(
        #     (outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype
        # )
        # vel_outlet = np.zeros(
        #     (outlet.shape[0], 3), dtype=self.precisionPolicy.compute_dtype
        # )
        # self.BCs[0].append(
        #     ConvectiveOutflow(
        #         tuple(outlet.T),
        #         self.gridInfo,
        #         self.precisionPolicy,
        #     )
        # )
        # rho_outlet = rho_c_g * np.ones(
        #     (outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype
        # )
        # self.BCs[1].append(
        #     ConvectiveOutflow(
        #         tuple(outlet.T),
        #         self.gridInfo,
        #         self.precisionPolicy,
        #     )
        # )

        # Wall boundary condition
        # wall = np.concatenate(
        #     (
        #         idx,
        #         self.boundingBoxIndices["top"],
        #         self.boundingBoxIndices["bottom"],
        #         self.boundingBoxIndices["front"],
        #         self.boundingBoxIndices["back"],
        #     )
        # )
        wall = idx
        wall = tuple(wall.T)
        self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_w[wall], phi_w[wall], delta_rho_w[wall]))
        self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_c[wall], phi_c[wall], delta_rho_c[wall]))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs.get("rho_total")[0, 1:-1, 1:-1, 1:-1, :])
        p_water = np.array(kwargs.get("p_tree")[0][1:-1, 1:-1, 1:-1])
        p_CO2 = np.array(kwargs.get("p_tree")[1][1:-1, 1:-1, 1:-1])
        u = np.array(kwargs.get("u_total")[0, 1:-1, 1:-1, 1:-1, :])
        rho_water = np.array(kwargs.get("rho_tree")[0][0, 1:-1, 1:-1, 1:-1, :])
        rho_CO2 = np.array(kwargs.get("rho_tree")[1][0, 1:-1, 1:-1, 1:-1, :])
        u_water = np.array(kwargs["u_tree"][0][0, 1:-1, 1:-1, 1:-1, :])
        u_CO2 = np.array(kwargs["u_tree"][1][0, 1:-1, 1:-1, 1:-1, :])
        timestep = kwargs["timestep"]
        # breakpoint()
        fields = {
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "uz": u[..., 2],
            "ux_CO2": u_CO2[..., 0],
            "uy_CO2": u_CO2[..., 1],
            "uz_CO2": u_CO2[..., 2],
            "ux_water": u_water[..., 0],
            "uy_water": u_water[..., 1],
            "uz_water": u_water[..., 2],
            "rho_water": rho_water[..., 0],
            "rho_CO2": rho_CO2[..., 0],
            "p_CO2": p_CO2[..., 0],
            "p_water": p_water[..., 0],
            "phi_w": phi_w[1:-1, 1:-1, 1:-1, 0],
            "theta_c": theta_c[1:-1, 1:-1, 1:-1, 0],
            "flag": self.solid_mask_streamed[0][1:-1, 1:-1, 1:-1, 0],
        }
        # HDF5/XDMF output option:
        # from jax_lab.utils import save_fields_hdf5_xdmf
        # dynamic_fields = {key: value for key, value in fields.items() if key != "flag"}
        # static_fields = {"flag": fields["flag"]}
        # save_fields_hdf5_xdmf(timestep, dynamic_fields, "output", "data", static_fields=static_fields)
        save_fields_vtk(timestep, fields, "output", "data")


class PorousMediaGeometric(PorousMedia):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        # Wall boundary condition
        wall = idx
        wall = tuple(wall.T)
        self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_w[wall]))
        self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_c[wall]))


if __name__ == "__main__":
    nx = 150
    ny = 150
    nz = 150

    x = np.linspace(0, nx - 1, nx, dtype=int)
    y = np.linspace(0, ny - 1, ny, dtype=int)
    z = np.linspace(0, nz - 1, nz, dtype=int)
    x, y, z = np.meshgrid(x, y, z)

    precision = "f32/f32"
    g_kkprime = -1 * np.ones((2, 2))
    g = 0.00323
    g_kkprime[0, 1] = g
    g_kkprime[1, 0] = g

    width = 5  # Liquid vapor interface width

    # Scale used for SI -> Lattice unit conversion
    C_l = 1e-6  # m
    C_s = 1.089e-8  # s
    C_rho = 105.9  # kg/m^3

    # Water properties in Lattice units
    rho_w_l = 8.1665  # liquid phase density
    rho_w_g = 0.2580  # gas phase density (solubility of water in CO2)
    nu_w = (1.59e-3 * 1e-4) * C_s / C_l**2  # 1.59e-3 cm^2/s converted to lattice units
    tau_w = 1.43  # 3 * nu_w + 0.5
    Tc_w = 0.03646 * 473.15 / 647.1  # Set temperature as 200C = 473.15K, which is converted to lattice units using critical temp. in LU: 0.03646

    # CO2 properties (subcritical)
    rho_c_l = 2.4454  # liquid phase density
    rho_c_g = 0.4980  # gas phase density (solubility of CO2 in water)
    nu_c = (1.11e-3 * 1e-4) * C_s / C_l**2  # 1.11e-3 cm^2/s converted to lattice units
    tau_c = 1.0  # 3 * nu_c + 0.5

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

    s_rho = [0.0, 0.0]  # Mass
    s_e = [0.8, 0.8]
    s_eta = [0.8, 0.8]
    s_j = [0.0, 0.0]  # Momentum
    s_q = [1.1, 1.1]
    s_m = [1.0, 1.0]
    s_pi = [1.0, 1.0]
    s_v = [1 / tau_w, 1 / tau_c]

    # Peng-Robinson EOS used for modeling fluids. For scCO2, the same PR equation is used as it is accurate for supercritical CO2
    a = [1 / 49, 0.01348]
    b = [2 / 21, 0.13385]
    R = [1, 1]
    pr_omega = [0.344, 0.22491]
    T = Tc_w

    A = np.zeros((2, 2))
    A_kk = 0.0
    A[0, 0] = A_kk
    A[1, 1] = A_kk

    kwargs = {"a": a, "b": b, "pr_omega": pr_omega, "R": R, "T": T}
    eos = Peng_Robinson(**kwargs)

    for r in [25, 30, 35, 40, 45, 50]:
        kwargs = {
            "n_components": 2,
            "lattice": LatticeD3Q19(precision),
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "body_force": [0.0, 0.0, 0.0],
            "g_kkprime": g_kkprime,
            "precision": precision,
            "EOS": eos,
            "M": [M, M],
            "s_rho": s_rho,
            "s_e": s_e,
            "s_eta": s_eta,
            "s_j": s_j,
            "s_q": s_q,
            "s_v": s_v,
            "s_m": s_m,
            "s_pi": s_pi,
            "kappa": [0, 0],
            "k": [0.125, 0.125],
            "A": A,
            "io_rate": 10000,
            "compute_MLUPS": False,
            "print_info_rate": 10000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }

        subprocess.run(f"rm -rf output_{r}/ *.vtk", shell=True, check=True)
        sim = Droplet3D(**kwargs)
        sim.run(30000)

    # Contact angle determination
    # Spherical wall
    R = 38
    sphere = (x - nx / 2) ** 2 + (y - ny / 2) ** 2 + (z - nz / 2 + R) ** 2 - R**2
    ind = np.array(np.where(sphere <= 0)).T

    # Setting contact angle
    theta_w = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    theta_w[ind] = np.pi / 6
    phi_w = np.ones((nx, ny, nz, 1))
    theta_c = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    phi_w[ind] = 1.17
    phi_c = np.ones((nx, ny, nz, 1))
    delta_rho_w = np.zeros((nx, ny, nz, 1))
    delta_rho_c = np.zeros((nx, ny, nz, 1))

    kwargs = {
        "n_components": 2,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "body_force": [0.0, 0.0, 0.0],
        "g_kkprime": g_kkprime,
        "precision": precision,
        "EOS": eos,
        "M": [M, M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": s_v,
        "s_m": s_m,
        "s_pi": s_pi,
        "kappa": [0, 0],
        "k": [0.125, 0.125],
        "A": A,
        "io_rate": 10000,
        "compute_MLUPS": False,
        "print_info_rate": 10000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    sim = DropletOnWall3D(**kwargs)
    sim.run(30000)

    # Steady state droplet simulation co-existence density
    rho_w_l = 8.105
    rho_w_g = 0.016
    rho_c_l = 2.562
    rho_c_g = 0.387

    width = 10
    buffer = 52

    nx = 256 + buffer + 4
    ny = 256
    nz = 256

    geometry = h5py.File("./assets/374_09_03_256.mat", "r")
    _bin = np.array(geometry["bin"], dtype=int)
    ind = np.where(_bin == 1.0)
    idx = np.zeros((len(ind[0]), 3), dtype=int)
    idx[:, 0] = ind[0] + buffer
    idx[:, 1] = ind[1]
    idx[:, 2] = ind[2]

    theta_w = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    theta_w[tuple(idx.T)] = np.pi / 6
    # theta_w[[buffer, 255 + buffer], ...] = np.pi / 2
    theta_c = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    # theta_c[tuple(idx.T)] = np.pi - np.pi / 6

    phi_w = np.ones((nx, ny, nz, 1))
    phi_w[tuple(idx.T)] = 1.17
    # phi_w[[buffer, 255 + buffer], ...] = 1.0
    phi_c = np.ones((nx, ny, nz, 1))

    delta_rho_w = np.zeros((nx, ny, nz, 1))
    delta_rho_c = np.zeros((nx, ny, nz, 1))
    # delta_rho_c[tuple(idx.T)] = 1.0

    # prescribed_vel = 1e-3

    # Drainage simulation
    kwargs = {
        "n_components": 2,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": g_kkprime,
        "body_force": [1e-4, 0.0, 0.0],
        "precision": precision,
        "M": [M, M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_v": [1 / tau_w, 1 / tau_c],
        "s_pi": s_pi,
        "s_m": s_m,
        "kappa": [0.0, 0.0],
        "k": [0.125, 0.125],
        "EOS": eos,
        "A": np.zeros((2, 2)),
        "io_rate": 1000,
        "compute_MLUPS": False,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    subprocess.run("rm -rf output*", shell=True, check=True)
    sim = PorousMedia(**kwargs)
    sim.run(30000)
