"""
Soil-water charateristic curve (SWCC) for a 256^3 sphere pack. First the 2 component droplet case and 2 component droplet on wall case is
performed for identifying the values of interaction strengths and wettability parameters.

The spherepack geometry used here is taken from Digital Rocks Portal and has porosity of 0.381
1. https://digitalporousmedia.org/published-datasets/tapis/projects/drp.project.published/drp.project.published.DRP-372/374_05_03/374_05_03_256/
"""

from jax_lab.lattice import LatticeD3Q19
from jax_lab.multiphase import MultiphaseBGK
from jax_lab.boundary_conditions import BounceBack, EquilibriumBC
from jax_lab.utils import save_fields_vtk

import h5py
import phantomgaze as pg
import matplotlib.pyplot as plt

from functools import partial
import os
import subprocess
import numpy as np
import jax.numpy as jnp
from jax import jit, config
from jax.tree import map as tree_map

# config.update("jax_default_matmul_precision", "float32")


# Multi-component droplet simulation to tune fluid-fluid interaction parameters and surface tension
class Droplet3D(MultiphaseBGK):
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

        # Air
        rho_inside = rho_a_g
        rho_outside = rho_a_l
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

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = tree_map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        U_tree = tree_map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None):
        p_tree = tree_map(lambda rho: rho * self.lattice.cs2, rho_tree)
        return p_tree

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree, rho_tree=None):
        p_water = p_tree[0]
        p_air = p_tree[1]
        return p_water + p_air + 3 * self.g_kkprime[0, 1] * p_air * p_water

    def output_data(self, **kwargs):
        rho = np.array(kwargs.get("rho_total")[0, ...])
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        rho_air = np.array(kwargs["rho_tree"][1][0, ...])
        p_water = np.array(kwargs["p_tree"][0][...])
        p_air = np.array(kwargs["p_tree"][1][...])
        u = np.array(kwargs["u_total"][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p_water": p_water[..., 0],
            "p_air": p_air[..., 0],
            "rho": rho[..., 0],
            "rho_water": rho_water[..., 0],
            "rho_air": rho_air[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "uz": u[..., 2],
        }
        offset = 60
        print(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p_air[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        p_south = p_air[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        p_west = p_air[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        p_east = p_air[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        p_back = p_air[self.nx // 2, self.ny // 2, self.nz // 2 - offset, 0]
        p_front = p_air[self.nx // 2, self.ny // 2, self.nz // 2 + offset, 0]
        pressure_difference = p_water[self.nx // 2, self.ny // 2, self.nz // 2, 0] - (p_north + p_south + p_west + p_east + p_front + p_back) / 6
        print(f"Pressure difference for radius = {r}: {pressure_difference}")

        rho_north = rho_water[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        rho_south = rho_water[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        rho_west = rho_water[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        rho_east = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        rho_back = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2 - offset, 0]
        rho_front = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2 + offset, 0]
        rho_g_pred = (rho_north + rho_south + rho_west + rho_east + rho_front + rho_back) / 6
        rho_l_pred = rho_water[self.nx // 2, self.ny // 2, self.nz // 2, 0]
        print(f"%Error Water Min: {(rho_g_pred - rho_w_g) * 100 / rho_w_g} Max: {(rho_l_pred - rho_w_l) * 100 / rho_w_l}")
        print(f"rho_l: {rho_l_pred}, rho_g: {rho_g_pred}")

        # HDF5/XDMF output option:
        # from jax_lab.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, f"output_{r}", "data")
        save_fields_vtk(timestep, fields, f"output_{r}", "data")
        if timestep == 20000:
            file.write(f"{r},{pressure_difference}\n")


# Multi-component droplet on wall example to tune contact angle
class DropletOnWall3D(MultiphaseBGK):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        dist = (x - self.nx / 2) ** 2 + (y - self.ny / 2) ** 2 + (z - self.nz / 2) ** 2 - r**2

        rho_inside = rho_w_l
        rho_outside = rho_w_g
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init(
            (self.nx, self.ny, self.nz, 1),
            self.precisionPolicy.compute_dtype,
            init_val=rho,
        )
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        # Air
        rho_inside = rho_a_g
        rho_outside = rho_a_l
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
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(
            BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta_w[tuple(ind.T)], phi_w[tuple(ind.T)], delta_rho_w[tuple(ind.T)])
        )
        self.BCs[1].append(
            BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta_a[tuple(ind.T)], phi_a[tuple(ind.T)], delta_rho_a[tuple(ind.T)])
        )

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = tree_map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        U_tree = tree_map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None):
        p_tree = tree_map(lambda rho: rho * self.lattice.cs2, rho_tree)
        return p_tree

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree, rho_tree=None):
        p_water = p_tree[0]
        p_air = p_tree[1]
        return p_water + p_air + 3 * self.g_kkprime[0, 1] * p_air * p_water

    def output_data(self, **kwargs):
        rho = np.array(kwargs.get("rho_total")[0, ...])
        rho_water = np.array(kwargs["rho_tree"][0][0, ...])
        rho_air = np.array(kwargs["rho_tree"][1][0, ...])
        u = np.array(kwargs["u_total"][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "rho": rho[..., 0],
            "rho_water": rho_water[..., 0],
            "rho_air": rho_air[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "uz": u[..., 2],
            "flag": self.solid_mask_streamed[0][..., 0],
        }
        # HDF5/XDMF output option:
        # from jax_lab.utils import save_fields_hdf5_xdmf
        # dynamic_fields = {key: value for key, value in fields.items() if key != "flag"}
        # static_fields = {"flag": fields["flag"]}
        # save_fields_hdf5_xdmf(timestep, dynamic_fields, "output", "data", static_fields=static_fields)
        save_fields_vtk(timestep, fields, "output", "data")


class DropletOnWall3DGeometric(DropletOnWall3D):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta_w[tuple(ind.T)]))
        self.BCs[1].append(BounceBack(tuple(ind.T), self.gridInfo, self.precisionPolicy, theta_a[tuple(ind.T)]))


class PorousMedia(MultiphaseBGK):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        if simulation == "imbibition":
            # Water
            rho = rho_w_g * np.ones((self.nx, self.ny, self.nz, 1))
            # rho[..., 0] = 0.5 * (rho_w_l + rho_w_g) - 0.5 * (
            #     rho_w_l - rho_w_g
            # ) * np.tanh(2 * (x - buffer) / width)
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)
            # air
            rho = rho_a_l * np.ones((self.nx, self.ny, self.nz, 1))
            # rho[..., 0] = 0.5 * (rho_a_g + rho_a_l) - 0.5 * (
            #     rho_a_g - rho_a_l
            # ) * np.tanh(2 * (x - buffer) / width)
            rho = self.distributed_array_init(
                (self.nx, self.ny, self.nz, 1),
                self.precisionPolicy.compute_dtype,
                init_val=rho,
            )
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)
            u = np.zeros((self.nx, self.ny, self.nz, 3))
            u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
            u = self.precisionPolicy.cast_to_output(u)
            u_tree = [u, u]
        else:
            # Water
            rho = rho_w_l * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)
            # air
            rho = rho_a_g * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
            rho = self.precisionPolicy.cast_to_output(rho)
            rho_tree.append(rho)

            u = np.zeros((self.nx, self.ny, self.nz, 3))
            u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
            u = self.precisionPolicy.cast_to_output(u)
            u_tree = [u, u]
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = tree_map(lambda rho: self.precisionPolicy.cast_to_compute(rho), rho_tree)
        U_tree = tree_map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None):
        p_tree = tree_map(lambda rho: rho * self.lattice.cs2, rho_tree)
        return p_tree

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree, rho_tree=None):
        p_water = p_tree[0]
        p_air = p_tree[1]
        return p_water + p_air + 3 * self.g_kkprime[0, 1] * p_air * p_water

    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        if simulation == "imbibition":
            inlet = self.boundingBoxIndices["left"]
            rho_inlet = (rho_w_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            vel = 0.004 * np.ones((inlet.shape[0], 3), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
            rho_inlet = (rho_a_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
            outlet = self.boundingBoxIndices["right"]
            rho_outlet = (rho_w_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))
            rho_outlet = (rho_a_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))
            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            wall = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_w[wall], phi_w[wall], delta_rho_w[wall]))
            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            wall = tuple(idx.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_a[wall], phi_a[wall], delta_rho_a[wall]))
        else:
            inlet = self.boundingBoxIndices["left"]
            rho_inlet = (rho_w_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            vel = 0.004 * np.ones((inlet.shape[0], 3), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
            rho_inlet = (rho_a_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
            outlet = self.boundingBoxIndices["right"]
            rho_outlet = (rho_w_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))
            rho_outlet = (rho_a_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))
            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            wall = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_w[wall], phi_w[wall], delta_rho_w[wall]))
            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            wall = tuple(idx.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_a[wall], phi_a[wall], delta_rho_a[wall]))

    def output_data(self, **kwargs):
        rho = np.array(kwargs.get("rho_total")[0, :, 1:-1, 1:-1, :])
        p_water = np.array(kwargs.get("p_tree")[0][:, 1:-1, 1:-1, :])
        p_air = np.array(kwargs.get("p_tree")[1][:, 1:-1, 1:-1, :])
        u = np.array(kwargs.get("u_total")[0, :, 1:-1, 1:-1, :])
        rho_water = np.array(kwargs.get("rho_tree")[0][0, :, 1:-1, 1:-1, :])
        rho_air = np.array(kwargs.get("rho_tree")[1][0, :, 1:-1, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {
            "rho": rho[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "uz": u[..., 2],
            "rho_water": rho_water[..., 0],
            "rho_air": rho_air[..., 0],
            "p_air": p_air[..., 0],
            "p_water": p_water[..., 0],
            "flag": self.solid_mask_streamed[0][:, 1:-1, 1:-1, 0],
        }
        # HDF5/XDMF output option:
        # from jax_lab.utils import save_fields_hdf5_xdmf
        # dynamic_fields = {key: value for key, value in fields.items() if key != "flag"}
        # static_fields = {"flag": fields["flag"]}
        # save_fields_hdf5_xdmf(timestep, dynamic_fields, f"output_{simulation}", "data", static_fields=static_fields)
        save_fields_vtk(timestep, fields, f"output_{simulation}", "data")

        # Computing capillary pressure, saturation using values inside porous media only
        rho_water = rho_water[buffer : buffer + self.nx, ...]
        rho_air = rho_air[buffer : buffer + self.nx, ...]
        porous = np.array(self.solid_mask_streamed[0][buffer : buffer + self.nx, 1:-1, 1:-1, 0])
        p_water = p_water[buffer : buffer + self.nx, ...]
        p_air = p_air[buffer : buffer + self.nx, ...]
        water = (rho_water > rho_air)[..., 0]
        air = (rho_water < rho_air)[..., 0]
        p_wetting = np.mean(p_water[(~porous & water) == 1])
        p_nonwetting = np.mean(p_air[(~porous & air) == 1])

        P_c = p_nonwetting - p_wetting
        v_w = np.sum((~porous & water))
        v_nw = np.sum((~porous & air))
        S = v_w / (v_w + v_nw)
        file.write(f"{P_c},{S}\n")

        red = pg.SolidColor(color=(1.0, 0.0, 0.0), opacity=1.0)
        grey = pg.SolidColor(color=(0.439, 0.475, 0.757), opacity=0.04)

        dx, dy, dz = (0.01, 0.01, 0.01)
        origin = (0.0, 0.0, 0.0)

        if simulation == "imbibition":
            rho_volume = pg.objects.Volume(
                jnp.array(
                    rho_water[0 : self.nx - 2 * buffer, ..., 0],
                    dtype=self.precisionPolicy.compute_dtype,
                ),
                spacing=(dx, dy, dz),
                origin=origin,
            )
        else:
            rho_volume = pg.objects.Volume(
                jnp.array(
                    rho_air[0 : self.nx - 2 * buffer, ..., 0],
                    dtype=self.precisionPolicy.compute_dtype,
                ),
                spacing=(dx, dy, dz),
                origin=origin,
            )
        boundary_volume = pg.objects.Volume(
            jnp.array(porous[0 : self.nx - 2 * buffer, ...], dtype=jnp.float32),
            spacing=(dx, dy, dz),
            origin=origin,
        )

        # Get camera parameters
        radius = 80
        angle = 20 * np.pi / 180
        focal_point = (self.nx * dx / 2, 3 * self.ny * dy / 4, self.nz * dz / 2)
        camera_position = (
            self.nx * dx + radius * np.cos(angle) * dx,
            -0.1,
            -self.nz * dz + radius * np.sin(angle) * dz,
        )

        # Rotate camera
        camera = pg.Camera(
            position=camera_position,
            focal_point=focal_point,
            view_up=(0.0, -1.0, 0.0),
            height=2160,
            width=3840,
            background=pg.SolidBackground(color=(1.0, 1.0, 1.0)),
        )

        screen_buffer = pg.render.contour(rho_volume, threshold=0.95, colormap=red, camera=camera)
        screen_buffer = pg.render.contour(boundary_volume, camera, threshold=0.95, colormap=grey, screen_buffer=screen_buffer)
        screen_buffer = pg.render.wireframe(
            lower_bound=(0, 0, 0),
            upper_bound=((self.nx - 2 * buffer) * dx, self.ny * dy, self.nz * dz),
            color=pg.SolidColor(color=(0.0, 0.0, 0.0)),
            thickness=0.0025,
            camera=camera,
            screen_buffer=screen_buffer,
        )

        plt.imsave(
            f"images/porous_{simulation}" + str(kwargs["timestep"]).zfill(7) + ".png",
            np.minimum(screen_buffer.image.get(), 1.0),
        )


class PorousMediaGeometric(PorousMedia):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        if simulation == "imbibition":
            inlet = self.boundingBoxIndices["left"]
            rho_inlet = (rho_w_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            vel = 0.004 * np.ones((inlet.shape[0], 3), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
            rho_inlet = (rho_a_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
            outlet = self.boundingBoxIndices["right"]
            rho_outlet = (rho_w_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))
            rho_outlet = (rho_a_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))
            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            wall = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_w[wall]))
            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            wall = tuple(idx.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_a[wall]))
        else:
            inlet = self.boundingBoxIndices["left"]
            rho_inlet = (rho_w_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            vel = 0.004 * np.ones((inlet.shape[0], 3), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
            rho_inlet = (rho_a_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel))
            outlet = self.boundingBoxIndices["right"]
            rho_outlet = (rho_w_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))
            rho_outlet = (rho_a_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.gridInfo, self.precisionPolicy, rho_outlet, vel))
            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            wall = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_w[wall]))
            wall = np.concatenate((
                self.boundingBoxIndices["top"],
                self.boundingBoxIndices["bottom"],
                self.boundingBoxIndices["front"],
                self.boundingBoxIndices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy))
            wall = tuple(idx.T)
            self.BCs[1].append(BounceBack(wall, self.gridInfo, self.precisionPolicy, theta_a[wall]))


if __name__ == "__main__":
    nx = 256
    ny = 256
    nz = 256

    x = np.linspace(0, nx - 1, nx, dtype=int)
    y = np.linspace(0, ny - 1, ny, dtype=int)
    z = np.linspace(0, nz - 1, nz, dtype=int)
    x, y, z = np.meshgrid(x, y, z)

    precision = "f32/f32"
    g_kkprime = -0.06 * np.ones((2, 2))
    g_kkprime[0, 1] = 0.54
    g_kkprime[1, 0] = 0.54

    width = 4  # Liquid vapor interface width

    # Water properties in Lattice units
    rho_w_l = 2.0
    rho_w_g = 0.1
    tau_w = 1.0

    # Air properties (subcritical)
    rho_a_l = 2.0
    rho_a_g = 0.1
    tau_a = 1.0

    A = np.zeros((2, 2))

    subprocess.run("rm -rf output*/", shell=True, check=True)
    file = open("surface_tension.txt", "w")
    file.write("Radius, Pressure Difference\n")
    R = [25, 30, 35, 40]
    for r in R:
        kwargs = {
            "n_components": 2,
            "lattice": LatticeD3Q19(precision),
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "body_force": [0.0, 0.0, 0.0],
            "g_kkprime": g_kkprime,
            "omega": [tau_w, tau_a],
            "precision": precision,
            "k": [1.0, 1.0],
            "A": A,
            "io_rate": 20000,
            "compute_MLUPS": False,
            "print_info_rate": 20000,
            "checkpoint_rate": -1,
            "checkpoint_dir": os.path.abspath("./checkpoints_"),
            "restore_checkpoint": False,
        }
        sim = Droplet3D(**kwargs)
        sim.run(20000)
    file.close()

    # Contact angle determination on spherical wall
    R = 30
    r = 25
    sphere = (x - nx / 2) ** 2 + (y - ny / 2) ** 2 + (z - nz / 2 + R + 24) ** 2 - R**2
    ind = np.array(np.where(sphere <= 0)).T

    # Setting contact angle
    theta_w = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    theta_w[ind] = np.pi / 6
    phi_w = np.ones((nx, ny, nz, 1))
    theta_a = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    theta_a[ind] = np.pi - np.pi / 6
    phi_w[ind] = 1.0
    phi_a = np.ones((nx, ny, nz, 1))
    delta_rho_w = np.zeros((nx, ny, nz, 1))
    delta_rho_a = 0.2 * np.ones((nx, ny, nz, 1))

    kwargs = {
        "n_components": 2,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "body_force": [0.0, 0.0, 0.0],
        "g_kkprime": g_kkprime,
        "omega": [1 / tau_w, 1 / tau_a],
        "precision": precision,
        "k": [1.0, 1.0],
        "A": A,
        "io_rate": 10000,
        "compute_MLUPS": False,
        "print_info_rate": 10000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    sim = DropletOnWall3D(**kwargs)
    sim.run(20000)

    # Saturation curves
    buffer = 8
    geometry = h5py.File("assets/374_05_03_256.mat", "r")
    _bin = np.array(geometry["bin"], dtype=int)
    # _bin = _bin[0:nx, 0:ny, 0:nz]
    nx = nx + 2 * buffer
    ind = np.where(_bin == 1.0)
    idx = np.zeros((len(ind[0]), 3), dtype=int)

    # porous geometry
    idx[:, 0] = ind[0] + buffer
    idx[:, 1] = ind[1]
    idx[:, 2] = ind[2]

    theta_w = (np.pi / 2) * np.ones((nx, ny, nz, 1))
    theta_w[tuple(idx.T)] = np.pi / 6
    theta_a = (np.pi / 2) * np.ones((nx, ny, nz, 1))

    phi_w = np.ones((nx, ny, nz, 1))
    phi_w[tuple(idx.T)] = 1.0
    phi_a = np.ones((nx, ny, nz, 1))

    delta_rho_w = np.zeros((nx, ny, nz, 1))
    delta_rho_a = np.zeros((nx, ny, nz, 1))
    delta_rho_a[tuple(idx.T)] = 0.2

    # Saturation curves
    drho = 0.0092
    subprocess.run("rm -rf output*", shell=True, check=True)
    subprocess.run(["rm", "-f", "characteristic_curve_imbibition.txt"], check=True)
    file = open("characteristic_curve_imbibition.txt", "w")
    file.write("Capillary Pressure,Saturation\n")
    simulation = "imbibition"
    kwargs = {
        "n_components": 2,
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": g_kkprime,
        "body_force": [0.0, 0.0, 0.0],
        "omega": [1.0, 1.0],
        "precision": precision,
        "k": [1.0, 1.0],
        "A": np.zeros((2, 2)),
        "io_rate": 1000,
        "print_info_rate": 1000,
        "compute_MLUPS": False,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
    }
    subprocess.run("rm -rf output*", shell=True, check=True)
    sim = PorousMedia(**kwargs)
    sim.run(250000)
    file.close()

    # Drainage
    drho = 0.0092
    simulation = "drainage"
    sim = PorousMedia(**kwargs)
    subprocess.run(["rm", "-f", "characteristic_curve_drainage.txt"], check=True)
    file = open("characteristic_curve_drainage.txt", "w")
    file.write("Capillary Pressure,Saturation\n")
    sim.run(250000)
    file.close()
