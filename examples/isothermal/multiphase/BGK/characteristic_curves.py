"""
Soil-water charateristic curve (SWCC) for a 256^3 sphere pack. First the 2 component droplet case and 2 component droplet on wall case is
performed for identifying the values of interaction strengths and wettability parameters.

The spherepack geometry used here is taken from Digital Rocks Portal and has porosity of 0.381
1. https://digitalporousmedia.org/published-datasets/tapis/projects/drp.project.published/drp.project.published.DRP-372/374_05_03/374_05_03_256/
"""

from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseBGK
from jax_lab.core.boundary_conditions import BounceBack, EquilibriumBC
from jax_lab.core.utils import save_fields_vtk
from jax_lab.render import Light, Scene, SurfaceRendering

import h5py

from functools import partial
import os
import subprocess
import numpy as np
import jax.numpy as jnp
from jax import jit, config
from jax.tree import map as tree_map

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

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
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        # Air
        rho_inside = rho_a_g
        rho_outside = rho_a_l
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = tree_map(lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree)
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
        logger.info(f"Spurious currents: {np.max(np.sqrt(np.sum(u**2, axis=-1)))}")
        p_north = p_air[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        p_south = p_air[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        p_west = p_air[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        p_east = p_air[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        p_back = p_air[self.nx // 2, self.ny // 2, self.nz // 2 - offset, 0]
        p_front = p_air[self.nx // 2, self.ny // 2, self.nz // 2 + offset, 0]
        pressure_difference = p_water[self.nx // 2, self.ny // 2, self.nz // 2, 0] - (p_north + p_south + p_west + p_east + p_front + p_back) / 6
        logger.info(f"Pressure difference for radius = {r}: {pressure_difference}")

        rho_north = rho_water[self.nx // 2, self.ny // 2 - offset, self.nz // 2, 0]
        rho_south = rho_water[self.nx // 2, self.ny // 2 + offset, self.nz // 2, 0]
        rho_west = rho_water[self.nx // 2 - offset, self.ny // 2, self.nz // 2, 0]
        rho_east = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2, 0]
        rho_back = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2 - offset, 0]
        rho_front = rho_water[self.nx // 2 + offset, self.ny // 2, self.nz // 2 + offset, 0]
        rho_g_pred = (rho_north + rho_south + rho_west + rho_east + rho_front + rho_back) / 6
        rho_l_pred = rho_water[self.nx // 2, self.ny // 2, self.nz // 2, 0]
        logger.info(f"%Error Water Min: {(rho_g_pred - rho_w_g) * 100 / rho_w_g} Max: {(rho_l_pred - rho_w_l) * 100 / rho_w_l}")
        logger.info(f"rho_l: {rho_l_pred}, rho_g: {rho_g_pred}")

        # HDF5/XDMF output option:
        # from jax_lab.core.utils import save_fields_hdf5_xdmf
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
            self.precision_policy.compute_dtype,
            init_val=rho,
        )
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        # Air
        rho_inside = rho_a_g
        rho_outside = rho_a_l
        rho = 0.5 * (rho_inside + rho_outside) - 0.5 * (rho_inside - rho_outside) * np.tanh(2 * (dist - r) / width)
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.precision_policy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(
            BounceBack(tuple(ind.T), self.grid_info, self.precision_policy, theta_w[tuple(ind.T)], phi_w[tuple(ind.T)], delta_rho_w[tuple(ind.T)])
        )
        self.BCs[1].append(
            BounceBack(tuple(ind.T), self.grid_info, self.precision_policy, theta_a[tuple(ind.T)], phi_a[tuple(ind.T)], delta_rho_a[tuple(ind.T)])
        )

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = tree_map(lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree)
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
        # from jax_lab.core.utils import save_fields_hdf5_xdmf
        # dynamic_fields = {key: value for key, value in fields.items() if key != "flag"}
        # static_fields = {"flag": fields["flag"]}
        # save_fields_hdf5_xdmf(timestep, dynamic_fields, "output", "data", static_fields=static_fields)
        save_fields_vtk(timestep, fields, "output", "data")


class DropletOnWall3DGeometric(DropletOnWall3D):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        self.BCs[0].append(BounceBack(tuple(ind.T), self.grid_info, self.precision_policy, theta_w[tuple(ind.T)]))
        self.BCs[1].append(BounceBack(tuple(ind.T), self.grid_info, self.precision_policy, theta_a[tuple(ind.T)]))


class PorousMedia(MultiphaseBGK):
    def initialize_macroscopic_fields(self):
        rho_tree = []
        if simulation == "imbibition":
            # Water
            rho = rho_w_g * np.ones((self.nx, self.ny, self.nz, 1))
            # rho[..., 0] = 0.5 * (rho_w_l + rho_w_g) - 0.5 * (
            #     rho_w_l - rho_w_g
            # ) * np.tanh(2 * (x - buffer) / width)
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
            rho = self.precision_policy.cast_to_output(rho)
            rho_tree.append(rho)
            # air
            rho = rho_a_l * np.ones((self.nx, self.ny, self.nz, 1))
            # rho[..., 0] = 0.5 * (rho_a_g + rho_a_l) - 0.5 * (
            #     rho_a_g - rho_a_l
            # ) * np.tanh(2 * (x - buffer) / width)
            rho = self.distributed_array_init(
                (self.nx, self.ny, self.nz, 1),
                self.precision_policy.compute_dtype,
                init_val=rho,
            )
            rho = self.precision_policy.cast_to_output(rho)
            rho_tree.append(rho)
            u = np.zeros((self.nx, self.ny, self.nz, 3))
            u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype, init_val=u)
            u = self.precision_policy.cast_to_output(u)
            u_tree = [u, u]
        else:
            # Water
            rho = rho_w_l * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
            rho = self.precision_policy.cast_to_output(rho)
            rho_tree.append(rho)
            # air
            rho = rho_a_g * np.ones((self.nx, self.ny, self.nz, 1))
            rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
            rho = self.precision_policy.cast_to_output(rho)
            rho_tree.append(rho)

            u = np.zeros((self.nx, self.ny, self.nz, 3))
            u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype, init_val=u)
            u = self.precision_policy.cast_to_output(u)
            u_tree = [u, u]
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        rho_tree = tree_map(lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree)
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
            inlet = self.bounding_box_indices["left"]
            rho_inlet = (rho_w_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            vel = 0.004 * np.ones((inlet.shape[0], 3), dtype=self.precision_policy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel))
            rho_inlet = (rho_a_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel))
            outlet = self.bounding_box_indices["right"]
            rho_outlet = (rho_w_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel))
            rho_outlet = (rho_a_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel))
            wall = np.concatenate((
                self.bounding_box_indices["top"],
                self.bounding_box_indices["bottom"],
                self.bounding_box_indices["front"],
                self.bounding_box_indices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[0].append(BounceBack(wall, self.grid_info, self.precision_policy))
            wall = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall, self.grid_info, self.precision_policy, theta_w[wall], phi_w[wall], delta_rho_w[wall]))
            wall = np.concatenate((
                self.bounding_box_indices["top"],
                self.bounding_box_indices["bottom"],
                self.bounding_box_indices["front"],
                self.bounding_box_indices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[1].append(BounceBack(wall, self.grid_info, self.precision_policy))
            wall = tuple(idx.T)
            self.BCs[1].append(BounceBack(wall, self.grid_info, self.precision_policy, theta_a[wall], phi_a[wall], delta_rho_a[wall]))
        else:
            inlet = self.bounding_box_indices["left"]
            rho_inlet = (rho_w_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            vel = 0.004 * np.ones((inlet.shape[0], 3), dtype=self.precision_policy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel))
            rho_inlet = (rho_a_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel))
            outlet = self.bounding_box_indices["right"]
            rho_outlet = (rho_w_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel))
            rho_outlet = (rho_a_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel))
            wall = np.concatenate((
                self.bounding_box_indices["top"],
                self.bounding_box_indices["bottom"],
                self.bounding_box_indices["front"],
                self.bounding_box_indices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[0].append(BounceBack(wall, self.grid_info, self.precision_policy))
            wall = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall, self.grid_info, self.precision_policy, theta_w[wall], phi_w[wall], delta_rho_w[wall]))
            wall = np.concatenate((
                self.bounding_box_indices["top"],
                self.bounding_box_indices["bottom"],
                self.bounding_box_indices["front"],
                self.bounding_box_indices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[1].append(BounceBack(wall, self.grid_info, self.precision_policy))
            wall = tuple(idx.T)
            self.BCs[1].append(BounceBack(wall, self.grid_info, self.precision_policy, theta_a[wall], phi_a[wall], delta_rho_a[wall]))

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
        # from jax_lab.core.utils import save_fields_hdf5_xdmf
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

        dx, dy, dz = (0.01, 0.01, 0.01)
        if simulation == "imbibition":
            rendered_density = jnp.array(
                rho_water[0 : self.nx - 2 * buffer, ..., 0],
                dtype=self.precision_policy.compute_dtype,
            )
        else:
            rendered_density = jnp.array(
                rho_air[0 : self.nx - 2 * buffer, ..., 0],
                dtype=self.precision_policy.compute_dtype,
            )
        rendered_boundary = jnp.array(porous[0 : self.nx - 2 * buffer, ...], dtype=jnp.float32)

        # Get camera parameters
        radius = 80
        angle = 20 * np.pi / 180
        focal_point = (self.nx * dx / 2, 3 * self.ny * dy / 4, self.nz * dz / 2)
        camera_position = (
            self.nx * dx + radius * np.cos(angle) * dx,
            -0.1,
            -self.nz * dz + radius * np.sin(angle) * dz,
        )

        scene = Scene(
            {
                "density": SurfaceRendering(
                    value_range=(0.95, 1.2),
                    color=(1.0, 0.0, 0.0),
                    metallic=0.0,
                    roughness=0.4,
                    spacing=(dx, dy, dz),
                ),
                "porous_medium": SurfaceRendering(
                    value_range=(0.95, 1.0),
                    color=(0.439, 0.475, 0.757),
                    metallic=0.0,
                    roughness=0.75,
                    opacity=0.04,
                    spacing=(dx, dy, dz),
                ),
            },
            position=camera_position,
            target=focal_point,
            up=(0.0, -1.0, 0.0),
            resolution=(1920, 1080),
            background_color=(1.0, 1.0, 1.0),
            lights=(Light(position=(0.0, -1.0, -1.0), intensity=5.0),),
            output_dir="images",
        )
        scene.render(
            {"density": rendered_density, "porous_medium": rendered_boundary},
            timestep=kwargs["timestep"],
            filename=f"porous_{simulation}{kwargs['timestep']:07d}.png",
        )


class PorousMediaGeometric(PorousMedia):
    def set_boundary_conditions(self):
        # Only theta is used for geometric wetting scheme so other parameters (phi, delta_rho) do not need to be passed as they will be ignored.
        if simulation == "imbibition":
            inlet = self.bounding_box_indices["left"]
            rho_inlet = (rho_w_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            vel = 0.004 * np.ones((inlet.shape[0], 3), dtype=self.precision_policy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel))
            rho_inlet = (rho_a_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel))
            outlet = self.bounding_box_indices["right"]
            rho_outlet = (rho_w_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel))
            rho_outlet = (rho_a_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel))
            wall = np.concatenate((
                self.bounding_box_indices["top"],
                self.bounding_box_indices["bottom"],
                self.bounding_box_indices["front"],
                self.bounding_box_indices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[0].append(BounceBack(wall, self.grid_info, self.precision_policy))
            wall = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall, self.grid_info, self.precision_policy, theta_w[wall]))
            wall = np.concatenate((
                self.bounding_box_indices["top"],
                self.bounding_box_indices["bottom"],
                self.bounding_box_indices["front"],
                self.bounding_box_indices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[1].append(BounceBack(wall, self.grid_info, self.precision_policy))
            wall = tuple(idx.T)
            self.BCs[1].append(BounceBack(wall, self.grid_info, self.precision_policy, theta_a[wall]))
        else:
            inlet = self.bounding_box_indices["left"]
            rho_inlet = (rho_w_l + drho) * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            vel = 0.004 * np.ones((inlet.shape[0], 3), dtype=self.precision_policy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel))
            rho_inlet = (rho_a_g + drho) * np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel))
            outlet = self.bounding_box_indices["right"]
            rho_outlet = (rho_w_l - drho) * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[0].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel))
            rho_outlet = (rho_a_g - drho) * np.ones((outlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
            self.BCs[1].append(EquilibriumBC(tuple(outlet.T), self.grid_info, self.precision_policy, rho_outlet, vel))
            wall = np.concatenate((
                self.bounding_box_indices["top"],
                self.bounding_box_indices["bottom"],
                self.bounding_box_indices["front"],
                self.bounding_box_indices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[0].append(BounceBack(wall, self.grid_info, self.precision_policy))
            wall = tuple(idx.T)
            self.BCs[0].append(BounceBack(wall, self.grid_info, self.precision_policy, theta_w[wall]))
            wall = np.concatenate((
                self.bounding_box_indices["top"],
                self.bounding_box_indices["bottom"],
                self.bounding_box_indices["front"],
                self.bounding_box_indices["back"],
            ))
            wall = tuple(wall.T)
            self.BCs[1].append(BounceBack(wall, self.grid_info, self.precision_policy))
            wall = tuple(idx.T)
            self.BCs[1].append(BounceBack(wall, self.grid_info, self.precision_policy, theta_a[wall]))


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
        "wetting_formulation": "improved_virtual_density",
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
        "wetting_formulation": "improved_virtual_density",
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
