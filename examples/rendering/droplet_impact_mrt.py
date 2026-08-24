"""Simulate and render a liquid droplet falling through its vapor onto a wall.

The example combines a single-component van der Waals liquid-vapor model, MRT collision, full-way bounce-back walls, and JAX-native surface rendering. Density
selects the liquid interface while velocity magnitude supplies its color.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.eos import VanderWaals
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseMRT
from jax_lab.render import Light, Scene, SurfaceRendering

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


class DropletImpactMRT(MultiphaseMRT):
    """
    Single-component liquid-vapor MRT simulation with in-situ rendering.
    """

    def __init__(
        self,
        *,
        droplet_radius,
        interface_width,
        droplet_center_height,
        initial_impact_velocity,
        liquid_density,
        vapor_density,
        render_output_dir,
        **kwargs,
    ):
        self.droplet_radius = droplet_radius
        self.interface_width = interface_width
        self.droplet_center_height = droplet_center_height
        self.initial_impact_velocity = initial_impact_velocity
        self.liquid_density = liquid_density
        self.vapor_density = vapor_density
        self.render_output_dir = Path(render_output_dir)
        super().__init__(**kwargs)

    def initialize_macroscopic_fields(self):
        x = jnp.arange(self.nx, dtype=self.precision_policy.compute_dtype)[:, None, None]
        y = jnp.arange(self.ny, dtype=self.precision_policy.compute_dtype)[None, :, None]
        z = jnp.arange(self.nz, dtype=self.precision_policy.compute_dtype)[None, None, :]
        distance = jnp.sqrt((x - 0.5 * (self.nx - 1)) ** 2 + (y - self.droplet_center_height) ** 2 + (z - 0.5 * (self.nz - 1)) ** 2)
        liquid_fraction = 0.5 * (1.0 - jnp.tanh(2.0 * (distance - self.droplet_radius) / self.interface_width))

        density = self.vapor_density + (self.liquid_density - self.vapor_density) * liquid_fraction
        density = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=density[..., None])

        velocity = jnp.zeros((self.nx, self.ny, self.nz, 3), dtype=self.precision_policy.compute_dtype)
        velocity = velocity.at[..., 1].set(self.initial_impact_velocity * liquid_fraction)
        velocity = self.distributed_array_init(velocity.shape, self.precision_policy.compute_dtype, init_val=velocity)
        return [density], [velocity]

    def set_boundary_conditions(self):
        # In the 3D face convention, front/back are y-min/y-max.
        floor_indices = self.bounding_box_indices["front"]
        contact_angle = np.full((floor_indices.shape[0], 1), np.deg2rad(30.0))
        phi = np.full((floor_indices.shape[0], 1), 1.2)
        delta_rho = np.zeros((floor_indices.shape[0], 1))
        self.BCs[0].append(
            BounceBack(tuple(floor_indices.T), self.grid_info, self.precision_policy, theta=contact_angle, phi=phi, delta_rho=delta_rho)
        )

        ceiling_indices = self.bounding_box_indices["back"]
        self.BCs[0].append(BounceBack(tuple(ceiling_indices.T), self.grid_info, self.precision_policy))

        output_spacing = float(self.downsampling_factor)

        # This doesn't have to be defined here, but it is convenient.
        self.render_scene = Scene(
            {
                "liquid_density": SurfaceRendering(
                    value_range=(0.5 * (self.liquid_density + self.vapor_density), 1.05 * self.liquid_density),
                    color=(0.2, 0.45, 0.95),
                    metallic=0.05,
                    roughness=0.28,
                    spacing=(output_spacing,) * 3,
                    color_field="velocity",
                    color_range=(0.0, 0.04),
                    colormap="jet",
                ),
                "floor": SurfaceRendering(
                    value_range=(0.5, 1.0), color=(0.48, 0.5, 0.54), metallic=0.0, roughness=0.4, spacing=(output_spacing,) * 3
                ),
            },
            resolution=(1920, 1080),
            position=(182.0, 112.0, -160.0),
            target=(64.0, 52.0, 64.0),
            up=(0.0, 1.0, 0.0),
            lights=(
                Light(position=(-50.0, 192.0, -115.0), intensity=50000.0),  # 80000
                Light(position=(192.0, 58.0, -50.0), color=(0.65, 0.78, 1.0), intensity=15000.0),  # 35000
            ),
            background_color=(1.0, 1.0, 1.0),
            global_illumination=0.3,
            shadows=False,
            surface_smoothing=2,
            samples_per_voxel=2.0,
            anti_aliasing=True,
            anti_aliasing_strength=1.0,
            output_dir=self.render_output_dir,
        )

    def output_data(self, **kwargs):
        density = jnp.asarray(kwargs["rho_tree"][0][0, ..., 0], dtype=jnp.float32)
        velocity = jnp.asarray(kwargs["u_total"][0, ...], dtype=jnp.float32)
        floor = jnp.zeros_like(density).at[:, 0, :].set(1.0)
        timestep = kwargs["timestep"]
        maximum_speed = jnp.max(jnp.linalg.norm(velocity, axis=-1))
        if not bool(jnp.isfinite(density).all() & jnp.isfinite(velocity).all()):
            raise FloatingPointError(f"Non-finite fields detected at timestep {timestep}.")
        logger.info(
            f"timestep={timestep}, density=[{float(jnp.min(density)):.4f}, {float(jnp.max(density)):.4f}], max_speed={float(maximum_speed):.5f}"
        )
        self.render_scene.render(
            {"liquid_density": density, "velocity": velocity, "floor": floor},
            timestep=timestep,
            filename=f"droplet_impact_mrt_{timestep:05d}.png",
        )


if __name__ == "__main__":
    precision = "f32/f32"

    lattice = LatticeD3Q19(precision)

    e = np.asarray(lattice.c.T)
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((19, 19))
    M[0, :] = en**0
    M[1, :] = 19.0 * en**2 - 30.0
    M[2, :] = 0.5 * (21.0 * en**4 - 53.0 * en**2 + 24.0)
    M[3, :] = e[:, 0]
    M[4, :] = (5.0 * en**2 - 9.0) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (5.0 * en**2 - 9.0) * e[:, 1]
    M[7, :] = e[:, 2]
    M[8, :] = (5.0 * en**2 - 9.0) * e[:, 2]
    M[9, :] = 3.0 * e[:, 0] ** 2 - en**2
    M[10, :] = (3.0 * en**2 - 5.0) * (3.0 * e[:, 0] ** 2 - en**2)
    M[11, :] = e[:, 1] ** 2 - e[:, 2] ** 2
    M[12, :] = (3.0 * en**2 - 5.0) * (e[:, 1] ** 2 - e[:, 2] ** 2)
    M[13, :] = e[:, 0] * e[:, 1]
    M[14, :] = e[:, 1] * e[:, 2]
    M[15, :] = e[:, 0] * e[:, 2]
    M[16, :] = (e[:, 1] ** 2 - e[:, 2] ** 2) * e[:, 0]
    M[17, :] = (e[:, 2] ** 2 - e[:, 0] ** 2) * e[:, 1]
    M[18, :] = (e[:, 0] ** 2 - e[:, 1] ** 2) * e[:, 2]

    eos = VanderWaals(a=9.0 / 49.0, b=2.0 / 21.0, R=1.0, T=0.8 * 0.5714285714)

    sim = DropletImpactMRT(
        droplet_radius=20.0,
        interface_width=4.0,
        droplet_center_height=93.0,
        initial_impact_velocity=-0.03,
        liquid_density=6.764470400,
        vapor_density=0.838834226,
        render_output_dir="rendered_examples/droplet_impact_mrt",
        n_components=1,
        lattice=lattice,
        nx=128,
        ny=128,
        nz=128,
        body_force=[0.0, -2.2e-5, 0.0],
        g_kkprime=-np.ones((1, 1)),
        EOS=eos,
        precision=precision,
        k=[0.27],
        A=0.01 * np.ones((1, 1)),
        s_rho=[0.0],
        s_e=[1.0],
        s_eta=[1.0],
        s_j=[0.0],
        s_q=[1.0],
        s_pi=[1.0],
        s_m=[1.0],
        s_v=[1.0],
        M=[M],
        kappa=[0.0],
        wetting_formulation="improved_virtual_density",
        io_rate=500,
        downsampling_factor=1,
        compute_MLUPS=False,
        print_info_rate=500,
        checkpoint_rate=-1,
        restore_checkpoint=False,
    )
    sim.run(5000)
