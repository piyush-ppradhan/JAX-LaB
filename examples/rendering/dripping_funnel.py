"""
Simulate and render a liquid drop growing at the tip of a tube-and-funnel nozzle and falling under gravity.

A single-component Peng-Robinson liquid-vapor mixture fills a vertical tube that narrows through a conical funnel to a small spout. MRT collision, full-way bounce-back nozzle
and ceiling walls, a non-equilibrium vapor outlet, and JAX-native surface rendering are combined so a pendant drop grows at the spout, necks, snaps off, and falls through the domain.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.tree import map as tree_map
from scipy.ndimage import gaussian_filter

from jax_lab.core.boundary_conditions import BounceBack, ExactNonEquilibriumExtrapolation
from jax_lab.core.eos import PengRobinson
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseMRT
from jax_lab.render import Light, Scene, SurfaceRendering

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def channel_radii(z, funnel_tip_z, tube_bottom_z, tube_inner_radius, tube_outer_radius, funnel_tip_inner_radius, funnel_tip_outer_radius):
    """
    Inner/outer nozzle radius at height(s) z: constant through the straight tube, linearly tapered through
    the conical funnel below it. Values below the funnel tip are not meaningful (no wall there) but are still
    returned (clipped to the tip radius) so the same array can be masked by the caller.

    Input
    -----
    z (numpy.ndarray): Height coordinate(s), any shape.

    Output
    ------
    inner_radius, outer_radius (numpy.ndarray): Same shape as z.
    """
    taper = np.clip((z - funnel_tip_z) / (tube_bottom_z - funnel_tip_z), 0.0, 1.0)
    inner_radius = funnel_tip_inner_radius + taper * (tube_inner_radius - funnel_tip_inner_radius)
    outer_radius = funnel_tip_outer_radius + taper * (tube_outer_radius - funnel_tip_outer_radius)
    return inner_radius, outer_radius


class DrippingFunnelMRT(MultiphaseMRT):
    """
    Single-component liquid-vapor MRT simulation of a tube-and-funnel nozzle dripping under gravity.
    """

    def __init__(
        self,
        *,
        tube_inner_radius,
        tube_outer_radius,
        funnel_tip_inner_radius,
        funnel_tip_outer_radius,
        tube_bottom_z,
        funnel_tip_z,
        interface_width,
        liquid_density,
        vapor_density,
        render_output_dir,
        **kwargs,
    ):
        self.tube_inner_radius = tube_inner_radius
        self.tube_outer_radius = tube_outer_radius
        self.funnel_tip_inner_radius = funnel_tip_inner_radius
        self.funnel_tip_outer_radius = funnel_tip_outer_radius
        self.tube_bottom_z = tube_bottom_z
        self.funnel_tip_z = funnel_tip_z
        self.interface_width = interface_width
        self.liquid_density = liquid_density
        self.vapor_density = vapor_density
        self.render_output_dir = Path(render_output_dir)
        super().__init__(**kwargs)

    def _radial_and_z_grid(self):
        cx = 0.5 * (self.nx - 1)
        cy = 0.5 * (self.ny - 1)
        x = np.arange(self.nx)[:, None, None]
        y = np.arange(self.ny)[None, :, None]
        z = np.arange(self.nz)[None, None, :]
        radial_distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return np.broadcast_to(radial_distance, (self.nx, self.ny, self.nz)), np.broadcast_to(z, (self.nx, self.ny, self.nz))

    def initialize_macroscopic_fields(self):
        radial_distance, z = self._radial_and_z_grid()
        inner_radius, _ = channel_radii(
            z,
            self.funnel_tip_z,
            self.tube_bottom_z,
            self.tube_inner_radius,
            self.tube_outer_radius,
            self.funnel_tip_inner_radius,
            self.funnel_tip_outer_radius,
        )
        radial_liquid_fraction = 0.5 * (1.0 - np.tanh(2.0 * (radial_distance - inner_radius) / self.interface_width))
        axial_liquid_fraction = 0.5 * (1.0 + np.tanh(2.0 * (z - self.funnel_tip_z) / self.interface_width))
        upper_interface_z = self.nz - 2.0 * self.interface_width
        upper_liquid_fraction = 0.5 * (1.0 - np.tanh(2.0 * (z - upper_interface_z) / self.interface_width))
        liquid_fraction = radial_liquid_fraction * axial_liquid_fraction * upper_liquid_fraction
        density = self.vapor_density + (self.liquid_density - self.vapor_density) * liquid_fraction
        density = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=density[..., None])

        velocity = np.zeros((self.nx, self.ny, self.nz, 3))
        velocity = self.distributed_array_init(velocity.shape, self.precision_policy.compute_dtype, init_val=velocity)
        return [density], [velocity]

    def _stream_nonperiodic_z(self, field):
        """Stream a scalar stencil without wrapping the nonperiodic z boundaries."""
        field_repeated = jnp.repeat(field, axis=-1, repeats=self.lattice.q)
        streamed = self.streaming(field_repeated)
        directions = np.asarray(self.lattice.c).T
        for direction, (_, _, cz) in enumerate(directions):
            if cz > 0:
                streamed = streamed.at[:, :, :cz, direction].set(field_repeated[:, :, :1, direction])
            elif cz < 0:
                streamed = streamed.at[:, :, cz:, direction].set(field_repeated[:, :, -1:, direction])
        return streamed

    def compute_average_density(self, rho_tree):
        """Average neighboring fluid density without dividing by zero inside thick solid walls."""
        rho_streamed_tree = tree_map(self._stream_nonperiodic_z, rho_tree)

        def average_fluid_neighbors(rho_streamed, solid_mask):
            fluid_weights = self.G_ff * (1 - solid_mask)
            numerator = jnp.sum(fluid_weights * rho_streamed, axis=-1, keepdims=True)
            denominator = jnp.sum(fluid_weights, axis=-1, keepdims=True)
            return numerator / jnp.where(denominator > 0, denominator, 1.0)

        return tree_map(average_fluid_neighbors, rho_streamed_tree, self.solid_mask_streamed)

    def compute_fluid_fluid_force(self, psi_tree, potential_tree):
        """Compute the one-component interaction force using a nonperiodic z stencil."""
        psi = psi_tree[0]
        potential = potential_tree[0]
        psi_streamed = self._stream_nonperiodic_z(psi)
        potential_streamed = self._stream_nonperiodic_z(potential)
        directions = jnp.asarray(self.c, dtype=self.precision_policy.compute_dtype).T
        force_weight = self.A[0, 0]
        coupling = self.g_kkprime[0, 0]
        shan_chen_force = jnp.dot((1.0 - force_weight) * coupling * self.G_ff * psi_streamed, directions)
        zhang_chen_force = force_weight * jnp.dot(self.G_ff * potential_streamed, directions)
        return [psi * shan_chen_force + zhang_chen_force]

    def set_boundary_conditions(self):
        radial_distance, z = self._radial_and_z_grid()
        inner_radius, outer_radius = channel_radii(
            z,
            self.funnel_tip_z,
            self.tube_bottom_z,
            self.tube_inner_radius,
            self.tube_outer_radius,
            self.funnel_tip_inner_radius,
            self.funnel_tip_outer_radius,
        )
        nozzle_wall = (z >= self.funnel_tip_z) & (radial_distance >= inner_radius) & (radial_distance <= outer_radius)
        self.nozzle_wall_field = np.minimum(
            np.minimum(radial_distance - inner_radius, outer_radius - radial_distance),
            z - self.funnel_tip_z,
        ).astype(np.float32)

        nozzle_indices = np.array(np.where(nozzle_wall), dtype=int)
        contact_angle = np.full((nozzle_indices.shape[1], 1), np.deg2rad(45.0))
        phi = np.full((nozzle_indices.shape[1], 1), 1.2)
        delta_rho = np.zeros((nozzle_indices.shape[1], 1))
        self.BCs[0].append(
            BounceBack(tuple(nozzle_indices), self.grid_info, self.precision_policy, theta=contact_angle, phi=phi, delta_rho=delta_rho)
        )

        floor_indices = self.bounding_box_indices["bottom"]
        outlet_density = np.full((floor_indices.shape[0], 1), self.vapor_density)
        self.BCs[0].append(
            ExactNonEquilibriumExtrapolation(
                tuple(floor_indices.T), self.grid_info, self.precision_policy, prescribed=outlet_density, bc_type="density"
            )
        )
        ceiling_indices = self.bounding_box_indices["top"]
        self.BCs[0].append(BounceBack(tuple(ceiling_indices.T), self.grid_info, self.precision_policy))

        output_spacing = float(self.downsampling_factor)
        self.render_scene = Scene(
            {
                "liquid_density": SurfaceRendering(
                    value_range=(0.5 * (self.liquid_density + self.vapor_density), 1.05 * self.liquid_density),
                    color=(0.358, 0.544, 0.968),
                    metallic=0.05,
                    roughness=0.25,
                    opacity=0.4,
                    spacing=(output_spacing,) * 3,
                ),
                "tube": SurfaceRendering(
                    value_range=(0.0, 2.0), color=(0.71, 0.71, 0.71), metallic=0.0, roughness=0.0, spacing=(output_spacing,) * 3
                ),
            },
            resolution=(3840, 2160),
            position=(113.0, -30.0, 194.0),
            target=(64.0, 64.0, 165.0),
            up=(0.0, 0.0, 1.0),
            lights=(
                Light(position=(-40.0, 220.0, 230.0), intensity=50000.0),
                Light(position=(220.0, -40.0, 60.0), color=(0.65, 0.78, 1.0), intensity=15000.0),
                Light(position=(113.0, -30.0, 194.0), intensity=2500.0),
            ),
            background_color=(1.0, 1.0, 1.0),
            global_illumination=0.7,
            shadows=False,
            surface_smoothing=4,
            samples_per_voxel=3.0,
            anti_aliasing=True,
            anti_aliasing_strength=1.0,
            output_dir=self.render_output_dir,
        )

    def output_data(self, **kwargs):
        density = np.asarray(kwargs["rho_tree"][0][0, ..., 0], dtype=np.float32)
        velocity = np.asarray(kwargs["u_total"][0, ...], dtype=np.float32)
        timestep = kwargs["timestep"]
        maximum_speed = float(np.max(np.linalg.norm(velocity, axis=-1)))
        if not (np.isfinite(density).all() and np.isfinite(velocity).all()):
            raise FloatingPointError(f"Non-finite fields detected at timestep {timestep}.")
        logger.info(f"timestep={timestep}, density=[{density.min():.4f}, {density.max():.4f}], max_speed={maximum_speed:.5f}")
        render_density = gaussian_filter(density, sigma=0.75, mode="nearest")
        if timestep >= 400:
            self.render_scene.render(
                {"liquid_density": render_density, "tube": self.nozzle_wall_field},
                timestep=timestep,
                filename=f"dripping_funnel_mrt_{timestep:05d}.png",
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

    critical_temperature = 0.1093785558
    eos = PengRobinson(a=3.0 / 49.0, b=2.0 / 21.0, pr_omega=0.344, R=1.0, T=0.86 * critical_temperature)

    sim = DrippingFunnelMRT(
        # Straight tube: 40 lattice units tall, occupying the domain ceiling down to z=216.
        tube_inner_radius=14.0,
        tube_outer_radius=17.0,
        tube_bottom_z=216,
        # Conical funnel: 40 lattice units tall, narrowing to a 4-unit-radius spout at z=176.
        funnel_tip_inner_radius=4.0,
        funnel_tip_outer_radius=7.0,
        funnel_tip_z=176,
        interface_width=6.0,
        liquid_density=6.499210784,
        vapor_density=0.379598891,
        render_output_dir="rendered_examples/dripping_funnel_mrt",
        n_components=1,
        lattice=lattice,
        nx=128,
        ny=128,
        nz=256,
        # Gravity magnitude tuned so the pendant drop necks and snaps off well within 20000 steps for this
        # geometry; raise it further if a run drains too slowly, lower it if the drop detaches too early/fast.
        body_force=[0.0, 0.0, -2.0e-5],
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
        io_rate=10,
        downsampling_factor=1,
        compute_MLUPS=False,
        print_info_rate=50,
        checkpoint_rate=-1,
        restore_checkpoint=False,
    )
    sim.run(2350)
