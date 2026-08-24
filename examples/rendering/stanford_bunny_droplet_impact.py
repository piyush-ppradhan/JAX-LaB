"""Render a Stanford-bunny liquid droplet falling onto a solid surface."""

from pathlib import Path
from functools import partial

import jax.numpy as jnp
import numpy as np
import trimesh
from jax import jit
from jax.tree_util import tree_map
from scipy import ndimage

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.eos import VanderWaals
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseMRT
from jax_lab.render import Light, Scene, SurfaceRendering, VolumeRendering
from jax_lab.render.render_utils import smooth_scalar_field

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def voxelize_filled_bunny(filename, target_extent):
    """Voxelize the Stanford bunny as a filled binary body.

    Parameters
    ----------
    filename: pathlib.Path
        Path to the OBJ geometry file.
    target_extent: int
        Longest voxelized dimension in lattice cells.

    Returns
    -------
    numpy.ndarray
        Filled boolean voxel array with axes ordered as x, y, z.

    Notes
    -----
    The supplied OBJ has open patches. Interior voxels are therefore selected as
    points bracketed by the surface along all three coordinate axes.
    """
    mesh = trimesh.load_mesh(filename, process=True)
    pitch = float(mesh.extents.max()) / target_extent
    surface = np.asarray(mesh.voxelized(pitch=pitch).matrix, dtype=bool)

    axis_fills = []
    for axis in range(3):
        line_has_surface = np.any(surface, axis=axis, keepdims=True)
        lower = np.expand_dims(np.argmax(surface, axis=axis), axis)
        upper = np.expand_dims(surface.shape[axis] - 1 - np.argmax(np.flip(surface, axis=axis), axis=axis), axis)
        coordinate_shape = [1, 1, 1]
        coordinate_shape[axis] = surface.shape[axis]
        coordinate = np.arange(surface.shape[axis]).reshape(coordinate_shape)
        axis_fills.append(line_has_surface & (coordinate >= lower) & (coordinate <= upper))

    return surface | (axis_fills[0] & axis_fills[1] & axis_fills[2])


class BunnyDropletImpactMRT(MultiphaseMRT):
    """Single-component bunny-shaped liquid droplet falling from rest."""

    def __init__(self, *, bunny_filename, **kwargs):
        self.bunny_mask = voxelize_filled_bunny(Path(bunny_filename), bunny_extent)
        super().__init__(**kwargs)

    def initialize_macroscopic_fields(self):
        padding = int(np.ceil(3.0 * interface_width))
        padded_bunny = np.pad(self.bunny_mask, padding)
        signed_distance = ndimage.distance_transform_edt(~padded_bunny) - ndimage.distance_transform_edt(padded_bunny)
        liquid_fraction = 0.5 * (1.0 - np.tanh(2.0 * signed_distance / interface_width))

        density = np.full((self.nx, self.ny, self.nz), vapor_density, dtype=np.float32)
        bunny_shape = np.asarray(self.bunny_mask.shape)
        bunny_origin = np.array([
            (self.nx - bunny_shape[0]) // 2,
            self.ny // 2,
            (self.nz - bunny_shape[2]) // 2,
        ])
        field_origin = bunny_origin - padding
        field_shape = np.asarray(liquid_fraction.shape)
        field_slices = tuple(slice(int(start), int(start + size)) for start, size in zip(field_origin, field_shape, strict=True))
        density[field_slices] = vapor_density + (liquid_density - vapor_density) * liquid_fraction

        density = self.distributed_array_init(
            (self.nx, self.ny, self.nz, 1),
            self.precision_policy.compute_dtype,
            init_val=density[..., None],
        )
        velocity = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype)
        return [density], [velocity]

    @partial(jit, static_argnums=(0,))
    def compute_force(self, rho_tree, T=None):
        """Apply gravity to liquid density above equilibrium vapor density.

        Parameters
        ----------
        rho_tree: list of jax.Array
            Component density fields.
        T: jax.Array, optional
            Temperature field supplied to a thermal equation of state.

        Returns
        -------
        list of jax.Array
            Component force fields.
        """
        force_tree = super().compute_force(rho_tree, T=T)
        if self.body_force is None:
            return force_tree
        return tree_map(
            lambda force, rho: force - self.body_force * jnp.minimum(rho, vapor_density),
            force_tree,
            rho_tree,
        )

    def set_boundary_conditions(self):
        outer_walls = np.unique(np.concatenate(tuple(self.bounding_box_indices.values())), axis=0)
        outer_wall_mask = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        outer_wall_mask[tuple(outer_walls.T)] = True
        impact_mask = np.zeros_like(outer_wall_mask)
        impact_mask[1:-1, 0, 1:-1] = True
        impact_indices = np.argwhere(impact_mask)
        plain_wall_indices = np.argwhere(outer_wall_mask & ~impact_mask)
        self.BCs[0].append(BounceBack(tuple(plain_wall_indices.T), self.grid_info, self.precision_policy))

        contact_angle = np.full((impact_indices.shape[0], 1), np.deg2rad(120.0), dtype=np.float32)
        phi = np.ones((impact_indices.shape[0], 1), dtype=np.float32)
        delta_rho = np.full((impact_indices.shape[0], 1), 0.1, dtype=np.float32)
        self.BCs[0].append(
            BounceBack(tuple(impact_indices.T), self.grid_info, self.precision_policy, theta=contact_angle, phi=phi, delta_rho=delta_rho)
        )

        solid_mask = outer_wall_mask
        self.fluid_mask = jnp.asarray(~solid_mask)
        self.render_fluid_mask = self.fluid_mask
        self.slab_field = jnp.ones((self.nx + 2, slab_thickness + 1, self.nz + 2), dtype=jnp.float32)

        output_spacing = float(self.downsampling_factor)
        self.render_scene = Scene(
            {
                "liquid": VolumeRendering(
                    value_range=(liquid_render_threshold, 2.0 * liquid_density),
                    color=(0.4196, 0.6667, 0.8196),
                    opacity=0.06,
                    index_of_refraction=1.0,
                    spacing=(output_spacing,) * 3,
                    origin=(render_clip * output_spacing,) * 3,
                ),
                "slab": SurfaceRendering(
                    value_range=(0.5, 1.0),
                    color=(0.7764, 0.7176, 0.6705),
                    metallic=0.0,
                    roughness=0.8,
                    opacity=1.0,
                    spacing=(output_spacing,) * 3,
                    origin=(-output_spacing, -slab_thickness * output_spacing, -output_spacing),
                ),
            },
            resolution=(1920, 1080),
            position=(128.0, 300.0, -430.0),
            target=(128.0, 135.0, 128.0),
            up=(0.0, 1.0, 0.0),
            lights=(
                Light(position=(-100.0, 440.0, -220.0), intensity=24000.0),
                Light(position=(390.0, 180.0, -120.0), color=(0.70, 0.82, 1.0), intensity=7500.0),
            ),
            background_color=(0.82, 0.92, 0.96),
            global_illumination=0.8,
            shadows=False,
            surface_smoothing=2,
            max_bounces=6,
            samples_per_voxel=2.0,
            anti_aliasing=True,
            anti_aliasing_strength=1.0,
            output_dir="./rendered_examples/stanford_bunny_droplet_impact",
        )

    def output_data(self, **kwargs):
        density = jnp.asarray(kwargs["rho_tree"][0][0, ..., 0], dtype=jnp.float32)
        velocity = jnp.asarray(kwargs["u_total"][0, ...], dtype=jnp.float32)
        timestep = kwargs["timestep"]
        if not bool(jnp.isfinite(density).all() & jnp.isfinite(velocity).all()):
            raise FloatingPointError(f"Non-finite fields detected at timestep {timestep}.")

        render_density = jnp.where(self.render_fluid_mask, density, vapor_density)
        render_density = render_density[render_clip:-render_clip, render_clip:-render_clip, render_clip:-render_clip]
        # render_density = smooth_scalar_field(render_density, iterations=2)
        render_fluid_mask = self.render_fluid_mask[render_clip:-render_clip, render_clip:-render_clip, render_clip:-render_clip]
        render_density = jnp.where(render_fluid_mask, render_density, vapor_density)
        fluid_speed = jnp.where(self.fluid_mask, jnp.linalg.norm(velocity, axis=-1), 0.0)
        rendered_voxels = jnp.count_nonzero(render_density >= liquid_render_threshold)
        logger.info(
            f"timestep={timestep}, density=[{float(jnp.min(density)):.4f}, "
            f"{float(jnp.max(density)):.4f}], max_fluid_speed={float(jnp.max(fluid_speed)):.5f}, "
            f"rendered_voxels={int(rendered_voxels)}"
        )
        self.render_scene.render(
            {"liquid": render_density, "slab": self.slab_field},
            timestep=timestep,
            filename=f"stanford_bunny_droplet_impact_{timestep:05d}.png",
        )


if __name__ == "__main__":
    liquid_density = 6.764470400
    vapor_density = 0.838834226
    interface_width = 4.0
    liquid_render_threshold = 4.6
    render_clip = 2
    bunny_extent = 96
    slab_thickness = 6

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
    bunny_filename = Path(__file__).resolve().parents[2] / "assets" / "stanford-bunny.obj"

    sim = BunnyDropletImpactMRT(
        bunny_filename=bunny_filename,
        n_components=1,
        lattice=lattice,
        nx=256,
        ny=256,
        nz=256,
        body_force=[0.0, -3.0e-4, 0.0],
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
        print_info_rate=10,
        checkpoint_rate=-1,
        restore_checkpoint=False,
    )
    sim.run(1400)
