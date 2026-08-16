"""Scene and material definitions for JAX-LaB in-situ rendering."""

import math
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .render_utils import (
    apply_edge_antialiasing,
    colormap_lookup_table,
    composite_layers,
    generate_camera_rays,
    render_surface,
    render_volume,
    smooth_scalar_field,
    write_image,
)


def _validate_vector(name, value, length=3, *, nonnegative=False):
    vector = tuple(float(component) for component in value)
    if len(vector) != length:
        raise ValueError(f"{name} must contain {length} values.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values.")
    if nonnegative and not np.all(np.asarray(vector) >= 0.0):
        raise ValueError(f"{name} cannot contain negative values.")
    return vector


def _validate_color(name, value):
    color = _validate_vector(name, value)
    if not np.all((np.asarray(color) >= 0.0) & (np.asarray(color) <= 1.0)):
        raise ValueError(f"{name} components must be in [0, 1].")
    return color


def _validate_range(value):
    value_range = _validate_vector("value_range", value, length=2)
    if value_range[0] > value_range[1]:
        raise ValueError("value_range must be ordered from minimum to maximum.")
    return value_range


class Light:
    """A positional light source.

    Parameters
    ----------
    position (tuple of float): Light location in scene coordinates.

    color (tuple of float): RGB light color in ``[0, 1]``.

    intensity (float): Nonnegative light power.
    """

    def __init__(self, position, color=(1.0, 1.0, 1.0), intensity=8.0):
        self.position = _validate_vector("position", position)
        self.color = _validate_color("color", color)
        if not math.isfinite(intensity) or intensity < 0.0:
            raise ValueError("Light intensity must be a finite nonnegative value.")
        self.intensity = intensity


class SurfaceRendering:
    """Surface material for values inside a closed interval.

    Parameters
    ----------
    value_range (tuple of float): Inclusive scalar interval defining the rendered material.

    color (tuple of float): RGB surface color in ``[0, 1]``.

    metallic (float): Metallic response in ``[0, 1]``.

    roughness (float): Surface roughness in ``[0, 1]``.

    opacity (float): Surface opacity in ``[0, 1]``.

    spacing (tuple of float): Positive voxel spacing in scene coordinates.

    origin (tuple of float): Field origin in scene coordinates.

    casts_shadows (bool): Whether the surface tests visibility to scene lights.

    color_field (str, optional): Named scalar or three-component vector field used to color the surface.

    color_range (tuple of float, optional): Inclusive scalar or vector-magnitude range mapped through ``colormap``.

    colormap (str or numpy.ndarray): Built-in matplotlib-compatible name or RGB control-point array.
    """

    def __init__(
        self,
        value_range,
        color,
        metallic,
        roughness,
        opacity=1.0,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        casts_shadows=True,
        color_field=None,
        color_range=None,
        colormap="viridis",
    ):
        self.value_range = _validate_range(value_range)
        self.color = _validate_color("color", color)
        self.metallic = metallic
        self.roughness = roughness
        self.opacity = opacity
        self.spacing = _validate_vector("spacing", spacing)
        self.origin = _validate_vector("origin", origin)
        self.casts_shadows = casts_shadows
        self.color_field = color_field
        self.color_range = color_range
        self.colormap = colormap
        if min(self.spacing) <= 0.0:
            raise ValueError("Voxel spacing must be positive.")
        for name in ("metallic", "roughness", "opacity"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.color_field is not None:
            if not isinstance(self.color_field, str) or not self.color_field:
                raise ValueError("color_field must be a nonempty field name.")
            if self.color_range is None:
                raise ValueError("color_range is required when color_field is specified.")
            self.color_range = _validate_range(self.color_range)
        elif self.color_range is not None:
            raise ValueError("color_field is required when color_range is specified.")
        self._lookup_table = colormap_lookup_table(self.colormap)


class VolumeRendering:
    """Refractive volume material for values inside a closed interval.

    Parameters
    ----------
    value_range (tuple of float): Inclusive scalar interval occupied by the volume.

    color (tuple of float): RGB volume color in ``[0, 1]``.

    opacity (float): Per-lattice-unit opacity in ``[0, 1]``.

    index_of_refraction (float): Positive refractive index of the material.

    spacing (tuple of float): Positive voxel spacing in scene coordinates.

    origin (tuple of float): Field origin in scene coordinates.

    color_field (str, optional): Named scalar or three-component vector field used to color the volume.

    color_range (tuple of float, optional): Inclusive scalar or vector-magnitude range mapped through ``colormap``.

    colormap (str or numpy.ndarray): Built-in matplotlib-compatible name or RGB control-point array.
    """

    def __init__(
        self,
        value_range,
        color,
        opacity,
        index_of_refraction,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        color_field=None,
        color_range=None,
        colormap="viridis",
    ):
        self.value_range = _validate_range(value_range)
        self.color = _validate_color("color", color)
        self.opacity = opacity
        self.index_of_refraction = index_of_refraction
        self.spacing = _validate_vector("spacing", spacing)
        self.origin = _validate_vector("origin", origin)
        self.color_field = color_field
        self.color_range = color_range
        self.colormap = colormap
        if min(self.spacing) <= 0.0:
            raise ValueError("Voxel spacing must be positive.")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be in [0, 1].")
        if not math.isfinite(self.index_of_refraction) or self.index_of_refraction <= 0.0:
            raise ValueError("index_of_refraction must be a finite positive value.")
        if self.color_field is not None:
            if not isinstance(self.color_field, str) or not self.color_field:
                raise ValueError("color_field must be a nonempty field name.")
            if self.color_range is None:
                raise ValueError("color_range is required when color_field is specified.")
            self.color_range = _validate_range(self.color_range)
        elif self.color_range is not None:
            raise ValueError("color_field is required when color_range is specified.")
        self._lookup_table = colormap_lookup_table(self.colormap)


class VectorRendering:
    """Colormapped magnitude rendering for a three-component vector field.

    Parameters
    ----------
    value_range (tuple of float): Magnitude interval mapped through the colormap.

    colormap (str or numpy.ndarray): Built-in matplotlib-compatible name or RGB control-point array.

    opacity (float): Per-lattice-unit opacity in ``[0, 1]``.

    spacing (tuple of float): Positive voxel spacing in scene coordinates.

    origin (tuple of float): Field origin in scene coordinates.
    """

    def __init__(
        self,
        value_range,
        colormap="viridis",
        opacity=0.08,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    ):
        self.value_range = _validate_range(value_range)
        self.colormap = colormap
        self.opacity = opacity
        self.spacing = _validate_vector("spacing", spacing)
        self.origin = _validate_vector("origin", origin)
        if min(self.spacing) <= 0.0:
            raise ValueError("Voxel spacing must be positive.")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be in [0, 1].")
        self._lookup_table = colormap_lookup_table(self.colormap)


class Scene:
    """Define and render a collection of named three-dimensional fields.

    Parameters
    ----------
    renderings (mapping of str to rendering configuration): Field names and their unique surface, volume, or vector
        configuration.

    resolution (tuple of int): Output ``(width, height)``. Both values are assumed positive.

    position (tuple of float): Camera location in scene coordinates.

    target (tuple of float): Camera focal point in scene coordinates.

    field_of_view (float): Vertical field of view in degrees, assumed to be in ``(0, 180)``.

    up (tuple of float): Camera up direction.

    lights (sequence of Light): Positional lights. At least one light is assumed.

    background_color (tuple of float): RGB background color in ``[0, 1]``.

    global_illumination (float): Nonnegative ambient-light intensity.

    global_illumination_color (tuple of float): RGB ambient-light color in ``[0, 1]``.

    shadows (bool): Enable shadow rays for surfaces that cast shadows.

    max_bounces (int): Maximum refraction and indirect-light bounce count.

    surface_smoothing (int): Number of on-device smoothing passes applied to scalar surfaces.

    samples_per_voxel (float): Ray samples per minimum voxel spacing, assumed to be at least one.

    anti_aliasing (bool): Apply a device-side FXAA-style edge filter to the composited image.

    anti_aliasing_strength (float): Edge smoothing strength in ``[0, 1]``.

    output_dir (str or pathlib.Path): Directory used for relative output filenames.

    rotate (bool): Rotate the camera around ``target`` using ``rotation_angle``.
    """

    def __init__(
        self,
        renderings,
        resolution=(1280, 720),
        position=(2.0, 2.0, -2.0),
        target=(0.0, 0.0, 0.0),
        field_of_view=45.0,
        up=(0.0, 1.0, 0.0),
        lights=(Light(position=(-2.0, 4.0, -3.0)),),
        background_color=(0.02, 0.02, 0.025),
        global_illumination=0.18,
        global_illumination_color=(1.0, 1.0, 1.0),
        shadows=True,
        max_bounces=1,
        surface_smoothing=1,
        samples_per_voxel=2.0,
        anti_aliasing=True,
        anti_aliasing_strength=0.9,
        output_dir="renders",
        rotate=False,
    ):
        if not renderings:
            raise ValueError("A scene must define at least one rendering.")
        if not all(isinstance(name, str) and name for name in renderings):
            raise ValueError("Rendering field names must be nonempty strings.")
        if not all(isinstance(item, (SurfaceRendering, VolumeRendering, VectorRendering)) for item in renderings.values()):
            raise TypeError("Every rendering must use a supported rendering configuration.")
        self.renderings = dict(renderings)
        if len(resolution) != 2 or not all(isinstance(value, int) and value > 0 for value in resolution):
            raise ValueError("resolution must contain two positive integers.")
        self.resolution = tuple(resolution)
        self.position = _validate_vector("position", position)
        self.target = _validate_vector("target", target)
        self.up = _validate_vector("up", up)
        if np.linalg.norm(np.asarray(self.position) - self.target) < 1.0e-8:
            raise ValueError("Camera position and target must differ.")
        if np.linalg.norm(self.up) < 1.0e-8:
            raise ValueError("Camera up vector cannot be zero.")
        if not 0.0 < field_of_view < 180.0:
            raise ValueError("field_of_view must be in (0, 180) degrees.")
        self.field_of_view = float(field_of_view)
        self.lights = tuple(lights)
        if not self.lights or not all(isinstance(light, Light) for light in self.lights):
            raise ValueError("lights must contain at least one Light.")
        self.background_color = _validate_color("background_color", background_color)
        self.global_illumination_color = _validate_color("global_illumination_color", global_illumination_color)
        if not math.isfinite(global_illumination) or global_illumination < 0.0:
            raise ValueError("global_illumination must be finite and nonnegative.")
        self.global_illumination = float(global_illumination)
        if not isinstance(shadows, bool):
            raise TypeError("shadows must be boolean.")
        self.shadows = shadows
        if not isinstance(max_bounces, int) or max_bounces < 0:
            raise ValueError("max_bounces must be a nonnegative integer.")
        if max_bounces > 4:
            warnings.warn(
                "Large bounce counts drastically increase rendering time and compilation size.",
                stacklevel=2,
            )
        self.max_bounces = max_bounces
        if not isinstance(surface_smoothing, int) or surface_smoothing < 0:
            raise ValueError("surface_smoothing must be a nonnegative integer.")
        self.surface_smoothing = surface_smoothing
        if not math.isfinite(samples_per_voxel) or samples_per_voxel < 1.0:
            raise ValueError("samples_per_voxel must be finite and at least one.")
        self.samples_per_voxel = float(samples_per_voxel)
        if not isinstance(anti_aliasing, bool):
            raise TypeError("anti_aliasing must be boolean.")
        self.anti_aliasing = anti_aliasing
        if not math.isfinite(anti_aliasing_strength) or not 0.0 <= anti_aliasing_strength <= 1.0:
            raise ValueError("anti_aliasing_strength must be finite and in [0, 1].")
        self.anti_aliasing_strength = float(anti_aliasing_strength)
        self.output_dir = Path(output_dir)
        if not isinstance(rotate, bool):
            raise TypeError("rotate must be boolean.")
        self.rotate = rotate

    def rotation_angle(self, timestep):
        """Return camera rotation in radians for a timestep.

        Parameters
        ----------
        timestep (int): Current simulation timestep.

        Returns
        -------
        float
            Rotation angle. Subclasses may override this method.
        """
        del timestep
        return 0.0

    def update_camera_position(self, timestep):
        """Return the camera position, including optional target rotation.

        Parameters
        ----------
        timestep (int): Current simulation timestep.

        Returns
        -------
        jax.Array
            Camera position with shape ``(3,)``.
        """
        position = jnp.asarray(self.position, dtype=jnp.float32)
        if not self.rotate:
            return position
        target = jnp.asarray(self.target, dtype=jnp.float32)
        axis = jnp.asarray(self.up, dtype=jnp.float32)
        axis /= jnp.linalg.norm(axis)
        offset = position - target
        angle = jnp.asarray(self.rotation_angle(timestep), dtype=jnp.float32)
        rotated = offset * jnp.cos(angle)
        rotated += jnp.cross(axis, offset) * jnp.sin(angle)
        rotated += axis * jnp.dot(axis, offset) * (1.0 - jnp.cos(angle))
        return target + rotated

    @staticmethod
    def _prepare_field(name, data, configuration):
        field = jnp.asarray(data)
        if isinstance(configuration, VectorRendering):
            if field.ndim != 4 or field.shape[-1] != 3:
                raise ValueError(f"Vector field {name!r} must have shape (nx, ny, nz, 3).")
            return jnp.linalg.norm(field, axis=-1).astype(jnp.float32)
        if field.ndim == 4 and field.shape[-1] == 1:
            field = field[..., 0]
        if field.ndim != 3:
            raise ValueError(f"Scalar field {name!r} must have shape (nx, ny, nz) or (nx, ny, nz, 1).")
        return field.astype(jnp.float32)

    def render(
        self,
        data,
        *,
        timestep=0,
        filename=None,
    ):
        """Render named fields and optionally save the final RGB image.

        Parameters
        ----------
        data (mapping of str to array): Scalar or vector fields matching all names in ``renderings``.

        timestep (int): Simulation timestep used by optional camera rotation.

        filename (str or pathlib.Path, optional): PNG filename. Relative paths are placed under ``output_dir``.

        Returns
        -------
        jax.Array
            RGB image with shape ``(height, width, 3)`` and values in ``[0, 1]``.
        """
        missing = self.renderings.keys() - data.keys()
        if missing:
            raise KeyError(f"Missing rendering data for: {', '.join(sorted(missing))}.")
        ray_origins, ray_directions = generate_camera_rays(
            self.resolution,
            self.update_camera_position(timestep),
            jnp.asarray(self.target, dtype=jnp.float32),
            jnp.asarray(self.up, dtype=jnp.float32),
            jnp.deg2rad(jnp.asarray(self.field_of_view, dtype=jnp.float32)),
        )
        light_positions = jnp.asarray([light.position for light in self.lights], dtype=jnp.float32)
        light_colors = jnp.asarray([light.color for light in self.lights], dtype=jnp.float32)
        light_intensities = jnp.asarray([light.intensity for light in self.lights], dtype=jnp.float32)
        background = jnp.asarray(self.background_color, dtype=jnp.float32)
        colors = []
        alphas = []
        depths = []

        for name, configuration in self.renderings.items():
            scalar_field = self._prepare_field(name, data[name], configuration)
            if min(scalar_field.shape) < 2:
                raise ValueError(f"Field {name!r} must have at least two voxels along each spatial axis.")
            spacing = jnp.asarray(configuration.spacing, dtype=jnp.float32)
            origin = jnp.asarray(configuration.origin, dtype=jnp.float32)
            value_range = jnp.asarray(configuration.value_range, dtype=jnp.float32)
            physical_diagonal = np.linalg.norm((np.asarray(scalar_field.shape) - 1) * np.asarray(configuration.spacing))
            minimum_spacing = min(configuration.spacing)
            num_steps = max(2, math.ceil(physical_diagonal / minimum_spacing * self.samples_per_voxel) + 1)

            if isinstance(configuration, SurfaceRendering):
                normal_field = scalar_field
                if self.surface_smoothing:
                    normal_field = smooth_scalar_field(scalar_field, iterations=self.surface_smoothing)
                uses_color_field = configuration.color_field is not None
                if uses_color_field:
                    if configuration.color_field not in data:
                        raise KeyError(f"Missing surface color data for: {configuration.color_field}.")
                    color_field = jnp.asarray(data[configuration.color_field])
                    if color_field.ndim == 4 and color_field.shape[-1] == 3:
                        color_field = jnp.linalg.norm(color_field, axis=-1)
                    elif color_field.ndim == 4 and color_field.shape[-1] == 1:
                        color_field = color_field[..., 0]
                    if color_field.ndim != 3 or color_field.shape != scalar_field.shape:
                        raise ValueError(f"Surface color field {configuration.color_field!r} must match field {name!r}.")
                    color_field = color_field.astype(jnp.float32)
                    color_value_range = jnp.asarray(configuration.color_range, dtype=jnp.float32)
                else:
                    color_field = scalar_field
                    color_value_range = value_range
                layer = render_surface(
                    scalar_field,
                    normal_field,
                    color_field,
                    ray_origins,
                    ray_directions,
                    origin,
                    spacing,
                    value_range,
                    color_value_range,
                    jnp.asarray(configuration.color, dtype=jnp.float32),
                    configuration._lookup_table if uses_color_field else jnp.zeros((2, 3), dtype=jnp.float32),
                    jnp.asarray(configuration.opacity, dtype=jnp.float32),
                    jnp.asarray(configuration.metallic, dtype=jnp.float32),
                    jnp.asarray(configuration.roughness, dtype=jnp.float32),
                    light_positions,
                    light_colors,
                    light_intensities,
                    jnp.asarray(self.global_illumination_color, dtype=jnp.float32),
                    jnp.asarray(self.global_illumination, dtype=jnp.float32),
                    background,
                    num_steps=num_steps,
                    shadow_steps=max(8, num_steps // 4),
                    max_bounces=self.max_bounces,
                    use_shadows=self.shadows and configuration.casts_shadows,
                    use_colormap=uses_color_field,
                )
            else:
                is_vector = isinstance(configuration, VectorRendering)
                uses_color_field = isinstance(configuration, VolumeRendering) and configuration.color_field is not None
                if uses_color_field:
                    if configuration.color_field not in data:
                        raise KeyError(f"Missing volume color data for: {configuration.color_field}.")
                    color_field = jnp.asarray(data[configuration.color_field])
                    if color_field.ndim == 4 and color_field.shape[-1] == 3:
                        color_field = jnp.linalg.norm(color_field, axis=-1)
                    elif color_field.ndim == 4 and color_field.shape[-1] == 1:
                        color_field = color_field[..., 0]
                    if color_field.ndim != 3 or color_field.shape != scalar_field.shape:
                        raise ValueError(f"Volume color field {configuration.color_field!r} must match field {name!r}.")
                    color_field = color_field.astype(jnp.float32)
                    color_value_range = jnp.asarray(configuration.color_range, dtype=jnp.float32)
                else:
                    color_field = scalar_field
                    color_value_range = value_range
                use_colormap = is_vector or uses_color_field
                lookup_table = configuration._lookup_table if use_colormap else jnp.zeros((2, 3), dtype=jnp.float32)
                layer = render_volume(
                    scalar_field,
                    color_field,
                    ray_origins,
                    ray_directions,
                    origin,
                    spacing,
                    value_range,
                    color_value_range,
                    jnp.zeros(3, dtype=jnp.float32) if is_vector else jnp.asarray(configuration.color, dtype=jnp.float32),
                    jnp.asarray(configuration.opacity, dtype=jnp.float32),
                    jnp.asarray(1.0 if is_vector else configuration.index_of_refraction, dtype=jnp.float32),
                    lookup_table,
                    jnp.asarray(minimum_spacing / self.samples_per_voxel, dtype=jnp.float32),
                    num_steps=num_steps,
                    max_bounces=self.max_bounces,
                    use_colormap=use_colormap,
                )
            color, alpha, depth = layer
            colors.append(color)
            alphas.append(alpha)
            depths.append(depth)

        image = composite_layers(
            jnp.stack(colors),
            jnp.stack(alphas),
            jnp.stack(depths),
            background,
        )
        if self.anti_aliasing:
            image = apply_edge_antialiasing(
                image,
                jnp.asarray(self.anti_aliasing_strength, dtype=jnp.float32),
            )
        if filename is not None:
            output_path = Path(filename)
            if not output_path.is_absolute():
                output_path = self.output_dir / output_path
            write_image(output_path, image)
        return image


def render(
    scene,
    data,
    *,
    timestep=0,
    filename=None,
):
    """Render data with a scene.

    Parameters
    ----------
    scene (Scene): Configured rendering scene.

    data (mapping of str to array): Named scalar or vector fields.

    timestep (int): Simulation timestep used by optional camera rotation.

    filename (str or pathlib.Path, optional): PNG output filename.

    Returns
    -------
    jax.Array
        Rendered RGB image.
    """
    if not isinstance(scene, Scene):
        raise TypeError("scene must be a Scene instance.")
    return scene.render(data, timestep=timestep, filename=filename)
