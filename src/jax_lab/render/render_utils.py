"""Low-level JAX ray marching and image output utilities."""

import struct
import zlib
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


_COLORMAP_ANCHORS = {
    "viridis": np.array(
        [
            [0.267, 0.005, 0.329],
            [0.283, 0.141, 0.458],
            [0.254, 0.265, 0.530],
            [0.207, 0.372, 0.553],
            [0.164, 0.471, 0.558],
            [0.128, 0.567, 0.551],
            [0.267, 0.749, 0.441],
            [0.741, 0.873, 0.150],
        ],
        dtype=np.float32,
    ),
    "plasma": np.array(
        [
            [0.050, 0.030, 0.528],
            [0.325, 0.007, 0.640],
            [0.546, 0.039, 0.647],
            [0.723, 0.196, 0.539],
            [0.859, 0.360, 0.407],
            [0.955, 0.533, 0.285],
            [0.994, 0.741, 0.166],
            [0.940, 0.975, 0.131],
        ],
        dtype=np.float32,
    ),
    "inferno": np.array(
        [
            [0.001, 0.000, 0.014],
            [0.123, 0.047, 0.282],
            [0.329, 0.064, 0.435],
            [0.553, 0.161, 0.506],
            [0.765, 0.265, 0.408],
            [0.930, 0.411, 0.216],
            [0.988, 0.683, 0.073],
            [0.988, 0.998, 0.645],
        ],
        dtype=np.float32,
    ),
    "magma": np.array(
        [
            [0.001, 0.000, 0.014],
            [0.114, 0.066, 0.276],
            [0.317, 0.072, 0.485],
            [0.530, 0.154, 0.507],
            [0.716, 0.276, 0.470],
            [0.869, 0.437, 0.376],
            [0.967, 0.681, 0.562],
            [0.987, 0.991, 0.750],
        ],
        dtype=np.float32,
    ),
    "jet": np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 1.0, 0.5],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    ),
    "grey": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
    "gray": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
}


def colormap_lookup_table(colormap, size=256):
    """Create an RGB lookup table without importing matplotlib.

    Parameters
    ----------
    colormap: str or numpy.ndarray
        Built-in matplotlib-compatible name or an RGB array of shape ``(n, 3)``.
    size: int
        Number of lookup entries. Assumed to be at least two.

    Returns
    -------
    jax.Array
        Lookup table with shape ``(size, 3)`` and float32 dtype.
    """
    if isinstance(colormap, str):
        try:
            anchors = _COLORMAP_ANCHORS[colormap.lower()]
        except KeyError as error:
            supported = ", ".join(sorted(_COLORMAP_ANCHORS))
            raise ValueError(f"Unknown colormap {colormap!r}. Supported names: {supported}.") from error
    else:
        anchors = np.asarray(colormap, dtype=np.float32)
        if anchors.ndim != 2 or anchors.shape[0] < 2 or anchors.shape[1] != 3:
            raise ValueError("A colormap array must have shape (n, 3), with n >= 2.")
    coordinates = np.linspace(0.0, 1.0, size, dtype=np.float32)
    anchor_coordinates = np.linspace(0.0, 1.0, len(anchors), dtype=np.float32)
    table = np.stack(
        [np.interp(coordinates, anchor_coordinates, anchors[:, channel]) for channel in range(3)],
        axis=-1,
    )
    return jnp.asarray(np.clip(table, 0.0, 1.0), dtype=jnp.float32)


def _png_chunk(chunk_type, data):
    payload = chunk_type + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))


def write_image(filename, image_data):
    """Save an RGB image using the standard-library PNG encoder.

    Parameters
    ----------
    filename: str or pathlib.Path
        Output filename. The file is assumed to use PNG format.
    image_data: jax.Array or numpy.ndarray
        RGB data of shape ``(height, width, 3)`` in ``[0, 1]`` or uint8.

    Returns
    -------
    pathlib.Path
        Path of the saved PNG file.
    """
    output_path = Path(filename)
    if output_path.suffix.lower() != ".png":
        raise ValueError("Rendering output must use the .png extension.")
    image = np.asarray(jax.device_get(image_data))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Image data must have shape (height, width, 3).")
    if image.dtype != np.uint8:
        image = np.asarray(np.clip(image, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)
    height, width, _ = image.shape
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", header)
    png += _png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
    png += _png_chunk(b"IEND", b"")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)
    return output_path


def _normalize(vector):
    return vector / jnp.maximum(jnp.linalg.norm(vector, axis=-1, keepdims=True), 1.0e-8)


@partial(jax.jit, static_argnames=("resolution",))
def generate_camera_rays(
    resolution,
    position,
    target,
    up,
    field_of_view,
):
    """Generate perspective rays for a fixed-size image grid."""
    width, height = resolution
    forward = _normalize(target - position)
    right = _normalize(jnp.cross(forward, up))
    camera_up = _normalize(jnp.cross(right, forward))
    aspect_ratio = width / height
    half_height = jnp.tan(0.5 * field_of_view)
    horizontal = ((jnp.arange(width) + 0.5) / width * 2.0 - 1.0) * aspect_ratio * half_height
    vertical = (1.0 - (jnp.arange(height) + 0.5) / height * 2.0) * half_height
    horizontal_grid, vertical_grid = jnp.meshgrid(horizontal, vertical)
    directions = _normalize(forward + horizontal_grid[..., None] * right + vertical_grid[..., None] * camera_up)
    origins = jnp.broadcast_to(position, directions.shape)
    return origins, directions


def _sample_scalar(field, positions, origin, spacing):
    coordinates = (positions - origin) / spacing
    maximum = jnp.asarray(field.shape, dtype=coordinates.dtype) - 1.0
    coordinates = jnp.clip(coordinates, 0.0, maximum)
    lower = jnp.floor(coordinates).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, jnp.asarray(field.shape, dtype=jnp.int32) - 1)
    fraction = coordinates - lower
    x0, y0, z0 = lower[..., 0], lower[..., 1], lower[..., 2]
    x1, y1, z1 = upper[..., 0], upper[..., 1], upper[..., 2]
    fx, fy, fz = fraction[..., 0], fraction[..., 1], fraction[..., 2]
    c00 = field[x0, y0, z0] * (1.0 - fx) + field[x1, y0, z0] * fx
    c01 = field[x0, y0, z1] * (1.0 - fx) + field[x1, y0, z1] * fx
    c10 = field[x0, y1, z0] * (1.0 - fx) + field[x1, y1, z0] * fx
    c11 = field[x0, y1, z1] * (1.0 - fx) + field[x1, y1, z1] * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def _sample_gradient(field, positions, origin, spacing):
    offsets = jnp.eye(3, dtype=positions.dtype) * spacing
    derivatives = [
        (_sample_scalar(field, positions + offsets[axis], origin, spacing) - _sample_scalar(field, positions - offsets[axis], origin, spacing))
        / (2.0 * spacing[axis])
        for axis in range(3)
    ]
    return _normalize(jnp.stack(derivatives, axis=-1))


def _inside_box(positions, lower, upper):
    return jnp.all((positions >= lower) & (positions <= upper), axis=-1)


def _box_intersection(
    ray_origins,
    ray_directions,
    lower,
    upper,
):
    safe_directions = jnp.where(
        jnp.abs(ray_directions) < 1.0e-8,
        jnp.where(ray_directions < 0.0, -1.0e-8, 1.0e-8),
        ray_directions,
    )
    inverse_directions = 1.0 / safe_directions
    near_axes = (lower - ray_origins) * inverse_directions
    far_axes = (upper - ray_origins) * inverse_directions
    near = jnp.maximum(jnp.max(jnp.minimum(near_axes, far_axes), axis=-1), 0.0)
    far = jnp.min(jnp.maximum(near_axes, far_axes), axis=-1)
    return near, far, far >= near


def _trace_surface(
    field,
    ray_origins,
    ray_directions,
    origin,
    spacing,
    value_range,
    num_steps,
):
    upper = origin + spacing * (jnp.asarray(field.shape, dtype=spacing.dtype) - 1.0)
    near, far, intersects = _box_intersection(ray_origins, ray_directions, origin, upper)
    initial_state = (
        jnp.zeros(intersects.shape, dtype=bool),
        jnp.zeros(ray_origins.shape, dtype=ray_origins.dtype),
        jnp.full(intersects.shape, jnp.inf, dtype=ray_origins.dtype),
        jnp.zeros(intersects.shape, dtype=bool),
    )

    def sample_step(index, state):
        hit, hit_positions, hit_depth, previous_inside = state
        fraction = index / jnp.maximum(num_steps - 1, 1)
        depth = near + fraction * (far - near)
        position = ray_origins + depth[..., None] * ray_directions
        value = _sample_scalar(field, position, origin, spacing)
        inside = intersects & (value >= value_range[0]) & (value <= value_range[1])
        new_hit = (~hit) & inside & (~previous_inside)
        return (
            hit | new_hit,
            jnp.where(new_hit[..., None], position, hit_positions),
            jnp.where(new_hit, depth, hit_depth),
            inside,
        )

    hit, positions, depth, _ = jax.lax.fori_loop(0, num_steps, sample_step, initial_state)
    return hit, positions, depth


def _shadow_visibility(
    field,
    positions,
    normals,
    light_positions,
    origin,
    spacing,
    value_range,
    shadow_steps,
):
    minimum_spacing = jnp.min(spacing)

    def one_light(light_position):
        start = positions + normals * (1.5 * minimum_spacing)
        displacement = light_position - start
        distance = jnp.linalg.norm(displacement, axis=-1)
        direction = displacement / jnp.maximum(distance[..., None], 1.0e-8)
        upper = origin + spacing * (jnp.asarray(field.shape, dtype=spacing.dtype) - 1.0)

        def shadow_step(index, occluded):
            fraction = (index + 1) / (shadow_steps + 1)
            sample_position = start + fraction[..., None] * displacement
            value = _sample_scalar(field, sample_position, origin, spacing)
            blocked = _inside_box(sample_position, origin, upper) & (value >= value_range[0]) & (value <= value_range[1])
            return occluded | blocked

        occluded = jax.lax.fori_loop(
            0,
            shadow_steps,
            shadow_step,
            jnp.zeros(distance.shape, dtype=bool),
        )
        return (~occluded).astype(jnp.float32)

    return jax.vmap(one_light)(light_positions)


@partial(jax.jit, static_argnames=("num_steps", "shadow_steps", "max_bounces", "use_shadows", "use_colormap"))
def render_surface(
    field,
    normal_field,
    color_field,
    ray_origins,
    ray_directions,
    origin,
    spacing,
    value_range,
    color_value_range,
    color,
    colormap,
    opacity,
    metallic,
    roughness,
    light_positions,
    light_colors,
    light_intensities,
    ambient_color,
    ambient_intensity,
    background_color,
    *,
    num_steps,
    shadow_steps,
    max_bounces,
    use_shadows,
    use_colormap,
):
    """Ray trace one scalar-field surface on the active JAX device."""
    hit, positions, depth = _trace_surface(field, ray_origins, ray_directions, origin, spacing, value_range, num_steps)
    normals = _sample_gradient(normal_field, positions, origin, spacing)
    normals = jnp.where(
        jnp.sum(normals * ray_directions, axis=-1, keepdims=True) > 0.0,
        -normals,
        normals,
    )
    if use_colormap:
        color_value = _sample_scalar(color_field, positions, origin, spacing)
        color_value_span = jnp.maximum(color_value_range[1] - color_value_range[0], 1.0e-8)
        normalized_color_value = jnp.clip(
            (color_value - color_value_range[0]) / color_value_span,
            0.0,
            1.0,
        )
        lookup_index = jnp.minimum(
            (normalized_color_value * (colormap.shape[0] - 1)).astype(jnp.int32),
            colormap.shape[0] - 1,
        )
        surface_color = colormap[lookup_index]
    else:
        surface_color = jnp.broadcast_to(color, positions.shape)
    view = -ray_directions
    to_lights = light_positions[:, None, None, :] - positions[None, ...]
    distances_squared = jnp.sum(to_lights * to_lights, axis=-1)
    light_directions = _normalize(to_lights)
    half_vectors = _normalize(light_directions + view[None, ...])
    diffuse_angles = jnp.maximum(jnp.sum(normals[None, ...] * light_directions, axis=-1), 0.0)
    specular_angles = jnp.maximum(jnp.sum(normals[None, ...] * half_vectors, axis=-1), 0.0)
    shininess = 2.0 + 126.0 * (1.0 - roughness) ** 2
    fresnel_zero = 0.04 * (1.0 - metallic) + surface_color * metallic
    attenuation = light_intensities[:, None, None] / (1.0 + distances_squared)
    if use_shadows:
        visibility = _shadow_visibility(
            field,
            positions,
            normals,
            light_positions,
            origin,
            spacing,
            value_range,
            shadow_steps,
        )
    else:
        visibility = jnp.ones_like(attenuation)
    diffuse = surface_color[None, ...] * (1.0 - metallic) * diffuse_angles[..., None]
    specular = fresnel_zero[None, ...] * specular_angles[..., None] ** shininess
    direct = jnp.sum(
        (diffuse + specular) * light_colors[:, None, None, :] * attenuation[..., None] * visibility[..., None],
        axis=0,
    )
    indirect_weight = 1.0 - (0.5 + 0.5 * roughness) ** (max_bounces + 1)
    indirect = ambient_color * ambient_intensity * surface_color
    indirect += background_color * metallic * indirect_weight
    shaded = jnp.clip(direct + indirect, 0.0, 1.0)
    alpha = jnp.where(hit, opacity, 0.0)
    return shaded, alpha, depth


def _refract(incident, normal, eta):
    cosine = jnp.clip(-jnp.sum(incident * normal, axis=-1, keepdims=True), 0.0, 1.0)
    discriminant = 1.0 - eta**2 * (1.0 - cosine**2)
    refracted = eta * incident + (eta * cosine - jnp.sqrt(jnp.maximum(discriminant, 0.0))) * normal
    reflected = incident + 2.0 * cosine * normal
    return _normalize(jnp.where(discriminant >= 0.0, refracted, reflected))


@partial(jax.jit, static_argnames=("num_steps", "max_bounces", "use_colormap"))
def render_volume(
    field,
    color_field,
    ray_origins,
    ray_directions,
    origin,
    spacing,
    value_range,
    color_value_range,
    color,
    opacity,
    index_of_refraction,
    colormap,
    sampling_step,
    *,
    num_steps,
    max_bounces,
    use_colormap,
):
    """Ray march one refractive scalar volume on the active JAX device."""
    upper = origin + spacing * (jnp.asarray(field.shape, dtype=spacing.dtype) - 1.0)
    near, _, intersects = _box_intersection(ray_origins, ray_directions, origin, upper)
    step_size = sampling_step
    positions = ray_origins + (near + 1.0e-4 * step_size)[..., None] * ray_directions
    initial_state = (
        positions,
        ray_directions,
        jnp.zeros(ray_origins.shape, dtype=ray_origins.dtype),
        jnp.zeros(intersects.shape, dtype=ray_origins.dtype),
        jnp.full(intersects.shape, jnp.inf, dtype=ray_origins.dtype),
        jnp.zeros(intersects.shape, dtype=bool),
        jnp.zeros(intersects.shape, dtype=jnp.int32),
        intersects,
    )
    sample_alpha = 1.0 - (1.0 - opacity) ** (step_size / jnp.min(spacing))
    color_value_span = jnp.maximum(color_value_range[1] - color_value_range[0], 1.0e-8)

    def sample_step(index, state):
        position, direction, accumulated_color, accumulated_alpha, first_depth, in_medium, refractions, active = state
        in_bounds = active & _inside_box(position, origin, upper)
        value = _sample_scalar(field, position, origin, spacing)
        occupied = in_bounds & (value >= value_range[0]) & (value <= value_range[1])
        boundary_crossing = in_bounds & (occupied != in_medium) & (refractions < 2 * (max_bounces + 1))
        normal = _sample_gradient(field, position, origin, spacing)
        normal = jnp.where(
            jnp.sum(normal * direction, axis=-1, keepdims=True) > 0.0,
            -normal,
            normal,
        )
        eta = jnp.where(occupied, 1.0 / index_of_refraction, index_of_refraction)
        refracted = _refract(direction, normal, eta[..., None])
        direction = jnp.where(boundary_crossing[..., None], refracted, direction)
        refractions = refractions + boundary_crossing.astype(jnp.int32)
        if use_colormap:
            color_value = _sample_scalar(color_field, position, origin, spacing)
            normalized_color_value = jnp.clip(
                (color_value - color_value_range[0]) / color_value_span,
                0.0,
                1.0,
            )
            lookup_index = jnp.minimum(
                (normalized_color_value * (colormap.shape[0] - 1)).astype(jnp.int32),
                colormap.shape[0] - 1,
            )
            sample_color = colormap[lookup_index]
        else:
            sample_color = jnp.broadcast_to(color, position.shape)
        contribution = (1.0 - accumulated_alpha) * sample_alpha * occupied
        accumulated_color += contribution[..., None] * sample_color
        accumulated_alpha += contribution
        travel_depth = near + index * step_size
        first_depth = jnp.where(occupied & jnp.isinf(first_depth), travel_depth, first_depth)
        position += direction * step_size
        active = in_bounds & (accumulated_alpha < 0.995)
        return position, direction, accumulated_color, accumulated_alpha, first_depth, occupied, refractions, active

    _, _, accumulated_color, accumulated_alpha, depth, _, _, _ = jax.lax.fori_loop(0, num_steps, sample_step, initial_state)
    unpremultiplied = accumulated_color / jnp.maximum(accumulated_alpha[..., None], 1.0e-8)
    return jnp.clip(unpremultiplied, 0.0, 1.0), accumulated_alpha, depth


@partial(jax.jit, static_argnames=("iterations",))
def smooth_scalar_field(field, *, iterations):
    """Apply a separable nearest-neighbor smoother on device."""
    for _ in range(iterations):
        padded = jnp.pad(field, 1, mode="edge")
        field = (
            4.0 * padded[1:-1, 1:-1, 1:-1]
            + padded[:-2, 1:-1, 1:-1]
            + padded[2:, 1:-1, 1:-1]
            + padded[1:-1, :-2, 1:-1]
            + padded[1:-1, 2:, 1:-1]
            + padded[1:-1, 1:-1, :-2]
            + padded[1:-1, 1:-1, 2:]
        ) / 10.0
    return field


@jax.jit
def composite_layers(colors, alphas, depths, background_color):
    """Depth-sort and alpha-composite independently rendered field layers."""
    order = jnp.argsort(depths, axis=0)
    ordered_colors = jnp.take_along_axis(colors, order[..., None], axis=0)
    ordered_alphas = jnp.take_along_axis(alphas, order, axis=0)
    result = jnp.broadcast_to(background_color, colors.shape[1:])
    for layer in range(colors.shape[0] - 1, -1, -1):
        alpha = ordered_alphas[layer, ..., None]
        result = ordered_colors[layer] * alpha + result * (1.0 - alpha)
    return jnp.clip(result, 0.0, 1.0)


@jax.jit
def apply_edge_antialiasing(image, strength):
    """Smooth high-contrast raster edges with an FXAA-style image pass.

    Parameters
    ----------
    image: jax.Array
        RGB image of shape ``(height, width, 3)`` with values in ``[0, 1]``.
    strength: jax.Array
        Edge blend strength in ``[0, 1]``.

    Returns
    -------
    jax.Array
        Antialiased RGB image with the same shape and dtype.

    Notes
    -----
    The pass follows the contrast detection used by FXAA 3.11 while using a compact
    nine-tap filter suitable for in-situ scientific rendering.
    """
    padded = jnp.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    center = padded[1:-1, 1:-1]
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]
    northwest = padded[:-2, :-2]
    northeast = padded[:-2, 2:]
    southwest = padded[2:, :-2]
    southeast = padded[2:, 2:]

    luminance_weights = jnp.asarray((0.2126, 0.7152, 0.0722), dtype=image.dtype)
    luminance = jnp.stack([
        jnp.sum(sample * luminance_weights, axis=-1) for sample in (center, north, south, west, east, northwest, northeast, southwest, southeast)
    ])
    local_minimum = jnp.min(luminance, axis=0)
    local_maximum = jnp.max(luminance, axis=0)
    contrast = local_maximum - local_minimum
    threshold = jnp.maximum(0.025, 0.125 * local_maximum)
    edge_weight = jnp.clip((contrast - threshold) / jnp.maximum(contrast, 1.0e-6), 0.0, 1.0)
    edge_weight = edge_weight**0.65 * strength

    filtered = (4.0 * center + 2.0 * (north + south + west + east) + northwest + northeast + southwest + southeast) / 16.0
    return jnp.clip(center + edge_weight[..., None] * (filtered - center), 0.0, 1.0)
