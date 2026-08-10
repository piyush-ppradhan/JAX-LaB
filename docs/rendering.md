# Rendering

JAX-LaB includes a JAX-native ray marcher for in-situ rendering of three-dimensional arrays. A scene maps each field name to its own surface,
refractive volume, or vector-magnitude material. Lighting, shadows, camera motion, global surface smoothing, and ray bounce count are configured once
on the scene. Only the final RGB image is transferred to the host when a PNG is requested.

```python
from jax_lab.render import Light, Scene, SurfaceRendering, VolumeRendering

scene = Scene(
    {
        "interface": SurfaceRendering(
            value_range=(0.5, 1.0),
            color=(0.15, 0.45, 0.95),
            metallic=0.05,
            roughness=0.3,
        ),
        "liquid": VolumeRendering(
            value_range=(0.5, 1.0),
            color=(0.05, 0.3, 0.9),
            opacity=0.06,
            index_of_refraction=1.333,
        ),
    },
    resolution=(1280, 720),
    position=(80.0, 60.0, -90.0),
    target=(32.0, 32.0, 32.0),
    lights=(Light(position=(-20.0, 100.0, -50.0), intensity=12.0),),
    surface_smoothing=2,
    anti_aliasing=True,
    anti_aliasing_strength=0.9,
    max_bounces=1,
)

image = scene.render(
    {"interface": density, "liquid": density},
    timestep=timestep,
    filename=f"droplet_{timestep:07d}.png",
)
```

`VectorRendering` accepts arrays shaped `(nx, ny, nz, 3)` and maps their magnitude through `viridis`, `plasma`, `inferno`, `magma`, `jet`, `gray`,
or a custom RGB lookup array. Scalar fields may be shaped `(nx, ny, nz)` or `(nx, ny, nz, 1)`.

Large `max_bounces` values increase both compilation and render time considerably. The default of one is intended for routine in-situ output. Increase
`samples_per_voxel` when a thin interface is missed, at the cost of proportionally more ray samples.

`VolumeRendering.value_range` defines occupancy, not only color normalization. Leave enough margin above the expected liquid density for transient
overshoots. A tight upper cutoff can turn small compressibility oscillations into artificial cavities and refractive layers.

Edge antialiasing is enabled by default. It runs on the active JAX device after layer compositing and adds only a fixed nine-tap image pass. Set
`anti_aliasing=False` for the lowest possible output overhead, or adjust `anti_aliasing_strength` between zero and one.

The renderer samples rays twice per voxel by default to avoid missing thin or grazing-angle interfaces. Lower `samples_per_voxel` only when render
throughput is more important than edge quality.

`SurfaceRendering` and `VolumeRendering` can color one field with another named field. Set `color_field` to a scalar field or a vector field of the
same spatial shape, provide its `color_range`, and select a `colormap`. Vector color fields are converted to magnitude on the active JAX device.
This is useful for coloring a density interface with velocity while the density interval continues to control the rendered region.

`examples/rendering/liquid_on_staggered_slabs.py` combines volume rendering with solid-surface rendering for gravity-driven liquid flow through two
hydrophobic staggered slabs.

::: jax_lab.render.Light

::: jax_lab.render.SurfaceRendering

::: jax_lab.render.VolumeRendering

::: jax_lab.render.VectorRendering

::: jax_lab.render.Scene

::: jax_lab.render.render
