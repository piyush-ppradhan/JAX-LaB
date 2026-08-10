import importlib
import logging
import os
import re
import sys
import warnings
from functools import cache, partial
from time import time

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.image import resize
from pathlib import Path

logger = logging.getLogger(__name__)

_ANSI_COLORS = {
    "black": 30,
    "grey": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "light_grey": 37,
    "dark_grey": 90,
    "light_red": 91,
    "light_green": 92,
    "light_yellow": 93,
    "light_blue": 94,
    "light_magenta": 95,
    "light_cyan": 96,
    "white": 97,
}
_ANSI_RESET = "\033[0m"


@cache
def _can_colorize(no_color=None, force_color=None):
    if no_color:
        return False
    if force_color:
        return True
    if os.environ.get("ANSI_COLORS_DISABLED") or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb" or not hasattr(sys.stdout, "fileno"):
        return False

    try:
        return os.isatty(sys.stdout.fileno())
    except OSError:
        return sys.stdout.isatty()


def colored(text, color, *, no_color=None, force_color=None):
    """
    Wrap text in an ANSI foreground color when the terminal supports it.

    Parameters
    ----------
    text (object): Value to convert to a string and colorize.

    color (str): ANSI color name. Standard and light foreground colors are supported.

    no_color (bool, optional): Disable ANSI colors explicitly.

    force_color (bool, optional): Enable ANSI colors explicitly.

    Returns
    -------
    str: Colored text, or plain text when colors are disabled or unsupported.
    """
    text = str(text)
    if not _can_colorize(no_color, force_color):
        return text
    return f"\033[{_ANSI_COLORS[color]}m{text}{_ANSI_RESET}"


def _import_optional(module_name, feature):
    """
    Imports an optional dependency lazily, raising a helpful error if missing.

    Heavy I/O and visualization dependencies (h5py, matplotlib, pyvista,
    trimesh) are imported at call time instead of module import time so that
    the core solver can run without them installed. After the first import,
    subsequent calls only perform a cached module lookup.

    Parameters
    ----------
    module_name (str): The module to import, e.g. "pyvista" or "matplotlib.pylab".

    feature (str): Name of the calling feature, used in the error message.

    Returns
    -------
    module: The imported module.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        package = module_name.split(".")[0]
        raise ImportError(f"{feature} requires the optional dependency '{package}'. Install it with: pip install {package}") from e


def read_raw_volume(path, shape, dtype="u1", endian="<", order="C", use_memmap=False):
    """
    Read a raw binary volume using NumPy.

    Parameters
    ----------
    path : str or Path
        Path to the raw file.
    shape : tuple[int, ...]
        Volume shape, e.g. (1000, 1000, 1000).
    dtype : str or np.dtype
        Element type without endianness, e.g. "u1", "i1", "u2", "f4".
        For 8-bit char data use "u1" (unsigned) or "i1" (signed).
    endian : str
        "<" little-endian, ">" big-endian.
    order : str
        "C" for C-order, "F" for Fortran-order.
    use_memmap : bool
        If True, return a memmap (does not load full array into RAM).

    Returns
    -------
    np.ndarray or np.memmap
    """
    dtype = np.dtype(endian + np.dtype(dtype).str[1:])
    expected = int(np.prod(shape))

    if use_memmap:
        data = np.memmap(path, dtype=dtype, mode="r", shape=shape, order=order)
        return data

    data = np.fromfile(path, dtype=dtype, count=expected)
    if data.size != expected:
        raise ValueError(f"File has {data.size} elements, expected {expected} for shape {shape}.")
    return data.reshape(shape, order=order)


@partial(jit, static_argnums=(1, 2))
def downsample_field(field, factor, method="bicubic"):
    """
    Downsample a JAX array by a factor of `factor` along each axis.

    Parameters
    ----------
    field (jax.numpy.ndarray): The input vector field to be downsampled. This should be a 3D or 4D JAX array where the last dimension is 2 or 3 (vector components).

    factor (int): The factor by which to downsample the field. The dimensions of the field will be divided by this factor.

    method (str, optional): The method to use for downsampling. Default is 'bicubic'.

    Returns
    -------
    jax.numpy.ndarray: The downsampled field.
    """
    if factor == 1:
        return field
    else:
        new_shape = tuple(dim // factor for dim in field.shape[:-1])
        downsampled_components = []
        for i in range(field.shape[-1]):  # Iterate over the last dimension (vector components)
            resized = resize(field[..., i], new_shape, method=method)
            downsampled_components.append(resized)

        return jnp.stack(downsampled_components, axis=-1)


def save_image(timestep, fld, prefix=None):
    """
    Save an image of a field at a given timestep.

    Parameters
    ----------
    timestep (int): The timestep at which the field is being saved.

    fld (jax.numpy.ndarray): The field to be saved. This should be a 2D or 3D JAX array. If the field is 3D, the magnitude of the field will be calculated and saved.

    prefix (str, optional): A prefix to be added to the filename. The filename will be the name of the main script file by default.

    Returns
    -------
    None

    Notes
    -----
    This function saves the field as an image in the PNG format. The filename is based on the name of the main script file, the provided prefix, and the timestep number.
    If the field is 3D, the magnitude of the field is calculated and saved. The image is saved with the 'nipy_spectral' colormap and the origin set to 'lower'.
    """
    plt = _import_optional("matplotlib.pylab", "save_image")
    cm = _import_optional("matplotlib.cm", "save_image")
    import __main__

    fname = os.path.basename(__main__.__file__)
    fname = os.path.splitext(fname)[0]
    if prefix is not None:
        fname = prefix + fname
    fname = fname + "_" + str(timestep).zfill(4)

    if len(fld.shape) > 3:
        raise ValueError("The input field should be 2D!")

    if len(fld.shape) == 3:
        fld = np.sqrt(fld[..., 0] ** 2 + fld[..., 1] ** 2)

    plt.clf()
    plt.imsave(fname + ".png", fld.T, cmap=cm.nipy_spectral, origin="lower")


def save_fields_hdf5_xdmf(
    timestep,
    fields,
    output_dir=".",
    prefix="fields",
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    compression="gzip",
    compression_level=5,
    shuffle=True,
    target_chunk_bytes=2 * 1024 * 1024,
    multi_timestep_file=None,
    static_fields=None,
    static_fields_file=None,
):
    """
    Save 2D/3D cell-centered fields (dict of arrays) as HDF5/XDMF.

    Parameters
    ----------
    timestep (int): The timestep number to be associated with the saved fields.

    fields (Dict[str, np.ndarray]): A dictionary of fields to be saved. Each field must be an array-like object with dimensions (nx, ny) for 2D fields
    or (nx, ny, nz) for 3D fields, where:

    - nx : int, number of grid points along the x-axis
    - ny : int, number of grid points along the y-axis
    - nz : int, number of grid points along the z-axis (for 3D fields only)

    The key value for each field in the dictionary must be a string containing the name of the field.

    output_dir (str, optional, default: '.'):  The directory in which to save the HDF5 files. Defaults to the current directory.

    prefix (str, optional, default: 'fields'): A prefix to be added to the filename. Defaults to 'fields'.

    origin (tuple[float]): The origin used for the file.

    spacing (tuple[float]): Spacing used for data points.

    compression (str): Compression algorithm used, supported options: `lzf`, `gzip` (default) and `szip`.

    compression_level (int): Options for the compression filter.

    shuffle (bool): Whether shuffle filter is applied. True by default.

    target_chunk_bytes (int): Chunk size to be used in bytes. 2 * 1024 * 1024 (2 MB) by default.

    multi_timestep_file (bool): Store data for all timesteps in a single, consolidated file.

    static_fields (dict, Optional): Pass a dict for static fields that doesn't change. Useful for passing for non-changing values (for e.g., mask array).
    Static fields are stored once and referenced from all timesteps.

    static_fields_file (str, Optional): Name of the file where static field is stored. Default: {prefix}_static.hdf5

    Returns
    -------
    None

    Notes
    -----
    Single-step mode:
      {output_dir}/{prefix}_{timestep:07d}.hdf5
      {output_dir}/{prefix}_{timestep:07d}.xdmf
      {output_dir}/{prefix}_series.xdmf

    Multi-step mode (append):
      {multi_timestep_file}
      {multi_timestep_file with .xdmf suffix}
    """
    h5py = _import_optional("h5py", "save_fields_hdf5_xdmf")
    start = time()

    if not isinstance(fields, dict) or len(fields) == 0:
        raise ValueError("fields must be a non-empty dict[str, np.ndarray].")

    arrays = {}
    shape = None
    ndim = None
    for name, value in fields.items():
        arr = np.asarray(value)
        if arr.ndim not in (2, 3):
            raise ValueError(f"Field '{name}' must be 2D or 3D, got shape {arr.shape}.")
        if shape is None:
            shape = arr.shape
            ndim = arr.ndim
        elif arr.shape != shape or arr.ndim != ndim:
            raise ValueError("All fields must have the same dimensions and ndim (2D or 3D).")
        arrays[name] = arr

    origin = np.asarray(origin, dtype=np.float64).ravel()
    spacing = np.asarray(spacing, dtype=np.float64).ravel()
    if origin.size < ndim:
        raise ValueError(f"origin must have at least {ndim} values.")
    if spacing.size < ndim:
        raise ValueError(f"spacing must have at least {ndim} values.")
    origin = tuple(float(v) for v in origin[:ndim])
    spacing = tuple(float(v) for v in spacing[:ndim])
    if np.any(np.asarray(spacing) <= 0.0):
        raise ValueError("spacing values must be positive.")

    field_names = tuple(arrays.keys())
    field_signature = "|".join(field_names)
    dtype_signature = "|".join(str(arrays[name].dtype) for name in field_names)

    static_arrays = {}
    static_field_names = ()
    static_field_signature = ""
    static_dtype_signature = ""
    if static_fields is not None:
        if not isinstance(static_fields, dict) or len(static_fields) == 0:
            raise ValueError("static_fields must be None or a non-empty dict[str, np.ndarray].")
        for name, value in static_fields.items():
            if name in arrays:
                raise ValueError(f"static field '{name}' conflicts with dynamic field name.")
            arr = np.asarray(value)
            if arr.ndim != ndim or arr.shape != shape:
                raise ValueError(f"Static field '{name}' must have shape {shape} and ndim {ndim}, got {arr.shape}.")
            static_arrays[name] = arr
        static_field_names = tuple(static_arrays.keys())
        static_field_signature = "|".join(static_field_names)
        static_dtype_signature = "|".join(str(static_arrays[name].dtype) for name in static_field_names)

    kwargs_cache = {}

    def dataset_kwargs(ds_shape, ds_dtype, include_time_axis=False):
        key = (tuple(int(v) for v in ds_shape), np.dtype(ds_dtype).str, bool(include_time_axis))
        if key in kwargs_cache:
            return dict(kwargs_cache[key])

        kwargs = {"track_times": False}
        if any(int(dim) == 0 for dim in ds_shape):
            kwargs_cache[key] = dict(kwargs)
            return kwargs

        chunks = list(ds_shape)
        if include_time_axis and len(chunks) > 0:
            chunks[0] = 1
        itemsize = np.dtype(ds_dtype).itemsize
        while np.prod(chunks, dtype=np.int64) * itemsize > int(target_chunk_bytes):
            idx = int(np.argmax(chunks))
            if chunks[idx] <= 1:
                break
            chunks[idx] = (chunks[idx] + 1) // 2
        kwargs["chunks"] = tuple(max(1, int(c)) for c in chunks)

        if compression is not None:
            kwargs["compression"] = compression
            if compression == "gzip":
                kwargs["compression_opts"] = int(compression_level)
            if shuffle and np.dtype(ds_dtype).kind in ("i", "u", "f", "b"):
                kwargs["shuffle"] = True

        kwargs_cache[key] = dict(kwargs)
        return kwargs

    def fsync_if_possible(h5_file):
        try:
            handle = h5_file.id.get_vfd_handle()
            if isinstance(handle, tuple):
                handle = handle[0]
            if isinstance(handle, int):
                os.fsync(handle)
        except Exception:
            pass

    def to_xdmf_type(dtype):
        dtype = np.dtype(dtype)
        if dtype.kind == "f":
            return "Float", dtype.itemsize
        if dtype.kind == "i":
            return "Int", dtype.itemsize
        if dtype.kind in ("u", "b"):
            return "UInt", dtype.itemsize
        raise TypeError(f"Unsupported dtype for XDMF export: {dtype}.")

    xdmf_path = None

    if multi_timestep_file is not None:
        h5_path = Path(multi_timestep_file)
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(
            h5_path,
            "a",
            libver="latest",
            rdcc_nbytes=max(int(target_chunk_bytes) * max(8, len(field_names)), 8 * 1024 * 1024),
        ) as h5_file:
            if "layout" not in h5_file.attrs:
                h5_file.attrs["layout"] = "multi_timestep_dense"
                h5_file.attrs["ndim"] = int(ndim)
                h5_file.attrs["shape"] = tuple(int(v) for v in shape)
                h5_file.attrs["origin"] = origin
                h5_file.attrs["spacing"] = spacing
                h5_file.attrs["field_signature"] = field_signature
                h5_file.attrs["dtype_signature"] = dtype_signature
                h5_file.attrs["static_field_signature"] = static_field_signature
                h5_file.attrs["static_dtype_signature"] = static_dtype_signature

                time_ds = h5_file.create_dataset(
                    "__timesteps__",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int64,
                    **dataset_kwargs((1024,), np.int64, include_time_axis=False),
                )
                time_ds.attrs["pending_value"] = -1
                h5_file.create_group("fields")
            else:
                if str(h5_file.attrs.get("layout")) != "multi_timestep_dense":
                    raise ValueError(f"Existing file {h5_path} is not dense multi-timestep layout.")
                if int(h5_file.attrs.get("ndim")) != int(ndim):
                    raise ValueError("ndim does not match existing multi-timestep file.")
                if tuple(int(v) for v in h5_file.attrs.get("shape")) != tuple(int(v) for v in shape):
                    raise ValueError("field shape does not match existing multi-timestep file.")
                if str(h5_file.attrs.get("field_signature")) != field_signature:
                    raise ValueError("field names/order do not match existing multi-timestep file.")
                if str(h5_file.attrs.get("dtype_signature")) != dtype_signature:
                    raise ValueError("field dtypes do not match existing multi-timestep file.")
                existing_static_sig = str(h5_file.attrs.get("static_field_signature", ""))
                existing_static_dtype_sig = str(h5_file.attrs.get("static_dtype_signature", ""))
                if static_fields is not None:
                    if existing_static_sig not in ("", static_field_signature):
                        raise ValueError("static field names/order do not match existing multi_timestep file.")
                    if existing_static_dtype_sig not in ("", static_dtype_signature):
                        raise ValueError("static field dtypes do not match existing multi_timestep file.")

            fields_group = h5_file["fields"]
            time_ds = h5_file["__timesteps__"]
            pending_value = int(time_ds.attrs.get("pending_value", -1))
            static_meta = {}

            if "static" not in h5_file:
                h5_file.create_group("static")
            static_group = h5_file["static"]

            if static_fields is not None and str(h5_file.attrs.get("static_field_signature", "")) == "":
                h5_file.attrs["static_field_signature"] = static_field_signature
                h5_file.attrs["static_dtype_signature"] = static_dtype_signature

            for name, arr in static_arrays.items():
                data_out = np.transpose(arr, (2, 1, 0)) if ndim == 3 else np.transpose(arr, (1, 0))
                if name not in static_group:
                    ds = static_group.create_dataset(
                        name,
                        data=data_out,
                        **dataset_kwargs(data_out.shape, data_out.dtype),
                    )
                    ds.attrs["original_shape"] = arr.shape
                    ds.attrs["original_order"] = "XYZ" if ndim == 3 else "XY"
                else:
                    ds = static_group[name]
                    if ds.shape != data_out.shape:
                        raise ValueError(f"Static dataset shape mismatch for field '{name}'.")
                    if ds.dtype != data_out.dtype:
                        raise ValueError(f"Static dataset dtype mismatch for field '{name}'.")

            for name in static_group.keys():
                ds = static_group[name]
                static_meta[name] = (tuple(int(v) for v in ds.shape), ds.dtype)

            while len(time_ds) > 0 and int(time_ds[len(time_ds) - 1]) == pending_value:
                new_n = len(time_ds) - 1
                time_ds.resize((new_n,))
                for name in field_names:
                    if name in fields_group:
                        ds = fields_group[name]
                        ds.resize((new_n,) + ds.shape[1:])

            for name, arr in arrays.items():
                data_out = np.transpose(arr, (2, 1, 0)) if ndim == 3 else np.transpose(arr, (1, 0))
                if name not in fields_group:
                    base_shape = data_out.shape
                    ds = fields_group.create_dataset(
                        name,
                        shape=(0,) + base_shape,
                        maxshape=(None,) + base_shape,
                        dtype=data_out.dtype,
                        **dataset_kwargs((1,) + base_shape, data_out.dtype, include_time_axis=True),
                    )
                    ds.attrs["original_shape"] = arr.shape
                    ds.attrs["original_order"] = "XYZ" if ndim == 3 else "XY"
                else:
                    ds = fields_group[name]
                    if ds.shape[1:] != data_out.shape:
                        raise ValueError(f"Dataset shape mismatch for field '{name}'.")
                    if ds.dtype != data_out.dtype:
                        raise ValueError(f"Dataset dtype mismatch for field '{name}'.")

            target_timestep = int(timestep)
            idx = None

            if len(time_ds) > 0 and int(time_ds[len(time_ds) - 1]) == target_timestep:
                idx = len(time_ds) - 1
            elif len(time_ds) > 0:
                all_timesteps = time_ds[...]
                matches = np.flatnonzero(all_timesteps == target_timestep)
                if matches.size > 0:
                    idx = int(matches[-1])

            if idx is None:
                idx = len(time_ds)
                time_ds.resize((idx + 1,))
                time_ds[idx] = pending_value
                for name, arr in arrays.items():
                    data_out = np.transpose(arr, (2, 1, 0)) if ndim == 3 else np.transpose(arr, (1, 0))
                    ds = fields_group[name]
                    ds.resize((idx + 1,) + ds.shape[1:])
                    ds[idx, ...] = data_out
                time_ds[idx] = target_timestep
            else:
                for name, arr in arrays.items():
                    data_out = np.transpose(arr, (2, 1, 0)) if ndim == 3 else np.transpose(arr, (1, 0))
                    fields_group[name][idx, ...] = data_out

            h5_file.flush()
            fsync_if_possible(h5_file)

            timesteps = np.asarray(time_ds[...], dtype=np.int64)
            if timesteps.size > 0:
                nx = int(shape[0])
                ny = int(shape[1])
                if ndim == 3:
                    nz = int(shape[2])
                    ox, oy, oz = origin
                    sx, sy, sz = spacing
                    topology = f'        <Topology TopologyType="3DCoRectMesh" Dimensions="{nz + 1} {ny + 1} {nx + 1}"/>'
                    geometry = [
                        '        <Geometry GeometryType="ORIGIN_DXDYDZ">',
                        f'          <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">{oz} {oy} {ox}</DataItem>',
                        f'          <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">{sz} {sy} {sx}</DataItem>',
                        "        </Geometry>",
                    ]
                else:
                    ox, oy = origin
                    sx, sy = spacing
                    topology = f'        <Topology TopologyType="2DCoRectMesh" Dimensions="{ny + 1} {nx + 1}"/>'
                    geometry = [
                        '        <Geometry GeometryType="ORIGIN_DXDY">',
                        f'          <DataItem Dimensions="2" NumberType="Float" Precision="8" Format="XML">{oy} {ox}</DataItem>',
                        f'          <DataItem Dimensions="2" NumberType="Float" Precision="8" Format="XML">{sy} {sx}</DataItem>',
                        "        </Geometry>",
                    ]

                grids = []
                for i, ts in enumerate(timesteps):
                    grids.extend([
                        f'      <Grid Name="timestep_{int(ts):07d}" GridType="Uniform">',
                        f'        <Time Value="{int(ts)}"/>',
                        topology,
                        *geometry,
                    ])

                    for name in field_names:
                        ds = fields_group[name]
                        number_type, precision = to_xdmf_type(ds.dtype)

                        if ndim == 3:
                            dims = f"{int(ds.shape[1])} {int(ds.shape[2])} {int(ds.shape[3])}"
                            slab_spec = f"{i} 0 0 0\n            1 1 1 1\n            1 {int(ds.shape[1])} {int(ds.shape[2])} {int(ds.shape[3])}"
                        else:
                            dims = f"{int(ds.shape[1])} {int(ds.shape[2])}"
                            slab_spec = f"{i} 0 0\n            1 1 1\n            1 {int(ds.shape[1])} {int(ds.shape[2])}"

                        full_dims = " ".join(str(int(v)) for v in ds.shape)
                        slab_dim_width = 4 if ndim == 3 else 3
                        grids.extend([
                            f'        <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">',
                            f'          <DataItem ItemType="HyperSlab" Type="HyperSlab" Dimensions="{dims}">',
                            f'            <DataItem Dimensions="3 {slab_dim_width}" Format="XML">{slab_spec}</DataItem>',
                            (
                                f'            <DataItem Dimensions="{full_dims}" NumberType="{number_type}" '
                                f'Precision="{precision}" Format="HDF">{h5_path.name}:/fields/{name}</DataItem>'
                            ),
                            "          </DataItem>",
                            "        </Attribute>",
                        ])

                    for name, (ds_shape, ds_dtype) in static_meta.items():
                        number_type, precision = to_xdmf_type(ds_dtype)
                        dims = " ".join(str(int(d)) for d in ds_shape)
                        grids.extend([
                            f'        <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">',
                            (
                                f'          <DataItem Dimensions="{dims}" NumberType="{number_type}" '
                                f'Precision="{precision}" Format="HDF">{h5_path.name}:/static/{name}</DataItem>'
                            ),
                            "        </Attribute>",
                        ])
                    grids.append("      </Grid>")

                xdmf_path = h5_path.with_suffix(".xdmf")
                xdmf_lines = [
                    '<?xml version="1.0" ?>',
                    '<Xdmf Version="3.0">',
                    "  <Domain>",
                    '    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">',
                    *grids,
                    "    </Grid>",
                    "  </Domain>",
                    "</Xdmf>",
                    "",
                ]
                xdmf_path.write_text("\n".join(xdmf_lines), encoding="utf-8")
    else:
        output_stem = Path(output_dir) / f"{prefix}_{int(timestep):07d}"
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        h5_path = output_stem.with_suffix(".hdf5")
        xdmf_path = output_stem.with_suffix(".xdmf")
        static_meta = {}
        static_ref_path = None

        if static_fields_file is None:
            static_h5_path = output_stem.parent / f"{prefix}_static.hdf5"
        else:
            static_h5_path = Path(static_fields_file)
            if not static_h5_path.is_absolute():
                static_h5_path = output_stem.parent / static_h5_path

        if static_fields is not None or static_h5_path.exists():
            if not static_h5_path.exists():
                with h5py.File(
                    static_h5_path,
                    "w",
                    libver="latest",
                    rdcc_nbytes=max(int(target_chunk_bytes) * max(8, len(static_field_names)), 8 * 1024 * 1024),
                ) as static_file:
                    static_file.attrs["layout"] = "single_timestep_static"
                    static_file.attrs["ndim"] = int(ndim)
                    static_file.attrs["shape"] = tuple(int(v) for v in shape)
                    static_file.attrs["origin"] = origin
                    static_file.attrs["spacing"] = spacing
                    static_file.attrs["static_field_signature"] = static_field_signature
                    static_file.attrs["static_dtype_signature"] = static_dtype_signature
                    for name, arr in static_arrays.items():
                        data_out = np.transpose(arr, (2, 1, 0)) if ndim == 3 else np.transpose(arr, (1, 0))
                        ds = static_file.create_dataset(
                            name,
                            data=data_out,
                            **dataset_kwargs(data_out.shape, data_out.dtype),
                        )
                        ds.attrs["original_shape"] = arr.shape
                        ds.attrs["original_order"] = "XYZ" if ndim == 3 else "XY"
                    static_file.flush()
                    fsync_if_possible(static_file)

            with h5py.File(static_h5_path, "a", libver="latest") as static_file:
                if int(static_file.attrs.get("ndim")) != int(ndim):
                    raise ValueError("Static file ndim does not match dynamic fields.")
                if tuple(int(v) for v in static_file.attrs.get("shape")) != tuple(int(v) for v in shape):
                    raise ValueError("Static file shape does not match dynamic fields.")

                existing_static_sig = str(static_file.attrs.get("static_field_signature", ""))
                existing_static_dtype_sig = str(static_file.attrs.get("static_dtype_signature", ""))

                if static_fields is not None:
                    if existing_static_sig not in ("", static_field_signature):
                        raise ValueError("static field names/order do not match static file.")
                    if existing_static_dtype_sig not in ("", static_dtype_signature):
                        raise ValueError("static field dtypes do not match static file.")

                if static_fields is not None and existing_static_sig == "":
                    static_file.attrs["static_field_signature"] = static_field_signature
                    static_file.attrs["static_dtype_signature"] = static_dtype_signature
                    for name, arr in static_arrays.items():
                        if name not in static_file:
                            data_out = np.transpose(arr, (2, 1, 0)) if ndim == 3 else np.transpose(arr, (1, 0))
                            ds = static_file.create_dataset(
                                name,
                                data=data_out,
                                **dataset_kwargs(data_out.shape, data_out.dtype),
                            )
                            ds.attrs["original_shape"] = arr.shape
                            ds.attrs["original_order"] = "XYZ" if ndim == 3 else "XY"

                for name in static_file.keys():
                    ds = static_file[name]
                    static_meta[name] = (tuple(int(v) for v in ds.shape), ds.dtype)
                static_file.flush()
                fsync_if_possible(static_file)

            static_ref_path = os.path.relpath(static_h5_path, output_stem.parent).replace("\\", "/")

        with h5py.File(
            h5_path,
            "w",
            libver="latest",
            rdcc_nbytes=max(int(target_chunk_bytes) * max(8, len(field_names)), 8 * 1024 * 1024),
        ) as h5_file:
            h5_file.attrs["layout"] = "single_timestep_dense"
            h5_file.attrs["ndim"] = int(ndim)
            h5_file.attrs["shape"] = tuple(int(v) for v in shape)
            h5_file.attrs["origin"] = origin
            h5_file.attrs["spacing"] = spacing
            h5_file.attrs["field_signature"] = field_signature
            h5_file.attrs["dtype_signature"] = dtype_signature

            dense_meta = {}
            for name, arr in arrays.items():
                data_out = np.transpose(arr, (2, 1, 0)) if ndim == 3 else np.transpose(arr, (1, 0))
                ds = h5_file.create_dataset(name, data=data_out, **dataset_kwargs(data_out.shape, data_out.dtype))
                ds.attrs["original_shape"] = arr.shape
                ds.attrs["original_order"] = "XYZ" if ndim == 3 else "XY"
                dense_meta[name] = (data_out.shape, ds.dtype)

            h5_file.flush()
            fsync_if_possible(h5_file)

        nx = int(shape[0])
        ny = int(shape[1])
        if ndim == 3:
            nz = int(shape[2])
            ox, oy, oz = origin
            sx, sy, sz = spacing
            topology = f'      <Topology TopologyType="3DCoRectMesh" Dimensions="{nz + 1} {ny + 1} {nx + 1}"/>'
            geometry = [
                '      <Geometry GeometryType="ORIGIN_DXDYDZ">',
                f'        <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">{oz} {oy} {ox}</DataItem>',
                f'        <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">{sz} {sy} {sx}</DataItem>',
                "      </Geometry>",
            ]
        else:
            ox, oy = origin
            sx, sy = spacing
            topology = f'      <Topology TopologyType="2DCoRectMesh" Dimensions="{ny + 1} {nx + 1}"/>'
            geometry = [
                '      <Geometry GeometryType="ORIGIN_DXDY">',
                f'        <DataItem Dimensions="2" NumberType="Float" Precision="8" Format="XML">{oy} {ox}</DataItem>',
                f'        <DataItem Dimensions="2" NumberType="Float" Precision="8" Format="XML">{sy} {sx}</DataItem>',
                "      </Geometry>",
            ]

        attributes = []
        for name in arrays:
            ds_shape, ds_dtype = dense_meta[name]
            number_type, precision = to_xdmf_type(ds_dtype)
            dims = " ".join(str(int(d)) for d in ds_shape)
            attributes.extend([
                f'      <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">',
                (
                    f'        <DataItem Dimensions="{dims}" NumberType="{number_type}" '
                    f'Precision="{precision}" Format="HDF">{h5_path.name}:/{name}</DataItem>'
                ),
                "      </Attribute>",
            ])

        if static_ref_path is not None:
            for name, (ds_shape, ds_dtype) in static_meta.items():
                number_type, precision = to_xdmf_type(ds_dtype)
                dims = " ".join(str(int(d)) for d in ds_shape)
                attributes.extend([
                    f'      <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">',
                    (
                        f'        <DataItem Dimensions="{dims}" NumberType="{number_type}" '
                        f'Precision="{precision}" Format="HDF">{static_ref_path}:/{name}</DataItem>'
                    ),
                    "      </Attribute>",
                ])

        xdmf_lines = [
            '<?xml version="1.0" ?>',
            '<Xdmf Version="3.0">',
            "  <Domain>",
            '    <Grid Name="UniformGrid" GridType="Uniform">',
            topology,
            *geometry,
            *attributes,
            "    </Grid>",
            "  </Domain>",
            "</Xdmf>",
            "",
        ]
        xdmf_path.write_text("\n".join(xdmf_lines), encoding="utf-8")

        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.hdf5$")
        step_files = []
        for candidate in output_stem.parent.glob(f"{prefix}_*.hdf5"):
            match = pattern.match(candidate.name)
            if match is None:
                continue
            try:
                with h5py.File(candidate, "r") as f:
                    if str(f.attrs.get("layout", "")) not in ("single_timestep_dense", "single_timestep"):
                        continue
                    if tuple(int(v) for v in f.attrs.get("shape", ())) != tuple(int(v) for v in shape):
                        continue
                    if int(f.attrs.get("ndim", -1)) != int(ndim):
                        continue
                    if str(f.attrs.get("field_signature", "")) != field_signature:
                        continue
                    if str(f.attrs.get("dtype_signature", "")) != dtype_signature:
                        continue
            except Exception:
                continue
            step_files.append((int(match.group(1)), candidate.name))
        step_files.sort(key=lambda x: x[0])

        if step_files:
            if ndim == 3:
                topology = f'        <Topology TopologyType="3DCoRectMesh" Dimensions="{nz + 1} {ny + 1} {nx + 1}"/>'
                geometry = [
                    '        <Geometry GeometryType="ORIGIN_DXDYDZ">',
                    f'          <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">{oz} {oy} {ox}</DataItem>',
                    f'          <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">{sz} {sy} {sx}</DataItem>',
                    "        </Geometry>",
                ]
            else:
                topology = f'        <Topology TopologyType="2DCoRectMesh" Dimensions="{ny + 1} {nx + 1}"/>'
                geometry = [
                    '        <Geometry GeometryType="ORIGIN_DXDY">',
                    f'          <DataItem Dimensions="2" NumberType="Float" Precision="8" Format="XML">{oy} {ox}</DataItem>',
                    f'          <DataItem Dimensions="2" NumberType="Float" Precision="8" Format="XML">{sy} {sx}</DataItem>',
                    "        </Geometry>",
                ]

            grids = []
            for ts, h5_name in step_files:
                grids.extend([
                    f'      <Grid Name="timestep_{ts:07d}" GridType="Uniform">',
                    f'        <Time Value="{ts}"/>',
                    topology,
                    *geometry,
                ])
                for name in arrays:
                    ds_shape, ds_dtype = dense_meta[name]
                    number_type, precision = to_xdmf_type(ds_dtype)
                    dims = " ".join(str(int(d)) for d in ds_shape)
                    grids.extend([
                        f'        <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">',
                        (
                            f'          <DataItem Dimensions="{dims}" NumberType="{number_type}" '
                            f'Precision="{precision}" Format="HDF">{h5_name}:/{name}</DataItem>'
                        ),
                        "        </Attribute>",
                    ])

                if static_ref_path is not None:
                    for name, (ds_shape, ds_dtype) in static_meta.items():
                        number_type, precision = to_xdmf_type(ds_dtype)
                        dims = " ".join(str(int(d)) for d in ds_shape)
                        grids.extend([
                            f'        <Attribute Name="{name}" AttributeType="Scalar" Center="Cell">',
                            (
                                f'          <DataItem Dimensions="{dims}" NumberType="{number_type}" '
                                f'Precision="{precision}" Format="HDF">{static_ref_path}:/{name}</DataItem>'
                            ),
                            "        </Attribute>",
                        ])
                grids.append("      </Grid>")

            series_xdmf_path = output_stem.parent / f"{prefix}_series.xdmf"
            series_lines = [
                '<?xml version="1.0" ?>',
                '<Xdmf Version="3.0">',
                "  <Domain>",
                '    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">',
                *grids,
                "    </Grid>",
                "  </Domain>",
                "</Xdmf>",
                "",
            ]
            series_xdmf_path.write_text("\n".join(series_lines), encoding="utf-8")

    elapsed = time() - start
    if xdmf_path is None:
        logger.info(f"Saved {h5_path} in {elapsed:.6f} seconds.")
    else:
        logger.info(f"Saved {h5_path} and {xdmf_path} in {elapsed:.6f} seconds.")

    return h5_path, xdmf_path


def save_fields_vtk(timestep, fields, output_dir=".", prefix="fields"):
    """
    Save VTK fields to the specified directory.

    Parameters
    ----------
    timestep (int): The timestep number to be associated with the saved fields.

    fields (Dict[str, np.ndarray]): A dictionary of fields to be saved. Each field must be an array-like object
    with dimensions (nx, ny) for 2D fields or (nx, ny, nz) for 3D fields, where:
    - nx : int, number of grid points along the x-axis
    - ny : int, number of grid points along the y-axis
    - nz : int, number of grid points along the z-axis (for 3D fields only)
    The key value for each field in the dictionary must be a string containing the name of the field.

    output_dir (str, optional, default: '.'): The directory in which to save the VTK files. Defaults to the current directory.

    prefix (str, optional, default: 'fields'): A prefix to be added to the filename. Defaults to 'fields'.

    Returns
    -------
    None

    Notes
    -----
    This function saves the VTK fields in the specified directory, with filenames based on the provided timestep number
    and the filename. For example, if the timestep number is 10 and the file name is fields, the VTK file
    will be saved as 'fields_0000010.vtk'in the specified directory.

    """
    pv = _import_optional("pyvista", "save_fields_vtk")
    # Assert that all fields have the same dimensions except for the last dimension assuming fields is a dictionary
    for key, value in fields.items():
        if key == list(fields.keys())[0]:
            dimensions = value.shape
        else:
            assert value.shape == dimensions, "All fields must have the same dimensions!"

    if not os.path.exists(output_dir):
        logger.info(colored("Directory does not exist, creating the directory " + output_dir, "yellow"))
        os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.join(output_dir, prefix + "_" + f"{timestep:07d}.vtk")

    # Add 1 to the dimensions tuple as we store cell values
    dimensions = tuple([dim + 1 for dim in dimensions])

    # Create a uniform grid
    if value.ndim == 2:
        dimensions = dimensions + (1,)

    grid = pv.ImageData(dimensions=dimensions)

    # Add the fields to the grid
    for key, value in fields.items():
        grid[key] = value.flatten(order="F")

    # Save the grid to a VTK file
    start = time()
    grid.save(output_filename, binary=True)
    logger.info(f"Saved {output_filename} in {time() - start:.6f} seconds.")


def live_volume_rendering(timestep, field):
    # WORK IN PROGRESS
    """
    Live rendering of a 3D volume using pyvista.

    Parameters
    ----------
    timestep : int
        Current simulation timestep.
    field : numpy.ndarray
        Three-dimensional field to render.

    Returns
    -------
    None

    Notes
    -----
    This function uses pyvista to render a 3D volume. The volume is rendered with a colormap based on the field values.
    The colormap is updated every 0.1 seconds to reflect changes to the field.

    """
    pv = _import_optional("pyvista", "live_volume_rendering")
    plt = _import_optional("matplotlib.pylab", "live_volume_rendering")
    # Create a uniform grid (Note that the field must be 3D) otherwise raise error
    if field.ndim != 3:
        raise ValueError("The input field must be 3D!")
    dimensions = field.shape
    grid = pv.ImageData(dimensions=dimensions)

    # Add the field to the grid
    grid["field"] = field.flatten(order="F")

    # Create the rendering scene
    if timestep == 0:
        plt.ion()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.title("Live rendering of the field")
        pl = pv.Plotter(off_screen=True)
        pl.add_volume(grid, cmap="nipy_spectral", opacity="sigmoid_10", shade=False)
        plt.imshow(pl.screenshot())

    else:
        pl = pv.Plotter(off_screen=True)
        pl.add_volume(grid, cmap="nipy_spectral", opacity="sigmoid_10", shade=False)
        # Update the rendering scene every 0.1 seconds
        plt.imshow(pl.screenshot())
        plt.pause(0.1)


def live_volume_randering(timestep, field):
    """Call :func:`live_volume_rendering` using the deprecated spelling."""
    warnings.warn(
        "live_volume_randering is deprecated; use live_volume_rendering instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return live_volume_rendering(timestep, field)


def save_BCs_vtk(timestep, BCs, grid_info, output_dir="."):
    """
    Save boundary conditions as VTK format to the specified directory.

    Parameters
    ----------
    timestep (int): The timestep number to be associated with the saved fields.

    BCs (List[BC]): A list of boundary conditions to be saved. Each boundary condition must be an object of type BC.

    Returns
    -------
    None

    Notes
    -----
    This function saves the boundary conditions in the specified directory, with filenames based on the provided timestep number
    and the filename. For example, if the timestep number is 10, the VTK file
    will be saved as 'BCs_0000010.vtk'in the specified directory.
    """
    pv = _import_optional("pyvista", "save_BCs_vtk")

    # Create a uniform grid
    if grid_info["nz"] == 0:
        gridDimensions = (grid_info["nx"] + 1, grid_info["ny"] + 1, 1)
        fieldDimensions = (grid_info["nx"], grid_info["ny"], 1)
    else:
        gridDimensions = (grid_info["nx"] + 1, grid_info["ny"] + 1, grid_info["nz"] + 1)
        fieldDimensions = (grid_info["nx"], grid_info["ny"], grid_info["nz"])

    grid = pv.ImageData(dimensions=gridDimensions)

    # Dictionary to keep track of encountered BC names
    bcNamesCount = {}

    for bc in BCs:
        bcName = bc.name
        if bcName in bcNamesCount:
            bcNamesCount[bcName] += 1
        else:
            bcNamesCount[bcName] = 0
        bcName += f"_{bcNamesCount[bcName]}"

        if bc.is_dynamic:
            bcIndices, _ = bc.update_function(timestep)
        else:
            bcIndices = bc.indices

        # Convert indices to 1D indices
        if grid_info["dim"] == 2:
            bcIndices = np.ravel_multi_index(bcIndices, fieldDimensions[:-1], order="F")
        else:
            bcIndices = np.ravel_multi_index(bcIndices, fieldDimensions, order="F")

        grid[bcName] = np.zeros(fieldDimensions, dtype=bool).flatten(order="F")
        grid[bcName][bcIndices] = True

    # Save the grid to a VTK file
    output_filename = os.path.join(output_dir, "BCs_" + f"{timestep:07d}.vtk")

    start = time()
    grid.save(output_filename, binary=True)
    logger.info(f"Saved {output_filename} in {time() - start:.6f} seconds.")


def rotate_geometry(indices, origin, axis, angle):
    """
    Rotates a voxelized mesh around a given axis.

    Parameters
    ----------
    indices (array-like): The indices of the voxels in the mesh.

    origin (array-like): The coordinates of the origin of the rotation axis.

    axis (array-like): The direction vector of the rotation axis. This should be a 3-element sequence.

    angle (float): The angle by which to rotate the mesh, in radians.

    Returns
    -------
    tuple: The indices of the voxels in the rotated mesh.

    Notes
    -----
    This function rotates the mesh by applying a rotation matrix to the voxel indices. The rotation matrix is calculated
    using the axis-angle representation of rotations. The origin of the rotation axis is assumed to be at (0, 0, 0).
    """
    indices_rotated = (jnp.array(indices).T - origin) @ axangle2mat(axis, angle) + origin
    return tuple(jnp.rint(indices_rotated).astype("int32").T)


def voxelize_stl(stl_filename, length_lbm_unit=None, transformation_matrix=None, pitch=None, **kwargs):
    """
    Converts an STL file to a voxelized mesh.

    Parameters
    ----------
    stl_filename (str): The name of the STL file to be voxelized.

    length_lbm_unit (float, optional): The unit length in LBM. Either this or 'pitch' must be provided.

    transformation_matrix (array-like, optional): A transformation matrix to be applied to the mesh before voxelization.

    pitch : (float, optional): The pitch of the voxel grid. Either this or 'length_lbm_unit' must be provided.

    Returns
    -------
    trimesh.VoxelGrid, float: The voxelized mesh and the pitch of the voxel grid.

    Notes
    -----
    This function uses the trimesh library to load the STL file and voxelized the mesh. If a transformation matrix is
    provided, it is applied to the mesh before voxelization. The pitch of the voxel grid is calculated based on the
    maximum extent of the mesh and the provided lattice Boltzmann unit length, unless a pitch is provided directly.
    """
    legacy_transformation_matrix = kwargs.pop("tranformation_matrix", None)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"voxelize_stl() got an unexpected keyword argument '{unexpected}'")
    if legacy_transformation_matrix is not None:
        if transformation_matrix is not None:
            raise TypeError("Specify only one of 'transformation_matrix' and deprecated 'tranformation_matrix'.")
        warnings.warn(
            "tranformation_matrix is deprecated; use transformation_matrix instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        transformation_matrix = legacy_transformation_matrix

    trimesh = _import_optional("trimesh", "voxelize_stl")
    if length_lbm_unit is None and pitch is None:
        raise ValueError("Either 'length_lbm_unit' or 'pitch' must be provided!")
    mesh = trimesh.load_mesh(stl_filename, process=False)
    length_phys_unit = mesh.extents.max()
    if transformation_matrix is not None:
        mesh.apply_transform(transformation_matrix)
    if pitch is None:
        pitch = length_phys_unit / length_lbm_unit
    mesh_voxelized = mesh.voxelized(pitch=pitch)
    return mesh_voxelized, pitch


def axangle2mat(axis, angle, is_normalized=False):
    """Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis (3 element sequence): vector specifying axis for rotation.

    angle (scalar): angle of rotation in radians.

    is_normalized (bool, optional): True if `axis` is already normalized (has norm of 1).  Default False.

    Returns
    -------
    mat (array shape (3,3)): rotation matrix for specified rotation

    Notes
    -----
    From : https://github.com/matthew-brett/transforms3d
    Ref : http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    x, y, z = axis
    if not is_normalized:
        n = jnp.sqrt(x * x + y * y + z * z)
        x = x / n
        y = y / n
        z = z / n
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    return jnp.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c],
    ])


@partial(jit)
def q_criterion(u):
    """
    Compute the Q-criterion on the interior of a three-dimensional velocity field.

    Parameters
    ----------
    u : jax.Array
        Velocity field with shape ``(nx, ny, nz, 3)`` and unit grid spacing.

    Returns
    -------
    jax.Array
        Q-criterion field with shape ``(nx - 2, ny - 2, nz - 2)``.
    """
    u_x = u[..., 0]
    u_y = u[..., 1]
    u_z = u[..., 2]

    # Compute derivatives
    u_x_dx = (u_x[2:, 1:-1, 1:-1] - u_x[:-2, 1:-1, 1:-1]) / 2
    u_x_dy = (u_x[1:-1, 2:, 1:-1] - u_x[1:-1, :-2, 1:-1]) / 2
    u_x_dz = (u_x[1:-1, 1:-1, 2:] - u_x[1:-1, 1:-1, :-2]) / 2
    u_y_dx = (u_y[2:, 1:-1, 1:-1] - u_y[:-2, 1:-1, 1:-1]) / 2
    u_y_dy = (u_y[1:-1, 2:, 1:-1] - u_y[1:-1, :-2, 1:-1]) / 2
    u_y_dz = (u_y[1:-1, 1:-1, 2:] - u_y[1:-1, 1:-1, :-2]) / 2
    u_z_dx = (u_z[2:, 1:-1, 1:-1] - u_z[:-2, 1:-1, 1:-1]) / 2
    u_z_dy = (u_z[1:-1, 2:, 1:-1] - u_z[1:-1, :-2, 1:-1]) / 2
    u_z_dz = (u_z[1:-1, 1:-1, 2:] - u_z[1:-1, 1:-1, :-2]) / 2

    # Compute vorticity
    mu_x = u_z_dy - u_y_dz
    mu_y = u_x_dz - u_z_dx
    mu_z = u_y_dx - u_x_dy
    norm_mu = jnp.sqrt(mu_x**2 + mu_y**2 + mu_z**2)

    # Compute strain rate
    s_0_0 = u_x_dx
    s_0_1 = 0.5 * (u_x_dy + u_y_dx)
    s_0_2 = 0.5 * (u_x_dz + u_z_dx)
    s_1_0 = s_0_1
    s_1_1 = u_y_dy
    s_1_2 = 0.5 * (u_y_dz + u_z_dy)
    s_2_0 = s_0_2
    s_2_1 = s_1_2
    s_2_2 = u_z_dz
    s_dot_s = s_0_0**2 + s_0_1**2 + s_0_2**2 + s_1_0**2 + s_1_1**2 + s_1_2**2 + s_2_0**2 + s_2_1**2 + s_2_2**2

    # Compute omega
    omega_0_0 = 0.0
    omega_0_1 = 0.5 * (u_x_dy - u_y_dx)
    omega_0_2 = 0.5 * (u_x_dz - u_z_dx)
    omega_1_0 = -omega_0_1
    omega_1_1 = 0.0
    omega_1_2 = 0.5 * (u_y_dz - u_z_dy)
    omega_2_0 = -omega_0_2
    omega_2_1 = -omega_1_2
    omega_2_2 = 0.0
    omega_dot_omega = (
        omega_0_0**2 + omega_0_1**2 + omega_0_2**2 + omega_1_0**2 + omega_1_1**2 + omega_1_2**2 + omega_2_0**2 + omega_2_1**2 + omega_2_2**2
    )

    # Compute q-criterion
    q = 0.5 * (omega_dot_omega - s_dot_s)

    return norm_mu, q
