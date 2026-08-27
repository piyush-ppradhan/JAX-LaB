"""Singlephase lattice Boltzmann simulation structure, later modified for multiphase implementation."""

import logging
import os
import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from jax import jit, lax, vmap, shard_map
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .boundary_conditions import INLET_OUTLET_BC_TYPES, WALL_BC_TYPES, BounceBack, BounceBackHalfway, EquilibriumBC, Regularized, ZouHe
from .precision_policy import PrecisionPolicy
from .utils import colored, downsample_field

logger = logging.getLogger(__name__)


class LBMBase(object):
    """
    Base class for lattice Boltzmann simulations.

    Parameters
    ----------
    lattice (Lattice): Lattice structure and weights.

    omega (float): Relaxation parameter.

    nx, ny (int): Grid points in the x and y directions.

    nz (int): Grid points in the z direction. Use 0 for two-dimensional simulations.

    precision (str): Compute/storage precision policy, such as ``"f32/f32"``.

    checkpoint_rate (int, optional): Timesteps between checkpoints. Defaults to 0, which disables checkpointing.

    checkpoint_dir (str, optional): Checkpoint directory. Defaults to ``"./checkpoints"``.

    downsampling_factor (int, optional): Spatial output downsampling factor. Defaults to 1.

    print_info_rate (int, optional): Timesteps between progress messages. Defaults to 100.

    io_rate (int, optional): Timesteps between output operations. Defaults to 0, which disables output.

    return_fpost (bool, optional): Whether simulation steps return post-collision populations. Defaults to False.

    compute_MLUPS (bool, optional): Whether to run in performance-measurement mode. Defaults to False.

    restore_checkpoint (bool, optional): Whether to restore the latest checkpoint. Defaults to False.
    """

    def __init__(self, **kwargs):
        self.omega = kwargs.get("omega")
        self.nx = kwargs.get("nx")
        self.ny = kwargs.get("ny")
        self.nz = kwargs.get("nz")

        self.precision = kwargs.get("precision")
        self.precision_policy = PrecisionPolicy.from_string(self.precision)

        self.lattice = kwargs.get("lattice")
        self.checkpoint_rate = kwargs.get("checkpoint_rate", 0)
        self.checkpoint_dir = kwargs.get("checkpoint_dir", "./checkpoints")
        self.downsampling_factor = kwargs.get("downsampling_factor", 1)
        self.print_info_rate = kwargs.get("print_info_rate", 100)
        self.io_rate = kwargs.get("io_rate", 0)
        self.return_fpost = kwargs.get("return_fpost", False)
        self.compute_MLUPS = kwargs.get("compute_MLUPS", False)
        self.restore_checkpoint = kwargs.get("restore_checkpoint", False)
        self.n_devices = jax.device_count()
        self.backend = jax.default_backend()

        if self.compute_MLUPS:
            self.restore_checkpoint = False
            self.io_rate = 0
            self.checkpoint_rate = 0
            self.print_info_rate = 0

        # Check for distributed mode
        if self.n_devices > jax.local_device_count():
            warnings.warn(
                "Running in distributed mode. Call jax.distributed.initialize before performing JAX computations.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.c = self.lattice.c
        self.q = self.lattice.q
        self.w = self.lattice.w
        self.dim = self.lattice.d

        # Set the checkpoint manager
        if self.checkpoint_rate > 0:
            mngr_options = orb.CheckpointManagerOptions(save_interval_steps=self.checkpoint_rate, max_to_keep=1)
            # self.mngr = orb.CheckpointManager(self.checkpoint_dir, orb.PyTreeCheckpointer(), options=mngr_options)
            self.mngr = orb.CheckpointManager(self.checkpoint_dir, options=mngr_options)
        else:
            self.mngr = None

        # Adjust the number of grid points in the x direction, if necessary.
        # If the number of grid points is not divisible by the number of devices
        # it increases the number of grid points to the next multiple of the number of devices.
        # This is done in order to accommodate the domain sharding per XLA device
        nx, ny, nz = kwargs.get("nx"), kwargs.get("ny"), kwargs.get("nz")
        if None in {nx, ny, nz}:
            raise ValueError("nx, ny, and nz must be provided. For 2D examples, nz must be set to 0.")
        self.nx = nx
        if nx % self.n_devices:
            self.nx = nx + (self.n_devices - nx % self.n_devices)
            warnings.warn(
                f"nx increased from {nx} to {self.nx} to accommodate domain sharding per XLA device.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.ny = ny
        self.nz = nz

        # self.show_simulation_parameters()

        # Store grid information
        self.grid_info = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dim": self.lattice.d,
            "lattice": self.lattice,
        }

        P = PartitionSpec

        # Define the right permutation
        self.right_perm = [(i, (i + 1) % self.n_devices) for i in range(self.n_devices)]
        # Define the left permutation
        self.left_perm = [((i + 1) % self.n_devices, i) for i in range(self.n_devices)]

        # Set up the sharding and streaming for 2D simulations
        if self.dim == 2:
            self.devices = mesh_utils.create_device_mesh((self.n_devices, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "value"))
            self.sharding = NamedSharding(self.mesh, P("x", "y", "value"))

            self.streaming = jit(
                shard_map(
                    self.streaming_m,
                    mesh=self.mesh,
                    in_specs=P("x", None, None),
                    out_specs=P("x", None, None),
                    check_vma=False,
                )
            )
            self.local_bounceback = jit(
                shard_map(
                    self.local_bounceback_m,
                    mesh=self.mesh,
                    in_specs=(P("x", None, None), P("x", None, None), P("x", None, None)),
                    out_specs=P("x", None, None),
                    check_vma=False,
                )
            )
            self.local_wall_bc_kernels = self._build_local_wall_kernels(field_spec=P("x", None, None))
            self.local_inlet_outlet_kernels = self._build_local_inlet_outlet_kernels(field_spec=P("x", None, None))
            self.local_equilibrium_bc = jit(
                shard_map(
                    self.local_equilibrium_bc_m,
                    mesh=self.mesh,
                    in_specs=(P("x", None, None), P("x", None, None), P("x", None, None)),
                    out_specs=P("x", None, None),
                    check_vma=False,
                )
            )
            self.local_solid_pin = jit(
                shard_map(
                    self.local_solid_pin_m,
                    mesh=self.mesh,
                    in_specs=(P("x", None, None), P("x", None, None)),
                    out_specs=P("x", None, None),
                    check_vma=False,
                )
            )

        # Set up the sharding and streaming for 3D simulations
        elif self.dim == 3:
            self.devices = mesh_utils.create_device_mesh((self.n_devices, 1, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "z", "value"))
            self.sharding = NamedSharding(self.mesh, P("x", "y", "z", "value"))

            self.streaming = jit(
                shard_map(
                    self.streaming_m,
                    mesh=self.mesh,
                    in_specs=P("x", None, None, None),
                    out_specs=P("x", None, None, None),
                    check_vma=False,
                )
            )
            self.local_bounceback = jit(
                shard_map(
                    self.local_bounceback_m,
                    mesh=self.mesh,
                    in_specs=(P("x", None, None, None), P("x", None, None, None), P("x", None, None)),
                    out_specs=P("x", None, None, None),
                    check_vma=False,
                )
            )
            self.local_wall_bc_kernels = self._build_local_wall_kernels(field_spec=P("x", None, None, None))
            self.local_inlet_outlet_kernels = self._build_local_inlet_outlet_kernels(field_spec=P("x", None, None, None))
            self.local_equilibrium_bc = jit(
                shard_map(
                    self.local_equilibrium_bc_m,
                    mesh=self.mesh,
                    in_specs=(P("x", None, None, None), P("x", None, None), P("x", None, None)),
                    out_specs=P("x", None, None, None),
                    check_vma=False,
                )
            )
            self.local_solid_pin = jit(
                shard_map(
                    self.local_solid_pin_m,
                    mesh=self.mesh,
                    in_specs=(P("x", None, None, None), P("x", None, None)),
                    out_specs=P("x", None, None, None),
                    check_vma=False,
                )
            )

        else:
            raise ValueError(f"dim = {self.dim} not supported")

        # Compute the bounding box indices for boundary conditions
        self.bounding_box_indices = self.compute_bounding_box_indices()
        # Create boundary data for the simulation
        self._create_boundary_data()
        # Local (per-shard, int32) BounceBack indices, used by apply_bc's fast bounce-back path.
        self.local_bounceback_indices = self._make_local_bounceback_indices()
        # Local (per-shard, int32) data for BounceBackHalfway/InterpolatedBounceBack* and the solid-node pin,
        # used by apply_bc's fast wall boundary condition path.
        self.wall_bc_data, self.solid_pin_indices = self._make_local_wall_bc_data()
        # Local (per-shard, int32) data for ZouHe/Regularized and EquilibriumBC, used by apply_bc's fast
        # inlet/outlet boundary condition path. These BCs only read/write their own node's populations (no
        # neighbor access), unlike ExtrapolationOutflow/ConvectiveOutflow which are left on the generic
        # global-index path.
        self.inlet_outlet_bc_data = self._make_local_inlet_outlet_data()
        self.local_equilibrium_bc_indices, self.local_equilibrium_bc_values = self._make_local_equilibrium_bc_data()
        self.force = self.get_force()

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, value):
        if value is None:
            raise ValueError("Lattice type must be provided.")
        if self.nz == 0 and value.name not in ["D2Q9"]:
            raise ValueError("For 2D simulations, lattice type must be LatticeD2Q9.")
        if self.nz != 0 and value.name not in ["D3Q19", "D3Q15", "D3Q27"]:
            raise ValueError("For 3D simulations, lattice type must be LatticeD3Q19, or LatticeD3Q27.")

        self._lattice = value

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if value is None:
            raise ValueError("omega must be provided")
        self._omega = value

    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, value):
        if value is None:
            raise ValueError("nx must be provided")
        if not isinstance(value, int):
            raise TypeError("nx must be an integer")
        self._nx = value

    @property
    def ny(self):
        return self._ny

    @ny.setter
    def ny(self, value):
        if value is None:
            raise ValueError("ny must be provided")
        if not isinstance(value, int):
            raise TypeError("ny must be an integer")
        self._ny = value

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, value):
        if value is None:
            raise ValueError("nz must be provided")
        if not isinstance(value, int):
            raise TypeError("nz must be an integer")
        self._nz = value

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        if not isinstance(value, str):
            raise TypeError("precision must be a string")
        self._precision = value

    @property
    def checkpoint_rate(self):
        return self._checkpoint_rate

    @checkpoint_rate.setter
    def checkpoint_rate(self, value):
        if not isinstance(value, int):
            raise TypeError("checkpoint_rate must be an integer")
        self._checkpoint_rate = value

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, value):
        if not isinstance(value, str):
            raise TypeError("checkpoint_dir must be a string")
        self._checkpoint_dir = value

    @property
    def downsampling_factor(self):
        return self._downsampling_factor

    @downsampling_factor.setter
    def downsampling_factor(self, value):
        if not isinstance(value, int):
            raise TypeError("downsampling_factor must be an integer")
        self._downsampling_factor = value

    @property
    def print_info_rate(self):
        return self._print_info_rate

    @print_info_rate.setter
    def print_info_rate(self, value):
        if not isinstance(value, int):
            raise TypeError("print_info_rate must be an integer")
        self._print_info_rate = value

    @property
    def io_rate(self):
        return self._io_rate

    @io_rate.setter
    def io_rate(self, value):
        if not isinstance(value, int):
            raise TypeError("io_rate must be an integer")
        self._io_rate = value

    @property
    def return_fpost(self):
        return self._return_fpost

    @return_fpost.setter
    def return_fpost(self, value):
        if not isinstance(value, bool):
            raise TypeError("return_fpost must be a boolean")
        self._return_fpost = value

    @property
    def compute_MLUPS(self):
        return self._compute_MLUPS

    @compute_MLUPS.setter
    def compute_MLUPS(self, value):
        if not isinstance(value, bool):
            raise TypeError("compute_MLUPS must be a boolean")
        self._compute_MLUPS = value

    @property
    def restore_checkpoint(self):
        return self._restore_checkpoint

    @restore_checkpoint.setter
    def restore_checkpoint(self, value):
        if not isinstance(value, bool):
            raise TypeError("restore_checkpoint must be a boolean")
        self._restore_checkpoint = value

    @property
    def n_devices(self):
        return self._n_devices

    @n_devices.setter
    def n_devices(self, value):
        if not isinstance(value, int):
            raise TypeError("n_devices must be an integer")
        self._n_devices = value

    def show_simulation_parameters(self):
        attributes_to_show = [
            "omega",
            "nx",
            "ny",
            "nz",
            "dim",
            "precision",
            "lattice",
            "checkpoint_rate",
            "checkpoint_dir",
            "downsampling_factor",
            "print_info_rate",
            "io_rate",
            "compute_MLUPS",
            "restore_checkpoint",
            "backend",
            "n_devices",
        ]

        descriptive_names = {
            "omega": "Omega",
            "nx": "Grid Points in X",
            "ny": "Grid Points in Y",
            "nz": "Grid Points in Z",
            "dim": "Dimensionality",
            "precision": "Precision Policy",
            "lattice": "Lattice Type",
            "checkpoint_rate": "Checkpoint Rate",
            "checkpoint_dir": "Checkpoint Directory",
            "downsampling_factor": "Downsampling Factor",
            "print_info_rate": "Print Info Rate",
            "io_rate": "I/O Rate",
            "compute_MLUPS": "Compute MLUPS",
            "restore_checkpoint": "Restore Checkpoint",
            "backend": "Backend",
            "n_devices": "Number of Devices",
        }
        simulation_name = self.__class__.__name__

        logger.info(colored(f"**** Simulation Parameters for {simulation_name} ****", "green"))

        header = f"{colored('Parameter', 'blue'):>30} | {colored('Value', 'yellow')}"
        logger.info(header)
        logger.info("-" * 50)

        for attr in attributes_to_show:
            value = getattr(self, attr, "Attribute not set")
            descriptive_name = descriptive_names.get(attr, attr)  # Use the attribute name as a fallback
            row = f"{colored(descriptive_name, 'blue'):>30} | {colored(value, 'yellow')}"
            logger.info(row)

    def _create_boundary_data(self):
        """
        Create boundary data for the Lattice Boltzmann simulation by setting boundary conditions,
        creating grid mask, and preparing local masks and normal arrays.
        """
        self.BCs = []
        self.set_boundary_conditions()
        # Accumulate the indices of all BCs to create the grid mask with FALSE along directions that
        # stream into a boundary voxel.
        solid_halo_list = [np.array(bc.indices).T for bc in self.BCs if bc.is_solid]
        solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

        # Create the grid mask on each process
        start = time.time()
        grid_mask = self.create_grid_mask(solid_halo_voxels)
        logger.info("Time to create the grid mask: %.6f seconds", time.time() - start)

        start = time.time()
        for bc in self.BCs:
            assert bc.implementation_step in ["PostStreaming", "PostCollision"]
            bc.create_local_mask_and_normal_arrays(grid_mask)
        logger.info("Time to create the local masks and normal arrays: %.6f seconds", time.time() - start)

    @partial(jit, static_argnums=(0, 1, 2, 4))
    def distributed_array_init(self, shape, ttype, init_val=0, sharding=None):
        """
        Initialize a distributed array using JAX, with a specified shape, data type, and initial value.
        Optionally, provide a custom sharding strategy.

        Parameters
        ----------
        shape (tuple): The shape of the array to be created.

        ttype (dtype): The data type of the array to be created.

        init_val (scalar, optional): The initial value to fill the array with. Defaults to 0.

        sharding (Sharding, optional): The sharding strategy to use. Defaults to `self.sharding`.

        Returns
        -------
        jax.numpy.ndarray: A JAX array with the specified shape, data type, initial value, and sharding strategy.
        """
        if sharding is None:
            sharding = self.sharding
        x = jnp.full(shape=shape, fill_value=init_val, dtype=ttype)
        return jax.lax.with_sharding_constraint(x, sharding)

    @partial(jit, static_argnums=(0,))
    def create_grid_mask(self, solid_halo_voxels):
        """
        This function creates a mask for the background grid that accounts for the location of the boundaries.

        Parameters
        ----------
        solid_halo_voxels (numpy.ndarray): A NumPy array representing the voxels in the halo of the solid object.

        Returns
        -------
        A JAX array representing the grid mask of the grid.
        """
        # Halo width (hw_x is different to accommodate the domain sharding per XLA device)
        hw_x = self.n_devices
        hw_y = hw_z = 1
        if self.dim == 2:
            grid_mask = self.distributed_array_init((self.nx + 2 * hw_x, self.ny + 2 * hw_y, self.lattice.q), jnp.bool_, init_val=True)
            grid_mask = grid_mask.at[(slice(hw_x, -hw_x), slice(hw_y, -hw_y), slice(None))].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)

            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

        elif self.dim == 3:
            grid_mask = self.distributed_array_init(
                (
                    self.nx + 2 * hw_x,
                    self.ny + 2 * hw_y,
                    self.nz + 2 * hw_z,
                    self.lattice.q,
                ),
                jnp.bool_,
                init_val=True,
            )
            grid_mask = grid_mask.at[
                (
                    slice(hw_x, -hw_x),
                    slice(hw_y, -hw_y),
                    slice(hw_z, -hw_z),
                    slice(None),
                )
            ].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                solid_halo_voxels = solid_halo_voxels.at[:, 2].add(hw_z)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)
            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

    def compute_bounding_box_indices(self):
        """
        This function calculates the indices of the bounding box of a 2D or 3D grid.
        The bounding box is defined as the set of grid points on the outer edge of the grid.

        Returns
        -------
        boundingBox (dict): A dictionary where keys are the names of the bounding box faces
        ("bottom", "top", "left", "right" for 2D; additional "front", "back" for 3D), and values
        are numpy arrays of indices corresponding to each face.
        """
        if self.dim == 2:
            # For a 2D grid, the bounding box consists of four edges: bottom, top, left, and right.
            # Each edge is represented as an array of indices. For example, the bottom edge includes
            # all points where the y-coordinate is 0, so its indices are [[i, 0] for i in range(self.nx)].
            bounding_box = {
                "bottom": np.array([[i, 0] for i in range(self.nx)], dtype=int),
                "top": np.array([[i, self.ny - 1] for i in range(self.nx)], dtype=int),
                "left": np.array([[0, i] for i in range(self.ny)], dtype=int),
                "right": np.array([[self.nx - 1, i] for i in range(self.ny)], dtype=int),
            }

            return bounding_box

        elif self.dim == 3:
            # For a 3D grid, the bounding box consists of six faces: bottom, top, left, right, front, and back.
            # Each face is represented as an array of indices. For example, the bottom face includes all points
            # where the z-coordinate is 0, so its indices are [[i, j, 0] for i in range(self.nx) for j in range(self.ny)].
            bounding_box = {
                "bottom": np.array(
                    [[i, j, 0] for i in range(self.nx) for j in range(self.ny)],
                    dtype=int,
                ),
                "top": np.array(
                    [[i, j, self.nz - 1] for i in range(self.nx) for j in range(self.ny)],
                    dtype=int,
                ),
                "left": np.array(
                    [[0, j, k] for j in range(self.ny) for k in range(self.nz)],
                    dtype=int,
                ),
                "right": np.array(
                    [[self.nx - 1, j, k] for j in range(self.ny) for k in range(self.nz)],
                    dtype=int,
                ),
                "front": np.array(
                    [[i, 0, k] for i in range(self.nx) for k in range(self.nz)],
                    dtype=int,
                ),
                "back": np.array(
                    [[i, self.ny - 1, k] for i in range(self.nx) for k in range(self.nz)],
                    dtype=int,
                ),
            }

            return bounding_box

    def initialize_macroscopic_fields(self):
        """
        This function initializes the macroscopic fields (density and velocity) to their default values.
        The default density is 1 and the default velocity is 0.

        Note: This function is a placeholder and should be overridden in a subclass or in an instance of the class
        to provide specific initial conditions.

        Returns
        -------
        None, None: The default density and velocity, both None. This indicates that the actual values should be set elsewhere.
        """
        warnings.warn(
            "Default fluid initial conditions assumed: density = 1 and velocity = 0. Override initialize_macroscopic_fields to set explicit values.",
            UserWarning,
            stacklevel=2,
        )
        return None, None

    def assign_fields_sharded(self):
        """
        This function is used to initialize the simulation by assigning the macroscopic fields and populations.

        The function first initializes the macroscopic fields, which are the density (rho0) and velocity (u0).
        Depending on the dimension of the simulation (2D or 3D), it then sets the shape of the array that will hold the
        distribution functions (f).

        If the density or velocity are not provided, the function initializes the distribution functions with a default
        value (self.w), representing density=1 and velocity=0. Otherwise, it uses the provided density and velocity to initialize the populations.

        Returns
        -------
        f: a distributed JAX array of shape (nx, ny, nz, q) or (nx, ny, q) holding the distribution functions for the simulation.
        """
        rho0, u0 = self.initialize_macroscopic_fields()

        if self.dim == 2:
            shape = (self.nx, self.ny, self.lattice.q)
        if self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.lattice.q)

        if rho0 is None or u0 is None:
            f = self.distributed_array_init(shape, self.precision_policy.output_dtype, init_val=self.w)
        else:
            f = self.initialize_populations(rho0, u0)

        return f

    def initialize_populations(self, rho0, u0):
        """
        This function initializes the populations (distribution functions) for the simulation.
        It uses the equilibrium distribution function, which is a function of the macroscopic
        density and velocity.

        Parameters
        ----------
        rho0 (jax.numpy.ndarray): Initial density field.

        u0 (jax.numpy.ndarray): Initial velocity field.

        Returns
        -------
        f (jax.numpy.ndarray): The array holding the initialized distribution functions for the simulation.
        """
        return self.equilibrium(rho0, u0)

    def send_right(self, x, axis_name):
        """
        This function sends the data to the right neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x (jax.numpy.ndarray): The data to be sent.

        axis_name (str): The name of the axis along which the data is sent.

        Returns
        -------
        (jax.numpy.ndarray): The data after being sent to the right neighboring process.
        """
        return lax.ppermute(x, perm=self.right_perm, axis_name=axis_name)

    def send_left(self, x, axis_name):
        """
        This function sends the data to the left neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x (jax.numpy.ndarray): The data to be sent.

        axis_name (str): The name of the axis along which the data is sent.

        Returns
        -------
        The data after being sent to the left neighboring process.
        """
        return lax.ppermute(x, perm=self.left_perm, axis_name=axis_name)

    def streaming_m(self, f):
        """
        This function performs the streaming step in the Lattice Boltzmann Method, which is
        the propagation of the distribution functions in the lattice.

        To enable multi-GPU/TPU functionality, it extracts the left and right boundary slices of the
        distribution functions that need to be communicated to the neighboring processes.

        The function then sends the left boundary slice to the right neighboring process and the right
        boundary slice to the left neighboring process. The received data is then set to the
        corresponding indices in the receiving domain.

        Parameters
        ----------
        f (jax.numpy.ndarray): The array holding the distribution functions for the simulation.

        Returns
        -------
        (jax.numpy.ndarray): The distribution functions after the streaming operation.
        """
        f = self.streaming_p(f)
        left_comm, right_comm = (
            f[:1, ..., self.lattice.right_indices],
            f[-1:, ..., self.lattice.left_indices],
        )

        left_comm, right_comm = (
            self.send_right(left_comm, "x"),
            self.send_left(right_comm, "x"),
        )
        f = f.at[:1, ..., self.lattice.right_indices].set(left_comm)
        f = f.at[-1:, ..., self.lattice.left_indices].set(right_comm)
        return f

    @partial(jit, static_argnums=(0,))
    def streaming_p(self, f):
        """
        Perform streaming operation on a partitioned (in the x-direction) distribution function.

        The function uses the vmap operation provided by the JAX library to vectorize the computation
        over all lattice directions.

        Parameters
        ----------
        f (jax.numpy.ndarray): The distribution function.

        Returns
        -------
        The updated distribution function after streaming.
        """

        def streaming_i(f, c):
            """
            Perform individual streaming operation in a direction.

            Parameters
            ----------
                f (jax.numpy.ndarray): The distribution function.

                c (jax.numpy.ndarray): The streaming direction vector.

            Returns
            -------
                jax.numpy.ndarray
                The updated distribution function after streaming.
            """
            if self.dim == 2:
                return jnp.roll(f, (c[0], c[1]), axis=(0, 1))
            elif self.dim == 3:
                return jnp.roll(f, (c[0], c[1], c[2]), axis=(0, 1, 2))

        return vmap(streaming_i, in_axes=(-1, 0), out_axes=-1)(f, self.c.T)

    def local_bounceback_m(self, fout, fin, local_indices):
        """
        Apply full-way bounce-back using local (per-shard, int32) solid indices instead of the global index list
        baked into every device's compiled program. Dimension-generic: works for 2D and 3D since the index tuple
        length is derived from self.dim.

        Padded index rows point one position outside the local shard (see _collect_bounceback_indices), so
        out-of-shard reads return 0.0 (mode="fill") and out-of-shard writes are dropped (mode="drop"), making the
        padding a safe no-op.

        Parameters
        ----------
        fout (jax.numpy.ndarray): Local shard of the post-collision distribution functions.

        fin (jax.numpy.ndarray): Local shard of the pre-collision distribution functions.

        local_indices (jax.numpy.ndarray): Local shard of padded solid-node indices, shape (1, n_local, dim).

        Returns
        -------
        (jax.numpy.ndarray): Local shard of fout with bounce-back applied at the solid nodes.
        """
        local_indices = local_indices[0]
        idx = tuple(local_indices[:, axis] for axis in range(self.dim))
        bounced = fin.at[idx].get(mode="fill", fill_value=0.0)[..., self.lattice.opp_indices]
        return fout.at[idx].set(bounced, mode="drop")

    def local_wall_bc_m(self, fout, fin, local_indices, local_imissing, local_iknown, local_vel, local_weights, apply_fn):
        """
        Apply a wall boundary condition (see WALL_BC_TYPES) using local (per-shard, int32) fluid-node indices
        and aligned auxiliary data, instead of the global index/imissing/iknown/weights arrays baked into every
        device's compiled program. Dimension-generic. apply_fn is a static (non-traced) Python callable, bound
        via functools.partial before this body is wrapped in shard_map, so it selects the compiled formula
        without adding a traced argument (see _build_local_wall_kernel).

        Padded index rows point one position outside the local shard, so out-of-shard reads return 0.0
        (mode="fill") and out-of-shard writes are dropped (mode="drop"); the padded rows of local_imissing,
        local_iknown, local_vel and local_weights are zero-filled, which is a safe no-op input for that padding.

        Parameters
        ----------
        fout (jax.numpy.ndarray): Local shard of the post-streaming distribution functions.

        fin (jax.numpy.ndarray): Local shard of the post-collision distribution functions.

        local_indices (jax.numpy.ndarray): Local shard of padded fluid-node indices, shape (1, n_local, dim).

        local_imissing, local_iknown (jax.numpy.ndarray): Local shard of padded missing/known direction
            indices, shape (1, n_local, q).

        local_vel (jax.numpy.ndarray): Local shard of padded prescribed velocities, shape (1, n_local, dim).

        local_weights (jax.numpy.ndarray): Local shard of padded interpolation weights, shape (1, n_local, q).
            Unused when apply_fn ignores it (see WALL_BC_TYPES), but always present for a uniform signature.

        apply_fn (callable): One of _halfway_wall_math, _bouzidi_wall_math, _differentiable_wall_math.

        Returns
        -------
        (jax.numpy.ndarray): Local shard of fout with the wall boundary condition applied at the fluid nodes.
        """
        local_indices = local_indices[0]
        idx = tuple(local_indices[:, axis] for axis in range(self.dim))
        fout_bd = fout.at[idx].get(mode="fill", fill_value=0.0)
        fin_bd = fin.at[idx].get(mode="fill", fill_value=0.0)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        fbd = apply_fn(fout_bd, fin_bd, local_imissing[0], local_iknown[0], local_vel[0], local_weights[0], self.lattice.w, c)
        return fout.at[idx].set(fbd, mode="drop")

    def local_inlet_bc_m(
        self, fout, local_indices, local_normals, local_imiddle_mask, local_iknown_mask, local_imissing, local_iknown, local_prescribed, apply_fn
    ):
        """
        Apply a ZouHe/Regularized inlet-outlet boundary condition (see INLET_OUTLET_BC_TYPES) using local
        (per-shard, int32) fluid-node indices and aligned auxiliary data, instead of the global index/mask/
        prescribed arrays baked into every device's compiled program. Unlike wall boundary conditions, these
        only read/write the node's own post-streaming populations (no fin/streaming-neighbor data), so a single
        fout argument suffices. Dimension-generic. apply_fn is a static (non-traced) Python callable, bound via
        functools.partial before this body is wrapped in shard_map (see _build_local_inlet_outlet_kernels).

        Padded index rows point one position outside the local shard, so out-of-shard reads return 0.0
        (mode="fill") and out-of-shard writes are dropped (mode="drop"); the padded rows of the auxiliary
        arrays are zero-filled, which is a safe no-op input for that padding.

        Parameters
        ----------
        fout (jax.numpy.ndarray): Local shard of the post-streaming distribution functions.

        local_indices (jax.numpy.ndarray): Local shard of padded fluid-node indices, shape (1, n_local, dim).

        local_normals (jax.numpy.ndarray): Local shard of padded boundary normals, shape (1, n_local, dim).

        local_imiddle_mask, local_iknown_mask (jax.numpy.ndarray): Local shard of padded middle/known direction
            boolean masks, shape (1, n_local, q).

        local_imissing, local_iknown (jax.numpy.ndarray): Local shard of padded missing/known direction
            indices, shape (1, n_local, q).

        local_prescribed (jax.numpy.ndarray): Local shard of padded prescribed values (velocity or density),
            shape (1, n_local, dim) or (1, n_local, 1).

        apply_fn (callable): One of _zouhe_velocity_math, _zouhe_pressure_math, _regularized_velocity_math,
            _regularized_pressure_math.

        Returns
        -------
        (jax.numpy.ndarray): Local shard of fout with the inlet/outlet boundary condition applied at the fluid
        nodes.
        """
        local_indices = local_indices[0]
        idx = tuple(local_indices[:, axis] for axis in range(self.dim))
        fpop = fout.at[idx].get(mode="fill", fill_value=0.0)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cc = jnp.array(self.lattice.cc, dtype=self.precision_policy.compute_dtype)
        fbd = apply_fn(
            fpop,
            local_prescribed[0],
            local_normals[0],
            local_imiddle_mask[0],
            local_iknown_mask[0],
            local_imissing[0],
            local_iknown[0],
            self.lattice.w,
            c,
            cc,
            self.dim,
        )
        return fout.at[idx].set(fbd, mode="drop")

    def local_equilibrium_bc_m(self, fout, local_indices, local_out):
        """
        Apply EquilibriumBC using local (per-shard, int32) fluid-node indices and the precomputed per-node
        equilibrium values, instead of the global index list baked into every device's compiled program.
        EquilibriumBC.apply ignores its fout/fin arguments (it just returns a precomputed constant per node), so
        this is a pure scatter with no gather needed.

        Parameters
        ----------
        fout (jax.numpy.ndarray): Local shard of the post-streaming distribution functions.

        local_indices (jax.numpy.ndarray): Local shard of padded fluid-node indices, shape (1, n_local, dim).

        local_out (jax.numpy.ndarray): Local shard of padded precomputed equilibrium values, shape
            (1, n_local, q).

        Returns
        -------
        (jax.numpy.ndarray): Local shard of fout with the equilibrium values set at the fluid nodes.
        """
        local_indices = local_indices[0]
        idx = tuple(local_indices[:, axis] for axis in range(self.dim))
        return fout.at[idx].set(local_out[0], mode="drop")

    def local_solid_pin_m(self, fout, local_indices):
        """
        Pin local (per-shard, int32) solid-node populations to the rest equilibrium after streaming, using local
        indices instead of the global solid-index list baked into every device's compiled program (see
        BounceBackHalfway.prepare_populations).

        Parameters
        ----------
        fout (jax.numpy.ndarray): Local shard of the post-streaming distribution functions.

        local_indices (jax.numpy.ndarray): Local shard of padded solid-node indices, shape (1, n_local, dim).

        Returns
        -------
        (jax.numpy.ndarray): Local shard of fout with solid nodes pinned to the rest equilibrium.
        """
        local_indices = local_indices[0]
        idx = tuple(local_indices[:, axis] for axis in range(self.dim))
        return fout.at[idx].set(self.precision_policy.cast_to_output(self.w), mode="drop")

    def _build_local_wall_kernels(self, field_spec):
        """
        Build one jitted shard_map callable per concrete wall boundary condition type in WALL_BC_TYPES, all
        sharing local_wall_bc_m's body with their own formula bound via functools.partial (a static Python
        callable, not a traced argument, so this produces distinct compiled kernels without an extra runtime
        argument or a Python-level branch inside the traced body).

        Parameters
        ----------
        field_spec (jax.sharding.PartitionSpec): Sharding of fout/fin, dimension-dependent (2D or 3D).

        Returns
        -------
        (dict): Maps each WALL_BC_TYPES class to its jitted shard_map callable.
        """
        P = PartitionSpec
        aux_spec = P("x", None, None)
        return {
            bc_type: jit(
                shard_map(
                    partial(self.local_wall_bc_m, apply_fn=apply_fn),
                    mesh=self.mesh,
                    in_specs=(field_spec, field_spec, aux_spec, aux_spec, aux_spec, aux_spec, aux_spec),
                    out_specs=field_spec,
                    check_vma=False,
                )
            )
            for bc_type, apply_fn, _ in WALL_BC_TYPES
        }

    def _build_local_inlet_outlet_kernels(self, field_spec):
        """
        Build one jitted shard_map callable per (concrete type, prescribed-value type) pair in
        INLET_OUTLET_BC_TYPES, all sharing local_inlet_bc_m's body with their own formula bound via
        functools.partial (a static Python callable, not a traced argument).

        Parameters
        ----------
        field_spec (jax.sharding.PartitionSpec): Sharding of fout, dimension-dependent (2D or 3D).

        Returns
        -------
        (dict): Maps each (bc_type, ttype) pair in INLET_OUTLET_BC_TYPES to its jitted shard_map callable.
        """
        P = PartitionSpec
        aux_spec = P("x", None, None)
        return {
            (bc_type, ttype): jit(
                shard_map(
                    partial(self.local_inlet_bc_m, apply_fn=apply_fn),
                    mesh=self.mesh,
                    in_specs=(field_spec, aux_spec, aux_spec, aux_spec, aux_spec, aux_spec, aux_spec, aux_spec),
                    out_specs=field_spec,
                    check_vma=False,
                )
            )
            for bc_type, ttype, apply_fn in INLET_OUTLET_BC_TYPES
        }

    def _split_local_indices(self, indices, *aux_arrays):
        """
        Split global node indices (and any per-row auxiliary data aligned with them) by x-shard, padding each
        device's list to a common length. Shared by every local (per-shard, int32) boundary condition path -
        full-way BounceBack (no auxiliary data) and the wall boundary conditions in WALL_BC_TYPES (imissing,
        iknown, vel, and optionally interpolation weights).

        Parameters
        ----------
        indices (numpy.ndarray): Global node coordinates, shape (n, dim).

        *aux_arrays (numpy.ndarray): Per-row data aligned with indices, each shape (n, ...).

        Returns
        -------
        (numpy.ndarray or None, tuple of numpy.ndarray or None): Padded local indices, shape
        (n_devices, max_local, dim), and one padded local array per aux_arrays entry, shape
        (n_devices, max_local, ...) zero-filled at padded rows. Both are None if indices is empty.
        """
        if len(indices) == 0:
            return None, tuple(None for _ in aux_arrays)
        local_nx = self.nx // self.n_devices
        owner = indices[:, 0] // local_nx
        by_device = [(indices[owner == device], [aux[owner == device] for aux in aux_arrays]) for device in range(self.n_devices)]
        max_local = max(len(idx) for idx, _ in by_device)
        if max_local == 0:
            return None, tuple(None for _ in aux_arrays)

        local_indices = np.zeros((self.n_devices, max_local, self.dim), dtype=np.int32)
        local_indices[..., 0] = local_nx
        local_aux = [np.zeros((self.n_devices, max_local, *aux.shape[1:]), dtype=aux.dtype) for aux in aux_arrays]
        for device, (idx, auxs) in enumerate(by_device):
            count = len(idx)
            local_indices[device, :count] = idx
            local_indices[device, :count, 0] -= device * local_nx
            for a, aux in enumerate(auxs):
                local_aux[a][device, :count] = aux
        return local_indices, tuple(local_aux)

    def _distribute_local(self, array, dtype=None):
        """
        Distribute a padded local (per-shard) host array sharded along x, blocking until it is ready.

        Parameters
        ----------
        array (numpy.ndarray or None): Padded local array, shape (n_devices, max_local, ...), or None.

        dtype (jax.numpy.dtype, optional): Target dtype. Defaults to array's own dtype, so floating-point
            auxiliary data (vel, weights) is distributed at whatever precision apply()'s global path already
            uses it at, rather than being independently rounded to a different precision.

        Returns
        -------
        (jax.numpy.ndarray or None): Distributed array, or None if array is None.
        """
        if array is None:
            return None
        if dtype is None:
            dtype = array.dtype
        sharding = NamedSharding(self.mesh, PartitionSpec("x", *([None] * (array.ndim - 1))))
        distributed = self.distributed_array_init(array.shape, dtype, init_val=array, sharding=sharding)
        distributed.block_until_ready()
        return distributed

    def _collect_bounceback_indices(self, BCs):
        """
        Build padded local (per-shard, int32) solid-node indices for every full-way BounceBack boundary
        condition in BCs, split by x-shard.

        Parameters
        ----------
        BCs (list): Boundary conditions for one component (or the whole simulation for single-phase).

        Returns
        -------
        (numpy.ndarray or None): Padded local indices, shape (n_devices, max_local, dim), or None if BCs
        contains no BounceBack instances.
        """
        bounceback_indices = [np.asarray(bc.indices, dtype=np.int32).T for bc in BCs if isinstance(bc, BounceBack)]
        if not bounceback_indices:
            return None
        local_indices, _ = self._split_local_indices(np.vstack(bounceback_indices))
        return local_indices

    def _make_local_bounceback_indices(self):
        """
        Distribute the padded local BounceBack indices for this simulation's boundary conditions.

        Returns
        -------
        (jax.numpy.ndarray or None): Distributed local indices sharded along x, or None if there is no
        BounceBack boundary condition.

        Notes
        -----
        Overridden by Multiphase to return one such array per component.
        """
        return self._distribute_local(self._collect_bounceback_indices(self.BCs), jnp.int32)

    def _collect_wall_bc_data(self, BCs, bc_type, has_weights):
        """
        Build padded local (per-shard, int32) fluid-node indices and aligned auxiliary data (imissing, iknown,
        vel, and optionally interpolation weights) for every boundary condition of exactly bc_type in BCs.

        Parameters
        ----------
        BCs (list): Boundary conditions for one component (or the whole simulation for single-phase).

        bc_type (type): Exact wall boundary condition class to match (subclasses are matched by their own entry
            in WALL_BC_TYPES instead, so each concrete formula gets its own local kernel).

        has_weights (bool): Whether bc_type carries per-node interpolation weights (set_proximity_ratio is
            called eagerly here if a matching boundary condition hasn't computed them yet).

        Returns
        -------
        (numpy.ndarray or None, tuple): Padded local indices and (imissing, iknown, vel, weights) padded local
        arrays (weights is None when has_weights is False). All None if BCs has no matching boundary condition.
        """
        matches = [bc for bc in BCs if type(bc) is bc_type]
        if not matches:
            return None, (None, None, None, None)

        indices = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in matches])
        imissing = np.vstack([np.asarray(bc.imissing) for bc in matches])
        iknown = np.vstack([np.asarray(bc.iknown) for bc in matches])
        # Not cast to compute_dtype here: kept at whatever dtype apply()'s global path already uses (self.vel /
        # self.weights, untouched), so the local path is numerically identical to it, not independently rounded.
        vel = np.vstack([
            np.broadcast_to(np.asarray(bc.vel), (len(bc.indices[0]), self.dim)) if bc.vel is not None else np.zeros((len(bc.indices[0]), self.dim))
            for bc in matches
        ])
        if has_weights:
            for bc in matches:
                if bc.weights is None:
                    bc.set_proximity_ratio()
            weights = np.vstack([np.asarray(bc.weights) for bc in matches])
        else:
            # Unused by apply_fn for this bc_type (see WALL_BC_TYPES), but always materialized so every wall
            # kernel shares the same argument signature.
            weights = np.zeros_like(imissing, dtype=np.float64)

        local_indices, (local_imissing, local_iknown, local_vel, local_weights) = self._split_local_indices(indices, imissing, iknown, vel, weights)
        return local_indices, (local_imissing, local_iknown, local_vel, local_weights)

    def _collect_solid_pin_indices(self, BCs):
        """
        Build padded local (per-shard, int32) solid-node indices for every wall boundary condition in
        WALL_BC_TYPES present in BCs, used to pin those nodes to the rest equilibrium after streaming (see
        BounceBackHalfway.prepare_populations).

        Parameters
        ----------
        BCs (list): Boundary conditions for one component (or the whole simulation for single-phase).

        Returns
        -------
        (numpy.ndarray or None): Padded local indices, shape (n_devices, max_local, dim), or None if BCs has no
        matching boundary condition.
        """
        solid_indices = [np.asarray(bc.solid_indices, dtype=np.int32).T for wall_type, _, _ in WALL_BC_TYPES for bc in BCs if type(bc) is wall_type]
        if not solid_indices:
            return None
        local_indices, _ = self._split_local_indices(np.vstack(solid_indices))
        return local_indices

    def _make_local_wall_bc_data(self):
        """
        Distribute, per concrete wall boundary condition type in WALL_BC_TYPES, the padded local fluid-node
        indices and auxiliary data, plus one merged padded local solid-node index array used to pin solid nodes
        after streaming.

        Returns
        -------
        (dict, jax.numpy.ndarray or None): Maps each WALL_BC_TYPES class to a (local_indices, local_imissing,
        local_iknown, local_vel, local_weights) tuple of distributed arrays (entries None where not
        applicable), and the distributed local solid-pin indices.

        Notes
        -----
        Overridden by Multiphase to return one such mapping (and one solid-pin index array) per component.
        """
        wall_bc_data = {}
        for bc_type, _, has_weights in WALL_BC_TYPES:
            local_indices, (local_imissing, local_iknown, local_vel, local_weights) = self._collect_wall_bc_data(self.BCs, bc_type, has_weights)
            wall_bc_data[bc_type] = (
                self._distribute_local(local_indices, jnp.int32),
                self._distribute_local(local_imissing, jnp.uint8),
                self._distribute_local(local_iknown, jnp.uint8),
                self._distribute_local(local_vel),
                self._distribute_local(local_weights),
            )
        solid_pin_indices = self._distribute_local(self._collect_solid_pin_indices(self.BCs), jnp.int32)
        return wall_bc_data, solid_pin_indices

    def _collect_inlet_outlet_bc_data(self, BCs, bc_type, ttype):
        """
        Build padded local (per-shard, int32) fluid-node indices and aligned auxiliary data (normals,
        imiddle_mask, iknown_mask, imissing, iknown, prescribed) for every boundary condition of exactly
        bc_type with a matching prescribed-value type (bc.type) in BCs.

        Parameters
        ----------
        BCs (list): Boundary conditions for one component (or the whole simulation for single-phase).

        bc_type (type): Exact ZouHe/Regularized class to match (matched by (type, ttype) instead of type alone,
            since the formula branch is selected by the instance's own bc.type, not the class).

        ttype (str): 'velocity' or 'pressure', matched against each candidate boundary condition's bc.type.

        Returns
        -------
        (numpy.ndarray or None, tuple): Padded local indices and (imissing, iknown, vel, weights)-shaped
        (normals, imiddle_mask, iknown_mask, imissing, iknown, prescribed) padded local arrays. All None if BCs
        has no matching boundary condition.
        """
        matches = [bc for bc in BCs if type(bc) is bc_type and bc.type == ttype]
        if not matches:
            return None, (None, None, None, None, None, None)

        indices = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in matches])
        normals = np.vstack([np.asarray(bc.normals) for bc in matches])
        imiddle_mask = np.vstack([np.asarray(bc.imiddle_mask) for bc in matches])
        iknown_mask = np.vstack([np.asarray(bc.iknown_mask) for bc in matches])
        imissing = np.vstack([np.asarray(bc.imissing) for bc in matches])
        iknown = np.vstack([np.asarray(bc.iknown) for bc in matches])
        prescribed = np.vstack([np.asarray(bc.prescribed) for bc in matches])

        local_indices, (local_normals, local_imiddle_mask, local_iknown_mask, local_imissing, local_iknown, local_prescribed) = (
            self._split_local_indices(indices, normals, imiddle_mask, iknown_mask, imissing, iknown, prescribed)
        )
        return local_indices, (local_normals, local_imiddle_mask, local_iknown_mask, local_imissing, local_iknown, local_prescribed)

    def _collect_equilibrium_bc_data(self, BCs):
        """
        Build padded local (per-shard, int32) fluid-node indices and the aligned precomputed equilibrium values
        for every EquilibriumBC in BCs.

        Parameters
        ----------
        BCs (list): Boundary conditions for one component (or the whole simulation for single-phase).

        Returns
        -------
        (numpy.ndarray or None, numpy.ndarray or None): Padded local indices and padded local equilibrium
        values. Both None if BCs has no EquilibriumBC.
        """
        matches = [bc for bc in BCs if type(bc) is EquilibriumBC]
        if not matches:
            return None, None
        indices = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in matches])
        out = np.vstack([np.asarray(bc.out) for bc in matches])
        local_indices, (local_out,) = self._split_local_indices(indices, out)
        return local_indices, local_out

    def _make_local_inlet_outlet_data(self):
        """
        Distribute, per (concrete type, prescribed-value type) pair in INLET_OUTLET_BC_TYPES, the padded local
        fluid-node indices and auxiliary data.

        Returns
        -------
        (dict): Maps each (bc_type, ttype) pair in INLET_OUTLET_BC_TYPES to a (local_indices, local_normals,
        local_imiddle_mask, local_iknown_mask, local_imissing, local_iknown, local_prescribed) tuple of
        distributed arrays (entries None where not applicable).

        Notes
        -----
        Overridden by Multiphase to return one such mapping per component.
        """
        inlet_outlet_bc_data = {}
        for bc_type, ttype, _ in INLET_OUTLET_BC_TYPES:
            local_indices, (local_normals, local_imiddle_mask, local_iknown_mask, local_imissing, local_iknown, local_prescribed) = (
                self._collect_inlet_outlet_bc_data(self.BCs, bc_type, ttype)
            )
            inlet_outlet_bc_data[(bc_type, ttype)] = (
                self._distribute_local(local_indices, jnp.int32),
                self._distribute_local(local_normals),
                self._distribute_local(local_imiddle_mask, jnp.bool_),
                self._distribute_local(local_iknown_mask, jnp.bool_),
                self._distribute_local(local_imissing, jnp.uint8),
                self._distribute_local(local_iknown, jnp.uint8),
                self._distribute_local(local_prescribed),
            )
        return inlet_outlet_bc_data

    def _make_local_equilibrium_bc_data(self):
        """
        Distribute the padded local EquilibriumBC indices and precomputed equilibrium values for this
        simulation's boundary conditions.

        Returns
        -------
        (jax.numpy.ndarray or None, jax.numpy.ndarray or None): Distributed local indices and equilibrium
        values, both sharded along x, or (None, None) if there is no EquilibriumBC.

        Notes
        -----
        Overridden by Multiphase to return one such pair per component.
        """
        local_indices, local_out = self._collect_equilibrium_bc_data(self.BCs)
        return self._distribute_local(local_indices, jnp.int32), self._distribute_local(local_out)

    @partial(jit, static_argnums=(0, 3), inline=True)
    def equilibrium(self, rho, u, cast_output=True):
        """
        This function computes the equilibrium distribution function in the Lattice Boltzmann Method.
        The equilibrium distribution function is a function of the macroscopic density and velocity.

        The function first casts the density and velocity to the compute precision if the cast_output flag is True.
        The function finally casts the equilibrium distribution function to the output precision if the cast_output
        flag is True.

        Parameters
        ----------
        rho (jax.numpy.ndarray): The macroscopic density.

        u (jax.numpy.ndarray): The macroscopic velocity.

        cast_output (bool, optional): A flag indicating whether to cast the density, velocity, and equilibrium
            distribution function to the
        compute and output precisions. Default is True.

        Returns
        -------
        feq (jax.numpy.ndarray): The equilibrium distribution function.
        """
        # Cast the density and velocity to the compute precision if the cast_output flag is True
        if cast_output:
            rho, u = self.precision_policy.cast_to_compute((rho, u))

        # Cast c to compute precision so that XLA call FXX matmul,
        # which is faster (it is faster in some older versions of JAX, newer versions are smart enough to do this automatically)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cu = 3.0 * jnp.dot(u, c)
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        feq = rho * self.w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)

        if cast_output:
            return self.precision_policy.cast_to_output(feq)
        else:
            return feq

    @partial(jit, static_argnums=(0,))
    def momentum_flux(self, fneq):
        """
        This function computes the momentum flux, which is the product of the non-equilibrium
        distribution functions (fneq) and the lattice moments (cc).

        The momentum flux is used in the computation of the stress tensor in the Lattice Boltzmann
        Method (LBM).

        Parameters
        ----------
        fneq (jax.numpy.ndarray): The non-equilibrium distribution functions.

        Returns
        -------
        (jax.numpy.ndarray): The computed momentum flux.
        """
        return jnp.dot(fneq, self.lattice.cc)

    @partial(jit, static_argnums=(0,), inline=True)
    def update_macroscopic(self, f):
        """
        This function computes the macroscopic variables (density and velocity) based on the
        distribution functions (f).

        The density is computed as the sum of the distribution functions over all lattice directions.
        The velocity is computed as the dot product of the distribution functions and the lattice
        velocities, divided by the density.

        Parameters
        ----------
        f (jax.numpy.ndarray): The distribution functions.

        Returns
        -------
        rho (jax.numpy.ndarray): Computed density.

        u (jax.numpy.ndarray): Computed velocity.
        """
        rho = jnp.sum(f, axis=-1, keepdims=True)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype).T
        u = jnp.dot(f, c) / rho

        return rho, u

    @partial(jit, static_argnums=(0, 4), donate_argnums=(1,))
    def apply_bc(self, fout, fin, timestep, implementation_step):
        """
        This function applies the boundary conditions to the distribution functions.

        It iterates over all boundary conditions (BCs) and checks if the implementation step of the
        boundary condition matches the provided implementation step. If it does, it applies the
        boundary condition to the post-streaming distribution functions (fout).

        Full-way BounceBack, every wall boundary condition in WALL_BC_TYPES (BounceBackHalfway,
        InterpolatedBounceBackBouzidi, InterpolatedBounceBackDifferentiable), EquilibriumBC, and every inlet/
        outlet boundary condition in INLET_OUTLET_BC_TYPES (ZouHe, Regularized) are handled separately, in
        batched calls using local (per-shard, int32) indices, instead of the generic per-BC global-index loop.
        ExtrapolationOutflow/ConvectiveOutflow/ExtrapolationOutflowMultiphase are not: they read a neighbor node
        offset from the boundary (not just their own node), which the wall/inlet-outlet local kernels don't
        support.

        Parameters
        ----------
        fout (jax.numpy.ndarray): Post-collision distribution functions.

        fin (jax.numpy.ndarray): Post-streaming distribution functions.

        timestep (int): Current simulation timestep, used by dynamic boundary conditions.

        implementation_step (str): Implementation step at which the boundary conditions should be applied.

        Returns
        -------
        (jax.numpy.ndarray): The output distribution functions after applying the boundary conditions.
        """
        for bc in self.BCs:
            if isinstance(bc, (BounceBack, BounceBackHalfway, EquilibriumBC, ZouHe)):
                continue
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementation_step == implementation_step:
                if bc.is_dynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin))

        if implementation_step == "PostCollision" and self.local_bounceback_indices is not None:
            fout = self.local_bounceback(fout, fin, self.local_bounceback_indices)

        if implementation_step == "PostStreaming":
            if self.solid_pin_indices is not None:
                fout = self.local_solid_pin(fout, self.solid_pin_indices)
            for bc_type, _, _ in WALL_BC_TYPES:
                local_indices, local_imissing, local_iknown, local_vel, local_weights = self.wall_bc_data[bc_type]
                if local_indices is not None:
                    fout = self.local_wall_bc_kernels[bc_type](fout, fin, local_indices, local_imissing, local_iknown, local_vel, local_weights)
            if self.local_equilibrium_bc_indices is not None:
                fout = self.local_equilibrium_bc(fout, self.local_equilibrium_bc_indices, self.local_equilibrium_bc_values)
            for bc_type, ttype, _ in INLET_OUTLET_BC_TYPES:
                local_indices, local_normals, local_imiddle_mask, local_iknown_mask, local_imissing, local_iknown, local_prescribed = (
                    self.inlet_outlet_bc_data[(bc_type, ttype)]
                )
                if local_indices is not None:
                    fout = self.local_inlet_outlet_kernels[(bc_type, ttype)](
                        fout, local_indices, local_normals, local_imiddle_mask, local_iknown_mask, local_imissing, local_iknown, local_prescribed
                    )

        return fout

    @partial(jit, static_argnums=(0, 3), donate_argnums=(1,))
    def step(self, f_poststreaming, timestep, return_fpost=False):
        """
        This function performs a single step of the LBM simulation.

        It first performs the collision step, which is the relaxation of the distribution functions
        towards their equilibrium values. It then applies the respective boundary conditions to the
        post-collision distribution functions.

        The function then performs the streaming step, which is the propagation of the distribution
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming
        distribution functions.

        Parameters
        ----------
        f_poststreaming (jax.numpy.ndarray): Post-streaming distribution functions.

        timestep (int): The current timestep of the simulation.

        return_fpost (bool, optional): If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        f_poststreaming (jax.numpy.ndarray): Post-streaming distribution functions after the simulation step.

        f_postcollision (jax.numpy.ndarray or None): Post-collision distribution functions after the simulation step, or None if
        return_fpost is False.
        """
        f_postcollision = self.collision(f_poststreaming)
        f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, "PostCollision")
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")

        if return_fpost:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None

    def run(self, t_max):
        """
        This function runs the LBM simulation for a specified number of time steps.

        It first initializes the distribution functions and then enters a loop where it performs the
        simulation steps (collision, streaming, and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and
        compute the performance of the simulation in million lattice updates per second (MLUPS).

        Parameters
        ----------
        t_max (int): The total number of time steps to run the simulation.

        Returns
        -------
        f (jax.numpy.ndarray): The distribution functions after the simulation.
        """
        f = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                # Assert that the checkpoint manager is not None
                assert self.mngr is not None, "Checkpoint manager does not exist."
                restore_target = lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=self.sharding)
                state = jax.tree.map(restore_target, {"f": f})
                try:
                    f = self.mngr.restore(latest_step, args=orb.args.StandardRestore(state))["f"]
                    logger.info(f"Restored checkpoint at step {latest_step}.")
                except ValueError:
                    raise ValueError(f"Failed to restore checkpoint at step {latest_step}.")

                start_step = latest_step + 1
                if not (t_max > start_step):
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max = {t_max}). Consider increasing t_max.")
        if self.compute_MLUPS:
            start = time.time()
        # Loop over all time steps
        for timestep in range(start_step, t_max + 1):
            io_flag = self.io_rate > 0 and (timestep % self.io_rate == 0 or timestep == t_max)
            print_iter_flag = self.print_info_rate > 0 and timestep % self.print_info_rate == 0
            checkpoint_flag = self.checkpoint_rate > 0 and timestep % self.checkpoint_rate == 0

            if io_flag:
                # Update the macroscopic variables and save the previous values (for error computation)
                rho_prev, u_prev = self.update_macroscopic(f)
                rho_prev = downsample_field(rho_prev, self.downsampling_factor)
                u_prev = downsample_field(u_prev, self.downsampling_factor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho_prev = process_allgather(rho_prev)
                u_prev = process_allgather(u_prev)

            # Perform one time-step (collision, streaming, and boundary conditions)
            f, fstar = self.step(f, timestep, return_fpost=self.return_fpost)
            # Print the progress of the simulation
            if print_iter_flag:
                logger.info(
                    colored("Timestep ", "blue")
                    + colored(f"{timestep}", "green")
                    + colored(" of ", "blue")
                    + colored(f"{t_max}", "green")
                    + colored(" completed", "blue")
                )

            if io_flag:
                # Save the simulation data
                logger.info(f"Saving data at timestep {timestep}/{t_max}")
                rho, u = self.update_macroscopic(f)
                rho = downsample_field(rho, self.downsampling_factor)
                u = downsample_field(u, self.downsampling_factor)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho = process_allgather(rho)
                u = process_allgather(u)

                # Save the data
                self.handle_io_timestep(timestep, f, fstar, rho, u, rho_prev, u_prev)

            if checkpoint_flag:
                # Save the checkpoint
                logger.info(f"Saving checkpoint at timestep {timestep}/{t_max}")
                state = {"f": f}
                # self.mngr.save(timestep, state)
                self.mngr.save(timestep, args=orb.args.StandardSave(state))

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.compute_MLUPS and timestep == 1:
                jax.block_until_ready(f)
                start = time.time()

        if self.compute_MLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f)
            end = time.time()
            if self.dim == 2:
                logger.info(
                    colored("Domain: ", "blue") + colored(f"{self.nx} x {self.ny}", "green")
                    if self.dim == 2
                    else colored(f"{self.nx} x {self.ny} x {self.nz}", "green")
                )
                logger.info(
                    colored("Number of voxels: ", "blue") + colored(f"{self.nx * self.ny}", "green")
                    if self.dim == 2
                    else colored(f"{self.nx * self.ny * self.nz}", "green")
                )
                logger.info(colored("MLUPS: ", "blue") + colored(f"{self.nx * self.ny * t_max / (end - start) / 1e6}", "red"))

            elif self.dim == 3:
                logger.info(colored("Domain: ", "blue") + colored(f"{self.nx} x {self.ny} x {self.nz}", "green"))
                logger.info(colored("Number of voxels: ", "blue") + colored(f"{self.nx * self.ny * self.nz}", "green"))
                logger.info(
                    colored("MLUPS: ", "blue")
                    + colored(
                        f"{self.nx * self.ny * self.nz * t_max / (end - start) / 1e6}",
                        "red",
                    )
                )
        if self.mngr is not None:
            self.mngr.wait_until_finished()
        return f

    def handle_io_timestep(self, timestep, f, fstar, rho, u, rho_prev, u_prev):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep (int): Current time step of the simulation.

        f (jax.numpy.ndarray): Post-streaming distribution functions at the current time step.

        fstar (jax.numpy.ndarray): Post-collision distribution functions at the current time step.

        rho (jax.numpy.ndarray): Density field at the current time step.

        u (jax.numpy.ndarray): Velocity field at the current time step.

        rho_prev (jax.numpy.ndarray): Density field at the previous I/O time step.

        u_prev (jax.numpy.ndarray): Velocity field at the previous I/O time step.
        """
        kwargs = {
            "timestep": timestep,
            "rho": rho,
            "rho_prev": rho_prev,
            "u": u,
            "u_prev": u_prev,
            "f_poststreaming": f,
            "f_postcollision": fstar,
        }
        self.output_data(**kwargs)

    def output_data(self, **kwargs):
        """
        This function is intended to be overwritten by the user to customize the input/output (I/O)
        operations of the simulation.

        By default, it does nothing. When overwritten, it could save the simulation data to files,
        display the simulation results in real time, send the data to another process for analysis, etc.

        Parameters
        ----------
        **kwargs (dict): A dictionary containing the simulation data to be outputted. The keys are the names of the
        data fields, and the values are the data fields themselves.
        """
        pass

    def set_boundary_conditions(self):
        """
        This function sets the boundary conditions for the simulation.

        It is intended to be overwritten by the user to specify the boundary conditions according to
        the specific problem being solved.

        By default, it does nothing. When overwritten, it could set periodic boundaries, no-slip
        boundaries, inflow/outflow boundaries, etc.
        """
        pass

    @partial(jit, static_argnums=(0,))
    def collision(self, fin):
        """
        This function performs the collision step in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the collision operator according to
        the specific LBM model being used.

        By default, it does nothing. When overwritten, it could implement the BGK collision operator,
        the MRT collision operator, etc.

        Parameters
        ----------
        fin (jax.numpy.ndarray): Pre-collision distribution functions.

        Returns
        -------
        fin (jax.numpy.ndarray): Post-collision distribution functions.
        """
        pass

    def get_force(self):
        """
        This function computes the force to be applied to the fluid in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the force according to the specific
        problem being solved.

        By default, it does nothing and returns None. When overwritten, it could implement a constant
        force term.

        Returns
        -------
        force (jax.numpy.ndarray): The force to be applied to the fluid.
        """
        pass

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, f_postcollision, feq, rho, u):
        """
        add force based on exact-difference method due to Kupershtokh

        Computes delta_feq = feq(rho, u + du) - feq(rho, u) directly from cu, dcu and delta_usqr instead of
        building a second full equilibrium distribution and subtracting feq from it. Lattice-generic: works for
        any lattice this class supports (D2Q9, D3Q19, D3Q27), since it only uses self.c/self.w.

        Parameters
        ----------
        f_postcollision (jax.numpy.ndarray): Post-collision distribution functions.

        feq (jax.numpy.ndarray): Equilibrium distribution functions. Unused - kept for interface compatibility
            with existing callers, since the compact difference formula only needs rho, u and the force.

        rho (jax.numpy.ndarray): Density field.

        u (jax.numpy.ndarray): Velocity field.

        Returns
        -------
        f_postcollision (jax.numpy.ndarray): Post-collision distribution functions with the force applied.

        References
        ----------
        1. Kupershtokh, A. (2004). New method of incorporating a body force term into the lattice Boltzmann equation. In
        Proceedings of the 5th International EHD Workshop (pp. 241-246). University of Poitiers, Poitiers, France.
        Chikatamarla, S. S., & Karlin, I. V. (2013). Entropic lattice Boltzmann method for turbulent flow simulations:
        Boundary conditions. Physica A, 392, 1925-1930.
        2. Krüger, T., et al. (2017). The lattice Boltzmann method. Springer International Publishing, 10.978-3, 4-15.
        """
        du = self.get_force()
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cu = 3.0 * jnp.dot(u, c)
        dcu = 3.0 * jnp.dot(du, c)
        delta_usqr = 1.5 * (2.0 * jnp.sum(u * du, axis=-1, keepdims=True) + jnp.sum(jnp.square(du), axis=-1, keepdims=True))
        delta_feq = rho * self.w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr)
        return f_postcollision + delta_feq
