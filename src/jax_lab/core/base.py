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

        else:
            raise ValueError(f"dim = {self.dim} not supported")

        # Compute the bounding box indices for boundary conditions
        self.bounding_box_indices = self.compute_bounding_box_indices()
        # Create boundary data for the simulation
        self._create_boundary_data()
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

    @partial(jit, static_argnums=(0, 4), inline=True)
    def apply_bc(self, fout, fin, timestep, implementation_step):
        """
        This function applies the boundary conditions to the distribution functions.

        It iterates over all boundary conditions (BCs) and checks if the implementation step of the
        boundary condition matches the provided implementation step. If it does, it applies the
        boundary condition to the post-streaming distribution functions (fout).

        Parameters
        ----------
        fout (jax.numpy.ndarray): Post-collision distribution functions.

        fin (jax.numpy.ndarray): Post-streaming distribution functions.

        implementation_step (str): Implementation step at which the boundary conditions should be applied.

        Returns
        -------
        (jax.numpy.ndarray): The output distribution functions after applying the boundary conditions.
        """
        for bc in self.BCs:
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementation_step == implementation_step:
                if bc.is_dynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin))

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
                state = {"f": f}
                # shardings = map(lambda x: x.sharding, state)
                # restore_args = orb.checkpoint_utils.construct_restore_args(state, shardings)
                try:
                    # f = self.mngr.restore(latest_step, restore_kwargs={'restore_args': restore_args})['f']
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

        Parameters
        ----------
        f_postcollision (jax.numpy.ndarray): Post-collision distribution functions.

        feq (jax.numpy.ndarray): Equilibrium distribution functions.

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
        delta_u = self.get_force()
        feq_force = self.equilibrium(rho, u + delta_u, cast_output=False)
        f_postcollision = f_postcollision + feq_force - feq
        return f_postcollision
