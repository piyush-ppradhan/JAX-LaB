"""
Hybrid thermal LBM solver: the fluid is evolved with the lattice Boltzmann method while the temperature field is evolved with a finite difference solver
on the same mesh, using lattice-based isotropic difference stencils and a fourth order Runge-Kutta time integration. Throughout, dx = dt = 1 (lattice units).

References
----------
1. Fei, L., Yang, J., Chen, Y., Mo, H. & Luo, K. H. Mesoscopic simulation of three-dimensional pool boiling based on a phase-change cascaded lattice Boltzmann method.
Physics of Fluids 32, 103312 (2020).
"""

import logging
import operator
import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from jax import jit
from jax.tree import map as tree_map
from jax.tree import reduce
from jax.experimental.multihost_utils import process_allgather
from termcolor import colored

from .utils import downsample_field

logger = logging.getLogger(__name__)


class Thermal(object):
    """
    Single phase hybrid thermal LBM solver. The fluid is advanced by the
    wrapped LBM fluid_solver while the temperature field is advanced by a
    finite difference solver on the same mesh.

    The temperature equation solved is:
    \\frac{\\partial T}{\\partial t} = -\\mathbf{u} \\cdot \\nabla T + \\frac{1}{\\rho c_v}(\\nabla \\cdot (K \\nabla T) + S_T)
    where S_T is a user defined source term (see source()).

    Spatial derivatives are evaluated with the isotropic lattice difference stencils in grad_x and laplacian_x and time integration
    uses the fourth order Runge-Kutta scheme with dt = 1.

    Parameters
    ----------
    fluid_solver (LBMBase): Configured fluid solver instance (e.g. BGKSim or MRTSim). Grid, lattice, precision and I/O settings are shared with it.

    specific_heat (float or numpy.ndarray): Specific heat c_v. Either a scalar or an array of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.

    thermal_conductivity (float or numpy.ndarray): Thermal conductivity K with the same shape options as specific_heat.

    checkpoint_dir (str, optional): Directory for temperature checkpoints. Defaults to "./temperature_checkpoints".

    apply_buoyancy (bool, optional): If True, adds a density variation based buoyancy force to the fluid solver. Defaults to False.

    gravity (sequence of float, optional): Gravitational acceleration vector, required when apply_buoyancy is True.

    rk_substeps (int, optional): Number of Runge-Kutta substeps per LBM step. Defaults to 1.
    The explicit RK4 diffusion update is stable only if K / (rho c_v) |lambda_max| dt < 2.785,
    where |lambda_max| is the largest eigenvalue of the lattice laplacian stencil (16/3 for
    both D2Q9 and D3Q19). Increase rk_substeps when the thermal diffusivity exceeds this limit.
    """

    def __init__(self, **kwargs):
        self.fluid_solver = kwargs.get("fluid_solver")
        if self.fluid_solver is None:
            raise ValueError("A fluid_solver must be provided for thermal simulations.")

        # Share grid, lattice, precision and run control settings with the fluid solver
        self.lattice = self.fluid_solver.lattice
        self.precision_policy = self.fluid_solver.precision_policy
        self.nx = self.fluid_solver.nx
        self.ny = self.fluid_solver.ny
        self.nz = self.fluid_solver.nz
        self.dim = self.fluid_solver.dim
        self.streaming = self.fluid_solver.streaming
        self.io_rate = self.fluid_solver.io_rate
        self.print_info_rate = self.fluid_solver.print_info_rate
        self.downsampling_factor = self.fluid_solver.downsampling_factor
        self.return_fpost = self.fluid_solver.return_fpost
        self.compute_MLUPS = self.fluid_solver.compute_MLUPS
        self.restore_checkpoint = self.fluid_solver.restore_checkpoint
        self.checkpoint_rate = self.fluid_solver.checkpoint_rate
        self.checkpoint_dir = kwargs.get("checkpoint_dir", "./temperature_checkpoints")
        self.n_devices = jax.device_count()
        self.backend = jax.default_backend()

        if self.checkpoint_rate > 0:
            mngr_options = orb.CheckpointManagerOptions(save_interval_steps=self.checkpoint_rate, max_to_keep=1)
            self.mngr = orb.CheckpointManager(self.checkpoint_dir, options=mngr_options)
        else:
            self.mngr = None

        self.rkSubsteps = int(kwargs.get("rk_substeps", 1))
        if self.rkSubsteps < 1:
            raise ValueError("rk_substeps must be a positive integer.")

        self.c_v = kwargs.get("specific_heat")
        self.K = kwargs.get("thermal_conductivity")
        # K is constant in time, so its gradient is precomputed once
        self.grad_K = self.grad_x(self.K)

        self.apply_buoyancy = kwargs.get("apply_buoyancy", False)
        if self.apply_buoyancy:
            gravity = kwargs.get("gravity")
            if gravity is None:
                raise ValueError("gravity must be provided when apply_buoyancy is True.")
            self.gravity = jnp.array(np.array(gravity, dtype=np.float64), dtype=self.precision_policy.compute_dtype)
            # The fluid collision only invokes apply_force when a force is present
            if self.fluid_solver.force is None:
                self.fluid_solver.force = jnp.zeros(self.dim, dtype=self.precision_policy.compute_dtype)
            self.fluid_solver.apply_force = self.apply_force_thermal

        self.set_thermal_boundary_conditions()

    @property
    def c_v(self):
        return self._c_v

    @c_v.setter
    def c_v(self, value):
        self._c_v = self._to_field("specific_heat", value)

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = self._to_field("thermal_conductivity", value)

    def _to_field(self, name, value):
        """
        Convert a scalar or numpy array parameter to a full field JAX array.

        Parameters
        ----------
        name (str): Parameter name used in error messages.

        value (float, int or numpy.ndarray): Scalar applied uniformly or an array
        of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.

        Returns
        -------
        jax.numpy.ndarray: Field of shape (nx, ny, 1) or (nx, ny, nz, 1).
        """
        if value is None:
            raise ValueError(f"{name} must be provided.")
        shape = (self.nx, self.ny, 1) if self.dim == 2 else (self.nx, self.ny, self.nz, 1)
        if isinstance(value, np.ndarray):
            if value.shape != shape:
                raise ValueError(f"The shape of {name} array must match the dimensions: (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D")
            return jnp.array(value, dtype=self.precision_policy.compute_dtype)
        elif isinstance(value, (int, float)):
            return float(value) * jnp.ones(shape, dtype=self.precision_policy.compute_dtype)
        else:
            raise ValueError(f"Invalid type for {name}. It must be float or numpy.ndarray.")

    def set_thermal_boundary_conditions(self):
        """
        This function sets the boundary conditions for the temperature field.

        It is intended to be overwritten by the user to specify the boundary conditions according to the specific problem being solved, by appending
        ThermalBoundaryCondition instances (DirichletTemperature, NeumannTemperature) to self.thermal_BCs. Conditions are applied in list order, so
        later entries win on shared nodes (e.g. corners).

        By default no thermal boundary condition is applied, which corresponds to a fully periodic temperature field.
        """
        self.thermal_BCs = []
        return

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_bc(self, T, timestep):
        """
        This function applies the boundary conditions to the temperature field.

        It iterates over all thermal boundary conditions and applies them in list order.

        Parameters
        ----------
        T (jax.numpy.ndarray): Temperature field.

        timestep (int): Current timestep of the simulation.

        Returns
        -------
        jax.numpy.ndarray: Temperature field after applying boundary conditions.
        """
        for bc in self.thermal_BCs:
            T = bc.apply(T, timestep)

        return T

    def initialize_temperature_field(self):
        """
        Return the initial temperature field.

        The default implementation returns ``None``, which :meth:`assign_fields_sharded` interprets as a uniform temperature of 1.
        Override this method to provide a scalar or spatially varying field.

        Returns
        -------
        None
            Sentinel requesting the default uniform temperature.
        """
        warnings.warn(
            "Default initial temperature assumed: temperature = 1. Override initialize_temperature_field to set an explicit value.",
            UserWarning,
            stacklevel=2,
        )
        return None

    @partial(jit, static_argnums=(0, 1, 2, 4))
    def distributed_array_init(self, shape, ttype, init_val=0, sharding=None):
        """
        Initialize a distributed array using JAX, with a specified shape, data type, and initial value. Optionally, provide a custom sharding strategy.

        Parameters
        ----------
        shape (tuple): The shape of the array to be created.

        ttype (dtype): The data type of the array to be created.

        init_val (scalar, optional): The initial value to fill the array with. Defaults to 0.

        sharding (Sharding, optional): The sharding strategy to use. Defaults to the fluid solver sharding.

        Returns
        -------
        jax.numpy.ndarray: A JAX array with the specified shape, data type, initial value, and sharding strategy.
        """
        if sharding is None:
            sharding = self.fluid_solver.sharding
        x = jnp.full(shape=shape, fill_value=init_val, dtype=ttype)
        return jax.lax.with_sharding_constraint(x, sharding)

    def assign_fields_sharded(self):
        """
        This function initializes the temperature field of the simulation.

        It calls initialize_temperature_field, which can return a scalar or an array of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D. If it
        returns None, a uniform temperature of 1 is assumed.

        Returns
        -------
        T: a distributed JAX array of shape (nx, ny, 1) or (nx, ny, nz, 1) holding the temperature field.
        """
        T0 = self.initialize_temperature_field()

        shape = (self.nx, self.ny, 1) if self.dim == 2 else (self.nx, self.ny, self.nz, 1)
        if T0 is None:
            T0 = 1.0
        if isinstance(T0, np.ndarray):
            T0 = jnp.array(T0, dtype=self.precision_policy.output_dtype)

        T = self.distributed_array_init(shape, self.precision_policy.output_dtype, init_val=T0)

        return T

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force_thermal(self, f_postcollision, feq, rho, u):
        """
        Modified version of the single phase apply_force function that adds a density variation based buoyancy force to any user defined fluid force,
        using the exact-difference method due to Kupershtokh.

        Note: the buoyancy force is computed from the local density deviation relative to the mean density, not from the temperature field, since the
        fluid collision does not have access to the temperature.

        Parameters
        ----------
        f_postcollision (jax.numpy.ndarray): Post-collision distribution functions.

        feq (jax.numpy.ndarray): Equilibrium distribution functions.

        rho (jax.numpy.ndarray): Density field.

        u (jax.numpy.ndarray): Velocity field.

        Returns
        -------
        jax.numpy.ndarray: Post-collision distribution functions with the force applied.
        """
        rho_average = rho.mean()
        buoyancy = jnp.repeat(rho - rho_average, repeats=self.dim, axis=-1) * self.gravity
        delta_u = buoyancy + self.fluid_solver.force
        feq_force = self.fluid_solver.equilibrium(rho, u + delta_u, cast_output=False)
        f_postcollision = f_postcollision + feq_force - feq
        return f_postcollision

    @partial(jit, static_argnums=(0,), inline=True)
    def grad_x(self, field):
        """
        Compute the gradient of a scalar field using the isotropic lattice
        difference stencil:
        \\nabla \\phi(x) = \\frac{1}{c_s^2} \\sum_i w_i \\mathbf{c}_i \\phi(x + \\mathbf{c}_i)

        The neighbor values are obtained with the streaming operation, which
        shifts along -c_i, hence the sign flip in the final contraction.

        Note: streaming wraps periodically at the domain edges. On non-periodic
        boundaries the affected nodes must be corrected by the thermal boundary
        conditions.

        Parameters
        ----------
        field (jax.numpy.ndarray): Scalar field of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.

        Returns
        -------
        jax.numpy.ndarray: Gradient of the field, of shape (nx, ny, 2) in 2D or (nx, ny, nz, 3) in 3D.
        """
        field_streamed = self.streaming(jnp.repeat(field, repeats=self.lattice.q, axis=-1))
        c = jnp.array(self.lattice.c, dtype=self.precision_policy.compute_dtype).T
        return -self.lattice.inv_cs2 * jnp.dot(self.lattice.w * field_streamed, c)

    @partial(jit, static_argnums=(0,), inline=True)
    def laplacian_x(self, field):
        """
        Compute the laplacian of a scalar field using the isotropic lattice difference stencil:
        \\nabla^2 \\phi(x) = \\frac{2}{c_s^2} \\sum_i w_i [\\phi(x + \\mathbf{c}_i) - \\phi(x)]

        Note: streaming wraps periodically at the domain edges. On non-periodic boundaries the
        affected nodes must be corrected by the thermal boundary conditions.

        Parameters
        ----------
        field (jax.numpy.ndarray): Scalar field of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.

        Returns
        -------
        jax.numpy.ndarray: Laplacian of the field, of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.
        """
        field_streamed = self.streaming(jnp.repeat(field, repeats=self.lattice.q, axis=-1))
        return 2.0 * self.lattice.inv_cs2 * jnp.sum(self.lattice.w * (field_streamed - field), axis=-1, keepdims=True)

    @partial(jit, static_argnums=(0,), inline=True)
    def source(self, T):
        """
        Source term S_T of the thermal equation.

        It is intended to be overwritten by the user to specify a volumetric
        heat source according to the specific problem being solved. By default
        it returns zero everywhere.

        Parameters
        ----------
        T (jax.numpy.ndarray): Temperature field.

        Returns
        -------
        jax.numpy.ndarray: Source term at each lattice node, same shape as T.
        """
        return jnp.zeros_like(T)

    @partial(jit, static_argnums=(0,), inline=True)
    def RHS(self, T, rho, u):
        """
        Right hand side of the thermal equation:
        -\\mathbf{u} \\cdot \\nabla T + \\frac{1}{\\rho c_v}(K \\nabla^2 T + \\nabla K \\cdot \\nabla T + S_T)

        Parameters
        ----------
        T (jax.numpy.ndarray): Temperature field.

        rho (jax.numpy.ndarray): Density field.

        u (jax.numpy.ndarray): Velocity field.

        Returns
        -------
        jax.numpy.ndarray: RHS of the thermal equation, of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.
        """
        grad_T = self.grad_x(T)
        lap_T = self.laplacian_x(T)

        advection = jnp.sum(u * grad_T, axis=-1, keepdims=True)
        diffusion = self.K * lap_T + jnp.sum(self.grad_K * grad_T, axis=-1, keepdims=True)

        return -advection + (diffusion + self.source(T)) / (rho * self.c_v)

    @partial(jit, static_argnums=(0,), inline=True)
    def _advance_temperature(self, T, timestep, *fields):
        """
        Advance the temperature field by one time unit using rkSubsteps fourth
        order Runge-Kutta substeps of size dt = 1 / rkSubsteps, holding the
        macroscopic fields frozen.

        Parameters
        ----------
        T (jax.numpy.ndarray): Temperature field.

        timestep (int): Current timestep used by the thermal boundary conditions.

        *fields: Macroscopic fields forwarded to RHS (rho, u for single phase;
        rho_tree, u_tree for multiphase).

        Returns
        -------
        jax.numpy.ndarray: Temperature field advanced by one time unit.
        """
        dt = 1.0 / self.rkSubsteps
        for _ in range(self.rkSubsteps):
            T = self.apply_bc(T, timestep)
            k1 = self.RHS(T, *fields)
            T_stage = self.apply_bc(T + 0.5 * dt * k1, timestep)
            k2 = self.RHS(T_stage, *fields)
            T_stage = self.apply_bc(T + 0.5 * dt * k2, timestep)
            k3 = self.RHS(T_stage, *fields)
            T_stage = self.apply_bc(T + dt * k3, timestep)
            k4 = self.RHS(T_stage, *fields)
            T = T + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return self.apply_bc(T, timestep)

    def update_macroscopic_output(self, f, T):
        """
        Compute the macroscopic density and velocity used for I/O. Overridden
        by MultiphaseThermal to include the force correction and pytrees.

        Parameters
        ----------
        f (jax.numpy.ndarray): Post-streaming distribution functions.

        T (jax.numpy.ndarray): Temperature field.

        Returns
        -------
        rho (jax.numpy.ndarray): Density field.

        u (jax.numpy.ndarray): Velocity field.
        """
        return self.fluid_solver.update_macroscopic(f)

    @partial(jit, static_argnums=(0, 4), donate_argnums=(1,))
    def step(self, T_prev, f_poststreaming, timestep, return_fpost=False):
        """
        This function performs a single step of the hybrid thermal LBM simulation.

        It evaluates the macroscopic fields and temperature right hand side at
        the current time level, then advances the fluid and temperature fields
        together by one time step.

        Parameters
        ----------
        T_prev (jax.numpy.ndarray): The temperature field from the previous timestep.

        f_poststreaming (jax.numpy.ndarray): The post-streaming distribution functions.

        timestep (int): The current timestep of the simulation.

        return_fpost (bool, optional): If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        T (jax.numpy.ndarray): The temperature field after the simulation step.

        f_poststreaming (jax.numpy.ndarray): The post-streaming distribution functions after the simulation step.

        f_postcollision (jax.numpy.ndarray or None): The post-collision distribution functions after the simulation
        step, or None if return_fpost is False.
        """
        T = self.precision_policy.cast_to_compute(T_prev)
        T = self.apply_bc(T, timestep)
        f_compute = self.precision_policy.cast_to_compute(f_poststreaming)
        rho, u = self.fluid_solver.update_macroscopic(f_compute)

        f_poststreaming, f_postcollision = self.fluid_solver.step(f_poststreaming, timestep, return_fpost=return_fpost)
        T = self._advance_temperature(T, timestep, rho, u)

        T = self.apply_bc(T, timestep)

        return self.precision_policy.cast_to_output(T), f_poststreaming, f_postcollision

    def run(self, t_max):
        """
        This function runs the hybrid thermal LBM simulation for a specified number of time steps.

        It first initializes the temperature field and the fluid distribution
        functions and then enters a loop where it performs the coupled
        simulation steps (fluid collision, streaming and boundary conditions,
        followed by the Runge-Kutta temperature update) for each time step.

        The function can also print the progress of the simulation, save the
        simulation data, and compute the performance of the simulation in
        million lattice updates per second (MLUPS).

        Parameters
        ----------
        t_max (int): The total number of time steps to run the simulation.

        Returns
        -------
        f (jax.numpy.ndarray): The distribution functions after the simulation.

        T (jax.numpy.ndarray): The temperature field after the simulation.
        """
        T = self.assign_fields_sharded()
        f = self.fluid_solver.assign_fields_sharded()
        # output_data can set this flag to True to stop the simulation early
        # (e.g. once a droplet has fully evaporated)
        self.stop_simulation = False
        start_step = 0
        if self.restore_checkpoint:
            assert self.mngr is not None, "Checkpoint manager does not exist."
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                try:
                    T = self.mngr.restore(latest_step, args=orb.args.StandardRestore({"T": T}))["T"]
                    f = self.fluid_solver.mngr.restore(latest_step, args=orb.args.StandardRestore({"f": f}))["f"]
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
                # Save the previous values of the macroscopic fields (for error computation)
                rho_prev, u_prev = self.update_macroscopic_output(f, T)
                rho_prev = tree_map(lambda x: downsample_field(x, self.downsampling_factor), rho_prev)
                u_prev = tree_map(lambda x: downsample_field(x, self.downsampling_factor), u_prev)
                T_prev = downsample_field(T, self.downsampling_factor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho_prev = process_allgather(rho_prev)
                u_prev = process_allgather(u_prev)
                T_prev = process_allgather(T_prev)

            # Perform one time-step (fluid step followed by the temperature update)
            T, f, fstar = self.step(T, f, timestep, return_fpost=self.return_fpost)

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
                rho, u = self.update_macroscopic_output(f, T)
                rho = tree_map(lambda x: downsample_field(x, self.downsampling_factor), rho)
                u = tree_map(lambda x: downsample_field(x, self.downsampling_factor), u)
                T_out = downsample_field(T, self.downsampling_factor)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho = process_allgather(rho)
                u = process_allgather(u)
                T_out = process_allgather(T_out)

                # Save the data
                self.handle_io_timestep(timestep, f, fstar, T_out, rho, u, T_prev, rho_prev, u_prev)

            if self.stop_simulation:
                logger.info(f"Stopping the simulation early at timestep {timestep} (requested by output_data).")
                break

            if checkpoint_flag:
                # Save the checkpoint
                logger.info(f"Saving checkpoint at timestep {timestep}/{t_max}")
                self.mngr.save(timestep, args=orb.args.StandardSave({"T": T}))
                if self.fluid_solver.mngr is not None:
                    self.fluid_solver.mngr.save(timestep, args=orb.args.StandardSave({"f": f}))

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.compute_MLUPS and timestep == 1:
                jax.block_until_ready(f)
                jax.block_until_ready(T)
                start = time.time()

        if self.compute_MLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(T)
            jax.block_until_ready(f)
            end = time.time()
            n_voxels = self.nx * self.ny if self.dim == 2 else self.nx * self.ny * self.nz
            domain = f"{self.nx} x {self.ny}" if self.dim == 2 else f"{self.nx} x {self.ny} x {self.nz}"
            logger.info(colored("Domain: ", "blue") + colored(domain, "green"))
            logger.info(colored("Number of voxels: ", "blue") + colored(f"{n_voxels}", "green"))
            logger.info(colored("MLUPS: ", "blue") + colored(f"{n_voxels * t_max / (end - start) / 1e6}", "red"))

        if self.mngr is not None:
            self.mngr.wait_until_finished()
        if self.fluid_solver.mngr is not None:
            self.fluid_solver.mngr.wait_until_finished()
        return f, T

    def handle_io_timestep(self, timestep, f, fstar, T, rho, u, T_prev, rho_prev, u_prev):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep (int): The current time step of the simulation.

        f (jax.numpy.ndarray): The post-streaming distribution functions at the current time step.

        fstar (jax.numpy.ndarray): The post-collision distribution functions at the current time step.

        T (jax.numpy.ndarray): The temperature field at the current time step.

        rho (jax.numpy.ndarray): The density field at the current time step.

        u (jax.numpy.ndarray): The velocity field at the current time step.

        T_prev (jax.numpy.ndarray): The temperature field at the previous I/O time step.

        rho_prev (jax.numpy.ndarray): The density field at the previous I/O time step.

        u_prev (jax.numpy.ndarray): The velocity field at the previous I/O time step.
        """
        kwargs = {
            "timestep": timestep,
            "rho": rho,
            "rho_prev": rho_prev,
            "u": u,
            "u_prev": u_prev,
            "T": T,
            "T_prev": T_prev,
            "f_poststreaming": f,
            "f_postcollision": fstar,
        }
        self.output_data(**kwargs)

    def output_data(self, **kwargs):
        """
        This function is intended to be overwritten by the user to customize
        the I/O operations of the simulation. By default it does nothing.

        Parameters
        ----------
        **kwargs: The simulation data at the current I/O timestep, see handle_io_timestep.
        """
        pass


class MultiphaseThermal(Thermal):
    """
    Multiphase implementation of the hybrid thermal LBM solver with phase
    change support. The fluid is advanced by the wrapped Multiphase solver
    (pytrees of distribution functions, one per component) while a single
    shared temperature field is advanced by the finite difference solver.

    The temperature equation solved is (reference 1 in the module docstring):
    \\frac{\\partial T}{\\partial t} = -\\mathbf{u} \\cdot \\nabla T + \\frac{1}{\\rho c_v}(\\nabla \\cdot (K \\nabla T)
    - T \\frac{\\partial p_{EOS}}{\\partial T} \\nabla \\cdot \\mathbf{u} + S_T)

    where the T (dp_EOS/dT) div(u) term accounts for the latent heat released or absorbed during phase change, rho and u are the mixture density
    and velocity, and S_T is a user defined source term (see source()).

    The temperature field is passed into the fluid solver every step, so the EOS of the fluid solver must be constructed with
    temperature_field_type="thermal"; the pressure (and hence the pseudopotential) then follows the local temperature, which is what drives
    evaporation and condensation.

    Assumes all components share a single temperature field (thermal equilibrium between components at every node).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.fluid_solver.eos.temperature_field_type != "thermal":
            raise ValueError('The EOS of the fluid solver must be constructed with temperature_field_type="thermal" for MultiphaseThermal.')
        if self.apply_buoyancy:
            raise ValueError("apply_buoyancy is not supported for MultiphaseThermal; use the body_force parameter of the fluid solver instead.")

    @partial(jit, static_argnums=(0,), inline=True)
    def divergence_x(self, u):
        """
        Compute the divergence of a vector field using the isotropic lattice
        difference stencil, by summing the diagonal entries of the gradient of
        each velocity component.

        Parameters
        ----------
        u (jax.numpy.ndarray): Vector field of shape (nx, ny, 2) in 2D or (nx, ny, nz, 3) in 3D.

        Returns
        -------
        jax.numpy.ndarray: Divergence of the field, of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.
        """
        div_u = jnp.zeros_like(u[..., :1])
        for alpha in range(self.dim):
            div_u = div_u + self.grad_x(u[..., alpha : alpha + 1])[..., alpha : alpha + 1]
        return div_u

    @partial(jit, static_argnums=(0,), inline=True)
    def RHS(self, T, rho_tree, u_tree):
        """
        Right hand side of the thermal equation for multiphase flow:
        -\\mathbf{u} \\cdot \\nabla T + \\frac{1}{\\rho c_v}(K \\nabla^2 T + \\nabla K \\cdot \\nabla T
        - T \\frac{\\partial p_{EOS}}{\\partial T} \\nabla \\cdot \\mathbf{u} + S_T)

        The mixture (total) density and velocity are used; the phase change
        term uses the sum of dp_EOS/dT over all components.

        Parameters
        ----------
        T (jax.numpy.ndarray): Temperature field.

        rho_tree (pytree of jax.numpy.ndarray): Density field of all components.

        u_tree (pytree of jax.numpy.ndarray): Velocity field of all components.

        Returns
        -------
        jax.numpy.ndarray: RHS of the thermal equation, of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.
        """
        grad_T = self.grad_x(T)
        lap_T = self.laplacian_x(T)
        # Mixture (total) density and velocity
        rho = self.fluid_solver.compute_total_density(rho_tree)
        u = self.fluid_solver.compute_total_velocity(rho_tree, u_tree)

        advection = jnp.sum(u * grad_T, axis=-1, keepdims=True)
        diffusion = self.K * lap_T + jnp.sum(self.grad_K * grad_T, axis=-1, keepdims=True)

        div_u = self.divergence_x(u)
        dp_eos_dT_tree = self.fluid_solver.eos.dp_eos_dT(rho_tree, T)
        phase_change = T * reduce(operator.add, dp_eos_dT_tree) * div_u

        return -advection + (diffusion - phase_change + self.source(T)) / (rho * self.c_v)

    @partial(jit, static_argnums=(0, 4), donate_argnums=(1,))
    def step(self, T_prev, f_poststreaming_tree, timestep, return_fpost=False):
        """
        This function performs a single step of the multiphase hybrid thermal LBM simulation.

        It evaluates the component fields and temperature right hand side at
        the current time level, then advances the fluid and temperature fields
        together by one time step. The current temperature is passed to the
        thermal EOS used by the fluid collision.

        Parameters
        ----------
        T_prev (jax.numpy.ndarray): The temperature field from the previous timestep.

        f_poststreaming_tree (pytree of jax.numpy.ndarray): The post-streaming distribution functions.

        timestep (int): The current timestep of the simulation.

        return_fpost (bool, optional): If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        T (jax.numpy.ndarray): The temperature field after the simulation step.

        f_poststreaming_tree (pytree of jax.numpy.ndarray): The post-streaming distribution functions after the simulation step.

        f_postcollision_tree (pytree of jax.numpy.ndarray or None): The post-collision distribution functions after the
        simulation step, or None if return_fpost is False.
        """
        T = self.precision_policy.cast_to_compute(T_prev)
        T = self.apply_bc(T, timestep)
        f_compute_tree = tree_map(lambda f: self.precision_policy.cast_to_compute(f), f_poststreaming_tree)
        rho_tree, _ = self.fluid_solver.update_macroscopic(f_compute_tree)
        u_tree = self.fluid_solver.macroscopic_velocity(f_compute_tree, rho_tree, T=T)

        f_poststreaming_tree, f_postcollision_tree = self.fluid_solver.step(f_poststreaming_tree, timestep, return_fpost, T)
        T = self._advance_temperature(T, timestep, rho_tree, u_tree)

        T = self.apply_bc(T, timestep)

        return self.precision_policy.cast_to_output(T), f_poststreaming_tree, f_postcollision_tree

    def update_macroscopic_output(self, f_tree, T):
        """
        Compute the component densities and force-corrected velocities for I/O.

        Parameters
        ----------
        f_tree (pytree of jax.numpy.ndarray): Post-streaming distribution functions.

        T (jax.numpy.ndarray): Temperature field.

        Returns
        -------
        rho_tree (pytree of jax.numpy.ndarray): Density field of all components.

        u_tree (pytree of jax.numpy.ndarray): Velocity field of all components.
        """
        f_tree = tree_map(lambda f: self.precision_policy.cast_to_compute(f), f_tree)
        rho_tree, _ = self.fluid_solver.update_macroscopic(f_tree)
        u_tree = self.fluid_solver.macroscopic_velocity(f_tree, rho_tree, T=self.precision_policy.cast_to_compute(T))
        return rho_tree, u_tree
