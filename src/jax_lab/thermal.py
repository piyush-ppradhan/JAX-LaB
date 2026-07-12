"""
Hybrid thermal LBM solver: the fluid is evolved with the lattice Boltzmann
method while the temperature field is evolved with a finite difference solver
on the same mesh, using lattice-based isotropic difference stencils and a
fourth order Runge-Kutta time integration. Throughout, dx = dt = 1 (lattice
units).

References
----------
1. Fei, L., Derome, D. & Carmeliet, J. Pore-scale study on the effect of
   heterogeneity on evaporation in porous media. Journal of Fluid Mechanics
   983, A6 (2024).
2. Kruger, T., et al. (2017). The lattice Boltzmann method. Springer
   International Publishing (isotropic lattice difference stencils).
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from jax import jit
from jax.experimental.multihost_utils import process_allgather
from termcolor import colored

from .utils import downsample_field


class ThermalBoundaryCondition(object):
    """
    Base class for thermal (temperature field) boundary conditions.

    Unlike the LBM boundary conditions in boundary_conditions.py which act on
    distribution functions, thermal boundary conditions act directly on the
    temperature field of the finite difference solver.

    Parameters
    ----------
    indices (tuple of numpy.ndarray): Tuple of index arrays selecting the boundary
    nodes, one array per spatial axis (e.g. tuple(wall_indices.T)).
    """

    def __init__(self, indices):
        self.indices = tuple(np.asarray(idx) for idx in indices)
        self.name = None

    def apply(self, T, timestep):
        """
        Apply the boundary condition to the temperature field.

        Parameters
        ----------
        T (jax.numpy.ndarray): Temperature field of shape (nx, ny, 1) in 2D or
        (nx, ny, nz, 1) in 3D.

        timestep (int): Current timestep, available for time dependent conditions.

        Returns
        -------
        jax.numpy.ndarray: Temperature field with the boundary condition applied.
        """
        raise NotImplementedError


class DirichletTemperature(ThermalBoundaryCondition):
    """
    Dirichlet (prescribed temperature) boundary condition: T = T_w at the
    boundary nodes.

    Parameters
    ----------
    indices (tuple of numpy.ndarray): Index arrays of the boundary nodes.

    prescribed (float or numpy.ndarray): Prescribed wall temperature. Either a
    scalar applied to all nodes or an array of shape (n, 1) with one value per
    boundary node.
    """

    def __init__(self, indices, prescribed):
        super().__init__(indices)
        self.name = "DirichletTemperature"
        self.prescribed = prescribed

    def apply(self, T, timestep):
        return T.at[self.indices].set(self.prescribed)


class NeumannTemperature(ThermalBoundaryCondition):
    """
    Neumann (prescribed normal temperature gradient) boundary condition,
    imposed with a first order one-sided difference over unit spacing:
    T_wall = T_interior + q, where q = dT/dn is the prescribed gradient along
    the outward normal (q = 0 gives an adiabatic wall).

    Assumes the interior neighbor of every boundary node lies one node along
    the negated outward normal. Corner nodes shared with a Dirichlet boundary
    should be listed in the Dirichlet condition as well, appended after this
    one, so the Dirichlet value takes precedence.

    Parameters
    ----------
    indices (tuple of numpy.ndarray): Index arrays of the boundary nodes.

    normal (sequence of int): Outward unit normal of the boundary, e.g. (0, 1)
    for the top wall in 2D or (0, 0, -1) for the bottom wall in 3D.

    prescribed (float or numpy.ndarray): Prescribed outward normal gradient.
    Either a scalar or an array of shape (n, 1). Defaults to 0 (adiabatic).
    """

    def __init__(self, indices, normal, prescribed=0.0):
        super().__init__(indices)
        self.name = "NeumannTemperature"
        self.prescribed = prescribed
        self.neighbor_indices = tuple(np.asarray(idx) - int(n) for idx, n in zip(self.indices, normal))

    def apply(self, T, timestep):
        return T.at[self.indices].set(T[self.neighbor_indices] + self.prescribed)


class Thermal(object):
    """
    Single phase hybrid thermal LBM solver. The fluid is advanced by the
    wrapped LBM fluid_solver while the temperature field is advanced by a
    finite difference solver on the same mesh.

    The temperature equation solved is:
    \\frac{\\partial T}{\\partial t} = -\\mathbf{u} \\cdot \\nabla T +
    \\frac{1}{\\rho c_v}(\\nabla \\cdot (K \\nabla T) + S_T)
    where S_T is a user defined source term (see source()).

    Spatial derivatives are evaluated with the isotropic lattice difference
    stencils in grad_x and laplacian_x and time integration uses the fourth
    order Runge-Kutta scheme with dt = 1.

    Parameters
    ----------
    fluid_solver (LBMBase): Configured fluid solver instance (e.g. BGKSim or
    MRTSim). Grid, lattice, precision and I/O settings are shared with it.

    specific_heat (float or numpy.ndarray): Specific heat c_v. Either a scalar
    or an array of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D.

    thermal_conductivity (float or numpy.ndarray): Thermal conductivity K with
    the same shape options as specific_heat.

    checkpoint_dir (str, optional): Directory for temperature checkpoints.
    Defaults to "./temperature_checkpoints".

    apply_buoyancy (bool, optional): If True, adds a density variation based
    buoyancy force to the fluid solver. Defaults to False.

    gravity (sequence of float, optional): Gravitational acceleration vector,
    required when apply_buoyancy is True.
    """

    def __init__(self, **kwargs):
        self.fluid_solver = kwargs.get("fluid_solver")
        if self.fluid_solver is None:
            raise ValueError("A configured fluid solver must be provided via the 'fluid_solver' keyword.")

        # Share grid, lattice, precision and run control settings with the fluid solver
        self.lattice = self.fluid_solver.lattice
        self.precisionPolicy = self.fluid_solver.precisionPolicy
        self.nx = self.fluid_solver.nx
        self.ny = self.fluid_solver.ny
        self.nz = self.fluid_solver.nz
        self.dim = self.fluid_solver.dim
        self.streaming = self.fluid_solver.streaming
        self.ioRate = self.fluid_solver.ioRate
        self.printInfoRate = self.fluid_solver.printInfoRate
        self.downsamplingFactor = self.fluid_solver.downsamplingFactor
        self.returnFpost = self.fluid_solver.returnFpost
        self.computeMLUPS = self.fluid_solver.computeMLUPS
        self.restore_checkpoint = self.fluid_solver.restore_checkpoint
        self.checkpointRate = self.fluid_solver.checkpointRate
        self.checkpointDir = kwargs.get("checkpoint_dir", "./temperature_checkpoints")
        self.nDevices = jax.device_count()
        self.backend = jax.default_backend()

        if self.checkpointRate > 0:
            mngr_options = orb.CheckpointManagerOptions(save_interval_steps=self.checkpointRate, max_to_keep=1)
            self.mngr = orb.CheckpointManager(self.checkpointDir, options=mngr_options)
        else:
            self.mngr = None

        self.c_v = kwargs.get("specific_heat")
        self.K = kwargs.get("thermal_conductivity")
        # K is constant in time, so its gradient is precomputed once
        self.grad_K = self.grad_x(self.K)

        self.apply_buoyancy = kwargs.get("apply_buoyancy", False)
        if self.apply_buoyancy:
            gravity = kwargs.get("gravity")
            if gravity is None:
                raise ValueError("gravity must be provided when apply_buoyancy is True.")
            self.gravity = jnp.array(np.array(gravity, dtype=np.float64), dtype=self.precisionPolicy.compute_dtype)
            # The fluid collision only invokes apply_force when a force is present
            if self.fluid_solver.force is None:
                self.fluid_solver.force = jnp.zeros(self.dim, dtype=self.precisionPolicy.compute_dtype)
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
            return jnp.array(value, dtype=self.precisionPolicy.compute_dtype)
        elif isinstance(value, (int, float)):
            return float(value) * jnp.ones(shape, dtype=self.precisionPolicy.compute_dtype)
        else:
            raise ValueError(f"Invalid type for {name}. It must be float or numpy.ndarray.")

    def set_thermal_boundary_conditions(self):
        """
        This function sets the boundary conditions for the temperature field.

        It is intended to be overwritten by the user to specify the boundary
        conditions according to the specific problem being solved, by appending
        ThermalBoundaryCondition instances (DirichletTemperature,
        NeumannTemperature) to self.thermal_BCs. Conditions are applied in
        list order, so later entries win on shared nodes (e.g. corners).

        By default no thermal boundary condition is applied, which corresponds
        to a fully periodic temperature field.
        """
        self.thermal_BCs = []

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_bc(self, T, timestep):
        """
        This function applies the boundary conditions to the temperature field.

        It iterates over all thermal boundary conditions and applies them in
        list order.

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
        This function initializes the temperature field to its default value of 1.

        Note: This function is a placeholder and should be overridden in a
        subclass or in an instance of the class to provide specific initial
        conditions.

        Returns
        -------
        None: The default temperature. This indicates that the actual value should be set elsewhere.
        """
        print("WARNING: Default initial condition assumed: temperature = 1")
        print("         To set an explicit initial temperature, use self.initialize_temperature_field.")
        return None

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

        It calls initialize_temperature_field, which can return a scalar or an
        array of shape (nx, ny, 1) in 2D or (nx, ny, nz, 1) in 3D. If it
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
            T0 = jnp.array(T0, dtype=self.precisionPolicy.output_dtype)

        T = self.distributed_array_init(shape, self.precisionPolicy.output_dtype, init_val=T0)

        return T

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force_thermal(self, f_postcollision, feq, rho, u):
        """
        Modified version of the single phase apply_force function that adds a
        density variation based buoyancy force to any user defined fluid force,
        using the exact-difference method due to Kupershtokh.

        Note: the buoyancy force is computed from the local density deviation
        relative to the mean density, not from the temperature field, since the
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
        c = jnp.array(self.lattice.c, dtype=self.precisionPolicy.compute_dtype).T
        return -self.lattice.inv_cs2 * jnp.dot(self.lattice.w * field_streamed, c)

    @partial(jit, static_argnums=(0,), inline=True)
    def laplacian_x(self, field):
        """
        Compute the laplacian of a scalar field using the isotropic lattice
        difference stencil:
        \\nabla^2 \\phi(x) = \\frac{2}{c_s^2} \\sum_i w_i [\\phi(x + \\mathbf{c}_i) - \\phi(x)]

        Note: streaming wraps periodically at the domain edges. On non-periodic
        boundaries the affected nodes must be corrected by the thermal boundary
        conditions.

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

        The advective term is not scaled by 1/(rho c_v); only the diffusive and
        source terms are, consistent with the temperature form of the energy
        equation (see reference 1 in the module docstring).

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

    @partial(jit, static_argnums=(0, 4), donate_argnums=(1,))
    def step(self, T_prev, f_poststreaming, timestep, return_fpost=False):
        """
        This function performs a single step of the hybrid thermal LBM simulation.

        It first advances the fluid solver by one LBM step, then advances the
        temperature field with the fourth order Runge-Kutta scheme (dt = 1)
        using the updated macroscopic density and velocity, and finally applies
        the thermal boundary conditions.

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
        f_poststreaming, f_postcollision = self.fluid_solver.step(f_poststreaming, timestep, return_fpost=return_fpost)
        rho, u = self.fluid_solver.update_macroscopic(self.precisionPolicy.cast_to_compute(f_poststreaming))

        T = self.precisionPolicy.cast_to_compute(T_prev)
        k1 = self.RHS(T, rho, u)
        k2 = self.RHS(T + 0.5 * k1, rho, u)
        k3 = self.RHS(T + 0.5 * k2, rho, u)
        k4 = self.RHS(T + k3, rho, u)
        T = T + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        T = self.apply_bc(T, timestep)

        return self.precisionPolicy.cast_to_output(T), f_poststreaming, f_postcollision

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
        start_step = 0
        if self.restore_checkpoint:
            assert self.mngr is not None, "Checkpoint manager does not exist."
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                try:
                    T = self.mngr.restore(latest_step, args=orb.args.StandardRestore({"T": T}))["T"]
                    f = self.fluid_solver.mngr.restore(latest_step, args=orb.args.StandardRestore({"f": f}))["f"]
                    print(f"Restored checkpoint at step {latest_step}.")
                except ValueError:
                    raise ValueError(f"Failed to restore checkpoint at step {latest_step}.")

                start_step = latest_step + 1
                if not (t_max > start_step):
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max = {t_max}). Consider increasing t_max.")

        if self.computeMLUPS:
            start = time.time()
        # Loop over all time steps
        for timestep in range(start_step, t_max + 1):
            io_flag = self.ioRate > 0 and (timestep % self.ioRate == 0 or timestep == t_max)
            print_iter_flag = self.printInfoRate > 0 and timestep % self.printInfoRate == 0
            checkpoint_flag = self.checkpointRate > 0 and timestep % self.checkpointRate == 0

            if io_flag:
                # Save the previous values of the macroscopic fields (for error computation)
                rho_prev, u_prev = self.fluid_solver.update_macroscopic(f)
                rho_prev = downsample_field(rho_prev, self.downsamplingFactor)
                u_prev = downsample_field(u_prev, self.downsamplingFactor)
                T_prev = downsample_field(T, self.downsamplingFactor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho_prev = process_allgather(rho_prev)
                u_prev = process_allgather(u_prev)
                T_prev = process_allgather(T_prev)

            # Perform one time-step (fluid step followed by the temperature update)
            T, f, fstar = self.step(T, f, timestep, return_fpost=self.returnFpost)

            # Print the progress of the simulation
            if print_iter_flag:
                print(
                    colored("Timestep ", "blue")
                    + colored(f"{timestep}", "green")
                    + colored(" of ", "blue")
                    + colored(f"{t_max}", "green")
                    + colored(" completed", "blue")
                )

            if io_flag:
                # Save the simulation data
                print(f"Saving data at timestep {timestep}/{t_max}")
                rho, u = self.fluid_solver.update_macroscopic(f)
                rho = downsample_field(rho, self.downsamplingFactor)
                u = downsample_field(u, self.downsamplingFactor)
                T_out = downsample_field(T, self.downsamplingFactor)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho = process_allgather(rho)
                u = process_allgather(u)
                T_out = process_allgather(T_out)

                # Save the data
                self.handle_io_timestep(timestep, f, fstar, T_out, rho, u, T_prev, rho_prev, u_prev)

            if checkpoint_flag:
                # Save the checkpoint
                print(f"Saving checkpoint at timestep {timestep}/{t_max}")
                self.mngr.save(timestep, args=orb.args.StandardSave({"T": T}))
                if self.fluid_solver.mngr is not None:
                    self.fluid_solver.mngr.save(timestep, args=orb.args.StandardSave({"f": f}))

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.computeMLUPS and timestep == 1:
                jax.block_until_ready(f)
                jax.block_until_ready(T)
                start = time.time()

        if self.computeMLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(T)
            jax.block_until_ready(f)
            end = time.time()
            n_voxels = self.nx * self.ny if self.dim == 2 else self.nx * self.ny * self.nz
            domain = f"{self.nx} x {self.ny}" if self.dim == 2 else f"{self.nx} x {self.ny} x {self.nz}"
            print(colored("Domain: ", "blue") + colored(domain, "green"))
            print(colored("Number of voxels: ", "blue") + colored(f"{n_voxels}", "green"))
            print(colored("MLUPS: ", "blue") + colored(f"{n_voxels * t_max / (end - start) / 1e6}", "red"))

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
