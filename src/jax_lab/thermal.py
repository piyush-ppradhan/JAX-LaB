"""
Implementation of thermal LBM for arbitrary collision model
"""

from .base import LBMBase
from .lattice import LatticeD2Q9, LatticeD3Q19, LatticeD3Q27
from .multiphase import Multiphase
from .utils import downsample_field


from functools import partial
import jax
from jax import jit
from jax.experimental.multihost_utils import process_allgather
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from termcolor import colored
import time


class Thermal(LBMBase):
    """
    Single phase thermal LBM implementation. The base solver is based on LBMBase class with some modified functions to solve the energy equation.

    Changes with respect to LBMBase:

    1. User defines the boundary condition in self.set_thermal_boundary_conditions() by appending to the self.thermal_BCs list which is initially empty.
    2. An additional "fluid_solver" parameter is defined, which takes either a single phase LBM fluid solver.

    The current approach is based on:
    1. Fei, Linlin, and Kai Hong Luo. “Cascaded Lattice Boltzmann Method for Incompressible Thermal Flows with Heat Sources and General
    Thermal Boundary Conditions.” Computers & Fluids 165 (March 2018): 89-95. https://doi.org/10.1016/j.compfluid.2018.01.020.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fluid_solver = kwargs.get("fluid_solver")

    def set_thermal_boundary_conditions(self):
        """
        This function sets the boundary conditions for thermal simulation only.

        It is intended to be overwritten by the user to specify the boundary conditions according to
        the specific problem being solved.

        By default, it does nothing. When overwritten, it could set periodic boundaries, no-slip
        boundaries, inflow/outflow boundaries, etc.
        """
        return

    def _create_boundary_data(self):
        """
        Create boundary data for the Lattice Boltzmann simulation by setting boundary conditions,
        creating grid mask, and preparing local masks and normal arrays.
        """
        self.thermal_BCs = []
        self.set_thermal_boundary_conditions()
        # Accumulate the indices of all BCs to create the grid mask with FALSE along directions that
        # stream into a boundary voxel.
        solid_halo_list = [np.array(bc.indices).T for bc in self.thermal_BCs if bc.isSolid]
        solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

        # Create the grid mask on each process
        start = time.time()
        grid_mask = self.create_grid_mask(solid_halo_voxels)
        print("Time to create the grid mask for thermal lattice:", time.time() - start)

        start = time.time()
        for bc in self.thermal_BCs:
            assert bc.implementationStep in ["PostStreaming", "PostCollision"]
            bc.create_local_mask_and_normal_arrays(grid_mask)
        print("Time to create the local masks and normal arrays for thermal lattice:", time.time() - start)

    @partial(jit, static_argnums=(0,))
    def compute_temperature(self, g):
        """
        Compute the temperature field from temperature distributions.

        Parameters
        ----------
        g: jax.numpy.ndarray
            Temperature distribution.
        """
        return jnp.sum(g, axis=-1, keepdims=True)

    @partial(jit, static_argnums=(0,), donate_argnums=(1, 2))
    def thermal_collision(self, gin, u):
        """
        This function performs the collision step in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the collision operator according to
        the specific LBM model being used.

        By default, it does nothing. When overwritten, it could implement the BGK collision operator,
        the MRT collision operator, etc.

        Parameters
        ----------
        gin: jax.numpy.ndarray
            The pre-collision distribution functions.
        u: jax.numpy.ndarray
            The velocity field.

        Returns
        -------
        gin: jax.numpy.ndarray
            The post-collision distribution functions.
        """
        pass

    @partial(jit, static_argnums=(0, 4), inline=True)
    def apply_bc(self, gout, gin, timestep, implementation_step):
        """
        This function applies the boundary conditions to the distribution functions for thermal LBM.

        It iterates over all boundary conditions (BCs) and checks if the implementation step of the
        boundary condition matches the provided implementation step. If it does, it applies the
        boundary condition to the post-streaming distribution functions (fout).

        Parameters
        ----------
        gout: jax.numpy.ndarray
            The post-collision distribution functions.
        gin: jax.numpy.ndarray
            The post-streaming distribution functions.
        implementation_step: str
            The implementation step at which the boundary conditions should be applied.

        Returns
        -------
        jax.numpy.ndarray
            The output distribution functions after applying the boundary conditions.
        """
        for bc in self.thermal_BCs:
            gout = bc.prepare_populations(gout, gin, implementation_step)
            if bc.implementationStep == implementation_step:
                if bc.isDynamic:
                    gout = bc.apply(gout, gin, timestep)
                else:
                    gout = gout.at[bc.indices].set(bc.apply(gout, gin))

        return gout

    @partial(jit, static_argnums=(0,))
    def initialize_macroscopic_fields(self):
        """
        This function initializes the temperature distribution using prescribed temperature field and fluid velocities.
        The default temperature and density is 1.

        Note: This function is a placeholder and should be overridden in a subclass or in an instance of the class
        to provide specific initial conditions.

        Returns
        -------
            None: The default temperature. This indicates that the actual values should be set elsewhere.
        """
        print("WARNING: Default initial conditions assumed: density = 1, fluid velocity = 0")
        print("To set explicit initial temperature, density and velocity, use self.initialize_macroscopic_fields.")
        return None

    def assign_fields_sharded(self):
        """
        This function is used to initialize the simulation by assigning the macroscopic fields and populations.

        The function first initializes the macroscopic fields, which are the density (rho0) and velocity (u0).
        Depending on the dimension of the simulation (2D or 3D), it then sets the shape of the array that will hold the
        distribution functions (f).

        The fluid solver's initialize_macroscopic_field is utilized to intialize density and velocity field. If they are not defined, the simulation does not run.

        Parameters
        ----------
        None

        Returns
        -------
        g: a distributed JAX array of shape (nx, ny, nz, q) or (nx, ny, q) holding the temperature distribution functions for the simulation.
        """
        T0 = self.initialize_macroscopic_fields()
        rho0, u0 = self.fluid_solver.initialize_macroscopic_fields()

        if rho0 is None or u0 is None:
            colored("Error", color="red")
            raise ValueError("initialize_macroscopic_field is not defined for the fluid solver.")

        if self.dim == 2:
            shape = (self.nx, self.ny, self.lattice.q)
        if self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.lattice.q)

        if T0 is None:
            colored("Warning: initialize_macroscopic_field not defined, using Temperature = 1 as default.", "yellow")
            g = self.distributed_array_init(shape, self.precisionPolicy.output_dtype, init_val=self.w)
        else:
            g = self.initialize_populations(T0, u0)

        return g

    @partial(jit, static_argnums=(0, 4), donate_argnums=(1,))
    def step(self, g_poststreaming, u, timestep, return_gpost=False):
        """
        This function performs a single step of the thermal LBM simulation.

        It first performs the collision step, which is the relaxation of the distribution functions
        towards their equilibrium values. It then applies the respective boundary conditions to the
        post-collision distribution functions.

        The function then performs the streaming step, which is the propagation of the distribution
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming
        distribution functions.

        Parameters
        ----------
        g_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions.
        u: jax.numpy.ndarray
            The velocity field.
        timestep: int
            The current timestep of the simulation.
        return_gpost: bool, optional
            If True, the function also returns the post-collision distribution functions.

        Returns
        -------
        f_poststreaming: jax.numpy.ndarray
            The post-streaming distribution functions after the simulation step.
        f_postcollision: jax.numpy.ndarray or None
            The post-collision distribution functions after the simulation step, or None if
            return_gpost is False.
        """
        g_postcollision = self.thermal_collision(g_poststreaming, u)
        g_postcollision = self.apply_bc(g_postcollision, g_poststreaming, timestep, "PostCollision")
        g_poststreaming = self.streaming(g_postcollision)
        g_poststreaming = self.apply_bc(g_poststreaming, g_postcollision, timestep, "PostStreaming")

        if return_gpost:
            return g_poststreaming, g_postcollision
        else:
            return g_poststreaming, None

    def run(self, t_max):
        """
        This function runs the LBM simulation for a specified number of time steps.

        It first initializes the distribution functions and then enters a loop where it performs the
        simulation steps (collision, streaming, and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and
        compute the performance of the simulation in million lattice updates per second (MLUPS).

        Parameters
        ----------
        t_max: int
            The total number of time steps to run the simulation.
        Returns
        -------
        g: jax.numpy.ndarray
            The distribution functions for temperature after the simulation.
        """
        g = self.assign_fields_sharded()
        f = self.fluid_solver.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                # Assert that the checkpoint manager is not None
                assert self.mngr is not None, "Checkpoint manager does not exist."
                state = {"g": g}
                # shardings = map(lambda x: x.sharding, state)
                # restore_args = orb.checkpoint_utils.construct_restore_args(state, shardings)
                try:
                    # f = self.mngr.restore(latest_step, restore_kwargs={'restore_args': restore_args})['f']
                    g = self.mngr.restore(latest_step, args=orb.args.StandardSave(state))["g"]
                    f = self.mngr.restore(latest_step, args=orb.args.StandardSave(state))["f"]
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
                # Update the macroscopic variables and save the previous values (for error computation)
                rho_prev, u_prev = self.fluid_solver.update_macroscopic(f)
                rho_prev = downsample_field(rho_prev, self.fluid_solver.downsamplingFactor)
                u_prev = downsample_field(u_prev, self.fluid_solver.downsamplingFactor)
                T_prev = self.compute_temperature(g)
                T_prev = downsample_field(g, self.downsamplingFactor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho_prev = process_allgather(rho_prev)
                u_prev = process_allgather(u_prev)
                T_prev = process_allgather(T_prev)

            # Perform one time-step (collision, streaming, and boundary conditions)
            f, fstar = self.fluid_solver.step(f, timestep, return_fpost=self.fluid_solver.returnFpost)
            rho, u = self.fluid_solver.update_macroscopic(f)
            g, gstar = self.step(g, u, timestep, return_gpost=self.returnFpost)

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
                # rho, u = self.update_macroscopic(f)
                rho = downsample_field(rho, self.fluid_solver.downsamplingFactor)
                u = downsample_field(u, self.fluid_solver.downsamplingFactor)
                T = self.compute_temperature(g)
                T = downsample_field(T, self.downsamplingFactor)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho = process_allgather(rho)
                u = process_allgather(u)
                T = process_allgather(T)

                # Save the data
                self.handle_io_timestep(timestep, f, fstar, g, gstar, T, rho, u, T_prev, rho_prev, u_prev)

            if checkpoint_flag:
                # Save the checkpoint
                print(f"Saving checkpoint at timestep {timestep}/{t_max}")
                state = {"f": f}
                # self.mngr.save(timestep, state)
                self.mngr.save(timestep, args=orb.args.StandardSave(state))

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.computeMLUPS and timestep == 1:
                jax.block_until_ready(f)
                jax.block_until_ready(g)
                start = time.time()

        if self.computeMLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(g)
            jax.block_until_ready(f)
            end = time.time()
            if self.dim == 2:
                print(
                    colored("Domain: ", "blue") + colored(f"{self.nx} x {self.ny}", "green")
                    if self.dim == 2
                    else colored(f"{self.nx} x {self.ny} x {self.nz}", "green")
                )
                print(
                    colored("Number of voxels: ", "blue") + colored(f"{self.nx * self.ny}", "green")
                    if self.dim == 2
                    else colored(f"{self.nx * self.ny * self.nz}", "green")
                )
                print(colored("MLUPS: ", "blue") + colored(f"{2 * self.nx * self.ny * t_max / (end - start) / 1e6}", "red"))

            elif self.dim == 3:
                print(colored("Domain: ", "blue") + colored(f"{self.nx} x {self.ny} x {self.nz}", "green"))
                print(colored("Number of voxels: ", "blue") + colored(f"{self.nx * self.ny * self.nz}", "green"))
                print(
                    colored("MLUPS: ", "blue")
                    + colored(
                        f"{2 * self.nx * self.ny * self.nz * t_max / (end - start) / 1e6}",
                        "red",
                    )
                )
        if self.mngr is not None:
            self.mngr.wait_until_finished()
        return f

    def handle_io_timestep(self, timestep, f, fstar, g, gstar, T, rho, u, T_prev, rho_prev, u_prev):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep: int
            The current time step of the simulation.
        f: jax.numpy.ndarray
            The post-streaming distribution functions at the current time step.
        fstar: jax.numpy.ndarray
            The post-collision distribution functions at the current time step.
        g: jax.numpy.ndarray
            The post-streaming distribution functions for temperature at the current time step.
        gstar: jax.numpy.ndarray
            The post-collision distribution functions for temperature at the current time step.
        rho: jax.numpy.ndarray
            The density field at the current time step.
        u: jax.numpy.ndarray
            The velocity field at the current time step.
        T: jax.numpy.ndarray
            The temperature field at the current time step.
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
            "g_poststreaming": g,
            "g_postcollision": gstar,
        }
        self.output_data(**kwargs)


class BGKSim(Thermal):
    """
    BGK simulation class.

    This class implements the Bhatnagar-Gross-Krook (BGK) approximation for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def thermal_collision(self, g, u):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        g = self.precisionPolicy.cast_to_compute(g)
        temperature = self.compute_temperature(g)
        geq = self.equilibrium(temperature, u, cast_output=False)
        gneq = g - geq
        gout = g - self.omega * gneq
        # if self.force is not None:
        #     fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(gout)


class MRTSim(Thermal):
    """
    Multi-relaxation time model.
    """

    def __init__(self, **kwargs):
        kwargs.update({"omega": 1.0})
        super().__init__(**kwargs)
        self.s_rho = kwargs.get("s_rho")
        self.s_e = kwargs.get("s_e")
        self.s_eta = kwargs.get("s_eta")
        self.s_j = kwargs.get("s_j")
        self.s_q = kwargs.get("s_q")
        self.s_v = kwargs.get("s_v")
        self.M_inv = jnp.array(
            np.transpose(np.linalg.inv(kwargs.get("M"))),
            dtype=self.precisionPolicy.compute_dtype,
        )
        self.M = jnp.array(np.transpose(kwargs.get("M")), dtype=self.precisionPolicy.compute_dtype)
        if isinstance(self.lattice, LatticeD2Q9):
            self.S = jnp.array(
                np.diag([self.s_rho, self.s_e, self.s_eta, self.s_j, self.s_q, self.s_j, self.s_q, self.s_v, self.s_v]),
                dtype=self.precisionPolicy.compute_dtype,
            )
        elif isinstance(self.lattice, LatticeD3Q19):
            self.s_pi = kwargs.get("s_pi")
            self.s_m = kwargs.get("s_m")
            self.S = jnp.array(
                np.diag([
                    self.s_rho,
                    self.s_e,
                    self.s_eta,
                    self.s_j,
                    self.s_q,
                    self.s_j,
                    self.s_q,
                    self.s_j,
                    self.s_q,
                    self.s_v,
                    self.s_pi,
                    self.s_v,
                    self.s_pi,
                    self.s_v,
                    self.s_v,
                    self.s_v,
                    self.s_m,
                    self.s_m,
                    self.s_m,
                ]),
                dtype=self.precisionPolicy.compute_dtype,
            )

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def thermal_collision(self, g, u):
        """
        MRT collision step for lattice.

        Parameters
        ----------
        g: jax.numpy.ndarray
            Temperature distribution.
        u: jax.numpy.ndarray
            Velocity field.
        """
        g = self.precisionPolicy.cast_to_compute(g)
        m = jnp.dot(g, self.M)
        rho, u = self.update_macroscopic(g)
        geq = self.equilibrium(rho, u)
        meq = jnp.dot(geq, self.M)
        mout = -jnp.dot(m - meq, self.S)
        if self.force is not None:
            mout = self.apply_force(mout, meq, rho, u)
        return self.precisionPolicy.cast_to_output(g + jnp.dot(mout, self.M_inv))
