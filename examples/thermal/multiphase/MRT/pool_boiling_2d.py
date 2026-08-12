"""
Two-dimensional saturated pool-boiling simulation.

The thermal and buoyancy parameters follow Fei et al., "Mesoscopic simulation of three-dimensional pool boiling based on a phase-change cascaded lattice Boltzmann method",
Physics of Fluids 32, 103312 (2020). A saturated Peng-Robinson liquid pool is heated from below at Ja = 0.22. Conductivity is proportional to density and buoyancy acts on
the local density deviation from the domain average after a 1000-step gravity-free relaxation. A Gaussian temperature disturbance triggers nucleation at the first fluid layer.
The top and bottom are no-slip isothermal walls, while x is periodic.
"""

import operator
import os
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree import map as tree_map
from jax.tree import reduce

from jax_lab.core.boundary_conditions import BounceBack, DirichletTemperature
from jax_lab.core.eos import PengRobinson
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.multiphase import MultiphaseMRT
from jax_lab.core.thermal import MultiphaseThermal
from jax_lab.core.utils import save_fields_hdf5_xdmf

# from jax_lab.core.utils import save_fields_vtk

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_pool_boiling_2d")


class PoolFluid(MultiphaseMRT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # No-slip walls at the bottom and top; periodic in x. Wetting is
        # prescribed only on the heated bottom wall.
        bottom = tuple(self.bounding_box_indices["bottom"].T)
        top = tuple(self.bounding_box_indices["top"].T)
        for i in range(self.n_components):
            self.BCs[i].append(BounceBack(bottom, self.grid_info, self.precision_policy, theta=contact_angle))
            self.BCs[i].append(BounceBack(top, self.grid_info, self.precision_policy))

    def initialize_macroscopic_fields(self):
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        y = np.broadcast_to(y, (self.nx, self.ny))

        # Saturated liquid below 0.6H and saturated vapor above it.
        rho = np.where(y < h_liquid, rho_l, rho_g)
        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho_tree = [self.precision_policy.cast_to_output(rho)]

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precision_policy.compute_dtype, init_val=u)
        u_tree = [self.precision_policy.cast_to_output(u)]
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_buoyancy_force(self, rho_tree, timestep):
        """
        Compute density-relative buoyancy after interface relaxation.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Component density fields.
        timestep (int): Current simulation timestep.

        Returns
        -------
        pytree of jax.numpy.ndarray: Component buoyancy-force fields.
        """
        rho_average = self.compute_total_density(rho_tree).mean()
        gravity_vector = jnp.array([0.0, -gravity], dtype=self.precision_policy.compute_dtype)
        gravity_active = jnp.asarray(timestep > gravity_relaxation_steps, dtype=self.precision_policy.compute_dtype)
        return tree_map(lambda rho: gravity_active * (rho - rho_average) * gravity_vector, rho_tree)

    @partial(jit, static_argnums=(0,))
    def macroscopic_velocity(self, f_tree, rho_tree, T=None, timestep=None):
        """Compute force-corrected velocity with delayed buoyancy.

        Parameters
        ----------
        f_tree (pytree of jax.numpy.ndarray): Component populations.
        rho_tree (pytree of jax.numpy.ndarray): Component density fields.
        T (jax.numpy.ndarray, optional): Temperature field for the thermal EOS.
        timestep (int, optional): Current timestep; defaults to active gravity.

        Returns
        -------
        pytree of jax.numpy.ndarray: Component velocity fields.
        """
        u_tree = super().macroscopic_velocity(f_tree, rho_tree, T=T)
        if timestep is None:
            timestep = gravity_relaxation_steps + 1
        buoyancy_tree = self.compute_buoyancy_force(rho_tree, timestep)
        return tree_map(lambda u, force, rho: u + 0.5 * force / rho, u_tree, buoyancy_tree, rho_tree)

    @partial(jit, static_argnums=(0, 3), donate_argnums=(1,))
    def step(self, f_poststreaming_tree, timestep, return_fpost=False, T=None):
        """
        Advance the fluid and apply buoyancy only after relaxation.

        Parameters
        ----------
        f_poststreaming_tree (pytree of jax.numpy.ndarray): Component populations.
        timestep (int): Current simulation timestep.
        return_fpost (bool, optional): Whether to return post-collision populations.
        T (jax.numpy.ndarray, optional): Temperature field for the thermal EOS.

        Returns
        -------
        tuple: Post-streaming populations and optional post-collision populations.
        """
        f_postcollision_tree = self.collision(f_poststreaming_tree, T=T)
        rho_tree, u_tree = self.update_macroscopic(f_poststreaming_tree)
        buoyancy_tree = self.compute_buoyancy_force(rho_tree, timestep)
        u_forced_tree = tree_map(lambda u, force, rho: u + force / rho, u_tree, buoyancy_tree, rho_tree)
        feq_tree = self.equilibrium(rho_tree, u_tree, cast_output=False)
        feq_forced_tree = self.equilibrium(rho_tree, u_forced_tree, cast_output=False)
        f_postcollision_tree = tree_map(
            lambda f, feq_forced, feq: f + feq_forced - feq,
            f_postcollision_tree,
            feq_forced_tree,
            feq_tree,
        )
        f_postcollision_tree = tree_map(
            lambda f, rho: f.at[..., 0].set(rho[..., 0] - jnp.sum(f[..., 1:], axis=-1)),
            f_postcollision_tree,
            rho_tree,
        )
        f_postcollision_tree = self.apply_bc(f_postcollision_tree, f_poststreaming_tree, timestep, "PostCollision")
        f_poststreaming_tree = tree_map(self.streaming, f_postcollision_tree)
        f_poststreaming_tree = self.apply_bc(f_poststreaming_tree, f_postcollision_tree, timestep, "PostStreaming")
        return f_poststreaming_tree, f_postcollision_tree if return_fpost else None


class PoolBoiling(MultiphaseThermal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thermal_diffusivity_factor = kwargs.get("thermal_diffusivity_factor")
        self.initial_mass = None

    def initialize_temperature_field(self):
        temperature = T_sat * np.ones((nx, ny, 1))
        rng = np.random.default_rng(temperature_disturbance_seed)
        temperature[:, 1, 0] += rng.normal(0.0, temperature_disturbance_std * T_sat, size=nx)
        return temperature

    def set_thermal_boundary_conditions(self):
        self.thermal_BCs = []
        bbox = self.fluid_solver.bounding_box_indices
        # Superheated bottom wall and saturated top wall.
        self.thermal_BCs.append(DirichletTemperature(tuple(bbox["bottom"].T), prescribed=T_bottom))
        self.thermal_BCs.append(DirichletTemperature(tuple(bbox["top"].T), prescribed=T_top))

    @partial(jit, static_argnums=(0, 4), donate_argnums=(1,))
    def step(self, T_prev, f_poststreaming_tree, timestep, return_fpost=False):
        """
        Advance the coupled fields with timestep-aware buoyancy.

        Parameters
        ----------
        T_prev (jax.numpy.ndarray): Temperature field from the previous step.
        f_poststreaming_tree (pytree of jax.numpy.ndarray): Component populations.
        timestep (int): Current simulation timestep.
        return_fpost (bool, optional): Whether to return post-collision populations.

        Returns
        -------
        tuple: Temperature, post-streaming populations, and optional post-collision populations.
        """
        T = self.precision_policy.cast_to_compute(T_prev)
        T = self.apply_bc(T, timestep)
        f_compute_tree = tree_map(self.precision_policy.cast_to_compute, f_poststreaming_tree)
        rho_tree, _ = self.fluid_solver.update_macroscopic(f_compute_tree)
        u_tree = self.fluid_solver.macroscopic_velocity(f_compute_tree, rho_tree, T=T, timestep=timestep)

        f_poststreaming_tree, f_postcollision_tree = self.fluid_solver.step(f_poststreaming_tree, timestep, return_fpost, T)
        T = self._advance_temperature(T, timestep, rho_tree, u_tree)
        T = self.apply_bc(T, timestep)
        return self.precision_policy.cast_to_output(T), f_poststreaming_tree, f_postcollision_tree

    @partial(jit, static_argnums=(0,), inline=True)
    def RHS(self, T, rho_tree, u_tree):
        """
        Evaluate the phase-change energy equation with K = rho c_v chi.

        Parameters
        ----------
        T (jax.numpy.ndarray): Temperature field.
        rho_tree (pytree of jax.numpy.ndarray): Component density fields.
        u_tree (pytree of jax.numpy.ndarray): Component velocity fields.

        Returns
        -------
        jax.numpy.ndarray: Temperature time derivative.
        """
        grad_T = self.grad_x(T)
        rho = self.fluid_solver.compute_total_density(rho_tree)
        u = self.fluid_solver.compute_total_velocity(rho_tree, u_tree)
        # Force-corrected velocities on solid wall nodes are not fluid
        # velocities and must not enter advection or the phase-change source.
        u = u.at[:, (0, -1), :].set(0.0)
        conductivity = rho * self.thermal_diffusivity_factor

        advection = jnp.sum(u * grad_T, axis=-1, keepdims=True)
        diffusion = conductivity * self.laplacian_x(T) + jnp.sum(self.grad_x(conductivity) * grad_T, axis=-1, keepdims=True)
        div_u = self.divergence_x(u)
        dp_eos_dT = reduce(operator.add, self.fluid_solver.eos.dp_eos_dT(rho_tree, T))
        phase_change = T * dp_eos_dT * div_u
        return -advection + (diffusion - phase_change + self.source(T)) / (rho * self.c_v)

    def output_data(self, **kwargs):
        timestep = kwargs["timestep"]
        rho = np.array(kwargs["rho"][0][0, ..., 0])
        u = np.array(kwargs["u"][0][0, ...])
        T = np.array(kwargs["T"][0, ..., 0])
        # Do not export force-corrected diagnostic velocities on solid walls.
        u[:, (0, -1), :] = 0.0
        if not np.isfinite(rho).all() or not np.isfinite(T).all():
            print(f"Simulation diverged at timestep {timestep}.")
            self.stop_simulation = True
            return

        mass = float(rho.sum(dtype=np.float64))
        if self.initial_mass is None:
            self.initial_mass = mass
        relative_mass_error = abs(mass - self.initial_mass) / self.initial_mass
        vapor_fraction = float(np.mean(rho < 0.5 * (rho_l + rho_g)))
        fields = {"rho": rho, "u_x": u[..., 0], "u_y": u[..., 1], "T": T}
        save_fields_hdf5_xdmf(timestep, fields, output_dir, prefix="pool_boiling")
        # save_fields_vtk(timestep, fields, output_dir)
        print(
            f"timestep {timestep}: vapor fraction = {vapor_fraction:.4f}, "
            f"T/Tc = {T.min() / Tc:.4f}/{T.max() / Tc:.4f}, mass error = {relative_mass_error:.2e}"
        )


if __name__ == "__main__":
    precision = "f32/f32"
    nx = 400
    ny = 200

    h_liquid = int(0.6 * ny)
    temperature_disturbance_seed = 0
    temperature_disturbance_std = 0.07
    gravity_relaxation_steps = 1000
    contact_angle = np.deg2rad(60.0)

    # Peng-Robinson saturation state used by the reference pool-boiling case.
    a = 2 / 49
    b = 2 / 21
    R = 1.0
    pr_omega = 0.344
    Tc = 0.0729190375
    T_sat = 0.86 * Tc
    latent_heat = 0.3813
    specific_heat = 6.0
    jacob_number = 0.22
    T_bottom = T_sat + jacob_number * latent_heat / specific_heat
    T_top = T_sat

    rho_l = 6.499210784
    rho_g = 0.379598891

    gravity = 3e-5
    thermal_diffusivity_factor = 0.028
    shear_relaxation = 1.25

    # D2Q9 orthogonal moment basis (same as the isothermal MRT droplet_2d example)
    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((9, 9))
    M[0, :] = en**0
    M[1, :] = -4 * en**0 + 3 * en**2
    M[2, :] = 4 * en**0 - (21 / 2) * en**2 + (9 / 2) * en**4
    M[3, :] = e[:, 0]
    M[4, :] = (-5 * en**0 + 3 * en**2) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (-5 * en**0 + 3 * en**2) * e[:, 1]
    M[7, :] = e[:, 0] ** 2 - e[:, 1] ** 2
    M[8, :] = e[:, 0] * e[:, 1]

    eos = PengRobinson(a=[a], b=[b], R=[R], pr_omega=[pr_omega], temperature_field_type="thermal")
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "g_kkprime": -1.0 * np.ones((1, 1)),
        "EOS": eos,
        "body_force": None,
        "k": [0.27],
        "A": 0.01 * np.ones((1, 1)),
        "M": [M],
        "s_rho": [0.0],
        "s_e": [0.8],
        "s_eta": [1.0],
        "s_j": [0.0],
        "s_q": [1.0],
        "s_v": [shear_relaxation],
        "kappa": [0.0],
        "precision": precision,
        "io_rate": 1000,
        "print_info_rate": 1000,
        "checkpoint_rate": 0,
        "wetting_formulation": "geometric",
    }
    fluid = PoolFluid(**kwargs)

    thermal_kwargs = {
        "fluid_solver": fluid,
        "specific_heat": specific_heat,
        "thermal_conductivity": 1.0,
        "thermal_diffusivity_factor": thermal_diffusivity_factor,
        "rk_substeps": 2,
    }
    sim = PoolBoiling(**thermal_kwargs)
    sim.run(800000)
