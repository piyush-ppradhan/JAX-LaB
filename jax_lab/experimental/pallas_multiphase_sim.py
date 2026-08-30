"""Experimental Pallas/SoA port of MultiphaseBGK. See todo.txt's "Multiphase Pallas
port plan (MultiphaseBGK first)" for the design and phased plan this implements.

Populations stay direction-major (SoA, (q, *spatial)) throughout step() - collision,
streaming, and boundary conditions - exactly like PallasBGK/PallasMRT, just as
a per-component list instead of one array. compute_potential and the neighbor-stencil
scalar_force_stencil are also Pallas-fused (see pallas_multiphase_kernels.py); collision
itself uses the mask-free mode of pallas_kernels.build_fused_soa_bgk_step(with_force=True),
one call per component, because boundary handling stays in the general SoA pipeline.
apply_contact_angle and compute_fluid_fluid_force's per-output-
component combine stay exactly as base Multiphase implements them (small, boundary-only
or already-cheap elementwise work - not dense per-node arithmetic, so not a natural fit
for a Pallas kernel), which is also what keeps every base constructor option working
unchanged: step, run, checkpointing, and every option other than EOS type,
wetting_formulation="geometric", and thermal coupling (all explicitly gated, not
silently mishandled - see __init__) are inherited from Multiphase/MultiphaseBGK verbatim.

Profiling before this file was written (see performance.txt) found base
MultiphaseBGK.collision() at ~19x the wall time of one Pallas single-phase collision
call for 2 components - apply_contact_angle, compute_potential, and
compute_fluid_fluid_force are each their own separate jax.jit boundary (materializing a
full pytree to HBM between each), on top of collision's own fin/feq/fneq/fout
materializations.
"""

import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from jax import lax
from jax.experimental.multihost_utils import process_allgather
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
from jax.tree import map as tree_map

from jax_lab.core.multiphase import MultiphaseBGK, MultiphaseMRT
from jax_lab.core.utils import colored, downsample_field

from .pallas_kernels import (
    build_fused_soa_bgk_step,
    build_fused_soa_mrt_step,
    build_fused_soa_mrt_surface_tension_step,
    build_soa_streaming,
)
from .pallas_multiphase_kernels import build_fused_soa_neighbor_stencil_step, build_fused_soa_potential_step, eos_pressure_fn
from .soa_boundary_conditions import SoABoundaryConditions, supports_soa_boundaries

logger = logging.getLogger(__name__)


class _ComponentBoundaryView:
    """Presents one Multiphase component's per-component BC data through the same
    attribute names SoABoundaryConditions expects from a single-phase simulation
    (whose self.BCs/self.wall_bc_data/etc. are flat, not per-component lists).

    SoABoundaryConditions only reads simulation.* attributes (never mutates them),
    so a thin read-only view is enough to reuse it unmodified, once per component.
    """

    def __init__(self, sim, component):
        self._sim = sim
        self.dim = sim.dim
        self.q = sim.q
        self.w = sim.w
        self.c = sim.c
        self.lattice = sim.lattice
        self.mesh = sim.mesh
        self.nx = sim.nx
        self.ny = sim.ny
        self.nz = sim.nz
        self.n_devices = sim.n_devices
        self.precision_policy = sim.precision_policy
        self.pallas_bc_block_size = sim.pallas_bc_block_size
        self.pallas_bc_num_warps = sim.pallas_bc_num_warps
        self.BCs = sim.BCs[component]
        self.local_bounceback_indices = sim.local_bounceback_indices[component]
        self.wall_bc_data = sim.wall_bc_data[component]
        self.solid_pin_indices = sim.solid_pin_indices[component]
        self.local_equilibrium_bc_indices = sim.local_equilibrium_bc_indices[component]
        self.local_equilibrium_bc_values = sim.local_equilibrium_bc_values[component]
        self.inlet_outlet_bc_data = sim.inlet_outlet_bc_data[component]

    def send_right(self, x, axis_name):
        return self._sim.send_right(x, axis_name)

    def send_left(self, x, axis_name):
        return self._sim.send_left(x, axis_name)

    def _split_local_indices(self, *args, **kwargs):
        return self._sim._split_local_indices(*args, **kwargs)

    def _distribute_local(self, *args, **kwargs):
        return self._sim._distribute_local(*args, **kwargs)


class PallasMultiphaseBGK(MultiphaseBGK):
    """MultiphaseBGK simulation using fused Pallas kernels for collision, potential,
    the fluid-fluid force neighbor stencil, streaming, and boundary conditions, on one
    or more GPUs.

    Parameters
    ----------
    **kwargs : dict
        Arguments accepted by MultiphaseBGK/Multiphase/LBMBase.

    Raises
    ------
    NotImplementedError
        If the simulation uses unsupported collision overrides, thermal EOS coupling, geometric wetting, a
        boundary condition without an SoA-native kernel (any component), or
        unsupported hardware. Use base MultiphaseBGK for any of these.
    """

    def __init__(self, **kwargs):
        self.pallas_block_size = kwargs.pop("pallas_block_size", 64)
        self.pallas_num_warps = kwargs.pop("pallas_num_warps", 2)
        self.pallas_bc_block_size = kwargs.pop("pallas_bc_block_size", 256)
        # See _PallasSoASimMixin's identical default in pallas_sim.py: 8 is tuned for
        # the native ZouHe/Regularized formula kernel, shared unchanged by multiphase's
        # per-component SoABoundaryConditions instances.
        self.pallas_bc_num_warps = kwargs.pop("pallas_bc_num_warps", 8)
        super().__init__(**kwargs)
        if type(self).collision is not PallasMultiphaseBGK.collision:
            raise NotImplementedError("PallasMultiphaseBGK requires standard MultiphaseBGK.collision behavior.")
        if self.eos.temperature_field_type != "isothermal":
            raise NotImplementedError("PallasMultiphaseBGK currently supports isothermal EOS only (no thermal coupling).")
        if self.wetting_formulation == "geometric":
            raise NotImplementedError(
                "PallasMultiphaseBGK does not yet support wetting_formulation='geometric' "
                "(its post-collision mass-rescatter step is not ported); use 'improved_virtual_density' or None."
            )
        for component, BCs in enumerate(self.BCs):
            if not supports_soa_boundaries(BCs):
                raise NotImplementedError(
                    f"PallasMultiphaseBGK requires every component's boundary conditions to have an SoA-native "
                    f"kernel; component {component} has an unsupported boundary condition type. Use base MultiphaseBGK."
                )

        local_shape = (self.nx // self.n_devices, self.ny)
        if self.dim == 3:
            local_shape += (self.nz,)

        rho_spec = PartitionSpec("x", *([None] * self.dim))
        g_diagonal = np.asarray(self.g_kkprime).diagonal()
        surface_components = getattr(self, "_surface_tension_components", (False,) * self.n_components)
        self._potential_needs_psi = tuple(
            self._psi_stencil_components[component]
            or any(self._psi_interactions[component])
            or surface_components[component]
            or not (any(self._psi_interactions[component]) or any(self._U_interactions[component]))
            for component in range(self.n_components)
        )
        self._potential_needs_u = self._U_stencil_components
        self._pallas_potential = []
        for component in range(self.n_components):
            needs_psi = self._potential_needs_psi[component]
            needs_u = self._potential_needs_u[component]
            if not needs_psi and not needs_u:
                self._pallas_potential.append(None)
                continue
            local_potential = build_fused_soa_potential_step(
                self.lattice,
                local_shape,
                self.precision_policy,
                eos_pressure_fn(self.eos, component),
                float(self.k[component]),
                float(g_diagonal[component]),
                block_size=self.pallas_block_size,
                num_warps=self.pallas_num_warps,
                allow_multi_device_local=True,
                return_psi=needs_psi,
                return_u=needs_u,
            )
            self._pallas_potential.append(
                jax.jit(
                    shard_map(
                        local_potential,
                        mesh=self.mesh,
                        in_specs=rho_spec,
                        out_specs=(rho_spec if needs_psi else None, rho_spec if needs_u else None),
                        check_rep=False,
                    )
                )
            )

        x_needs_halo = self.n_devices > 1

        def build_stencil_callable(weights):
            """One shard_map-wrapped Pallas neighbor-stencil callable for a given
            (out_channels, q) weight array - shared halo-wrapping logic for both the
            vector-weighted force stencil and the scalar-weighted wetting/average-
            density stencil below (same underlying _neighbor_stencil_m structure
            base Multiphase itself shares between scalar_force_stencil and
            scalar_neighbor_sum, just with different weights)."""
            local_stencil = build_fused_soa_neighbor_stencil_step(
                self.lattice,
                local_shape,
                self.precision_policy,
                weights,
                block_size=self.pallas_block_size,
                num_warps=self.pallas_num_warps,
                allow_multi_device_local=True,
            )

            def stencil_body(local_field):
                if x_needs_halo:
                    left_halo = self.send_right(local_field[-1:], "x")
                    right_halo = self.send_left(local_field[:1], "x")
                    padded = jnp.concatenate((left_halo, local_field, right_halo), axis=0)
                    return local_stencil(padded, True)
                return local_stencil(local_field, False)

            return jax.jit(shard_map(stencil_body, mesh=self.mesh, in_specs=rho_spec, out_specs=rho_spec, check_rep=False))

        # Overrides Multiphase.__init__'s own self.scalar_force_stencil (same calling
        # convention: field -> weighted neighbor-direction sum), so compute_fluid_fluid_force
        # (inherited, unchanged) transparently uses this Pallas version instead.
        weights_vector = np.asarray(self.G_ff)[None, :] * np.asarray(self.c)
        self.scalar_force_stencil = build_stencil_callable(weights_vector)

        # Same override for self.scalar_neighbor_sum, used by compute_average_density
        # (in turn the dominant cost of apply_contact_angle - confirmed by direct
        # profiling, not assumed: compute_average_density alone accounted for ~106% of
        # apply_contact_angle's isolated wall-clock time, i.e. it - not the boundary-
        # only local wetting scatter apply_contact_angle also does - is essentially the
        # whole cost). base only builds scalar_neighbor_sum (and the None check below
        # mirrors it) when wetting_formulation="improved_virtual_density" actually needs
        # it; None otherwise, matching base's own behavior with no override needed.
        if self.scalar_neighbor_sum is not None:
            weights_scalar = np.asarray(self.G_ff)[None, :]
            self.scalar_neighbor_sum = build_stencil_callable(weights_scalar)

        local_collision = self._build_local_pallas_collisions(local_shape)
        surface_components = getattr(self, "_surface_tension_components", (False,) * self.n_components)
        self._pallas_collision = [
            self._build_distributed_collision(collide, surface_tension=has_surface_tension)
            for collide, has_surface_tension in zip(local_collision, surface_components, strict=True)
        ]

        # Streaming and BC application replace Multiphase's own self.streaming (an
        # instance attribute, same override mechanism as scalar_force_stencil above) and
        # this class's own apply_bc (see below) - both SoA-native, one per component
        # (streaming shares a single kernel, since domain shape doesn't vary by
        # component; boundary conditions get one SoABoundaryConditions per component,
        # since BCs themselves do).
        local_streaming = build_soa_streaming(
            self.lattice,
            local_shape,
            self.precision_policy,
            block_size=self.pallas_block_size,
            num_warps=self.pallas_num_warps,
            allow_multi_device_local=True,
        )
        self.streaming = self._build_soa_streaming(local_streaming)
        self._soa_boundaries = [SoABoundaryConditions(_ComponentBoundaryView(self, component)) for component in range(self.n_components)]

    def _build_local_pallas_collisions(self, local_shape):
        """Build one fused BGK collision kernel per component."""
        return [
            build_fused_soa_bgk_step(
                self.lattice,
                local_shape,
                self.precision_policy,
                float(self.omega[component]),
                block_size=self.pallas_block_size,
                num_warps=self.pallas_num_warps,
                allow_multi_device_local=True,
                with_force=True,
                force_is_acceleration=False,
                streaming=False,
                use_solid_mask=False,
            )
            for component in range(self.n_components)
        ]

    def _build_distributed_collision(self, local_collision, *, surface_tension=False):
        """Wrap one component's shard-local fused Pallas collision+force kernel."""
        field_spec = PartitionSpec(None, "x", *([None] * (self.dim - 1)))
        force_spec = PartitionSpec("x", *([None] * (self.dim - 1)), None)

        if surface_tension:
            psi_spec = PartitionSpec("x", *([None] * self.dim))

            def collide_surface(field, force, psi):
                if self.n_devices > 1:
                    left_halo = self.send_right(psi[-1:], "x")
                    right_halo = self.send_left(psi[:1], "x")
                    psi = jnp.concatenate((left_halo, psi, right_halo), axis=0)
                return local_collision(field, force, psi)

            return jax.jit(
                shard_map(
                    collide_surface,
                    mesh=self.mesh,
                    in_specs=(field_spec, force_spec, psi_spec),
                    out_specs=field_spec,
                    check_rep=False,
                )
            )

        def collide(field, force):
            return local_collision(field, force)

        return jax.jit(
            shard_map(
                collide,
                mesh=self.mesh,
                in_specs=(field_spec, force_spec),
                out_specs=field_spec,
                check_rep=False,
            )
        )

    def _build_soa_streaming(self, local_streaming):
        """Build native Pallas streaming with general X halo exchange (shared by every
        component - domain shape only, no component-specific state)."""
        if self.n_devices == 1:
            return local_streaming

        field_spec = PartitionSpec(None, "x", *([None] * (self.dim - 1)))
        boundary_slices = (slice(None),) * (self.dim - 1)
        right_indices = jnp.asarray(self.lattice.right_indices, dtype=jnp.int32)
        left_indices = jnp.asarray(self.lattice.left_indices, dtype=jnp.int32)
        local_nx = self.nx // self.n_devices

        def stream_and_exchange(field):
            streamed = local_streaming(field)
            send_right = streamed[(right_indices, 0, *boundary_slices)]
            send_left = streamed[(left_indices, local_nx - 1, *boundary_slices)]
            receive_left = jax.lax.ppermute(send_right, axis_name="x", perm=self.right_perm)
            receive_right = jax.lax.ppermute(send_left, axis_name="x", perm=self.left_perm)
            streamed = streamed.at[(right_indices, 0, *boundary_slices)].set(receive_left)
            return streamed.at[(left_indices, local_nx - 1, *boundary_slices)].set(receive_right)

        return jax.jit(shard_map(stream_and_exchange, mesh=self.mesh, in_specs=field_spec, out_specs=field_spec, check_rep=False))

    @staticmethod
    def to_soa(populations):
        """Convert public population layout (*spatial, q) to (q, *spatial)."""
        return jnp.moveaxis(populations, -1, 0)

    @staticmethod
    def to_aos(populations):
        """Convert internal population layout (q, *spatial) to (*spatial, q)."""
        return jnp.moveaxis(populations, 0, -1)

    def assign_fields_sharded(self):
        """Initialize direction-major (SoA) populations per component."""
        return [self.to_soa(f) for f in super().assign_fields_sharded()]

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def _compute_density(self, f_tree):
        """Density only, per component - see the comment at its call site in
        collision() for why this exists instead of always calling
        update_macroscopic (which also computes velocity via an einsum against the
        lattice velocity matrix, needed nowhere in that caller)."""
        return [jnp.sum(self.precision_policy.cast_to_compute(f), axis=0)[..., None] for f in f_tree]

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def update_macroscopic(self, f_tree):
        """Compute macroscopic fields from direction-major populations, per component."""
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        rho_tree = []
        u_tree = []
        for f in f_tree:
            f = self.precision_policy.cast_to_compute(f)
            rho = jnp.sum(f, axis=0)[..., None]
            u = jnp.einsum("q...,dq->...d", f, c) / rho
            rho_tree.append(rho)
            u_tree.append(u)
        return rho_tree, u_tree

    @partial(jax.jit, static_argnums=(0, 4))
    def apply_bc(self, fout_tree, fin_tree, timestep, implementation_step):
        """Apply each component's boundary conditions to direction-major populations."""
        return [
            self._soa_boundaries[component].apply(fout, fin, implementation_step, timestep)
            for component, (fout, fin) in enumerate(zip(fout_tree, fin_tree, strict=True))
        ]

    @partial(jax.jit, static_argnums=(0,))
    def compute_potential(self, rho_tree, T=None):
        """Per-component Shan-Chen/Zhang-Chen potential, fused into one Pallas kernel
        call per component instead of one shared, separately-jitted call over the whole
        pytree (see build_fused_soa_potential_step)."""
        if T is not None:
            raise NotImplementedError("PallasMultiphaseBGK does not support thermal EOS coupling yet.")
        psi_tree = []
        u_tree = []
        for component in range(self.n_components):
            potential = self._pallas_potential[component]
            if potential is None:
                psi_tree.append(None)
                u_tree.append(None)
                continue
            psi, u = potential(rho_tree[component])
            psi_tree.append(psi)
            u_tree.append(u)
        return psi_tree, u_tree

    @partial(jax.jit, static_argnums=(0,))
    def collision(self, fin_tree, T=None):
        """BGK collision step, extended to pytrees, with the per-node arithmetic
        (equilibrium, BGK relaxation, exact-difference force application) fused into one
        Pallas kernel call per component instead of MultiphaseBGK.collision's separate
        feq/fneq/fout/apply_force JAX ops.

        Reuses apply_contact_angle and compute_fluid_fluid_force's per-output-component
        combine (compute_force -> apply_force's caller chain) unchanged - only
        compute_potential and the neighbor-stencil scalar_force_stencil, the two
        dense-grid steps in that chain, are Pallas-fused (see __init__).
        """
        # Keep populations in storage dtype. The Pallas collision kernels cast each
        # loaded direction to compute dtype, avoiding a full-size converted copy.
        # fin_tree = [self.precision_policy.cast_to_compute(f) for f in fin_tree]
        # Only rho feeds the force pipeline (compute_potential/compute_force never use
        # velocity - confirmed by reading compute_potential's body), and the fused
        # Pallas collision kernel recomputes u internally from raw populations anyway
        # for its own equilibrium calculation - so density-only here skips
        # update_macroscopic's einsum against the lattice velocity matrix entirely,
        # rather than computing u_tree just to discard it.
        rho_tree = self._compute_density(fin_tree)
        surface_components = getattr(self, "_surface_tension_components", (False,) * self.n_components)
        if any(surface_components):
            force_tree, psi_tree, _ = self._compute_force_fields(rho_tree, T=T)
        else:
            force_tree = self.compute_force(rho_tree, T=T)
            psi_tree = (None,) * self.n_components
        fout_tree = []
        for component in range(self.n_components):
            args = (fin_tree[component], force_tree[component])
            if surface_components[component]:
                args += (psi_tree[component],)
            fout_tree.append(self._pallas_collision[component](*args))
        return [self.precision_policy.cast_to_output(fout) for fout in fout_tree]

    def handle_io_timestep(self, timestep, f_tree, fstar_tree, p_tree, p_total, u_tree, u_total, rho_total, rho_tree):
        """Send direction-major (SoA, per component (q, *spatial)) populations to the I/O path.

        No AoS conversion here, same reasoning as _PallasSoASimMixin.handle_io_timestep:
        a user-overridden output_data can read (q, *spatial) per component directly, or
        call self.to_aos(...) itself if it needs (*spatial, q).
        """
        super().handle_io_timestep(
            timestep,
            f_tree,
            fstar_tree,
            p_tree,
            p_total,
            u_tree,
            u_total,
            rho_total,
            rho_tree,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _run_fused_chunk(self, f_tree, lo, hi):
        """Run steps [lo, hi) as one jit(lax.fori_loop(...)) call instead of hi - lo eager
        dispatches. Same technique and rationale as _PallasSoASimMixin._run_fused_chunk in
        pallas_sim.py (see its docstring and performance.txt's BC-penalty investigation);
        multiphase has even more separately-jitted per-step calls (potential, the force
        neighbor stencil, and collision are all per-component) than single-phase, so the
        eager-dispatch penalty this avoids is at least as large there.
        """

        def body(timestep, f_tree):
            return self.step(f_tree, timestep)[0]

        return lax.fori_loop(lo, hi, body, f_tree)

    def run(self, t_max):
        """Run with internal direction-major state, batching dispatch between io/print/checkpoint events.

        Matches Multiphase.run's checkpoint-restore path, io/print/checkpoint timing, and
        final result exactly; spans of consecutive timesteps needing no io/print/checkpoint
        action run through _run_fused_chunk instead of Multiphase.run's eager per-timestep
        self.step() dispatch. See _PallasSoASimMixin.run (pallas_sim.py) for the same
        technique applied to the single-phase case.
        """
        f_tree = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:
                assert self.mngr is not None, "Checkpoint manager does not exist."
                c_name = lambda i: f"component_{i}"
                restore_target = lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=self.sharding)
                state = jax.tree.map(restore_target, {c_name(i): f_tree[i] for i in range(self.n_components)})
                try:
                    restored_state = self.mngr.restore(latest_step, args=orb.args.StandardRestore(state))
                    f_tree = [restored_state[c_name(i)] for i in range(self.n_components)]
                    logger.info(f"Restored checkpoint at step {latest_step}.")
                except ValueError:
                    raise ValueError(f"Failed to restore checkpoint at step {latest_step}.")
                start_step = latest_step + 1
                if not (t_max > start_step):
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max = {t_max}). Consider increasing t_max.")

        def is_event(timestep):
            return (
                (self.io_rate > 0 and (timestep % self.io_rate == 0 or timestep == t_max))
                or (self.print_info_rate > 0 and timestep % self.print_info_rate == 0)
                or (self.checkpoint_rate > 0 and timestep % self.checkpoint_rate == 0)
            )

        boundaries = {timestep for timestep in range(start_step, t_max + 1) if is_event(timestep)}
        if self.compute_MLUPS and start_step + 2 <= t_max:
            # Non-event boundary: keeps the first chunk small so MLUPS timing can start
            # right after it, excluding _run_fused_chunk's own one-time compilation - see
            # _PallasSoASimMixin.run for the identical reasoning.
            boundaries.add(start_step + 2)
        boundaries.add(t_max + 1)

        start_time = None
        timestep = start_step
        for event in sorted(boundaries):
            if event > timestep:
                f_tree = self._run_fused_chunk(f_tree, timestep, event)
                timestep = event
            if timestep > t_max:
                break

            io_flag = self.io_rate > 0 and (timestep % self.io_rate == 0 or timestep == t_max)
            print_iter_flag = self.print_info_rate > 0 and timestep % self.print_info_rate == 0
            checkpoint_flag = self.checkpoint_rate > 0 and timestep % self.checkpoint_rate == 0

            # Matches Multiphase.run: return_fpost is not threaded through here either.
            f_tree, fstar_tree = self.step(f_tree, timestep)

            if print_iter_flag:
                logger.info(
                    colored("Timestep ", "blue")
                    + colored(f"{timestep}", "green")
                    + colored(" of ", "blue")
                    + colored(f"{t_max}", "green")
                    + colored(" completed", "blue")
                )

            if io_flag:
                logger.info(f"Saving data at timestep {timestep}/{t_max}")
                rho_tree, _ = self.update_macroscopic(f_tree)
                # macroscopic_velocity is inherited from Multiphase unchanged and expects
                # AoS (*spatial, q) populations; f_tree stays SoA (q, *spatial) internally
                # until the final to_aos below, same as handle_io_timestep's own conversion.
                u_tree = self.macroscopic_velocity([self.to_aos(f) for f in f_tree], rho_tree)
                psi_tree, _ = self.compute_potential(rho_tree)
                p_tree = self.compute_pressure(rho_tree, psi_tree)
                p_total = self.compute_total_pressure(p_tree, rho_tree)
                p_total = downsample_field(p_total, self.downsampling_factor)
                rho_tree = tree_map(lambda rho: downsample_field(rho, self.downsampling_factor), rho_tree)
                u_tree = tree_map(lambda u: downsample_field(u, self.downsampling_factor), u_tree)
                rho_total = self.compute_total_density(rho_tree)
                u_total = self.compute_total_velocity(rho_tree, u_tree)
                p_total = process_allgather(p_total)
                rho_tree = tree_map(lambda rho: process_allgather(rho), rho_tree)
                u_tree = tree_map(lambda u: process_allgather(u), u_tree)
                rho_total = process_allgather(rho_total)
                u_total = process_allgather(u_total)
                self.handle_io_timestep(timestep, f_tree, fstar_tree, p_tree, p_total, u_tree, u_total, rho_total, rho_tree)

            if checkpoint_flag:
                logger.info(f"Saving checkpoint at timestep {timestep}/{t_max}")
                c_name = lambda i: f"component_{i}"
                state = {c_name(i): f_tree[i] for i in range(self.n_components)}
                self.mngr.save(timestep, args=orb.args.StandardSave(state))

            if self.compute_MLUPS and start_time is None and timestep >= start_step + 1:
                jax.block_until_ready(f_tree)
                start_time = time.time()

            timestep += 1

        if self.compute_MLUPS:
            jax.block_until_ready(f_tree)
            end_time = time.time()
            if start_time is None:
                start_time = end_time
            voxels = self.nx * self.ny * (self.nz if self.dim == 3 else 1)
            logger.info(
                colored("Domain: ", "blue")
                + colored(f"{self.nx} x {self.ny} x {self.nz}" if self.dim == 3 else f"{self.nx} x {self.ny}", "green")
            )
            logger.info(colored("Number of voxels: ", "blue") + colored(f"{voxels}", "green"))
            logger.info(
                colored("MLUPS: ", "blue") + colored(f"{self.n_components * voxels * t_max / (end_time - start_time) / 1e6}", "red")
            )

        if self.mngr is not None:
            self.mngr.wait_until_finished()

        return [self.to_aos(f) for f in f_tree]


class PallasMultiphaseMRT(PallasMultiphaseBGK, MultiphaseMRT):
    """Multiphase MRT using the Pallas SoA potential, force, collision and streaming path.

    Parameters
    ----------
    **kwargs : dict
        Arguments accepted by :class:jax_lab.core.multiphase.MultiphaseMRT.

    Raises
    ------
    NotImplementedError
        If a custom equilibrium, geometric wetting, thermal EOS, unsupported
        boundary, lattice, or hardware is requested. Nonzero kappa uses a
        separate fused surface-tension kernel for D2Q9 and D3Q19.
    """

    def _build_local_pallas_collisions(self, local_shape):
        if not self._uses_default_mrt_equilibrium:
            raise NotImplementedError("PallasMultiphaseMRT requires the default multiphase equilibrium.")
        collisions = []
        for component in range(self.n_components):
            common = {
                "block_size": self.pallas_block_size,
                "num_warps": self.pallas_num_warps,
                "allow_multi_device_local": True,
            }
            if self._surface_tension_components[component]:
                collisions.append(
                    build_fused_soa_mrt_surface_tension_step(
                        self.lattice,
                        local_shape,
                        self.precision_policy,
                        self.collision_terms[component],
                        kappa=self.kappa[component],
                        A=np.asarray(self.A)[component, component],
                        s_e=self.s_e[component],
                        s_eta=self.s_eta[component],
                        s_v=self.s_v[component],
                        g_ff=self.G_ff,
                        m_inv=self.M_inv[component],
                        x_needs_halo=self.n_devices > 1,
                        use_solid_mask=False,
                        **common,
                    )
                )
            else:
                collisions.append(
                    build_fused_soa_mrt_step(
                        self.lattice,
                        local_shape,
                        self.precision_policy,
                        self.collision_terms[component],
                        with_force=True,
                        force_is_acceleration=False,
                        streaming=False,
                        use_solid_mask=False,
                        **common,
                    )
                )
        return collisions
