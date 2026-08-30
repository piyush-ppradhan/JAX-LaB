"""Experimental LBMBase-compatible direction-major Pallas simulations."""

import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
import orbax.checkpoint as orb
from jax import lax
from jax.experimental.multihost_utils import process_allgather
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec

from jax_lab.core.models import BGKSim, MRTSim
from jax_lab.core.utils import colored, downsample_field

from .pallas_kernels import build_fused_soa_bgk_step, build_fused_soa_mrt_step, build_soa_streaming
from .soa_boundary_conditions import SoABoundaryConditions, supports_soa_boundaries

logger = logging.getLogger(__name__)


class _PallasSoASimMixin:
    """Shared direction-major (SoA) machinery for Pallas single-phase collision models.

    A concrete subclass (alongside a collision model such as BGKSim or
    MRTSim) provides _build_local_pallas_collision(local_shape) and calls
    self._setup_soa() at the end of its __init__, after super().__init__.
    Everything else - SoA streaming, boundary dispatch, step, I/O, and
    run - is collision-model-independent and lives here once.
    """

    def _setup_soa(self):
        """Build SoA sharding, streaming, boundaries, and the Pallas collision callable.

        Must run after the base model's __init__ (so self.BCs, self.mesh,
        self.nx/self.n_devices, etc. are populated) and after any
        collision-compatibility check the concrete subclass performs.
        """
        self._aos_sharding = self.sharding
        self._aos_streaming = self.streaming

        self._uses_soa_boundaries = supports_soa_boundaries(self.BCs)

        self._soa_sharding = NamedSharding(
            self.mesh,
            PartitionSpec(None, "x", *([None] * (self.dim - 1))),
        )
        self._soa_streaming = self._build_soa_streaming()
        self._soa_boundaries = SoABoundaryConditions(self) if self._uses_soa_boundaries else None
        local_shape = (self.nx // self.n_devices, self.ny)
        if self.dim == 3:
            local_shape += (self.nz,)
        local_collision = self._build_local_pallas_collision(local_shape)
        self._pallas_collision = (
            local_collision if self.n_devices == 1 else self._build_distributed_collision(local_collision, self.force is not None)
        )

    @staticmethod
    def to_soa(populations):
        """Convert public population layout (*spatial, q) to (q, *spatial)."""
        return jnp.moveaxis(populations, -1, 0)

    @staticmethod
    def to_aos(populations):
        """Convert internal population layout (q, *spatial) to (*spatial, q)."""
        return jnp.moveaxis(populations, 0, -1)

    def assign_fields_sharded(self):
        """Initialize direction-major populations without an AoS staging buffer.

        Builds the initial condition inside a jax.jit with out_shardings set
        to the target SoA sharding, instead of computing a full, unsharded,
        single-device array and reharding it afterward with jax.device_put:
        letting XLA's SPMD partitioner produce each device's shard directly
        avoids a transient full-domain buffer (previously the dominant
        contributor to peak VRAM, dwarfing steady-state population storage;
        see performance.txt).
        """
        self.sharding = self._aos_sharding
        rho0, u0 = self.initialize_macroscopic_fields()
        shape = (self.q, self.nx, self.ny)
        if self.dim == 3:
            shape += (self.nz,)
        weights = self.w.reshape((self.q,) + (1,) * self.dim)
        output_dtype = self.precision_policy.output_dtype

        if rho0 is None or u0 is None:

            def build(weights):
                return jnp.broadcast_to(weights, shape).astype(output_dtype)

            result = jax.jit(build, out_shardings=self._soa_sharding)(weights)
        else:

            def build(rho0, u0, weights):
                rho0, u0 = self.precision_policy.cast_to_compute((rho0, u0))
                c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
                cu = 3.0 * jnp.einsum("...d,dq->q...", u0, c)
                usqr = 1.5 * jnp.sum(jnp.square(u0), axis=-1)
                equilibrium = rho0[..., 0][None, ...] * weights * (1.0 + cu * (1.0 + 0.5 * cu) - usqr[None, ...])
                return self.precision_policy.cast_to_output(equilibrium)

            result = jax.jit(build, out_shardings=self._soa_sharding)(rho0, u0, weights)

        self.sharding = self._soa_sharding
        return result

    def _build_distributed_collision(self, local_collision, with_force):
        """Wrap a shard-local Pallas collision without BC-specific dispatch."""
        field_spec = PartitionSpec(None, "x", *([None] * (self.dim - 1)))
        force_spec = PartitionSpec("x", *([None] * (self.dim - 1)), None)

        def collide(field, force=None):
            return local_collision(field, force) if with_force else local_collision(field)

        input_specs = (field_spec, force_spec) if with_force else (field_spec,)
        return jax.jit(shard_map(collide, mesh=self.mesh, in_specs=input_specs, out_specs=field_spec, check_rep=False))

    def _build_soa_streaming(self):
        """Build native Pallas streaming with general X halo exchange."""
        local_shape = (self.nx // self.n_devices, self.ny)
        if self.dim == 3:
            local_shape += (self.nz,)
        local_streaming = build_soa_streaming(
            self.lattice,
            local_shape,
            self.precision_policy,
            block_size=self.pallas_block_size,
            num_warps=self.pallas_num_warps,
            allow_multi_device_local=True,
        )
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
            receive_left = lax.ppermute(send_right, axis_name="x", perm=self.right_perm)
            receive_right = lax.ppermute(send_left, axis_name="x", perm=self.left_perm)
            streamed = streamed.at[(right_indices, 0, *boundary_slices)].set(receive_left)
            return streamed.at[(left_indices, local_nx - 1, *boundary_slices)].set(receive_right)

        return jax.jit(shard_map(stream_and_exchange, mesh=self.mesh, in_specs=field_spec, out_specs=field_spec, check_rep=False))

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def update_macroscopic(self, f):
        """Compute macroscopic fields from direction-major populations."""
        f = self.precision_policy.cast_to_compute(f)
        rho = jnp.sum(f, axis=0)[..., None]
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        u = jnp.einsum("q...,dq->...d", f, c) / rho
        return rho, u

    def apply_bc(self, fout, fin, timestep, implementation_step):
        """Apply supported base boundary conditions to direction-major populations."""
        return self.to_soa(
            super().apply_bc(
                self.to_aos(fout),
                self.to_aos(fin),
                timestep,
                implementation_step,
            )
        )

    def step(self, f_poststreaming, timestep, return_fpost=False):
        """Advance one collision step, retaining all LBMBase boundary conditions.

        Pallas collision, boundary application, and streaming always follow
        the LBMBase order. Native SoA boundaries stay direction-major;
        unported boundaries use the equivalent AoS base fallback.

        Parameters
        ----------
        f_poststreaming : jax.Array
            Direction-major post-streaming populations.

        timestep : int
            Current simulation timestep.

        return_fpost : bool, optional
            Return post-collision populations when True.

        Returns
        -------
        tuple[jax.Array, jax.Array | None]
            Direction-major post-streaming and optional post-collision fields.
        """
        force = self.get_force() if self.force is not None else None
        f_postcollision = self._pallas_collision(f_poststreaming, force) if force is not None else self._pallas_collision(f_poststreaming)
        if self._uses_soa_boundaries:
            f_postcollision = self._soa_boundaries.apply(f_postcollision, f_poststreaming, "PostCollision", timestep)
            f_poststreaming = self._soa_streaming(f_postcollision)
            f_poststreaming = self._soa_boundaries.apply(f_poststreaming, f_postcollision, "PostStreaming", timestep)
            return f_poststreaming, f_postcollision if return_fpost else None

        f_postcollision = self.apply_bc(f_postcollision, f_poststreaming, timestep, "PostCollision")
        f_poststreaming = self.to_soa(self._aos_streaming(self.to_aos(f_postcollision)))
        f_poststreaming = self.apply_bc(f_poststreaming, f_postcollision, timestep, "PostStreaming")
        return f_poststreaming, f_postcollision if return_fpost else None

    def handle_io_timestep(self, timestep, f, fstar, rho, u, rho_prev, u_prev):
        """Send direction-major (SoA, (q, *spatial)) populations to output_data.

        No AoS conversion here: a user-overridden output_data can read (q, *spatial)
        directly (e.g. per-direction slicing, checkpointing), or call self.to_aos(...)
        itself if it specifically needs (*spatial, q). Forcing the conversion here would
        materialize a full transposed copy of every io_rate hit even when the override
        never needs AoS layout at all.
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

    @partial(jax.jit, static_argnums=(0,))
    def _run_fused_chunk(self, f, lo, hi):
        """Run steps [lo, hi) as one jit(lax.fori_loop(...)) call instead of hi - lo eager dispatches.

        lo/hi are ordinary (non-static) traced ints, so this compiles once regardless of how
        many differently-sized spans run() splits a simulation into between io/print/checkpoint
        timesteps - not once per span. Only used for spans with no per-timestep host-side work.
        """

        def body(timestep, f):
            return self.step(f, timestep)[0]

        return lax.fori_loop(lo, hi, body, f)

    def run(self, t_max):
        """Run with internal direction-major state, batching dispatch between io/print/checkpoint events.

        LBMBase.run dispatches self.step() eagerly once per timestep. Each of collision, every
        active boundary condition, and streaming is its own separately-jitted call, so once real
        per-kernel GPU work drops under ~1ms, Python/XLA dispatch overhead dominates - measured to
        cost close to one whole periodic-step's time per active boundary condition regardless of
        how few nodes it touches (see performance.txt). Spans of consecutive timesteps needing no
        io/print/checkpoint action run through _run_fused_chunk instead, paying that dispatch
        overhead once per span rather than once per timestep. io/print/checkpoint timing, the
        checkpoint-restore path, and the final result otherwise match LBMBase.run exactly.
        """
        f = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:
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

        def is_event(timestep):
            return (
                (self.io_rate > 0 and (timestep % self.io_rate == 0 or timestep == t_max))
                or (self.print_info_rate > 0 and timestep % self.print_info_rate == 0)
                or (self.checkpoint_rate > 0 and timestep % self.checkpoint_rate == 0)
            )

        boundaries = {timestep for timestep in range(start_step, t_max + 1) if is_event(timestep)}
        if self.compute_MLUPS and start_step + 2 <= t_max:
            # A synthetic (non-event) boundary: forces the first chunk to be small, so MLUPS
            # timing can start right after it - excluding _run_fused_chunk's own one-time
            # compilation the same way LBMBase.run excludes self.step's - instead of timing a
            # single chunk that could cover the whole run.
            boundaries.add(start_step + 2)
        boundaries.add(t_max + 1)

        start_time = None
        timestep = start_step
        for event in sorted(boundaries):
            if event > timestep:
                f = self._run_fused_chunk(f, timestep, event)
                timestep = event
            if timestep > t_max:
                break

            io_flag = self.io_rate > 0 and (timestep % self.io_rate == 0 or timestep == t_max)
            print_iter_flag = self.print_info_rate > 0 and timestep % self.print_info_rate == 0
            checkpoint_flag = self.checkpoint_rate > 0 and timestep % self.checkpoint_rate == 0

            if io_flag:
                rho_prev, u_prev = self.update_macroscopic(f)
                rho_prev = downsample_field(rho_prev, self.downsampling_factor)
                u_prev = downsample_field(u_prev, self.downsampling_factor)
                rho_prev = process_allgather(rho_prev)
                u_prev = process_allgather(u_prev)

            f, fstar = self.step(f, timestep, return_fpost=self.return_fpost)

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
                rho, u = self.update_macroscopic(f)
                rho = downsample_field(rho, self.downsampling_factor)
                u = downsample_field(u, self.downsampling_factor)
                rho = process_allgather(rho)
                u = process_allgather(u)
                self.handle_io_timestep(timestep, f, fstar, rho, u, rho_prev, u_prev)

            if checkpoint_flag:
                logger.info(f"Saving checkpoint at timestep {timestep}/{t_max}")
                self.mngr.save(timestep, args=orb.args.StandardSave({"f": f}))

            if self.compute_MLUPS and start_time is None and timestep >= start_step + 1:
                jax.block_until_ready(f)
                start_time = time.time()

            timestep += 1

        if self.compute_MLUPS:
            jax.block_until_ready(f)
            end_time = time.time()
            if start_time is None:
                start_time = end_time
            cells = self.nx * self.ny * (self.nz if self.dim == 3 else 1)
            logger.info(
                colored("Domain: ", "blue")
                + colored(f"{self.nx} x {self.ny} x {self.nz}" if self.dim == 3 else f"{self.nx} x {self.ny}", "green")
            )
            logger.info(colored("Number of voxels: ", "blue") + colored(f"{cells}", "green"))
            logger.info(colored("MLUPS: ", "blue") + colored(f"{cells * t_max / (end_time - start_time) / 1e6}", "red"))

        if self.mngr is not None:
            self.mngr.wait_until_finished()

        return self.to_aos(f)


class PallasBGK(_PallasSoASimMixin, BGKSim):
    """BGK simulation using direction-major Pallas kernels on one or more GPUs.

    run, initialization, I/O, checkpoints, macroscopic updates, and output
    callbacks retain LBMBase semantics. Populations remain direction-major
    internally and are converted to (nx, ny[, nz], q) for public output.
    Multi-GPU runs shard X and exchange only crossing population directions.

    Parameters
    ----------
    **kwargs : dict
        Arguments accepted by BGKSim and LBMBase.

    Raises
    ------
    NotImplementedError
        If simulation uses unsupported collision overrides or hardware.
    """

    def __init__(self, **kwargs):
        self.pallas_block_size = kwargs.pop("pallas_block_size", 64)
        self.pallas_num_warps = kwargs.pop("pallas_num_warps", 2)
        self.pallas_bc_block_size = kwargs.pop("pallas_bc_block_size", 256)
        # 8 is tuned for the native ZouHe/Regularized formula kernel (1->8 warps: 3277
        # -> 4709 MLUPS, +44%, 200^3 f32/f16 cavity); simpler BC kernels (sparse scatter,
        # wall types) are much less warp-sensitive (<2% and ~1% respectively) so this is
        # a net win as a shared default. User override remains for anyone who profiles a
        # different optimum for their own BC mix.
        self.pallas_bc_num_warps = kwargs.pop("pallas_bc_num_warps", 8)
        super().__init__(**kwargs)
        if type(self).collision is not PallasBGK.collision:
            raise NotImplementedError("PallasBGK requires standard BGKSim.collision behavior.")
        self._setup_soa()

    def _build_local_pallas_collision(self, local_shape):
        return build_fused_soa_bgk_step(
            self.lattice,
            local_shape,
            self.precision_policy,
            self.omega,
            block_size=self.pallas_block_size,
            num_warps=self.pallas_num_warps,
            allow_multi_device_local=True,
            with_force=self.force is not None,
            streaming=False,
            use_solid_mask=False,
        )

    @partial(jax.jit, static_argnums=(0,))
    def collision(self, f):
        """Apply standard BGK collision to direction-major populations."""
        # Keep populations in storage dtype. Reductions and arithmetic below cast
        # through their compute-dtype operands without a full converted copy.
        # f = self.precision_policy.cast_to_compute(f)
        rho, u = self.update_macroscopic(f)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cu = jnp.moveaxis(3.0 * jnp.dot(u, c), -1, 0)
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=-1)
        rho = rho[..., 0]
        weights = self.w.reshape((self.q,) + (1,) * self.dim)
        equilibrium = rho[None, ...] * weights * (1.0 + cu * (1.0 + 0.5 * cu) - usqr[None, ...])
        fout = f - self.omega * (f - equilibrium)
        if self.force is not None:
            du = self.precision_policy.cast_to_compute(self.get_force())
            cu_force = 3.0 * jnp.dot(u, c)
            dcu = 3.0 * jnp.dot(du, c)
            delta_usqr = 1.5 * (2.0 * jnp.sum(u * du, axis=-1, keepdims=True) + jnp.sum(jnp.square(du), axis=-1, keepdims=True))
            delta_feq = rho[..., None] * self.w * (dcu * (1.0 + cu_force + 0.5 * dcu) - delta_usqr)
            fout = fout + jnp.moveaxis(delta_feq, -1, 0)
        return self.precision_policy.cast_to_output(fout)


class PallasMRT(_PallasSoASimMixin, MRTSim):
    """MRT (multi-relaxation-time) simulation using direction-major Pallas kernels.

    Same public behavior as PallasBGK (see its docstring), but drives the
    fused Pallas collision from MRTSim.collision_terms - the sparse
    per-output-direction (input_direction, coefficient) decomposition of
    M @ S @ M_inv that MRTSim.__init__ already computes - instead of a
    single scalar omega. With an identity M and a uniform S diagonal,
    collision_terms degenerates to exactly one (direction, omega) term
    per direction, reproducing PallasBGK exactly.

    Parameters
    ----------
    **kwargs : dict
        Arguments accepted by MRTSim and LBMBase (including M and the
        s_* relaxation-rate kwargs MRTSim expects).

    Raises
    ------
    NotImplementedError
        If simulation uses unsupported collision overrides or hardware.
    """

    def __init__(self, **kwargs):
        self.pallas_block_size = kwargs.pop("pallas_block_size", 64)
        self.pallas_num_warps = kwargs.pop("pallas_num_warps", 2)
        self.pallas_bc_block_size = kwargs.pop("pallas_bc_block_size", 256)
        # 8 is tuned for the native ZouHe/Regularized formula kernel (1->8 warps: 3277
        # -> 4709 MLUPS, +44%, 200^3 f32/f16 cavity); simpler BC kernels (sparse scatter,
        # wall types) are much less warp-sensitive (<2% and ~1% respectively) so this is
        # a net win as a shared default. User override remains for anyone who profiles a
        # different optimum for their own BC mix.
        self.pallas_bc_num_warps = kwargs.pop("pallas_bc_num_warps", 8)
        super().__init__(**kwargs)
        if type(self).collision is not MRTSim.collision:
            raise NotImplementedError("PallasMRT requires standard MRTSim.collision behavior.")
        self._setup_soa()

    def _build_local_pallas_collision(self, local_shape):
        return build_fused_soa_mrt_step(
            self.lattice,
            local_shape,
            self.precision_policy,
            self.collision_terms,
            block_size=self.pallas_block_size,
            num_warps=self.pallas_num_warps,
            allow_multi_device_local=True,
            with_force=self.force is not None,
            streaming=False,
            use_solid_mask=False,
        )
