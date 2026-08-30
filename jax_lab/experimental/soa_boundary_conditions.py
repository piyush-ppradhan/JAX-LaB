"""SoA-native single-phase boundary-condition kernels.

The formulas are shared with :mod:jax_lab.core.boundary_conditions; only
the direction-major gather/scatter layout is experimental here.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec

from jax_lab.core.boundary_conditions import (
    INLET_OUTLET_BC_TYPES,
    NEQ_BC_TYPES,
    WALL_BC_TYPES,
    BounceBack,
    BounceBackMoving,
    BounceBackHalfway,
    ConvectiveOutflow,
    DoNothing,
    EquilibriumBC,
    ExactNonEquilibriumExtrapolation,
    ExtrapolationOutflow,
    ExtrapolationOutflowMultiphase,
    InterpolatedBounceBackBouzidi,
    InterpolatedBounceBackDifferentiable,
    NonEquilibriumExtrapolation,
    Regularized,
    ZouHe,
    _construct_symmetric_lattice_moment,
)

SUPPORTED_BC_TYPES = frozenset((
    BounceBack,
    BounceBackMoving,
    BounceBackHalfway,
    InterpolatedBounceBackBouzidi,
    InterpolatedBounceBackDifferentiable,
    DoNothing,
    EquilibriumBC,
    ZouHe,
    Regularized,
    ConvectiveOutflow,
    NonEquilibriumExtrapolation,
    ExactNonEquilibriumExtrapolation,
    ExtrapolationOutflow,
    ExtrapolationOutflowMultiphase,
))


def supports_soa_boundaries(boundaries):
    """Return whether every boundary has an SoA-native implementation."""
    return all(type(boundary) in SUPPORTED_BC_TYPES for boundary in boundaries)


class SoABoundaryConditions:
    """Apply selected LBMBase boundaries directly to (q, *spatial) fields.

    Parameters
    ----------
    simulation : PallasBGK
        Initialized simulation supplying preprocessed local boundary data.
    """

    def __init__(self, simulation):
        self.simulation = simulation
        self.dim = simulation.dim
        self.q = simulation.q
        self.w = jnp.asarray(simulation.w, dtype=simulation.precision_policy.compute_dtype)
        self.c = jnp.asarray(simulation.c, dtype=simulation.precision_policy.compute_dtype)
        self.cc = jnp.asarray(simulation.lattice.cc, dtype=simulation.precision_policy.compute_dtype)
        self._sparse_scatter_kernels = {}
        field_spec = PartitionSpec(None, "x", *([None] * (self.dim - 1)))
        auxiliary_spec = PartitionSpec("x", None, None)
        self._bounceback = self._build_bounceback(field_spec, auxiliary_spec)
        self._moving_bounceback = self._build_moving_bounceback(field_spec)
        self._solid_pin = self._build_solid_pin(field_spec, auxiliary_spec)
        self._equilibrium = self._build_equilibrium(field_spec, auxiliary_spec)
        self._do_nothing = self._build_do_nothing(field_spec, auxiliary_spec)
        self._wall_kernels = self._build_wall_kernels(field_spec, auxiliary_spec)
        self._inlet_outlet_kernels = self._build_inlet_outlet_kernels(field_spec, auxiliary_spec)
        self._do_nothing_indices = self._make_do_nothing_indices()

        # Neighbor-reading outflow boundaries (priority work #2): local/halo-aware kernels.
        self._neq_data = self._make_neq_data()
        self._neq_kernels = self._build_neq_kernels(field_spec, auxiliary_spec)
        self._convective_data = self._make_convective_data()
        self._convective_kernels = self._build_convective_kernels(field_spec, auxiliary_spec)
        self._extrapolation_data = self._make_extrapolation_data()
        self._extrapolation_kernels = self._build_extrapolation_kernels(field_spec, auxiliary_spec)
        self._multiphase_extrapolation_data = self._make_multiphase_extrapolation_data()
        self._multiphase_extrapolation_kernels = self._build_multiphase_extrapolation_kernels(field_spec, auxiliary_spec)

    def _make_do_nothing_indices(self):
        indices = [np.asarray(boundary.indices, dtype=np.int32).T for boundary in self.simulation.BCs if type(boundary) is DoNothing]
        if not indices:
            return None
        local_indices, _ = self.simulation._split_local_indices(np.vstack(indices))
        return self.simulation._distribute_local(local_indices, jnp.int32)

    @staticmethod
    def _indices(local_indices, dim):
        local_indices = local_indices[0]
        return tuple(local_indices[:, axis] for axis in range(dim))

    def _sparse_scatter(self, fout, local_indices, values):
        """Scatter direction-major boundary values without copying the full field."""
        max_rows = local_indices.shape[1]
        scatter = self._sparse_scatter_kernels.get(max_rows)
        if scatter is None:
            block_size = self.simulation.pallas_bc_block_size
            local_shape = (self.simulation.nx // self.simulation.n_devices, self.simulation.ny)
            if self.dim == 3:
                local_shape += (self.simulation.nz,)
            field_shape = (self.q, *local_shape)
            output_dtype = self.simulation.precision_policy.output_dtype

            def kernel(field_ref, indices_ref, values_ref, out_ref):
                row = pl.program_id(0) * block_size + jnp.arange(block_size)
                valid_row = row < max_rows
                coordinates = [
                    pltriton.load(indices_ref.at[row, axis], mask=valid_row, other=(local_shape[0] if axis == 0 else 0))
                    for axis in range(self.dim)
                ]
                # >= 0 excludes rows a *different* shard owns when a caller (BounceBackMoving)
                # localizes one replicated global index list per device instead of pre-padding
                # with the >= local_shape[0] sentinel _split_local_indices uses.
                active = valid_row & (coordinates[0] >= 0) & (coordinates[0] < local_shape[0])
                for direction in range(self.q):
                    value = pltriton.load(values_ref.at[row, direction], mask=active, other=0.0)
                    pltriton.store(out_ref.at[(direction, *coordinates)], value, mask=active)

            scatter = jax.jit(
                pl.pallas_call(
                    kernel,
                    out_shape=jax.ShapeDtypeStruct(field_shape, output_dtype),
                    grid=((max_rows + block_size - 1) // block_size,),
                    input_output_aliases={0: 0},
                    compiler_params=pltriton.CompilerParams(num_warps=self.simulation.pallas_bc_num_warps),
                    name=f"{self.simulation.lattice.name.lower()}_soa_sparse_bc_scatter",
                )
            )
            self._sparse_scatter_kernels[max_rows] = scatter
        return scatter(fout, local_indices[0], values)

    def _has_unique_indices(self, bc_type, value_type):
        """Return whether one grouped inlet/outlet write has no duplicate nodes."""
        matches = [
            boundary
            for boundary in self.simulation.BCs
            if type(boundary) is bc_type and boundary.type == value_type
        ]
        if not matches:
            return True
        indices = np.vstack([np.asarray(boundary.indices, dtype=np.int32).T for boundary in matches])
        return np.unique(indices, axis=0).shape[0] == indices.shape[0]

    def _build_bounceback(self, field_spec, auxiliary_spec):
        """Fully Pallas-native BounceBack (full-way), fused with the aliased scatter -
        same technique as `_build_native_wall_kernel`: one kernel does the gather (own
        node, every direction) and scatter together instead of gather (JAX) -> permute
        (JAX) -> `_sparse_scatter` (Pallas). BounceBack's formula is just a static
        (compile-time-known) direction permutation - `fin[opp_indices]`, no reduction and
        no dynamic per-row direction lookup - so there is no early-cast ULP trap concern
        here at all (unlike ZouHe/NEQ's reductions); the value is simply cast to
        output_dtype at the point base's own `.get()` gather would have inherited it.
        """
        q = self.q
        dim = self.dim
        output_dtype = self.simulation.precision_policy.output_dtype
        local_shape = (self.simulation.nx // self.simulation.n_devices, self.simulation.ny)
        if dim == 3:
            local_shape += (self.simulation.nz,)
        opposites = tuple(int(value) for value in np.asarray(self.simulation.lattice.opp_indices))

        def apply(fout, fin, local_indices):
            max_rows = local_indices.shape[1]
            block_size = self.simulation.pallas_bc_block_size

            def kernel(field_ref, fin_ref, indices_ref, out_ref):
                row = pl.program_id(0) * block_size + jnp.arange(block_size)
                valid_row = row < max_rows
                x = pltriton.load(indices_ref.at[row, 0], mask=valid_row, other=local_shape[0])
                coordinates = [x] + [pltriton.load(indices_ref.at[row, axis], mask=valid_row, other=0) for axis in range(1, dim)]
                active = valid_row & (x < local_shape[0])
                fin_bd = [pltriton.load(fin_ref.at[(direction, *coordinates)], mask=active, other=0.0) for direction in range(q)]
                for direction in range(q):
                    value = fin_bd[opposites[direction]].astype(output_dtype)
                    pltriton.store(out_ref.at[(direction, *coordinates)], value, mask=active)

            field_shape = (q, *local_shape)
            local = jax.jit(
                pl.pallas_call(
                    kernel,
                    out_shape=jax.ShapeDtypeStruct(field_shape, output_dtype),
                    grid=((max_rows + block_size - 1) // block_size,),
                    input_output_aliases={0: 0},
                    compiler_params=pltriton.CompilerParams(num_warps=self.simulation.pallas_bc_num_warps),
                    name=f"{self.simulation.lattice.name.lower()}_soa_native_bounceback",
                )
            )
            return local(fout, fin, local_indices[0])

        return jax.jit(
            shard_map(
                apply,
                mesh=self.simulation.mesh,
                in_specs=(field_spec, field_spec, auxiliary_spec),
                out_specs=field_spec,
                check_rep=False,
            )
        )

    def _build_solid_pin(self, field_spec, auxiliary_spec):
        rest = self.simulation.precision_policy.cast_to_output(jnp.asarray(self.simulation.w))

        def apply(fout, local_indices):
            values = jnp.broadcast_to(rest[None, :], (local_indices.shape[1], self.q))
            return self._sparse_scatter(fout, local_indices, values)

        return jax.jit(
            shard_map(
                apply,
                mesh=self.simulation.mesh,
                in_specs=(field_spec, auxiliary_spec),
                out_specs=field_spec,
                check_rep=False,
            )
        )

    def _build_moving_bounceback(self, field_spec):
        """Build a dynamic moving-wall kernel using global replicated indices."""
        local_nx = self.simulation.nx // self.simulation.n_devices
        opposites = jnp.asarray(self.simulation.lattice.opp_indices, dtype=jnp.int32)
        output_dtype = self.simulation.precision_policy.output_dtype

        def apply(fout, fin, global_indices, velocity):
            local_indices = global_indices.at[:, 0].add(-jax.lax.axis_index("x") * local_nx)[None, ...]
            indices = self._indices(local_indices, self.dim)
            gather_indices = (slice(None), *indices)
            incoming = jnp.moveaxis(fin.at[gather_indices].get(mode="fill", fill_value=0.0), 0, -1)
            correction = 6.0 * self.w[None, :] * jnp.dot(velocity, self.c)
            boundary = (incoming[:, opposites] - correction).astype(output_dtype)
            # boundary is already (n, q): local_indices carries out-of-shard rows as negative or
            # >= local_shape[0] x, which _sparse_scatter's active mask now excludes on both sides.
            return self._sparse_scatter(fout, local_indices, boundary)

        return jax.jit(
            shard_map(
                apply,
                mesh=self.simulation.mesh,
                in_specs=(field_spec, field_spec, PartitionSpec(), PartitionSpec()),
                out_specs=field_spec,
                check_rep=False,
            )
        )

    def _build_equilibrium(self, field_spec, auxiliary_spec):
        output_dtype = self.simulation.precision_policy.output_dtype

        def apply(fout, local_indices, local_out):
            values = local_out[0].astype(output_dtype)
            return self._sparse_scatter(fout, local_indices, values)

        return jax.jit(
            shard_map(
                apply,
                mesh=self.simulation.mesh,
                in_specs=(field_spec, auxiliary_spec, auxiliary_spec),
                out_specs=field_spec,
                check_rep=False,
            )
        )

    def _build_do_nothing(self, field_spec, auxiliary_spec):
        def apply(fout, fin, local_indices):
            gather_indices = (slice(None), *self._indices(local_indices, self.dim))
            values = jnp.moveaxis(fin.at[gather_indices].get(mode="fill", fill_value=0.0), 0, -1)
            return self._sparse_scatter(fout, local_indices, values)

        return jax.jit(
            shard_map(
                apply,
                mesh=self.simulation.mesh,
                in_specs=(field_spec, field_spec, auxiliary_spec),
                out_specs=field_spec,
                check_rep=False,
            )
        )

    def _build_native_wall_kernel(self, bc_type):
        """Fully Pallas-native BounceBackHalfway/InterpolatedBounceBack{Bouzidi,Differentiable}
        formula, fused with the aliased scatter - same technique and rationale as
        `_build_native_zouhe_kernel` (see its docstring): one kernel does the gather,
        formula math, and scatter together instead of gather (JAX) -> formula (JAX) ->
        `_sparse_scatter` (Pallas). `velocity_correction` is applied unconditionally
        (vel=0 is a documented safe no-op for BCs with no prescribed velocity), matching
        `_halfway_wall_math`/`_bouzidi_wall_math`/`_differentiable_wall_math`'s own
        unconditional call - no per-instance branch needed. Unlike the ZouHe/Regularized
        density-sum reduction, none of these formulas sum many native-output-dtype terms
        without a compute-dtype operand already forcing promotion (Bouzidi/Differentiable's
        `weights` and the velocity correction's `vel`/`c` are always compute dtype), so
        there is no equivalent early-cast ULP trap here - still avoided anyway, on
        principle, and confirmed by the same base-vs-Pallas equivalence check.
        """
        q = self.q
        dim = self.dim
        field_spec = PartitionSpec(None, "x", *([None] * (dim - 1)))
        auxiliary_spec = PartitionSpec("x", None, None)
        output_dtype = self.simulation.precision_policy.output_dtype
        compute_dtype = self.simulation.precision_policy.compute_dtype
        local_shape = (self.simulation.nx // self.simulation.n_devices, self.simulation.ny)
        if dim == 3:
            local_shape += (self.simulation.nz,)
        lattice_weights = tuple(float(value) for value in np.asarray(self.simulation.w))
        velocities = tuple(tuple(float(value) for value in np.asarray(self.simulation.c)[:, direction]) for direction in range(q))
        interpolated = bc_type is not BounceBackHalfway
        differentiable = bc_type is InterpolatedBounceBackDifferentiable

        def apply(fout, fin, local_indices, imissing, iknown, velocity, interp_weights):
            max_rows = local_indices.shape[1]
            missing_count = imissing.shape[-1]
            block_size = self.simulation.pallas_bc_block_size

            def kernel(field_ref, fin_ref, indices_ref, imissing_ref, iknown_ref, velocity_ref, weights_ref, out_ref):
                row = pl.program_id(0) * block_size + jnp.arange(block_size)
                valid_row = row < max_rows
                x = pltriton.load(indices_ref.at[row, 0], mask=valid_row, other=local_shape[0])
                coordinates = [x] + [pltriton.load(indices_ref.at[row, axis], mask=valid_row, other=0) for axis in range(1, dim)]
                active = valid_row & (x < local_shape[0])

                fout_bd = [pltriton.load(field_ref.at[(direction, *coordinates)], mask=active, other=0.0) for direction in range(q)]
                fin_bd = [pltriton.load(fin_ref.at[(direction, *coordinates)], mask=active, other=0.0) for direction in range(q)]
                bounced = list(fout_bd)

                for slot in range(missing_count):
                    missing_direction = pltriton.load(imissing_ref.at[row, slot], mask=active, other=0)
                    known_direction = pltriton.load(iknown_ref.at[row, slot], mask=active, other=0)
                    known_fin = jnp.zeros_like(fout_bd[0])
                    known_fout = jnp.zeros_like(fout_bd[0])
                    missing_fin = jnp.zeros_like(fout_bd[0])
                    for direction in range(q):
                        known_fin = jnp.where(known_direction == direction, fin_bd[direction], known_fin)
                        known_fout = jnp.where(known_direction == direction, fout_bd[direction], known_fout)
                        missing_fin = jnp.where(missing_direction == direction, fin_bd[direction], missing_fin)

                    if not interpolated:
                        value = known_fin
                    else:
                        w_interp = pltriton.load(weights_ref.at[row, slot], mask=active, other=0.5).astype(compute_dtype)
                        if not differentiable:
                            fs_near = 2.0 * w_interp * known_fin + (1.0 - 2.0 * w_interp) * known_fout
                            fs_far = (1.0 / (2.0 * w_interp)) * known_fin + ((2.0 * w_interp - 1.0) / (2.0 * w_interp)) * missing_fin
                            value = jnp.where(w_interp < 0.5, fs_near, fs_far)
                        else:
                            value = ((1.0 - w_interp) * known_fout + w_interp * (missing_fin + known_fin)) / (1.0 + w_interp)
                        value = value.astype(output_dtype)
                    bounced = [jnp.where(missing_direction == direction, value, bounced[direction]) for direction in range(q)]

                velocity_local = [pltriton.load(velocity_ref.at[row, axis], mask=active, other=0.0).astype(compute_dtype) for axis in range(dim)]
                for slot in range(missing_count):
                    missing_direction = pltriton.load(imissing_ref.at[row, slot], mask=active, other=0)
                    known_direction = pltriton.load(iknown_ref.at[row, slot], mask=active, other=0)
                    cu_at_known = jnp.zeros_like(velocity_local[0])
                    for direction in range(q):
                        cu = 6.0 * lattice_weights[direction] * sum(velocity_local[axis] * velocities[direction][axis] for axis in range(dim))
                        cu_at_known = jnp.where(known_direction == direction, cu, cu_at_known)
                    correction = (-cu_at_known).astype(output_dtype)
                    bounced = [
                        jnp.where(missing_direction == direction, bounced[direction] + correction, bounced[direction])
                        for direction in range(q)
                    ]

                for direction in range(q):
                    pltriton.store(out_ref.at[(direction, *coordinates)], bounced[direction], mask=active)

            field_shape = (q, *local_shape)
            local = jax.jit(
                pl.pallas_call(
                    kernel,
                    out_shape=jax.ShapeDtypeStruct(field_shape, output_dtype),
                    grid=((max_rows + block_size - 1) // block_size,),
                    input_output_aliases={0: 0},
                    compiler_params=pltriton.CompilerParams(num_warps=self.simulation.pallas_bc_num_warps),
                    name=f"{self.simulation.lattice.name.lower()}_soa_native_wall_{bc_type.__name__.lower()}",
                )
            )
            return local(fout, fin, local_indices[0], imissing[0], iknown[0], velocity[0], interp_weights[0])

        return jax.jit(
            shard_map(
                apply,
                mesh=self.simulation.mesh,
                in_specs=(field_spec, field_spec, auxiliary_spec, auxiliary_spec, auxiliary_spec, auxiliary_spec, auxiliary_spec),
                out_specs=field_spec,
                check_rep=False,
            )
        )

    def _build_wall_kernels(self, field_spec, auxiliary_spec):
        # All wall types share `wall_bc_data`'s local-data structure (see LBMBase._make_local_wall_bc_data),
        # so every entry in WALL_BC_TYPES gets its own shard-local native kernel here.
        return {bc_type: self._build_native_wall_kernel(bc_type) for bc_type, _, _ in WALL_BC_TYPES}

    def _build_native_zouhe_kernel(self, value_type, regularized):
        """Fully Pallas-native ZouHe/Regularized formula, fused with the aliased scatter.

        Replaces the gather (JAX) -> formula (JAX) -> `_sparse_scatter` (Pallas) split
        with one kernel doing the gather, formula math, and scatter together: ~4700
        MLUPS vs ~3945-3990 MLUPS for the split version on the same 200^3 cavity
        (+18-19%), with a smaller peak (no separate JAX-level buffers for the gathered
        populations or the formula's result). Matches base bitwise, unlike a first
        attempt that cast populations to compute dtype up front: the reference
        `_zouhe_*_math`/`_regularized_*_math` functions receive populations at their
        native (output) dtype and only promote where an operation explicitly mixes with
        a compute-dtype operand (e.g. `_zouhe_equilibrium`'s `rho.astype(c.dtype)`).
        Casting early changes the 19-term density-sum reduction's accumulation
        precision and was the actual, sole source of a previously-recorded 1-ULP
        mismatch (confirmed directly: matching the reference's dtype-promotion timing
        exactly gives a bitwise-0 diff; summation associativity was not the cause).
        """
        q = self.q
        dim = self.dim
        field_spec = PartitionSpec(None, "x", *([None] * (dim - 1)))
        auxiliary_spec = PartitionSpec("x", None, None)
        output_dtype = self.simulation.precision_policy.output_dtype
        compute_dtype = self.simulation.precision_policy.compute_dtype
        local_shape = (self.simulation.nx // self.simulation.n_devices, self.simulation.ny)
        if dim == 3:
            local_shape += (self.simulation.nz,)
        weights = tuple(float(value) for value in np.asarray(self.simulation.w))
        velocities = tuple(tuple(float(value) for value in np.asarray(self.simulation.c)[:, direction]) for direction in range(q))
        cc = np.asarray(self.simulation.lattice.cc)
        moment_count = cc.shape[1]
        cc_values = tuple(tuple(float(value) for value in row) for row in cc)
        qi_values = None
        if regularized:
            qi = np.asarray(_construct_symmetric_lattice_moment(jnp.asarray(cc, dtype=jnp.float32), dim))
            qi_values = tuple(tuple(float(value) for value in row) for row in qi)

        def apply(fout, local_indices, normals, imiddle_mask, iknown_mask, imissing, iknown, prescribed):
            max_rows = local_indices.shape[1]
            missing_count = imissing.shape[-1]
            block_size = self.simulation.pallas_bc_block_size

            def kernel(field_ref, indices_ref, normals_ref, middle_ref, known_ref, missing_ref, known_dir_ref, prescribed_ref, out_ref):
                row = pl.program_id(0) * block_size + jnp.arange(block_size)
                valid_row = row < max_rows
                x = pltriton.load(indices_ref.at[row, 0], mask=valid_row, other=local_shape[0])
                coordinates = [x] + [
                    pltriton.load(indices_ref.at[row, axis], mask=valid_row, other=0) for axis in range(1, dim)
                ]
                active = valid_row & (x < local_shape[0])

                # populations load at native (output) dtype, matching the reference
                # formula's own promotion timing - see docstring above. normal/
                # prescribed are separate, already-small auxiliary arrays (not part of
                # the density-sum reduction this is about), cast to compute dtype
                # immediately like the reference's own vel/rho inputs effectively are.
                populations = [pltriton.load(field_ref.at[(direction, *coordinates)], mask=active, other=0.0) for direction in range(q)]
                normal = [pltriton.load(normals_ref.at[row, axis], mask=active, other=0.0).astype(compute_dtype) for axis in range(dim)]

                middle_sum = jnp.zeros_like(populations[0])
                known_sum = jnp.zeros_like(populations[0])
                for direction in range(q):
                    middle = pltriton.load(middle_ref.at[row, direction], mask=active, other=False)
                    known_mask = pltriton.load(known_ref.at[row, direction], mask=active, other=False)
                    middle_sum = middle_sum + populations[direction] * middle
                    known_sum = known_sum + populations[direction] * known_mask
                density_sum = middle_sum + 2.0 * known_sum

                if value_type == "velocity":
                    velocity = [
                        pltriton.load(prescribed_ref.at[row, axis], mask=active, other=0.0).astype(compute_dtype) for axis in range(dim)
                    ]
                    unormal = sum(normal[axis] * velocity[axis] for axis in range(dim))
                    rho = (1.0 / (1.0 + unormal)) * density_sum
                else:
                    rho = pltriton.load(prescribed_ref.at[row, 0], mask=active, other=1.0).astype(compute_dtype)
                    unormal = -1.0 + (1.0 / rho) * density_sum
                    velocity = [unormal * normal[axis] for axis in range(dim)]

                velocity_squared = sum(value * value for value in velocity)
                feq = []
                for direction in range(q):
                    cu = 3.0 * sum(velocities[direction][axis] * velocity[axis] for axis in range(dim))
                    feq.append(rho * weights[direction] * (1.0 + cu + 0.5 * cu * cu - 1.5 * velocity_squared))

                bounced = list(populations)
                for slot in range(missing_count):
                    missing_direction = pltriton.load(missing_ref.at[row, slot], mask=active, other=0)
                    known_direction = pltriton.load(known_dir_ref.at[row, slot], mask=active, other=0)
                    known_population = jnp.zeros_like(populations[0])
                    known_equilibrium = jnp.zeros_like(rho)
                    missing_equilibrium = jnp.zeros_like(rho)
                    for direction in range(q):
                        known_population = jnp.where(known_direction == direction, populations[direction], known_population)
                        known_equilibrium = jnp.where(known_direction == direction, feq[direction], known_equilibrium)
                        missing_equilibrium = jnp.where(missing_direction == direction, feq[direction], missing_equilibrium)
                    value = (known_population + missing_equilibrium - known_equilibrium).astype(output_dtype)
                    bounced = [jnp.where(missing_direction == direction, value, bounced[direction]) for direction in range(q)]

                if regularized:
                    moments = []
                    for moment in range(moment_count):
                        total = jnp.zeros_like(rho)
                        for direction in range(q):
                            total = total + (bounced[direction] - feq[direction]) * cc_values[direction][moment]
                        moments.append(total)
                    for direction in range(q):
                        contraction = jnp.zeros_like(rho)
                        for moment in range(moment_count):
                            contraction = contraction + moments[moment] * qi_values[moment][direction]
                        value = (feq[direction] + 4.5 * weights[direction] * contraction).astype(output_dtype)
                        pltriton.store(out_ref.at[(direction, *coordinates)], value, mask=active)
                else:
                    for direction in range(q):
                        pltriton.store(out_ref.at[(direction, *coordinates)], bounced[direction], mask=active)

            field_shape = (q, *local_shape)
            local = jax.jit(
                pl.pallas_call(
                    kernel,
                    out_shape=jax.ShapeDtypeStruct(field_shape, output_dtype),
                    grid=((max_rows + block_size - 1) // block_size,),
                    input_output_aliases={0: 0},
                    compiler_params=pltriton.CompilerParams(num_warps=self.simulation.pallas_bc_num_warps),
                    name=f"{self.simulation.lattice.name.lower()}_soa_native_{'regularized' if regularized else 'zouhe'}_{value_type}",
                )
            )
            return local(fout, local_indices[0], normals[0], imiddle_mask[0], iknown_mask[0], imissing[0], iknown[0], prescribed[0])

        return jax.jit(
            shard_map(
                apply,
                mesh=self.simulation.mesh,
                in_specs=(
                    field_spec,
                    auxiliary_spec,
                    auxiliary_spec,
                    auxiliary_spec,
                    auxiliary_spec,
                    auxiliary_spec,
                    auxiliary_spec,
                    auxiliary_spec,
                ),
                out_specs=field_spec,
                check_rep=False,
            )
        )

    def _build_inlet_outlet_kernels(self, field_spec, auxiliary_spec):
        def make_dense_kernel(formula):
            def apply(fout, local_indices, normals, imiddle_mask, iknown_mask, imissing, iknown, prescribed):
                indices = self._indices(local_indices, self.dim)
                gather_indices = (slice(None), *indices)
                populations = jnp.moveaxis(fout.at[gather_indices].get(mode="fill", fill_value=0.0), 0, -1)
                boundary = formula(
                    populations,
                    prescribed[0],
                    normals[0],
                    imiddle_mask[0],
                    iknown_mask[0],
                    imissing[0],
                    iknown[0],
                    self.w,
                    self.c,
                    self.cc,
                    self.dim,
                )
                return fout.at[gather_indices].set(jnp.moveaxis(boundary, -1, 0), mode="drop")

            return jax.jit(
                shard_map(
                    apply,
                    mesh=self.simulation.mesh,
                    in_specs=(
                        field_spec,
                        auxiliary_spec,
                        auxiliary_spec,
                        auxiliary_spec,
                        auxiliary_spec,
                        auxiliary_spec,
                        auxiliary_spec,
                        auxiliary_spec,
                    ),
                    out_specs=field_spec,
                    check_rep=False,
                )
            )

        # Unique-index nodes (the common case) get the fully Pallas-native kernel;
        # nodes with duplicates fall back to the original gather/JAX-formula/dense-
        # scatter path, since the native kernel's aliased output write assumes each
        # row owns a distinct position the same way _sparse_scatter does.
        return {
            (bc_type, value_type): (
                self._build_native_zouhe_kernel(value_type, regularized=bc_type is Regularized)
                if self._has_unique_indices(bc_type, value_type)
                else make_dense_kernel(formula)
            )
            for bc_type, value_type, formula in INLET_OUTLET_BC_TYPES
            if bc_type in (ZouHe, Regularized)
        }

    # -- Neighbor-reading outflow boundaries ---------------------------------------------------
    #
    # ExtrapolationOutflow (single phase remains on the AoS fallback), ConvectiveOutflow, NonEquilibriumExtrapolation, and
    # ExactNonEquilibriumExtrapolation read one neighbor node per boundary node (one lattice step
    # along the inward normal), unlike the wall/inlet-outlet kernels above which only read their own
    # node. Only the X dimension is sharded, and a neighbor is always exactly one lattice step away,
    # so a neighbor can cross a shard boundary only when a boundary node sits on the innermost row of
    # its shard. `_localize_neighbors` detects this per configuration (data-driven, not assumed) and
    # the affected kernels exchange a 1-layer X halo only when it actually happens.
    #
    # ExtrapolationOutflowMultiphase uses a two-layer halo and is available to the
    # multiphase component view below.

    def _localize_neighbors(self, indices, local_neighbors):
        """Shift each shard's neighbor x-coordinates into shard-local space in place.

        Returns whether any neighbor falls outside its shard's local x-range, i.e. whether the
        matching kernel needs a 1-layer x-halo exchange to read it.
        """
        local_nx = self.simulation.nx // self.simulation.n_devices
        owner = indices[:, 0] // local_nx
        needs_halo = False
        for device in range(self.simulation.n_devices):
            count = int(np.count_nonzero(owner == device))
            if count == 0:
                continue
            local_neighbors[device, :count, 0] -= device * local_nx
            needs_halo = needs_halo or bool(
                np.any((local_neighbors[device, :count, 0] < 0) | (local_neighbors[device, :count, 0] >= local_nx))
            )
        return needs_halo

    def _neighbor_field(self, field, local_neighbor_indices, needs_halo):
        """Return (field padded with a 1-layer x-halo, adjusted local neighbor indices)."""
        neighbor_indices = local_neighbor_indices[0]
        if not needs_halo:
            return field, neighbor_indices
        left_halo = self.simulation.send_right(field[:, -1:], "x")
        right_halo = self.simulation.send_left(field[:, :1], "x")
        return jnp.concatenate((left_halo, field, right_halo), axis=1), neighbor_indices.at[:, 0].add(1)

    def _gather_bd(self, field, gather_indices):
        return jnp.moveaxis(field.at[gather_indices].get(mode="fill", fill_value=0.0), 0, -1)

    @staticmethod
    def _expand_neq_prescribed(values, count):
        """Normalize prescribed density to one row per boundary node."""
        values = np.asarray(values)
        if values.ndim == 0:
            return np.full((count, 1), values.item(), dtype=values.dtype)
        if values.ndim == 1:
            if values.shape[0] == count:
                return values[:, None]
            return np.broadcast_to(values, (count, values.shape[0])).copy()
        if values.shape[0] == count:
            return values
        return np.broadcast_to(values, (count, *values.shape)).copy()

    def _make_neq_data(self):
        """Build local (per-shard) data for NonEquilibriumExtrapolation/ExactNonEquilibriumExtrapolation."""
        data = {}
        for bc_type in NEQ_BC_TYPES:
            matches = [bc for bc in self.simulation.BCs if type(bc) is bc_type]
            if not matches:
                data[bc_type] = None
                continue
            for bc in matches:
                if not bc.neighbors_found:
                    bc.find_neighbors()
                    bc.neighbors_found = True

            indices = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in matches])
            neighbor_indices = np.vstack([np.asarray(bc.indices_nbr, dtype=np.int32).T for bc in matches])
            imissing = np.vstack([np.asarray(bc.imissing) for bc in matches])
            prescribed = np.vstack([self._expand_neq_prescribed(bc.prescribed, len(bc.indices[0])) for bc in matches])

            local_indices, (local_neighbors, local_imissing, local_prescribed) = self.simulation._split_local_indices(
                indices, neighbor_indices, imissing, prescribed
            )
            needs_halo = self._localize_neighbors(indices, local_neighbors)
            data[bc_type] = (
                needs_halo,
                self.simulation._distribute_local(local_indices, jnp.int32),
                self.simulation._distribute_local(local_neighbors, jnp.int32),
                self.simulation._distribute_local(local_imissing, jnp.uint8),
                self.simulation._distribute_local(local_prescribed, self.simulation.precision_policy.compute_dtype),
            )
        return data

    def _build_neq_kernels(self, field_spec, auxiliary_spec):
        """Fully Pallas-native NonEquilibriumExtrapolation/ExactNonEquilibriumExtrapolation
        formula, fused with the aliased scatter - same technique as
        `_build_native_zouhe_kernel`. The cross-device halo exchange for the neighbor
        read (`_neighbor_field`) stays in JAX (Pallas/Triton kernels cannot do
        inter-device communication); only the per-node gather+formula+write becomes one
        kernel. `_neq_extrapolation_math` has two full 19-direction native-(output-)dtype
        reductions with no compute-dtype operand forcing promotion - `rho_nbr =
        jnp.sum(f_nbr, ...)` and, for the exact variant, `rho_incorrect = jnp.sum(fbd,
        ...)` after the first `.set()` - exactly the pattern that caused the ZouHe/
        Regularized 1-ULP mismatch; both are summed here at native dtype with no early
        cast, matching the reference's promotion timing (rho only casts to compute
        dtype inside `equilibium`, i.e. after the sum, same as `_zouhe_equilibrium`).
        """
        exact_bc = next((bc for bc in self.simulation.BCs if type(bc) is ExactNonEquilibriumExtrapolation), None)
        correction_weights = tuple(float(value) for value in np.asarray(exact_bc.w_NEQ)) if exact_bc is not None else None
        q = self.q
        dim = self.dim
        output_dtype = self.simulation.precision_policy.output_dtype
        compute_dtype = self.simulation.precision_policy.compute_dtype
        local_shape = (self.simulation.nx // self.simulation.n_devices, self.simulation.ny)
        if dim == 3:
            local_shape += (self.simulation.nz,)
        weights_lattice = tuple(float(value) for value in np.asarray(self.simulation.w))
        velocities = tuple(tuple(float(value) for value in np.asarray(self.simulation.c)[:, direction]) for direction in range(q))

        def make_kernel(needs_halo, exact):
            def apply(fout, local_indices, local_neighbor_indices, local_imissing, local_prescribed):
                neighbor_field, neighbor_indices = self._neighbor_field(fout, local_neighbor_indices, needs_halo)
                max_rows = local_indices.shape[1]
                missing_count = local_imissing.shape[-1]
                block_size = self.simulation.pallas_bc_block_size

                def kernel(field_ref, neighbor_ref, indices_ref, neighbor_indices_ref, imissing_ref, prescribed_ref, out_ref):
                    row = pl.program_id(0) * block_size + jnp.arange(block_size)
                    valid_row = row < max_rows
                    x = pltriton.load(indices_ref.at[row, 0], mask=valid_row, other=local_shape[0])
                    coordinates = [x] + [pltriton.load(indices_ref.at[row, axis], mask=valid_row, other=0) for axis in range(1, dim)]
                    active = valid_row & (x < local_shape[0])
                    neighbor_coordinates = [
                        pltriton.load(neighbor_indices_ref.at[row, axis], mask=active, other=0) for axis in range(dim)
                    ]

                    fbd = [pltriton.load(field_ref.at[(direction, *coordinates)], mask=active, other=0.0) for direction in range(q)]
                    f_nbr = [
                        pltriton.load(neighbor_ref.at[(direction, *neighbor_coordinates)], mask=active, other=0.0) for direction in range(q)
                    ]

                    # Native-dtype 19-term sum, no early cast - see docstring above. Kept
                    # untouched for its later use in equilibrium() (which casts it to
                    # compute dtype itself, same as the reference). The division below
                    # gets its own compute-dtype copy instead of casting rho_nbr itself:
                    # reference's numerator (jnp.dot(f_nbr, c.T)) is genuinely f32 since
                    # c is a real f32 array, but a Python-float static coefficient times
                    # a native f16 tensor here is weakly-typed and stays f16, so an
                    # unpromoted division would be pure f16/f16 - Triton's Ampere backend
                    # cannot lower that (same limitation documented for the main
                    # collision kernels); promoting only this division, not the
                    # accumulation, avoids it while keeping rho_nbr's own precision exact.
                    rho_nbr = f_nbr[0]
                    for direction in range(1, q):
                        rho_nbr = rho_nbr + f_nbr[direction]
                    rho_nbr_compute = rho_nbr.astype(compute_dtype)
                    vel_nbr = [
                        sum(velocities[direction][axis] * f_nbr[direction].astype(compute_dtype) for direction in range(q))
                        / rho_nbr_compute
                        for axis in range(dim)
                    ]

                    def equilibrium(rho, velocity):
                        rho_c = rho.astype(compute_dtype)
                        usqr = 1.5 * sum(value * value for value in velocity)
                        result = []
                        for direction in range(q):
                            cu = 3.0 * sum(velocities[direction][axis] * velocity[axis] for axis in range(dim))
                            result.append(rho_c * weights_lattice[direction] * (1.0 + cu + 0.5 * cu * cu - usqr))
                        return result

                    feq_nbr = equilibrium(rho_nbr, vel_nbr)
                    prescribed = pltriton.load(prescribed_ref.at[row, 0], mask=active, other=1.0).astype(compute_dtype)
                    feq = equilibrium(prescribed, vel_nbr)

                    bounced = list(fbd)
                    for slot in range(missing_count):
                        missing_direction = pltriton.load(imissing_ref.at[row, slot], mask=active, other=0)
                        value = jnp.zeros_like(fbd[0]).astype(compute_dtype)
                        for direction in range(q):
                            candidate = feq[direction] + (f_nbr[direction] - feq_nbr[direction])
                            value = jnp.where(missing_direction == direction, candidate, value)
                        value = value.astype(output_dtype)
                        bounced = [jnp.where(missing_direction == direction, value, bounced[direction]) for direction in range(q)]

                    if correction_weights is not None:
                        # Native-dtype 19-term sum over the just-updated `bounced`, no
                        # early cast - matches `rho_incorrect = jnp.sum(fbd, ...)`.
                        rho_incorrect = bounced[0]
                        for direction in range(1, q):
                            rho_incorrect = rho_incorrect + bounced[direction]
                        sum_missing_weights = jnp.zeros_like(prescribed)
                        missing_weight_by_slot = []
                        for slot in range(missing_count):
                            missing_direction = pltriton.load(imissing_ref.at[row, slot], mask=active, other=0)
                            weight = jnp.zeros_like(prescribed)
                            for direction in range(q):
                                weight = jnp.where(missing_direction == direction, correction_weights[direction], weight)
                            missing_weight_by_slot.append(weight)
                            sum_missing_weights = sum_missing_weights + weight
                        for slot in range(missing_count):
                            missing_direction = pltriton.load(imissing_ref.at[row, slot], mask=active, other=0)
                            beta = missing_weight_by_slot[slot] * (prescribed - rho_incorrect) / sum_missing_weights
                            bounced = [
                                jnp.where(missing_direction == direction, (bounced[direction] + beta).astype(output_dtype), bounced[direction])
                                for direction in range(q)
                            ]

                    for direction in range(q):
                        pltriton.store(out_ref.at[(direction, *coordinates)], bounced[direction], mask=active)

                field_shape = (q, *local_shape)
                local = jax.jit(
                    pl.pallas_call(
                        kernel,
                        out_shape=jax.ShapeDtypeStruct(field_shape, output_dtype),
                        grid=((max_rows + block_size - 1) // block_size,),
                        input_output_aliases={0: 0},
                        compiler_params=pltriton.CompilerParams(num_warps=self.simulation.pallas_bc_num_warps),
                        name=f"{self.simulation.lattice.name.lower()}_soa_native_neq_{'exact' if exact else 'plain'}",
                    )
                )
                return local(fout, neighbor_field, local_indices[0], neighbor_indices, local_imissing[0], local_prescribed[0])

            return jax.jit(
                shard_map(
                    apply,
                    mesh=self.simulation.mesh,
                    in_specs=(field_spec, auxiliary_spec, auxiliary_spec, auxiliary_spec, auxiliary_spec),
                    out_specs=field_spec,
                    check_rep=False,
                )
            )

        return {
            (exact, needs_halo): make_kernel(needs_halo, exact)
            for exact in (False, True)
            for needs_halo in (False, True)
        }

    def _make_convective_data(self):
        """Build local (per-shard) data for ConvectiveOutflow."""
        matches = [bc for bc in self.simulation.BCs if type(bc) is ConvectiveOutflow]
        if not matches:
            return None
        for bc in matches:
            if not bc.neighbors_found:
                bc.find_neighbors()
                bc.neighbors_found = True

        indices = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in matches])
        neighbor_indices = np.vstack([np.asarray(bc.indices_nbr, dtype=np.int32).T for bc in matches])
        normals = np.vstack([np.asarray(bc.normals) for bc in matches])

        local_indices, (local_neighbors, local_normals) = self.simulation._split_local_indices(indices, neighbor_indices, normals)
        needs_halo = self._localize_neighbors(indices, local_neighbors)
        return (
            needs_halo,
            self.simulation._distribute_local(local_indices, jnp.int32),
            self.simulation._distribute_local(local_neighbors, jnp.int32),
            self.simulation._distribute_local(local_normals),
        )

    def _build_convective_kernels(self, field_spec, auxiliary_spec):
        """Two native Pallas kernels around the one JAX op that must stay JAX:
        `lax.pmax`'s cross-device reduction (Pallas/Triton kernels cannot do
        inter-device communication, so lambda_cbc genuinely cannot move into a
        kernel). Kernel A computes each row's u_nbr (native-dtype rho_nbr sum, no
        early cast, promoted division - same pattern and reasoning as
        `_build_neq_kernels`); the max+pmax stays JAX exactly as before; kernel B
        re-reads f_nbr/fin_bd and does the blend+aliased-scatter write in one call.
        f_nbr is genuinely re-gathered by kernel B (not threaded through from kernel
        A) - simpler than plumbing a second output out of kernel A, and gathering a
        19-direction row is cheap next to the halo exchange and collective already
        being paid for.
        """
        q = self.q
        dim = self.dim
        local_nx = self.simulation.nx // self.simulation.n_devices
        output_dtype = self.simulation.precision_policy.output_dtype
        compute_dtype = self.simulation.precision_policy.compute_dtype
        local_shape = (self.simulation.nx // self.simulation.n_devices, self.simulation.ny)
        if dim == 3:
            local_shape += (self.simulation.nz,)
        velocities = tuple(tuple(float(value) for value in np.asarray(self.simulation.c)[:, direction]) for direction in range(q))

        def make_kernel(needs_halo):
            def apply(fout, fin, local_indices, local_neighbor_indices, local_normals):
                neighbor_field, neighbor_indices = self._neighbor_field(fout, local_neighbor_indices, needs_halo)
                max_rows = local_indices.shape[1]
                block_size = self.simulation.pallas_bc_block_size
                grid = ((max_rows + block_size - 1) // block_size,)

                def kernel_u_nbr(neighbor_ref, neighbor_indices_ref, normals_ref, u_out_ref):
                    row = pl.program_id(0) * block_size + jnp.arange(block_size)
                    valid_row = row < max_rows
                    neighbor_coordinates = [
                        pltriton.load(neighbor_indices_ref.at[row, axis], mask=valid_row, other=0) for axis in range(dim)
                    ]
                    f_nbr = [
                        pltriton.load(neighbor_ref.at[(direction, *neighbor_coordinates)], mask=valid_row, other=0.0)
                        for direction in range(q)
                    ]
                    # Native-dtype 19-term sum, no early cast - same reasoning as
                    # _build_neq_kernels; division promoted the same way too.
                    rho_nbr = f_nbr[0]
                    for direction in range(1, q):
                        rho_nbr = rho_nbr + f_nbr[direction]
                    rho_nbr_compute = rho_nbr.astype(compute_dtype)
                    normal = [pltriton.load(normals_ref.at[row, axis], mask=valid_row, other=0.0).astype(compute_dtype) for axis in range(dim)]
                    u_nbr = jnp.zeros_like(rho_nbr_compute)
                    for axis in range(dim):
                        numerator = sum(velocities[direction][axis] * f_nbr[direction].astype(compute_dtype) for direction in range(q))
                        u_nbr = u_nbr + (numerator / rho_nbr_compute) * normal[axis]
                    pltriton.store(u_out_ref.at[row], u_nbr, mask=valid_row)

                compute_u_nbr = jax.jit(
                    pl.pallas_call(
                        kernel_u_nbr,
                        out_shape=jax.ShapeDtypeStruct((max_rows,), compute_dtype),
                        grid=grid,
                        compiler_params=pltriton.CompilerParams(num_warps=self.simulation.pallas_bc_num_warps),
                        name=f"{self.simulation.lattice.name.lower()}_soa_native_convective_u_nbr",
                    )
                )
                u_nbr_all = compute_u_nbr(neighbor_field, neighbor_indices, local_normals[0])

                # Padding rows (see LBMBase._split_local_indices) carry an out-of-range x
                # sentinel; exclude them from the cross-device max so they can't skew
                # lambda_cbc. Cross-device reduction: must stay JAX, cannot move into Pallas.
                valid = local_indices[0][:, 0] < local_nx
                local_max = jnp.max(jnp.where(valid, u_nbr_all, -jnp.inf))
                lambda_cbc = lax.pmax(local_max, axis_name="x").reshape((1,))

                def kernel_blend(field_ref, fin_ref, neighbor_ref, indices_ref, neighbor_indices_ref, lambda_ref, out_ref):
                    row = pl.program_id(0) * block_size + jnp.arange(block_size)
                    valid_row = row < max_rows
                    x = pltriton.load(indices_ref.at[row, 0], mask=valid_row, other=local_shape[0])
                    coordinates = [x] + [pltriton.load(indices_ref.at[row, axis], mask=valid_row, other=0) for axis in range(1, dim)]
                    active = valid_row & (x < local_shape[0])
                    neighbor_coordinates = [
                        pltriton.load(neighbor_indices_ref.at[row, axis], mask=active, other=0) for axis in range(dim)
                    ]
                    lam = pltriton.load(lambda_ref.at[0], mask=True, other=0.0)
                    for direction in range(q):
                        fin_value = pltriton.load(fin_ref.at[(direction, *coordinates)], mask=active, other=0.0)
                        neighbor_value = pltriton.load(neighbor_ref.at[(direction, *neighbor_coordinates)], mask=active, other=0.0)
                        value = ((1.0 - lam) * neighbor_value + lam * fin_value).astype(output_dtype)
                        pltriton.store(out_ref.at[(direction, *coordinates)], value, mask=active)

                field_shape = (q, *local_shape)
                blend_and_scatter = jax.jit(
                    pl.pallas_call(
                        kernel_blend,
                        out_shape=jax.ShapeDtypeStruct(field_shape, output_dtype),
                        grid=grid,
                        input_output_aliases={0: 0},
                        compiler_params=pltriton.CompilerParams(num_warps=self.simulation.pallas_bc_num_warps),
                        name=f"{self.simulation.lattice.name.lower()}_soa_native_convective_blend",
                    )
                )
                return blend_and_scatter(fout, fin, neighbor_field, local_indices[0], neighbor_indices, lambda_cbc)

            return jax.jit(
                shard_map(
                    apply,
                    mesh=self.simulation.mesh,
                    in_specs=(field_spec, field_spec, auxiliary_spec, auxiliary_spec, auxiliary_spec),
                    out_specs=field_spec,
                    check_rep=False,
                )
            )

        return {needs_halo: make_kernel(needs_halo) for needs_halo in (False, True)}

    def _make_extrapolation_data(self):
        matches = [bc for bc in self.simulation.BCs if type(bc) is ExtrapolationOutflow]
        if not matches:
            return None
        indices = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in matches])
        neighbors = np.vstack([np.asarray(bc.indices_nbr, dtype=np.int32).T for bc in matches])
        imissing = np.vstack([np.asarray(bc.imissing) for bc in matches])
        iknown = np.vstack([np.asarray(bc.iknown) for bc in matches])
        local_indices, (local_neighbors, local_imissing, local_iknown) = self.simulation._split_local_indices(indices, neighbors, imissing, iknown)
        needs_halo = self._localize_neighbors(indices, local_neighbors)
        return (needs_halo, self.simulation._distribute_local(local_indices, jnp.int32), self.simulation._distribute_local(local_neighbors, jnp.int32), self.simulation._distribute_local(local_imissing, jnp.uint8), self.simulation._distribute_local(local_iknown, jnp.uint8))

    def _build_extrapolation_kernels(self, field_spec, auxiliary_spec):
        """Fully Pallas-native ExtrapolationOutflow, fused with the aliased scatter -
        same technique as the other native kernels above. Pure per-slot selection and
        blend (no reduction, no formula function to match), so unlike ZouHe/NEQ/
        Convective there is no early-cast ULP trap here; everything stays at whatever
        dtype `field_ref`/`fin_ref` naturally are, matching the JAX version's own
        promotion exactly (Python-float `cs` combined with a native array stays native,
        the same weak-type behavior the JAX version already relied on).
        Note the two implementation steps write/read imissing and iknown in opposite
        roles - post-collision writes to iknown using values gathered from imissing,
        post-streaming writes to imissing using values gathered from iknown - matching
        the existing (pre-native) JAX version's indexing exactly, not a typo.
        """
        q = self.q
        dim = self.dim
        output_dtype = self.simulation.precision_policy.output_dtype
        local_shape = (self.simulation.nx // self.simulation.n_devices, self.simulation.ny)
        if dim == 3:
            local_shape += (self.simulation.nz,)
        cs = float(1.0 / np.sqrt(3.0))

        def make_kernel(needs_halo, post_collision):
            def apply(fout, fin, local_indices, local_neighbors, local_imissing, local_iknown):
                neighbor_field, neighbor_indices = self._neighbor_field(fout, local_neighbors, needs_halo)
                max_rows = local_indices.shape[1]
                missing_count = local_imissing.shape[-1]
                block_size = self.simulation.pallas_bc_block_size

                def kernel(field_ref, fin_ref, neighbor_ref, indices_ref, neighbor_indices_ref, imissing_ref, iknown_ref, out_ref):
                    row = pl.program_id(0) * block_size + jnp.arange(block_size)
                    valid_row = row < max_rows
                    x = pltriton.load(indices_ref.at[row, 0], mask=valid_row, other=local_shape[0])
                    coordinates = [x] + [pltriton.load(indices_ref.at[row, axis], mask=valid_row, other=0) for axis in range(1, dim)]
                    active = valid_row & (x < local_shape[0])
                    neighbor_coordinates = [
                        pltriton.load(neighbor_indices_ref.at[row, axis], mask=active, other=0) for axis in range(dim)
                    ]

                    b = [pltriton.load(field_ref.at[(direction, *coordinates)], mask=active, other=0.0) for direction in range(q)]
                    old = [pltriton.load(fin_ref.at[(direction, *coordinates)], mask=active, other=0.0) for direction in range(q)]
                    nbr = [
                        pltriton.load(neighbor_ref.at[(direction, *neighbor_coordinates)], mask=active, other=0.0) for direction in range(q)
                    ]

                    for slot in range(missing_count):
                        missing_direction = pltriton.load(imissing_ref.at[row, slot], mask=active, other=0)
                        known_direction = pltriton.load(iknown_ref.at[row, slot], mask=active, other=0)
                        if post_collision:
                            nbr_at_missing = jnp.zeros_like(b[0])
                            old_at_missing = jnp.zeros_like(b[0])
                            for direction in range(q):
                                nbr_at_missing = jnp.where(missing_direction == direction, nbr[direction], nbr_at_missing)
                                old_at_missing = jnp.where(missing_direction == direction, old[direction], old_at_missing)
                            value = cs * nbr_at_missing + (1.0 - cs) * old_at_missing
                            b = [jnp.where(known_direction == direction, value, b[direction]) for direction in range(q)]
                        else:
                            old_at_known = jnp.zeros_like(b[0])
                            for direction in range(q):
                                old_at_known = jnp.where(known_direction == direction, old[direction], old_at_known)
                            b = [jnp.where(missing_direction == direction, old_at_known, b[direction]) for direction in range(q)]

                    for direction in range(q):
                        pltriton.store(out_ref.at[(direction, *coordinates)], b[direction], mask=active)

                field_shape = (q, *local_shape)
                local = jax.jit(
                    pl.pallas_call(
                        kernel,
                        out_shape=jax.ShapeDtypeStruct(field_shape, output_dtype),
                        grid=((max_rows + block_size - 1) // block_size,),
                        input_output_aliases={0: 0},
                        compiler_params=pltriton.CompilerParams(num_warps=self.simulation.pallas_bc_num_warps),
                        name=f"{self.simulation.lattice.name.lower()}_soa_native_extrapolation_{'postcollision' if post_collision else 'poststreaming'}",
                    )
                )
                return local(fout, fin, neighbor_field, local_indices[0], neighbor_indices, local_imissing[0], local_iknown[0])

            return jax.jit(
                shard_map(
                    apply,
                    mesh=self.simulation.mesh,
                    in_specs=(field_spec, field_spec, auxiliary_spec, auxiliary_spec, auxiliary_spec, auxiliary_spec),
                    out_specs=field_spec,
                    check_rep=False,
                )
            )

        return {(post, halo): make_kernel(halo, post) for post in (False, True) for halo in (False, True)}

    def _make_multiphase_extrapolation_data(self):
        matches = [bc for bc in self.simulation.BCs if type(bc) is ExtrapolationOutflowMultiphase]
        if not matches:
            return None
        for bc in matches:
            if not bc.neighbors_found:
                bc.find_neighbors()
                bc.neighbors_found = True
        indices = np.vstack([np.asarray(bc.indices, dtype=np.int32).T for bc in matches])
        nbr = np.vstack([np.asarray(bc.indices_nbr, dtype=np.int32).T for bc in matches])
        next_nbr = np.vstack([np.asarray(bc.indices_next_nbr, dtype=np.int32).T for bc in matches])
        imissing = np.vstack([np.asarray(bc.imissing) for bc in matches])
        local_indices, (local_nbr, local_next, local_imissing) = self.simulation._split_local_indices(indices, nbr, next_nbr, imissing)
        local_nx = self.simulation.nx // self.simulation.n_devices
        owners = indices[:, 0] // local_nx
        needs_halo = False
        for device in range(self.simulation.n_devices):
            count = int(np.count_nonzero(owners == device))
            if count == 0:
                continue
            local_nbr[device, :count, 0] -= device * local_nx
            local_next[device, :count, 0] -= device * local_nx
            needs_halo = needs_halo or bool(
                np.any((local_nbr[device, :count, 0] < 0) | (local_nbr[device, :count, 0] >= local_nx))
                or np.any((local_next[device, :count, 0] < 0) | (local_next[device, :count, 0] >= local_nx))
            )
        return (
            needs_halo,
            self.simulation._distribute_local(local_indices, jnp.int32),
            self.simulation._distribute_local(local_nbr, jnp.int32),
            self.simulation._distribute_local(local_next, jnp.int32),
            self.simulation._distribute_local(local_imissing, jnp.uint8),
        )

    def _build_multiphase_extrapolation_kernels(self, field_spec, auxiliary_spec):
        """Fully Pallas-native ExtrapolationOutflowMultiphase, fused with the aliased
        scatter - same technique as `_build_extrapolation_kernels`. Two-layer halo
        (this BC reads a neighbor and a next-neighbor, two lattice steps in along the
        inward normal) built in JAX exactly as before - Pallas/Triton cannot do the
        inter-device exchange - then one kernel does both neighbor reads, the own-node
        read, the `2*f_nbr - f_next` formula, and the write. No reduction and no
        formula-mixed-with-compute-dtype step here (matches base's `apply()` exactly:
        `fbd.at[bindex, imissing].set(2*f_nbr[...] - f_next_nbr[...])`), so - like
        single-phase ExtrapolationOutflow - there is no early-cast ULP trap to avoid.
        """
        q = self.q
        dim = self.dim
        output_dtype = self.simulation.precision_policy.output_dtype
        local_shape = (self.simulation.nx // self.simulation.n_devices, self.simulation.ny)
        if dim == 3:
            local_shape += (self.simulation.nz,)

        def make_kernel(needs_halo):
            def apply(fout, local_indices, local_nbr, local_next, local_imissing):
                nbr = local_nbr[0]
                nxt = local_next[0]
                if needs_halo:
                    left = self.simulation.send_right(fout[:, -2:], "x")
                    right = self.simulation.send_left(fout[:, :2], "x")
                    field = jnp.concatenate((left, fout, right), axis=1)
                    nbr = nbr.at[:, 0].add(2)
                    nxt = nxt.at[:, 0].add(2)
                else:
                    field = fout

                max_rows = local_indices.shape[1]
                missing_count = local_imissing.shape[-1]
                block_size = self.simulation.pallas_bc_block_size

                def kernel(field_ref, neighbor_ref, indices_ref, nbr_ref, next_ref, imissing_ref, out_ref):
                    row = pl.program_id(0) * block_size + jnp.arange(block_size)
                    valid_row = row < max_rows
                    x = pltriton.load(indices_ref.at[row, 0], mask=valid_row, other=local_shape[0])
                    coordinates = [x] + [pltriton.load(indices_ref.at[row, axis], mask=valid_row, other=0) for axis in range(1, dim)]
                    active = valid_row & (x < local_shape[0])
                    nbr_coordinates = [pltriton.load(nbr_ref.at[row, axis], mask=active, other=0) for axis in range(dim)]
                    next_coordinates = [pltriton.load(next_ref.at[row, axis], mask=active, other=0) for axis in range(dim)]

                    fbd = [pltriton.load(field_ref.at[(direction, *coordinates)], mask=active, other=0.0) for direction in range(q)]
                    f_nbr = [
                        pltriton.load(neighbor_ref.at[(direction, *nbr_coordinates)], mask=active, other=0.0) for direction in range(q)
                    ]
                    f_next = [
                        pltriton.load(neighbor_ref.at[(direction, *next_coordinates)], mask=active, other=0.0) for direction in range(q)
                    ]

                    for slot in range(missing_count):
                        missing_direction = pltriton.load(imissing_ref.at[row, slot], mask=active, other=0)
                        nbr_at_missing = jnp.zeros_like(fbd[0])
                        next_at_missing = jnp.zeros_like(fbd[0])
                        for direction in range(q):
                            nbr_at_missing = jnp.where(missing_direction == direction, f_nbr[direction], nbr_at_missing)
                            next_at_missing = jnp.where(missing_direction == direction, f_next[direction], next_at_missing)
                        value = (2.0 * nbr_at_missing - next_at_missing).astype(output_dtype)
                        fbd = [jnp.where(missing_direction == direction, value, fbd[direction]) for direction in range(q)]

                    for direction in range(q):
                        pltriton.store(out_ref.at[(direction, *coordinates)], fbd[direction], mask=active)

                field_shape = (q, *local_shape)
                local = jax.jit(
                    pl.pallas_call(
                        kernel,
                        out_shape=jax.ShapeDtypeStruct(field_shape, output_dtype),
                        grid=((max_rows + block_size - 1) // block_size,),
                        input_output_aliases={0: 0},
                        compiler_params=pltriton.CompilerParams(num_warps=self.simulation.pallas_bc_num_warps),
                        name=f"{self.simulation.lattice.name.lower()}_soa_native_extrapolation_multiphase",
                    )
                )
                return local(fout, field, local_indices[0], nbr, nxt, local_imissing[0])

            return jax.jit(
                shard_map(
                    apply,
                    mesh=self.simulation.mesh,
                    in_specs=(field_spec, auxiliary_spec, auxiliary_spec, auxiliary_spec, auxiliary_spec),
                    out_specs=field_spec,
                    check_rep=False,
                )
            )

        return {needs_halo: make_kernel(needs_halo) for needs_halo in (False, True)}

    def apply(self, fout, fin, implementation_step, timestep=0):
        """Apply supported BCs at one LBM implementation step.

        Parameters
        ----------
        fout, fin : jax.Array
            Direction-major output and input population fields.

        implementation_step : str
            Either "PostCollision" or "PostStreaming".

        Returns
        -------
        jax.Array
            Updated direction-major population field.
        """
        simulation = self.simulation
        if implementation_step == "PostCollision":
            for boundary in simulation.BCs:
                if type(boundary) is BounceBackMoving:
                    indices, velocity = boundary.update_function(timestep)
                    global_indices = jnp.stack(indices, axis=1).astype(jnp.int32)
                    fout = self._moving_bounceback(fout, fin, global_indices, velocity)
            if simulation.local_bounceback_indices is not None:
                fout = self._bounceback(fout, fin, simulation.local_bounceback_indices)
            if self._extrapolation_data is not None:
                halo, *data = self._extrapolation_data
                fout = self._extrapolation_kernels[(True, halo)](fout, fin, *data)

        if implementation_step == "PostStreaming":
            # Phase 1 matches LBMBase.apply_bc's generic per-BC loop: DoNothing, both NEQ
            # variants, ConvectiveOutflow, and (multiphase) ExtrapolationOutflow are
            # everything NOT in that loop's isinstance(bc, (BounceBack, BounceBackHalfway,
            # EquilibriumBC, ZouHe)) exclusion (BounceBackHalfway/ZouHe subclasses -
            # InterpolatedBounceBack*/Regularized - are excluded too). Base runs this
            # whole generic loop BEFORE its batched BounceBackHalfway/EquilibriumBC/ZouHe
            # handling (phase 2 below); a neighbor-reading kernel here (NEQ/Convective/
            # Extrapolation) that borders a wall/equilibrium/inlet-outlet node must read
            # fout at this same pre-phase-2 point to see what base's neighbor read sees -
            # running it after phase 2, as an earlier version of this method did, silently
            # diverged from base at exactly those shared edge/corner nodes (small mean
            # diff, locally large max diff; caught by combining several BC types on
            # adjacent faces, not by testing any one BC type alone).
            if self._do_nothing_indices is not None:
                fout = self._do_nothing(fout, fin, self._do_nothing_indices)
            for bc_type in NEQ_BC_TYPES:
                local_data = self._neq_data[bc_type]
                if local_data is not None:
                    needs_halo, local_indices, local_neighbors, local_imissing, local_prescribed = local_data
                    fout = self._neq_kernels[(bc_type is ExactNonEquilibriumExtrapolation, needs_halo)](
                        fout, local_indices, local_neighbors, local_imissing, local_prescribed
                    )
            if self._convective_data is not None:
                needs_halo, local_indices, local_neighbors, local_normals = self._convective_data
                fout = self._convective_kernels[needs_halo](fout, fin, local_indices, local_neighbors, local_normals)
            if self._extrapolation_data is not None:
                halo, *data = self._extrapolation_data
                fout = self._extrapolation_kernels[(False, halo)](fout, fin, *data)
            if self._multiphase_extrapolation_data is not None:
                needs_halo, local_indices, local_nbr, local_next, local_imissing = self._multiphase_extrapolation_data
                fout = self._multiphase_extrapolation_kernels[needs_halo](fout, local_indices, local_nbr, local_next, local_imissing)

            # Phase 2 matches base's batched BounceBackHalfway/EquilibriumBC/ZouHe/Regularized
            # handling, which runs after the entire phase-1 generic loop above.
            if simulation.solid_pin_indices is not None:
                fout = self._solid_pin(fout, simulation.solid_pin_indices)
            for bc_type, _, _ in WALL_BC_TYPES:
                local_data = simulation.wall_bc_data[bc_type]
                if local_data[0] is not None:
                    fout = self._wall_kernels[bc_type](fout, fin, *local_data)
            if simulation.local_equilibrium_bc_indices is not None:
                fout = self._equilibrium(fout, simulation.local_equilibrium_bc_indices, simulation.local_equilibrium_bc_values)
            for bc_type, value_type, _ in INLET_OUTLET_BC_TYPES:
                local_data = simulation.inlet_outlet_bc_data[(bc_type, value_type)]
                if local_data[0] is not None:
                    fout = self._inlet_outlet_kernels[(bc_type, value_type)](fout, *local_data)
        return fout
