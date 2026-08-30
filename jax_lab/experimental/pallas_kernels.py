"""Experimental Pallas kernels for single-phase BGK and MRT simulations."""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

from jax_lab.core.boundary_conditions import BounceBack


def create_bounce_back_mask(simulation):
    """Create a solid-node mask for supported full-way bounce-back boundaries.

    Parameters
    ----------
    simulation : LBMBase
        Initialized single-device simulation whose boundary conditions are
        represented by exact BounceBack instances.

    Returns
    -------
    jax.Array
        Boolean device array matching simulation's spatial shape.

    Raises
    ------
    NotImplementedError
        If any configured boundary condition is not full-way BounceBack.
    """
    unsupported = [bc for bc in simulation.BCs if type(bc) is not BounceBack]
    if unsupported:
        names = ", ".join(type(bc).__name__ for bc in unsupported)
        raise NotImplementedError(f"Experimental fused BGK supports only exact BounceBack boundary conditions; received: {names}.")

    shape = (simulation.nx, simulation.ny)
    if simulation.dim == 3:
        shape += (simulation.nz,)
    solid = np.zeros(shape, dtype=np.bool_)
    for boundary in simulation.BCs:
        solid[boundary.indices] = True
    return jax.device_put(solid)


def build_fused_bgk(
    lattice,
    shape: Sequence[int],
    precision_policy,
    omega: float,
    *,
    block_size: int = 256,
    num_warps: int = 8,
):
    """Build a fused pull-streaming, BGK, and bounce-back Pallas kernel.

    Input and output distributions are post-collision populations. Each call
    performs pull streaming, computes BGK collision, then applies full-way
    bounce-back at nodes selected by solid.

    Parameters
    ----------
    lattice : Lattice
        D2Q9, D3Q19, or D3Q27 lattice definition.

    shape : sequence of int
        Global domain shape matching lattice dimensionality.

    precision_policy : PrecisionPolicy
        Compute and output precision used by kernel.

    omega : float
        BGK relaxation rate.

    block_size : int, optional
        Number of lattice nodes handled by each Pallas program.

    num_warps : int, optional
        Triton warps assigned to each Pallas program.

    Returns
    -------
    Callable
        JIT-compiled function (post_collision, solid) -> post_collision.

    Raises
    ------
    NotImplementedError
        If lattice or accelerator is unsupported.
    ValueError
        If launch parameters or domain shape are invalid.
    """
    supported_lattices = {"D2Q9": (2, 9), "D3Q19": (3, 19), "D3Q27": (3, 27)}
    if supported_lattices.get(lattice.name) != (lattice.d, lattice.q):
        raise NotImplementedError("Experimental fused BGK supports D2Q9, D3Q19, and D3Q27.")
    if len(shape) != lattice.d:
        raise ValueError(f"{lattice.name} requires a {lattice.d}D domain shape; received {len(shape)}D.")
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Experimental fused BGK currently requires a GPU backend.")
    if jax.device_count() != 1:
        raise NotImplementedError("Experimental fused BGK currently supports one visible GPU. Multi-GPU halo exchange is not implemented.")
    if block_size <= 0 or num_warps <= 0:
        raise ValueError("block_size and num_warps must be positive.")

    domain_shape = tuple(int(size) for size in shape)
    if min(domain_shape) <= 0:
        raise ValueError("All domain dimensions must be positive.")

    cells = int(np.prod(domain_shape))
    grid = ((cells + block_size - 1) // block_size,)
    velocities = tuple(tuple(int(value) for value in lattice.c[:, direction]) for direction in range(lattice.q))
    weights = tuple(float(value) for value in lattice.w)
    opposites = tuple(int(value) for value in lattice.opp_indices)
    compute_dtype = precision_policy.compute_dtype
    output_dtype = precision_policy.output_dtype

    def kernel(f_ref, solid_ref, out_ref):
        cell = pl.program_id(0) * block_size + jnp.arange(block_size)
        active = cell < cells
        remainder = jnp.minimum(cell, cells - 1)
        coordinates = []
        for axis in range(lattice.d):
            stride = int(np.prod(domain_shape[axis + 1 :]))
            coordinate = remainder // stride
            remainder = remainder - coordinate * stride
            coordinates.append(coordinate)

        populations = []
        for direction, velocity in enumerate(velocities):
            source_coordinates = tuple(
                (coordinate - component) % size for coordinate, component, size in zip(coordinates, velocity, domain_shape, strict=True)
            )
            value = pltriton.load(
                f_ref.at[(*source_coordinates, direction)],
                mask=active,
                other=0.0,
            )
            populations.append(value.astype(compute_dtype))

        rho = populations[0]
        for population in populations[1:]:
            rho = rho + population

        velocity_fields = [jnp.zeros_like(rho) for _ in range(lattice.d)]
        for population, velocity in zip(populations, velocities, strict=True):
            velocity_fields = [field + component * population for field, component in zip(velocity_fields, velocity, strict=True)]

        # Triton's Ampere backend does not lower vector FP16 division. Promote
        # only this operation, then restore requested compute precision.
        velocity_fields = [field / rho for field in velocity_fields]

        velocity_squared = velocity_fields[0] * velocity_fields[0]
        for field in velocity_fields[1:]:
            velocity_squared = velocity_squared + field * field
        usqr = 1.5 * velocity_squared
        solid = pltriton.load(solid_ref.at[tuple(coordinates)], mask=active, other=True)

        for direction, (velocity, weight) in enumerate(zip(velocities, weights, strict=True)):
            velocity_dot = velocity[0] * velocity_fields[0]
            for component, field in zip(velocity[1:], velocity_fields[1:], strict=True):
                velocity_dot = velocity_dot + component * field
            cu = 3.0 * velocity_dot
            equilibrium = rho * weight * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
            collided = populations[direction] - omega * (populations[direction] - equilibrium)
            value = jnp.where(solid, populations[opposites[direction]], collided)
            pltriton.store(
                out_ref.at[(*coordinates, direction)],
                value.astype(output_dtype),
                mask=active,
            )

    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((*domain_shape, lattice.q), output_dtype),
            grid=grid,
            compiler_params=pltriton.CompilerParams(num_warps=num_warps),
            name=f"{lattice.name.lower()}_pull_bgk",
        )
    )


def build_fused_bgk_step(
    lattice,
    shape: Sequence[int],
    precision_policy,
    omega: float,
    *,
    return_postcollision: bool = False,
    block_size: int = 256,
    num_warps: int = 8,
):
    """Build an LBMBase.step-compatible fused Pallas kernel.

    Unlike build_fused_bgk, this kernel accepts and returns post-streaming
    populations. It performs local BGK collision, full-way bounce-back, and
    push streaming. This state layout preserves LBMBase.run behavior.

    Parameters
    ----------
    lattice : Lattice
        D2Q9, D3Q19, or D3Q27 lattice definition.

    shape : sequence of int
        Global domain shape matching lattice dimensionality.

    precision_policy : PrecisionPolicy
        Compute and output precision used by kernel.

    omega : float
        BGK relaxation rate.

    return_postcollision : bool, optional
        Return post-collision populations with post-streaming populations.

    block_size : int, optional
        Number of lattice nodes handled by each Pallas program.

    num_warps : int, optional
        Triton warps assigned to each Pallas program.

    Returns
    -------
    Callable
        JIT-compiled fused step. Return value matches return_postcollision.
    """
    supported_lattices = {"D2Q9": (2, 9), "D3Q19": (3, 19), "D3Q27": (3, 27)}
    if supported_lattices.get(lattice.name) != (lattice.d, lattice.q):
        raise NotImplementedError("Experimental fused BGK supports D2Q9, D3Q19, and D3Q27.")
    if len(shape) != lattice.d:
        raise ValueError(f"{lattice.name} requires a {lattice.d}D domain shape; received {len(shape)}D.")
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Experimental fused BGK currently requires a GPU backend.")
    if jax.device_count() != 1:
        raise NotImplementedError("Experimental fused BGK currently supports one visible GPU. Multi-GPU halo exchange is not implemented.")
    if block_size <= 0 or num_warps <= 0:
        raise ValueError("block_size and num_warps must be positive.")

    domain_shape = tuple(int(size) for size in shape)
    if min(domain_shape) <= 0:
        raise ValueError("All domain dimensions must be positive.")

    cells = int(np.prod(domain_shape))
    grid = ((cells + block_size - 1) // block_size,)
    velocities = tuple(tuple(int(value) for value in lattice.c[:, direction]) for direction in range(lattice.q))
    weights = tuple(float(value) for value in lattice.w)
    opposites = tuple(int(value) for value in lattice.opp_indices)
    compute_dtype = precision_policy.compute_dtype
    output_dtype = precision_policy.output_dtype
    field_shape = (*domain_shape, lattice.q)

    def collide(f_ref, solid_ref, stream_ref, postcollision_ref=None):
        cell = pl.program_id(0) * block_size + jnp.arange(block_size)
        active = cell < cells
        remainder = jnp.minimum(cell, cells - 1)
        coordinates = []
        for axis in range(lattice.d):
            stride = int(np.prod(domain_shape[axis + 1 :]))
            coordinate = remainder // stride
            remainder = remainder - coordinate * stride
            coordinates.append(coordinate)

        populations = []
        for direction in range(lattice.q):
            value = pltriton.load(f_ref.at[(*coordinates, direction)], mask=active, other=0.0)
            populations.append(value.astype(compute_dtype))

        rho = populations[0]
        for population in populations[1:]:
            rho = rho + population

        velocity_fields = [jnp.zeros_like(rho) for _ in range(lattice.d)]
        for population, velocity in zip(populations, velocities, strict=True):
            velocity_fields = [field + component * population for field, component in zip(velocity_fields, velocity, strict=True)]

        velocity_fields = [field / rho for field in velocity_fields]

        velocity_squared = velocity_fields[0] * velocity_fields[0]
        for field in velocity_fields[1:]:
            velocity_squared = velocity_squared + field * field
        usqr = 1.5 * velocity_squared
        solid = pltriton.load(solid_ref.at[tuple(coordinates)], mask=active, other=True)

        for direction, (velocity, weight) in enumerate(zip(velocities, weights, strict=True)):
            velocity_dot = velocity[0] * velocity_fields[0]
            for component, field in zip(velocity[1:], velocity_fields[1:], strict=True):
                velocity_dot = velocity_dot + component * field
            cu = 3.0 * velocity_dot
            equilibrium = rho * weight * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
            collided = populations[direction] - omega * (populations[direction] - equilibrium)
            postcollision = jnp.where(solid, populations[opposites[direction]], collided).astype(output_dtype)
            destination = tuple(
                (coordinate + component) % size for coordinate, component, size in zip(coordinates, velocity, domain_shape, strict=True)
            )
            pltriton.store(
                stream_ref.at[(*destination, direction)],
                postcollision,
                mask=active,
            )
            if postcollision_ref is not None:
                pltriton.store(
                    postcollision_ref.at[(*coordinates, direction)],
                    postcollision,
                    mask=active,
                )

    if return_postcollision:

        def kernel_with_postcollision(f_ref, solid_ref, stream_ref, postcollision_ref):
            collide(f_ref, solid_ref, stream_ref, postcollision_ref)

        kernel = kernel_with_postcollision
        out_shape = (
            jax.ShapeDtypeStruct(field_shape, output_dtype),
            jax.ShapeDtypeStruct(field_shape, output_dtype),
        )
    else:

        def kernel_without_postcollision(f_ref, solid_ref, stream_ref):
            collide(f_ref, solid_ref, stream_ref)

        kernel = kernel_without_postcollision
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)

    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid=grid,
            compiler_params=pltriton.CompilerParams(num_warps=num_warps),
            name=f"{lattice.name.lower()}_bgk_step",
        )
    )


def build_fused_soa_bgk_step(
    lattice,
    shape: Sequence[int],
    precision_policy,
    omega: float,
    *,
    return_postcollision: bool = False,
    block_size: int = 64,
    num_warps: int = 4,
    allow_multi_device_local: bool = False,
    with_force: bool = False,
    force_is_acceleration: bool = True,
    streaming: bool = True,
    use_solid_mask: bool = True,
):
    """Build a direction-major fused BGK Pallas step.

    Input and output use internal shape (q, *spatial_shape). Direction-major
    storage makes every same-direction load and store contiguous across lanes.

    Parameters
    ----------
    lattice : Lattice
        D2Q9, D3Q19, or D3Q27 lattice definition.

    shape : sequence of int
        Global domain shape matching lattice dimensionality.

    precision_policy : PrecisionPolicy
        Compute and output precision used by kernel.

    omega : float
        BGK relaxation rate.

    return_postcollision : bool, optional
        Return post-collision populations with post-streaming populations.

    block_size : int, optional
        Number of lattice nodes handled by each Pallas program.

    num_warps : int, optional
        Triton warps assigned to each Pallas program.

    allow_multi_device_local : bool, optional
        Permit construction for a local shard inside a multi-device wrapper.

    with_force : bool, optional
        Accept a spatial force field and apply the standard exact-difference
        BGK forcing term within the fused collision.

    force_is_acceleration : bool, optional
        Interpret the force input as a velocity increment when True. When False,
        divide the raw force density by the density already computed in the kernel.

    streaming : bool, optional
        Push-stream post-collision populations when True. When False, return
        only the post-collision field for general BC composition.

    use_solid_mask : bool, optional
        Apply the dense solid mask inside the collision kernel. Set to False when
        boundary conditions are applied by a separate SoA boundary pipeline.

    Returns
    -------
    Callable
        JIT-compiled fused step with direction-major input and output.
    """
    supported_lattices = {"D2Q9": (2, 9), "D3Q19": (3, 19), "D3Q27": (3, 27)}
    if supported_lattices.get(lattice.name) != (lattice.d, lattice.q):
        raise NotImplementedError("Experimental fused BGK supports D2Q9, D3Q19, and D3Q27.")
    if len(shape) != lattice.d:
        raise ValueError(f"{lattice.name} requires a {lattice.d}D domain shape; received {len(shape)}D.")
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Experimental fused BGK currently requires a GPU backend.")
    if jax.device_count() != 1 and not allow_multi_device_local:
        raise NotImplementedError("This builder requires one visible GPU unless used for a local shard.")
    if block_size <= 0 or num_warps <= 0:
        raise ValueError("block_size and num_warps must be positive.")

    domain_shape = tuple(int(size) for size in shape)
    if min(domain_shape) <= 0:
        raise ValueError("All domain dimensions must be positive.")

    cells = int(np.prod(domain_shape))
    grid = ((cells + block_size - 1) // block_size,)
    velocities = tuple(tuple(int(value) for value in lattice.c[:, direction]) for direction in range(lattice.q))
    weights = tuple(float(value) for value in lattice.w)
    opposites = tuple(int(value) for value in lattice.opp_indices)
    compute_dtype = precision_policy.compute_dtype
    output_dtype = precision_policy.output_dtype
    field_shape = (lattice.q, *domain_shape)

    def collide(f_ref, solid_ref, stream_ref, postcollision_ref=None, force_ref=None):
        cell = pl.program_id(0) * block_size + jnp.arange(block_size)
        active = cell < cells
        remainder = jnp.minimum(cell, cells - 1)
        coordinates = []
        for axis in range(lattice.d):
            stride = int(np.prod(domain_shape[axis + 1 :]))
            coordinate = remainder // stride
            remainder = remainder - coordinate * stride
            coordinates.append(coordinate)

        populations = []
        for direction in range(lattice.q):
            value = pltriton.load(f_ref.at[(direction, *coordinates)], mask=active, other=0.0)
            populations.append(value.astype(compute_dtype))

        rho = populations[0]
        for population in populations[1:]:
            rho = rho + population

        velocity_fields = [jnp.zeros_like(rho) for _ in range(lattice.d)]
        for population, velocity in zip(populations, velocities, strict=True):
            velocity_fields = [field + component * population for field, component in zip(velocity_fields, velocity, strict=True)]

        velocity_fields = [field / rho for field in velocity_fields]

        velocity_squared = velocity_fields[0] * velocity_fields[0]
        for field in velocity_fields[1:]:
            velocity_squared = velocity_squared + field * field
        usqr = 1.5 * velocity_squared
        if force_ref is not None:
            force_fields = [
                pltriton.load(force_ref.at[(*coordinates, component)], mask=active, other=0.0).astype(compute_dtype) for component in range(lattice.d)
            ]
            if not force_is_acceleration:
                force_fields = [field / rho for field in force_fields]
            velocity_force = velocity_fields[0] * force_fields[0]
            force_squared = force_fields[0] * force_fields[0]
            for velocity_field, force_field in zip(velocity_fields[1:], force_fields[1:], strict=True):
                velocity_force = velocity_force + velocity_field * force_field
                force_squared = force_squared + force_field * force_field
            delta_usqr = 1.5 * (2.0 * velocity_force + force_squared)
        solid = (
            pltriton.load(solid_ref.at[tuple(coordinates)], mask=active, other=True)
            if use_solid_mask
            else None
        )

        for direction, (velocity, weight) in enumerate(zip(velocities, weights, strict=True)):
            velocity_dot = velocity[0] * velocity_fields[0]
            for component, field in zip(velocity[1:], velocity_fields[1:], strict=True):
                velocity_dot = velocity_dot + component * field
            cu = 3.0 * velocity_dot
            equilibrium = rho * weight * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
            collided = populations[direction] - omega * (populations[direction] - equilibrium)
            if force_ref is not None:
                force_dot = velocity[0] * force_fields[0]
                for component, force_field in zip(velocity[1:], force_fields[1:], strict=True):
                    force_dot = force_dot + component * force_field
                dcu = 3.0 * force_dot
                collided = collided + rho * weight * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr)
            postcollision = (
                jnp.where(solid, populations[opposites[direction]], collided)
                if use_solid_mask
                else collided
            ).astype(output_dtype)
            if stream_ref is not None:
                destination = tuple(
                    (coordinate + component) % size for coordinate, component, size in zip(coordinates, velocity, domain_shape, strict=True)
                )
                pltriton.store(
                    stream_ref.at[(direction, *destination)],
                    postcollision,
                    mask=active,
                )
            if postcollision_ref is not None:
                pltriton.store(
                    postcollision_ref.at[(direction, *coordinates)],
                    postcollision,
                    mask=active,
                )

    if not use_solid_mask:
        if streaming:
            raise ValueError("Mask-free BGK collision requires streaming=False.")
        if with_force:

            def kernel_collision_force_no_solid(f_ref, force_ref, postcollision_ref):
                collide(f_ref, None, None, postcollision_ref, force_ref)

            kernel = kernel_collision_force_no_solid
        else:

            def kernel_collision_no_solid(f_ref, postcollision_ref):
                collide(f_ref, None, None, postcollision_ref)

            kernel = kernel_collision_no_solid
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)
    elif not streaming and with_force:

        def kernel_collision_force(f_ref, solid_ref, force_ref, postcollision_ref):
            collide(f_ref, solid_ref, None, postcollision_ref, force_ref)

        kernel = kernel_collision_force
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)
    elif not streaming:

        def kernel_collision(f_ref, solid_ref, postcollision_ref):
            collide(f_ref, solid_ref, None, postcollision_ref)

        kernel = kernel_collision
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)
    elif return_postcollision and with_force:

        def kernel_with_postcollision_force(f_ref, solid_ref, force_ref, stream_ref, postcollision_ref):
            collide(f_ref, solid_ref, stream_ref, postcollision_ref, force_ref)

        kernel = kernel_with_postcollision_force
        out_shape = (
            jax.ShapeDtypeStruct(field_shape, output_dtype),
            jax.ShapeDtypeStruct(field_shape, output_dtype),
        )
    elif return_postcollision:

        def kernel_with_postcollision(f_ref, solid_ref, stream_ref, postcollision_ref):
            collide(f_ref, solid_ref, stream_ref, postcollision_ref)

        kernel = kernel_with_postcollision
        out_shape = (
            jax.ShapeDtypeStruct(field_shape, output_dtype),
            jax.ShapeDtypeStruct(field_shape, output_dtype),
        )
    elif with_force:

        def kernel_with_force(f_ref, solid_ref, force_ref, stream_ref):
            collide(f_ref, solid_ref, stream_ref, force_ref=force_ref)

        kernel = kernel_with_force
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)
    else:

        def kernel_without_postcollision(f_ref, solid_ref, stream_ref):
            collide(f_ref, solid_ref, stream_ref)

        kernel = kernel_without_postcollision
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)

    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid=grid,
            compiler_params=pltriton.CompilerParams(num_warps=num_warps),
            name=f"{lattice.name.lower()}_soa_bgk_step",
        )
    )


def build_fused_soa_mrt_step(
    lattice,
    shape: Sequence[int],
    precision_policy,
    collision_terms,
    *,
    return_postcollision: bool = False,
    block_size: int = 64,
    num_warps: int = 4,
    allow_multi_device_local: bool = False,
    with_force: bool = False,
    force_is_acceleration: bool = True,
    streaming: bool = True,
    use_solid_mask: bool = True,
    _surface_tension=None,
):
    """Build a direction-major fused MRT Pallas step.

    Mirrors build_fused_soa_bgk_step, replacing the scalar-omega BGK
    collision with jax_lab.core.models.MRTSim's symbolic, sparse-coefficient
    fused collision matrix (MRTSim.collision_terms: one tuple of
    (input_direction, coefficient) pairs per output direction, derived from
    M @ S @ M_inv). With an identity M and a uniform S diagonal,
    collision_terms reduces to exactly one (direction, omega) term per
    output direction, reproducing build_fused_soa_bgk_step's BGK collision
    exactly.

    Parameters
    ----------
    lattice : Lattice
        D2Q9, D3Q19, or D3Q27 lattice definition.

    shape : sequence of int
        Global domain shape matching lattice dimensionality.

    precision_policy : PrecisionPolicy
        Compute and output precision used by kernel.

    collision_terms : sequence of sequence of (int, float)
        Per-output-direction sparse MRT collision coefficients, exactly as
        built by MRTSim.__init__ (self.collision_terms).

    return_postcollision : bool, optional
        Return post-collision populations with post-streaming populations.

    block_size : int, optional
        Number of lattice nodes handled by each Pallas program.

    num_warps : int, optional
        Triton warps assigned to each Pallas program.

    allow_multi_device_local : bool, optional
        Permit construction for a local shard inside a multi-device wrapper.

    with_force : bool, optional
        Accept a spatial force field and apply the standard exact-difference
        forcing term (identical formula to the BGK kernel and to
        MRTSim._compute_force_delta_feq) within the fused collision.

    force_is_acceleration : bool, optional
        Interpret force input as a velocity increment when True. When False,
        divide raw force density by the density already computed in the kernel.

    streaming : bool, optional
        Push-stream post-collision populations when True. When False, return
        only the post-collision field for general BC composition.

    use_solid_mask : bool, optional
        Apply the dense solid mask inside the collision kernel. Set to False when
        boundary conditions are applied by a separate SoA boundary pipeline.

    Returns
    -------
    Callable
        JIT-compiled fused step with direction-major input and output.
    """
    supported_lattices = {"D2Q9": (2, 9), "D3Q19": (3, 19), "D3Q27": (3, 27)}
    if supported_lattices.get(lattice.name) != (lattice.d, lattice.q):
        raise NotImplementedError("Experimental fused MRT supports D2Q9, D3Q19, and D3Q27.")
    if len(shape) != lattice.d:
        raise ValueError(f"{lattice.name} requires a {lattice.d}D domain shape; received {len(shape)}D.")
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Experimental fused MRT currently requires a GPU backend.")
    if jax.device_count() != 1 and not allow_multi_device_local:
        raise NotImplementedError("This builder requires one visible GPU unless used for a local shard.")
    if block_size <= 0 or num_warps <= 0:
        raise ValueError("block_size and num_warps must be positive.")
    if len(collision_terms) != lattice.q:
        raise ValueError(f"collision_terms must have one entry per direction ({lattice.q}); received {len(collision_terms)}.")

    domain_shape = tuple(int(size) for size in shape)
    if min(domain_shape) <= 0:
        raise ValueError("All domain dimensions must be positive.")

    cells = int(np.prod(domain_shape))
    grid = ((cells + block_size - 1) // block_size,)
    velocities = tuple(tuple(int(value) for value in lattice.c[:, direction]) for direction in range(lattice.q))
    weights = tuple(float(value) for value in lattice.w)
    opposites = tuple(int(value) for value in lattice.opp_indices)
    # Static (Python-level, not traced) per-output-direction sparse coefficients, closed over by the
    # kernel below exactly like `omega` is in build_fused_soa_bgk_step.
    terms = tuple(tuple((int(input_direction), float(coefficient)) for input_direction, coefficient in column) for column in collision_terms)
    compute_dtype = precision_policy.compute_dtype
    output_dtype = precision_policy.output_dtype
    field_shape = (lattice.q, *domain_shape)

    surface_config = None
    if _surface_tension is not None:
        if lattice.name not in ("D2Q9", "D3Q19"):
            raise NotImplementedError("Fused MRT surface tension supports D2Q9 and D3Q19.")
        if not with_force or streaming:
            raise ValueError("Fused MRT surface tension requires with_force=True and streaming=False.")
        moment_pairs = ((0, 0), (0, 1), (1, 1)) if lattice.d == 2 else (
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
            (2, 2),
        )
        g_ff = np.asarray(_surface_tension["g_ff"])
        surface_weights = tuple(
            tuple(float(g_ff[q_index] * lattice.c[i, q_index] * lattice.c[j, q_index]) for q_index in range(lattice.q))
            for i, j in moment_pairs
        )
        surface_centers = tuple(sum(row) for row in surface_weights)
        m_inv = tuple(tuple(float(value) for value in row) for row in np.asarray(_surface_tension["m_inv"]))
        surface_config = {
            "kappa": float(_surface_tension["kappa"]),
            "A": float(_surface_tension["A"]),
            "s_e": float(_surface_tension["s_e"]),
            "s_eta": float(_surface_tension["s_eta"]),
            "s_v": float(_surface_tension["s_v"]),
            "weights": surface_weights,
            "centers": surface_centers,
            "m_inv": m_inv,
            "x_needs_halo": bool(_surface_tension["x_needs_halo"]),
        }

    def collide(f_ref, solid_ref, stream_ref, postcollision_ref=None, force_ref=None, surface_psi_ref=None):
        cell = pl.program_id(0) * block_size + jnp.arange(block_size)
        active = cell < cells
        remainder = jnp.minimum(cell, cells - 1)
        coordinates = []
        for axis in range(lattice.d):
            stride = int(np.prod(domain_shape[axis + 1 :]))
            coordinate = remainder // stride
            remainder = remainder - coordinate * stride
            coordinates.append(coordinate)

        populations = []
        for direction in range(lattice.q):
            value = pltriton.load(f_ref.at[(direction, *coordinates)], mask=active, other=0.0)
            populations.append(value.astype(compute_dtype))

        rho = populations[0]
        for population in populations[1:]:
            rho = rho + population

        velocity_fields = [jnp.zeros_like(rho) for _ in range(lattice.d)]
        for population, velocity in zip(populations, velocities, strict=True):
            velocity_fields = [field + component * population for field, component in zip(velocity_fields, velocity, strict=True)]

        velocity_fields = [field / rho for field in velocity_fields]

        velocity_squared = velocity_fields[0] * velocity_fields[0]
        for field in velocity_fields[1:]:
            velocity_squared = velocity_squared + field * field
        usqr = 1.5 * velocity_squared
        if force_ref is not None:
            force_fields = [
                pltriton.load(force_ref.at[(*coordinates, component)], mask=active, other=0.0).astype(compute_dtype) for component in range(lattice.d)
            ]
            if not force_is_acceleration:
                force_fields = [field / rho for field in force_fields]
            velocity_force = velocity_fields[0] * force_fields[0]
            force_squared = force_fields[0] * force_fields[0]
            for velocity_field, force_field in zip(velocity_fields[1:], force_fields[1:], strict=True):
                velocity_force = velocity_force + velocity_field * force_field
                force_squared = force_squared + force_field * force_field
            delta_usqr = 1.5 * (2.0 * velocity_force + force_squared)
        solid = (
            pltriton.load(solid_ref.at[tuple(coordinates)], mask=active, other=True)
            if use_solid_mask
            else None
        )

        surface_correction = None
        if surface_config is not None:
            center_coordinates = list(coordinates)
            if surface_config["x_needs_halo"]:
                center_coordinates[0] = center_coordinates[0] + 1
            psi = pltriton.load(surface_psi_ref.at[(*center_coordinates, 0)], mask=active, other=0.0).astype(compute_dtype)
            psi_moments = [jnp.zeros_like(psi) for _ in surface_config["weights"]]
            psi_squared_moments = [jnp.zeros_like(psi) for _ in surface_config["weights"]]
            for q_index, velocity in enumerate(velocities):
                source_coordinates = []
                for axis, (coordinate, component, size) in enumerate(zip(coordinates, velocity, domain_shape, strict=True)):
                    if axis == 0 and surface_config["x_needs_halo"]:
                        source_coordinates.append(coordinate + 1 - component)
                    else:
                        source_coordinates.append((coordinate - component) % size)
                neighbor = pltriton.load(surface_psi_ref.at[(*source_coordinates, 0)], mask=active, other=0.0).astype(compute_dtype)
                neighbor_squared = neighbor * neighbor
                for moment_index, weights_row in enumerate(surface_config["weights"]):
                    weight = weights_row[q_index]
                    if weight != 0.0:
                        psi_moments[moment_index] = psi_moments[moment_index] + weight * neighbor
                        psi_squared_moments[moment_index] = psi_squared_moments[moment_index] + weight * neighbor_squared

            qmoments = []
            psi_squared = psi * psi
            for psi_moment, psi_squared_moment, center_weight in zip(
                psi_moments,
                psi_squared_moments,
                surface_config["centers"],
                strict=True,
            ):
                centered_psi = psi_moment - psi * center_weight
                centered_psi_squared = psi_squared_moment - psi_squared * center_weight
                qmoments.append(
                    -surface_config["kappa"]
                    * (
                        (1.0 - surface_config["A"]) * psi * centered_psi
                        + 0.5 * surface_config["A"] * centered_psi_squared
                    )
                )

            moments = [jnp.zeros_like(psi) for _ in range(lattice.q)]
            if lattice.d == 2:
                qxx, qxy, qyy = qmoments
                moments[1] = 1.5 * surface_config["s_e"] * (qxx + qyy)
                moments[2] = -1.5 * surface_config["s_eta"] * (qxx + qyy)
                moments[7] = -surface_config["s_v"] * (qxx - qyy)
                moments[8] = -surface_config["s_v"] * qxy
            else:
                qxx, qxy, qxz, qyy, qyz, qzz = qmoments
                moments[1] = (2.0 / 5.0) * surface_config["s_e"] * (qxx + qyy + qzz)
                moments[9] = -surface_config["s_v"] * (2.0 * qxx - qyy - qzz)
                moments[11] = -surface_config["s_v"] * (qyy - qzz)
                moments[13] = -surface_config["s_v"] * qxy
                moments[14] = -surface_config["s_v"] * qyz
                moments[15] = -surface_config["s_v"] * qxz
            surface_correction = []
            for output_direction in range(lattice.q):
                correction = jnp.zeros_like(psi)
                for moment_index, moment in enumerate(moments):
                    coefficient = surface_config["m_inv"][moment_index][output_direction]
                    if coefficient != 0.0:
                        correction = correction + coefficient * moment
                surface_correction.append(correction)

        # MRT couples output direction j to non-equilibrium populations at other input directions
        # (collision_terms[j] is not just (j, omega) in general), unlike BGK where each output only
        # needs its own direction. So every direction's equilibrium/difference is computed in one pass
        # first, then relaxed and stored in a second pass, instead of BGK's single interleaved loop.
        difference = []
        cu_by_direction = []
        for direction, (velocity, weight) in enumerate(zip(velocities, weights, strict=True)):
            velocity_dot = velocity[0] * velocity_fields[0]
            for component, field in zip(velocity[1:], velocity_fields[1:], strict=True):
                velocity_dot = velocity_dot + component * field
            cu = 3.0 * velocity_dot
            equilibrium = rho * weight * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
            difference.append(populations[direction] - equilibrium)
            cu_by_direction.append(cu)

        for output_direction in range(lattice.q):
            velocity = velocities[output_direction]
            relaxed = None
            for input_direction, coefficient in terms[output_direction]:
                term = difference[input_direction] * coefficient
                relaxed = term if relaxed is None else relaxed + term
            collided = populations[output_direction] if relaxed is None else populations[output_direction] - relaxed
            if force_ref is not None:
                weight = weights[output_direction]
                cu = cu_by_direction[output_direction]
                force_dot = velocity[0] * force_fields[0]
                for component, force_field in zip(velocity[1:], force_fields[1:], strict=True):
                    force_dot = force_dot + component * force_field
                dcu = 3.0 * force_dot
                collided = collided + rho * weight * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr)
            if surface_correction is not None:
                collided = collided + surface_correction[output_direction]
            postcollision = (
                jnp.where(solid, populations[opposites[output_direction]], collided)
                if use_solid_mask
                else collided
            ).astype(output_dtype)
            if stream_ref is not None:
                destination = tuple(
                    (coordinate + component) % size for coordinate, component, size in zip(coordinates, velocity, domain_shape, strict=True)
                )
                pltriton.store(
                    stream_ref.at[(output_direction, *destination)],
                    postcollision,
                    mask=active,
                )
            if postcollision_ref is not None:
                pltriton.store(
                    postcollision_ref.at[(output_direction, *coordinates)],
                    postcollision,
                    mask=active,
                )

    if not use_solid_mask:
        if streaming:
            raise ValueError("Mask-free MRT collision requires streaming=False.")
        if surface_config is not None:

            def kernel_collision_force_surface_no_solid(f_ref, force_ref, surface_psi_ref, postcollision_ref):
                collide(f_ref, None, None, postcollision_ref, force_ref, surface_psi_ref)

            kernel = kernel_collision_force_surface_no_solid
        elif with_force:

            def kernel_collision_force_no_solid(f_ref, force_ref, postcollision_ref):
                collide(f_ref, None, None, postcollision_ref, force_ref)

            kernel = kernel_collision_force_no_solid
        else:

            def kernel_collision_no_solid(f_ref, postcollision_ref):
                collide(f_ref, None, None, postcollision_ref)

            kernel = kernel_collision_no_solid
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)
    elif surface_config is not None:

        def kernel_collision_force_surface(f_ref, solid_ref, force_ref, surface_psi_ref, postcollision_ref):
            collide(f_ref, solid_ref, None, postcollision_ref, force_ref, surface_psi_ref)

        kernel = kernel_collision_force_surface
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)
    elif not streaming and with_force:

        def kernel_collision_force(f_ref, solid_ref, force_ref, postcollision_ref):
            collide(f_ref, solid_ref, None, postcollision_ref, force_ref)

        kernel = kernel_collision_force
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)
    elif not streaming:

        def kernel_collision(f_ref, solid_ref, postcollision_ref):
            collide(f_ref, solid_ref, None, postcollision_ref)

        kernel = kernel_collision
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)
    elif return_postcollision and with_force:

        def kernel_with_postcollision_force(f_ref, solid_ref, force_ref, stream_ref, postcollision_ref):
            collide(f_ref, solid_ref, stream_ref, postcollision_ref, force_ref)

        kernel = kernel_with_postcollision_force
        out_shape = (
            jax.ShapeDtypeStruct(field_shape, output_dtype),
            jax.ShapeDtypeStruct(field_shape, output_dtype),
        )
    elif return_postcollision:

        def kernel_with_postcollision(f_ref, solid_ref, stream_ref, postcollision_ref):
            collide(f_ref, solid_ref, stream_ref, postcollision_ref)

        kernel = kernel_with_postcollision
        out_shape = (
            jax.ShapeDtypeStruct(field_shape, output_dtype),
            jax.ShapeDtypeStruct(field_shape, output_dtype),
        )
    elif with_force:

        def kernel_with_force(f_ref, solid_ref, force_ref, stream_ref):
            collide(f_ref, solid_ref, stream_ref, force_ref=force_ref)

        kernel = kernel_with_force
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)
    else:

        def kernel_without_postcollision(f_ref, solid_ref, stream_ref):
            collide(f_ref, solid_ref, stream_ref)

        kernel = kernel_without_postcollision
        out_shape = jax.ShapeDtypeStruct(field_shape, output_dtype)

    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid=grid,
            compiler_params=pltriton.CompilerParams(num_warps=num_warps),
            name=(
                f"{lattice.name.lower()}_soa_mrt_surface_tension_step"
                if surface_config is not None
                else f"{lattice.name.lower()}_soa_mrt_step"
            ),
        )
    )


def build_fused_soa_mrt_surface_tension_step(
    lattice,
    shape: Sequence[int],
    precision_policy,
    collision_terms,
    *,
    kappa: float,
    A: float,
    s_e: float,
    s_eta: float,
    s_v: float,
    g_ff,
    m_inv,
    x_needs_halo: bool,
    block_size: int = 64,
    num_warps: int = 4,
    allow_multi_device_local: bool = False,
    use_solid_mask: bool = True,
):
    """Build a fused MRT collision/EDM/surface-tension Pallas kernel.

    The pseudopotential neighbor moments, moment-space surface correction, inverse
    moment transform, MRT relaxation, and EDM force addition are evaluated in one
    kernel. surface_psi must include a one-cell X halo when x_needs_halo is
    true; the distributed wrapper supplies that halo for any device count.
    """
    return build_fused_soa_mrt_step(
        lattice,
        shape,
        precision_policy,
        collision_terms,
        block_size=block_size,
        num_warps=num_warps,
        allow_multi_device_local=allow_multi_device_local,
        with_force=True,
        force_is_acceleration=False,
        streaming=False,
        use_solid_mask=use_solid_mask,
        _surface_tension={
            "kappa": kappa,
            "A": A,
            "s_e": s_e,
            "s_eta": s_eta,
            "s_v": s_v,
            "g_ff": g_ff,
            "m_inv": m_inv,
            "x_needs_halo": x_needs_halo,
        },
    )


def build_soa_streaming(
    lattice,
    shape: Sequence[int],
    precision_policy,
    *,
    block_size: int = 64,
    num_warps: int = 4,
    allow_multi_device_local: bool = False,
):
    """Build a direction-major Pallas push-streaming kernel.

    Parameters
    ----------
    lattice : Lattice
        D2Q9, D3Q19, or D3Q27 lattice definition.

    shape : sequence of int
        Local spatial shape. X is the local shard extent for multi-GPU use.

    precision_policy : PrecisionPolicy
        Output precision policy for the population field.

    block_size : int, optional
        Number of lattice nodes handled by each Pallas program.

    num_warps : int, optional
        Triton warps assigned to each program.

    allow_multi_device_local : bool, optional
        Permit construction for a local shard inside a multi-device wrapper.

    Returns
    -------
    Callable
        JIT-compiled (q, *shape) -> (q, *shape) local streaming kernel.
    """
    supported_lattices = {"D2Q9": (2, 9), "D3Q19": (3, 19), "D3Q27": (3, 27)}
    if supported_lattices.get(lattice.name) != (lattice.d, lattice.q):
        raise NotImplementedError("Experimental streaming supports D2Q9, D3Q19, and D3Q27.")
    if len(shape) != lattice.d:
        raise ValueError(f"{lattice.name} requires a {lattice.d}D domain shape; received {len(shape)}D.")
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Experimental streaming currently requires a GPU backend.")
    if jax.device_count() != 1 and not allow_multi_device_local:
        raise NotImplementedError("This builder requires one visible GPU unless used for a local shard.")
    if block_size <= 0 or num_warps <= 0:
        raise ValueError("block_size and num_warps must be positive.")

    domain_shape = tuple(int(size) for size in shape)
    if min(domain_shape) <= 0:
        raise ValueError("All domain dimensions must be positive.")
    cells = int(np.prod(domain_shape))
    grid = ((cells + block_size - 1) // block_size,)
    velocities = tuple(tuple(int(value) for value in lattice.c[:, direction]) for direction in range(lattice.q))
    field_shape = (lattice.q, *domain_shape)
    dtype = precision_policy.output_dtype

    def kernel(f_ref, out_ref):
        cell = pl.program_id(0) * block_size + jnp.arange(block_size)
        active = cell < cells
        remainder = jnp.minimum(cell, cells - 1)
        coordinates = []
        for axis in range(lattice.d):
            stride = int(np.prod(domain_shape[axis + 1 :]))
            coordinate = remainder // stride
            remainder = remainder - coordinate * stride
            coordinates.append(coordinate)
        for direction, velocity in enumerate(velocities):
            value = pltriton.load(f_ref.at[(direction, *coordinates)], mask=active, other=0.0)
            destination = tuple(
                (coordinate + component) % size for coordinate, component, size in zip(coordinates, velocity, domain_shape, strict=True)
            )
            pltriton.store(out_ref.at[(direction, *destination)], value.astype(dtype), mask=active)

    return jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(field_shape, dtype),
            grid=grid,
            compiler_params=pltriton.CompilerParams(num_warps=num_warps),
            name=f"{lattice.name.lower()}_soa_stream",
        )
    )


# Compatibility alias for early benchmark scripts.
build_fused_d3q19_bgk = build_fused_bgk
