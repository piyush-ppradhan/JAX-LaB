"""Experimental Pallas kernels for the multiphase pseudopotential (Shan-Chen/Zhang-Chen)
force pipeline: per-component potential (psi, U) and the q-direction-weighted neighbor
stencil compute_fluid_fluid_force needs. Collision itself reuses
pallas_kernels.build_fused_soa_bgk_step(with_force=True) unchanged - see
jax_lab/experimental/pallas_multiphase_sim.py and todo.txt's Multiphase Pallas port plan.

Profiling before this file was written (see performance.txt) showed base
MultiphaseBGK.collision() (n_components=2) at ~19x the wall time of one Pallas
single-phase collision call - each of apply_contact_angle, compute_potential, and
compute_fluid_fluid_force is its own separate jax.jit boundary (materializing a full
pytree to HBM between each), on top of collision's own fin/feq/fneq/fout materializations.
These two kernels fuse the potential and neighbor-stencil steps each into one Pallas
launch per component (per scalar field, for the stencil), leaving only the small,
already-cheap per-output-component combine (compute_fluid_fluid_force's g_kkprime/A
weighted sum) and apply_contact_angle (boundary-only, not dense-grid work, so not a
natural fit for a Pallas kernel) as plain JAX ops - unchanged, full config generality.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton


def eos_pressure_fn(eos, component_index):
    """Return one component's elementwise pressure callable.

    Parameters
    ----------
    eos : jax_lab.core.eos.EOS
        EOS implementing pressure_component(rho, component_index).

    component_index : int
        Which component's parameters to extract.

    Returns
    -------
    Callable[[jax.Array], jax.Array]
        Elementwise pressure formula, matching that EOS class's own EOS() method exactly.

    """
    return lambda rho: eos.pressure_component(rho, component_index)


def build_fused_soa_potential_step(
    lattice,
    shape: Sequence[int],
    precision_policy,
    pressure_fn,
    k: float,
    g_diag: float,
    *,
    block_size: int = 64,
    num_warps: int = 4,
    allow_multi_device_local: bool = False,
    return_psi: bool = True,
    return_u: bool = True,
):
    """Build a fused Pallas kernel computing one component's Shan-Chen pseudopotential
    (psi) and Zhang-Chen potential (U) from its density field.

    Purely local (no neighbor access - unlike build_fused_soa_neighbor_stencil_step
    below), so this needs no halo handling and treats the domain as a flat array of
    cells, matching Multiphase.compute_potential's formula exactly:
    U = k*p - cs2*rho, psi = sqrt(2*U / g_diag).

    Parameters
    ----------
    lattice : Lattice
        D2Q9, D3Q19, or D3Q27 lattice definition (only cs2 and name are used).

    shape : sequence of int
        Local spatial shape. X is the local shard extent for multi-GPU use.

    precision_policy : PrecisionPolicy
        Compute precision used by the kernel; psi/U are returned at compute precision
        (never narrowed to output_dtype), matching Multiphase.compute_potential.

    pressure_fn : Callable[[jax.Array], jax.Array]
        Elementwise EOS pressure formula for this component, e.g. from eos_pressure_fn.

    k : float
        Per-component modified-pressure coefficient (Multiphase.k[component]).

    g_diag : float
        This component's self-interaction strength (g_kkprime.diagonal()[component]).

    block_size, num_warps, allow_multi_device_local :
        See build_fused_soa_bgk_step.

    return_psi, return_u : bool, optional
        Materialize only the potential fields required by the configured force and
        surface-tension model. At least one output must be selected.

    Returns
    -------
    Callable
        JIT-compiled rho -> (psi_or_none, U_or_none). Selected arrays have
        the same shape as rho ((*shape, 1)).
    """
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Experimental fused multiphase potential currently requires a GPU backend.")
    if jax.device_count() != 1 and not allow_multi_device_local:
        raise NotImplementedError("This builder requires one visible GPU unless used for a local shard.")
    if block_size <= 0 or num_warps <= 0:
        raise ValueError("block_size and num_warps must be positive.")
    if not return_psi and not return_u:
        raise ValueError("At least one potential output must be requested.")

    domain_shape = tuple(int(size) for size in shape)
    if min(domain_shape) <= 0:
        raise ValueError("All domain dimensions must be positive.")

    cells = int(np.prod(domain_shape))
    grid = ((cells + block_size - 1) // block_size,)
    compute_dtype = precision_policy.compute_dtype
    cs2 = float(lattice.cs2)

    def potential_values(rho_ref):
        cell = pl.program_id(0) * block_size + jnp.arange(block_size)
        active = cell < cells
        rho = pltriton.load(rho_ref.at[cell], mask=active, other=0.0).astype(compute_dtype)
        p = pressure_fn(rho)
        u_val = k * p - cs2 * rho
        psi_val = jnp.sqrt(2.0 * u_val / g_diag) if return_psi else None
        return cell, active, psi_val, u_val

    def kernel_both(rho_ref, psi_ref, u_ref):
        cell, active, psi_val, u_val = potential_values(rho_ref)
        pltriton.store(psi_ref.at[cell], psi_val.astype(compute_dtype), mask=active)
        pltriton.store(u_ref.at[cell], u_val.astype(compute_dtype), mask=active)

    def kernel_psi(rho_ref, psi_ref):
        cell, active, psi_val, _ = potential_values(rho_ref)
        pltriton.store(psi_ref.at[cell], psi_val.astype(compute_dtype), mask=active)

    def kernel_u(rho_ref, u_ref):
        cell, active, _, u_val = potential_values(rho_ref)
        pltriton.store(u_ref.at[cell], u_val.astype(compute_dtype), mask=active)

    flat_shape = jax.ShapeDtypeStruct((cells,), compute_dtype)
    if return_psi and return_u:
        kernel = kernel_both
        out_shape = (flat_shape, flat_shape)
    elif return_psi:
        kernel = kernel_psi
        out_shape = flat_shape
    else:
        kernel = kernel_u
        out_shape = flat_shape

    call = jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid=grid,
            compiler_params=pltriton.CompilerParams(num_warps=num_warps),
            name=f"{lattice.name.lower()}_potential_step",
        )
    )

    def run(rho):
        flat = jnp.reshape(rho[..., 0], (cells,))
        result = call(flat)
        if return_psi and return_u:
            psi, u = result
            return psi.reshape((*domain_shape, 1)), u.reshape((*domain_shape, 1))
        if return_psi:
            return result.reshape((*domain_shape, 1)), None
        return None, result.reshape((*domain_shape, 1))

    return jax.jit(run)


def build_fused_soa_neighbor_stencil_step(
    lattice,
    shape: Sequence[int],
    precision_policy,
    weights,
    *,
    block_size: int = 64,
    num_warps: int = 4,
    allow_multi_device_local: bool = False,
):
    """Build a fused Pallas kernel computing a q-direction-weighted neighbor sum of a
    scalar field: out[x] = sum_q weights[:, q] * field(x - c_q), matching
    Multiphase._neighbor_stencil_m's pull-gather convention exactly (verified against
    it directly in tests, not just derived from the formula).

    Builds two variants and returns a single callable run(field, x_needs_halo) that
    picks between them at call time (both are pre-compiled; x_needs_halo just selects
    which one runs, since it is known statically by the caller from n_devices):
    x_needs_halo=False treats X the same as Y/Z, wrapping around locally with a plain
    modulo (correct for one GPU, where the local shard is the whole domain).
    x_needs_halo=True expects field padded with one extra X layer on each side
    ((shape[0] + 2, *shape[1:], 1), built by the caller via a 1-layer x halo exchange -
    e.g. send_left/send_right, matching every other halo in this package) and reads X
    neighbors directly from that padding instead of wrapping within the local shard. Y/Z
    (never sharded) always wrap around locally, which is exact since those axes are
    never split across devices.

    Parameters
    ----------
    lattice : Lattice
        D2Q9, D3Q19, or D3Q27 lattice definition.

    shape : sequence of int
        Local (unpadded) spatial shape. Both the output and the x_needs_halo=False
        input have this shape; the x_needs_halo=True input is shape with 1 added to
        the X extent.

    precision_policy : PrecisionPolicy
        Compute precision used by the kernel.

    weights : numpy.ndarray
        Per-direction weight vector(s), shape (out_channels, q) (e.g. G_ff[None, :] *
        c for the force stencil's (dim, q) vector weights, or G_ff[None, :] for the
        wetting-denominator's scalar weights).

    block_size, num_warps, allow_multi_device_local :
        See build_fused_soa_bgk_step.

    Returns
    -------
    Callable[[jax.Array, bool], jax.Array]
        run(field, x_needs_halo) -> stencil, field shaped (*shape, 1) (or
        (shape[0] + 2, *shape[1:], 1) when x_needs_halo), stencil shaped
        (*shape, out_channels).
    """
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Experimental fused multiphase neighbor stencil currently requires a GPU backend.")
    if jax.device_count() != 1 and not allow_multi_device_local:
        raise NotImplementedError("This builder requires one visible GPU unless used for a local shard.")
    if block_size <= 0 or num_warps <= 0:
        raise ValueError("block_size and num_warps must be positive.")

    domain_shape = tuple(int(size) for size in shape)
    if min(domain_shape) <= 0:
        raise ValueError("All domain dimensions must be positive.")
    weights = np.asarray(weights)
    if weights.ndim != 2 or weights.shape[1] != lattice.q:
        raise ValueError(f"weights must have shape (out_channels, {lattice.q}); received {weights.shape}.")

    cells = int(np.prod(domain_shape))
    grid = ((cells + block_size - 1) // block_size,)
    velocities = tuple(tuple(int(value) for value in lattice.c[:, direction]) for direction in range(lattice.q))
    compute_dtype = precision_policy.compute_dtype
    out_channels = weights.shape[0]
    dim = lattice.d
    weight_rows = tuple(tuple(float(weights[channel, q]) for q in range(lattice.q)) for channel in range(out_channels))

    def build(x_needs_halo):
        def kernel(field_ref, out_ref):
            cell = pl.program_id(0) * block_size + jnp.arange(block_size)
            active = cell < cells
            remainder = jnp.minimum(cell, cells - 1)
            coordinates = []
            for axis in range(dim):
                stride = int(np.prod(domain_shape[axis + 1 :]))
                coordinate = remainder // stride
                remainder = remainder - coordinate * stride
                coordinates.append(coordinate)

            totals = [jnp.zeros((block_size,), dtype=compute_dtype) for _ in range(out_channels)]
            for q_index, velocity in enumerate(velocities):
                source_coordinates = []
                for axis, (coordinate, component, size) in enumerate(zip(coordinates, velocity, domain_shape, strict=True)):
                    if axis == 0 and x_needs_halo:
                        # field is padded by 1 on each side of X: local index x maps to
                        # padded index x+1, so the neighbor at x-component sits at
                        # (x+1) - component, always in [0, size+2) - no modulo needed.
                        source_coordinates.append(coordinate + 1 - component)
                    else:
                        source_coordinates.append((coordinate - component) % size)
                value = pltriton.load(field_ref.at[tuple(source_coordinates)], mask=active, other=0.0).astype(compute_dtype)
                for channel in range(out_channels):
                    w = weight_rows[channel][q_index]
                    if w == 0.0:
                        continue
                    totals[channel] = totals[channel] + w * value

            for channel in range(out_channels):
                pltriton.store(out_ref.at[(*coordinates, channel)], totals[channel].astype(compute_dtype), mask=active)

        return jax.jit(
            pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((*domain_shape, out_channels), compute_dtype),
                grid=grid,
                compiler_params=pltriton.CompilerParams(num_warps=num_warps),
                name=f"{lattice.name.lower()}_neighbor_stencil",
            )
        )

    local_call = build(False)
    halo_call = build(True)

    def run(field, x_needs_halo):
        flat_field = field[..., 0]
        call = halo_call if x_needs_halo else local_call
        return call(flat_field).astype(compute_dtype)

    return run
