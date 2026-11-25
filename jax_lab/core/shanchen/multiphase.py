"""
Definition of Multiphase class for simulating a multiphase flow.
"""

import logging
import operator
import time
import warnings

# System libraries
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from jax import jit, shard_map
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import PartitionSpec

# Third-party libraries
from jax.tree import map as tree_map
from jax.tree import reduce

from ..base import (
    Components,
    LBMBase,
    PopulationState,
    advance_quiet_steps,
    quiet_steps,
    _lattice_dot,
    _lattice_sum,
    component_mode,
    lattice_density_momentum,
    matvec_components,
    stack_components,
)

# User-defined libraries
from ..boundary_conditions import (
    BounceBack,
    BounceBackHalfway,
    BounceBackMoving,
    InterpolatedBounceBackBouzidi,
    InterpolatedBounceBackDifferentiable,
    link_fullway_walls,
)
from ..lattice import LatticeD2Q9, LatticeD3Q19, LatticeD3Q27
from ..utils import colored, downsample_field

logger = logging.getLogger(__name__)

# This significantly reduces the performance. Use if necessary
# jax.config.update("jax_debug_nans", True)


class Multiphase(LBMBase):
    """
    Multiphase model based on the Shan-Chen method.

    The user supplies an equation of state (EOS). Pressure is evaluated from density and temperature before the effective mass. Both single-component
    multiphase and multicomponent multiphase systems are supported.

    The interaction force is F_k = -psi_k * sum_k' g_kk' * grad(psi_k'), where the gradient is evaluated with a fourth-order
    isotropic explicit finite difference (EFM) along every lattice direction (Refs. 3, 4), and the effective mass is evaluated in
    the decoupled mesh space, psi = sqrt(2 * (lambda_^2 * p_EOS - cs2 * rho) / g_kk).

    Parameters
    ----------
    lambda_ (float or list of float): Scale factor between the computational mesh step and the momentum-space lattice step
        (one per component). Smaller values widen the interface (more nodes across it), improving stability at large density
        ratios. Equivalent to the old modified-pressure coefficient through k = lambda_**2.

    g_kkprime (numpy.ndarray): Symmetric component-interaction matrix with shape (n_components, n_components).

    wetting_formulation (str or None, optional): Contact-angle scheme. Select "geometric" or
    "improved_virtual_density" when a boundary condition defines theta. Defaults to None.

    geometric_preprocessing_backend (str, optional): "auto" uses GPU preprocessing when available, "gpu"
        requires it, and "cpu" keeps the NumPy implementation. Used only by geometric wetting. Defaults to "auto".

    geometric_preprocessing_batch_size (int, optional): Rays processed per GPU at once. Used only by geometric
        wetting. Defaults to 65536.

    geometric_interpolation (str, optional): Interpolation of the density at the characteristic sample points of geometric
        wetting: "cubic" (Catmull-Rom along the grid line (2D) or plane (3D) that holds the sample, falling back to "linear"
        where its stencil is not all fluid) or "linear" (bi/trilinear). Linear interpolation across the steep liquid-vapor
        interface biases the sampled extremum and so the contact angle (several degrees between about 95 and 130 degrees);
        cubic removes most of it. Defaults to "cubic".

    Geometric wetting corrects the full-way bounce-back populations at its wall nodes by the bounce-back defect measured at
    their characteristic sample (see _create_geometric_wall_correction_data): without it, the kinetic error of bounce-back
    at the contact line, which depends on the relaxation rates, shifts the contact angle by several degrees.

    References
    ----------
    1. Shan, Xiaowen, and Hudong Chen. “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.”
        Physical Review E 47, no. 3 (March 1, 1993): 1815-19. https://doi.org/10.1103/PhysRevE.47.1815.

    2. Yuan, Peng, and Laura Schaefer. “Equations of State in a Lattice Boltzmann Model.”
        Physics of Fluids 18, no. 4 (April 3, 2006): 042101. https://doi.org/10.1063/1.2187070.

    3. Qin, Zhangrong, Jianfei Zhu, Wenbo Chen, Chengsheng Li, and Binghai Wen. "An effective pseudo-potential lattice Boltzmann
        model with extremely large density ratio and adjustable surface tension." Physics of Fluids 34, 113328 (2022).
        https://doi.org/10.1063/5.0123727.

    4. Yang, Zhichao, and Zhangrong Qin. "A three-dimensional pseudo-potential multiphase model with super-large density ratio
        and adjustable surface tension." Computers and Fluids 278, 106318 (2024). https://doi.org/10.1016/j.compfluid.2024.106318.

    Notes
    -----
    1. Boundary conditions are handled separately for each component. For example, define a wall condition once per component in a two-component system.
    2. Pytrees contain one leaf per component in the order defined by ``initialize_macroscopic_fields``.
    3. Component-specific lists and arrays must use the same ordering.
    """

    # Interaction force of the current time step, already computed by a coupled solver (MultiphaseThermal.step) from the
    # same populations and temperature; the fused collisions use it instead of computing it again. Only set while that
    # solver traces the fluid step inline (see MultiphaseThermal._shares_fluid_force); None otherwise.
    _shared_force = None
    # List a coupled solver (MultiphaseThermal.step) sets while tracing the fluid step inline: the fused collisions append the
    # density and force-corrected velocity they compute anyway, so the coupled solver does not read the populations again
    _export_macroscopic = None
    # List the step sets while tracing the collision when the geometric-wetting bounce-back correction is active: the fused
    # collisions append the density and physical momentum (see _export_force_corrected_fields)
    _wall_correction_export = None
    # Same, for the characteristic samples apply_contact_angle chooses (geometric wetting)
    _wetting_choice_export = None

    def __init__(self, **kwargs):
        self.n_components = kwargs.get("n_components")
        super().__init__(**kwargs)
        if "k" in kwargs or "A" in kwargs:
            raise ValueError("'k' and 'A' were replaced by the single scale factor 'lambda_' (k = lambda_**2); the Zhang-Chen force term is removed.")
        self.lambda_ = kwargs.get("lambda_")
        self.eos = kwargs.get("EOS", None)
        self.g_kkprime = kwargs.get("g_kkprime")  # Fluid-fluid interaction strength
        self.body_force = kwargs.get("body_force", None)
        self.wetting_formulation = kwargs.get("wetting_formulation")
        self.geometric_interpolation = kwargs.get("geometric_interpolation", "cubic")
        if self.geometric_interpolation not in ("cubic", "linear"):
            raise ValueError("geometric_interpolation must be 'cubic' or 'linear'.")
        self._uses_default_compute_force = type(self).compute_force is Multiphase.compute_force
        self._uses_default_macroscopic_velocity = type(self).macroscopic_velocity is Multiphase.macroscopic_velocity

        self._has_wetting_bc = tuple(
            any(self._is_wetting_boundary_condition(bc) and bc.theta is not None for bc in component_bcs) for component_bcs in self.BCs
        )
        if self.wetting_formulation is None and any(self._has_wetting_bc):
            raise ValueError(
                "A wetting_formulation must be selected when a boundary condition defines theta. "
                "Supported schemes: geometric and improved_virtual_density."
            )

        has_wetting_bc = any(self._has_wetting_bc)
        uses_geometric_wetting = self.wetting_formulation == "geometric" and has_wetting_bc
        uses_improved_wetting = self.wetting_formulation == "improved_virtual_density" and has_wetting_bc

        if uses_geometric_wetting:
            self.computed_nearest_next_nearest_nbr = False
            self.geometric_preprocessing_backend = kwargs.get("geometric_preprocessing_backend", "auto")
            if self.geometric_preprocessing_backend not in ("auto", "cpu", "gpu"):
                raise ValueError("geometric_preprocessing_backend must be 'auto', 'cpu', or 'gpu'.")
            self.geometric_preprocessing_batch_size = int(kwargs.get("geometric_preprocessing_batch_size", 65536))
            if self.geometric_preprocessing_batch_size <= 0:
                raise ValueError("geometric_preprocessing_batch_size must be positive.")

        self.G_ff = self.compute_ff_greens_function()
        self._g_host = np.asarray(self.g_kkprime, dtype=np.float64)
        self.g_kkprime = jnp.array(self.g_kkprime, dtype=self.precision_policy.compute_dtype)
        self._psi_interactions = tuple(
            tuple(self._g_host[output, source] != 0.0 for source in range(self.n_components)) for output in range(self.n_components)
        )
        self._psi_stencil_components = tuple(
            any(self._psi_interactions[output][source] for output in range(self.n_components)) for source in range(self.n_components)
        )
        self._uses_psi_force = any(self._psi_stencil_components)
        self._efm_offsets = self._build_efm_gradient_offsets()
        if self.nx // self.n_devices < 2:
            raise ValueError("The fourth-order EFM gradient needs at least 2 x-planes per device (two-cell halo).")

        P = PartitionSpec
        scalar_spec = P("x", None, None) if self.dim == 2 else P("x", None, None, None)
        aux_spec = P("x", None, None)
        self.local_improved_wetting = None
        if uses_improved_wetting and self.n_devices > 1:
            self.local_improved_wetting = jit(
                shard_map(
                    self.local_improved_wetting_m,
                    mesh=self.mesh,
                    in_specs=(scalar_spec, scalar_spec, aux_spec, aux_spec, aux_spec, aux_spec),
                    out_specs=scalar_spec,
                    check_vma=False,
                )
            )
        # Global (min, max) of a density field with one cross-device reduction instead of two (see _global_min_max_m)
        self.global_min_max = None
        if uses_improved_wetting and self.n_devices > 1:
            self.global_min_max = jit(shard_map(self._global_min_max_m, mesh=self.mesh, in_specs=scalar_spec, out_specs=P(), check_vma=False))
        self.local_geometric_wetting = None
        G_ff_host = np.array(self.G_ff)
        # scalar_neighbor_sum: G_ff-weighted scalar neighbor sum, used by the wetting/average-density
        # denominator (one-x-halo _neighbor_stencil_m). scalar_gradient: fourth-order EFM gradient used by the
        # fluid-fluid force (two-x-halo _gradient_stencil_m).
        self.scalar_neighbor_sum = (
            jit(
                shard_map(
                    partial(self._neighbor_stencil_m, weights=G_ff_host),
                    mesh=self.mesh,
                    in_specs=scalar_spec,
                    out_specs=scalar_spec,
                    check_vma=False,
                )
            )
            if uses_improved_wetting
            else None
        )
        self.scalar_gradient = jit(
            shard_map(
                self._gradient_stencil_m,
                mesh=self.mesh,
                in_specs=scalar_spec,
                out_specs=scalar_spec,
                check_vma=False,
            )
        )

        self.solid_mask_streamed = None
        self.average_density_denominator = None
        self.average_density_second_layer = None
        if self.scalar_neighbor_sum is not None:
            self.solid_mask_streamed = self.get_solid_mask_streamed()
            self.average_density_denominator = [
                self.scalar_neighbor_sum(1 - mask) if has_wetting_bc else None
                for mask, has_wetting_bc in zip(self.solid_mask_streamed, self._has_wetting_bc, strict=True)
            ]
            # Weighted count of first-layer neighbors (nodes with a fluid neighbor), for the second-layer average
            # in compute_average_density
            self.average_density_second_layer = [
                self.scalar_neighbor_sum((denominator > 0).astype(denominator.dtype)) if denominator is not None else None
                for denominator in self.average_density_denominator
            ]
        if self.average_density_denominator is not None:
            for denominator in self.average_density_denominator + self.average_density_second_layer:
                if denominator is not None:
                    denominator.block_until_ready()
        self.geometric_force_mask = None
        self.geometric_wetting_data, self.geometric_fluid_mask = None, None
        if uses_geometric_wetting:
            self._geometric_wetting_halos = []
            self._geometric_wetting_orders = []  # per component and boundary, the kernel's row order per shard
            self.geometric_wetting_data, self.geometric_fluid_mask = self._create_geometric_wetting_data()
            self.local_geometric_wetting = self._build_local_geometric_wetting_kernels(scalar_spec)
        self.geometric_wall_correction_data = self._create_geometric_wall_correction_data() if uses_geometric_wetting else [None] * self.n_components
        self._geometric_wall_correction_kernels = self._build_geometric_wall_correction_kernels()
        self.local_improved_wetting_data = (
            self._make_local_improved_wetting_data()
            if uses_improved_wetting and self.n_devices > 1
            else [[None] * len(component_bcs) for component_bcs in self.BCs]
        )

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if not isinstance(value, list):
            raise ValueError("omega must be a list")
        self._omega = value

    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, value):
        if value is None:
            raise ValueError("Number of components cannot be None")
        if value <= 0:
            raise ValueError("Number of components must be positive")
        if not isinstance(value, int):
            raise ValueError("Number of components must be an integer")
        self._n_components = value

    @property
    def lambda_(self):
        return self._lambda

    @lambda_.setter
    def lambda_(self, value):
        if value is None:
            raise ValueError("Scale factor lambda_ must be provided")
        values = [value] if isinstance(value, (int, float)) else value
        if not isinstance(values, (list, tuple)) or len(values) != self.n_components:
            raise ValueError("lambda_ must be a number (single component) or a list with one value per component")
        if any(not isinstance(v, (int, float)) or v < 0 for v in values):
            raise ValueError("lambda_ values must be non-negative numbers")
        self._lambda = [float(v) for v in values]

    @property
    def body_force(self):
        return self._body_force

    @body_force.setter
    def body_force(self, value):
        if value is None:
            self._body_force = None
        if isinstance(value, list):
            self._body_force = jnp.array(np.array(value), dtype=self.precision_policy.compute_dtype)
        if isinstance(value, np.ndarray):
            self._body_force = jnp.array(value, dtype=self.precision_policy.compute_dtype)

    @property
    def g_kkprime(self):
        return self._g_kkprime

    @g_kkprime.setter
    def g_kkprime(self, value):
        if not isinstance(value, np.ndarray) and not isinstance(value, jax.numpy.ndarray):
            raise ValueError("g_kkprime must be a numpy array or jax.numpy.ndarray")
        if value.shape != (self.n_components, self.n_components):
            raise ValueError("g_kkprime must be a matrix of size n_components x n_components")
        if not np.allclose(value, np.transpose(value), atol=1e-6):
            raise ValueError("g_kkprime must be a symmetric matrix")
        self._g_kkprime = np.array(value)

    @property
    def wetting_formulation(self):
        return self._wetting_formulation

    @wetting_formulation.setter
    def wetting_formulation(self, value):
        if value is None or value in ["geometric", "improved_virtual_density"]:
            self._wetting_formulation = value
        else:
            raise ValueError("Invalid wetting scheme type. Supported schemes: None, geometric, and improved_virtual_density.")

    def _is_wetting_boundary_condition(self, bc):
        """
        Check whether a boundary condition can carry wetting parameters.

        Parameters
        ----------
        bc (BoundaryCondition): Boundary condition object.

        Returns
        -------
        (bool): True if the boundary condition supports contact angle data.
        """
        return isinstance(bc, (BounceBackHalfway, BounceBack, BounceBackMoving, InterpolatedBounceBackBouzidi, InterpolatedBounceBackDifferentiable))

    def _get_solid_indices(self, bc):
        """
        Return the solid-node indices of a boundary condition.

        BounceBackHalfway (and subclasses) shift bc.indices to the adjacent fluid nodes during configure and keep the
        original solid nodes in bc.solid_indices. Wetting data must be built on the solid nodes.

        Parameters
        ----------
        bc (BoundaryCondition): Boundary condition object.

        Returns
        -------
        (tuple): Solid-node index tuple.
        """
        return getattr(bc, "solid_indices", bc.indices)

    def _create_component_solid_mask(self, BC):
        """
        Create a solid mask for computing wall normals in geometric wetting.

        Parameters
        ----------
        BC (list): Boundary conditions for one component.

        Returns
        -------
        solid_mask (numpy.ndarray): Boolean mask with True on boundary nodes.
        """
        shape = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        solid_mask = np.zeros(shape, dtype=bool)
        for bc in BC:
            if self._is_wetting_boundary_condition(bc):
                indices = np.array(self._get_solid_indices(bc), dtype=np.int64)
                if self.dim == 2:
                    bounds = [(self.nx, indices[0]), (self.ny, indices[1])]
                else:
                    bounds = [(self.nx, indices[0]), (self.ny, indices[1]), (self.nz, indices[2])]
                valid = np.ones((indices.shape[1],), dtype=bool)
                for size, index in bounds:
                    valid &= (index >= 0) & (index < size)
                solid_mask[tuple(indices[:, valid])] = True
        return solid_mask

    def _create_geometric_force_mask(self, fluid_mask):
        """Disable geometric force where no opposite D3Q lattice-neighbor pair resolves a pore passage."""
        if self.dim != 3:
            return fluid_mask
        resolved = np.zeros_like(fluid_mask)
        directions = np.asarray(self.lattice.c, dtype=np.int32).T
        for direction in directions:
            nonzero = np.flatnonzero(direction)
            if len(nonzero) == 0 or direction[nonzero[0]] < 0:
                continue
            positive = np.roll(fluid_mask, tuple(direction), axis=tuple(range(self.dim)))
            negative = np.roll(fluid_mask, tuple(-direction), axis=tuple(range(self.dim)))
            resolved |= positive & negative
        return fluid_mask & resolved

    def _compute_geometric_normals(self, bc, solid_mask, indices=None, source_rows=None):
        """
        Compute normals for geometric wetting using boundary data and solid mask.

        Parameters
        ----------
        bc (BoundaryCondition): Boundary condition with wettability data.

        solid_mask (numpy.ndarray): Boolean mask with True on boundary nodes.

        indices (numpy.ndarray, optional): Boundary-node subset. Defaults to all solid indices belonging to ``bc``.

        source_rows (numpy.ndarray, optional): Rows of ``bc`` corresponding to ``indices``, used when boundary-provided
            normals are available.

        Returns
        -------
        normals (numpy.ndarray): Unit normals pointing from wall nodes toward fluid nodes.
        """
        all_indices = np.array(self._get_solid_indices(bc), dtype=np.int64).T
        if indices is None:
            indices = all_indices
            source_rows = np.arange(len(all_indices), dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)
        normals = np.zeros((indices.shape[0], self.dim), dtype=np.float64)

        # bc.normals rows correspond to bc.indices; for halfway bounce-back those are the shifted
        # fluid nodes, not the solid nodes used here, so fall back to the neighbor-based normals.
        if bc.is_solid and hasattr(bc, "normals") and not hasattr(bc, "solid_indices"):
            bc_normals = np.asarray(bc.normals, dtype=np.float64)
            if bc_normals.shape == (len(all_indices), self.dim):
                if source_rows is not None:
                    bc_normals = bc_normals[source_rows]
                normal_norm = np.linalg.norm(bc_normals, axis=1, keepdims=True)
                normals = np.divide(bc_normals, normal_norm, out=normals, where=normal_norm > 1e-12)

        c = np.array(self.lattice.c).T
        c = c[np.linalg.norm(c, axis=1) > 0]
        missing_normal = np.linalg.norm(normals, axis=1) <= 1e-12
        missing_rows = np.flatnonzero(missing_normal)
        accumulated = np.zeros((len(missing_rows), self.dim), dtype=np.float64)
        domain_shape = np.asarray(solid_mask.shape, dtype=np.int64)
        # Keep lattice-direction accumulation order unchanged, but evaluate every boundary node together.
        for direction in c:
            neighbors = indices[missing_rows] + direction
            in_bounds = np.all((neighbors >= 0) & (neighbors < domain_shape), axis=1)
            in_bounds_rows = np.flatnonzero(in_bounds)
            if len(in_bounds_rows) == 0:
                continue
            fluid = ~solid_mask[tuple(neighbors[in_bounds_rows].T)]
            accumulated[in_bounds_rows[fluid]] += direction / np.linalg.norm(direction)
        normal_norm = np.linalg.norm(accumulated, axis=1, keepdims=True)
        normals[missing_rows] = np.divide(accumulated, normal_norm, out=np.zeros_like(accumulated), where=normal_norm > 1e-12)

        return normals

    def _one_sided_face_normals(self, indices, solid_mask):
        """
        Normals of wall nodes whose periodic neighbor normal cancels (see _create_geometric_wetting_data): a one-layer wall
        on a domain face, with fluid on both sides along an axis through the periodic face. Along such an axis only the
        side inside the domain counts; the other axes stay periodic.

        Parameters
        ----------
        indices (numpy.ndarray): Wall nodes in domain coordinates, shape (n, dim).

        solid_mask (numpy.ndarray): Boolean mask of the domain with True on solid nodes.

        Returns
        -------
        (numpy.ndarray): Unit normals (zero where none is defined), shape (n, dim).
        """
        shape = np.asarray(solid_mask.shape, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        fluid = lambda nodes: ~solid_mask[tuple(np.mod(nodes, shape).T)]
        unit = np.eye(self.dim, dtype=np.int64)
        two_sided = np.stack([fluid(indices + unit[axis]) & fluid(indices - unit[axis]) for axis in range(self.dim)], axis=1)
        accumulated = np.zeros(indices.shape, dtype=np.float64)
        for direction in np.array(self.lattice.c, dtype=np.int64).T:
            if not direction.any():
                continue
            neighbors = indices + direction
            crosses = (neighbors < 0) | (neighbors >= shape)  # (n, dim): through a periodic face along that axis
            counted = fluid(neighbors) & ~np.any(crosses & two_sided, axis=1)
            accumulated[counted] += direction / np.linalg.norm(direction)
        norm = np.linalg.norm(accumulated, axis=1, keepdims=True)
        return np.divide(accumulated, norm, out=np.zeros_like(accumulated), where=norm > 1e-12)

    def _solid_fluid_interface_mask(self, indices, solid_mask):
        """
        Identify solid boundary nodes that touch at least one fluid node.

        Parameters
        ----------
        indices (numpy.ndarray): Boundary node coordinates with shape (n, dim).

        solid_mask (numpy.ndarray): Boolean mask with True on boundary nodes.

        Returns
        -------
        interface (numpy.ndarray): Boolean mask with True for solid-fluid interface nodes.
        """
        directions = np.array(self.lattice.c, dtype=np.int64).T
        directions = directions[np.linalg.norm(directions, axis=1) > 0]
        interface = np.zeros((indices.shape[0],), dtype=bool)
        domain_shape = np.asarray(solid_mask.shape, dtype=np.int64)
        # Preserve direction order and early acceptance while replacing the per-node Python loop with bulk indexing.
        for direction in directions:
            unresolved_rows = np.flatnonzero(~interface)
            if len(unresolved_rows) == 0:
                break
            neighbors = indices[unresolved_rows] + direction
            in_bounds = np.all((neighbors >= 0) & (neighbors < domain_shape), axis=1)
            in_bounds_rows = np.flatnonzero(in_bounds)
            if len(in_bounds_rows) == 0:
                continue
            fluid = ~solid_mask[tuple(neighbors[in_bounds_rows].T)]
            interface[unresolved_rows[in_bounds_rows[fluid]]] = True

        return interface

    def _stencil_reach_wall_rows(self, wall_indices, member_mask, solid_mask):
        """
        Solid nodes behind the wetted wall layer that the fluid-fluid force stencil still reaches, with the wall row in front of each.

        The fourth-order EFM psi gradient reads psi(x + 2 c_i) (see _build_efm_gradient_offsets), so the first fluid layer also sees the
        solid layer behind the wall nodes. Those nodes must carry a wetting density too: left at their bounce-back (initial) density they
        add a spurious fluid-wall interaction that drives the contact angle far from the prescribed one. A node s is reached when, for a
        lattice direction c_i, w = s + c_i is a wall node and s + 2 c_i is fluid; its wall row is that of w (the first such direction).

        Parameters
        ----------
        wall_indices (numpy.ndarray): Wetted wall-layer nodes (solid nodes with a fluid neighbor), shape (n, dim).

        member_mask (numpy.ndarray): Boolean mask with True on the solid nodes of this boundary condition (candidates).

        solid_mask (numpy.ndarray): Boolean mask with True on solid nodes.

        Returns
        -------
        inner_indices (numpy.ndarray): Reached solid nodes behind the wall layer, shape (m, dim).

        source_rows (numpy.ndarray): Row of wall_indices in front of each reached node, shape (m,).
        """
        reach = max(max(abs(v) for v in offset) for offset, _ in self._efm_offsets)
        if reach < 2 or len(wall_indices) == 0:
            return np.zeros((0, self.dim), dtype=wall_indices.dtype), np.zeros((0,), dtype=np.int64)
        shape = np.asarray(solid_mask.shape, dtype=np.int64)
        taken = np.zeros(solid_mask.shape, dtype=bool)
        taken[tuple(wall_indices.T)] = True
        inner, sources = [], []
        directions = np.array(self.lattice.c, dtype=np.int64).T
        for direction in directions[np.any(directions != 0, axis=1)]:
            behind = wall_indices - direction
            front = wall_indices + direction
            ok = np.all((behind >= 0) & (behind < shape), axis=1) & np.all((front >= 0) & (front < shape), axis=1)
            rows = np.flatnonzero(ok)
            rows = rows[member_mask[tuple(behind[rows].T)] & ~solid_mask[tuple(front[rows].T)] & ~taken[tuple(behind[rows].T)]]
            # One wall row per reached node (the first direction, then the first wall row)
            _, first = np.unique(np.ravel_multi_index(tuple(behind[rows].T), solid_mask.shape), return_index=True)
            rows = rows[np.sort(first)]
            taken[tuple(behind[rows].T)] = True
            inner.append(behind[rows])
            sources.append(rows)
        return np.concatenate(inner).astype(wall_indices.dtype), np.concatenate(sources)

    def _interpolation_points(self, point):
        """The sample points of interpolation data (_build_interpolation_data): the weighted mean of the stencil nodes (exact for
        multilinear and Catmull-Rom weights, which reproduce linear functions)."""
        indices, weights = point
        return np.sum(np.asarray(weights, dtype=np.float64)[..., None] * np.asarray(indices, dtype=np.float64), axis=1)

    def _append_wall_rows(self, data, inner_indices, source_rows, solid_mask):
        """
        Append geometric wetting rows for the solid nodes behind the wall layer (see _stencil_reach_wall_rows), and data["source"]:
        for every row the wall row whose characteristics it uses (itself for the wall rows).

        Each such node s, behind the wall node w = s + c_i, samples along the characteristics of w extended by one layer: a sample point
        P of w becomes s + 2 (P - w), the same ray direction at twice the length, which on a flat wall is the first fluid layer at twice the
        tangential distance |cot(theta)| (the characteristic through s). Where that point is outside the domain or its interpolation
        stencil is not all fluid (curved or thin walls), s uses the samples of w (the wall density extended one layer into the solid).
        """
        n_wall = len(data["indices"][0])
        data = dict(data, source=np.arange(n_wall))
        if len(source_rows) == 0:
            return data
        wall = np.stack(data["indices"], axis=-1).astype(np.float64)[source_rows]
        inner = np.asarray(inner_indices, dtype=np.float64)
        shape = np.asarray(solid_mask.shape, dtype=np.float64)

        def extend(point):
            old = tuple(np.asarray(array)[source_rows] for array in point)
            sample = inner + 2.0 * (self._interpolation_points(point)[source_rows] - wall)
            rounded = np.round(sample)
            sample = np.where(np.isclose(sample, rounded, atol=1e-5), rounded, sample)
            valid = np.all((sample >= 0) & (sample <= shape - 1), axis=1)
            valid[valid] = self._uses_only_fluid_nodes_batch(sample[valid], solid_mask)
            new = self._build_interpolation_data(np.where(valid[:, None], sample, wall), solid_mask)
            row_mask = lambda array: valid.reshape((-1,) + (1,) * (np.ndim(array) - 1))
            return tuple(
                np.concatenate([np.asarray(array), np.where(row_mask(n), n, o).astype(np.asarray(array).dtype)])
                for array, n, o in zip(point, new, old, strict=True)
            )

        data = dict(data)
        data["indices"] = tuple(
            np.concatenate([index, extra.astype(index.dtype)]) for index, extra in zip(data["indices"], np.asarray(inner_indices).T, strict=True)
        )
        data["theta"] = np.concatenate([data["theta"], np.asarray(data["theta"])[source_rows]])
        data["source"] = np.concatenate([data["source"], np.asarray(source_rows)])  # the wall row each row extends
        for key in ("point_1", "point_2"):
            if key in data:
                data[key] = extend(data[key])
        if "points" in data:
            data["points"] = tuple(extend(point) for point in data["points"])
        return data

    def _first_mesh_intersection(self, indices, directions):
        """
        Find first mesh-line intersections from boundary nodes.

        Parameters
        ----------
        indices (numpy.ndarray): Boundary node coordinates with shape (n, dim).

        directions (numpy.ndarray): Characteristic directions with shape (n, dim).

        Returns
        -------
        points (numpy.ndarray): First intersection points with mesh lines.
        """
        eps = 1e-12
        abs_dir = np.abs(directions)
        t_axis = np.divide(1.0, abs_dir, out=np.full_like(abs_dir, np.inf, dtype=np.float64), where=abs_dir > eps)
        t = np.min(t_axis, axis=1)
        t = np.where(np.isfinite(t), t, 1.0)
        points = indices + t[:, None] * directions
        rounded = np.round(points)
        return np.where(np.isclose(points, rounded, atol=eps), rounded, points)

    def _uses_only_fluid_nodes(self, point, solid_mask):
        """
        Check if a multilinear interpolation stencil contains only fluid nodes.

        Parameters
        ----------
        point (numpy.ndarray): Off-lattice or on-lattice interpolation point.

        solid_mask (numpy.ndarray): Boolean mask with True on boundary nodes.

        Returns
        -------
        (bool): True if every interpolation node with non-zero weight is inside the domain and fluid.
        """
        eps = 1e-12
        floor_point = np.floor(point)
        lower = floor_point.astype(np.int64)
        upper = lower + 1
        frac = point - floor_point
        stencil = []
        for corner in np.ndindex(*(2 for _ in range(self.dim))):
            index = np.where(corner, upper, lower)
            weight = np.prod(np.where(corner, frac, 1.0 - frac))
            stencil.append((index, weight))

        for index, weight in stencil:
            if weight <= eps:
                continue
            in_bounds = all(0 <= index[axis] < solid_mask.shape[axis] for axis in range(self.dim))
            if not in_bounds or solid_mask[tuple(index)]:
                return False
        return True

    def _uses_only_fluid_nodes_batch(self, points, solid_mask):
        """Vectorized equivalent of ``_uses_only_fluid_nodes`` for a batch of interpolation points."""
        eps = 1e-12
        floor_points = np.floor(points)
        lower = floor_points.astype(np.int64)
        upper = lower + 1
        frac = points - floor_points
        domain_shape = np.asarray(solid_mask.shape, dtype=np.int64)
        valid = np.ones((len(points),), dtype=bool)
        for corner in np.ndindex(*(2 for _ in range(self.dim))):
            corner = np.asarray(corner, dtype=bool)
            interpolation_indices = np.where(corner, upper, lower)
            weights = np.prod(np.where(corner, frac, 1.0 - frac), axis=1)
            active_rows = np.flatnonzero((weights > eps) & valid)
            if len(active_rows) == 0:
                continue
            active_indices = interpolation_indices[active_rows]
            in_bounds = np.all((active_indices >= 0) & (active_indices < domain_shape), axis=1)
            active_valid = np.zeros((len(active_rows),), dtype=bool)
            in_bounds_rows = np.flatnonzero(in_bounds)
            if len(in_bounds_rows) > 0:
                active_valid[in_bounds_rows] = ~solid_mask[tuple(active_indices[in_bounds_rows].T)]
            valid[active_rows] &= active_valid
        return valid

    def _first_fluid_mesh_intersection(self, indices, directions, solid_mask, return_valid=False, max_intersections=None):
        """
        Find first mesh-line intersections with fluid-only interpolation stencils.

        Parameters
        ----------
        indices (numpy.ndarray): Boundary node coordinates with shape (n, dim).

        directions (numpy.ndarray): Characteristic directions with shape (n, dim).

        solid_mask (numpy.ndarray): Boolean mask with True on boundary nodes.

        return_valid (bool, optional): If True, return a boolean mask for nodes where a fluid-only stencil was found.

        max_intersections (int, optional): Maximum number of candidate mesh intersections to test for each node.

        Returns
        -------
        points (numpy.ndarray): First fluid-side mesh intersection points. If return_valid is True, returns
        (points, valid), where valid is a boolean mask for accepted intersections.
        """
        eps = 1e-12
        indices = np.asarray(indices)
        directions = np.asarray(directions)
        points = self._first_mesh_intersection(indices, directions)
        valid = np.zeros((indices.shape[0],), dtype=bool)
        max_steps = self.nx + self.ny if self.dim == 2 else self.nx + self.ny + self.nz
        if len(indices) == 0:
            return (points, valid) if return_valid else points

        # Bound temporary host memory while vectorizing rays. Each batch holds at most about two million float64
        # candidates, independent of domain size or porous-interface area.
        candidate_slots = self.dim * max_steps
        batch_size = max(1, min(len(indices), 2_000_000 // candidate_slots))
        steps = np.arange(1, max_steps + 1, dtype=np.float64)
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            batch_indices = indices[start:stop]
            batch_directions = directions[start:stop]
            absolute_directions = np.abs(batch_directions)
            candidates = np.full((len(batch_indices), self.dim, max_steps), np.inf, dtype=np.float64)
            np.divide(
                steps[None, None, :],
                absolute_directions[..., None],
                out=candidates,
                where=absolute_directions[..., None] > eps,
            )
            candidates = np.round(np.sort(candidates.reshape(len(batch_indices), -1), axis=1), decimals=12)
            unresolved = np.any(absolute_directions > eps, axis=1)
            candidate_counts = np.zeros((len(batch_indices),), dtype=np.int32)
            batch_points = points[start:stop]
            batch_valid = valid[start:stop]

            for candidate_index in range(candidates.shape[1]):
                values = candidates[:, candidate_index]
                unique = np.isfinite(values)
                if candidate_index > 0:
                    unique &= values != candidates[:, candidate_index - 1]
                eligible = unresolved & unique
                if max_intersections is not None:
                    eligible &= candidate_counts < max_intersections
                eligible_rows = np.flatnonzero(eligible)
                if len(eligible_rows) > 0:
                    candidate_points = batch_indices[eligible_rows] + values[eligible_rows, None] * batch_directions[eligible_rows]
                    rounded = np.round(candidate_points)
                    candidate_points = np.where(np.isclose(candidate_points, rounded, atol=eps), rounded, candidate_points)
                    accepted = self._uses_only_fluid_nodes_batch(candidate_points, solid_mask)
                    accepted_rows = eligible_rows[accepted]
                    batch_points[accepted_rows] = candidate_points[accepted]
                    batch_valid[accepted_rows] = True
                    unresolved[accepted_rows] = False
                candidate_counts += unique
                if max_intersections is not None:
                    unresolved &= candidate_counts < max_intersections
                if not np.any(unresolved):
                    break

            points[start:stop] = batch_points
            valid[start:stop] = batch_valid
        if return_valid:
            return points, valid
        return points

    def _ray_passes_through_solid(self, indices, directions, distance, solid_mask, max_crossings=None):
        """
        Check whether characteristic rays pass through solid before reaching their sample points.

        A ray passes through solid (a grain) when one of its lattice-plane crossings before the sample point has an
        interpolation stencil with no fluid node. A crossing whose stencil is partly fluid, such as the face between a
        flat wall layer and its first fluid layer that a ray at theta < 45 deg (or > 135 deg) crosses first, does not
        count: rejecting those would discard every cone on a flat voxel wall.

        Parameters
        ----------
        indices (numpy.ndarray): Wall node coordinates with shape (n, dim), float.

        directions (numpy.ndarray): Unit ray directions with shape (n, dim).

        distance (numpy.ndarray): Distance along each ray to its sample point, shape (n,).

        solid_mask (numpy.ndarray): Boolean mask with True on solid nodes.

        max_crossings (int, optional): Crossings checked per axis. Defaults to None (every crossing before the sample).

        Returns
        -------
        (numpy.ndarray): Boolean mask, True where the ray passes through solid.
        """
        eps = 1e-9
        n = len(indices)
        through = np.zeros(n, dtype=bool)
        shape = np.asarray(solid_mask.shape, dtype=np.int64)
        absolute = np.abs(directions)
        for axis in range(self.dim):
            k = 0
            while max_crossings is None or k < max_crossings:
                k += 1
                t = np.divide(k, absolute[:, axis], out=np.full(n, np.inf), where=absolute[:, axis] > eps)
                rows = np.flatnonzero(t < distance - eps)
                if len(rows) == 0:
                    break
                points = indices[rows] + t[rows, None] * directions[rows]
                rounded = np.round(points)
                points = np.where(np.isclose(points, rounded, atol=1e-12), rounded, points)
                lower = np.floor(points).astype(np.int64)
                frac = points - lower
                any_fluid = np.zeros(len(rows), dtype=bool)
                for corner in np.ndindex(*(2 for _ in range(self.dim))):
                    corner = np.asarray(corner, dtype=bool)
                    index = np.where(corner, lower + 1, lower)
                    weight = np.prod(np.where(corner, frac, 1.0 - frac), axis=1)
                    inside = np.all((index >= 0) & (index < shape), axis=1)
                    fluid = np.zeros(len(rows), dtype=bool)
                    fluid[inside] = ~solid_mask[tuple(index[inside].T)]
                    any_fluid |= (weight > 1e-12) & fluid
                through[rows[~any_fluid]] = True
        return through

    def _build_interpolation_data(self, points, solid_mask):
        """
        Build interpolation data for density samples: the stencil nodes and weights of each sample point.

        With geometric_interpolation = "cubic", a point is interpolated by Catmull-Rom splines along its fractional axes (at most
        dim - 1 of them: the samples lie on a grid line in 2D or plane in 3D), 4^(dim - 1) nodes; a point whose stencil leaves the
        domain or touches solid, or with dim fractional axes, uses multilinear interpolation (zero-padded to the same size).
        With "linear", every point uses multilinear interpolation over its 2^dim cell corners.

        Parameters
        ----------
        points (numpy.ndarray): Off-lattice or on-lattice sample points, shape (n, dim).

        solid_mask (numpy.ndarray): Boolean mask with True on solid nodes.

        Returns
        -------
        data (tuple): (indices, weights): int32 node indices of shape (n, S, dim) and weights of shape (n, S).
        """
        points = np.asarray(points, dtype=np.float64)
        rounded = np.round(points)
        points = np.where(np.abs(points - rounded) < 1e-9, rounded, points)
        compute_dtype = np.dtype(self.precision_policy.compute_dtype)
        shape = np.asarray(solid_mask.shape, dtype=np.int64)
        lower = np.floor(points).astype(np.int64)
        frac = points - lower

        # Multilinear: the 2^dim cell corners (indices clipped to the domain)
        corners = np.array([[(k >> axis) & 1 for axis in range(self.dim)] for k in range(2**self.dim)], dtype=np.int64)
        linear_indices = np.clip(lower[:, None, :] + corners[None], 0, shape - 1)
        linear_weights = np.prod(np.where(corners[None] == 1, frac[:, None, :], 1.0 - frac[:, None, :]), axis=-1)
        if self.geometric_interpolation == "linear":
            return linear_indices.astype(np.int32), linear_weights.astype(compute_dtype)

        # Catmull-Rom along the fractional axes: slot j holds the j-th fractional axis of each point (or none)
        slots = self.dim - 1
        fractional = frac > 1e-9
        order = np.argsort(~fractional, axis=1, kind="stable")[:, :slots]
        slot_used = np.take_along_axis(fractional, order, axis=1)
        offsets = np.array(np.meshgrid(*([np.arange(-1, 3)] * slots), indexing="ij")).reshape(slots, -1).T  # (4^slots, slots)

        def catmull_rom(o, t):
            return np.select(
                [o == -1, o == 0, o == 1, o == 2],
                [0.5 * (-t + 2 * t**2 - t**3), 0.5 * (2 - 5 * t**2 + 3 * t**3), 0.5 * (t + 4 * t**2 - 3 * t**3), 0.5 * (-(t**2) + t**3)],
            )

        n = len(points)
        cubic_indices = np.repeat(lower[:, None, :], len(offsets), axis=1)
        cubic_weights = np.ones((n, len(offsets)))
        for slot in range(slots):
            axis = order[:, slot]
            o = offsets[None, :, slot]
            t = np.take_along_axis(frac, axis[:, None], axis=1)
            used = slot_used[:, slot : slot + 1]
            cubic_weights *= np.where(used, catmull_rom(o, t), (o == 0).astype(np.float64))
            shift = np.where(used, o, 0)
            cubic_indices[np.arange(n)[:, None], np.arange(len(offsets))[None, :], axis[:, None]] += shift

        active = np.abs(cubic_weights) > 1e-12
        inside = np.all((cubic_indices >= 0) & (cubic_indices < shape), axis=-1)
        valid = np.all(~active | inside, axis=1) & (fractional.sum(axis=1) <= slots)
        safe = np.clip(cubic_indices, 0, shape - 1)
        valid &= np.all(~active | ~solid_mask[tuple(np.moveaxis(safe, -1, 0))], axis=1)

        size = len(offsets)
        pad = size - linear_indices.shape[1]
        padded_indices = np.concatenate([linear_indices, np.repeat(linear_indices[:, :1], pad, axis=1)], axis=1)
        padded_weights = np.concatenate([linear_weights, np.zeros((n, pad))], axis=1)
        indices = np.where(valid[:, None, None], np.clip(cubic_indices, 0, shape - 1), padded_indices)
        weights = np.where(valid[:, None], cubic_weights, padded_weights)
        return indices.astype(np.int32), weights.astype(compute_dtype)

    def _build_geometric_gpu_ray_kernel(self, ray_count):
        """
        Build the device ray tracer for geometric-wetting preprocessing: for each ray (a wall node and a direction), march
        through the lattice-plane crossings and return the first crossing point whose interpolation stencil touches only
        fluid nodes, whether one was found, and whether an earlier crossing had an interpolation stencil with no fluid node
        (the ray passes through solid; the device version of _ray_passes_through_solid).

        Plain JAX: one ray is a lax.while_loop, vectorized over the rays with vmap (all rays advance together until the last
        one finishes). The arithmetic is the same float64 sequence as the CPU path (_first_fluid_mesh_intersection).

        Parameters
        ----------
        ray_count (int): Number of rays per call (the batch size); the returned function expects inputs of this length.

        Returns
        -------
        (callable): fn(indices (ray_count, 3) int32, directions (ray_count, 3) float64, solid (nx, ny, nz) bool) ->
            (points (ray_count, 3) float64, valid (ray_count,) bool, through (ray_count,) bool).
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        max_steps = nx + ny + nz
        candidate_slots = self.dim * max_steps

        def round_half_to_even(value, scale=1.0):
            scaled = value * scale
            lower = jnp.floor(scaled)
            fraction = scaled - lower
            upper = lower + 1.0
            even_lower = jnp.floor(lower * 0.5) * 2.0 == lower
            tie = jnp.where(even_lower, lower, upper)
            return jnp.where(fraction < 0.5, lower, jnp.where(fraction > 0.5, upper, tie)) / scale

        def trace(index, direction, solid):
            nx, ny, nz = solid.shape  # the (periodically padded) mask, see _create_geometric_wetting_data
            index = tuple(index[axis] for axis in range(3))
            direction = tuple(direction[axis] for axis in range(3))
            absolute = tuple(jnp.abs(value) for value in direction)
            active = tuple(value > 1e-12 for value in absolute)
            first_t = jnp.minimum(
                jnp.minimum(
                    jnp.where(active[0], 1.0 / absolute[0], jnp.inf),
                    jnp.where(active[1], 1.0 / absolute[1], jnp.inf),
                ),
                jnp.where(active[2], 1.0 / absolute[2], jnp.inf),
            )
            first_t = jnp.where(jnp.isfinite(first_t), first_t, 1.0)
            initial_point = tuple(index[axis] + first_t * direction[axis] for axis in range(3))
            initial_point = tuple(
                jnp.where(
                    jnp.isclose(value, rounded := round_half_to_even(value), atol=1e-12),
                    rounded,
                    value,
                )
                for value in initial_point
            )

            def condition(state):
                iteration, _, _, _, _, _, _, _, _, done = state
                return (iteration < candidate_slots) & ~done

            def body(state):
                iteration, k0, k1, k2, out0, out1, out2, found, through, _ = state
                t0 = jnp.where(active[0] & (k0 <= max_steps), round_half_to_even(k0 / absolute[0], 1e12), jnp.inf)
                t1 = jnp.where(active[1] & (k1 <= max_steps), round_half_to_even(k1 / absolute[1], 1e12), jnp.inf)
                t2 = jnp.where(active[2] & (k2 <= max_steps), round_half_to_even(k2 / absolute[2], 1e12), jnp.inf)
                value = jnp.minimum(jnp.minimum(t0, t1), t2)
                finite = jnp.isfinite(value)
                safe_value = jnp.where(finite, value, 0.0)
                point = tuple(index[axis] + safe_value * direction[axis] for axis in range(3))
                point = tuple(
                    jnp.where(
                        jnp.isclose(component, rounded := round_half_to_even(component), atol=1e-12),
                        rounded,
                        component,
                    )
                    for component in point
                )
                lower = tuple(jnp.floor(component).astype(jnp.int32) for component in point)
                upper = tuple(component + 1 for component in lower)
                frac = tuple(point[axis] - lower[axis] for axis in range(3))
                fluid = finite
                any_fluid = jnp.zeros((), dtype=bool)
                for corner in np.ndindex(2, 2, 2):
                    coordinates = tuple(upper[axis] if corner[axis] else lower[axis] for axis in range(3))
                    weight = (
                        (frac[0] if corner[0] else 1.0 - frac[0])
                        * (frac[1] if corner[1] else 1.0 - frac[1])
                        * (frac[2] if corner[2] else 1.0 - frac[2])
                    )
                    in_bounds = (
                        (coordinates[0] >= 0)
                        & (coordinates[0] < nx)
                        & (coordinates[1] >= 0)
                        & (coordinates[1] < ny)
                        & (coordinates[2] >= 0)
                        & (coordinates[2] < nz)
                    )
                    safe = (
                        jnp.clip(coordinates[0], 0, nx - 1),
                        jnp.clip(coordinates[1], 0, ny - 1),
                        jnp.clip(coordinates[2], 0, nz - 1),
                    )
                    node_fluid = in_bounds & ~solid[safe]
                    fluid &= (weight <= 1e-12) | node_fluid
                    any_fluid |= (weight > 1e-12) & node_fluid
                return (
                    iteration + 1,
                    k0 + (t0 == value),
                    k1 + (t1 == value),
                    k2 + (t2 == value),
                    jnp.where(fluid, point[0], out0),
                    jnp.where(fluid, point[1], out1),
                    jnp.where(fluid, point[2], out2),
                    found | fluid,
                    through | (finite & ~fluid & ~any_fluid),
                    fluid | ~finite,
                )

            result = jax.lax.while_loop(condition, body, (0, 1, 1, 1, *initial_point, False, False, False))
            return jnp.stack(result[4:7]), result[7], result[8]

        return jax.jit(jax.vmap(trace, in_axes=(0, 0, None)))

    def _first_fluid_mesh_intersection_gpu(self, indices, directions, solid_mask, return_through=False):
        """Trace 3D characteristic rays on every local GPU with bounded temporary device memory. Returns (points, valid), and
        with return_through also whether each ray passed through solid before its point (see _ray_passes_through_solid)."""
        devices = tuple(device for device in jax.local_devices() if device.platform == "gpu")
        if not devices:
            raise RuntimeError("GPU geometric preprocessing requested, but no local GPU is available.")

        indices = np.asarray(indices, dtype=np.int32)
        directions = np.asarray(directions, dtype=np.float64)
        points = np.empty_like(directions)
        valid = np.empty((len(indices),), dtype=bool)
        through = np.empty((len(indices),), dtype=bool)
        batch_size = self.geometric_preprocessing_batch_size

        with jax.enable_x64():
            kernel = getattr(self, "_geometric_gpu_ray_kernel", None)
            if kernel is None:
                kernel = self._build_geometric_gpu_ray_kernel(batch_size)
                self._geometric_gpu_ray_kernel = kernel
            device_masks = tuple(jax.device_put(jnp.asarray(solid_mask, dtype=jnp.bool_), device) for device in devices)
            group_size = batch_size * len(devices)
            for group_start in range(0, len(indices), group_size):
                pending = []
                for device_index, (device, device_mask) in enumerate(zip(devices, device_masks, strict=True)):
                    start = group_start + device_index * batch_size
                    stop = min(start + batch_size, len(indices))
                    if start >= stop:
                        continue
                    count = stop - start
                    batch_indices = np.zeros((batch_size, 3), dtype=np.int32)
                    batch_directions = np.zeros((batch_size, 3), dtype=np.float64)
                    batch_indices[:count] = indices[start:stop]
                    batch_directions[:count] = directions[start:stop]
                    with jax.default_device(device):
                        output = kernel(
                            jax.device_put(batch_indices, device),
                            jax.device_put(batch_directions, device),
                            device_mask,
                        )
                    pending.append((start, stop, output))
                for start, stop, output in pending:
                    count = stop - start
                    points[start:stop] = np.asarray(output[0])[:count]
                    valid[start:stop] = np.asarray(output[1])[:count]
                    through[start:stop] = np.asarray(output[2])[:count]
        return (points, valid, through) if return_through else (points, valid)

    def _nearest_fluid_lattice_points(self, indices, normals, solid_mask):
        """Select the adjacent fluid lattice node most aligned with each wall normal."""
        if len(indices) == 0:
            return np.empty((0, self.dim), dtype=np.float64)

        directions = np.asarray(self.lattice.c, dtype=np.int32).T
        directions = directions[np.any(directions != 0, axis=1)]
        candidates = np.asarray(indices, dtype=np.int32)[:, None, :] + directions[None, :, :]
        domain_shape = np.asarray(solid_mask.shape, dtype=np.int32)
        in_bounds = np.all((candidates >= 0) & (candidates < domain_shape), axis=-1)
        safe_candidates = np.clip(candidates, 0, domain_shape - 1)
        fluid = in_bounds & ~solid_mask[tuple(np.moveaxis(safe_candidates, -1, 0))]
        if not np.all(np.any(fluid, axis=1)):
            raise RuntimeError("Geometric wetting interface node has no adjacent fluid lattice node.")

        unit_directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        alignment = np.asarray(normals, dtype=np.float64) @ unit_directions.T
        choice = np.argmax(np.where(fluid, alignment, -np.inf), axis=1)
        return np.asarray(candidates[np.arange(len(indices)), choice], dtype=np.float64)

    def _repair_invalid_geometric_point_data(self, point_data, validity, indices, normals, solid_mask, fallback_points):
        """Replace unresolved characteristic samples with fluid-only samples during preprocessing."""
        invalid_count = sum(np.count_nonzero(~valid) for valid in validity)
        if invalid_count == 0:
            return tuple(point_data)

        has_valid = np.logical_or.reduce(validity)
        all_invalid = ~has_valid
        if np.any(all_invalid):
            fallback_points[all_invalid] = self._nearest_fluid_lattice_points(
                indices[all_invalid],
                normals[all_invalid],
                solid_mask,
            )

        repaired_data = []
        for data, valid in zip(point_data, validity, strict=True):
            invalid_rows = np.flatnonzero(~valid)
            if len(invalid_rows):
                replacements = self._build_interpolation_data(fallback_points[invalid_rows], solid_mask)
                for values, replacement in zip(data, replacements, strict=True):
                    values[invalid_rows] = replacement
            repaired_data.append(data)

        logger.info(
            "Replaced %d unresolved geometric rays at %d wall nodes (%d had no valid ray).",
            invalid_count,
            np.count_nonzero(~np.logical_and.reduce(validity)),
            np.count_nonzero(all_invalid),
        )
        return tuple(repaired_data)

    @staticmethod
    def _characteristic_length_ok(indices, points, theta):
        """
        Whether characteristic samples are close enough to their wall node: at most 3 / sin(theta) + 2 nodes away. On a flat
        wall the sample is 1 / sin(theta) away; a ray that travels much further before it finds fluid grazes the wall, and
        the straight iso-surface the scheme assumes between the wall node and the sample does not hold over that distance.
        Such rays are dropped like rays through solid (the node's other samples replace them).
        """
        distance = np.linalg.norm(np.asarray(points, dtype=np.float64) - np.asarray(indices, dtype=np.float64), axis=-1)
        return distance <= 3.0 / np.maximum(np.abs(np.sin(theta)), 1e-3) + 2.0

    def _build_geometric_3d_characteristic_data(self, indices, normals, theta, solid_mask):
        """
        Build cone-sampled interpolation data for 3D geometric wetting.

        Parameters
        ----------
        indices (numpy.ndarray): Boundary node coordinates with shape (n, 3).

        normals (numpy.ndarray): Unit normals pointing from wall nodes toward fluid nodes.

        theta (numpy.ndarray): Contact angle in radians for each boundary node.

        solid_mask (numpy.ndarray): Boolean mask with True on boundary nodes.

        Returns
        -------
        point_data (tuple): Multilinear interpolation data for characteristic directions sampled on the contact-angle
        cone around the wall normal.

        Notes
        -----
        Assumes theta is prescribed in radians. The 3D construction samples eight azimuthal directions on the cone
        instead of selecting only two characteristic directions, then the wall density is selected from the extrema of
        those samples in apply_contact_angle.
        """
        sample_count = 8
        azimuths = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
        reference_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        reference_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        references = np.where((np.abs(normals[:, 0]) > 0.9)[:, None], reference_y, reference_x)
        tangent_1 = np.cross(normals, references)
        tangent_1_norm = np.linalg.norm(tangent_1, axis=1, keepdims=True)
        tangent_1 = np.divide(tangent_1, tangent_1_norm, out=np.zeros_like(tangent_1), where=tangent_1_norm > 1e-12)
        tangent_2 = np.cross(normals, tangent_1)

        angle = np.pi / 2 - theta
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        gpu_devices = tuple(device for device in jax.local_devices() if device.platform == "gpu")
        use_gpu = self.geometric_preprocessing_backend == "gpu" or (self.geometric_preprocessing_backend == "auto" and bool(gpu_devices))
        if self.geometric_preprocessing_backend == "gpu" and not gpu_devices:
            raise RuntimeError("GPU geometric preprocessing requested, but no local GPU is available.")
        point_data = []
        validity = []
        fallback_points = np.zeros_like(normals)
        has_fallback = np.zeros((len(indices),), dtype=bool)
        crosses_solid = np.zeros((len(indices),), dtype=bool)
        for azimuth in azimuths:
            tangent_direction = np.cos(azimuth) * tangent_1 + np.sin(azimuth) * tangent_2
            directions = cos_angle[:, None] * normals + sin_angle[:, None] * tangent_direction
            if use_gpu:
                try:
                    points, valid, through = self._first_fluid_mesh_intersection_gpu(indices, directions, solid_mask, return_through=True)
                except Exception as error:
                    if self.geometric_preprocessing_backend == "gpu":
                        raise
                    logger.warning("GPU geometric preprocessing failed; using CPU fallback: %s", error)
                    use_gpu = False
                    points, valid = self._first_fluid_mesh_intersection(indices, directions, solid_mask, return_valid=True)
            else:
                points, valid = self._first_fluid_mesh_intersection(indices, directions, solid_mask, return_valid=True)
            if not use_gpu:
                indices_f = np.asarray(indices, dtype=np.float64)
                through = np.zeros(len(indices), dtype=bool)
                rows = np.flatnonzero(valid)
                distance = np.linalg.norm(points[rows] - indices_f[rows], axis=1)
                through[rows] = self._ray_passes_through_solid(indices_f[rows], directions[rows], distance, solid_mask)
            # A characteristic belongs to this wall only when it reaches its sample without passing through solid.
            # Continuing through solid until a remote pore is found can import a density from the opposite side of a
            # grain.  In thin throats, the extrema selection then creates a liquid-density wall inside vapor and can
            # drive the adjacent fluid density negative in one step.  Treat such rays like unresolved ones so they use
            # a local same-wall sample (or the nearest adjacent fluid node when every ray is unresolved).
            crosses_solid |= valid & through
            valid &= ~through & self._characteristic_length_ok(indices, points, theta)
            first_valid = valid & ~has_fallback
            fallback_points[first_valid] = points[first_valid]
            has_fallback |= valid
            validity.append(valid)
            point_data.append(self._build_interpolation_data(points, solid_mask))
        # Only the rays that pass through solid are dropped: _repair_invalid_geometric_point_data replaces each of them by
        # a valid ray of the same wall node, which leaves the extremum over the cone unchanged. A node falls back to its
        # nearest adjacent fluid node (a neutral wall) only when none of its rays is valid.
        if np.any(crosses_solid):
            logger.info("Dropped geometric rays through solid at %d wall nodes.", np.count_nonzero(crosses_solid))
        return self._repair_invalid_geometric_point_data(
            point_data,
            validity,
            indices,
            normals,
            solid_mask,
            fallback_points,
        )

    # Halo width (x-planes) of each component's halo-form geometric-wetting data, None for the other forms (see
    # _localize_component_wetting_data); static for the kernels
    _geometric_wetting_halos = ()
    # Device form of the geometric-wetting data (see _localize_component_wetting_data): None chooses, "global" forces the
    # global form (for tests)
    _GEOMETRIC_WETTING_FORM = None

    # Largest periodic margin (nodes per side) of the padded solid mask the geometric characteristics are traced on
    _GEOMETRIC_PERIODIC_MARGIN = 32

    def _geometric_periodic_margin(self, BC):
        """
        Periodic margin per axis for tracing the characteristics of one component (see _create_geometric_wetting_data): the
        samples lie about |cot(theta)| from their wall node (twice that for the second wall row) plus the interpolation
        stencil, so 2 max|cot(theta)| + 6 nodes, at most _GEOMETRIC_PERIODIC_MARGIN and the domain size. A ray that runs
        further (nearly tangent to the wall) is treated as before, as one that leaves the domain.
        """
        thetas = [np.asarray(bc.theta, dtype=np.float64).reshape(-1) for bc in BC if self._is_wetting_boundary_condition(bc) and bc.theta is not None]
        theta = np.concatenate(thetas) if thetas else np.full(1, np.pi / 2)
        cot = np.max(np.abs(np.cos(theta)) / np.maximum(np.abs(np.sin(theta)), 1e-6))
        margin = int(min(self._GEOMETRIC_PERIODIC_MARGIN, np.ceil(2.0 * cot) + 6))
        shape = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        return tuple(min(margin, n) for n in shape)

    @staticmethod
    def _geometric_score_nodes(walls, offsets, solid_mask):
        """
        Nodes that choose a wall node's characteristic sample (see _geometric_wetting_rows): for each sample its nearest node
        and the node twice as far along the characteristic (the nearest node where the path to it crosses solid), wrapped into
        the periodic domain, shape (n, K, 2, dim); and whether the nearest node is fluid, shape (n, K).
        """
        shape = np.asarray(solid_mask.shape, dtype=np.int64)
        walls = np.asarray(walls, dtype=np.int64)[:, None, :]
        node = lambda t: np.mod(np.round(walls + t * offsets).astype(np.int64), shape)
        fluid = lambda nodes: ~solid_mask[tuple(np.moveaxis(nodes, -1, 0))]
        near, far = node(1.0), node(2.0)
        # The far node only along a clear path from the sample: past a thin grain it would be another pore's density
        clear = fluid(far)
        for t in (1.25, 1.5, 1.75):
            clear &= fluid(node(t))
        far = np.where(clear[..., None], far, near)
        return np.stack([near, far], axis=2), fluid(near)

    @staticmethod
    def _geometric_linear_samples(walls, offsets, solid_mask):
        """
        Multilinear interpolation data of every characteristic sample (see _geometric_wetting_rows): the 2^dim cell corners
        wrapped into the periodic domain, shape (n, K, 2^dim, dim), their weights (zero for the corners a sample on a grid
        line or plane does not use), and whether all corners with weight are fluid, shape (n, K).
        """
        shape = np.asarray(solid_mask.shape, dtype=np.int64)
        dim = len(shape)
        points = np.asarray(walls, dtype=np.float64)[:, None, :] + offsets
        rounded = np.round(points)
        # The points are rebuilt from float32 interpolation weights: snap the coordinates that are grid values up to that
        points = np.where(np.abs(points - rounded) < 1e-4, rounded, points)
        lower = np.floor(points).astype(np.int64)
        frac = points - lower
        corners = np.array([[(k >> axis) & 1 for axis in range(dim)] for k in range(2**dim)], dtype=np.int64)
        nodes = np.mod(lower[:, :, None, :] + corners, shape)
        weights = np.prod(np.where(corners == 1, frac[:, :, None, :], 1.0 - frac[:, :, None, :]), axis=-1)
        weights = np.where(weights > 1e-9, weights, 0.0)
        solid = solid_mask[tuple(np.moveaxis(nodes, -1, 0))]
        return nodes, weights, np.all((weights == 0) | ~solid, axis=-1)

    @staticmethod
    def _unpad_wall_samples(indices, theta, points, pad):
        """First-layer wall nodes and their samples in domain coordinates; the sample stencils are shifted but not wrapped,
        so that their weighted means stay the sample points (the wall-population correction wraps the nodes itself)."""
        offset = np.asarray(pad, dtype=np.int64)
        return (
            np.asarray(indices, dtype=np.int64) - offset,
            theta,
            tuple((np.asarray(index, dtype=np.int64) - offset, weight) for index, weight in points),
        )

    def _unpad_geometric_data(self, data, pad, shape, solid_mask=None):
        """
        Geometric-wetting data traced on the padded mask, back in domain coordinates: wall rows and stencil nodes wrapped
        into the domain. A wall row that wraps onto the node of an earlier row (a second-layer node behind a wall at a
        domain face that is the periodic image of a wall node) is dropped, so that each node is written once.
        """
        offset = np.asarray(pad, dtype=np.int64)
        shape_array = np.asarray(shape, dtype=np.int64)
        padded_nodes = np.stack([np.asarray(index, dtype=np.int64) for index in data["indices"]], axis=-1)
        nodes = np.mod(padded_nodes - offset, shape_array)
        _, first = np.unique(np.ravel_multi_index(tuple(nodes.T), shape), return_index=True)
        keep = np.sort(first)
        points = data["points"] if "points" in data else (data["point_1"], data["point_2"])
        # Sample points relative to their wall node (unwrapped, so that they stay right across the periodic faces)
        sample_offsets = np.stack([self._interpolation_points(point) for point in points], axis=1) - padded_nodes[:, None, :]

        def unpad(point):
            index, weight = point
            return np.mod(np.asarray(index, dtype=np.int64)[keep] - offset, shape_array).astype(np.int32), np.asarray(weight)[keep]

        result = dict(data)
        result["indices"] = tuple(np.asarray(column, dtype=np.int32) for column in nodes[keep].T)
        result["theta"] = np.asarray(data["theta"])[keep]
        result["offsets"] = sample_offsets[keep]
        # The wall rows come first and are all kept (distinct domain nodes), so the source rows stay valid
        result["source"] = np.asarray(data.get("source", np.arange(len(nodes))))[keep]
        if solid_mask is not None:
            result["score_nodes"], result["score_ok"] = self._geometric_score_nodes(nodes[keep], sample_offsets[keep], solid_mask)
            result["linear_nodes"], result["linear_weights"], result["linear_ok"] = self._geometric_linear_samples(
                nodes[keep], sample_offsets[keep], solid_mask
            )
        for key in ("point_1", "point_2"):
            if key in data:
                result[key] = unpad(data[key])
        if "points" in data:
            result["points"] = tuple(unpad(point) for point in data["points"])
        return result

    def _create_geometric_wetting_data(self, localize=True):
        """
        Precompute interpolation data for geometric wetting.

        Returns
        -------
        geometric_wetting_data (list): Component-wise interpolation data for wetted boundary nodes.

        geometric_fluid_mask (list): Component-wise boolean masks (jax.numpy.ndarray) with True on fluid nodes, used to clamp wall densities to the fluid density range.

        Notes
        -----
        Assumes theta is prescribed in radians for each wetted boundary node. Boundary conditions without theta are included in the solid mask but skipped for contact-angle
        interpolation. The 2D implementation keeps the original two-characteristic construction. The 3D implementation samples multiple characteristic directions on the
        contact-angle cone around each wall normal and stores interpolation data for every sample.

        References
        ----------
        1. Fei, Linlin, Feifei Qin, Jianlin Zhao, Dominique Derome, and Jan Carmeliet.
        “Lattice Boltzmann Modelling of Isothermal Two-Component Evaporation in Porous Media.”
        Journal of Fluid Mechanics 955 (January 2023): A18. doi: 10.1017/jfm.2022.1048.
        2. Wang, Lei, Hai-bo Huang, and Xi-Yun Lu. “Scheme for Contact Angle and Its Hysteresis in a Multiphase Lattice
        Boltzmann Method.”
        Physical Review E 87, no. 1 (2013): 013301. doi:10.1103/PhysRevE.87.013301.
        """
        geometric_wetting_data = []
        geometric_fluid_mask = []
        geometric_force_mask = []
        # First-layer wall nodes with their characteristic samples, per component, for the wall-population correction
        # (see _create_geometric_wall_correction_data)
        wall_samples = []
        characteristics_time = 0.0
        for BC in self.BCs:
            wall_samples.append([])
            solid_mask = self._create_component_solid_mask(BC)
            fluid_mask_host = ~solid_mask[..., None]
            geometric_fluid_mask.append(self.distributed_array_init(fluid_mask_host.shape, jnp.bool_, init_val=fluid_mask_host))
            force_mask_host = self._create_geometric_force_mask(fluid_mask_host[..., 0])[..., None]
            geometric_force_mask.append(self.distributed_array_init(force_mask_host.shape, jnp.bool_, init_val=force_mask_host))
            component_data = []
            # The domain is periodic: characteristics are traced on the solid mask padded with its periodic images, so that
            # walls at the domain faces sample across them, and the results are wrapped back (see _unpad_geometric_data)
            pad = self._geometric_periodic_margin(BC)
            domain_mask = solid_mask
            solid_mask = np.pad(domain_mask, [(m, m) for m in pad], mode="wrap")
            for bc in BC:
                if not self._is_wetting_boundary_condition(bc):
                    continue
                if bc.theta is None:
                    continue

                domain_indices = np.array(self._get_solid_indices(bc), dtype=np.int32).T
                member_mask = np.zeros(domain_mask.shape, dtype=bool)
                member_mask[tuple(domain_indices.T)] = True
                member_mask = np.pad(member_mask, [(m, m) for m in pad], mode="wrap")
                indices = domain_indices + np.asarray(pad, dtype=np.int32)
                theta = np.asarray(bc.theta, dtype=np.float64).reshape(-1)
                if theta.size == 1:
                    theta = np.full((indices.shape[0],), theta.item(), dtype=np.float64)
                if theta.shape[0] != indices.shape[0]:
                    raise ValueError("Geometric wetting theta must be scalar or match the number of boundary nodes.")

                interface = self._solid_fluid_interface_mask(indices, solid_mask)
                if not np.any(interface):
                    continue
                source_rows = np.flatnonzero(interface)
                indices = indices[interface]
                theta = theta[interface]
                normals = self._compute_geometric_normals(bc, solid_mask, indices=indices, source_rows=source_rows)
                # A one-layer wall on a domain face has fluid on both sides through the periodic face, and its neighbor
                # normal cancels: keep the normal toward the fluid inside the domain, as without the periodic images
                cancelled = np.flatnonzero(np.linalg.norm(normals, axis=1) <= 1e-12)
                if len(cancelled):
                    normals[cancelled] = self._one_sided_face_normals(domain_indices[source_rows[cancelled]], domain_mask)
                valid_normal = np.linalg.norm(normals, axis=1) > 1e-12
                if not np.any(valid_normal):
                    continue
                indices = indices[valid_normal]
                theta = theta[valid_normal]
                normals = normals[valid_normal]

                if self.dim == 3:
                    characteristics_start = time.perf_counter()
                    points = self._build_geometric_3d_characteristic_data(indices, normals, theta, solid_mask)
                    characteristics_time += time.perf_counter() - characteristics_start
                    data = {
                        "indices": tuple(np.asarray(index, dtype=np.int32) for index in indices.T),
                        "theta": np.asarray(theta.reshape(-1, 1), dtype=np.dtype(self.precision_policy.compute_dtype)),
                        "points": points,
                    }
                    wall_samples[-1].append(self._unpad_wall_samples(indices, theta, tuple(points), pad))
                    data = self._append_wall_rows(data, *self._stencil_reach_wall_rows(indices, member_mask, solid_mask), solid_mask)
                    data = self._unpad_geometric_data(data, pad, domain_mask.shape, domain_mask)
                    component_data.append(data)
                    continue

                # Characteristics determination for density interpolation
                # The density of nearest intersection to mesh is used for solid density.
                # In most cases, intersection does not occur on a fluid point so the density is interpolated from neighboring fluid points.
                # This ensures a local density value is used.
                characteristics_start = time.perf_counter()
                angle = np.pi / 2 - theta
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                direction_1 = np.column_stack((
                    normals[:, 0] * cos_angle - normals[:, 1] * sin_angle,
                    normals[:, 0] * sin_angle + normals[:, 1] * cos_angle,
                ))
                direction_2 = np.column_stack((
                    normals[:, 0] * cos_angle + normals[:, 1] * sin_angle,
                    -normals[:, 0] * sin_angle + normals[:, 1] * cos_angle,
                ))

                points_1, valid_1 = self._first_fluid_mesh_intersection(indices, direction_1, solid_mask, return_valid=True)
                points_2, valid_2 = self._first_fluid_mesh_intersection(indices, direction_2, solid_mask, return_valid=True)
                valid_1 &= self._characteristic_length_ok(indices, points_1, theta)
                valid_2 &= self._characteristic_length_ok(indices, points_2, theta)
                characteristics_time += time.perf_counter() - characteristics_start
                fallback_points = np.zeros_like(normals)
                fallback_points[valid_1] = points_1[valid_1]
                fallback_points[~valid_1 & valid_2] = points_2[~valid_1 & valid_2]
                point_1, point_2 = self._repair_invalid_geometric_point_data(
                    [self._build_interpolation_data(points_1, solid_mask), self._build_interpolation_data(points_2, solid_mask)],
                    [valid_1, valid_2],
                    indices,
                    normals,
                    solid_mask,
                    fallback_points,
                )
                data = {
                    "indices": tuple(np.asarray(index, dtype=np.int32) for index in indices.T),
                    "theta": np.asarray(theta.reshape(-1, 1), dtype=np.dtype(self.precision_policy.compute_dtype)),
                    "point_1": point_1,
                    "point_2": point_2,
                }
                wall_samples[-1].append(self._unpad_wall_samples(indices, theta, (point_1, point_2), pad))
                data = self._append_wall_rows(data, *self._stencil_reach_wall_rows(indices, member_mask, solid_mask), solid_mask)
                data = self._unpad_geometric_data(data, pad, domain_mask.shape, domain_mask)
                component_data.append(data)
            if localize:
                component_data, halo = self._localize_component_wetting_data(component_data)
                self._geometric_wetting_halos.append(halo)
            geometric_wetting_data.append(component_data)

        logger.info(f"Time taken to determine geometric wetting characteristics: {characteristics_time:.6f} seconds")

        self.geometric_force_mask = geometric_force_mask
        self._geometric_wall_samples = wall_samples

        return geometric_wetting_data, geometric_fluid_mask

    def get_solid_mask_streamed(self):
        """
        Define the solid mask used for the fluid-solid interaction (wetting) force. One flag per node, not per
        lattice direction - neighbor-direction information is derived on demand by scalar_neighbor_sum instead of
        being pre-streamed into a persistent per-direction mask. The boundary conditions must be passed separately.

        Returns
        -------
        list of jax.Array or None: Component masks with shape (nx, ny, 1) for d == 2 or (nx, ny, nz, 1) for d == 3.
        Components without a wetting boundary contain None.
        """
        shape = (self.nx, self.ny, 1) if self.dim == 2 else (self.nx, self.ny, self.nz, 1)
        solid_mask = []
        for component_bcs, has_wetting_bc in zip(self.BCs, self._has_wetting_bc, strict=True):
            if not has_wetting_bc:
                solid_mask.append(None)
                continue

            solid_indices = [np.array(self._get_solid_indices(bc)).T for bc in component_bcs if self._is_wetting_boundary_condition(bc)]
            mask_host = np.zeros(shape, dtype=np.int8)
            if solid_indices:
                index = np.vstack(solid_indices)
                mask_host[tuple(index.T)] = 1
            mask = self.distributed_array_init(shape, jnp.int8, init_val=mask_host)
            solid_mask.append(mask)
        return solid_mask

    def _neighbor_stencil_m(self, field, weights):
        """
        Sum a scalar (single-channel) field over its lattice neighbors, weighted per direction: out = sum_i w_i field(x - c_i).
        The field is padded once (one-cell x halo exchanged between devices, periodic wrap in y/z) and every neighbor is a
        static slice of the padded array, so XLA fuses the whole stencil into a single loop.

        Bound to static weights via functools.partial: the G_ff-weighted neighbor sum used by
        compute_average_density's wetting denominator (scalar_neighbor_sum, scalar weights) and the MRT surface-tension
        moments (scalar_surface_stencil, vector weights) - both without ever streaming a q-channel array.

        Parameters
        ----------
        field (jax.numpy.ndarray): Local shard of a scalar field, shape (nx, ny, 1) or (nx, ny, nz, 1).

        weights (numpy.ndarray): Per-direction weights, shape (q,) for a scalar-weighted neighbor sum, or
            (dim, q) for a direction-vector weighted sum (one weight vector per lattice direction).

        Returns
        -------
        (jax.numpy.ndarray): Local shard of the weighted neighbor sum, shape (..., 1) for scalar weights or
        (..., dim) for vector weights.
        """
        field = field.astype(self.precision_policy.compute_dtype)
        left = self.send_right(field[-1:], "x")
        right = self.send_left(field[:1], "x")
        padded = jnp.concatenate((left, field, right), axis=0)
        padded = jnp.pad(padded, ((0, 0),) + ((1, 1),) * (self.dim - 1) + ((0, 0),), mode="wrap")
        spatial = field.shape[:-1]

        weights = np.asarray(weights, dtype=np.float64)
        rows = weights if weights.ndim == 2 else weights[None, :]
        totals = [None] * rows.shape[0]
        for q_index, direction in enumerate(np.asarray(self.lattice.c).T):
            column = rows[:, q_index]
            if np.all(column == 0.0):
                continue
            value = padded[tuple(slice(1 - int(c), 1 - int(c) + n) for c, n in zip(direction[: self.dim], spatial, strict=True))]
            for channel, w in enumerate(column):
                if w == 0.0:
                    continue
                term = value * float(w)
                totals[channel] = term if totals[channel] is None else totals[channel] + term
        totals = [t if t is not None else jnp.zeros_like(field) for t in totals]
        return jnp.concatenate(totals, axis=-1)

    def neighbor_stencil(self, field, weights):
        """
        Weighted sum of a scalar field over its lattice neighbours, out = sum_i w_i field(x - c_i), in one fused pass (the
        neighbours are slices of a padded copy of the field; no q-channel array is formed).

        Parameters
        ----------
        field (jax.numpy.ndarray): Scalar field, shape (nx, ny, 1) or (nx, ny, nz, 1).

        weights (array-like): Per-direction weights, shape (q,) for one output channel or (channels, q), e.g.
            G_ff[None, :] * c for the vector sum_i G_ff_i c_i field(x - c_i) of a Shan-Chen force.

        Returns
        -------
        (jax.numpy.ndarray): The weighted sums, shape (..., 1) or (..., channels).
        """
        weights = np.asarray(weights, dtype=np.float64)
        key = (weights.shape, weights.tobytes())
        stencils = self.__dict__.setdefault("_neighbor_stencils", {})
        if key not in stencils:
            spec = PartitionSpec("x")
            stencils[key] = shard_map(
                partial(self._neighbor_stencil_m, weights=weights), mesh=self.mesh, in_specs=spec, out_specs=spec, check_vma=False
            )
        return stencils[key](field)

    def _build_efm_gradient_offsets(self):
        """
        Collapse the fourth-order isotropic EFM gradient into static (offset, weight-vector) pairs.

        Along each lattice direction c_i the directional derivative is (8 (f(x+c) - f(x-c)) - (f(x+2c) - f(x-2c))) / 12, and
        grad f = (1/cs2) sum_i w_i c_i (c_i . grad f). Summing over both members of every +-c pair gives

            grad f(x) = sum_i (w_i / cs2) c_i [ (4/3) f(x + c_i) - (1/6) f(x + 2 c_i) ],

        which for D3Q19 reproduces Eqs. (21)-(23) of Yang & Qin (2024) term by term.

        Returns
        -------
        tuple of (tuple of int, tuple of float)
            Integer spatial offset and the gradient weight it contributes to each axis (zero entries kept).
        """
        c = np.asarray(self.lattice.c).T
        w = np.asarray(self.lattice.w, dtype=np.float64)
        weights = {}
        for direction, weight in zip(c, w, strict=True):
            if not np.any(direction):
                continue
            g = weight / self.lattice.cs2
            for multiple, coefficient in ((1, 4.0 / 3.0), (2, -1.0 / 6.0)):
                offset = tuple(int(multiple * component) for component in direction)
                weights[offset] = weights.get(offset, np.zeros(self.dim)) + g * coefficient * direction
        return tuple((offset, tuple(float(v) for v in vector)) for offset, vector in weights.items())

    def _gradient_stencil_m(self, field, weights=None):
        """
        Fourth-order isotropic EFM gradient of a scalar field (local shard).

        The field is padded once (two-cell X halo exchanged between devices, periodic wrap in Y/Z) and every stencil
        point is a static slice of that padded array, so XLA fuses the whole stencil into a single loop.

        Parameters
        ----------
        field (jax.numpy.ndarray): Local shard of a scalar field, shape (nx, ny, 1) or (nx, ny, nz, 1).

        weights (numpy.ndarray, optional): (out_channels, q) nearest-neighbor weights. When given, the weighted sums
            sum_i w_i field(x - c_i) (as in _neighbor_stencil_m) are appended from the same padded array.

        Returns
        -------
        (jax.numpy.ndarray): Gradient, shape (nx, ny, 2) or (nx, ny, nz, 3), followed by out_channels neighbor sums.
        """
        halo = 2
        field = field.astype(self.precision_policy.compute_dtype)
        left = self.send_right(field[-halo:], "x")
        right = self.send_left(field[:halo], "x")
        padded = jnp.concatenate((left, field, right), axis=0)
        padded = jnp.pad(padded, ((0, 0),) + ((halo, halo),) * (self.dim - 1) + ((0, 0),), mode="wrap")
        # Materialize the padded field once: the gradient is fused into its consumer (the collision), and without the barrier
        # XLA would recompute the field (psi, from the density, from the populations) once per stencil neighbor
        padded = jax.lax.optimization_barrier(padded)
        spatial = field.shape[:-1]

        gradient = [None] * self.dim
        for offset, vector in self._efm_offsets:
            value = padded[tuple(slice(halo + o, halo + o + n) for o, n in zip(offset, spatial, strict=True))]
            for axis, weight in enumerate(vector):
                if weight == 0.0:
                    continue
                term = value * weight
                gradient[axis] = term if gradient[axis] is None else gradient[axis] + term
        if weights is None:
            return jnp.concatenate(gradient, axis=-1)

        sums = [None] * weights.shape[0]
        for q_index, direction in enumerate(np.asarray(self.lattice.c).T):
            column = weights[:, q_index]
            if np.all(column == 0.0):
                continue
            value = padded[tuple(slice(halo - int(c), halo - int(c) + n) for c, n in zip(direction[: self.dim], spatial, strict=True))]
            for channel, w in enumerate(column):
                if w == 0.0:
                    continue
                term = value * float(w)
                sums[channel] = term if sums[channel] is None else sums[channel] + term
        sums = [t if t is not None else jnp.zeros_like(field) for t in sums]
        return jnp.concatenate(gradient + sums, axis=-1)

    def _create_boundary_data(self):
        """
        Create boundary data for the Lattice Boltzmann simulation by setting boundary conditions,
        creating grid mask, and preparing local masks and normal arrays.
        """
        self.BCs = [[] for _ in range(self.n_components)]
        self.set_boundary_conditions()
        # Accumulate the indices of all BCs to create the grid mask with FALSE along directions that
        # stream into a boundary voxel.
        for i in range(self.n_components):
            logger.info(f"Component: {i}")
            solid_halo_list = [np.array(bc.indices).T for bc in self.BCs[i] if bc.is_solid]
            solid_halo_voxels = np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None

            # Create the grid mask on each process
            start = time.time()
            grid_mask = self.create_grid_mask(solid_halo_voxels)
            logger.info("Time to create the grid mask: %.6f seconds", time.time() - start)

            start = time.time()
            for bc in self.BCs[i]:
                assert bc.implementation_step in ["PostStreaming", "PostCollision"]
                bc.create_local_mask_and_normal_arrays(grid_mask)
            link_fullway_walls(self.BCs[i])
            logger.info("Time to create the local masks and normal arrays: %.6f seconds", time.time() - start)

    def _export_force_corrected_fields(self, rho_scalar_tree, velocity_tree, force_tree):
        """
        Hand the collision's density and force-corrected velocity to a coupled solver (see _export_macroscopic):
        u = sum_i c_i f_i / rho + F / (2 rho), with the same operations as Multiphase.macroscopic_velocity.
        """
        if self._wall_correction_export is not None:
            # Density and physical momentum j = rho u + F / 2 for the geometric-wetting bounce-back correction
            self._wall_correction_export.append([
                tuple([rho] + [rho * v + 0.5 * force[..., axis] for axis, v in enumerate(velocity)])
                for rho, velocity, force in zip(rho_scalar_tree, velocity_tree, force_tree, strict=True)
            ])
        if self._export_macroscopic is None:
            return
        rho_tree = [rho[..., None] for rho in rho_scalar_tree]
        u_tree = [jnp.stack(velocity, axis=-1) + 0.5 * force / rho for velocity, force, rho in zip(velocity_tree, force_tree, rho_tree, strict=True)]
        # Materialize the velocity once: otherwise XLA fuses this cheap expression of the populations into each of its consumers
        # (the temperature Runge-Kutta stages), reading the populations again in every one
        u_tree = list(jax.lax.optimization_barrier(tuple(u_tree)))
        self._export_macroscopic.append((rho_tree, u_tree))

    def _make_bounceback_mask(self):
        """One full-way BounceBack mask (or None) per component (see LBMBase._bounceback_mask_for)."""
        return [self._bounceback_mask_for(BCs) for BCs in self.BCs]

    @partial(jit, static_argnums=(0, 3), inline=True)
    def equilibrium(self, rho_tree, u_tree, cast_output=True):
        """
        Compute the equilibrium distribution using density and velocity pytrees.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Density field

        u_tree (pytree of jax.numpy.ndarray): Velocity field

        cast_output (bool, optional): A flag to cast the density and velocity values to the compute and output
            precision. Default: True

        Returns
        -------
        feq_tree (pytree of jax.numpy.ndarray): Equilibrium distribution.
        """
        if cast_output:
            cast = lambda x: self.precision_policy.cast_to_compute(x)
            rho_tree = tree_map(cast, rho_tree)
            u_tree = tree_map(cast, u_tree)

        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cu_tree = tree_map(lambda u: 3.0 * jnp.dot(u, c), u_tree)
        usqr_tree = tree_map(lambda u: 1.5 * jnp.sum(jnp.square(u), axis=-1, keepdims=True), u_tree)
        feq_tree = tree_map(lambda rho, udote, udotu: rho * self.w * (1.0 + udote * (1.0 + 0.5 * udote) - udotu), rho_tree, cu_tree, usqr_tree)

        if cast_output:
            return tree_map(lambda f_eq: self.precision_policy.cast_to_output(f_eq), feq_tree)
        else:
            return feq_tree

    def _global_min_max_m(self, rho):
        """
        Minimum and maximum of a sharded density field: the shard-local (-min, max) pair is combined across devices with a
        single pmax, so the global min/max needs one collective instead of the two that jnp.min and jnp.max compile to.
        Negation is exact, so the values equal jnp.min(rho) and jnp.max(rho).
        """
        extremes = jax.lax.pmax(jnp.stack([-jnp.min(rho), jnp.max(rho)]), "x")
        return -extremes[0], extremes[1]

    def local_improved_wetting_m(self, rho, rho_ave, local_indices, local_theta, local_phi, local_delta_rho):
        """Apply improved-virtual-density wetting using local solid-node indices on each x shard."""
        indices = local_indices[0]
        idx = tuple(indices[:, axis] for axis in range(self.dim))
        rho_ave_boundary = rho_ave.at[idx].get(mode="fill", fill_value=0.0)
        theta = local_theta[0]
        wall_density = (theta <= jnp.pi / 2) * (local_phi[0] * rho_ave_boundary) + (theta > jnp.pi / 2) * (rho_ave_boundary - local_delta_rho[0])
        return rho.at[idx].set(wall_density, mode="drop")

    def _localize_component_wetting_data(self, component_data):
        """
        Device form of one component's geometric-wetting data (see _geometric_wetting_m): per shard, with the density padded
        by halo x-planes of the neighbor shards, when every node a wall node reads lies within one shard width of it along x
        (always on one device); otherwise global (indices into the whole density, for shards thinner than the stencils).
        Returns (data, halo width, or None for the global form).
        """
        if not component_data:
            self._geometric_wetting_orders.append([])
            return component_data, None
        halo = max(self._geometric_wetting_reach(data) for data in component_data)
        if self._GEOMETRIC_WETTING_FORM == "global" or (self.n_devices > 1 and halo > self.nx // self.n_devices):
            halo = None
        localized = [self._device_geometric_wetting_data(data, halo) for data in component_data]
        self._geometric_wetting_orders.append([order for _, order in localized])
        return [arrays for arrays, _ in localized], halo

    def _geometric_wetting_points(self, data):
        return data["points"] if self.dim == 3 else (data["point_1"], data["point_2"])

    def _geometric_wetting_reach(self, data):
        """Largest periodic x distance from a wall node to a node it reads (sample stencils, multilinear corners, nearest nodes),
        at least 1."""
        wall_x = np.asarray(data["indices"][0], dtype=np.int64)
        reach = 1
        nodes = [(np.asarray(index, dtype=np.int64)[..., 0], np.asarray(weight) != 0) for index, weight in self._geometric_wetting_points(data)]
        near_x = np.asarray(data["score_nodes"])[:, :, 0, 0]  # the nearest nodes (the far ones are the correction's)
        nodes.append((near_x, np.ones(near_x.shape, dtype=bool)))
        linear_x = np.asarray(data["linear_nodes"])[..., 0].reshape(len(wall_x), -1)
        nodes.append((linear_x, np.asarray(data["linear_weights"]).reshape(len(wall_x), -1) != 0))
        for x, active in nodes:
            dx = (x - wall_x.reshape((-1,) + (1,) * (x.ndim - 1)) + self.nx // 2) % self.nx - self.nx // 2
            if np.any(active):
                reach = max(reach, int(np.abs(dx[active]).max()))
        return reach

    def _device_geometric_wetting_data(self, data, halo):
        """
        Device form of one boundary's geometric-wetting data (see _geometric_wetting_m). Returns (arrays, row order): per
        block, the wall nodes (linear index); per characteristic sample its nearest node, its multilinear corners and weights
        and its interpolation stencil and weights (linear indices into the block's density source), whether the nearest node
        and the corners are usable; the extremum to take; and per shard the data rows in the order of the kernel's output (the blocks one after the other, each
        padded to its widest shard).

        With a halo width, per shard: "interior", the wall nodes at least halo planes from both shard faces, which read only
        the shard's density; "edge", the others, which read an edge array of the halo x-planes of the neighbor shards
        (periodic) and the shard's own 2 * halo first and last planes (all planes, the halo-padded density, when the shard
        is thinner than 4 * halo). Without (global form): one "interior" block, with the whole density.
        """
        global_form = halo is None
        devices = 1 if global_form else self.n_devices
        shape = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        local_nx = self.nx // devices
        local_shape = shape if global_form else self._local_shape()
        plane = int(np.prod(local_shape[1:]))
        halo = 0 if global_form else halo
        thin = 4 * halo > local_nx
        edge_planes = local_nx + 2 * halo if thin else 6 * halo
        walls = np.stack([np.asarray(index, dtype=np.int64) for index in data["indices"]], axis=-1)
        points = self._geometric_wetting_points(data)
        stencil = np.stack([np.asarray(index, dtype=np.int64) for index, _ in points], axis=1)  # (n, K, S, dim)
        weights = np.stack([np.asarray(weight) for _, weight in points], axis=1)  # (n, K, S)
        score_nodes = np.asarray(data["score_nodes"], dtype=np.int64)  # (n, K, 2, dim)
        linear_nodes = np.asarray(data["linear_nodes"], dtype=np.int64)  # (n, K, 2^dim, dim)
        linear_weights = np.asarray(data["linear_weights"])
        # Only the corners with weight (a sample lies on a grid line or plane: at most 2^(dim - 1) of them)
        order = np.argsort(linear_weights == 0, axis=-1, kind="stable")
        corners = max(1, int(np.max(np.sum(linear_weights != 0, axis=-1))))
        order = order[..., :corners]
        linear_nodes = np.take_along_axis(linear_nodes, order[..., None], axis=2)
        linear_weights = np.take_along_axis(linear_weights, order, axis=2)
        theta = np.asarray(data["theta"]).reshape(-1)
        owner = walls[:, 0] // local_nx
        local_x = walls[:, 0] - owner * local_nx
        K, S = stencil.shape[1:3]
        dtype = np.dtype(self.precision_policy.compute_dtype)

        def plane_of(x, edge):
            """Plane in the block's density source of shard-local x (-halo <= x < local_nx + halo); -1 if not there."""
            if global_form:
                return np.mod(x, self.nx)
            if not edge:
                return np.where((x >= 0) & (x < local_nx), x, -1)
            if thin:
                return np.where((x >= -halo) & (x < local_nx + halo), x + halo, -1)
            return np.select(
                [
                    (x >= -halo) & (x < 0),
                    (x >= 0) & (x < 2 * halo),
                    (x >= local_nx - 2 * halo) & (x < local_nx),
                    (x >= local_nx) & (x < local_nx + halo),
                ],
                [x + halo, x + halo, x - local_nx + 5 * halo, x - local_nx + 5 * halo],
                -1,
            )

        def block(rows_per_device, edge):
            planes = edge_planes if edge else local_shape[0]
            sentinel = planes * plane
            n = max(1, max(len(rows) for rows in rows_per_device))
            out = {
                "wall_lin": np.full((devices, n), int(np.prod(local_shape)), dtype=np.int32),
                "near_lin": np.full((devices, n, K), sentinel, dtype=np.int32),
                "near_ok": np.zeros((devices, n, K), dtype=bool),
                "linear_lin": np.full((devices, n, K, corners), sentinel, dtype=np.int32),
                "linear_weights": np.zeros((devices, n, K, corners), dtype=dtype),
                "linear_ok": np.zeros((devices, n, K), dtype=bool),
                "stencil_lin": np.full((devices, n, K, S), sentinel, dtype=np.int32),
                "weights": np.zeros((devices, n, K, S), dtype=dtype),
                "select_max": np.zeros((devices, n, 1), dtype=bool),
            }

            def lin(nodes, rows):
                wall_x = walls[rows][:, 0].reshape((-1,) + (1,) * (nodes.ndim - 2))
                dx = (nodes[..., 0] - wall_x + self.nx // 2) % self.nx - self.nx // 2
                x = plane_of(local_x[rows].reshape(wall_x.shape) + dx, edge)
                inside = x >= 0
                nodes = nodes.copy()
                nodes[..., 0] = np.where(inside, x, 0)
                return np.ravel_multi_index(tuple(np.moveaxis(nodes, -1, 0)), (planes, *local_shape[1:])), inside

            for d, rows in enumerate(rows_per_device):
                count = len(rows)
                if count == 0:
                    continue
                local = walls[rows].copy()
                local[:, 0] = local_x[rows]
                out["wall_lin"][d, :count] = np.ravel_multi_index(tuple(local.T), local_shape)
                index, inside = lin(stencil[rows], rows)
                active = (weights[rows] != 0) & inside
                out["stencil_lin"][d, :count] = np.where(active, index, sentinel)
                out["weights"][d, :count] = np.where(active, weights[rows], 0.0)
                index, inside = lin(score_nodes[rows][:, :, 0], rows)
                out["near_lin"][d, :count] = np.where(inside, index, sentinel)
                out["near_ok"][d, :count] = np.asarray(data["score_ok"])[rows] & inside
                index, inside = lin(linear_nodes[rows], rows)
                used = linear_weights[rows] != 0
                out["linear_lin"][d, :count] = np.where(used & inside, index, sentinel)
                out["linear_weights"][d, :count] = np.where(used & inside, linear_weights[rows], 0.0)
                out["linear_ok"][d, :count] = np.asarray(data["linear_ok"])[rows] & np.all(inside | ~used, axis=-1)
                out["select_max"][d, :count, 0] = theta[rows] <= np.pi / 2
            if global_form:  # plain arrays, no shard axis
                return {key: jnp.asarray(value[0]) for key, value in out.items()}, n
            types = {
                "wall_lin": jnp.int32,
                "near_lin": jnp.int32,
                "linear_lin": jnp.int32,
                "stencil_lin": jnp.int32,
                "near_ok": jnp.bool_,
                "linear_ok": jnp.bool_,
                "select_max": jnp.bool_,
            }
            return {key: self._distribute_local(value, types.get(key, self.precision_policy.compute_dtype)) for key, value in out.items()}, n

        if global_form:
            interior = [np.arange(len(walls))]
            arrays, _ = block(interior, edge=False)
            return {"interior": arrays}, interior
        near_face = (local_x < halo) | (local_x >= local_nx - halo)
        interior = [np.flatnonzero((owner == d) & ~near_face) for d in range(devices)]
        edge = [np.flatnonzero((owner == d) & near_face) for d in range(devices)]
        interior_arrays, width = block(interior, edge=False)
        edge_arrays, edge_width = block(edge, edge=True)
        # Output order per shard: the interior rows, then the edge rows, each padded to the widest shard (-1)
        pad = lambda rows, n: np.pad(rows, (0, n - len(rows)), constant_values=-1)
        order = [np.concatenate([pad(i, width), pad(e, edge_width)]) for i, e in zip(interior, edge, strict=True)]
        return {"interior": interior_arrays, "edge": edge_arrays}, order

    @staticmethod
    def _best_sample(score):
        """Index of the largest score along the sample axis (n, K), the first on ties: a running comparison over the few
        samples, which XLA fuses into one pass (argmax is a two-operand reduction)."""
        best, chosen = score[:, 0], jnp.zeros(score.shape[:1], dtype=jnp.int32)
        for k in range(1, score.shape[1]):
            better = score[:, k] > best
            best = jnp.where(better, score[:, k], best)
            chosen = jnp.where(better, k, chosen)
        return chosen

    @staticmethod
    def _pick_sample(array, chosen, fill=0):
        """array[r, chosen[r]] for array (n, K, ...): one gather of the chosen entries (take_along_axis over the sample axis
        would read all K); fill where chosen is negative (no memory read)."""
        n, K = array.shape[:2]
        rows = jnp.arange(chosen.shape[0], dtype=jnp.int32)
        index = jnp.where(chosen >= 0, rows * K + chosen, n * K)
        return array.reshape((n * K,) + array.shape[2:]).at[index].get(mode="fill", fill_value=fill)

    # Geometric wetting near interfaces only: a wall node whose score densities (see _geometric_wetting_rows) all agree
    # within this fraction of the fluid's density range sits in one uniform phase; it takes the score density directly
    # (the value any interpolation there gives) and is left out of the bounce-back correction (which vanishes there)
    _GEOMETRIC_WETTING_ACTIVITY = 1e-4

    def _geometric_wetting_rows(self, padded, data, rho_min, rho_max):
        """
        Wall densities of one block of wall nodes (see _geometric_wetting_m and _device_geometric_wetting_data): the extremum
        (largest for theta <= 90 degrees, smallest otherwise) of each wall node's characteristic samples, the samples
        compared by multilinear interpolation and the extremal one interpolated with its interpolation data (Catmull-Rom by
        default), clipped to the fluid range. A wall node whose samples' nearest-node densities agree within
        _GEOMETRIC_WETTING_ACTIVITY of the fluid range sits in a uniform phase: it takes the extremal nearest-node density
        and reads nothing more. Returns (densities, -1 for the wall nodes in a uniform phase and 0 for the others, for the
        bounce-back correction).
        """
        padded_rho = padded[..., 0]
        size = padded_rho.shape[0]
        sign = 2 * data["select_max"].astype(padded.dtype) - 1  # (n, 1): +1 for the largest, -1 for the smallest
        near = padded_rho.at[data["near_lin"]].get(mode="fill", fill_value=0.0)  # (n, K)
        spread = jnp.max(jnp.where(data["near_ok"], near, -jnp.inf), axis=1) - jnp.min(jnp.where(data["near_ok"], near, jnp.inf), axis=1)
        active = spread > self._GEOMETRIC_WETTING_ACTIVITY * (rho_max - rho_min)
        uniform = jnp.max(jnp.where(data["near_ok"], sign * near, -jnp.inf), axis=1) * sign[:, 0]
        # Compare the samples by multilinear interpolation (its corners with weight only). Masking the indices of the
        # uniform-phase wall nodes would cost a pass over the index array: XLA does not fuse it into the gather
        n, K, _ = data["linear_lin"].shape
        linear = jnp.sum(data["linear_weights"] * padded_rho.at[data["linear_lin"]].get(mode="fill", fill_value=0.0), axis=-1)  # (n, K)
        chosen = self._best_sample(jnp.where(data["linear_ok"], sign * linear, -jnp.inf))
        pick = jnp.where(active, jnp.arange(n, dtype=jnp.int32) * K + chosen, n * K)
        stencil = data["stencil_lin"].reshape(n * K, -1).at[pick].get(mode="fill", fill_value=size)  # (n, S)
        weights = data["weights"].reshape(n * K, -1).at[pick].get(mode="fill", fill_value=0.0)
        interpolated = jnp.sum(weights[..., None] * padded.at[stencil].get(mode="fill", fill_value=0.0), axis=1)
        rho_wall = jnp.where(active[:, None], interpolated, uniform[:, None])
        return jnp.clip(rho_wall, rho_min, rho_max), jnp.where(active, 0, -1)

    def _geometric_wetting_m(self, rho, fluid_mask, component_data, halo):
        """
        Geometric wetting of one component, shard-local (see _device_geometric_wetting_data): the interior wall nodes read the
        shard's density, the edge ones an edge array that includes halo x-planes of the neighbor shards (one exchange of
        2 * halo planes of one scalar field, which the interior work overlaps), and every wall node gets the density of its
        chosen characteristic sample (_geometric_wetting_rows). Returns the density and, per boundary, the chosen samples
        (shard-local rows in the order of _device_geometric_wetting_data, for the bounce-back correction).
        """
        rho_min = jax.lax.pmin(jnp.min(jnp.where(fluid_mask, rho, jnp.inf)), "x")
        rho_max = jax.lax.pmax(jnp.max(jnp.where(fluid_mask, rho, -jnp.inf)), "x")
        n = self.n_devices
        if n > 1:
            left = jax.lax.ppermute(rho[-halo:], "x", [(i, (i + 1) % n) for i in range(n)])  # last planes of the left neighbor
            right = jax.lax.ppermute(rho[:halo], "x", [(i, (i - 1) % n) for i in range(n)])  # first planes of the right one
        else:
            left, right = rho[-halo:], rho[:halo]
        channels = rho.shape[-1]
        flat = rho.reshape(-1, channels)
        if 4 * halo > rho.shape[0]:
            edge = jnp.concatenate([left, rho, right], axis=0)  # thin shard: the halo-padded density
        else:
            edge = jnp.concatenate([left, rho[: 2 * halo], rho[-2 * halo :], right], axis=0)
        edge = edge.reshape(-1, channels)
        rows = []
        for data in component_data:
            interior = self._geometric_wetting_rows(flat, {key: value[0] for key, value in data["interior"].items()}, rho_min, rho_max)
            near = self._geometric_wetting_rows(edge, {key: value[0] for key, value in data["edge"].items()}, rho_min, rho_max)
            rows.append((interior, near))
        # Every sample before any write: a scatter into rho that a later gather still reads would copy rho
        flat, rows = jax.lax.optimization_barrier((flat, rows))
        chosen = []
        for data, ((rho_interior, chosen_interior), (rho_edge, chosen_edge)) in zip(component_data, rows, strict=True):
            flat = flat.at[data["interior"]["wall_lin"][0]].set(rho_interior, mode="drop")
            flat = flat.at[data["edge"]["wall_lin"][0]].set(rho_edge, mode="drop")
            chosen.append(jnp.concatenate([chosen_interior, chosen_edge])[None])
        return flat.reshape(rho.shape), chosen

    def _geometric_wetting_global(self, rho, fluid_mask, component_data):
        """Global form of _geometric_wetting_m (indices into the whole density, see _localize_component_wetting_data)."""
        rho_min = jnp.min(jnp.where(fluid_mask, rho, jnp.inf))
        rho_max = jnp.max(jnp.where(fluid_mask, rho, -jnp.inf))
        flat = rho.reshape(-1, rho.shape[-1])
        rows = [self._geometric_wetting_rows(flat, data["interior"], rho_min, rho_max) for data in component_data]
        for data, (rho_wall, _) in zip(component_data, rows, strict=True):
            flat = flat.at[data["interior"]["wall_lin"]].set(rho_wall, mode="drop")
        return flat.reshape(rho.shape), [chosen for _, chosen in rows]

    def _build_local_geometric_wetting_kernels(self, scalar_spec):
        """One geometric-wetting kernel per component (see _geometric_wetting_m), None without wetted walls."""
        kernels = []
        for component_data, halo in zip(self.geometric_wetting_data, self._geometric_wetting_halos, strict=True):
            if not component_data:
                kernels.append(None)
            elif halo is None:
                kernels.append(jit(self._geometric_wetting_global))
            else:
                data_specs = tree_map(lambda array: PartitionSpec("x", *([None] * (array.ndim - 1))), component_data)
                kernels.append(
                    jit(
                        shard_map(
                            partial(self._geometric_wetting_m, halo=halo),
                            mesh=self.mesh,
                            in_specs=(scalar_spec, scalar_spec, data_specs),
                            out_specs=(scalar_spec, [PartitionSpec("x")] * len(component_data)),
                            check_vma=False,
                        )
                    )
                )
        return tuple(kernels)

    # Geometric wetting: bounce-back correction along the characteristics

    def _create_geometric_wall_correction_data(self):
        """
        Host-side setup of the bounce-back correction of geometric wetting (see _geometric_wall_correction_m).

        Full-way bounce-back returns to the fluid node x = b + c_i the population g_opp(i)(x) instead of the population g_i(b)
        that a fluid continued into the wall would send. At a contact line that defect is a kinetic error whose size depends on
        the relaxation rates (the magic parameter of the odd moments), and the geometric scheme, which copies the first fluid
        layer into the wall, amplifies it into a contact-angle error. The correction adds to the bounced population the same
        defect measured inside the fluid at the wall node's characteristic sample s (the point the wall density is taken from,
        on the same density iso-surface): D_i = G_i(s) - H_i(s + c_i), with G_i = g_i - 3 w_i c_i.j and H_i = g_opp(i) +
        3 w_i c_i.j, where j = rho u + F / 2 is the physical momentum. D is exact for any planar interface, vanishes in a
        uniform phase and for a uniform or linear shear flow (no slip is kept), and is evaluated at the sample's nearest node.
        It is applied on planar wall faces only (edges and corners of staircased walls keep plain bounce-back), and only near
        interfaces (see _GEOMETRIC_WETTING_ACTIVITY).

        The sample is chosen per time step among the characteristic samples (2 in 2D, the cone samples in 3D): the one with
        the largest (theta <= 90 degrees) or smallest density summed at its node and at the node twice as far along its
        characteristic. The second point, two layers into the fluid, keeps the density overshoot next to the interface from
        selecting the sample on the wrong side of the contact line.

        Nodes are referenced by their shard-local linear index, or, when they belong to another shard, by local_size + their
        row in a table of imported values (populations, density and momentum) exchanged once per time step.

        Returns
        -------
        (list): Per component, (arrays, directions) for _geometric_wall_correction_m, or None.
        """
        shape = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        dim, q, n_devices = self.dim, self.q, self.n_devices
        local_nx = self.nx // n_devices
        local_shape = self._local_shape()
        local_size = int(np.prod(local_shape))
        c = np.asarray(self.lattice.c, dtype=np.int64).T  # (q, dim)
        shape_array = np.asarray(shape, dtype=np.int64)

        def lin(points):
            return np.ravel_multi_index(tuple(np.moveaxis(np.mod(points, shape_array), -1, 0)), shape)

        result = []
        for component, (BC, samples) in enumerate(zip(self.BCs, self._geometric_wall_samples, strict=True)):
            bounceback = np.zeros(shape, dtype=bool)
            for bc in BC:
                if isinstance(bc, BounceBack):
                    bounceback[tuple(np.asarray(index) for index in bc.indices)] = True
            if not samples or not bounceback.any():
                result.append(None)
                continue
            # The defect is measured in ordinary fluid: not at solid nodes nor at the nodes of any boundary condition (inlets,
            # outlets, extrapolations, ...), whose populations the boundary conditions set
            constrained = self._create_component_solid_mask(BC) | bounceback
            for bc in BC:
                constrained[tuple(np.asarray(index) for index in bc.indices)] = True
            fluid = ~constrained.reshape(-1)
            walls = np.concatenate([np.asarray(indices, dtype=np.int64) for indices, _, _ in samples])
            theta = np.concatenate([np.asarray(theta, dtype=np.float64).reshape(-1) for _, theta, _ in samples])
            points = np.concatenate([
                np.stack([self._interpolation_points(point) for point in sample_points], axis=1) for _, _, sample_points in samples
            ])
            keep = bounceback[tuple(walls.T)]
            walls, theta, points = walls[keep], theta[keep], points[keep]
            if len(walls) == 0:
                result.append(None)
                continue
            n, K = points.shape[:2]

            sample = lin(np.round(points).astype(np.int64))  # (n, K) nearest nodes
            sample_ok = fluid[sample]
            # The score nodes of geometric wetting (same rule, same solid mask), for choosing samples without its choices
            score_nodes, score_ok = self._geometric_score_nodes(walls, points - walls[:, None, :], self._create_component_solid_mask(BC))
            score_near, far = lin(score_nodes[:, :, 0]), lin(score_nodes[:, :, 1])
            incoming = fluid[lin(walls[:, None, :] + c[None])]  # (n, q): b + c_i is fluid
            incoming[:, ~np.any(c != 0, axis=1)] = False
            # The correction holds where the wall is locally a plane (the defect is carried along it): only wall nodes on a
            # lattice-aligned face, whose fluid neighbors are exactly the directions to one side of an axis plane; edges and
            # corners of staircased walls keep plain bounce-back
            planar = np.zeros(n, dtype=bool)
            for axis in range(dim):
                for sign in (1, -1):
                    planar |= np.all(incoming == (sign * c[:, axis] > 0)[None], axis=1)
            incoming &= planar[:, None]
            directions = [i for i in range(q) if incoming[:, i].any()]
            sample_coordinates = np.stack(np.unravel_index(sample, shape), axis=-1)
            direction_rows, shifted, shifted_ok = {}, {}, {}
            for i in directions:
                rows = np.flatnonzero(incoming[:, i])
                direction_rows[i] = rows
                shifted[i] = lin(sample_coordinates[rows] + c[i])
                shifted_ok[i] = sample_ok[rows] & fluid[shifted[i]]

            # References: shard-local linear index, or local_size + row of the import table (block per source shard)
            wall_owner = walls[:, 0] // local_nx
            node_owner = lambda nodes: np.unravel_index(nodes, shape)[0] // local_nx

            def local_lin(nodes, device):
                coordinates = np.stack(np.unravel_index(nodes, shape), axis=-1)
                coordinates[..., 0] -= device * local_nx
                return np.ravel_multi_index(tuple(np.moveaxis(coordinates, -1, 0)), local_shape)

            # Nodes of other shards each shard reads: with populations (samples and shifted samples) or only the density and
            # momentum (the far score points); each block of the import tables holds what one shard sends
            empty = np.zeros(0, np.int64)
            remote = [[empty] * n_devices for _ in range(n_devices)]
            remote_macro = [[empty] * n_devices for _ in range(n_devices)]
            for d in range(n_devices):
                mine = wall_owner == d
                full_nodes = [sample[mine][sample_ok[mine]].ravel()]
                for i in directions:
                    sel = wall_owner[direction_rows[i]] == d
                    full_nodes.append(shifted[i][sel][shifted_ok[i][sel]].ravel())
                full_nodes = np.unique(np.concatenate(full_nodes))
                macro_nodes = np.setdiff1d(np.unique(np.concatenate([score_near[mine].ravel(), far[mine].ravel()])), full_nodes)
                for nodes, table in ((full_nodes, remote), (macro_nodes, remote_macro)):
                    owners = node_owner(nodes)
                    for e in range(n_devices):
                        if e != d:
                            table[d][e] = nodes[owners == e]
            width = max(1, max(len(rows) for per in remote for rows in per)) if n_devices > 1 else 1
            width_macro = max(1, max(len(rows) for per in remote_macro for rows in per)) if n_devices > 1 else 1
            # Export buffers, one contiguous block per destination shard (all_to_all splits them so): the nodes with
            # populations (width rows), and for the density and momentum those followed by the density-only nodes
            export = np.full((n_devices, n_devices * width), local_size, dtype=np.int32)
            export_macro = np.full((n_devices, n_devices * (width + width_macro)), local_size, dtype=np.int32)
            for d in range(n_devices):
                for e in range(n_devices):
                    rows, rows_macro = remote[d][e], remote_macro[d][e]
                    block = d * (width + width_macro)
                    if len(rows):  # shard e sends shard d these nodes
                        export[e, d * width : d * width + len(rows)] = local_lin(rows, e)
                        export_macro[e, block : block + len(rows)] = local_lin(rows, e)
                    if len(rows_macro):
                        export_macro[e, block + width : block + width + len(rows_macro)] = local_lin(rows_macro, e)
            # Imported rows (after the kernel reorders the density and momentum): the population nodes of each source shard,
            # local_size + e * width + k, then its density-only nodes, local_size + n_devices * width + e * width_macro + k
            sentinel = local_size + n_devices * (width + width_macro)  # out of range of everything

            def reference(nodes, device, ok=None):
                nodes = np.asarray(nodes)
                ref = np.full(nodes.shape, sentinel, dtype=np.int64)
                owners = node_owner(nodes)
                here = owners == device
                ref[here] = local_lin(nodes[here], device)
                for e in range(n_devices):
                    there = owners == e
                    if e == device or not np.any(there):
                        continue
                    full_rows, macro_rows = remote[device][e], remote_macro[device][e]
                    found = np.minimum(np.searchsorted(full_rows, nodes[there]), max(len(full_rows) - 1, 0))
                    in_full = (full_rows[found] == nodes[there]) if len(full_rows) else np.zeros(np.count_nonzero(there), bool)
                    found_macro = np.minimum(np.searchsorted(macro_rows, nodes[there]), max(len(macro_rows) - 1, 0))
                    value = np.where(in_full, local_size + e * width + found, local_size + n_devices * width + e * width_macro + found_macro)
                    ref[there] = value
                if ok is not None:
                    ref = np.where(ok, ref, sentinel)
                return ref.astype(np.int32)

            per_device = [np.flatnonzero(wall_owner == d) for d in range(n_devices)]
            n_wall = max(len(rows) for rows in per_device) + 1  # one padding wall, written out of range
            wall_lin = np.full((n_devices, n_wall), local_size, dtype=np.int32)
            wall_lin[:, -1] = np.iinfo(np.int32).max  # the padding wall (out of range of the fields)
            wall_select_max = np.zeros((n_devices, n_wall), dtype=bool)
            wall_sample_ok = np.zeros((n_devices, n_wall, K), dtype=bool)
            wall_sample = np.full((n_devices, n_wall, K), sentinel, dtype=np.int32)
            wall_score = np.full((n_devices, n_wall, K, 2), sentinel, dtype=np.int32)
            wall_slot = np.full(n, n_wall - 1, dtype=np.int64)
            # Row of each wall node in the chosen samples of geometric wetting on its shard (the device data of every boundary,
            # concatenated; within one, the wall rows of a shard come first, in order, see _device_geometric_wetting_data)
            wall_choice = np.zeros((n_devices, n_wall), dtype=np.int32)
            choice_row = np.zeros(n, dtype=np.int64)
            row_offset, start = 0, 0
            kept_position = np.cumsum(keep) - 1
            orders = self._geometric_wetting_orders[component] if self._geometric_wetting_halos[component] is not None else None
            for j, (indices, _, _) in enumerate(samples):
                count = len(indices)
                keep_j = keep[start : start + count]
                if orders is not None:
                    owner_j = np.asarray(indices)[:, 0] // local_nx
                    position = np.zeros(count, dtype=np.int64)
                    for d in range(n_devices):
                        order = orders[j][d]
                        rank = np.full(int(order.max(initial=-1)) + 2, -1, dtype=np.int64)
                        rank[order[order >= 0]] = np.flatnonzero(order >= 0)
                        mine_j = np.flatnonzero(owner_j == d)
                        position[mine_j] = rank[mine_j]
                    choice_row[kept_position[start : start + count][keep_j]] = row_offset + position[keep_j]
                    row_offset += len(orders[j][0])
                start += count
            for d, rows in enumerate(per_device):
                wall_slot[rows] = np.arange(len(rows))
                wall_choice[d, : len(rows)] = choice_row[rows]
                local = walls[rows].copy()
                local[:, 0] -= d * local_nx
                wall_lin[d, : len(rows)] = np.ravel_multi_index(tuple(local.T), local_shape)
                wall_select_max[d, : len(rows)] = theta[rows] <= np.pi / 2
                wall_sample_ok[d, : len(rows)] = score_ok[rows]
                wall_sample[d, : len(rows)] = reference(sample[rows], d, sample_ok[rows])
                wall_score[d, : len(rows)] = reference(np.stack([score_near[rows], far[rows]], axis=-1), d)

            direction_slots, direction_shifted, direction_nodes = {}, {}, {}
            for i in directions:
                rows = direction_rows[i]
                owners = wall_owner[rows]
                counts = [np.count_nonzero(owners == d) for d in range(n_devices)]
                width_i = max(1, max(counts))
                slots = np.full((n_devices, width_i), n_wall - 1, dtype=np.int32)
                shifted_ref = np.full((n_devices, width_i, K), sentinel, dtype=np.int32)  # sentinel: no correction
                for d in range(n_devices):
                    sel = owners == d
                    slots[d, : counts[d]] = wall_slot[rows[sel]]
                    shifted_ref[d, : counts[d]] = reference(shifted[i][sel], d, shifted_ok[i][sel])
                nodes = np.take_along_axis(wall_lin, slots, axis=1).astype(np.int64)
                # Distinct out-of-range nodes for the padding, so that the scatter of each direction has unique indices
                padding = slots == n_wall - 1
                nodes[padding] = local_size + np.cumsum(padding, axis=1)[padding]
                direction_nodes[str(i)] = self._distribute_local(nodes.astype(np.int32), jnp.int32)
                direction_slots[str(i)] = self._distribute_local(slots, jnp.int32)
                direction_shifted[str(i)] = self._distribute_local(shifted_ref, jnp.int32)

            arrays = {
                "export": self._distribute_local(export, jnp.int32),
                "export_macro": self._distribute_local(export_macro, jnp.int32),
                "wall_choice": self._distribute_local(wall_choice, jnp.int32),
                "fluid": self.geometric_fluid_mask[component],
                "wall_select_max": self._distribute_local(wall_select_max, jnp.bool_),
                "wall_sample_ok": self._distribute_local(wall_sample_ok, jnp.bool_),
                "wall_sample": self._distribute_local(wall_sample, jnp.int32),
                "wall_score": self._distribute_local(wall_score, jnp.int32),
                "direction_slots": direction_slots,
                "direction_shifted": direction_shifted,
                "direction_nodes": direction_nodes,
            }
            logger.info(
                f"Geometric wall correction: {n} wall nodes; imported per shard: {n_devices * width if n_devices > 1 else 0} nodes "
                f"(populations) and {n_devices * width_macro if n_devices > 1 else 0} (density and momentum only)."
            )
            result.append((arrays, tuple(directions)))
        return result

    def _geometric_wall_correction_m(self, fields, macroscopic, arrays, chosen=None, directions=()):
        """
        Shard-local bounce-back correction of geometric wetting (see _create_geometric_wall_correction_data).

        fields are the post-collision populations (bounce-back applied at the wall nodes), macroscopic the density and the
        physical momentum components (rho, j_x, ...) the populations were collided with. Returns the fields with D added
        to the populations each wall node sends into the fluid.
        """
        q, dim = self.q, self.dim
        c = np.asarray(self.lattice.c, dtype=np.float64)  # (dim, q)
        w = np.asarray(self.lattice.w, dtype=np.float64)
        opposite = np.asarray(self.lattice.opp_indices)
        columns = [f.reshape(-1) for f in fields]
        local_size = columns[0].shape[0]
        # Density and momentum interleaved, (nodes, dim + 1): one gather per node
        macro = jnp.stack([m.reshape(-1) for m in macroscopic], axis=-1)
        imported = imported_macro = None
        if self.n_devices > 1:
            export = arrays["export"][0]
            sent = jnp.stack([column.at[export].get(mode="fill", fill_value=0.0) for column in columns])  # (q, rows)
            imported = jax.lax.all_to_all(sent, "x", 1, 1, tiled=True)
            imported_macro = jax.lax.all_to_all(macro.at[arrays["export_macro"][0]].get(mode="fill", fill_value=0.0), "x", 0, 0, tiled=True)
            # Received per source shard: its population nodes, then its density-only nodes; reordered to all population nodes
            # first (the references of _create_geometric_wall_correction_data)
            width = imported.shape[1] // self.n_devices
            per_source = imported_macro.reshape(self.n_devices, -1, dim + 1)
            imported_macro = jnp.concatenate([per_source[:, :width].reshape(-1, dim + 1), per_source[:, width:].reshape(-1, dim + 1)])

        def fetch(column, ref):
            """Population column at references (shard-local nodes or imported rows)."""
            value = columns[column].at[ref].get(mode="fill", fill_value=0.0)
            if imported is None:
                return value
            return jnp.where(ref < local_size, value, imported[column].at[ref - local_size].get(mode="fill", fill_value=0.0))

        def fetch_macro(ref):
            """(rho, j) at references, (..., dim + 1)."""
            value = macro.at[ref].get(mode="fill", fill_value=0.0)
            if imported_macro is None:
                return value
            return jnp.where((ref < local_size)[..., None], value, imported_macro.at[ref - local_size].get(mode="fill", fill_value=0.0))

        sentinel = local_size + (0 if imported_macro is None else imported_macro.shape[0])
        ok = arrays["wall_sample_ok"][0]
        n_wall, K = ok.shape
        score_refs = arrays["wall_score"][0]  # (n, K, 2): nearest node and the node twice as far, per sample
        if chosen is None:
            # Uniform-phase test as in geometric wetting (_geometric_wetting_rows), on the samples' nearest nodes
            near = fetch_macro(score_refs[..., 0])[..., 0]  # (n, K)
            fluid, density = arrays["fluid"].reshape(-1), macro[:, 0]  # the mask is a sharded field (no device axis)
            rho_range = jax.lax.pmax(jnp.max(jnp.where(fluid, density, -jnp.inf)), "x") - jax.lax.pmin(
                jnp.min(jnp.where(fluid, density, jnp.inf)), "x"
            )
            spread = jnp.max(jnp.where(ok, near, -jnp.inf), axis=1) - jnp.min(jnp.where(ok, near, jnp.inf), axis=1)
            active = spread > self._GEOMETRIC_WETTING_ACTIVITY * rho_range
        else:
            # The uniform-phase test geometric wetting made (its shard-local rows of these wall nodes)
            active = chosen[0][arrays["wall_choice"][0]] >= 0
        # Near interfaces, the sample on the iso-surface of the wall node: the largest (theta <= 90) or smallest density
        # summed at its nearest node and at the node twice as far along the characteristic (robust to the density
        # overshoot next to an interface). Out-of-range references elsewhere: no reads
        values = fetch_macro(jnp.where(active[:, None, None], score_refs, sentinel))[..., 0]
        score = jnp.sum(values, axis=-1)
        score = jnp.where(arrays["wall_select_max"][0][:, None], score, -score)
        chosen = jnp.where(active, self._best_sample(jnp.where(ok, score, -jnp.inf)), -1)
        # Wall nodes in a uniform phase (chosen -1) get no correction: their references are out of range (no memory read)
        chosen_sample = self._pick_sample(arrays["wall_sample"][0], chosen, fill=sentinel)
        sample_macro = fetch_macro(chosen_sample)  # (n, dim + 1), shared by the directions

        # All defects first (gathers from the unmodified fields), then the scatters: a scatter into a field that a later
        # gather still reads would copy it
        updates = []
        for i in directions:
            key, o = str(i), int(opposite[i])
            slots = arrays["direction_slots"][key][0]
            here = chosen_sample[slots]
            there = self._pick_sample(arrays["direction_shifted"][key][0], chosen[slots], fill=sentinel)
            # Python floats (weakly typed): the momentum term stays in the compute precision
            ci = [(axis, float(c[axis, i])) for axis in range(dim) if c[axis, i] != 0]
            weight = 3.0 * float(w[i])
            cj = lambda macro_values: weight * sum(macro_values[:, 1 + axis] * value for axis, value in ci)
            # G_i(s) - H_i(s + c_i), with G_i = g_i - 3 w_i c_i.j and H_i = g_opp(i) + 3 w_i c_i.j
            defect = fetch(i, here) - cj(sample_macro[slots]) - fetch(o, there) - cj(fetch_macro(there))
            updates.append((i, arrays["direction_nodes"][key][0], jnp.where(there < sentinel, defect, 0.0)))
        fields, values = jax.lax.optimization_barrier((tuple(fields), [(nodes, defect) for _, nodes, defect in updates]))
        fields = list(fields)
        columns = [f.reshape(-1) for f in fields]
        for (i, _, _), (nodes, defect) in zip(updates, values, strict=True):
            fields[i] = columns[i].at[nodes].add(defect.astype(columns[i].dtype), mode="drop", unique_indices=True).reshape(fields[i].shape)
        return tuple(fields)

    def _build_geometric_wall_correction_kernels(self):
        """shard_map'd kernels of the bounce-back correction per corrected component, (choosing the samples itself, with the
        samples geometric wetting chose), None otherwise."""
        spec = PartitionSpec("x")
        kernels = []
        for data in self.geometric_wall_correction_data:
            if data is None:
                kernels.append(None)
                continue
            function = partial(self._geometric_wall_correction_m, directions=data[1])
            kernels.append((
                shard_map(function, mesh=self.mesh, in_specs=(spec, spec, spec), out_specs=spec, check_vma=False),
                shard_map(function, mesh=self.mesh, in_specs=(spec, spec, spec, spec), out_specs=spec, check_vma=False),
            ))
        return kernels

    def _apply_geometric_wall_correction(self, component, macroscopic, choices, target, previous):
        """Post-collision hook of _prepare_streaming: the bounce-back correction of one component (see
        _geometric_wall_correction_m), with the samples geometric wetting chose (halo form) when it ran."""
        arrays = self.geometric_wall_correction_data[component][0]
        use_choices = choices is not None and self._geometric_wetting_halos[component] is not None
        chosen = jnp.concatenate(choices, axis=1) if use_choices else None
        kernel = self._geometric_wall_correction_kernels[component][1 if use_choices else 0]
        fields = kernel(tuple(target.fields), macroscopic, arrays) if chosen is None else kernel(tuple(target.fields), macroscopic, arrays, chosen)
        return PopulationState(fields, target.lo, target.hi)

    def _wetting_parameter_at_indices(self, value, indices):
        """Convert scalar, per-node, or full-domain wetting data to one scalar row per solid node."""
        value = np.asarray(value)
        count = len(indices)
        if value.size == 1:
            return np.full((count, 1), value.reshape(()).item(), dtype=value.dtype)
        spatial_shape = (self.nx, self.ny) if self.dim == 2 else (self.nx, self.ny, self.nz)
        if value.shape[: self.dim] == spatial_shape:
            selected = value[tuple(indices.T)]
            return np.asarray(selected).reshape(count, -1)
        if value.shape[0] == count:
            return value.reshape(count, -1)
        raise ValueError("Wetting parameters must be scalar, per-boundary-node arrays, or full-domain fields.")

    def _make_local_improved_wetting_data(self):
        """Build static local wetting data per component and boundary condition."""
        data_by_component = []
        for component_bcs in self.BCs:
            component_data = []
            for bc in component_bcs:
                if not self._is_wetting_boundary_condition(bc) or bc.theta is None or bc.is_dynamic:
                    component_data.append(None)
                    continue
                indices = np.asarray(self._get_solid_indices(bc), dtype=np.int32).T
                if len(indices) == 0:
                    component_data.append(None)
                    continue
                theta = self._wetting_parameter_at_indices(bc.theta, indices)
                phi = self._wetting_parameter_at_indices(bc.phi, indices)
                delta_rho = self._wetting_parameter_at_indices(bc.delta_rho, indices)
                local_indices, (local_theta, local_phi, local_delta_rho) = self._split_local_indices(indices, theta, phi, delta_rho)
                component_data.append((
                    self._distribute_local(local_indices, jnp.int32),
                    self._distribute_local(local_theta, self.precision_policy.compute_dtype),
                    self._distribute_local(local_phi, self.precision_policy.compute_dtype),
                    self._distribute_local(local_delta_rho, self.precision_policy.compute_dtype),
                ))
            data_by_component.append(component_data)
        return data_by_component

    @partial(jit, static_argnums=(0,))
    def compute_average_density(self, rho_tree):
        """
        Compute component densities averaged over neighboring fluid nodes, using the scalar solid mask and the
        denominator cached once at construction time (self.average_density_denominator), instead of streaming a
        q-channel mask and dividing every timestep.

        Parameters
        ----------
        rho_tree (pytree of jax.Array): Component density fields with shape ``(nx, ny, 1)`` in 2D or ``(nx, ny, nz, 1)``
            in 3D.

        Returns
        -------
        pytree of jax.Array
            Averaged component density fields with the same shapes as the inputs.

        Notes
        -----
        Solid nodes with no fluid neighbor (the second layer of a thick solid) have a zero denominator. The fourth-order
        pseudopotential gradient reads psi two cells into the solid, so instead of 0/0 = NaN these nodes take the average of
        the neighboring first-layer values (denominator average_density_second_layer); deeper nodes, which no stencil reads,
        keep their own density. Nodes with a fluid neighbor are unchanged.
        """
        if self.scalar_neighbor_sum is None or self.average_density_denominator is None:
            return rho_tree

        out = []
        for rho, solid_mask, denominator, second_layer in zip(
            rho_tree,
            self.solid_mask_streamed,
            self.average_density_denominator,
            self.average_density_second_layer,
            strict=True,
        ):
            if denominator is None:
                out.append(rho)
                continue
            has_fluid_neighbor = denominator > 0
            first = self.scalar_neighbor_sum(rho * (1 - solid_mask)) / jnp.where(has_fluid_neighbor, denominator, 1)
            first = jnp.where(has_fluid_neighbor, first, 0)
            has_first_layer_neighbor = second_layer > 0
            second = self.scalar_neighbor_sum(first) / jnp.where(has_first_layer_neighbor, second_layer, 1)
            out.append(jnp.where(has_fluid_neighbor, first, jnp.where(has_first_layer_neighbor, second, rho)))
        return out

    def apply_contact_angle(self, rho_tree):
        """
        Apply prescribed contact angles to wall-node densities.

        For the geometric scheme, only theta is used. The 2D path interpolates two characteristic samples and chooses the appropriate extrema.
        The 3D path interpolates multiple samples on the contact-angle cone and chooses the maximum density for theta <= pi / 2 or the minimum
        density for theta > pi / 2. For improved virtual density, theta is used with phi and delta_rho according to the selected wettability branch.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Density field.

        Returns
        -------
        (pytree of jax.numpy.ndarray) Density field with adjusted contact angle values at the boundary nodes.

        References
        ----------
        1. Li, Q., Yu, Y. & Luo, K. H. "Implementation of contact angles in pseudopotential lattice Boltzmann simulations with
        curved boundaries." Phys. Rev. E 100, 053313 (2019).
        2. Fei, Linlin, Feifei Qin, Jianlin Zhao, Dominique Derome, and Jan Carmeliet. “Lattice Boltzmann Modelling of
        Isothermal Two-Component Evaporation in Porous Media.”
        Journal of Fluid Mechanics 955 (January 2023): A18.
        3. Wang, Lei, Hai-bo Huang, and Xi-Yun Lu. “Scheme for Contact Angle and Its Hysteresis in a Multiphase Lattice
        Boltzmann Method.” Physical Review E 87, no. 1 (2013): 013301.
        """
        if self.wetting_formulation is None or not any(self._has_wetting_bc):
            return rho_tree

        if self.wetting_formulation == "improved_virtual_density":
            rho_ave_tree = self.compute_average_density(rho_tree)

            def set_contact_angle(rho, rho_ave, BC, local_component_data):
                if self.global_min_max is not None:
                    rho_min, rho_max = self.global_min_max(rho)
                else:
                    rho_min = jnp.min(rho)
                    rho_max = jnp.max(rho)
                for bc_index, bc in enumerate(BC):
                    if isinstance(
                        bc, (BounceBackHalfway, BounceBack, BounceBackMoving, InterpolatedBounceBackBouzidi, InterpolatedBounceBackDifferentiable)
                    ):
                        if bc.theta is not None:
                            local_data = local_component_data[bc_index]
                            if local_data is not None:
                                local_indices, local_theta, local_phi, local_delta_rho = local_data
                                rho = self.local_improved_wetting(rho, rho_ave, local_indices, local_theta, local_phi, local_delta_rho)
                            else:
                                indices = self._get_solid_indices(bc)
                                rho = rho.at[indices].set(
                                    (bc.theta <= jnp.pi / 2) * (bc.phi * rho_ave[indices])
                                    + (bc.theta > jnp.pi / 2) * (rho_ave[indices] - bc.delta_rho)
                                )
                        rho = jnp.clip(rho, min=rho_min, max=rho_max)
                return rho

            return [
                set_contact_angle(rho, rho_ave, BC, local_component_data) if has_wetting_bc else rho
                for rho, rho_ave, BC, local_component_data, has_wetting_bc in zip(
                    rho_tree,
                    rho_ave_tree,
                    self.BCs,
                    self.local_improved_wetting_data,
                    self._has_wetting_bc,
                    strict=True,
                )
            ]
        elif self.wetting_formulation == "geometric":
            # One kernel per component (see _geometric_wetting_m); the chosen samples go to the bounce-back correction
            result, choices = [], []
            for rho, fluid_mask, component_data, kernel in zip(
                rho_tree, self.geometric_fluid_mask, self.geometric_wetting_data, self.local_geometric_wetting, strict=True
            ):
                if kernel is None:
                    result.append(rho)
                    choices.append(None)
                    continue
                rho, chosen = kernel(rho, fluid_mask, component_data)
                result.append(rho)
                choices.append(chosen)
            if self._wetting_choice_export is not None:
                self._wetting_choice_export.append(choices)
            return result

        return rho_tree

    @partial(jit, static_argnums=(0,))
    def collision(self, fin_tree, T=None):
        """
        Apply collision step of LBM. The optional temperature field T is used
        by thermal EOS variants (see compute_pressure).
        """
        pass

    def compute_ff_greens_function(self):
        """
        Define the fluid-fluid interaction force Green's function used to compute interaction phase-phase interaction forces.

        The interaction coefficient between k^th and kprime^th component: self.gkkprime[k, kprime]
        During computation, this value is multiplied with corresponding g_kkprime value to get the Green's function:
        G_kkprime = self.g_kk[k, k_prime] * self.G_ff

        G_kkprime(x, x') = g1 * g_kkprime,  if |x - x'| = 1
                         = g2 * g_kkprime,  if |x - x'| = sqrt(2)
                         = 0,               otherwise

        Here d is the dimension of problem and x' are the neighboring points.

        Some examples values could be:
        For D2Q9:
            g1 = 1/3 and g2 = 1/12
        For D3Q19
            g1 = 1/6 and g2 = 1/12

        Returns
        -------
        G_ff (jax.numpy.ndarray): Dimension: (q, )
        """
        c = np.array(self.lattice.c).T
        G_ff = np.zeros((self.q,), dtype=np.float64)
        cl = np.linalg.norm(c, axis=-1)
        if isinstance(self.lattice, LatticeD2Q9):
            g1 = 1 / 3
            g2 = 1 / 12
            G_ff[np.isclose(cl, 1.0, atol=1e-6)] = g1
            G_ff[np.isclose(cl, jnp.sqrt(2.0), atol=1e-6)] = g2
        elif isinstance(self.lattice, LatticeD3Q19):
            g1 = 1 / 6
            g2 = 1 / 12
            G_ff[np.isclose(cl, 1.0, atol=1e-6)] = g1
            G_ff[np.isclose(cl, jnp.sqrt(2.0), atol=1e-6)] = g2
        else:
            raise NotImplementedError("Please define Green's function for D3Q27 lattice by modifying compute_ff_greens_function.")
        return jnp.array(G_ff, dtype=self.precision_policy.compute_dtype)

    def assign_fields_sharded(self):
        """
        This function is used to initialize pytree of the distribution arrays using the initial velocities and velocity defined in self.initialize_macroscopic_fields function.
        To do this, function first uses the initialize_macroscopic_fields function to get the initial values of rho (rho0) and velocity (u0).

        If this function is not modified then, the distribution pytree is initialized with density value of 1.0 everywhere and velocity of 0.0 everywhere

        The distribution is initialized with rho0 and u0 values, using the self.equilibrium function.

        Returns
        -------
        f: pytree of distributed JAX array of shape: (self.nx, self.ny, self.q) for 2D and (self.nx, self.ny, self.nz, self.q) for 3D.
        """
        rho0_tree, u0_tree = self.initialize_macroscopic_fields()
        if self.dim == 2:
            shape = (self.nx, self.ny, self.q)
        if self.dim == 3:
            shape = (self.nx, self.ny, self.nz, self.q)
        f_tree = []
        if rho0_tree is not None and u0_tree is not None:
            assert len(rho0_tree) == self.n_components, "The initial density values for all components must be provided"

            assert len(u0_tree) == self.n_components, "The initial velocity values for all components must be provided."

            for i in range(self.n_components):
                rho0, u0 = rho0_tree[i], u0_tree[i]
                rho0 = self.precision_policy.cast_to_compute(rho0)
                u0 = self.precision_policy.cast_to_compute(u0)
                f_tree.append(self.initialize_populations(rho0, u0))
        else:
            for i in range(self.n_components):
                f_tree.append(self.distributed_array_init(shape, self.precision_policy.output_dtype, init_val=self.w))
        return f_tree

    @partial(jit, static_argnums=(0,), inline=True)
    def update_macroscopic(self, f_tree):
        """
        update_macroscopic from base.py extended to pytrees.

        Parameters
        ----------
        f_tree (pytree of jax.numpy.ndarray): Distribution field.

        Returns
        -------
        rho_tree (pytree of jax.numpy.ndarray): Density field.
        u_tree (pytree of jax.numpy.ndarray): Velocity field.
        """
        f_tree = tree_map(lambda f: self.precision_policy.cast_to_compute(f), f_tree)
        rho_tree, u_tree = [], []
        for f in f_tree:
            rho, momentum = lattice_density_momentum(self.lattice.c, f)
            rho_tree.append(rho)
            u_tree.append(momentum / rho)  # Component velocity
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,), inline=True)
    def macroscopic_velocity(self, f_tree, rho_tree, T=None):
        """
        macroscopic_velocity computes the velocity and incorporates forces into velocity for Exact Difference Method (EDM) (used for SRT and MRT collision) models
        and the consistent forcing scheme developed by LinLin Fei et. al (for Cascaded LBM). This is used for post-processing only and not for equilibrium distribution computation.

        Parameters
        ----------
        f_tree (pytree of jax.numpy.ndarray): Distribution field.

        rho_tree (pytree of jax.numpy.ndarray): Density field.

        T (jax.numpy.ndarray, optional): Temperature field, required when the EOS is thermal.

        Returns
        -------
        u_tree (pytree of jax.numpy.ndarray): Velocity field.
        """
        u_tree = [
            lattice_density_momentum(self.lattice.c, self.precision_policy.cast_to_compute(f))[1] / rho
            for f, rho in zip(f_tree, rho_tree, strict=True)
        ]
        F_tree = self.compute_force(rho_tree, T=T)
        return tree_map(lambda rho, u, F: u + 0.5 * F / rho, rho_tree, u_tree, F_tree)

    @partial(jit, static_argnums=(0,))
    def compute_total_density(self, rho_tree):
        """
        Compute the total density using component velocity and density values.

        Parameters
        ----------
        rho_tree (Pytree of jax.numpy.ndarray): Density field.

        Returns
        -------
        (jax.numpy.ndarray): Total density field.
        """
        return reduce(operator.add, rho_tree)

    @partial(jit, static_argnums=(0,))
    def compute_total_velocity(self, rho_tree, u_tree):
        """
        Compute the total velocity using component velocity and density values.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Density field

        u_tree (pytree of jax.numpy.ndarray): Velocity field

        Returns
        -------
        (jax.numpy.ndarray): Total velocity field.
        """
        n = reduce(operator.add, tree_map(lambda rho, u: rho * u, rho_tree, u_tree))
        d = reduce(operator.add, rho_tree)
        return n / d

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree=None, T=None):
        """
        Generalized function for computing pressure. By default it uses equation
        of state but it can be modified if the pseudopotential is computed using
        a different method.

        For a thermal EOS (temperature_field_type == "thermal") the local
        temperature field T must be provided and the pressure is evaluated with
        EOS_thermal, coupling the flow to the temperature solver in thermal.py.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Density field.

        psi_tree (pytree of jax.numpy.ndarray): Pseudopotential field.

        T (jax.numpy.ndarray, optional): Temperature field, required when the EOS is thermal.

        Returns
        -------
        (pytree of jax.numpy.ndarray): Pressure field.
        """
        if self.eos.temperature_field_type == "thermal":
            if T is None:
                raise ValueError("Temperature field T must be passed through step/collision when using a thermal EOS.")
            return self.eos.EOS_thermal(rho_tree, T)
        return self.eos.EOS(rho_tree)

    @partial(jit, static_argnums=(0,))
    def compute_total_pressure(self, p_tree, rho_tree=None):
        """
        Compute the total combined pressure from all components.

        Parameters
        ----------
        p_tree (pytree of jax.numpy.ndarray): Pressure field.

        rho_tree (pytree of jax.numpy.ndarray, default=None): Density field.

        Returns
        -------
        (jax.numpy.ndarray): Total pressure field.
        """
        return reduce(operator.add, p_tree)

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree, T=None):
        """
        Compute the effective mass (psi) in the decoupled mesh space, which is required for computing interaction forces.
        The psi values are obtained using the corresponding EOS: psi = sqrt(2 * (lambda_^2 * p - cs2 * rho) / g_kk). This function can be
        overloaded to handle cases where one or more component does not have EOS.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Density field.

        T (jax.numpy.ndarray, optional): Temperature field, required when the EOS is thermal.

        Returns
        -------
        psi_tree (pytree of jax.numpy.ndarray): Pseudopotential field.
        """
        rho_tree = tree_map(lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree)
        p_tree = self.compute_pressure(rho_tree, T=T)
        return [
            jnp.sqrt((2.0 / G) * (lam * lam * p - self.lattice.cs2 * rho))
            for lam, p, rho, G in zip(self.lambda_, p_tree, rho_tree, self._g_host.diagonal().tolist(), strict=True)
        ]

    # Compute the force using the effective mass (psi)
    @partial(jit, static_argnums=(0,))
    def _compute_force_fields(self, rho_tree, T=None):
        """Return force plus the contact-angle-adjusted effective mass used to construct it."""
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree = self.compute_potential(rho_tree, T=T)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree)
        # fluid_solid_force = self.compute_fluid_solid_force(rho_tree)
        if self.body_force is not None:
            force_tree = tree_map(lambda ff, rho: ff + self.body_force * rho, fluid_fluid_force, rho_tree)
        else:
            force_tree = fluid_fluid_force
        if self.wetting_formulation == "geometric" and any(self._has_wetting_bc):
            force_tree = tree_map(lambda force, force_mask: force * force_mask, force_tree, self.geometric_force_mask)
        return force_tree, psi_tree

    # Compute the force using the effective mass (psi)
    @partial(jit, static_argnums=(0,))
    def compute_force(self, rho_tree, T=None):
        """
        Compute the force acting on each component(fluid). This includes fluid-fluid, fluid-solid, and body forces.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Density field.

        T (jax.numpy.ndarray, optional): Temperature field, required when the EOS is thermal.

        Returns
        -------
        fluid_fluid_force (pytree of jax.numpy.ndarray): Total force field.
        """
        force_tree, _ = self._compute_force_fields(rho_tree, T=T)
        return force_tree

    @partial(jit, static_argnums=(0,))
    def compute_fluid_fluid_force(self, psi_tree):
        """
        Compute the fluid-fluid interaction force from the effective mass (psi):

            F_k = -psi_k * sum_k' g_kk' * grad(psi_k'),

        with grad the fourth-order isotropic EFM gradient (scalar_gradient). For a single component with g = -1 this is
        F = -grad(lambda_^2 * p - cs2 * rho), the mesh-space force of Qin et al. (2022) / Yang & Qin (2024).

        Parameters
        ----------
        psi_tree (pytree of jax.numpy.ndarray): Effective mass field.

        Returns
        -------
        (pytree of jax.numpy.ndarray): Fluid-fluid interaction forces.

        Notes
        -----
        Gradients are computed once per source component, outside the per-output-component combine
        (scalar_gradient is a shard_map'd call and must not be invoked from inside vmap).
        """
        gradient_tree = [self.scalar_gradient(psi) if used else None for psi, used in zip(psi_tree, self._psi_stencil_components, strict=True)]
        return self._fluid_fluid_force_from_gradients(psi_tree, gradient_tree)

    def _fluid_fluid_force_from_gradients(self, psi_tree, gradient_tree):
        """F_k = -psi_k * sum_k' g_kk' * grad(psi_k') from precomputed gradients."""
        force_tree = []
        for output in range(self.n_components):
            terms = [
                float(self._g_host[output, source]) * gradient_tree[source]
                for source in range(self.n_components)
                if self._psi_interactions[output][source]
            ]
            if terms:
                force = -psi_tree[output] * reduce(operator.add, terms)
            else:
                force = jnp.zeros((*psi_tree[output].shape[:-1], self.dim), dtype=self.precision_policy.compute_dtype)
            force_tree.append(force)
        return force_tree

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, f_postcollision_tree, feq_tree, rho_tree, u_tree, T=None):
        """
        Modified version of the apply_force defined in LBMBase to account for modified force.

        Adds the force contribution using the exact-difference method (Kupershtokh), computing
        feq(rho, u + F/rho) - feq(rho, u) directly from cu, dcu and delta_usqr instead of building a second full
        equilibrium distribution and subtracting feq_tree from it.

        Parameters
        ----------
        f_postcollision_tree (pytree of jax.numpy.ndarray): Post-collision distribution field.

        feq_tree (pytree of jax.numpy.ndarray): Equilibrium distribution functions. Unused - kept for interface
            compatibility with existing callers, since the compact difference formula only needs rho, u and F.

        rho_tree (pytree of jax.numpy.ndarray): Density field.

        u_tree (pytree of jax.numpy.ndarray): Velocity field.

        T (jax.numpy.ndarray, optional): Temperature field, required when the EOS is thermal.

        Returns
        -------
        f_postcollision_tree (pytree of jax.numpy.ndarray): The post-collision distribution field with the force applied.

        References
        ----------
        1. Kupershtokh, A. (2004). New method of incorporating a body force term into the lattice Boltzmann
        equation. In Proceedings of the 5th International EHD Workshop (pp. 241-246). University of Poitiers.
        """
        F_tree = self.compute_force(rho_tree, T=T)
        du_tree = tree_map(lambda F, rho: F / rho, F_tree, rho_tree)

        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cu_tree = tree_map(lambda u: 3.0 * jnp.dot(u, c), u_tree)
        dcu_tree = tree_map(lambda du: 3.0 * jnp.dot(du, c), du_tree)
        delta_usqr_tree = tree_map(
            lambda u, du: 1.5 * (2.0 * jnp.sum(u * du, axis=-1, keepdims=True) + jnp.sum(jnp.square(du), axis=-1, keepdims=True)),
            u_tree,
            du_tree,
        )
        delta_feq_tree = tree_map(
            lambda rho, cu, dcu, delta_usqr: rho * self.w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr),
            rho_tree,
            cu_tree,
            dcu_tree,
            delta_usqr_tree,
        )
        return tree_map(lambda f_postcollision, delta_feq: f_postcollision + delta_feq, f_postcollision_tree, delta_feq_tree)

    def _make_streaming_bc_data(self):
        """Boundary data of the time step, one per component (see LBMBase.prepare_streaming)."""
        return [self._streaming_bcs(BCs, mask is not None) for BCs, mask in zip(self.BCs, self.bounceback_mask, strict=True)]

    @partial(jit, static_argnums=(0,))
    def state_from_populations(self, f_tree):
        """The per-component states whose streamed populations are f_tree (see LBMBase.state_from_populations)."""
        return [LBMBase.state_from_populations.__wrapped__(self, f) for f in f_tree]

    @partial(jit, static_argnums=(0,))
    def poststreaming(self, state_tree):
        """Post-streaming populations of the per-component states, as (..., q) arrays."""
        return [self.stream(state) for state in state_tree]

    @partial(jit, static_argnums=(0, 3))
    def step(self, state_tree, timestep, return_fpost=False, T=None):
        """
        One time step of every component: pull streaming, collision and the boundary conditions (see LBMBase.step). The
        interaction force is computed from the streamed populations of all components inside the collision.

        Parameters
        ----------
        state_tree (list of PopulationState): Post-collision populations of the previous step, per component.

        timestep (int): Current timestep.

        return_fpost (bool): Also return the post-collision populations (list of (..., q) arrays).

        T (jax.numpy.ndarray, optional): Temperature field, required when the EOS is thermal (see the hybrid thermal
        solver in thermal.py).

        Returns
        -------
        state_tree (list of PopulationState): The states after this step.

        f_postcollision_tree (list of jax.numpy.ndarray or None): Post-collision populations, or None.
        """
        correction = getattr(self, "geometric_wall_correction_data", None) or [None] * self.n_components
        corrected = any(data is not None for data in correction)
        f_poststreaming_tree = [self.stream(state) for state in state_tree]
        collision = self.collision
        if corrected and "collision" not in self.__dict__ and hasattr(type(self).collision, "__wrapped__"):
            # Traced inline (not as its own jit), so that the fields it exports belong to this trace
            collision = partial(type(self).collision.__wrapped__, self)
        self._wall_correction_export = [] if corrected else None
        self._wetting_choice_export = [] if corrected else None
        try:
            f_postcollision_tree = collision(f_poststreaming_tree, T=T)
            exported, exported_choices = self._wall_correction_export, self._wetting_choice_export
        finally:
            self._wall_correction_export = self._wetting_choice_export = None
        # The characteristic samples geometric wetting chose this step (None per component where the correction chooses)
        choices = exported_choices[-1] if exported_choices else [None] * self.n_components
        f_postcollision_tree = self.post_collision(f_postcollision_tree, f_poststreaming_tree, timestep)
        macroscopic = [None] * self.n_components
        if corrected:
            if exported:
                macroscopic = exported[0]
            else:
                # Collision without the export (e.g. a user override): density and mean of the pre- and post-collision momenta
                macroscopic = []
                for f, g in zip(f_poststreaming_tree, f_postcollision_tree, strict=True):
                    c = jnp.asarray(self.lattice.c, dtype=f.dtype)
                    j = 0.5 * (jnp.einsum("...q,dq->...d", f, c) + jnp.einsum("...q,dq->...d", g.astype(f.dtype), c))
                    macroscopic.append(tuple([jnp.sum(f, axis=-1)] + [j[..., axis] for axis in range(self.dim)]))
            # Materialized once: otherwise XLA fuses these expressions of the populations and the force into the
            # correction's gathers and recomputes them (with the psi-gradient stencil) at every gathered node
            macroscopic = list(jax.lax.optimization_barrier(tuple(m if m is not None else () for m in macroscopic)))
        results = [
            self._prepare_streaming(
                f_postcollision,
                state,
                timestep,
                bc_data,
                postcollision_hook=None
                if data is None
                else partial(self._apply_geometric_wall_correction, component, macroscopic[component], choices[component]),
            )
            for component, (f_postcollision, state, bc_data, data) in enumerate(
                zip(f_postcollision_tree, state_tree, self.streaming_bc_data, correction, strict=True)
            )
        ]
        new_state_tree = [state for state, _ in results]
        return new_state_tree, ([jnp.stack(fields, axis=-1) for _, fields in results] if return_fpost else None)

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
        f_tree (pytree of jax.numpy.ndarray): Distribution function after t_max timesteps.
        """
        f_tree = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                # Assert that the checkpoint manager is not None
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
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max  = {t_max}). Consider increasing t_max.")

        # The loop advances the per-component states (post-collision populations, see step); the post-streaming
        # populations are formed from them only where they are used (output, checkpoints, the returned value)
        state_tree = self.state_from_populations(f_tree)
        del f_tree
        # Compile the device loop up front (a zero-step call), so that it is not timed (MLUPS)
        state_tree = self.advance(state_tree, start_step, 0)
        if self.compute_MLUPS:
            start = time.time()

        # Loop over all time steps
        timestep = start_step
        while timestep <= t_max:
            # Steps without host-side events, see advance_quiet_steps
            quiet = quiet_steps(timestep, t_max, self.io_rate, self.print_info_rate, self.checkpoint_rate)
            if quiet:
                state_tree, timestep = advance_quiet_steps(self, state_tree, timestep, quiet, self.advance, lambda s, t: self.step(s, t)[0])
                continue
            io_flag = self.io_rate > 0 and (timestep % self.io_rate == 0 or timestep == t_max)
            print_iter_flag = self.print_info_rate > 0 and timestep % self.print_info_rate == 0
            checkpoint_flag = self.checkpoint_rate > 0 and timestep % self.checkpoint_rate == 0

            # if io_flag:
            #     # Update the macroscopic variables and save the previous values (for error computation)
            #     rho_prev_tree, _ = self.update_macroscopic(f_tree)
            #     # update_macroscopic sums f_tree directly, so rho_prev_tree inherits f_tree's storage precision.
            #     # macroscopic_velocity -> compute_force -> apply_contact_angle scatters into rho at its own dtype
            #     # using values derived from G_ff (permanently fixed at compute precision), so under mixed
            #     # precision (storage narrower than compute) that scatter's source and target dtypes mismatch.
            #     u_prev_tree = self.macroscopic_velocity(f_tree, rho_prev_tree)
            #     rho_prev_tree = tree_map(
            #         lambda rho_prev: downsample_field(rho_prev, self.downsampling_factor),
            #         rho_prev_tree,
            #     )
            #     psi_prev_tree, _ = self.compute_potential(rho_prev_tree)
            #     p_prev_tree = self.compute_pressure(rho_prev_tree, psi_prev_tree)
            #     p_prev_total = self.compute_total_pressure(p_prev_tree, rho_prev_tree)
            #     p_prev_total = downsample_field(p_prev_total, self.downsampling_factor)
            #     u_prev_tree = tree_map(lambda u_prev: downsample_field(u_prev, self.downsampling_factor), u_prev_tree)
            #     rho_total_prev = self.compute_total_density(rho_prev_tree)
            #     u_total_prev = self.compute_total_velocity(rho_prev_tree, u_prev_tree)

            #     # Gather the data from all processes and convert it to numpy arrays (move to host memory)
            #     p_prev_total = process_allgather(p_prev_total)
            #     rho_prev_tree = tree_map(lambda rho_prev: process_allgather(rho_prev), rho_prev_tree)
            #     u_prev_tree = tree_map(lambda u_prev: process_allgather(u_prev), u_prev_tree)
            #     rho_total_prev = process_allgather(rho_total_prev)
            #     u_total_prev = process_allgather(u_total_prev)

            # Perform one time-step (streaming, collision, and boundary conditions)
            state_tree, fstar_tree = self.step(state_tree, timestep, io_flag and self.return_fpost)
            f_tree = self.poststreaming(state_tree) if (io_flag or checkpoint_flag) else None

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
                rho_tree, _ = self.update_macroscopic(f_tree)
                u_tree = self.macroscopic_velocity(f_tree, rho_tree)
                psi_tree = self.compute_potential(rho_tree)
                p_tree = self.compute_pressure(rho_tree, psi_tree)
                p_total = self.compute_total_pressure(p_tree, rho_tree)
                p_total = downsample_field(p_total, self.downsampling_factor)
                rho_tree = tree_map(
                    lambda rho: downsample_field(rho, self.downsampling_factor),
                    rho_tree,
                )
                u_tree = tree_map(lambda u: downsample_field(u, self.downsampling_factor), u_tree)

                rho_total = self.compute_total_density(rho_tree)
                u_total = self.compute_total_velocity(rho_tree, u_tree)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                p_total = process_allgather(p_total)
                rho_tree = tree_map(lambda rho: process_allgather(rho), rho_tree)
                u_tree = tree_map(lambda u: process_allgather(u), u_tree)
                rho_total = process_allgather(rho_total)
                u_total = process_allgather(u_total)

                # Save the data
                self.handle_io_timestep(
                    timestep,
                    f_tree,
                    fstar_tree,
                    p_tree,
                    p_total,
                    u_tree,
                    u_total,
                    rho_total,
                    rho_tree,
                    # p_prev_tree,
                    # p_prev_total,
                    # u_total_prev,
                    # u_prev_tree,
                    # rho_total_prev,
                    # rho_prev_tree,
                )

            if checkpoint_flag:
                # Save the checkpoint
                logger.info(f"Saving checkpoint at timestep {timestep}/{t_max}")
                state = {}
                c_name = lambda i: f"component_{i}"
                for i in range(self.n_components):
                    state[c_name(i)] = f_tree[i]

                self.mngr.save(timestep, args=orb.args.StandardSave(state))

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.compute_MLUPS and timestep == 1:
                jax.block_until_ready(state_tree)
                start = time.time()
            timestep += 1

        if self.compute_MLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(state_tree)
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
                logger.info(
                    colored("MLUPS: ", "blue")
                    + colored(
                        f"{self.nx * self.ny * t_max / (end - start) / 1e6}",
                        "red",
                    )
                )

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
        self._output_writer.finish()
        if self.mngr is not None:
            self.mngr.wait_until_finished()
        return self.poststreaming(state_tree)

    def handle_io_timestep(
        self,
        timestep,
        f_tree,
        fstar_tree,
        p_tree,
        p_total,
        u_tree,
        u_total,
        rho_total,
        rho_tree,
        # p_prev_tree,
        # p_prev_total,
        # u_total_prev,
        # u_prev_tree,
        # rho_total_prev,
        # rho_prev_tree,
    ):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten
        by the user to customize the I/O operations.

        Parameters
        ----------
        timestep (int): The current time step of the simulation.

        f_tree (pytree of jax.numpy.ndarray): Post-streaming distribution functions at the current time step.

        fstar_tree (pytree of jax.numpy.ndarray): Post-collision distribution functions at the current time step.

        p_tree (pytree of jax.numpy.ndarray): Pressure field at the current time step.

        p_total (jax.numpy.ndarray): Total pressure field at the current time step.

        u_total (jax.numpy.ndarray): Total velocity field at the current time step.

        u_tree (pytree of jax.numpy.ndarray): Velocity field at the current time step.

        rho_total (jax.numpy.ndarray): Total density field at the current time step.

        rho_tree (pytree of jax.numpy.ndarray): Density field at the current time step.

        # p_prev_tree (pytree of jax.numpy.ndarray): Pressure field at the previous time step.

        # p_prev_total (jax.numpy.ndarray): Total pressure field at the previous time step.

        # u_total_prev (jax.numpy.ndarray): Total velocity field at the previous time step.

        # u_prev_tree (pytree of jax.numpy.ndarray): Velocity field at the previous time step.

        # rho_total_prev (jax.numpy.ndarray): Total density field at the previous time step.

        # rho_prev_tree (pytree of jax.numpy.ndarray): Density field at the previous time step.

        Returns
        -------
        None
        """
        kwargs = {
            "n_components": self.n_components,
            "timestep": timestep,
            "rho_total": rho_total,
            "rho_tree": rho_tree,
            "p_tree": p_tree,
            "p": p_total,
            "u_total": u_total,
            "u_tree": u_tree,
            # "rho_total_prev": rho_total_prev,
            # "rho_prev_tree": rho_prev_tree,
            # "p_prev_tree": p_prev_tree,
            # "p_prev": p_prev_total,
            # "u_total_prev": u_total_prev,
            # "u_prev_tree": u_prev_tree,
            "f_poststreaming_tree": f_tree,
            "f_postcollision_tree": fstar_tree,
        }
        self._output_writer.submit(self.output_data, kwargs)


class MultiphaseBGK(Multiphase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _collision_applies_bounceback(self):
        """The per-direction collision applies the full-way bounce-back itself (see LBMBase.apply_fused_bounceback)."""
        return type(self).collision is MultiphaseBGK.collision and self._uses_default_bgk_methods()

    def _uses_default_bgk_methods(self):
        """Whether the macroscopic fields, equilibrium and forcing are the defaults (then the per-direction collision applies)."""
        cls = type(self)
        return (
            cls.update_macroscopic is Multiphase.update_macroscopic
            and cls.equilibrium is Multiphase.equilibrium
            and cls.apply_force is Multiphase.apply_force
        )

    def _collision_components(self, fin_tree, T=None):
        """
        The BGK collision with exact-difference forcing evaluated per lattice direction, for the default macroscopic fields,
        equilibrium and forcing (as in MultiphaseMRT.collision): fout_i = f_i + omega (feq_i - f_i) + delta_feq_i.

        Every term is computed per direction with the lattice velocities as compile-time constants, so XLA fuses the moments,
        equilibrium, forcing and relaxation into one kernel with one thread per node. The interaction force is computed in its
        own kernel first (see MultiphaseMRT.collision). Same formulas as the array path; results can differ at roundoff level
        (summation order of the moments).
        """
        c_host = np.asarray(self.lattice.c)
        directions = tuple(tuple(int(v) for v in c_host[:, i]) for i in range(self.q))
        weights = tuple(float(v) for v in np.asarray(self.lattice.w))
        channels_tree = [[f[..., i] for i in range(self.q)] for f in fin_tree]
        rho_scalar_tree = [sum(channels) for channels in channels_tree]
        velocity_tree = [
            [_lattice_sum(directions, channels, axis) / rho for axis in range(self.dim)]
            for channels, rho in zip(channels_tree, rho_scalar_tree, strict=True)
        ]
        rho_tree = [rho[..., None] for rho in rho_scalar_tree]
        F_tree = self._shared_force if self._shared_force is not None else self.compute_force(rho_tree, T=T)
        # No barrier on the force: XLA fuses the psi-gradient stencil into the collision, which then reads the (materialized,
        # padded) psi instead of a stored 3-component force (see _gradient_stencil_m and MultiphaseMRT.collision)
        F_tree = list(F_tree)
        self._export_force_corrected_fields(rho_scalar_tree, velocity_tree, F_tree)
        fout_tree = []
        for channels, rho, velocity, force, omega, mask in zip(
            channels_tree, rho_scalar_tree, velocity_tree, F_tree, self.omega, self.bounceback_mask, strict=True
        ):
            du = [force[..., axis] / rho for axis in range(self.dim)]
            usqr = 1.5 * sum(v * v for v in velocity)
            delta_usqr = 1.5 * sum(2.0 * v * d + d * d for v, d in zip(velocity, du, strict=True))
            outputs = []
            for direction, w, f in zip(directions, weights, channels, strict=True):
                cu = 3.0 * _lattice_dot(direction, velocity)
                dcu = 3.0 * _lattice_dot(direction, du)
                rho_w = rho * w
                feq = rho_w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
                outputs.append(f + omega * (feq - f) + rho_w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr))
            if self.wetting_formulation == "geometric" and self.dim == 3:
                # Preserve the density moment after the 3D geometric wetting update by applying any roundoff-level mismatch to the rest population.
                outputs[0] = outputs[0] + (rho - sum(outputs))
            outputs = self.apply_fused_bounceback(outputs, channels, mask)
            fout_tree.append(self.stack_populations([self.precision_policy.cast_to_output(output) for output in outputs]))
        return fout_tree

    @partial(jit, static_argnums=(0,))
    def collision(self, fin_tree, T=None):
        """
        BGK collision step for lattice, extended to pytrees.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.
        The optional temperature field T is forwarded to the force computation for thermal EOS.
        """
        fin_tree = tree_map(lambda fin: self.precision_policy.cast_to_compute(fin), fin_tree)
        if self._uses_default_bgk_methods():
            return self._collision_components(fin_tree, T=T)
        rho_tree, u_tree = self.update_macroscopic(fin_tree)
        feq_tree = self.equilibrium(rho_tree, u_tree, cast_output=False)
        fneq_tree = tree_map(lambda feq, fin: feq - fin, feq_tree, fin_tree)
        fout_tree = tree_map(lambda fin, fneq, omega: fin + omega * fneq, fin_tree, fneq_tree, self.omega)

        fout_tree = self.apply_force(fout_tree, feq_tree, rho_tree, u_tree, T=T)
        if self.wetting_formulation == "geometric" and self.dim == 3:
            # Preserve the density moment after the 3D geometric wetting update by applying any roundoff-level mismatch to the rest population.
            rho_out_tree = tree_map(lambda fout: jnp.sum(fout, axis=-1, keepdims=True), fout_tree)
            fout_tree = tree_map(lambda fout, rho, rho_out: fout.at[..., 0].add((rho - rho_out)[..., 0]), fout_tree, rho_tree, rho_out_tree)
        return tree_map(lambda fout: self.precision_policy.cast_to_output(fout), fout_tree)


class MultiphaseMRT(Multiphase):
    """Multiphase MRT solver with fused sparse collision kernels.

    One fused collision serves every configuration: overridden update_macroscopic,
    equilibrium, compute_force or adjust_surface_tension methods, wetting schemes and the
    kappa surface-tension term are handled inside it (see collision()).
    """

    def _collision_applies_bounceback(self):
        """The fused collision (every configuration) applies the full-way bounce-back itself (see LBMBase.apply_fused_bounceback)."""
        return type(self).collision is MultiphaseMRT.collision

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._uses_default_mrt_equilibrium = type(self).equilibrium is Multiphase.equilibrium
        self.kappa = kwargs.get("kappa")
        self.s_rho = kwargs.get("s_rho")
        self.s_e = kwargs.get("s_e")
        self.s_eta = kwargs.get("s_eta")
        self.s_j = kwargs.get("s_j")
        self.s_q = kwargs.get("s_q")
        self.s_v = kwargs.get("s_v")
        self.M_inv = tree_map(
            lambda M: jnp.array(
                np.linalg.inv(M).T,
                dtype=self.precision_policy.compute_dtype,
            ),
            kwargs.get("M"),
        )
        self.M = tree_map(
            lambda M: jnp.array(M.T, dtype=self.precision_policy.compute_dtype),
            kwargs.get("M"),
        )
        if isinstance(self.lattice, LatticeD2Q9):
            self.S = tree_map(
                lambda s_rho, s_e, s_eta, s_j, s_q, s_v: jnp.array(
                    np.diag([s_rho, s_e, s_eta, s_j, s_q, s_j, s_q, s_v, s_v]),
                    dtype=self.precision_policy.compute_dtype,
                ),
                self.s_rho,
                self.s_e,
                self.s_eta,
                self.s_j,
                self.s_q,
                self.s_v,
            )
        elif isinstance(self.lattice, LatticeD3Q19):
            self.s_pi = kwargs.get("s_pi")
            self.s_m = kwargs.get("s_m")
            self.S = tree_map(
                lambda s_rho, s_e, s_eta, s_j, s_q, s_v, s_pi, s_m: jnp.array(
                    np.diag([
                        s_rho,
                        s_e,
                        s_eta,
                        s_j,
                        s_q,
                        s_j,
                        s_q,
                        s_j,
                        s_q,
                        s_v,
                        s_pi,
                        s_v,
                        s_pi,
                        s_v,
                        s_v,
                        s_v,
                        s_m,
                        s_m,
                        s_m,
                    ]),
                    dtype=self.precision_policy.compute_dtype,
                ),
                self.s_rho,
                self.s_e,
                self.s_eta,
                self.s_j,
                self.s_q,
                self.s_v,
                self.s_pi,
                self.s_m,
            )

        # Fused collision matrix K = M @ S @ M_inv, replacing the three separate moment-space matrix multiplies
        # (f @ M, relax, @ M_inv) in collision() with symbolic, sparse-coefficient columns compiled from K's
        # nonzero entries - see collision() and its module-level References for the fusion identity.
        self.collision_matrix = tree_map(lambda M, S, M_inv: jnp.dot(jnp.dot(M, S), M_inv), self.M, self.S, self.M_inv)
        self._lattice_directions = tuple(tuple(int(v) for v in np.asarray(self.lattice.c)[:, i]) for i in range(self.lattice.q))
        self._lattice_weights = tuple(float(v) for v in np.asarray(self.lattice.w))
        # Coefficients in the compute precision (see LBMBase.sparse_collision_terms)
        self.collision_terms = [self.sparse_collision_terms(collision_matrix) for collision_matrix in self.collision_matrix]
        # Surface tension adjustment (adjust_surface_tension) is identically zero whenever every component's
        # kappa is zero - a static (non-traced) fact known here, so collision() can skip computing and adding
        # it entirely in that case, rather than multiplying by a provably-zero array every timestep.
        self._surface_tension_components = tuple(float(kappa) != 0.0 for kappa in self.kappa)
        self._has_surface_tension = any(self._surface_tension_components)
        # Extension points honored by collision(); the defaults take the fused per-direction formulation.
        self._uses_default_update_macroscopic = type(self).update_macroscopic is Multiphase.update_macroscopic
        self._uses_default_surface_tension = type(self).adjust_surface_tension is MultiphaseMRT.adjust_surface_tension
        if self._has_surface_tension and self._uses_default_surface_tension:
            if not isinstance(self.lattice, (LatticeD2Q9, LatticeD3Q19)):
                raise NotImplementedError("The kappa surface-tension adjustment is implemented for D2Q9 and D3Q19 only.")
            self._check_surface_tension_basis(kwargs.get("M"))
        self.scalar_surface_stencil = None
        self.scalar_gradient_surface = None
        self.surface_moment_center = None
        if self._has_surface_tension and isinstance(self.lattice, (LatticeD2Q9, LatticeD3Q19)):
            moment_pairs = ((0, 0), (0, 1), (1, 1)) if self.dim == 2 else ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
            c_host = np.asarray(self.c)
            surface_weights = np.stack([np.asarray(self.G_ff) * c_host[i] * c_host[j] for i, j in moment_pairs])
            P = PartitionSpec
            scalar_spec = P("x", None, None) if self.dim == 2 else P("x", None, None, None)
            self.scalar_surface_stencil = jit(
                shard_map(
                    partial(self._neighbor_stencil_m, weights=surface_weights),
                    mesh=self.mesh,
                    in_specs=scalar_spec,
                    out_specs=scalar_spec,
                    check_vma=False,
                )
            )
            self.surface_moment_center = jnp.asarray(np.sum(surface_weights, axis=1), dtype=self.precision_policy.compute_dtype)
            # Gradient and surface moments of psi from one padded array (one halo exchange, one pass over psi).
            self.scalar_gradient_surface = jit(
                shard_map(
                    partial(self._gradient_stencil_m, weights=surface_weights),
                    mesh=self.mesh,
                    in_specs=scalar_spec,
                    out_specs=scalar_spec,
                    check_vma=False,
                )
            )
        # Population-space coefficients of the surface source, (C @ M_inv)_i = sum_k C_k M_inv[k, i], restricted to the moments
        # C can populate, so the fast collision adds it per direction without a dense q x q product.
        surface_moments = (1, 2, 7, 8) if self.dim == 2 else (1, 9, 11, 13, 14, 15)
        self._surface_population_terms = []
        coefficient_dtype = np.float64 if self.precision_policy.compute_dtype == jnp.float64 else np.float32
        for M_inv, has_surface_tension in zip(self.M_inv, self._surface_tension_components, strict=True):
            if not has_surface_tension:
                self._surface_population_terms.append(None)
                continue
            matrix = np.asarray(M_inv)
            self._surface_population_terms.append(
                tuple(
                    tuple((k, coefficient_dtype(matrix[k, i])) for k in surface_moments if not np.isclose(matrix[k, i], 0.0, atol=1e-7))
                    for i in range(self.lattice.q)
                )
            )

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value=None):
        self._omega = value

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        if not isinstance(value, list):
            raise ValueError("Matrix M must be a list")
        if len(value) != self.n_components:
            raise ValueError("Number of components does not match number of matrix M passed")
        self._M = value

    @partial(jit, static_argnums=(0,))
    def adjust_surface_tension(self, psi_tree):
        """
        Adjust surface tension in multiphase MRT model. It does not effect the coexistence densities. Assumes orthogonal collision
        matrix (see the pool boiling 3D example) instead of raw-moments (see the droplet 3D example).

        Uses the papers' convention: surface tension scales as (1 - kappa), so kappa = 0 leaves it unchanged and kappa = 1 removes it.

        Parameters
        ----------
        psi_tree: pytree of jax.numpy.ndarray
            Square-root form pseudopotential field.

        Returns
        -------
        pytree of jax.numpy.ndarray
            Surface tension adjustment field.
        """
        if not isinstance(self.lattice, (LatticeD2Q9, LatticeD3Q19)):
            raise NotImplementedError("MRT model with D3Q27 model has not been implemented")

        C_tree = []
        for psi, moments in zip(psi_tree, self._surface_source_moments(psi_tree), strict=True):
            C = jnp.zeros((*psi.shape[:-1], self.q), dtype=self.precision_policy.compute_dtype)
            if moments is not None:
                for index, value in moments.items():
                    C = C.at[..., index].set(value)
            C_tree.append(C)
        return C_tree

    def _surface_source_moments(self, psi_tree, psi_moments_tree=None):
        """
        Nonzero moment-space entries of the surface-tension source term C, per component.

        Returns
        -------
        list of (dict of int -> jax.numpy.ndarray) or None
            For each component, {moment index: field of shape (...)} or None when that component's kappa is zero.

        psi_moments_tree (list of jax.numpy.ndarray, optional): scalar_surface_stencil(psi) per component, if already
            computed (e.g. fused with the gradient by scalar_gradient_surface).
        """
        out = []
        g_diagonal = self._g_host.diagonal().tolist()
        for component, (kappa, g, s_v, s_e, s_eta, psi) in enumerate(
            zip(self.kappa, g_diagonal, self.s_v, self.s_e, self.s_eta, psi_tree, strict=True)
        ):
            if not self._surface_tension_components[component]:
                out.append(None)
                continue

            psi_moments = self.scalar_surface_stencil(psi) if psi_moments_tree is None else psi_moments_tree[component]
            centered_psi = psi_moments - psi * self.surface_moment_center
            # Q = kappa * (G / 2) * psi * sum_i w_i (psi(x + e_i) - psi(x)) e_i e_i (Li & Luo 2013), so sigma ~ (1 - kappa).
            qmoments = (0.5 * g * kappa) * psi * centered_psi
            if self.dim == 2:
                qxx, qxy, qyy = (qmoments[..., index] for index in range(3))
                out.append({1: 1.5 * s_e * (qxx + qyy), 2: -1.5 * s_eta * (qxx + qyy), 7: -s_v * (qxx - qyy), 8: -s_v * qxy})
            else:
                qxx, qxy, qxz, qyy, qyz, qzz = (qmoments[..., index] for index in range(6))
                # e = 19|c|^2 - 30: keeping the normal pressure fixed while the tangential part changes by -kappa needs
                # delta(tr Pi) = (4/5) tr Q, i.e. (76/5) s_e tr Q (Yang & Qin 2024, Eq. 25).
                out.append({
                    1: (76.0 / 5.0) * s_e * (qxx + qyy + qzz),
                    9: -s_v * (2.0 * qxx - qyy - qzz),
                    11: -s_v * (qyy - qzz),
                    13: -s_v * qxy,
                    14: -s_v * qyz,
                    15: -s_v * qxz,
                })
        return out

    @partial(jit, static_argnums=(0,), inline=True)
    def _force_delta_feq(self, rho_tree, u_tree, F_tree):
        """Compute compact EDM population differences from an already evaluated force."""
        du_tree = tree_map(lambda F, rho: F / rho, F_tree, rho_tree)

        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cu_tree = tree_map(lambda u: 3.0 * jnp.dot(u, c), u_tree)
        dcu_tree = tree_map(lambda du: 3.0 * jnp.dot(du, c), du_tree)
        delta_usqr_tree = tree_map(
            lambda u, du: 1.5 * (2.0 * jnp.sum(u * du, axis=-1, keepdims=True) + jnp.sum(jnp.square(du), axis=-1, keepdims=True)),
            u_tree,
            du_tree,
        )
        return tree_map(
            lambda rho, cu, dcu, delta_usqr: rho * self.w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr),
            rho_tree,
            cu_tree,
            dcu_tree,
            delta_usqr_tree,
        )

    @partial(jit, static_argnums=(0,), inline=True)
    def _compute_force_delta_feq(self, rho_tree, u_tree, T=None):
        """
        Compute the real-space EDM difference for all components.

        Parameters
        ----------
        rho_tree : pytree of jax.Array
            Component density fields.
        u_tree : pytree of jax.Array
            Component velocity fields.
        T : jax.Array or None, optional
            Temperature field required by a thermal EOS.

        Returns
        -------
        pytree of jax.Array
            Population-space ``feq(rho, u + F / rho) - feq(rho, u)``.
        """
        return self._force_delta_feq(rho_tree, u_tree, self.compute_force(rho_tree, T=T))

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, m_tree, meq_tree, rho_tree, u_tree, T=None):
        """
        Apply the exact-difference force in moment space.

        meq_tree is retained for compatibility with the base collision interface.

        Parameters
        ----------
        m_tree : pytree of jax.Array
            Post-collision moments for each component.
        meq_tree : pytree of jax.Array
            Equilibrium moments; unused but required by the collision interface.
        rho_tree : pytree of jax.Array
            Component density fields.
        u_tree : pytree of jax.Array
            Component velocity fields.
        T : jax.Array or None, optional
            Temperature field required by a thermal EOS.

        Returns
        -------
        pytree of jax.Array
            Post-collision moments with the force contribution applied.
        """
        delta_feq_tree = self._compute_force_delta_feq(rho_tree, u_tree, T=T)
        delta_meq_tree = tree_map(lambda delta_feq, M: jnp.dot(delta_feq, M), delta_feq_tree, self.M)
        return tree_map(lambda m, delta_meq: m + delta_meq, m_tree, delta_meq_tree)

    def _check_surface_tension_basis(self, matrices):
        """
        Warn when kappa is used with a moment basis other than the one the surface-tension source is defined in.

        The source term C places tr(Q) and the deviatoric Q components on specific orthogonal moments (McCracken & Abraham /
        Li & Luo in 2D, d'Humieres et al. in 3D, as in examples/thermal/multiphase/shanchen/MRT/pool_boiling_3d.py). With other rows
        or row order the term lands on different moments and sigma no longer scales as (1 - kappa).
        """
        e = np.asarray(self.lattice.c).T.astype(np.float64)
        c2 = np.sum(e**2, axis=1)
        x, y = e[:, 0], e[:, 1]
        if self.dim == 2:
            expected = {1: -4 + 3 * c2, 2: 4 - 10.5 * c2 + 4.5 * c2**2, 7: x**2 - y**2, 8: x * y}
        else:
            z = e[:, 2]
            expected = {1: 19 * c2 - 30, 9: 3 * x**2 - c2, 11: y**2 - z**2, 13: x * y, 14: y * z, 15: x * z}
        for component, (matrix, has_surface_tension) in enumerate(zip(matrices, self._surface_tension_components, strict=True)):
            matrix = np.asarray(matrix, dtype=np.float64)
            if has_surface_tension and not all(np.allclose(matrix[k], row) for k, row in expected.items()):
                warnings.warn(
                    f"\nComponent {component}: kappa != 0 with an MRT matrix whose rows {sorted(expected)} are not the orthogonal "
                    "basis the surface-tension source is defined in; the adjustment will not follow sigma ~ (1 - kappa). "
                    "Use the orthogonal basis (see pool_boiling_3d.py) or override adjust_surface_tension.",
                    stacklevel=3,
                )

    @partial(jit, static_argnums=(0,))
    def collision(self, fin_tree, T=None):
        """
        Apply the MRT collision with exact-difference forcing and the optional kappa surface-tension source:

            fout = f - (f - feq) @ (M @ S @ M_inv) + delta_feq + C @ M_inv

        The relaxation uses the sparse nonzero entries of M @ S @ M_inv built from the user's M and S, so any invertible
        moment basis works. With the default update_macroscopic, equilibrium, force and surface-tension methods every term
        is evaluated per direction with the lattice velocities as compile-time constants (no cells x q matrix products), so
        XLA fuses moments, equilibrium, forcing and relaxation into one loop; overriding any of those methods routes that
        term through the override instead. The optional temperature field T is forwarded to the pressure and force
        computations for thermal EOS.
        """
        fin_tree = tree_map(self.precision_policy.cast_to_compute, fin_tree)
        directions, weights = self._lattice_directions, self._lattice_weights
        channels_tree = [[f[..., i] for i in range(self.q)] for f in fin_tree]
        if self._uses_default_update_macroscopic:
            rho_scalar_tree = [sum(channels) for channels in channels_tree]
            velocity_tree = [
                [_lattice_sum(directions, channels, axis) / rho for axis in range(self.dim)]
                for channels, rho in zip(channels_tree, rho_scalar_tree, strict=True)
            ]
            rho_tree = [rho[..., None] for rho in rho_scalar_tree]
        else:
            rho_tree, u_tree = self.update_macroscopic(fin_tree)
            rho_scalar_tree = [rho[..., 0] for rho in rho_tree]
            velocity_tree = [[u[..., axis] for axis in range(self.dim)] for u in u_tree]

        # Force, and the potential/moments the surface-tension source is built from
        uses_wetting = self.wetting_formulation is not None and any(self._has_wetting_bc)
        surface_psi_tree = moments_tree = None
        if self._shared_force is not None:
            # Already computed for this time step by the coupled solver (identical values; see Multiphase._shared_force)
            F_tree = list(self._shared_force)
        elif self._uses_default_compute_force:
            force_rho_tree = self.apply_contact_angle(rho_tree)
            psi_tree = self.compute_potential(force_rho_tree, T=T)
            fused_surface = (
                self._has_surface_tension
                and not uses_wetting
                and self._uses_default_surface_tension
                and type(self).compute_fluid_fluid_force is Multiphase.compute_fluid_fluid_force
            )
            if fused_surface:
                gradient_tree, moments_tree = [], []
                for psi, used, has_surface_tension in zip(psi_tree, self._psi_stencil_components, self._surface_tension_components, strict=True):
                    if has_surface_tension:
                        stencil = self.scalar_gradient_surface(psi)
                        gradient_tree.append(stencil[..., : self.dim])
                        moments_tree.append(stencil[..., self.dim :])
                    else:
                        gradient_tree.append(self.scalar_gradient(psi) if used else None)
                        moments_tree.append(None)
                F_tree = self._fluid_fluid_force_from_gradients(psi_tree, gradient_tree)
            else:
                F_tree = self.compute_fluid_fluid_force(psi_tree)
            if self.body_force is not None:
                F_tree = tree_map(lambda force, rho: force + self.body_force * rho, F_tree, force_rho_tree)
            if self.wetting_formulation == "geometric" and any(self._has_wetting_bc):
                F_tree = tree_map(lambda force, force_mask: force * force_mask, F_tree, self.geometric_force_mask)
            if not uses_wetting:
                surface_psi_tree = psi_tree
        else:
            F_tree = self.compute_force(rho_tree, T=T)
        # The psi-gradient stencil is fused into the collision: it reads the padded psi, which _gradient_stencil_m materializes
        # once, instead of the force being written to memory and read back (3 values per node). Without that psi barrier XLA
        # recomputed psi - and the density and populations behind it - for every stencil neighbor, which is why the force used
        # to be split into its own kernel. Values are identical; 2-GPU droplet runs are 11-26% faster than with the split.
        F_tree = list(F_tree)
        self._export_force_corrected_fields(rho_scalar_tree, velocity_tree, F_tree)

        # Surface-tension source in population space, per component: list of q fields, or None
        surface_tree = [None] * self.n_components
        if self._has_surface_tension:
            if surface_psi_tree is None:
                surface_psi_tree = self.compute_potential(rho_tree, T=T)
            if self._uses_default_surface_tension:
                source_moments = self._surface_source_moments(surface_psi_tree, moments_tree)
                surface_tree = [
                    None if moments is None else [sum(moments[k] * coefficient for k, coefficient in row) for row in terms]
                    for moments, terms in zip(source_moments, self._surface_population_terms, strict=True)
                ]
            else:
                C_tree = self.adjust_surface_tension(surface_psi_tree)
                for component, (C, M_inv) in enumerate(zip(C_tree, self.M_inv, strict=True)):
                    if self._surface_tension_components[component]:
                        source = jnp.dot(C, M_inv)
                        surface_tree[component] = [source[..., i] for i in range(self.q)]

        # Equilibrium per direction (default) or from the user's equilibrium()
        if self._uses_default_mrt_equilibrium:
            feq_tree = None
        else:
            u_tree = [jnp.stack(velocity, axis=-1) for velocity in velocity_tree]
            feq_tree = [[feq[..., i] for i in range(self.q)] for feq in self.equilibrium(rho_tree, u_tree, cast_output=False)]
        # Rest-population roundoff mass correction: default equilibrium (exactly mass conserving) or geometric wetting
        restore_mass = self._uses_default_mrt_equilibrium or self.wetting_formulation == "geometric"

        fout_tree = []
        for component, (channels, rho, velocity, force, columns, surface) in enumerate(
            zip(channels_tree, rho_scalar_tree, velocity_tree, F_tree, self.collision_terms, surface_tree, strict=True)
        ):
            du = [force[..., axis] / rho for axis in range(self.dim)]
            usqr = 1.5 * sum(v * v for v in velocity)
            delta_usqr = 1.5 * sum(2.0 * v * d + d * d for v, d in zip(velocity, du, strict=True))
            difference, delta_feq = [], []
            for i, (direction, w, f) in enumerate(zip(directions, weights, channels, strict=True)):
                cu = 3.0 * _lattice_dot(direction, velocity)
                dcu = 3.0 * _lattice_dot(direction, du)
                rho_w = rho * w
                feq = rho_w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr) if feq_tree is None else feq_tree[component][i]
                difference.append(f - feq)
                delta_feq.append(rho_w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr))

            outputs = []
            for output_direction, terms in enumerate(columns):
                relaxed = sum(difference[input_direction] * coefficient for input_direction, coefficient in terms)
                value = channels[output_direction] - relaxed + delta_feq[output_direction]
                if surface is not None:
                    value = value + surface[output_direction]
                outputs.append(value)
            if restore_mass:
                # Restore the density lost to roundoff on the rest population, from the per-direction values (no re-read).
                outputs[0] = outputs[0] + (rho - sum(outputs))
            outputs = self.apply_fused_bounceback(outputs, channels, self.bounceback_mask[component])
            fout_tree.append(self.stack_populations([self.precision_policy.cast_to_output(output) for output in outputs]))
        return fout_tree


class MultiphaseCascade(Multiphase):
    """
    Cascaded LBM collision model transforms the distribution to central moments and then relaxation them. The central moments are obtained by first
    transforming distributions to raw-moment using transformation matrix similar to MRT model. The raw-moments are subsequently transformed to central
    moments using shift-matrix. CLBM gives independent control over vapor diffusivity, binary diffusivity and Schmidt number, which is not possible using SRT model.

    The current implementation is based on:
    1. Fei, L. & Luo, K. H. Consistent forcing scheme in the cascaded lattice Boltzmann method. Phys. Rev. E 96, 053307 (2017).

    2. Fei, L., Luo, K. H. & Li, Q. Three-dimensional cascaded lattice Boltzmann method: Improved implementation and consistent forcing scheme. Phys. Rev. E 97, 053309 (2018).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._uses_default_cascade_apply_force = type(self).apply_force is MultiphaseCascade.apply_force
        self.sigma = kwargs.get("sigma")
        self.s_0 = kwargs.get("s_0")
        self.s_1 = kwargs.get("s_1")
        self.s_b = kwargs.get("s_b")
        self.s_2 = kwargs.get("s_2")
        self.s_3 = kwargs.get("s_3")
        self.s_4 = kwargs.get("s_4")
        self.M_inv = tree_map(
            lambda M: jnp.array(
                np.linalg.inv(M).T,
                dtype=self.precision_policy.compute_dtype,
            ),
            kwargs.get("M"),
        )
        self.M = tree_map(
            lambda M: jnp.array(M.T, dtype=self.precision_policy.compute_dtype),
            kwargs.get("M"),
        )

        if isinstance(self.lattice, LatticeD2Q9):
            self.S = tree_map(
                lambda s_0, s_1, s_b, s_2, s_3, s_4: jnp.array(
                    np.diag([s_0, s_1, s_1, s_b, s_2, s_2, s_3, s_3, s_4]),
                    dtype=self.precision_policy.compute_dtype,
                ),
                self.s_0,
                self.s_1,
                self.s_b,
                self.s_2,
                self.s_3,
                self.s_4,
            )
        elif isinstance(self.lattice, LatticeD3Q19):
            self.s_plus = tree_map(lambda s_b, s_2: (s_b + 2 * s_2) / 3, self.s_b, self.s_2)
            self.s_minus = tree_map(lambda s_b, s_2: (s_b - s_2) / 3, self.s_b, self.s_2)

            def f(s_0, s_1, s_v, s_plus, s_minus, s_3, s_4):
                S = np.diag([s_0, s_1, s_1, s_1, s_v, s_v, s_v, s_plus, s_plus, s_plus, s_3, s_3, s_3, s_3, s_3, s_3, s_4, s_4, s_4])
                S[7, 8] = s_minus
                S[7, 9] = s_minus
                S[8, 7] = s_minus
                S[8, 9] = s_minus
                S[9, 7] = s_minus
                S[9, 8] = s_minus
                return jnp.array(S, dtype=self.precision_policy.compute_dtype)

            self.S = tree_map(
                lambda s_0, s_1, s_v, s_plus, s_minus, s_3, s_4: f(s_0, s_1, s_v, s_plus, s_minus, s_3, s_4),
                self.s_0,
                self.s_1,
                self.s_2,
                self.s_plus,
                self.s_minus,
                self.s_3,
                self.s_4,
            )
        elif isinstance(self.lattice, LatticeD3Q27):
            self.s_plus = tree_map(lambda s_b, s_2: (s_b + 2 * s_2) / 3, self.s_b, self.s_2)
            self.s_minus = tree_map(lambda s_b, s_2: (s_b - s_2) / 3, self.s_b, self.s_2)
            self.s_3b = kwargs.get("s_3b")
            self.s_4b = kwargs.get("s_4b")
            self.s_5 = kwargs.get("s_5")
            self.s_6 = kwargs.get("s_6")

            def f(s_0, s_1, s_v, s_plus, s_minus, s_3, s_3b, s_4, s_4b, s_5, s_6):
                S = np.diag([
                    s_0,
                    s_1,
                    s_1,
                    s_1,
                    s_v,
                    s_v,
                    s_v,
                    s_plus,
                    s_plus,
                    s_plus,
                    s_3,
                    s_3,
                    s_3,
                    s_3,
                    s_3,
                    s_3,
                    s_3b,
                    s_4,
                    s_4,
                    s_4,
                    s_4b,
                    s_4b,
                    s_4b,
                    s_5,
                    s_5,
                    s_5,
                    s_6,
                ])
                S[7, 8] = s_minus
                S[7, 9] = s_minus
                S[8, 7] = s_minus
                S[8, 9] = s_minus
                S[9, 7] = s_minus
                S[9, 8] = s_minus
                return jnp.array(S, dtype=self.precision_policy.compute_dtype)

            self.S = tree_map(
                lambda s_0, s_1, s_v, s_plus, s_minus, s_3, s_3b, s_4, s_4b, s_5, s_6: f(
                    s_0, s_1, s_v, s_plus, s_minus, s_3, s_3b, s_4, s_4b, s_5, s_6
                ),
                self.s_0,
                self.s_1,
                self.s_2,
                self.s_plus,
                self.s_minus,
                self.s_3,
                self.s_3b,
                self.s_4,
                self.s_4b,
                self.s_5,
                self.s_6,
            )

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value=None):
        self._omega = value

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        if not isinstance(value, list):
            raise ValueError("Matrix M must be a list")
        if len(value) != self.n_components:
            raise ValueError("Number of components does not match number of matrix M passed")
        self._M = value

    @partial(jit, static_argnums=(0,))
    def compute_central_moment(self, m_tree, u_tree):
        if isinstance(self.lattice, LatticeD2Q9):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                usq = ux**2 + uy**2
                udiff = ux**2 - uy**2
                T = {}
                T[0] = m[..., 0]
                T[1] = -ux * m[..., 0] + m[..., 1]
                T[2] = -uy * m[..., 0] + m[..., 2]
                T[3] = usq * m[..., 0] - 2 * ux * m[..., 1] - 2 * uy * m[..., 2] + m[..., 3]
                T[4] = udiff * m[..., 0] - 2 * ux * m[..., 1] + 2 * uy * m[..., 2] + m[..., 4]
                T[5] = ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 5]
                T[6] = (
                    -(ux**2) * uy * m[..., 0]
                    + 2 * ux * uy * m[..., 1]
                    + ux**2 * m[..., 2]
                    - 0.5 * uy * m[..., 3]
                    - 0.5 * uy * m[..., 4]
                    - 2 * ux * m[..., 5]
                    + m[..., 6]
                )
                T[7] = (
                    -(uy**2) * ux * m[..., 0]
                    + uy**2 * m[..., 1]
                    + 2 * ux * uy * m[..., 2]
                    - 0.5 * ux * m[..., 3]
                    + 0.5 * ux * m[..., 4]
                    - 2 * uy * m[..., 5]
                    + m[..., 7]
                )
                T[8] = (
                    (uy**2 * ux**2) * m[..., 0]
                    - 2 * ux * uy**2 * m[..., 1]
                    - 2 * uy * ux**2 * m[..., 2]
                    + 0.5 * usq * m[..., 3]
                    - 0.5 * udiff * m[..., 4]
                    + 4 * ux * uy * m[..., 5]
                    - 2 * uy * m[..., 6]
                    - 2 * ux * m[..., 7]
                    + m[..., 8]
                )
                T = stack_components(T, m)
                return T

            return tree_map(lambda m, u: shift(m, u), m_tree, u_tree)

        elif isinstance(self.lattice, LatticeD3Q19):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                T = {}
                T[0] = m[..., 0]
                T[1] = -ux * m[..., 0] + m[..., 1]
                T[2] = -uy * m[..., 0] + m[..., 2]
                T[3] = -uz * m[..., 0] + m[..., 3]
                T[4] = ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 4]
                T[5] = ux * uz * m[..., 0] - uz * m[..., 1] - ux * m[..., 3] + m[..., 5]
                T[6] = uy * uz * m[..., 0] - uz * m[..., 2] - uy * m[..., 3] + m[..., 6]
                T[7] = (ux**2) * m[..., 0] - 2 * ux * m[..., 1] + m[..., 7]
                T[8] = (uy**2) * m[..., 0] - 2 * uy * m[..., 2] + m[..., 8]
                T[9] = (uz**2) * m[..., 0] - 2 * uz * m[..., 3] + m[..., 9]
                T[10] = -ux * (uy**2) * m[..., 0] + (uy**2) * m[..., 1] + 2 * ux * uy * m[..., 2] - 2 * uy * m[..., 4] - ux * m[..., 8] + m[..., 10]
                T[11] = -ux * (uz**2) * m[..., 0] + (uz**2) * m[..., 1] + 2 * ux * uz * m[..., 3] - 2 * uz * m[..., 5] - ux * m[..., 9] + m[..., 11]
                T[12] = -(ux**2) * uy * m[..., 0] + 2 * ux * uy * m[..., 1] + (ux**2) * m[..., 2] - 2 * ux * m[..., 4] - uy * m[..., 7] + m[..., 12]
                T[13] = -(ux**2) * uz * m[..., 0] + 2 * ux * uz * m[..., 1] + (ux**2) * m[..., 3] - 2 * ux * m[..., 5] - uz * m[..., 7] + m[..., 13]
                T[14] = -uy * (uz**2) * m[..., 0] + (uz**2) * m[..., 2] + 2 * uy * uz * m[..., 3] - 2 * uz * m[..., 6] - uy * m[..., 9] + m[..., 14]
                T[15] = -(uy**2) * uz * m[..., 0] + 2 * uy * uz * m[..., 2] + (uy**2) * m[..., 3] - 2 * uy * m[..., 6] - uz * m[..., 8] + m[..., 15]
                T[16] = (
                    (ux**2) * (uy**2) * m[..., 0]
                    - 2 * ux * (uy**2) * m[..., 1]
                    - 2 * uy * (ux**2) * m[..., 2]
                    + 4 * ux * uy * m[..., 4]
                    + (uy**2) * m[..., 7]
                    + (ux**2) * m[..., 8]
                    - 2 * ux * m[..., 10]
                    - 2 * uy * m[..., 12]
                    + m[..., 16]
                )
                T[17] = (
                    (ux**2) * (uz**2) * m[..., 0]
                    - 2 * ux * (uz**2) * m[..., 1]
                    - 2 * uz * (ux**2) * m[..., 3]
                    + 4 * ux * uz * m[..., 5]
                    + (uz**2) * m[..., 7]
                    + (ux**2) * m[..., 9]
                    - 2 * ux * m[..., 11]
                    - 2 * uz * m[..., 13]
                    + m[..., 17]
                )
                T[18] = (
                    (uy**2) * (uz**2) * m[..., 0]
                    - 2 * uy * (uz**2) * m[..., 2]
                    - 2 * uz * (uy**2) * m[..., 3]
                    + 4 * uy * uz * m[..., 6]
                    + (uz**2) * m[..., 8]
                    + (uy**2) * m[..., 9]
                    - 2 * uy * m[..., 14]
                    - 2 * uz * m[..., 15]
                    + m[..., 18]
                )
                T = stack_components(T, m)
                return T

            return tree_map(lambda m, u: shift(m, u), m_tree, u_tree)

        elif isinstance(self.lattice, LatticeD3Q27):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                T = {}
                T[0] = m[..., 0]
                T[1] = m[..., 1] - m[..., 0] * ux
                T[2] = m[..., 2] - m[..., 0] * uy
                T[3] = m[..., 3] - m[..., 0] * uz
                T[4] = m[..., 4] - m[..., 2] * ux - m[..., 1] * uy + m[..., 0] * ux * uy
                T[5] = m[..., 5] - m[..., 3] * ux - m[..., 1] * uz + m[..., 0] * ux * uz
                T[6] = m[..., 6] - m[..., 3] * uy - m[..., 2] * uz + m[..., 0] * uy * uz
                T[7] = m[..., 0] * ux**2 - 2 * m[..., 1] * ux + m[..., 7]
                T[8] = m[..., 0] * uy**2 - 2 * m[..., 2] * uy + m[..., 8]
                T[9] = m[..., 0] * uz**2 - 2 * m[..., 3] * uz + m[..., 9]
                T[10] = m[..., 10] - m[..., 8] * ux - 2 * m[..., 4] * uy + m[..., 1] * uy**2 - m[..., 0] * ux * uy**2 + 2 * m[..., 2] * ux * uy
                T[11] = m[..., 11] - m[..., 9] * ux - 2 * m[..., 5] * uz + m[..., 1] * uz**2 - m[..., 0] * ux * uz**2 + 2 * m[..., 3] * ux * uz
                T[12] = m[..., 12] - 2 * m[..., 4] * ux - m[..., 7] * uy + m[..., 2] * ux**2 - m[..., 0] * ux**2 * uy + 2 * m[..., 1] * ux * uy
                T[13] = m[..., 13] - 2 * m[..., 5] * ux - m[..., 7] * uz + m[..., 3] * ux**2 - m[..., 0] * ux**2 * uz + 2 * m[..., 1] * ux * uz
                T[14] = m[..., 14] - m[..., 9] * uy - 2 * m[..., 6] * uz + m[..., 2] * uz**2 - m[..., 0] * uy * uz**2 + 2 * m[..., 3] * uy * uz
                T[15] = m[..., 15] - 2 * m[..., 6] * uy - m[..., 8] * uz + m[..., 3] * uy**2 - m[..., 0] * uy**2 * uz + 2 * m[..., 2] * uy * uz
                T[16] = (
                    m[..., 16]
                    - m[..., 6] * ux
                    - m[..., 5] * uy
                    - m[..., 4] * uz
                    + m[..., 3] * ux * uy
                    + m[..., 2] * ux * uz
                    + m[..., 1] * uy * uz
                    - m[..., 0] * ux * uy * uz
                )
                T[17] = (
                    m[..., 0] * ux**2 * uy**2
                    - 2 * m[..., 2] * ux**2 * uy
                    + m[..., 8] * ux**2
                    - 2 * m[..., 1] * ux * uy**2
                    + 4 * m[..., 4] * ux * uy
                    - 2 * m[..., 10] * ux
                    + m[..., 7] * uy**2
                    - 2 * m[..., 12] * uy
                    + m[..., 17]
                )
                T[18] = (
                    m[..., 0] * ux**2 * uz**2
                    - 2 * m[..., 3] * ux**2 * uz
                    + m[..., 9] * ux**2
                    - 2 * m[..., 1] * ux * uz**2
                    + 4 * m[..., 5] * ux * uz
                    - 2 * m[..., 11] * ux
                    + m[..., 7] * uz**2
                    - 2 * m[..., 13] * uz
                    + m[..., 18]
                )
                T[19] = (
                    m[..., 0] * uy**2 * uz**2
                    - 2 * m[..., 3] * uy**2 * uz
                    + m[..., 9] * uy**2
                    - 2 * m[..., 2] * uy * uz**2
                    + 4 * m[..., 6] * uy * uz
                    - 2 * m[..., 14] * uy
                    + m[..., 8] * uz**2
                    - 2 * m[..., 15] * uz
                    + m[..., 19]
                )
                T[20] = (
                    m[..., 20]
                    - 2 * m[..., 16] * ux
                    - m[..., 13] * uy
                    - m[..., 12] * uz
                    + m[..., 6] * ux**2
                    - m[..., 3] * ux**2 * uy
                    - m[..., 2] * ux**2 * uz
                    + 2 * m[..., 5] * ux * uy
                    + 2 * m[..., 4] * ux * uz
                    + m[..., 7] * uy * uz
                    - 2 * m[..., 1] * ux * uy * uz
                    + m[..., 0] * ux**2 * uy * uz
                )
                T[21] = (
                    m[..., 21]
                    - m[..., 15] * ux
                    - 2 * m[..., 16] * uy
                    - m[..., 10] * uz
                    + m[..., 5] * uy**2
                    - m[..., 3] * ux * uy**2
                    - m[..., 1] * uy**2 * uz
                    + 2 * m[..., 6] * ux * uy
                    + m[..., 8] * ux * uz
                    + 2 * m[..., 4] * uy * uz
                    - 2 * m[..., 2] * ux * uy * uz
                    + m[..., 0] * ux * uy**2 * uz
                )
                T[22] = (
                    m[..., 22]
                    - m[..., 14] * ux
                    - m[..., 11] * uy
                    - 2 * m[..., 16] * uz
                    + m[..., 4] * uz**2
                    - m[..., 2] * ux * uz**2
                    - m[..., 1] * uy * uz**2
                    + m[..., 9] * ux * uy
                    + 2 * m[..., 6] * ux * uz
                    + 2 * m[..., 5] * uy * uz
                    - 2 * m[..., 3] * ux * uy * uz
                    + m[..., 0] * ux * uy * uz**2
                )
                T[23] = (
                    m[..., 23]
                    - m[..., 19] * ux
                    - 2 * m[..., 22] * uy
                    - 2 * m[..., 21] * uz
                    + m[..., 11] * uy**2
                    + m[..., 10] * uz**2
                    - m[..., 9] * ux * uy**2
                    - m[..., 8] * ux * uz**2
                    - 2 * m[..., 4] * uy * uz**2
                    - 2 * m[..., 5] * uy**2 * uz
                    + m[..., 1] * uy**2 * uz**2
                    + 2 * m[..., 14] * ux * uy
                    + 2 * m[..., 15] * ux * uz
                    + 4 * m[..., 16] * uy * uz
                    - 4 * m[..., 6] * ux * uy * uz
                    + 2 * m[..., 2] * ux * uy * uz**2
                    + 2 * m[..., 3] * ux * uy**2 * uz
                    - m[..., 0] * ux * uy**2 * uz**2
                )
                T[24] = (
                    m[..., 24]
                    - 2 * m[..., 22] * ux
                    - m[..., 18] * uy
                    - 2 * m[..., 20] * uz
                    + m[..., 14] * ux**2
                    + m[..., 12] * uz**2
                    - m[..., 9] * ux**2 * uy
                    - 2 * m[..., 4] * ux * uz**2
                    - 2 * m[..., 6] * ux**2 * uz
                    - m[..., 7] * uy * uz**2
                    + m[..., 2] * ux**2 * uz**2
                    + 2 * m[..., 11] * ux * uy
                    + 4 * m[..., 16] * ux * uz
                    + 2 * m[..., 13] * uy * uz
                    - 4 * m[..., 5] * ux * uy * uz
                    + 2 * m[..., 1] * ux * uy * uz**2
                    + 2 * m[..., 3] * ux**2 * uy * uz
                    - m[..., 0] * ux**2 * uy * uz**2
                )
                T[25] = (
                    m[..., 25]
                    - 2 * m[..., 21] * ux
                    - 2 * m[..., 20] * uy
                    - m[..., 17] * uz
                    + m[..., 15] * ux**2
                    + m[..., 13] * uy**2
                    - 2 * m[..., 5] * ux * uy**2
                    - 2 * m[..., 6] * ux**2 * uy
                    - m[..., 8] * ux**2 * uz
                    - m[..., 7] * uy**2 * uz
                    + m[..., 3] * ux**2 * uy**2
                    + 4 * m[..., 16] * ux * uy
                    + 2 * m[..., 10] * ux * uz
                    + 2 * m[..., 12] * uy * uz
                    - 4 * m[..., 4] * ux * uy * uz
                    + 2 * m[..., 1] * ux * uy**2 * uz
                    + 2 * m[..., 2] * ux**2 * uy * uz
                    - m[..., 0] * ux**2 * uy**2 * uz
                )
                T[26] = (
                    m[..., 0] * ux**2 * uy**2 * uz**2
                    - 2 * m[..., 3] * ux**2 * uy**2 * uz
                    + m[..., 9] * ux**2 * uy**2
                    - 2 * m[..., 2] * ux**2 * uy * uz**2
                    + 4 * m[..., 6] * ux**2 * uy * uz
                    - 2 * m[..., 14] * ux**2 * uy
                    + m[..., 8] * ux**2 * uz**2
                    - 2 * m[..., 15] * ux**2 * uz
                    + m[..., 19] * ux**2
                    - 2 * m[..., 1] * ux * uy**2 * uz**2
                    + 4 * m[..., 5] * ux * uy**2 * uz
                    - 2 * m[..., 11] * ux * uy**2
                    + 4 * m[..., 4] * ux * uy * uz**2
                    - 8 * m[..., 16] * ux * uy * uz
                    + 4 * m[..., 22] * ux * uy
                    - 2 * m[..., 10] * ux * uz**2
                    + 4 * m[..., 21] * ux * uz
                    - 2 * m[..., 23] * ux
                    + m[..., 7] * uy**2 * uz**2
                    - 2 * m[..., 13] * uy**2 * uz
                    + m[..., 18] * uy**2
                    - 2 * m[..., 12] * uy * uz**2
                    + 4 * m[..., 20] * uy * uz
                    - 2 * m[..., 24] * uy
                    + m[..., 17] * uz**2
                    - 2 * m[..., 25] * uz
                    + m[..., 26]
                )
                T = stack_components(T, m)

                return T

            return tree_map(lambda m, u: shift(m, u), m_tree, u_tree)

    @partial(jit, static_argnums=(0,))
    def compute_central_moment_inverse(self, T_tree, u_tree):
        if isinstance(self.lattice, LatticeD2Q9):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                usq = ux**2 + uy**2
                udiff = ux**2 - uy**2
                m = {}
                m[0] = T[..., 0]
                m[1] = ux * T[..., 0] + T[..., 1]
                m[2] = uy * T[..., 0] + T[..., 2]
                m[3] = usq * T[..., 0] + 2 * ux * T[..., 1] + 2 * uy * T[..., 2] + T[..., 3]
                m[4] = udiff * T[..., 0] + 2 * ux * T[..., 1] - 2 * uy * T[..., 2] + T[..., 4]
                m[5] = ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 5]
                m[6] = (
                    (ux**2) * uy * T[..., 0]
                    + 2 * ux * uy * T[..., 1]
                    + ux**2 * T[..., 2]
                    + 0.5 * uy * T[..., 3]
                    + 0.5 * uy * T[..., 4]
                    + 2 * ux * T[..., 5]
                    + T[..., 6]
                )
                m[7] = (
                    (uy**2) * ux * T[..., 0]
                    + uy**2 * T[..., 1]
                    + 2 * ux * uy * T[..., 2]
                    + 0.5 * ux * T[..., 3]
                    - 0.5 * ux * T[..., 4]
                    + 2 * uy * T[..., 5]
                    + T[..., 7]
                )
                m[8] = (
                    (uy**2 * ux**2) * T[..., 0]
                    + 2 * ux * uy**2 * T[..., 1]
                    + 2 * uy * ux**2 * T[..., 2]
                    + 0.5 * usq * T[..., 3]
                    - 0.5 * udiff * T[..., 4]
                    + 4 * ux * uy * T[..., 5]
                    + 2 * uy * T[..., 6]
                    + 2 * ux * T[..., 7]
                    + T[..., 8]
                )
                m = stack_components(m, T)
                return m

            return tree_map(lambda T, u: shift_inverse(T, u), T_tree, u_tree)

        elif isinstance(self.lattice, LatticeD3Q19):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                m = {}
                m[0] = T[..., 0]
                m[1] = ux * T[..., 0] + T[..., 1]
                m[2] = uy * T[..., 0] + T[..., 2]
                m[3] = uz * T[..., 0] + T[..., 3]
                m[4] = ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 4]
                m[5] = ux * uz * T[..., 0] + uz * T[..., 1] + ux * T[..., 3] + T[..., 5]
                m[6] = uy * uz * T[..., 0] + uz * T[..., 2] + uy * T[..., 3] + T[..., 6]
                m[7] = (ux**2) * T[..., 0] + 2 * ux * T[..., 1] + T[..., 7]
                m[8] = (uy**2) * T[..., 0] + 2 * uy * T[..., 2] + T[..., 8]
                m[9] = (uz**2) * T[..., 0] + 2 * uz * T[..., 3] + T[..., 9]
                m[10] = ux * (uy**2) * T[..., 0] + (uy**2) * T[..., 1] + 2 * ux * uy * T[..., 2] + 2 * uy * T[..., 4] + ux * T[..., 8] + T[..., 10]
                m[11] = ux * (uz**2) * T[..., 0] + (uz**2) * T[..., 1] + 2 * ux * uz * T[..., 3] + 2 * uz * T[..., 5] + ux * T[..., 9] + T[..., 11]
                m[12] = (ux**2) * uy * T[..., 0] + 2 * ux * uy * T[..., 1] + (ux**2) * T[..., 2] + 2 * ux * T[..., 4] + uy * T[..., 7] + T[..., 12]
                m[13] = (ux**2) * uz * T[..., 0] + 2 * ux * uz * T[..., 1] + (ux**2) * T[..., 3] + 2 * ux * T[..., 5] + uz * T[..., 7] + T[..., 13]
                m[14] = uy * (uz**2) * T[..., 0] + (uz**2) * T[..., 2] + 2 * uy * uz * T[..., 3] + 2 * uz * T[..., 6] + uy * T[..., 9] + T[..., 14]
                m[15] = (uy**2) * uz * T[..., 0] + 2 * uy * uz * T[..., 2] + (uy**2) * T[..., 3] + 2 * uy * T[..., 6] + uz * T[..., 8] + T[..., 15]
                m[16] = (
                    (ux**2) * (uy**2) * T[..., 0]
                    + 2 * ux * (uy**2) * T[..., 1]
                    + 2 * uy * (ux**2) * T[..., 2]
                    + 4 * ux * uy * T[..., 4]
                    + (uy**2) * T[..., 7]
                    + (ux**2) * T[..., 8]
                    + 2 * ux * T[..., 10]
                    + 2 * uy * T[..., 12]
                    + T[..., 16]
                )
                m[17] = (
                    (ux**2) * (uz**2) * T[..., 0]
                    + 2 * ux * (uz**2) * T[..., 1]
                    + 2 * uz * (ux**2) * T[..., 3]
                    + 4 * ux * uz * T[..., 5]
                    + (uz**2) * T[..., 7]
                    + (ux**2) * T[..., 9]
                    + 2 * ux * T[..., 11]
                    + 2 * uz * T[..., 13]
                    + T[..., 17]
                )
                m[18] = (
                    (uy**2) * (uz**2) * T[..., 0]
                    + 2 * uy * (uz**2) * T[..., 2]
                    + 2 * uz * (uy**2) * T[..., 3]
                    + 4 * uy * uz * T[..., 6]
                    + (uz**2) * T[..., 8]
                    + (uy**2) * T[..., 9]
                    + 2 * uy * T[..., 14]
                    + 2 * uz * T[..., 15]
                    + T[..., 18]
                )
                m = stack_components(m, T)
                return m

            return tree_map(lambda T, u: shift_inverse(T, u), T_tree, u_tree)

        elif isinstance(self.lattice, LatticeD3Q27):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                m = {}
                m[0] = T[..., 0]
                m[1] = T[..., 1] + T[..., 0] * ux
                m[2] = T[..., 2] + T[..., 0] * uy
                m[3] = T[..., 3] + T[..., 0] * uz
                m[4] = T[..., 4] + T[..., 2] * ux + T[..., 1] * uy + T[..., 0] * ux * uy
                m[5] = T[..., 5] + T[..., 3] * ux + T[..., 1] * uz + T[..., 0] * ux * uz
                m[6] = T[..., 6] + T[..., 3] * uy + T[..., 2] * uz + T[..., 0] * uy * uz
                m[7] = T[..., 0] * ux**2 + 2 * T[..., 1] * ux + T[..., 7]
                m[8] = T[..., 0] * uy**2 + 2 * T[..., 2] * uy + T[..., 8]
                m[9] = T[..., 0] * uz**2 + 2 * T[..., 3] * uz + T[..., 9]
                m[10] = T[..., 10] + T[..., 8] * ux + 2 * T[..., 4] * uy + T[..., 1] * uy**2 + T[..., 0] * ux * uy**2 + 2 * T[..., 2] * ux * uy
                m[11] = T[..., 11] + T[..., 9] * ux + 2 * T[..., 5] * uz + T[..., 1] * uz**2 + T[..., 0] * ux * uz**2 + 2 * T[..., 3] * ux * uz
                m[12] = T[..., 12] + 2 * T[..., 4] * ux + T[..., 7] * uy + T[..., 2] * ux**2 + T[..., 0] * ux**2 * uy + 2 * T[..., 1] * ux * uy
                m[13] = T[..., 13] + 2 * T[..., 5] * ux + T[..., 7] * uz + T[..., 3] * ux**2 + T[..., 0] * ux**2 * uz + 2 * T[..., 1] * ux * uz
                m[14] = T[..., 14] + T[..., 9] * uy + 2 * T[..., 6] * uz + T[..., 2] * uz**2 + T[..., 0] * uy * uz**2 + 2 * T[..., 3] * uy * uz
                m[15] = T[..., 15] + 2 * T[..., 6] * uy + T[..., 8] * uz + T[..., 3] * uy**2 + T[..., 0] * uy**2 * uz + 2 * T[..., 2] * uy * uz
                m[16] = (
                    T[..., 16]
                    + T[..., 6] * ux
                    + T[..., 5] * uy
                    + T[..., 4] * uz
                    + T[..., 3] * ux * uy
                    + T[..., 2] * ux * uz
                    + T[..., 1] * uy * uz
                    + T[..., 0] * ux * uy * uz
                )
                m[17] = (
                    T[..., 0] * ux**2 * uy**2
                    + 2 * T[..., 2] * ux**2 * uy
                    + T[..., 8] * ux**2
                    + 2 * T[..., 1] * ux * uy**2
                    + 4 * T[..., 4] * ux * uy
                    + 2 * T[..., 10] * ux
                    + T[..., 7] * uy**2
                    + 2 * T[..., 12] * uy
                    + T[..., 17]
                )
                m[18] = (
                    T[..., 0] * ux**2 * uz**2
                    + 2 * T[..., 3] * ux**2 * uz
                    + T[..., 9] * ux**2
                    + 2 * T[..., 1] * ux * uz**2
                    + 4 * T[..., 5] * ux * uz
                    + 2 * T[..., 11] * ux
                    + T[..., 7] * uz**2
                    + 2 * T[..., 13] * uz
                    + T[..., 18]
                )
                m[19] = (
                    T[..., 0] * uy**2 * uz**2
                    + 2 * T[..., 3] * uy**2 * uz
                    + T[..., 9] * uy**2
                    + 2 * T[..., 2] * uy * uz**2
                    + 4 * T[..., 6] * uy * uz
                    + 2 * T[..., 14] * uy
                    + T[..., 8] * uz**2
                    + 2 * T[..., 15] * uz
                    + T[..., 19]
                )
                m[20] = (
                    T[..., 20]
                    + 2 * T[..., 16] * ux
                    + T[..., 13] * uy
                    + T[..., 12] * uz
                    + T[..., 6] * ux**2
                    + T[..., 3] * ux**2 * uy
                    + T[..., 2] * ux**2 * uz
                    + 2 * T[..., 5] * ux * uy
                    + 2 * T[..., 4] * ux * uz
                    + T[..., 7] * uy * uz
                    + 2 * T[..., 1] * ux * uy * uz
                    + T[..., 0] * ux**2 * uy * uz
                )
                m[21] = (
                    T[..., 21]
                    + T[..., 15] * ux
                    + 2 * T[..., 16] * uy
                    + T[..., 10] * uz
                    + T[..., 5] * uy**2
                    + T[..., 3] * ux * uy**2
                    + T[..., 1] * uy**2 * uz
                    + 2 * T[..., 6] * ux * uy
                    + T[..., 8] * ux * uz
                    + 2 * T[..., 4] * uy * uz
                    + 2 * T[..., 2] * ux * uy * uz
                    + T[..., 0] * ux * uy**2 * uz
                )
                m[22] = (
                    T[..., 22]
                    + T[..., 14] * ux
                    + T[..., 11] * uy
                    + 2 * T[..., 16] * uz
                    + T[..., 4] * uz**2
                    + T[..., 2] * ux * uz**2
                    + T[..., 1] * uy * uz**2
                    + T[..., 9] * ux * uy
                    + 2 * T[..., 6] * ux * uz
                    + 2 * T[..., 5] * uy * uz
                    + 2 * T[..., 3] * ux * uy * uz
                    + T[..., 0] * ux * uy * uz**2
                )
                m[23] = (
                    T[..., 23]
                    + T[..., 19] * ux
                    + 2 * T[..., 22] * uy
                    + 2 * T[..., 21] * uz
                    + T[..., 11] * uy**2
                    + T[..., 10] * uz**2
                    + T[..., 9] * ux * uy**2
                    + T[..., 8] * ux * uz**2
                    + 2 * T[..., 4] * uy * uz**2
                    + 2 * T[..., 5] * uy**2 * uz
                    + T[..., 1] * uy**2 * uz**2
                    + 2 * T[..., 14] * ux * uy
                    + 2 * T[..., 15] * ux * uz
                    + 4 * T[..., 16] * uy * uz
                    + 4 * T[..., 6] * ux * uy * uz
                    + 2 * T[..., 2] * ux * uy * uz**2
                    + 2 * T[..., 3] * ux * uy**2 * uz
                    + T[..., 0] * ux * uy**2 * uz**2
                )
                m[24] = (
                    T[..., 24]
                    + 2 * T[..., 22] * ux
                    + T[..., 18] * uy
                    + 2 * T[..., 20] * uz
                    + T[..., 14] * ux**2
                    + T[..., 12] * uz**2
                    + T[..., 9] * ux**2 * uy
                    + 2 * T[..., 4] * ux * uz**2
                    + 2 * T[..., 6] * ux**2 * uz
                    + T[..., 7] * uy * uz**2
                    + T[..., 2] * ux**2 * uz**2
                    + 2 * T[..., 11] * ux * uy
                    + 4 * T[..., 16] * ux * uz
                    + 2 * T[..., 13] * uy * uz
                    + 4 * T[..., 5] * ux * uy * uz
                    + 2 * T[..., 1] * ux * uy * uz**2
                    + 2 * T[..., 3] * ux**2 * uy * uz
                    + T[..., 0] * ux**2 * uy * uz**2
                )
                m[25] = (
                    T[..., 25]
                    + 2 * T[..., 21] * ux
                    + 2 * T[..., 20] * uy
                    + T[..., 17] * uz
                    + T[..., 15] * ux**2
                    + T[..., 13] * uy**2
                    + 2 * T[..., 5] * ux * uy**2
                    + 2 * T[..., 6] * ux**2 * uy
                    + T[..., 8] * ux**2 * uz
                    + T[..., 7] * uy**2 * uz
                    + T[..., 3] * ux**2 * uy**2
                    + 4 * T[..., 16] * ux * uy
                    + 2 * T[..., 10] * ux * uz
                    + 2 * T[..., 12] * uy * uz
                    + 4 * T[..., 4] * ux * uy * uz
                    + 2 * T[..., 1] * ux * uy**2 * uz
                    + 2 * T[..., 2] * ux**2 * uy * uz
                    + T[..., 0] * ux**2 * uy**2 * uz
                )
                m[26] = (
                    T[..., 0] * ux**2 * uy**2 * uz**2
                    + 2 * T[..., 3] * ux**2 * uy**2 * uz
                    + T[..., 9] * ux**2 * uy**2
                    + 2 * T[..., 2] * ux**2 * uy * uz**2
                    + 4 * T[..., 6] * ux**2 * uy * uz
                    + 2 * T[..., 14] * ux**2 * uy
                    + T[..., 8] * ux**2 * uz**2
                    + 2 * T[..., 15] * ux**2 * uz
                    + T[..., 19] * ux**2
                    + 2 * T[..., 1] * ux * uy**2 * uz**2
                    + 4 * T[..., 5] * ux * uy**2 * uz
                    + 2 * T[..., 11] * ux * uy**2
                    + 4 * T[..., 4] * ux * uy * uz**2
                    + 8 * T[..., 16] * ux * uy * uz
                    + 4 * T[..., 22] * ux * uy
                    + 2 * T[..., 10] * ux * uz**2
                    + 4 * T[..., 21] * ux * uz
                    + 2 * T[..., 23] * ux
                    + T[..., 7] * uy**2 * uz**2
                    + 2 * T[..., 13] * uy**2 * uz
                    + T[..., 18] * uy**2
                    + 2 * T[..., 12] * uy * uz**2
                    + 4 * T[..., 20] * uy * uz
                    + 2 * T[..., 24] * uy
                    + T[..., 17] * uz**2
                    + 2 * T[..., 25] * uz
                    + T[..., 26]
                )
                m = stack_components(m, T)
                return m

            return tree_map(lambda T, u: shift_inverse(T, u), T_tree, u_tree)

    @partial(jit, static_argnums=(0,))
    def compute_eq_central_moments(self, rho_tree):
        """
        Calculate the central moments of the equilibrium distribution.

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Density field for all components.

        Returns
        -------
        T_eq_tree (pytree of jax.numpy.ndarray): Central moments of the equilibrium distribution.
        """

        def f(rho):
            if isinstance(self.lattice, LatticeD2Q9):
                T_eq = {}
                T_eq[0] = rho[..., 0]
                T_eq[3] = 2 * rho[..., 0] * self.lattice.cs2
                T_eq[8] = rho[..., 0] * self.lattice.cs**4
                T_eq = stack_components(T_eq, jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precision_policy.compute_dtype))

                return T_eq

            elif isinstance(self.lattice, LatticeD3Q19):
                T_eq = {}
                T_eq[0] = rho[..., 0]
                T_eq[7] = rho[..., 0] * self.lattice.cs2
                T_eq[8] = rho[..., 0] * self.lattice.cs2
                T_eq[9] = rho[..., 0] * self.lattice.cs2
                T_eq[16] = rho[..., 0] * self.lattice.cs**4
                T_eq[17] = rho[..., 0] * self.lattice.cs**4
                T_eq[18] = rho[..., 0] * self.lattice.cs**4
                T_eq = stack_components(T_eq, jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype))

                return T_eq

            elif isinstance(self.lattice, LatticeD3Q27):
                T_eq = {}
                T_eq[0] = rho[..., 0]
                T_eq[7] = rho[..., 0] * self.lattice.cs2
                T_eq[8] = rho[..., 0] * self.lattice.cs2
                T_eq[9] = rho[..., 0] * self.lattice.cs2
                T_eq[17] = rho[..., 0] * self.lattice.cs**4
                T_eq[18] = rho[..., 0] * self.lattice.cs**4
                T_eq[19] = rho[..., 0] * self.lattice.cs**4
                T_eq[26] = rho[..., 0] * self.lattice.cs**6
                T_eq = stack_components(T_eq, jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype))

                return T_eq

        return tree_map(lambda rho: f(rho), rho_tree)

    @partial(jit, static_argnums=(0,))
    def compute_force_central_moments(self, F_tree, psi_tree):
        """
        Calculate the central moments of the force distribution. Includes modification to accurately replicate mechanical stability conditions.

        Parameters
        ----------
        F_tree (pytree of jax.numpy.ndarray): Force field.

        psi_tree (pytree of jax.numpy.ndarray): Potential field.

        Returns
        -------
        T_eq_tree (pytree of jax.numpy.ndarray): Central moments of the force distribution.
        """

        def f(F, sigma, psi, s_b):
            if isinstance(self.lattice, LatticeD2Q9):
                C = {}
                Fx = F[..., 0]
                Fy = F[..., 1]
                eta = 4 * sigma * (Fx**2 + Fy**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))  # For mechanical stability
                C[1] = Fx
                C[2] = Fy
                C[3] = eta
                C[6] = Fy * self.lattice.cs2
                C[7] = Fx * self.lattice.cs2
                C[8] = eta * self.lattice.cs2
                C = stack_components(C, jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precision_policy.compute_dtype))

                return C
            elif isinstance(self.lattice, LatticeD3Q19):
                C = {}
                Fx = F[..., 0]
                Fy = F[..., 1]
                Fz = F[..., 2]
                eta = 4 * sigma * (Fx**2 + Fy**2 + Fz**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))
                C[1] = Fx
                C[2] = Fy
                C[3] = Fz
                C[7] = eta
                C[8] = eta
                C[9] = eta
                C[10] = Fx * self.lattice.cs2
                C[11] = Fx * self.lattice.cs2
                C[12] = Fy * self.lattice.cs2
                C[13] = Fz * self.lattice.cs2
                C[14] = Fy * self.lattice.cs2
                C[15] = Fz * self.lattice.cs2
                C = stack_components(C, jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype))

                return C
            elif isinstance(self.lattice, LatticeD3Q27):
                C = {}
                Fx = F[..., 0]
                Fy = F[..., 1]
                Fz = F[..., 2]
                C[1] = Fx
                C[2] = Fy
                C[3] = Fz
                eta = 4 * sigma * (Fx**2 + Fy**2 + Fz**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))
                C[7] = eta
                C[8] = eta
                C[9] = eta
                C[10] = Fx * self.lattice.cs2
                C[11] = Fx * self.lattice.cs2
                C[12] = Fy * self.lattice.cs2
                C[13] = Fz * self.lattice.cs2
                C[14] = Fy * self.lattice.cs2
                C[15] = Fz * self.lattice.cs2
                C[23] = Fx * self.lattice.cs**4
                C[24] = Fy * self.lattice.cs**4
                C[25] = Fz * self.lattice.cs**4
                C = stack_components(C, jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype))

                return C

        return tree_map(lambda F, sigma, psi, s_b: f(F, sigma, psi, s_b), F_tree, self.sigma, psi_tree, self.s_b)

    @partial(jit, static_argnums=(0,), inline=True)
    def _apply_force_fields(self, Tdash_tree, F_tree, psi_tree):
        """Apply Cascaded forcing from force and potential fields already evaluated by collision."""
        C_tree = self.compute_force_central_moments(F_tree, psi_tree)
        Tf_tree = tree_map(lambda S, C: jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * S), self.S, C_tree)
        return tree_map(lambda Tdash, Tf: Tdash + Tf, Tdash_tree, Tf_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, Tdash_tree, rho_tree, u_tree, T=None):
        """
        Modified version of the apply_force defined in LBMBase to account for modified force.

        Parameters
        ----------
        Tdash_tree (pytree of jax.numpy.ndarray): Central moments of post-collision distribution functions.

        rho_tree (pytree of jax.numpy.ndarray): Density field.

        u_tree (pytree of jax.numpy.ndarray): Velocity field.

        T (jax.numpy.ndarray, optional): Temperature field, required when the EOS is thermal.

        Returns
        -------
        f_postcollision_tree (pytree of jax.numpy.ndarray): The post-collision distribution functions with the force applied.
        """
        F_tree = self.compute_force(rho_tree, T=T)
        psi_tree = self.compute_potential(rho_tree, T=T)
        return self._apply_force_fields(Tdash_tree, F_tree, psi_tree)

    def _collision_applies_bounceback(self):
        """The per-direction collision applies the full-way bounce-back itself (see LBMBase.apply_fused_bounceback)."""
        return type(self).collision is MultiphaseCascade.collision and self._uses_component_collision()

    def _uses_component_collision(self):
        """
        Whether collision() takes the per-direction (component-mode) formulation: always, unless a subclass overrides the forcing
        step on stacked (..., q) moments (apply_force or _apply_force_fields), which only the array formulation calls. Overridden
        moment transforms (compute_central_moment, compute_central_moment_inverse, compute_eq_central_moments,
        compute_force_central_moments) are traced in component mode, so they must read components as x[..., k] and build their
        output with stack_components.
        """
        return self._uses_default_cascade_apply_force and type(self)._apply_force_fields is MultiphaseCascade._apply_force_fields

    def _collision_components(self, fin_tree, rho_tree, u_tree, F_tree, psi_tree):
        """
        The cascaded collision evaluated per lattice direction (component mode).

        Same algorithm as the array path of collision(): raw moments f @ M, shift to central moments, relaxation
        Tdash @ (I - S) + Tdash_eq @ S, forcing C @ (I - S / 2), shift back, populations @ M_inv. Here the (..., q) intermediates
        are kept as q separate arrays (Components), the transforms are the same methods traced in component mode, and the matrix
        products are sparse sums with constant coefficients (as in MultiphaseMRT). XLA then compiles the whole collision into one
        kernel with one thread per node, instead of materializing every intermediate through separate kernels and GEMMs. The
        products are exact float32/float64 sums, so results can differ from the GEMM path at roundoff (or TF32) level.
        """
        dtype = self.precision_policy.compute_dtype
        q = self.lattice.q
        eye = np.eye(q)
        # The class's own (possibly overridden) transforms, unjitted, so that they trace in component mode
        transform = lambda name: getattr(getattr(type(self), name), "__wrapped__", getattr(type(self), name))
        with component_mode():
            m_tree = [
                matvec_components(Components([f[..., i] for i in range(q)], dtype), self.sparse_collision_terms(np.asarray(M)), dtype)
                for f, M in zip(fin_tree, self.M, strict=True)
            ]
            Tdash_tree = transform("compute_central_moment")(self, m_tree, u_tree)
            Tdash_eq_tree = transform("compute_eq_central_moments")(self, rho_tree)
            C_tree = transform("compute_force_central_moments")(self, F_tree, psi_tree)
            Tout_tree = []
            for Tdash, Tdash_eq, C, S in zip(Tdash_tree, Tdash_eq_tree, C_tree, self.S, strict=True):
                S = np.asarray(S)
                relaxed = matvec_components(Tdash, self.sparse_collision_terms(eye - S), dtype)
                equilibrium = matvec_components(Tdash_eq, self.sparse_collision_terms(S), dtype)
                forcing = matvec_components(C, self.sparse_collision_terms(eye - 0.5 * S), dtype)
                Tout_tree.append(Components([a + b + c for a, b, c in zip(relaxed, equilibrium, forcing, strict=True)], dtype))
            m_out_tree = transform("compute_central_moment_inverse")(self, Tout_tree, u_tree)
            fout_tree = [
                matvec_components(m, self.sparse_collision_terms(np.asarray(M_inv)), dtype) for m, M_inv in zip(m_out_tree, self.M_inv, strict=True)
            ]
        populations_tree = []
        for fin, rho, fout, mask in zip(fin_tree, rho_tree, fout_tree, self.bounceback_mask, strict=True):
            fout = [jnp.broadcast_to(x, fin.shape[:-1]) for x in fout]
            if self.wetting_formulation == "geometric" and self.dim == 3:
                # Preserve the density moment after the 3D geometric wetting update by applying any roundoff-level mismatch to the rest population.
                fout[0] = fout[0] + (rho[..., 0] - sum(fout))
            fout = self.apply_fused_bounceback(fout, [fin[..., i] for i in range(self.q)], mask)
            populations_tree.append(self.stack_populations([self.precision_policy.cast_to_output(x) for x in fout]))
        return populations_tree

    @partial(jit, static_argnums=(0,))
    def collision(self, fin_tree, T=None):
        """
        Cascaded LBM collision step for lattice. The optional temperature field
        T is forwarded to the pressure and force computations for thermal EOS.
        """
        fin_tree = tree_map(lambda f: self.precision_policy.cast_to_compute(f), fin_tree)
        rho_tree, raw_u_tree = self.update_macroscopic(fin_tree)
        force_psi_tree = None
        F_tree = None
        if not self._uses_default_cascade_apply_force:
            u_tree = self.macroscopic_velocity(fin_tree, rho_tree, T=T)
        elif self._uses_default_macroscopic_velocity:
            if self._uses_default_compute_force:
                F_tree, force_psi_tree = self._compute_force_fields(rho_tree, T=T)
            else:
                F_tree = self.compute_force(rho_tree, T=T)
            u_tree = tree_map(lambda rho, u, F: u + 0.5 * F / rho, rho_tree, raw_u_tree, F_tree)
        else:
            u_tree = self.macroscopic_velocity(fin_tree, rho_tree, T=T)
            F_tree = self.compute_force(rho_tree, T=T)
        if self._uses_component_collision():
            if force_psi_tree is None or (self.wetting_formulation is not None and any(self._has_wetting_bc)):
                psi_tree = self.compute_potential(rho_tree, T=T)
            else:
                psi_tree = force_psi_tree
            return self._collision_components(fin_tree, rho_tree, u_tree, F_tree, psi_tree)
        T_tree = tree_map(lambda f, M: jnp.dot(f, M), fin_tree, self.M)
        Tdash_tree = self.compute_central_moment(T_tree, u_tree)
        Tdash_eq_tree = self.compute_eq_central_moments(rho_tree)
        Tout_tree = tree_map(
            lambda Tdash, Tdash_eq, S: jnp.dot(Tdash, jnp.eye(self.lattice.q) - S) + jnp.dot(Tdash_eq, S), Tdash_tree, Tdash_eq_tree, self.S
        )
        if not self._uses_default_cascade_apply_force:
            Tout_tree = self.apply_force(Tout_tree, rho_tree, u_tree, T=T)
        else:
            if force_psi_tree is None or (self.wetting_formulation is not None and any(self._has_wetting_bc)):
                psi_tree = self.compute_potential(rho_tree, T=T)
            else:
                psi_tree = force_psi_tree
            Tout_tree = self._apply_force_fields(Tout_tree, F_tree, psi_tree)
        Tout_tree = self.compute_central_moment_inverse(Tout_tree, u_tree)
        fout_tree = tree_map(lambda T, Minv: jnp.dot(T, Minv), Tout_tree, self.M_inv)
        if self.wetting_formulation == "geometric" and self.dim == 3:
            # Preserve the density moment after the 3D geometric wetting update by applying any roundoff-level mismatch to the rest population.
            rho_out_tree = tree_map(lambda fout: jnp.sum(fout, axis=-1, keepdims=True), fout_tree)
            fout_tree = tree_map(lambda fout, rho, rho_out: fout.at[..., 0].add((rho - rho_out)[..., 0]), fout_tree, rho_tree, rho_out_tree)
        return tree_map(lambda fout: self.precision_policy.cast_to_output(fout), fout_tree)
