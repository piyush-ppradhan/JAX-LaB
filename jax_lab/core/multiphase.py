"""
Definition of Multiphase class for simulating a multiphase flow.
"""

import logging
import operator
import time

# System libraries
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orb
from jax import jit, shard_map, vmap
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding, PartitionSpec

# Third-party libraries
from jax.tree import map as tree_map
from jax.tree import reduce

from .base import WALL_BC_TYPES, LBMBase

# User-defined libraries
from .boundary_conditions import BounceBack, BounceBackHalfway, BounceBackMoving, InterpolatedBounceBackBouzidi, InterpolatedBounceBackDifferentiable
from .lattice import LatticeD2Q9, LatticeD3Q19, LatticeD3Q27
from .utils import colored, downsample_field

logger = logging.getLogger(__name__)

# This significantly reduces the performance. Use if necessary
# jax.config.update("jax_debug_nans", True)


class Multiphase(LBMBase):
    """
    Multiphase model based on the Shan-Chen method.

    The user supplies an equation of state (EOS). Pressure is evaluated from density and temperature before the effective mass. Both single-component
    multiphase and multicomponent multiphase systems are supported.

    Parameters
    ----------
    k (list): Modification coefficient used to tune surface tension.

    A (numpy.ndarray): Weighting factor for combining the Shan-Chen and Zhang-Chen forces.

    g_kkprime (numpy.ndarray): Symmetric component-interaction matrix with shape ``(n_components, n_components)``.

    wetting_formulation (str or None, optional): Contact-angle scheme. Select ``"geometric"`` or
    ``"improved_virtual_density"`` when a boundary condition defines ``theta``. Defaults to ``None``.

    References
    ----------
    1. Shan, Xiaowen, and Hudong Chen. “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.”
        Physical Review E 47, no. 3 (March 1, 1993): 1815-19. https://doi.org/10.1103/PhysRevE.47.1815.

    2. Yuan, Peng, and Laura Schaefer. “Equations of State in a Lattice Boltzmann Model.”
        Physics of Fluids 18, no. 4 (April 3, 2006): 042101. https://doi.org/10.1063/1.2187070.

    Notes
    -----
    1. Boundary conditions are handled separately for each component. For example, define a wall condition once per component in a two-component system.
    2. Pytrees contain one leaf per component in the order defined by ``initialize_macroscopic_fields``.
    3. Component-specific lists and arrays must use the same ordering.
    """

    def __init__(self, **kwargs):
        self.n_components = kwargs.get("n_components")
        super().__init__(**kwargs)
        self.k = kwargs.get("k")
        self.A = kwargs.get("A")
        self.eos = kwargs.get("EOS", None)
        self.g_kkprime = kwargs.get("g_kkprime")  # Fluid-fluid interaction strength
        self.body_force = kwargs.get("body_force", None)
        self.wetting_formulation = kwargs.get("wetting_formulation")

        self._has_wetting_bc = tuple(
            any(self._is_wetting_boundary_condition(bc) and bc.theta is not None for bc in component_bcs) for component_bcs in self.BCs
        )
        if self.wetting_formulation is None and any(self._has_wetting_bc):
            raise ValueError(
                "A wetting_formulation must be selected when a boundary condition defines theta. "
                "Supported schemes: geometric and improved_virtual_density."
            )

        if self.wetting_formulation == "geometric":
            self.computed_nearest_next_nearest_nbr = False

        self.G_ff = self.compute_ff_greens_function()
        self.g_kkprime = jnp.array(self.g_kkprime, dtype=self.precision_policy.compute_dtype)

        P = PartitionSpec
        scalar_spec = P("x", None, None) if self.dim == 2 else P("x", None, None, None)
        G_ff_host = np.array(self.G_ff)
        # scalar_neighbor_sum: G_ff-weighted scalar neighbor sum, used by the wetting/average-density
        # denominator. scalar_force_stencil: the same neighbor structure, weighted by G_ff*c (a per-direction
        # vector instead of a scalar), used by the Shan-Chen/Zhang-Chen fluid-fluid force - both share
        # _neighbor_stencil_m's one-x-halo-exchange machinery, bound to their own static weights.
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
            if self.wetting_formulation == "improved_virtual_density" and any(self._has_wetting_bc)
            else None
        )
        self.scalar_force_stencil = jit(
            shard_map(
                partial(self._neighbor_stencil_m, weights=G_ff_host[None, :] * np.array(self.c)),
                mesh=self.mesh,
                in_specs=scalar_spec,
                out_specs=scalar_spec,
                check_vma=False,
            )
        )

        self.solid_mask_streamed = None
        self.average_density_denominator = None
        if self.scalar_neighbor_sum is not None:
            self.solid_mask_streamed = self.get_solid_mask_streamed()
            self.average_density_denominator = [
                self.scalar_neighbor_sum(1 - mask) if has_wetting_bc else None
                for mask, has_wetting_bc in zip(self.solid_mask_streamed, self._has_wetting_bc, strict=True)
            ]
        if self.average_density_denominator is not None:
            for denominator in self.average_density_denominator:
                if denominator is not None:
                    denominator.block_until_ready()
        self.geometric_wetting_data, self.geometric_fluid_mask = (
            self._create_geometric_wetting_data() if self.wetting_formulation == "geometric" and any(self._has_wetting_bc) else (None, None)
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
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        if value is None:
            raise ValueError("Modification coefficient must be provided")
        if isinstance(value, float) or isinstance(value, int):
            if self.n_components != 1:
                raise ValueError("The number of modification coefficients provided does not match the number of components in the system")
            self._k = [value]
        elif isinstance(value, list):
            if len(value) != self.n_components:
                raise ValueError("The number of modification coefficients provided does not match the number of components in the system")
        self._k = value

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        if value is None:
            raise ValueError("Weight coefficient value must be provided")
        if isinstance(value, np.ndarray):
            if value.shape != (self.n_components, self.n_components):
                raise ValueError("The dimensions of A should match the number of components")
        self._A = jnp.array(value, dtype=self.precision_policy.compute_dtype)

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

    def _compute_geometric_normals(self, bc, solid_mask):
        """
        Compute normals for geometric wetting using boundary data and solid mask.

        Parameters
        ----------
        bc (BoundaryCondition): Boundary condition with wettability data.

        solid_mask (numpy.ndarray): Boolean mask with True on boundary nodes.

        Returns
        -------
        normals (numpy.ndarray): Unit normals pointing from wall nodes toward fluid nodes.
        """
        indices = np.array(self._get_solid_indices(bc), dtype=np.int64).T
        normals = np.zeros((indices.shape[0], self.dim), dtype=np.float64)

        # bc.normals rows correspond to bc.indices; for halfway bounce-back those are the shifted
        # fluid nodes, not the solid nodes used here, so fall back to the neighbor-based normals.
        if bc.is_solid and hasattr(bc, "normals") and not hasattr(bc, "solid_indices"):
            bc_normals = np.asarray(bc.normals, dtype=np.float64)
            if bc_normals.shape == normals.shape:
                normal_norm = np.linalg.norm(bc_normals, axis=1, keepdims=True)
                normals = np.divide(bc_normals, normal_norm, out=normals, where=normal_norm > 1e-12)

        c = np.array(self.lattice.c).T
        c = c[np.linalg.norm(c, axis=1) > 0]
        missing_normal = np.linalg.norm(normals, axis=1) <= 1e-12
        for i in np.where(missing_normal)[0]:
            idx = indices[i]
            normal = np.zeros((self.dim,), dtype=np.float64)
            for ci in c:
                nbr = idx + ci
                in_bounds = (0 <= nbr[0] < self.nx) and (0 <= nbr[1] < self.ny)
                if self.dim == 3:
                    in_bounds = in_bounds and (0 <= nbr[2] < self.nz)
                if in_bounds and not solid_mask[tuple(nbr)]:
                    normal += ci / np.linalg.norm(ci)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-12:
                normals[i] = normal / normal_norm

        return normals

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
        lattice_directions = np.array(self.lattice.c, dtype=np.int64).T
        lattice_directions = lattice_directions[np.linalg.norm(lattice_directions, axis=1) > 0]
        interface = np.zeros((indices.shape[0],), dtype=bool)

        for i, index in enumerate(indices):
            for direction in lattice_directions:
                nbr = index + direction
                in_bounds = (0 <= nbr[0] < self.nx) and (0 <= nbr[1] < self.ny)
                if self.dim == 3:
                    in_bounds = in_bounds and (0 <= nbr[2] < self.nz)
                if in_bounds and not solid_mask[tuple(nbr)]:
                    interface[i] = True
                    break

        return interface

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
            in_bounds = (0 <= index[0] < self.nx) and (0 <= index[1] < self.ny)
            if self.dim == 3:
                in_bounds = in_bounds and (0 <= index[2] < self.nz)
            if not in_bounds or solid_mask[tuple(index)]:
                return False
        return True

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
        points = self._first_mesh_intersection(indices, directions)
        valid = np.zeros((indices.shape[0],), dtype=bool)
        max_steps = self.nx + self.ny if self.dim == 2 else self.nx + self.ny + self.nz
        for i, (idx, direction) in enumerate(zip(indices, directions)):
            candidates = []
            for component in direction:
                if np.abs(component) > eps:
                    candidates.append(np.arange(1, max_steps + 1, dtype=np.float64) / np.abs(component))
            if not candidates:
                continue
            t_candidates = np.unique(np.round(np.sort(np.concatenate(candidates)), decimals=12))
            for candidate_count, t in enumerate(t_candidates):
                if max_intersections is not None and candidate_count >= max_intersections:
                    break
                point = idx + t * direction
                rounded = np.round(point)
                point = np.where(np.isclose(point, rounded, atol=eps), rounded, point)
                if self._uses_only_fluid_nodes(point, solid_mask):
                    points[i] = point
                    valid[i] = True
                    break
        if return_valid:
            return points, valid
        return points

    def _build_interpolation_data(self, points):
        """
        Build multilinear interpolation data for density samples.

        Parameters
        ----------
        points (numpy.ndarray): Off-lattice or on-lattice sample points.

        Returns
        -------
        data (tuple): Index arrays and weights for multilinear interpolation.
        """
        floor_points = np.floor(points)
        lower = floor_points.astype(np.int64)
        upper = lower + 1
        frac = points - floor_points

        if self.dim == 2:
            x0 = np.clip(lower[:, 0], 0, self.nx - 1)
            y0 = np.clip(lower[:, 1], 0, self.ny - 1)
            x1 = np.clip(upper[:, 0], 0, self.nx - 1)
            y1 = np.clip(upper[:, 1], 0, self.ny - 1)
            wx = frac[:, 0]
            wy = frac[:, 1]

            return (
                jnp.array(x0, dtype=jnp.int32),
                jnp.array(y0, dtype=jnp.int32),
                jnp.array(x1, dtype=jnp.int32),
                jnp.array(y1, dtype=jnp.int32),
                jnp.array((1.0 - wx) * (1.0 - wy), dtype=self.precision_policy.compute_dtype),
                jnp.array(wx * (1.0 - wy), dtype=self.precision_policy.compute_dtype),
                jnp.array((1.0 - wx) * wy, dtype=self.precision_policy.compute_dtype),
                jnp.array(wx * wy, dtype=self.precision_policy.compute_dtype),
            )

        x0 = np.clip(lower[:, 0], 0, self.nx - 1)
        y0 = np.clip(lower[:, 1], 0, self.ny - 1)
        z0 = np.clip(lower[:, 2], 0, self.nz - 1)
        x1 = np.clip(upper[:, 0], 0, self.nx - 1)
        y1 = np.clip(upper[:, 1], 0, self.ny - 1)
        z1 = np.clip(upper[:, 2], 0, self.nz - 1)
        wx = frac[:, 0]
        wy = frac[:, 1]
        wz = frac[:, 2]

        return (
            jnp.array(x0, dtype=jnp.int32),
            jnp.array(y0, dtype=jnp.int32),
            jnp.array(z0, dtype=jnp.int32),
            jnp.array(x1, dtype=jnp.int32),
            jnp.array(y1, dtype=jnp.int32),
            jnp.array(z1, dtype=jnp.int32),
            jnp.array((1.0 - wx) * (1.0 - wy) * (1.0 - wz), dtype=self.precision_policy.compute_dtype),
            jnp.array(wx * (1.0 - wy) * (1.0 - wz), dtype=self.precision_policy.compute_dtype),
            jnp.array((1.0 - wx) * wy * (1.0 - wz), dtype=self.precision_policy.compute_dtype),
            jnp.array(wx * wy * (1.0 - wz), dtype=self.precision_policy.compute_dtype),
            jnp.array((1.0 - wx) * (1.0 - wy) * wz, dtype=self.precision_policy.compute_dtype),
            jnp.array(wx * (1.0 - wy) * wz, dtype=self.precision_policy.compute_dtype),
            jnp.array((1.0 - wx) * wy * wz, dtype=self.precision_policy.compute_dtype),
            jnp.array(wx * wy * wz, dtype=self.precision_policy.compute_dtype),
        )

    def _build_geometric_3d_lattice_data(self, indices, normals, solid_mask):
        """
        Build lattice-node stencil data for the 3D geometric wetting scheme.

        Parameters
        ----------
        indices (numpy.ndarray): Boundary node coordinates with shape (n, 3).

        normals (numpy.ndarray): Unit normals pointing from wall nodes toward fluid nodes.

        solid_mask (numpy.ndarray): Boolean mask with True on boundary nodes.

        Returns
        -------
        active (numpy.ndarray): Boolean mask with True for nodes that have a valid normal stencil.

        normal_2_indices (tuple): JAX index tuple for the second fluid node along the selected normal direction.

        tangent_indices (tuple): JAX index tuples for up to two opposite tangent node pairs around the first normal node.

        tangent_pair_valid (jax.numpy.ndarray): Boolean mask indicating which tangent pairs are valid for each active node.

        Notes
        -----
        This helper constructs the older lattice-node 3D stencil. The active 3D geometric wetting path uses
        _build_geometric_3d_characteristic_data to sample multiple off-lattice characteristic directions.
        """
        lattice_directions = np.array(self.lattice.c, dtype=np.int64).T
        nonzero_direction_indices = np.flatnonzero(np.linalg.norm(lattice_directions, axis=1) > 0)
        nonzero_directions = lattice_directions[nonzero_direction_indices]
        unit_directions = nonzero_directions / np.linalg.norm(nonzero_directions, axis=1, keepdims=True)

        normal_2_indices = np.zeros_like(indices)
        tangent_indices = [np.zeros_like(indices) for _ in range(4)]
        active = np.zeros((indices.shape[0],), dtype=bool)
        tangent_pair_valid = np.zeros((2, indices.shape[0]), dtype=bool)

        for i, (index, normal) in enumerate(zip(indices, normals)):
            fluid_direction = None
            sorted_direction_indices = np.argsort(-(unit_directions @ normal))
            for direction_index in sorted_direction_indices:
                direction = nonzero_directions[direction_index]
                unit_direction = unit_directions[direction_index]
                if unit_direction @ normal <= 1e-12:
                    break
                normal_1 = index + direction
                normal_2 = index + 2 * direction
                in_bounds_1 = (0 <= normal_1[0] < self.nx) and (0 <= normal_1[1] < self.ny) and (0 <= normal_1[2] < self.nz)
                in_bounds_2 = (0 <= normal_2[0] < self.nx) and (0 <= normal_2[1] < self.ny) and (0 <= normal_2[2] < self.nz)
                if in_bounds_1 and in_bounds_2 and not solid_mask[tuple(normal_1)] and not solid_mask[tuple(normal_2)]:
                    fluid_direction = direction
                    normal_2_indices[i] = normal_2
                    active[i] = True
                    break

            if fluid_direction is None:
                continue

            normal_1 = index + fluid_direction
            tangent_candidates = nonzero_directions[nonzero_directions @ fluid_direction == 0]
            tangent_candidates = tangent_candidates[np.linalg.norm(tangent_candidates, axis=1) > 0]
            tangent_scores = np.abs(tangent_candidates @ normal)
            sorted_tangents = tangent_candidates[np.argsort(tangent_scores)]
            selected_tangents = []
            for tangent in sorted_tangents:
                if any(np.all(tangent == selected) or np.all(tangent == -selected) for selected in selected_tangents):
                    continue
                plus = normal_1 + tangent
                minus = normal_1 - tangent
                in_bounds_plus = (0 <= plus[0] < self.nx) and (0 <= plus[1] < self.ny) and (0 <= plus[2] < self.nz)
                in_bounds_minus = (0 <= minus[0] < self.nx) and (0 <= minus[1] < self.ny) and (0 <= minus[2] < self.nz)
                if in_bounds_plus and in_bounds_minus and not solid_mask[tuple(plus)] and not solid_mask[tuple(minus)]:
                    pair_index = len(selected_tangents)
                    selected_tangents.append(tangent)
                    tangent_indices[2 * pair_index][i] = plus
                    tangent_indices[2 * pair_index + 1][i] = minus
                    tangent_pair_valid[pair_index, i] = True
                    if len(selected_tangents) == 2:
                        break

            for pair_index in range(len(selected_tangents), 2):
                tangent_indices[2 * pair_index][i] = normal_1
                tangent_indices[2 * pair_index + 1][i] = normal_1

        if not np.any(active):
            empty = tuple(jnp.array([], dtype=jnp.int32) for _ in range(3))
            return active, empty, (), jnp.array([], dtype=jnp.bool_)

        return (
            active,
            tuple(jnp.array(index, dtype=jnp.int32) for index in normal_2_indices[active].T),
            tuple(tuple(jnp.array(index, dtype=jnp.int32) for index in point_indices[active].T) for point_indices in tangent_indices),
            jnp.array(tangent_pair_valid[:, active], dtype=jnp.bool_),
        )

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
        point_data = []
        for azimuth in azimuths:
            tangent_direction = np.cos(azimuth) * tangent_1 + np.sin(azimuth) * tangent_2
            directions = cos_angle[:, None] * normals + sin_angle[:, None] * tangent_direction
            points = self._first_fluid_mesh_intersection(indices, directions, solid_mask)
            point_data.append(self._build_interpolation_data(points))
        return tuple(point_data)

    def _create_geometric_wetting_data(self):
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
        characteristics_time = 0.0
        for BC in self.BCs:
            solid_mask = self._create_component_solid_mask(BC)
            geometric_fluid_mask.append(jnp.array(~solid_mask[..., None], dtype=jnp.bool_))
            component_data = []
            for bc in BC:
                if not self._is_wetting_boundary_condition(bc):
                    continue
                if bc.theta is None:
                    continue

                indices = np.array(self._get_solid_indices(bc), dtype=np.int64).T
                theta = np.asarray(bc.theta, dtype=np.float64).reshape(-1)
                if theta.size == 1:
                    theta = np.full((indices.shape[0],), theta.item(), dtype=np.float64)
                if theta.shape[0] != indices.shape[0]:
                    raise ValueError("Geometric wetting theta must be scalar or match the number of boundary nodes.")

                normals = self._compute_geometric_normals(bc, solid_mask)
                normal_norm = np.linalg.norm(normals, axis=1)
                interface = self._solid_fluid_interface_mask(indices, solid_mask)
                valid = interface & (normal_norm > 1e-12)
                if not np.any(valid):
                    continue
                indices = indices[valid]
                theta = theta[valid]
                normals = normals[valid]

                if self.dim == 3:
                    characteristics_start = time.perf_counter()
                    points = self._build_geometric_3d_characteristic_data(indices, normals, theta, solid_mask)
                    characteristics_time += time.perf_counter() - characteristics_start
                    component_data.append({
                        "indices": tuple(jnp.array(index, dtype=jnp.int32) for index in indices.T),
                        "theta": jnp.array(theta.reshape(-1, 1), dtype=self.precision_policy.compute_dtype),
                        "points": points,
                    })
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

                points_1 = self._first_fluid_mesh_intersection(indices, direction_1, solid_mask)
                points_2 = self._first_fluid_mesh_intersection(indices, direction_2, solid_mask)
                characteristics_time += time.perf_counter() - characteristics_start
                component_data.append({
                    "indices": tuple(jnp.array(index, dtype=jnp.int32) for index in indices.T),
                    "theta": jnp.array(theta.reshape(-1, 1), dtype=self.precision_policy.compute_dtype),
                    "point_1": self._build_interpolation_data(points_1),
                    "point_2": self._build_interpolation_data(points_2),
                })
            geometric_wetting_data.append(component_data)

        logger.info(f"Time taken to determine geometric wetting characteristics: {characteristics_time:.6f} seconds")

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
        Sum a scalar (single-channel) field over its lattice neighbors, weighted per direction, using local
        jnp.roll for every axis and exactly one x-halo exchange per distinct x-shift shared by every lattice
        direction with that shift (-1, 0 or +1 for every lattice this library supports).

        Reused for two purposes, bound to their own static weights via functools.partial when scalar_neighbor_sum
        / scalar_force_stencil are built (see __init__): the G_ff-weighted neighbor sum used by
        compute_average_density's wetting denominator (scalar weights), and the G_ff*c weighted directional sum
        used by compute_fluid_fluid_force's Shan-Chen/Zhang-Chen force (vector weights) - both without ever
        streaming a q-channel array.

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
        x_shifted = {0: field}
        for x_shift in (1, -1):
            shifted = jnp.roll(field, x_shift, axis=0)
            if x_shift == 1:
                x_shifted[1] = shifted.at[:1].set(self.send_right(field[-1:], "x"))
            else:
                x_shifted[-1] = shifted.at[-1:].set(self.send_left(field[:1], "x"))

        directions = np.array(self.lattice.c).T
        is_vector = weights.ndim == 2
        out_channels = weights.shape[0] if is_vector else 1
        total = jnp.zeros((*field.shape[:-1], out_channels), dtype=self.precision_policy.compute_dtype)
        for q_index, direction in enumerate(directions):
            w = weights[:, q_index] if is_vector else weights[q_index]
            if np.all(w == 0.0):
                continue
            base = x_shifted[int(direction[0])]
            remaining_axes = tuple(int(component) for component in direction[1 : self.dim])
            if any(remaining_axes):
                base = jnp.roll(base, remaining_axes, axis=tuple(range(1, self.dim)))
            total = total + base * jnp.asarray(w, dtype=self.precision_policy.compute_dtype)
        return total

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
            logger.info(f"Component: {i + 1}")
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
            logger.info("Time to create the local masks and normal arrays: %.6f seconds", time.time() - start)

    def _make_local_bounceback_indices(self):
        """
        Distribute one padded local BounceBack index array per component (self.BCs is a list of per-component
        boundary condition lists for Multiphase, unlike the flat list in LBMBase).

        Returns
        -------
        (list): One distributed local index array (or None) per component, see LBMBase._make_local_bounceback_indices.
        """
        sharding = NamedSharding(self.mesh, PartitionSpec("x", None, None))
        local_bounceback_indices = []
        for BCs in self.BCs:
            local_indices = self._collect_bounceback_indices(BCs)
            if local_indices is None:
                local_bounceback_indices.append(None)
                continue
            indices = self.distributed_array_init(local_indices.shape, jnp.int32, init_val=local_indices, sharding=sharding)
            indices.block_until_ready()
            local_bounceback_indices.append(indices)
        return local_bounceback_indices

    def _make_local_wall_bc_data(self):
        """
        Distribute, per component and per concrete wall boundary condition type in WALL_BC_TYPES, the padded
        local fluid-node indices and auxiliary data, plus one merged padded local solid-node index array per
        component (self.BCs is a list of per-component boundary condition lists for Multiphase, unlike the flat
        list in LBMBase).

        Returns
        -------
        (list, list): One wall_bc_data dict and one solid-pin index array (or None) per component, see
        LBMBase._make_local_wall_bc_data.
        """
        wall_bc_data_by_component = []
        solid_pin_indices_by_component = []
        for BCs in self.BCs:
            wall_bc_data = {}
            for bc_type, _, has_weights in WALL_BC_TYPES:
                local_indices, (local_imissing, local_iknown, local_vel, local_weights) = self._collect_wall_bc_data(BCs, bc_type, has_weights)
                wall_bc_data[bc_type] = (
                    self._distribute_local(local_indices, jnp.int32),
                    self._distribute_local(local_imissing, jnp.uint8),
                    self._distribute_local(local_iknown, jnp.uint8),
                    self._distribute_local(local_vel),
                    self._distribute_local(local_weights),
                )
            wall_bc_data_by_component.append(wall_bc_data)
            solid_pin_indices_by_component.append(self._distribute_local(self._collect_solid_pin_indices(BCs), jnp.int32))
        return wall_bc_data_by_component, solid_pin_indices_by_component

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
        """
        if self.scalar_neighbor_sum is None or self.average_density_denominator is None:
            return rho_tree

        return [
            self.scalar_neighbor_sum(rho * (1 - solid_mask)) / denominator if denominator is not None else rho
            for rho, solid_mask, denominator in zip(
                rho_tree,
                self.solid_mask_streamed,
                self.average_density_denominator,
                strict=True,
            )
        ]

    @partial(jit, static_argnums=(0,))
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

            def set_contact_angle(rho, rho_ave, BC):
                rho_min = jnp.min(rho)
                rho_max = jnp.max(rho)
                for bc in BC:
                    if isinstance(
                        bc, (BounceBackHalfway, BounceBack, BounceBackMoving, InterpolatedBounceBackBouzidi, InterpolatedBounceBackDifferentiable)
                    ):
                        if bc.theta is not None:
                            indices = self._get_solid_indices(bc)
                            rho = rho.at[indices].set(
                                (bc.theta <= jnp.pi / 2) * (bc.phi * rho_ave[indices]) + (bc.theta > jnp.pi / 2) * (rho_ave[indices] - bc.delta_rho)
                            )
                        rho = jnp.clip(rho, min=rho_min, max=rho_max)
                return rho

            return [
                set_contact_angle(rho, rho_ave, BC) if has_wetting_bc else rho
                for rho, rho_ave, BC, has_wetting_bc in zip(rho_tree, rho_ave_tree, self.BCs, self._has_wetting_bc, strict=True)
            ]
        elif self.wetting_formulation == "geometric":

            def interpolate_density(rho, interpolation_data):
                """
                Interpolate density at precomputed geometric wetting sample points.

                Parameters
                ----------
                rho (jax.numpy.ndarray): Density field for one component.

                interpolation_data (tuple): Index arrays and weights generated by _build_interpolation_data.

                Returns
                -------
                (jax.numpy.ndarray): Interpolated density values at the sample points.
                """
                if self.dim == 2:
                    x0, y0, x1, y1, w00, w10, w01, w11 = interpolation_data
                    return w00[:, None] * rho[x0, y0] + w10[:, None] * rho[x1, y0] + w01[:, None] * rho[x0, y1] + w11[:, None] * rho[x1, y1]

                x0, y0, z0, x1, y1, z1, w000, w100, w010, w110, w001, w101, w011, w111 = interpolation_data
                return (
                    w000[:, None] * rho[x0, y0, z0]
                    + w100[:, None] * rho[x1, y0, z0]
                    + w010[:, None] * rho[x0, y1, z0]
                    + w110[:, None] * rho[x1, y1, z0]
                    + w001[:, None] * rho[x0, y0, z1]
                    + w101[:, None] * rho[x1, y0, z1]
                    + w011[:, None] * rho[x0, y1, z1]
                    + w111[:, None] * rho[x1, y1, z1]
                )

            def set_geometric_contact_angle(rho, component_data, fluid_mask):
                """
                Set wall density values for one component using precomputed geometric wetting data.

                Parameters
                ----------
                rho (jax.numpy.ndarray): Density field for one component.

                component_data (list): Boundary-wise geometric wetting data for one component.

                fluid_mask (jax.numpy.ndarray): Boolean mask with True on fluid nodes.

                Returns
                -------
                rho (jax.numpy.ndarray): Density field with wall values updated at wetted boundary nodes.
                """
                # Bound wall densities by the fluid density range so the wall can never introduce
                # a density outside what exists in the fluid. Using the global field range instead
                # would include the unphysical densities stored at solid nodes by bounce-back and
                # let the wall values ratchet the fluid range upward.
                rho_min = jnp.min(jnp.where(fluid_mask, rho, jnp.inf))
                rho_max = jnp.max(jnp.where(fluid_mask, rho, -jnp.inf))
                for data in component_data:
                    if self.dim == 2:
                        rho_1 = interpolate_density(rho, data["point_1"])
                        rho_2 = interpolate_density(rho, data["point_2"])
                        rho_wall = jnp.where(data["theta"] <= jnp.pi / 2, jnp.maximum(rho_1, rho_2), jnp.minimum(rho_1, rho_2))
                    else:
                        rho_samples = [interpolate_density(rho, point_data) for point_data in data["points"]]
                        rho_sample_min = rho_samples[0]
                        rho_sample_max = rho_samples[0]
                        for rho_sample in rho_samples[1:]:
                            rho_sample_min = jnp.minimum(rho_sample_min, rho_sample)
                            rho_sample_max = jnp.maximum(rho_sample_max, rho_sample)
                        rho_wall = jnp.where(data["theta"] <= jnp.pi / 2, rho_sample_max, rho_sample_min)
                    rho = rho.at[data["indices"]].set(jnp.clip(rho_wall, rho_min, rho_max))
                return rho

            return tree_map(
                lambda rho, component_data, fluid_mask: set_geometric_contact_angle(rho, component_data, fluid_mask),
                rho_tree,
                self.geometric_wetting_data,
                self.geometric_fluid_mask,
            )

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
        rho_tree = tree_map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype).T
        u_tree = tree_map(lambda f, rho: jnp.dot(f, c) / rho, f_tree, rho_tree)  # Component velocity
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
        # rho_tree = tree_map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype).T
        u_tree = tree_map(lambda f, rho: jnp.dot(f, c) / rho, f_tree, rho_tree)
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
        Compute the potential (psi and U) which is required for computing interaction forces.
        The psi values are obtained using the corresponding EOS. This function can be overloaded to handle cases where one or more component does not
        have EOS.

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
        # Shan-Chen potential using modified pressure
        psi_tree = tree_map(
            lambda k, p, rho, G: jnp.sqrt(2 * (k * p - self.lattice.cs2 * rho) / G), self.k, p_tree, rho_tree, self.g_kkprime.diagonal().tolist()
        )
        # Zhang-Chen potential
        U_tree = tree_map(lambda k, p, rho: k * p - self.lattice.cs2 * rho, self.k, p_tree, rho_tree)
        return psi_tree, U_tree

    # Compute the force using the effective mass (psi) and the interaction potential (phi)
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
        rho_tree = self.apply_contact_angle(rho_tree)
        psi_tree, U_tree = self.compute_potential(rho_tree, T=T)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree, U_tree)
        # fluid_solid_force = self.compute_fluid_solid_force(rho_tree)
        if self.body_force is not None:
            force_tree = tree_map(lambda ff, rho: ff + self.body_force * rho, fluid_fluid_force, rho_tree)
        else:
            force_tree = fluid_fluid_force
        if self.wetting_formulation == "geometric" and any(self._has_wetting_bc):
            force_tree = tree_map(lambda force, fluid_mask: force * fluid_mask, force_tree, self.geometric_fluid_mask)
        return force_tree

    @partial(jit, static_argnums=(0,))
    def compute_fluid_fluid_force(self, psi_tree, U_tree):
        """
        Compute the fluid-fluid interaction force using the effective mass (psi).
        The force calculation is based on the Shan-Chen method using the weighted sum
        of Shan-Chen and Zhang-Chen potential where modified pressure is used:

        modified pressure = k * pressure;
        k is defined by user. Set k=1 for default pseudopotential formulation.

        Parameters
        ----------
        psi_tree (pytree of jax.numpy.ndarray): Pseudo-potential field (Yuan-Schaefer, with modification)

        U_tree (pytree of jax.numpy.ndarray): Pseudo-potential field (Zhang-Chen, with modification)

        Returns
        -------
        (pytree of jax.numpy.ndarray): Fluid-fluid interaction forces.

        Notes
        -----
        jnp.dot(G_ff * field_s, c), with field_s the q-channel streamed field, is a G_ff*c weighted directional
        sum of neighbor values - exactly what scalar_force_stencil computes directly from the unstreamed scalar
        field (see _neighbor_stencil_m). Computed once per component here, outside the per-output-component
        vmap below, matching the original psi_s_tree/U_s_tree precompute (scalar_force_stencil is itself a
        shard_map'd call and must not be invoked from inside vmap).
        """
        psi_stencil_tree = tree_map(lambda psi: self.scalar_force_stencil(psi), psi_tree)
        U_stencil_tree = tree_map(lambda U: self.scalar_force_stencil(U), U_tree)

        def ffk_1(Ai, g_kkprime):
            """
            Shan-Chen interaction force
            g_kkprime is a row of self.gkkprime, as it represents the interaction between kth component with all components
            """
            return reduce(operator.add, tree_map(lambda A, G, stencil: (1 - A) * G * stencil, list(Ai), list(g_kkprime), psi_stencil_tree))

        def ffk_2(Ai):
            """
            Zhang-Chen interaction force.
            """
            return reduce(operator.add, tree_map(lambda A, stencil: A * stencil, list(Ai), U_stencil_tree))

        return tree_map(
            lambda psi, nt_1, nt_2: psi * nt_1 + nt_2,
            psi_tree,
            list(vmap(ffk_1, in_axes=(0, 0))(self.A, self.g_kkprime)),
            list(vmap(ffk_2, in_axes=(0))(self.A)),
        )

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

    @partial(jit, static_argnums=(0, 4), donate_argnums=(1,))
    def apply_bc(self, fout_tree, fin_tree, timestep, implementation_step):
        """
        This function extends apply_bc to pytrees.

        Full-way BounceBack and every wall boundary condition in WALL_BC_TYPES (BounceBackHalfway,
        InterpolatedBounceBackBouzidi, InterpolatedBounceBackDifferentiable) are handled separately per
        component, in batched calls using local (per-shard, int32) indices, instead of the generic per-BC
        global-index loop.

        Parameters
        ----------
        fout_tree (pytree of jax.numpy.ndarray): The post-collision or post-streaming distribution functions where bc
            needs to be applied.

        fin_tree (pytree of jax.numpy.ndarray): The pre-collision or post-collision distribution functions.

        timestep (int): Current simulation timestep, used by dynamic boundary conditions.

        implementation_step (str): The implementation step at which the boundary conditions should be applied.

        Returns
        -------
        (pytree of jax.numpy.ndarray): The output distribution functions after applying the boundary conditions.
        """

        def _apply_bc_(fin, fout, bc):
            if isinstance(bc, (BounceBack, BounceBackHalfway)):
                return fout
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementation_step == implementation_step:
                if bc.is_dynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.indices].set(bc.apply(fout, fin))
            return fout

        def __apply_bc__(fout, fin, BCs):
            for bc in BCs:
                fout = _apply_bc_(fin, fout, bc)
            return fout

        fout_tree = tree_map(lambda fout, fin, BCs: __apply_bc__(fout, fin, BCs), fout_tree, fin_tree, self.BCs)

        if implementation_step == "PostCollision":
            fout_tree = [
                self.local_bounceback(fout, fin, local_indices) if local_indices is not None else fout
                for fout, fin, local_indices in zip(fout_tree, fin_tree, self.local_bounceback_indices, strict=True)
            ]

        if implementation_step == "PostStreaming":
            new_fout_tree = []
            for fout, fin, wall_bc_data, solid_pin_indices in zip(fout_tree, fin_tree, self.wall_bc_data, self.solid_pin_indices, strict=True):
                if solid_pin_indices is not None:
                    fout = self.local_solid_pin(fout, solid_pin_indices)
                for bc_type, _, _ in WALL_BC_TYPES:
                    local_indices, local_imissing, local_iknown, local_vel, local_weights = wall_bc_data[bc_type]
                    if local_indices is not None:
                        fout = self.local_wall_bc_kernels[bc_type](fout, fin, local_indices, local_imissing, local_iknown, local_vel, local_weights)
                new_fout_tree.append(fout)
            fout_tree = new_fout_tree

        return fout_tree

    @partial(jit, static_argnums=(0, 3), donate_argnums=(1,))
    def step(self, f_poststreaming_tree, timestep, return_fpost=False, T=None):
        """
        This function performs a single step of the LBM simulation.

        It first performs the collision step, which is the relaxation of the distribution functions
        towards their equilibrium values. It then applies the respective boundary conditions to the
        post-collision distribution functions.

        The function then performs the streaming step, which is the propagation of the distribution
        functions in the lattice. It then applies the respective boundary conditions to the post-streaming
        distribution functions.

        Parameters
        ----------
        f_poststreaming_tree (pytree of jax.numpy.ndarray): Post-streaming distribution function.

        timestep (int): Current timestep

        return_fpost (bool): Return post-collision distribution function (pytree).

        T (jax.numpy.ndarray, optional): Temperature field, required when the EOS is thermal
        (see the hybrid thermal solver in thermal.py).

        Returns
        -------
        f_poststreaming_tree (pytree of jax.numpy.ndarray): Post-streamed distribution function.

        f_collision_tree (pytree of jax.numpy.ndarray {Optional}): Post-collision distribution function.
        """
        f_postcollision_tree = self.collision(f_poststreaming_tree, T=T)
        f_postcollision_tree = self.apply_bc(f_postcollision_tree, f_poststreaming_tree, timestep, "PostCollision")
        f_poststreaming_tree = tree_map(lambda f_postcollision: self.streaming(f_postcollision), f_postcollision_tree)
        f_poststreaming_tree = self.apply_bc(f_poststreaming_tree, f_postcollision_tree, timestep, "PostStreaming")

        if return_fpost:
            return f_poststreaming_tree, f_postcollision_tree
        else:
            return f_poststreaming_tree, None

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
                state = {}
                c_name = lambda i: f"component_{i}"
                for i in range(self.n_components):
                    state[c_name(i)] = f_tree[i]
                # shardings = jax.map(lambda x: x.sharding, f_tree)
                # restore_args = orb.checkpoint_utils.construct_restore_args(
                #     f_tree, shardings
                # )
                try:
                    restored_state = self.mngr.restore(latest_step, args=orb.args.StandardRestore(state))
                    f_tree = [restored_state[c_name(i)] for i in range(self.n_components)]
                    logger.info(f"Restored checkpoint at step {latest_step}.")
                except ValueError:
                    raise ValueError(f"Failed to restore checkpoint at step {latest_step}.")

                start_step = latest_step + 1
                if not (t_max > start_step):
                    raise ValueError(f"Simulation already exceeded maximum allowable steps (t_max  = {t_max}). Consider increasing t_max.")

        if self.compute_MLUPS:
            start = time.time()

        # Loop over all time steps
        for timestep in range(start_step, t_max + 1):
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
            #     # Cast to compute precision first, matching the convention collision() already uses.
            #     rho_prev_tree = tree_map(lambda rho: self.precision_policy.cast_to_compute(rho), rho_prev_tree)
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

            # Perform one time-step (collision, streaming, and boundary conditions)
            f_tree, fstar_tree = self.step(f_tree, timestep)

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
                # # See the cast_to_compute comment on rho_prev_tree above: same fix, same reason.
                rho_tree = tree_map(lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree)
                u_tree = self.macroscopic_velocity(f_tree, rho_tree)
                psi_tree, _ = self.compute_potential(rho_tree)
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
                jax.block_until_ready(f_tree)
                start = time.time()

        if self.compute_MLUPS:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f_tree)
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
                        f"{self.n_components * self.nx * self.ny * t_max / (end - start) / 1e6}",
                        "red",
                    )
                )

            elif self.dim == 3:
                logger.info(colored("Domain: ", "blue") + colored(f"{self.nx} x {self.ny} x {self.nz}", "green"))
                logger.info(colored("Number of voxels: ", "blue") + colored(f"{self.nx * self.ny * self.nz}", "green"))
                logger.info(
                    colored("MLUPS: ", "blue")
                    + colored(
                        f"{self.n_components * self.nx * self.ny * self.nz * t_max / (end - start) / 1e6}",
                        "red",
                    )
                )
        if self.mngr is not None:
            self.mngr.wait_until_finished()
        return f_tree

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
        self.output_data(**kwargs)


class MultiphaseBGK(Multiphase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,))
    def collision(self, fin_tree, T=None):
        """
        BGK collision step for lattice, extended to pytrees.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.
        The optional temperature field T is forwarded to the force computation for thermal EOS.
        """
        fin_tree = tree_map(lambda fin: self.precision_policy.cast_to_compute(fin), fin_tree)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        self.collision_terms = []
        for collision_matrix in self.collision_matrix:
            matrix = np.asarray(collision_matrix)
            columns = []
            for output_direction in range(self.lattice.q):
                columns.append(
                    tuple(
                        (input_direction, np.float32(matrix[input_direction, output_direction]))
                        for input_direction in range(self.lattice.q)
                        if not np.isclose(matrix[input_direction, output_direction], 0.0, atol=1e-7)
                    )
                )
            self.collision_terms.append(tuple(columns))
        # Surface tension adjustment (adjust_surface_tension) is identically zero whenever every component's
        # kappa is zero - a static (non-traced) fact known here, so collision() can skip computing and adding
        # it entirely in that case, rather than multiplying by a provably-zero array every timestep.
        self._has_surface_tension = any(float(kappa) != 0.0 for kappa in self.kappa)

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
        psi_s_tree = tree_map(lambda psi: self.streaming(jnp.repeat(psi, axis=-1, repeats=self.q)), psi_tree)
        c = jnp.transpose(self.c)
        if isinstance(self.lattice, LatticeD2Q9):
            tm1 = lambda i, j, psi, psi_s: psi[..., 0] * jnp.dot(self.G_ff * (psi_s - psi), c[:, i] * c[:, j])
            tm2 = lambda i, j, psi, psi_s: jnp.dot(self.G_ff * (psi_s**2 - psi**2), c[:, i] * c[:, j])

            def compute_C(kappa, A, s_v, s_e, s_eta, psi, psi_s):
                C = jnp.zeros_like(
                    psi_s,
                    dtype=self.precision_policy.compute_dtype,
                )
                qxx = -kappa * ((1 - A) * tm1(0, 0, psi, psi_s) + 0.5 * A * tm2(0, 0, psi, psi_s))
                qxy = -kappa * ((1 - A) * tm1(0, 1, psi, psi_s) + 0.5 * A * tm2(0, 1, psi, psi_s))
                qyy = -kappa * ((1 - A) * tm1(1, 1, psi, psi_s) + 0.5 * A * tm2(1, 1, psi, psi_s))
                C = C.at[..., 1].set(1.5 * s_e * (qxx + qyy))
                C = C.at[..., 2].set(-1.5 * s_eta * (qxx + qyy))
                C = C.at[..., 7].set(-s_v * (qxx - qyy))
                C = C.at[..., 8].set(-s_v * qxy)
                return C

            C_tree = tree_map(
                lambda kappa, A, s_v, s_e, s_eta, psi, psi_s: compute_C(kappa, A, s_v, s_e, s_eta, psi, psi_s),
                self.kappa,
                list(self.A.diagonal()),
                self.s_v,
                self.s_e,
                self.s_eta,
                psi_tree,
                psi_s_tree,
            )
            return C_tree
        elif isinstance(self.lattice, LatticeD3Q19):
            tm1 = lambda i, j, psi, psi_s: psi[..., 0] * jnp.dot(self.G_ff * (psi_s - psi), c[:, i] * c[:, j])
            tm2 = lambda i, j, psi, psi_s: jnp.dot(self.G_ff * (psi_s**2 - psi**2), c[:, i] * c[:, j])

            def compute_C(kappa, A, s_v, s_e, s_eta, psi, psi_s):
                C = jnp.zeros_like(
                    psi_s,
                    dtype=self.precision_policy.compute_dtype,
                )
                qxx = -kappa * ((1 - A) * tm1(0, 0, psi, psi_s) + 0.5 * A * tm2(0, 0, psi, psi_s))
                qxy = -kappa * ((1 - A) * tm1(0, 1, psi, psi_s) + 0.5 * A * tm2(0, 1, psi, psi_s))
                qxz = -kappa * ((1 - A) * tm1(0, 2, psi, psi_s) + 0.5 * A * tm2(0, 2, psi, psi_s))
                qyy = -kappa * ((1 - A) * tm1(1, 1, psi, psi_s) + 0.5 * A * tm2(1, 1, psi, psi_s))
                qyz = -kappa * ((1 - A) * tm1(1, 2, psi, psi_s) + 0.5 * A * tm2(1, 2, psi, psi_s))
                qzz = -kappa * ((1 - A) * tm1(2, 2, psi, psi_s) + 0.5 * A * tm2(2, 2, psi, psi_s))
                C = C.at[..., 1].set((2 / 5) * s_e * (qxx + qyy + qzz))
                C = C.at[..., 9].set(-s_v * (2 * qxx - qyy - qzz))
                C = C.at[..., 11].set(-s_v * (qyy - qzz))
                C = C.at[..., 13].set(-s_v * qxy)
                C = C.at[..., 14].set(-s_v * qyz)
                C = C.at[..., 15].set(-s_v * qxz)
                return C

            C_tree = tree_map(
                lambda kappa, A, s_v, s_e, s_eta, psi, psi_s: compute_C(kappa, A, s_v, s_e, s_eta, psi, psi_s),
                self.kappa,
                list(self.A.diagonal()),
                self.s_v,
                self.s_e,
                self.s_eta,
                psi_tree,
                psi_s_tree,
            )
            return C_tree
        else:
            raise NotImplementedError("MRT model with D3Q27 model has not been implemented")

    @partial(jit, static_argnums=(0,), inline=True)
    def _compute_force_delta_feq(self, rho_tree, u_tree, T=None):
        """
        Real-space compact EDM difference delta_feq = feq(rho, u + F/rho) - feq(rho, u), computed directly from
        cu, dcu and delta_usqr instead of building a full feq_force array via equilibrium(). Shared by
        apply_force (which transforms it into moment space for the unfused, moment-space collision path some
        callers may still use) and collision (which adds it directly in real space, needing no M transform at
        all - see collision()'s Notes).

        Parameters
        ----------
        rho_tree (pytree of jax.numpy.ndarray): Density field for all components.

        u_tree (pytree of jax.numpy.ndarray): Velocity field for all components.

        T (jax.numpy.ndarray, optional): Temperature field, required when the EOS is thermal.

        Returns
        -------
        (pytree of jax.numpy.ndarray): Real-space delta_feq for all components.
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
        return tree_map(
            lambda rho, cu, dcu, delta_usqr: rho * self.w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr),
            rho_tree,
            cu_tree,
            dcu_tree,
            delta_usqr_tree,
        )

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, m_tree, meq_tree, rho_tree, u_tree, T=None):
        """
        Modified version of the apply_force defined in LBMBase to account for modified force.

        Adds the force contribution using the exact-difference method (Kupershtokh): the compact real-space
        delta_feq (see _compute_force_delta_feq) transformed into moment space with M. Since M is linear,
        dot(delta_feq, M) == dot(feq_force, M) - dot(feq(rho, u), M), so this is an exact substitute for
        separately building feq_force and meq_force. Provided for callers using the unfused, moment-space
        collision path; collision() itself adds delta_feq directly in real space instead.

        Parameters
        ----------
        m_tree (pytree of jax.numpy.ndarray): Post-collision distribution function.

        meq_tree (pytree of jax.numpy.ndarray): Equilibrium distribution function. Unused - kept for interface
            compatibility with existing callers, since the compact difference formula only needs rho, u and F.

        rho_tree (pytree of jax.numpy.ndarray): Density field for all components.

        u_tree (pytree of jax.numpy.ndarray): Velocity field for all components.

        T (jax.numpy.ndarray, optional): Temperature field, required when the EOS is thermal.

        Returns
        -------
        f_postcollision_tree (pytree of jax.numpy.ndarray): Post-collision distribution functions with the force applied.
        """
        delta_feq_tree = self._compute_force_delta_feq(rho_tree, u_tree, T=T)
        delta_meq_tree = tree_map(lambda delta_feq, M: jnp.dot(delta_feq, M), delta_feq_tree, self.M)
        return tree_map(lambda m, delta_meq: m + delta_meq, m_tree, delta_meq_tree)

    @partial(jit, static_argnums=(0,))
    def collision(self, fin_tree, T=None):
        """
        MRT collision step for lattice, using a symbolic (sparse-coefficient) fused collision matrix instead of
        three separate moment-space matrix multiplies. The optional temperature field T is forwarded to the
        pressure and force computations for thermal EOS.

        Notes
        -----
        The unfused collision is m = f @ M; mout = m - (m - meq) @ S + delta_meq + C; fout = mout @ M_inv, with
        delta_meq = dot(delta_feq, M) (see apply_force) and C the surface-tension adjustment (see
        adjust_surface_tension). Substituting and using M @ M_inv = I:

            fout = f - (f - feq) @ (M @ S @ M_inv) + delta_feq + C @ M_inv

        collision_matrix = M @ S @ M_inv is built once in __init__; collision_terms holds its nonzero entries as
        static Python tuples per output direction, so the (f - feq) @ collision_matrix contraction is unrolled
        into an explicit sum over only the nonzero terms while tracing, instead of a dense matrix multiply.
        C @ M_inv is skipped entirely when every component's kappa is zero (see __init__), since C is then
        identically zero.
        """
        fin_tree = tree_map(lambda f: self.precision_policy.cast_to_compute(f), fin_tree)
        rho_tree, u_tree = self.update_macroscopic(fin_tree)
        feq_tree = self.equilibrium(rho_tree, u_tree, cast_output=False)
        delta_feq_tree = self._compute_force_delta_feq(rho_tree, u_tree, T=T)

        fout_tree = []
        for f, feq, delta_feq, columns in zip(fin_tree, feq_tree, delta_feq_tree, self.collision_terms, strict=True):
            difference = f - feq
            outputs = []
            for output_direction, terms in enumerate(columns):
                relaxed = sum(difference[..., input_direction] * coefficient for input_direction, coefficient in terms)
                outputs.append(f[..., output_direction] - relaxed + delta_feq[..., output_direction])
            fout_tree.append(jnp.stack(outputs, axis=-1))

        if self._has_surface_tension:
            psi_tree, _ = self.compute_potential(rho_tree, T=T)
            C_tree = self.adjust_surface_tension(psi_tree)
            fout_tree = tree_map(lambda fout, C, M_inv: fout + jnp.dot(C, M_inv), fout_tree, C_tree, self.M_inv)

        if self.wetting_formulation == "geometric" and self.dim == 3:
            # Preserve the density moment after the 3D geometric wetting update by applying any roundoff-level mismatch to the rest population.
            rho_out_tree = tree_map(lambda fout: jnp.sum(fout, axis=-1, keepdims=True), fout_tree)
            fout_tree = tree_map(lambda fout, rho, rho_out: fout.at[..., 0].add((rho - rho_out)[..., 0]), fout_tree, rho_tree, rho_out_tree)
        return tree_map(lambda fout: self.precision_policy.cast_to_output(fout), fout_tree)


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
                T = jnp.zeros_like(m)
                T = T.at[..., 0].set(m[..., 0])
                T = T.at[..., 1].set(-ux * m[..., 0] + m[..., 1])
                T = T.at[..., 2].set(-uy * m[..., 0] + m[..., 2])
                T = T.at[..., 3].set(usq * m[..., 0] - 2 * ux * m[..., 1] - 2 * uy * m[..., 2] + m[..., 3])
                T = T.at[..., 4].set(udiff * m[..., 0] - 2 * ux * m[..., 1] + 2 * uy * m[..., 2] + m[..., 4])
                T = T.at[..., 5].set(ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 5])
                T = T.at[..., 6].set(
                    -(ux**2) * uy * m[..., 0]
                    + 2 * ux * uy * m[..., 1]
                    + ux**2 * m[..., 2]
                    - 0.5 * uy * m[..., 3]
                    - 0.5 * uy * m[..., 4]
                    - 2 * ux * m[..., 5]
                    + m[..., 6]
                )
                T = T.at[..., 7].set(
                    -(uy**2) * ux * m[..., 0]
                    + uy**2 * m[..., 1]
                    + 2 * ux * uy * m[..., 2]
                    - 0.5 * ux * m[..., 3]
                    + 0.5 * ux * m[..., 4]
                    - 2 * uy * m[..., 5]
                    + m[..., 7]
                )
                T = T.at[..., 8].set(
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
                return T

            return tree_map(lambda m, u: shift(m, u), m_tree, u_tree)

        elif isinstance(self.lattice, LatticeD3Q19):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                T = jnp.zeros_like(m)
                T = T.at[..., 0].set(m[..., 0])
                T = T.at[..., 1].set(-ux * m[..., 0] + m[..., 1])
                T = T.at[..., 2].set(-uy * m[..., 0] + m[..., 2])
                T = T.at[..., 3].set(-uz * m[..., 0] + m[..., 3])
                T = T.at[..., 4].set(ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 4])
                T = T.at[..., 5].set(ux * uz * m[..., 0] - uz * m[..., 1] - ux * m[..., 3] + m[..., 5])
                T = T.at[..., 6].set(uy * uz * m[..., 0] - uz * m[..., 2] - uy * m[..., 3] + m[..., 6])
                T = T.at[..., 7].set((ux**2) * m[..., 0] - 2 * ux * m[..., 1] + m[..., 7])
                T = T.at[..., 8].set((uy**2) * m[..., 0] - 2 * uy * m[..., 2] + m[..., 8])
                T = T.at[..., 9].set((uz**2) * m[..., 0] - 2 * uz * m[..., 3] + m[..., 9])
                T = T.at[..., 10].set(
                    -ux * (uy**2) * m[..., 0] + (uy**2) * m[..., 1] + 2 * ux * uy * m[..., 2] - 2 * uy * m[..., 4] - ux * m[..., 8] + m[..., 10]
                )
                T = T.at[..., 11].set(
                    -ux * (uz**2) * m[..., 0] + (uz**2) * m[..., 1] + 2 * ux * uz * m[..., 3] - 2 * uz * m[..., 5] - ux * m[..., 9] + m[..., 11]
                )
                T = T.at[..., 12].set(
                    -(ux**2) * uy * m[..., 0] + 2 * ux * uy * m[..., 1] + (ux**2) * m[..., 2] - 2 * ux * m[..., 4] - uy * m[..., 7] + m[..., 12]
                )
                T = T.at[..., 13].set(
                    -(ux**2) * uz * m[..., 0] + 2 * ux * uz * m[..., 1] + (ux**2) * m[..., 3] - 2 * ux * m[..., 5] - uz * m[..., 7] + m[..., 13]
                )
                T = T.at[..., 14].set(
                    -uy * (uz**2) * m[..., 0] + (uz**2) * m[..., 2] + 2 * uy * uz * m[..., 3] - 2 * uz * m[..., 6] - uy * m[..., 9] + m[..., 14]
                )
                T = T.at[..., 15].set(
                    -(uy**2) * uz * m[..., 0] + 2 * uy * uz * m[..., 2] + (uy**2) * m[..., 3] - 2 * uy * m[..., 6] - uz * m[..., 8] + m[..., 15]
                )
                T = T.at[..., 16].set(
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
                T = T.at[..., 17].set(
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
                T = T.at[..., 18].set(
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
                return T

            return tree_map(lambda m, u: shift(m, u), m_tree, u_tree)

        elif isinstance(self.lattice, LatticeD3Q27):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                T = jnp.zeros_like(m)
                T = T.at[..., 0].set(m[..., 0])
                T = T.at[..., 1].set(m[..., 1] - m[..., 0] * ux)
                T = T.at[..., 2].set(m[..., 2] - m[..., 0] * uy)
                T = T.at[..., 3].set(m[..., 3] - m[..., 0] * uz)
                T = T.at[..., 4].set(m[..., 4] - m[..., 2] * ux - m[..., 1] * uy + m[..., 0] * ux * uy)
                T = T.at[..., 5].set(m[..., 5] - m[..., 3] * ux - m[..., 1] * uz + m[..., 0] * ux * uz)
                T = T.at[..., 6].set(m[..., 6] - m[..., 3] * uy - m[..., 2] * uz + m[..., 0] * uy * uz)
                T = T.at[..., 7].set(m[..., 0] * ux**2 - 2 * m[..., 1] * ux + m[..., 7])
                T = T.at[..., 8].set(m[..., 0] * uy**2 - 2 * m[..., 2] * uy + m[..., 8])
                T = T.at[..., 9].set(m[..., 0] * uz**2 - 2 * m[..., 3] * uz + m[..., 9])
                T = T.at[..., 10].set(
                    m[..., 10] - m[..., 8] * ux - 2 * m[..., 4] * uy + m[..., 1] * uy**2 - m[..., 0] * ux * uy**2 + 2 * m[..., 2] * ux * uy
                )
                T = T.at[..., 11].set(
                    m[..., 11] - m[..., 9] * ux - 2 * m[..., 5] * uz + m[..., 1] * uz**2 - m[..., 0] * ux * uz**2 + 2 * m[..., 3] * ux * uz
                )
                T = T.at[..., 12].set(
                    m[..., 12] - 2 * m[..., 4] * ux - m[..., 7] * uy + m[..., 2] * ux**2 - m[..., 0] * ux**2 * uy + 2 * m[..., 1] * ux * uy
                )
                T = T.at[..., 13].set(
                    m[..., 13] - 2 * m[..., 5] * ux - m[..., 7] * uz + m[..., 3] * ux**2 - m[..., 0] * ux**2 * uz + 2 * m[..., 1] * ux * uz
                )
                T = T.at[..., 14].set(
                    m[..., 14] - m[..., 9] * uy - 2 * m[..., 6] * uz + m[..., 2] * uz**2 - m[..., 0] * uy * uz**2 + 2 * m[..., 3] * uy * uz
                )
                T = T.at[..., 15].set(
                    m[..., 15] - 2 * m[..., 6] * uy - m[..., 8] * uz + m[..., 3] * uy**2 - m[..., 0] * uy**2 * uz + 2 * m[..., 2] * uy * uz
                )
                T = T.at[..., 16].set(
                    m[..., 16]
                    - m[..., 6] * ux
                    - m[..., 5] * uy
                    - m[..., 4] * uz
                    + m[..., 3] * ux * uy
                    + m[..., 2] * ux * uz
                    + m[..., 1] * uy * uz
                    - m[..., 0] * ux * uy * uz
                )
                T = T.at[..., 17].set(
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
                T = T.at[..., 18].set(
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
                T = T.at[..., 19].set(
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
                T = T.at[..., 20].set(
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
                T = T.at[..., 21].set(
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
                T = T.at[..., 22].set(
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
                T = T.at[..., 23].set(
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
                T = T.at[..., 24].set(
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
                T = T.at[..., 25].set(
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
                T = T.at[..., 26].set(
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
                m = jnp.zeros_like(T)
                m = m.at[..., 0].set(T[..., 0])
                m = m.at[..., 1].set(ux * T[..., 0] + T[..., 1])
                m = m.at[..., 2].set(uy * T[..., 0] + T[..., 2])
                m = m.at[..., 3].set(usq * T[..., 0] + 2 * ux * T[..., 1] + 2 * uy * T[..., 2] + T[..., 3])
                m = m.at[..., 4].set(udiff * T[..., 0] + 2 * ux * T[..., 1] - 2 * uy * T[..., 2] + T[..., 4])
                m = m.at[..., 5].set(ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 5])
                m = m.at[..., 6].set(
                    (ux**2) * uy * T[..., 0]
                    + 2 * ux * uy * T[..., 1]
                    + ux**2 * T[..., 2]
                    + 0.5 * uy * T[..., 3]
                    + 0.5 * uy * T[..., 4]
                    + 2 * ux * T[..., 5]
                    + T[..., 6]
                )
                m = m.at[..., 7].set(
                    (uy**2) * ux * T[..., 0]
                    + uy**2 * T[..., 1]
                    + 2 * ux * uy * T[..., 2]
                    + 0.5 * ux * T[..., 3]
                    - 0.5 * ux * T[..., 4]
                    + 2 * uy * T[..., 5]
                    + T[..., 7]
                )
                m = m.at[..., 8].set(
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
                return m

            return tree_map(lambda T, u: shift_inverse(T, u), T_tree, u_tree)

        elif isinstance(self.lattice, LatticeD3Q19):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                m = jnp.zeros_like(T)
                m = m.at[..., 0].set(T[..., 0])
                m = m.at[..., 1].set(ux * T[..., 0] + T[..., 1])
                m = m.at[..., 2].set(uy * T[..., 0] + T[..., 2])
                m = m.at[..., 3].set(uz * T[..., 0] + T[..., 3])
                m = m.at[..., 4].set(ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 4])
                m = m.at[..., 5].set(ux * uz * T[..., 0] + uz * T[..., 1] + ux * T[..., 3] + T[..., 5])
                m = m.at[..., 6].set(uy * uz * T[..., 0] + uz * T[..., 2] + uy * T[..., 3] + T[..., 6])
                m = m.at[..., 7].set((ux**2) * T[..., 0] + 2 * ux * T[..., 1] + T[..., 7])
                m = m.at[..., 8].set((uy**2) * T[..., 0] + 2 * uy * T[..., 2] + T[..., 8])
                m = m.at[..., 9].set((uz**2) * T[..., 0] + 2 * uz * T[..., 3] + T[..., 9])
                m = m.at[..., 10].set(
                    ux * (uy**2) * T[..., 0] + (uy**2) * T[..., 1] + 2 * ux * uy * T[..., 2] + 2 * uy * T[..., 4] + ux * T[..., 8] + T[..., 10]
                )
                m = m.at[..., 11].set(
                    ux * (uz**2) * T[..., 0] + (uz**2) * T[..., 1] + 2 * ux * uz * T[..., 3] + 2 * uz * T[..., 5] + ux * T[..., 9] + T[..., 11]
                )
                m = m.at[..., 12].set(
                    (ux**2) * uy * T[..., 0] + 2 * ux * uy * T[..., 1] + (ux**2) * T[..., 2] + 2 * ux * T[..., 4] + uy * T[..., 7] + T[..., 12]
                )
                m = m.at[..., 13].set(
                    (ux**2) * uz * T[..., 0] + 2 * ux * uz * T[..., 1] + (ux**2) * T[..., 3] + 2 * ux * T[..., 5] + uz * T[..., 7] + T[..., 13]
                )
                m = m.at[..., 14].set(
                    uy * (uz**2) * T[..., 0] + (uz**2) * T[..., 2] + 2 * uy * uz * T[..., 3] + 2 * uz * T[..., 6] + uy * T[..., 9] + T[..., 14]
                )
                m = m.at[..., 15].set(
                    (uy**2) * uz * T[..., 0] + 2 * uy * uz * T[..., 2] + (uy**2) * T[..., 3] + 2 * uy * T[..., 6] + uz * T[..., 8] + T[..., 15]
                )
                m = m.at[..., 16].set(
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
                m = m.at[..., 17].set(
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
                m = m.at[..., 18].set(
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
                return m

            return tree_map(lambda T, u: shift_inverse(T, u), T_tree, u_tree)

        elif isinstance(self.lattice, LatticeD3Q27):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                m = jnp.zeros_like(T)
                m = m.at[..., 0].set(T[..., 0])
                m = m.at[..., 1].set(T[..., 1] + T[..., 0] * ux)
                m = m.at[..., 2].set(T[..., 2] + T[..., 0] * uy)
                m = m.at[..., 3].set(T[..., 3] + T[..., 0] * uz)
                m = m.at[..., 4].set(T[..., 4] + T[..., 2] * ux + T[..., 1] * uy + T[..., 0] * ux * uy)
                m = m.at[..., 5].set(T[..., 5] + T[..., 3] * ux + T[..., 1] * uz + T[..., 0] * ux * uz)
                m = m.at[..., 6].set(T[..., 6] + T[..., 3] * uy + T[..., 2] * uz + T[..., 0] * uy * uz)
                m = m.at[..., 7].set(T[..., 0] * ux**2 + 2 * T[..., 1] * ux + T[..., 7])
                m = m.at[..., 8].set(T[..., 0] * uy**2 + 2 * T[..., 2] * uy + T[..., 8])
                m = m.at[..., 9].set(T[..., 0] * uz**2 + 2 * T[..., 3] * uz + T[..., 9])
                m = m.at[..., 10].set(
                    T[..., 10] + T[..., 8] * ux + 2 * T[..., 4] * uy + T[..., 1] * uy**2 + T[..., 0] * ux * uy**2 + 2 * T[..., 2] * ux * uy
                )
                m = m.at[..., 11].set(
                    T[..., 11] + T[..., 9] * ux + 2 * T[..., 5] * uz + T[..., 1] * uz**2 + T[..., 0] * ux * uz**2 + 2 * T[..., 3] * ux * uz
                )
                m = m.at[..., 12].set(
                    T[..., 12] + 2 * T[..., 4] * ux + T[..., 7] * uy + T[..., 2] * ux**2 + T[..., 0] * ux**2 * uy + 2 * T[..., 1] * ux * uy
                )
                m = m.at[..., 13].set(
                    T[..., 13] + 2 * T[..., 5] * ux + T[..., 7] * uz + T[..., 3] * ux**2 + T[..., 0] * ux**2 * uz + 2 * T[..., 1] * ux * uz
                )
                m = m.at[..., 14].set(
                    T[..., 14] + T[..., 9] * uy + 2 * T[..., 6] * uz + T[..., 2] * uz**2 + T[..., 0] * uy * uz**2 + 2 * T[..., 3] * uy * uz
                )
                m = m.at[..., 15].set(
                    T[..., 15] + 2 * T[..., 6] * uy + T[..., 8] * uz + T[..., 3] * uy**2 + T[..., 0] * uy**2 * uz + 2 * T[..., 2] * uy * uz
                )
                m = m.at[..., 16].set(
                    T[..., 16]
                    + T[..., 6] * ux
                    + T[..., 5] * uy
                    + T[..., 4] * uz
                    + T[..., 3] * ux * uy
                    + T[..., 2] * ux * uz
                    + T[..., 1] * uy * uz
                    + T[..., 0] * ux * uy * uz
                )
                m = m.at[..., 17].set(
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
                m = m.at[..., 18].set(
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
                m = m.at[..., 19].set(
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
                m = m.at[..., 20].set(
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
                m = m.at[..., 21].set(
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
                m = m.at[..., 22].set(
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
                m = m.at[..., 23].set(
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
                m = m.at[..., 24].set(
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
                m = m.at[..., 25].set(
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
                m = m.at[..., 26].set(
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
                T_eq = jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precision_policy.compute_dtype)
                T_eq = T_eq.at[..., 0].set(rho[..., 0])
                T_eq = T_eq.at[..., 3].set(2 * rho[..., 0] * self.lattice.cs2)
                T_eq = T_eq.at[..., 8].set(rho[..., 0] * self.lattice.cs**4)

                return T_eq

            elif isinstance(self.lattice, LatticeD3Q19):
                T_eq = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype)
                T_eq = T_eq.at[..., 0].set(rho[..., 0])
                T_eq = T_eq.at[..., 7].set(rho[..., 0] * self.lattice.cs2)
                T_eq = T_eq.at[..., 8].set(rho[..., 0] * self.lattice.cs2)
                T_eq = T_eq.at[..., 9].set(rho[..., 0] * self.lattice.cs2)
                T_eq = T_eq.at[..., 16].set(rho[..., 0] * self.lattice.cs**4)
                T_eq = T_eq.at[..., 17].set(rho[..., 0] * self.lattice.cs**4)
                T_eq = T_eq.at[..., 18].set(rho[..., 0] * self.lattice.cs**4)

                return T_eq

            elif isinstance(self.lattice, LatticeD3Q27):
                T_eq = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype)
                T_eq = T_eq.at[..., 0].set(rho[..., 0])
                T_eq = T_eq.at[..., 7].set(rho[..., 0] * self.lattice.cs2)
                T_eq = T_eq.at[..., 8].set(rho[..., 0] * self.lattice.cs2)
                T_eq = T_eq.at[..., 9].set(rho[..., 0] * self.lattice.cs2)
                T_eq = T_eq.at[..., 17].set(rho[..., 0] * self.lattice.cs**4)
                T_eq = T_eq.at[..., 18].set(rho[..., 0] * self.lattice.cs**4)
                T_eq = T_eq.at[..., 19].set(rho[..., 0] * self.lattice.cs**4)
                T_eq = T_eq.at[..., 26].set(rho[..., 0] * self.lattice.cs**6)

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
                C = jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precision_policy.compute_dtype)
                Fx = F[..., 0]
                Fy = F[..., 1]
                eta = 4 * sigma * (Fx**2 + Fy**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))  # For mechanical stability
                C = C.at[..., 1].set(Fx)
                C = C.at[..., 2].set(Fy)
                C = C.at[..., 3].set(eta)
                C = C.at[..., 6].set(Fy * self.lattice.cs2)
                C = C.at[..., 7].set(Fx * self.lattice.cs2)
                C = C.at[..., 8].set(eta * self.lattice.cs2)

                return C
            elif isinstance(self.lattice, LatticeD3Q19):
                C = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype)
                Fx = F[..., 0]
                Fy = F[..., 1]
                Fz = F[..., 2]
                eta = 4 * sigma * (Fx**2 + Fy**2 + Fz**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))
                C = C.at[..., 1].set(Fx)
                C = C.at[..., 2].set(Fy)
                C = C.at[..., 3].set(Fz)
                C = C.at[..., 7].set(eta)
                C = C.at[..., 8].set(eta)
                C = C.at[..., 9].set(eta)
                C = C.at[..., 10].set(Fx * self.lattice.cs2)
                C = C.at[..., 11].set(Fx * self.lattice.cs2)
                C = C.at[..., 12].set(Fy * self.lattice.cs2)
                C = C.at[..., 13].set(Fz * self.lattice.cs2)
                C = C.at[..., 14].set(Fy * self.lattice.cs2)
                C = C.at[..., 15].set(Fz * self.lattice.cs2)

                return C
            elif isinstance(self.lattice, LatticeD3Q27):
                C = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype)
                Fx = F[..., 0]
                Fy = F[..., 1]
                Fz = F[..., 2]
                C = C.at[..., 1].set(Fx)
                C = C.at[..., 2].set(Fy)
                C = C.at[..., 3].set(Fz)
                eta = 4 * sigma * (Fx**2 + Fy**2 + Fz**2) / ((psi[..., 0] ** 2) * (1 / s_b - 0.5))
                C = C.at[..., 7].set(eta)
                C = C.at[..., 8].set(eta)
                C = C.at[..., 9].set(eta)
                C = C.at[..., 10].set(Fx * self.lattice.cs2)
                C = C.at[..., 11].set(Fx * self.lattice.cs2)
                C = C.at[..., 12].set(Fy * self.lattice.cs2)
                C = C.at[..., 13].set(Fz * self.lattice.cs2)
                C = C.at[..., 14].set(Fy * self.lattice.cs2)
                C = C.at[..., 15].set(Fz * self.lattice.cs2)
                C = C.at[..., 23].set(Fx * self.lattice.cs**4)
                C = C.at[..., 24].set(Fy * self.lattice.cs**4)
                C = C.at[..., 25].set(Fz * self.lattice.cs**4)

                return C

        return tree_map(lambda F, sigma, psi, s_b: f(F, sigma, psi, s_b), F_tree, self.sigma, psi_tree, self.s_b)

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
        psi_tree, _ = self.compute_potential(rho_tree, T=T)
        C_tree = self.compute_force_central_moments(F_tree, psi_tree)
        Tf_tree = tree_map(lambda S, C: jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * S), self.S, C_tree)
        return tree_map(lambda Tdash, Tf: Tdash + Tf, Tdash_tree, Tf_tree)

    @partial(jit, static_argnums=(0,))
    def collision(self, fin_tree, T=None):
        """
        Cascaded LBM collision step for lattice. The optional temperature field
        T is forwarded to the pressure and force computations for thermal EOS.
        """
        fin_tree = tree_map(lambda f: self.precision_policy.cast_to_compute(f), fin_tree)
        rho_tree, _ = self.update_macroscopic(fin_tree)
        u_tree = self.macroscopic_velocity(fin_tree, rho_tree, T=T)
        T_tree = tree_map(lambda f, M: jnp.dot(f, M), fin_tree, self.M)
        Tdash_tree = self.compute_central_moment(T_tree, u_tree)
        Tdash_eq_tree = self.compute_eq_central_moments(rho_tree)
        Tout_tree = tree_map(
            lambda Tdash, Tdash_eq, S: jnp.dot(Tdash, jnp.eye(self.lattice.q) - S) + jnp.dot(Tdash_eq, S), Tdash_tree, Tdash_eq_tree, self.S
        )
        Tout_tree = self.apply_force(Tout_tree, rho_tree, u_tree, T=T)
        Tout_tree = self.compute_central_moment_inverse(Tout_tree, u_tree)
        fout_tree = tree_map(lambda T, Minv: jnp.dot(T, Minv), Tout_tree, self.M_inv)
        if self.wetting_formulation == "geometric" and self.dim == 3:
            # Preserve the density moment after the 3D geometric wetting update by applying any roundoff-level mismatch to the rest population.
            rho_out_tree = tree_map(lambda fout: jnp.sum(fout, axis=-1, keepdims=True), fout_tree)
            fout_tree = tree_map(lambda fout, rho, rho_out: fout.at[..., 0].add((rho - rho_out)[..., 0]), fout_tree, rho_tree, rho_out_tree)
        return tree_map(lambda fout: self.precision_policy.cast_to_output(fout), fout_tree)
