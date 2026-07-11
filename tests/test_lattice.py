"""Tests for lattice definitions and their mathematical invariants."""

import os

# os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from src.lattice import Lattice, LatticeD2Q9, LatticeD3Q19, LatticeD3Q27


LATTICE_CASES = (
    pytest.param(LatticeD2Q9, 2, 9, id="D2Q9"),
    pytest.param(LatticeD3Q19, 3, 19, id="D3Q19"),
    pytest.param(LatticeD3Q27, 3, 27, id="D3Q27"),
)

PRECISION_CASES = (
    pytest.param("f16/f16", jnp.float16, 5e-4, id="f16-f16"),
    pytest.param("f32/f16", jnp.float32, 1e-6, id="f32-f16"),
    pytest.param("f32/f32", jnp.float32, 1e-6, id="f32-f32"),
    pytest.param("f64/f16", jnp.float64, 1e-12, id="f64-f16"),
    pytest.param("f64/f32", jnp.float64, 1e-12, id="f64-f32"),
    pytest.param("f64/f64", jnp.float64, 1e-12, id="f64-f64"),
)


@pytest.mark.parametrize("lattice_class, dimensions, cardinality", LATTICE_CASES)
def test_lattice_array_shapes(lattice_class, dimensions, cardinality):
    """Verify the shapes of velocities, weights, and second moments."""
    lattice = lattice_class()

    assert lattice.d == dimensions
    assert lattice.q == cardinality
    assert lattice.c.shape == (dimensions, cardinality)
    assert lattice.w.shape == (cardinality,)
    assert lattice.cc.shape == (cardinality, dimensions * (dimensions + 1) // 2)


@pytest.mark.parametrize("lattice_class, dimensions, cardinality", LATTICE_CASES)
def test_lattice_directions_are_unique_and_have_one_rest_direction(lattice_class, dimensions, cardinality):
    """Verify that the velocity set has no duplicates and one zero vector."""
    directions = np.asarray(lattice_class().c).T

    assert np.unique(directions, axis=0).shape[0] == cardinality
    assert np.count_nonzero(np.all(directions == 0, axis=1)) == 1


@pytest.mark.parametrize("lattice_class, dimensions, cardinality", LATTICE_CASES)
def test_lattice_directions_sum_to_zero(lattice_class, dimensions, cardinality):
    """Verify symmetry of the velocity set along every spatial axis."""
    direction_sum = np.sum(np.asarray(lattice_class().c), axis=1)

    np.testing.assert_array_equal(direction_sum, np.zeros(dimensions, dtype=direction_sum.dtype))


@pytest.mark.parametrize("precision, expected_dtype, tolerance", PRECISION_CASES)
@pytest.mark.parametrize("lattice_class, dimensions, cardinality", LATTICE_CASES)
def test_lattice_weights_and_isotropy(
    lattice_class,
    dimensions,
    cardinality,
    precision,
    expected_dtype,
    tolerance,
):
    """Verify normalized positive weights and first- and second-order isotropy."""
    lattice = lattice_class(precision=precision)
    directions = np.asarray(lattice.c, dtype=np.float64)
    weights = np.asarray(lattice.w, dtype=np.float64)

    assert weights.shape == (cardinality,)
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(np.sum(weights), 1.0, rtol=0.0, atol=tolerance)

    first_moment = directions @ weights
    np.testing.assert_allclose(first_moment, np.zeros(dimensions), rtol=0.0, atol=tolerance)

    second_moment = (directions * weights) @ directions.T
    expected_second_moment = np.eye(dimensions) * lattice.cs2
    np.testing.assert_allclose(second_moment, expected_second_moment, rtol=0.0, atol=tolerance)

    assert lattice.w.dtype == expected_dtype
    assert lattice.cc.dtype == expected_dtype


@pytest.mark.parametrize("lattice_class, dimensions, cardinality", LATTICE_CASES)
def test_opposite_indices_are_an_involution(lattice_class, dimensions, cardinality):
    """Verify that every velocity maps to, and back from, its opposite."""
    lattice = lattice_class()
    directions = np.asarray(lattice.c).T
    opposite_indices = np.asarray(lattice.opp_indices, dtype=np.int64)

    np.testing.assert_array_equal(directions[opposite_indices], -directions)
    np.testing.assert_array_equal(opposite_indices[opposite_indices], np.arange(cardinality))


@pytest.mark.parametrize("lattice_class, dimensions, cardinality", LATTICE_CASES)
def test_direction_index_groups_match_their_definitions(lattice_class, dimensions, cardinality):
    """Verify main, left, and right direction index groups."""
    lattice = lattice_class()
    directions = np.asarray(lattice.c).T
    main_indices = np.asarray(lattice.main_indices, dtype=np.int64)
    left_indices = np.asarray(lattice.left_indices, dtype=np.int64)
    right_indices = np.asarray(lattice.right_indices, dtype=np.int64)

    expected_main = np.flatnonzero(np.sum(np.abs(directions), axis=1) == 1)
    expected_left = np.flatnonzero(directions[:, 0] == -1)
    expected_right = np.flatnonzero(directions[:, 0] == 1)

    np.testing.assert_array_equal(main_indices, expected_main)
    np.testing.assert_array_equal(left_indices, expected_left)
    np.testing.assert_array_equal(right_indices, expected_right)


@pytest.mark.parametrize("lattice_class, dimensions, cardinality", LATTICE_CASES)
def test_lattice_moments_match_velocity_products(lattice_class, dimensions, cardinality):
    """Verify every stored symmetric second-moment component."""
    lattice = lattice_class()
    directions = np.asarray(lattice.c).T
    expected_moments = np.stack(
        [directions[:, first_axis] * directions[:, second_axis] for first_axis in range(dimensions) for second_axis in range(first_axis, dimensions)],
        axis=-1,
    )

    np.testing.assert_array_equal(np.asarray(lattice.cc), expected_moments)


def test_unsupported_lattice_name_raises_value_error():
    """Verify that an unknown lattice definition is rejected."""
    with pytest.raises(ValueError, match="Supported Lattice types"):
        Lattice("D2Q7")


def test_unsupported_precision_raises_value_error():
    """Verify that an unknown precision policy is rejected."""
    with pytest.raises(ValueError, match="precision not supported"):
        LatticeD2Q9(precision="f16/f32")
