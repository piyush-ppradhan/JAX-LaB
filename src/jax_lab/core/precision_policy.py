"""
Precision and precision-policy enumerations for JAX-LaB.

Adapted from the Autodesk/XLB precision_policy module, keeping only the JAX backend. PrecisionPolicy exposes the same interface as jmp.Policy
(compute_dtype, output_dtype, cast_to_compute, cast_to_output) so it works as a drop-in replacement without a jmp dependency.
"""

from enum import Enum, auto

import jax
import jax.numpy as jnp
import numpy as np


class Precision(Enum):
    """Scalar precision levels with JAX dtype accessors."""

    FP64 = auto()
    FP32 = auto()
    FP16 = auto()

    @property
    def jax_dtype(self):
        """
        Returns
        -------
        jnp.dtype: The JAX dtype corresponding to this precision level.
        """
        if self == Precision.FP64:
            return jnp.float64
        elif self == Precision.FP32:
            return jnp.float32
        elif self == Precision.FP16:
            return jnp.float16
        else:
            raise ValueError("Invalid precision")


def _cast_floating_to(tree, dtype):
    """
    Casts all floating-point array leaves of a pytree to the given dtype.
    Non-floating leaves (e.g. integer lattice velocities) are left untouched,
    matching jmp.Policy semantics.

    Parameters
    ----------
    tree (pytree): A pytree of JAX or NumPy arrays.

    dtype (jnp.dtype): Target dtype for floating-point leaves.

    Returns
    -------
    A pytree with floating-point leaves cast to dtype.
    """

    def conditional_cast(x):
        if isinstance(x, (jax.Array, np.ndarray)) and jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(dtype)
        return x

    return jax.tree.map(conditional_cast, tree)


class PrecisionPolicy(Enum):
    """
    Mixed-precision policy pairing a compute precision (used during
    arithmetic) with a store/output precision (used in memory).

    The naming convention is <compute><store>, e.g. FP32FP16 computes in
    FP32 and stores results in FP16.
    """

    FP64FP64 = auto()
    FP64FP32 = auto()
    FP64FP16 = auto()
    FP32FP32 = auto()
    FP32FP16 = auto()
    FP16FP16 = auto()

    @classmethod
    def from_string(cls, precision):
        """
        Constructs a PrecisionPolicy from a "computation/storage" string.

        Parameters
        ----------
        precision (str): A string in the format "computation/storage" where each
        side is one of "f64", "f32" or "f16". Unrecognized values (including
        None) default to FP32FP32.

        Returns
        -------
        PrecisionPolicy: The corresponding policy.
        """
        return {
            "f64/f64": cls.FP64FP64,
            "f64/f32": cls.FP64FP32,
            "f64/f16": cls.FP64FP16,
            "f32/f32": cls.FP32FP32,
            "f32/f16": cls.FP32FP16,
            "f16/f16": cls.FP16FP16,
        }.get(precision, cls.FP32FP32)

    @property
    def compute_precision(self):
        if self in (PrecisionPolicy.FP64FP64, PrecisionPolicy.FP64FP32, PrecisionPolicy.FP64FP16):
            return Precision.FP64
        elif self in (PrecisionPolicy.FP32FP32, PrecisionPolicy.FP32FP16):
            return Precision.FP32
        elif self == PrecisionPolicy.FP16FP16:
            return Precision.FP16
        else:
            raise ValueError("Invalid precision policy")

    @property
    def store_precision(self):
        if self in (PrecisionPolicy.FP64FP64,):
            return Precision.FP64
        elif self in (PrecisionPolicy.FP64FP32, PrecisionPolicy.FP32FP32):
            return Precision.FP32
        elif self in (PrecisionPolicy.FP64FP16, PrecisionPolicy.FP32FP16, PrecisionPolicy.FP16FP16):
            return Precision.FP16
        else:
            raise ValueError("Invalid precision policy")

    @property
    def compute_dtype(self):
        return self.compute_precision.jax_dtype

    @property
    def output_dtype(self):
        return self.store_precision.jax_dtype

    def cast_to_compute(self, tree):
        """
        Casts floating-point leaves of a pytree to the compute dtype.

        Parameters
        ----------
        tree (pytree): A pytree of JAX or NumPy arrays.

        Returns
        -------
        A pytree with floating-point leaves cast to the compute dtype.
        """
        return _cast_floating_to(tree, self.compute_dtype)

    def cast_to_output(self, tree):
        """
        Casts floating-point leaves of a pytree to the store/output dtype.

        Parameters
        ----------
        tree (pytree): A pytree of JAX or NumPy arrays.

        Returns
        -------
        A pytree with floating-point leaves cast to the output dtype.
        """
        return _cast_floating_to(tree, self.output_dtype)
