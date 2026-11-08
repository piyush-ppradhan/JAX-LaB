"""
JAX-LaB: a JAX-based differentiable Lattice Boltzmann library for single and
multiphase flow simulations on CPUs, GPUs and TPUs.
"""

import os

# Disable XLA GPU command buffers (CUDA graphs) for every run, whatever the number of GPUs.
# Reason: with jax 0.11.2 and 0.12.0, a gather with mode="fill" (used by the per-shard boundary-condition kernels,
# e.g. bounce-back) inside a CUDA graph fails on two or more GPUs with
#   "CUDA error: Failed to add memset node to a CUDA graph: CUDA_ERROR_INVALID_VALUE"
# This is an XLA bug, reproducible without JAX-LaB. Disabling command buffers had no measurable cost in our
# benchmarks. XLA reads XLA_FLAGS once, when the backend initializes (the first JAX computation), so this must run
# before any computation; it is appended to the user's existing flags and does not override an explicit setting.
if "--xla_gpu_enable_command_buffer" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer=").strip()

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jax-lab")
except PackageNotFoundError:
    # Package is not installed (e.g. running from a source checkout)
    __version__ = "0.0.0"

from .core import (
    AdvectionDiffusionBGK,
    BGKSim,
    CLBMSim,
    CarnahanStarling,
    DirichletTemperature,
    EOS,
    KBCSim,
    LatticeD2Q9,
    LatticeD3Q19,
    LatticeD3Q27,
    MRTSim,
    Multiphase,
    MultiphaseBGK,
    MultiphaseCascade,
    MultiphaseMRT,
    MultiphaseThermal,
    NeumannTemperature,
    PengRobinson,
    Precision,
    PrecisionPolicy,
    RedlichKwong,
    RedlichKwongSoave,
    Thermal,
    VanderWaals,
    VoF,
    VoFBGK,
    VoFMRT,
    VoFOpenBoundary,
)

__all__ = [
    "DirichletTemperature",
    "NeumannTemperature",
    "MultiphaseThermal",
    "Thermal",
    "__version__",
    "EOS",
    "CarnahanStarling",
    "PengRobinson",
    "RedlichKwong",
    "RedlichKwongSoave",
    "VanderWaals",
    "LatticeD2Q9",
    "LatticeD3Q19",
    "LatticeD3Q27",
    "AdvectionDiffusionBGK",
    "BGKSim",
    "CLBMSim",
    "KBCSim",
    "MRTSim",
    "Multiphase",
    "MultiphaseBGK",
    "MultiphaseCascade",
    "MultiphaseMRT",
    "Precision",
    "PrecisionPolicy",
    "VoF",
    "VoFBGK",
    "VoFMRT",
    "VoFOpenBoundary",
]
