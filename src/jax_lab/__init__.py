"""
JAX-LaB: a JAX-based differentiable Lattice Boltzmann library for single and
multiphase flow simulations on CPUs, GPUs and TPUs.
"""

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
    Unit,
    VanderWaals,
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
    "Unit",
]
