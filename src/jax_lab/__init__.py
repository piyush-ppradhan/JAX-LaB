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

from .eos import (
    CarnahanStarling,
    EOS,
    Carnahan_Starling,
    PengRobinson,
    Peng_Robinson,
    RedlichKwong,
    RedlichKwongSoave,
    Redlich_Kwong,
    Redlich_Kwong_Soave,
    VanderWaals,
)
from .lattice import LatticeD2Q9, LatticeD3Q19, LatticeD3Q27
from .models import AdvectionDiffusionBGK, BGKSim, CLBMSim, KBCSim, MRTSim
from .multiphase import Multiphase, MultiphaseBGK, MultiphaseCascade, MultiphaseMRT
from .boundary_conditions import DirichletTemperature, NeumannTemperature
from .precision_policy import Precision, PrecisionPolicy
from .thermal import MultiphaseThermal, Thermal

__all__ = [
    "DirichletTemperature",
    "NeumannTemperature",
    "MultiphaseThermal",
    "Thermal",
    "__version__",
    "EOS",
    "CarnahanStarling",
    "Carnahan_Starling",
    "PengRobinson",
    "Peng_Robinson",
    "RedlichKwong",
    "RedlichKwongSoave",
    "Redlich_Kwong",
    "Redlich_Kwong_Soave",
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
]
