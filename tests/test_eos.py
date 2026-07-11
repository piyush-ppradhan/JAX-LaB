"""Saturation-pressure regression tests for equations of state."""

import os
from pathlib import Path

# os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from src.eos import Carnahan_Starling, Peng_Robinson, Redlich_Kwong, Redlich_Kwong_Soave, VanderWaal


EOS_DATA_DIRECTORY = Path(__file__).parent / "eos_data"

# The data-file headers round a and b to three decimal places. These are the
# exact benchmark values represented by those headers.
EOS_CASES = (
    pytest.param(Carnahan_Starling, "satpoints_CS.txt", 1.0, 4.0, {}, 0.0, id="carnahan-starling"),
    pytest.param(Peng_Robinson, "satpoints_PR.txt", 2.0 / 49.0, 2.0 / 21.0, {"pr_omega": 0.344}, 6e-5, id="peng-robinson"),
    pytest.param(Redlich_Kwong, "satpoints_RK.txt", 2.0 / 49.0, 2.0 / 21.0, {}, 0.0, id="redlich-kwong"),
    pytest.param(Redlich_Kwong_Soave, "satpoints_SRK.txt", 2.0 / 49.0, 2.0 / 21.0, {"RKS_omega": [0.344]}, 1.5e-3, id="redlich-kwong-soave"),
    pytest.param(VanderWaal, "satpoints_VW.txt", 9.0 / 49.0, 2.0 / 21.0, {}, 0.0, id="van-der-waals"),
)

PRECISION_CASES = (
    pytest.param(jnp.float16, 1.25e-1, 1.25e-1, id="float16"),
    pytest.param(jnp.float32, 2e-5, 2e-5, id="float32"),
    pytest.param(jnp.float64, 2e-7, 2e-7, id="float64"),
)


@pytest.mark.parametrize("dtype, rtol, atol", PRECISION_CASES)
@pytest.mark.parametrize("eos_class, data_file, a, b, extra_parameters, benchmark_atol", EOS_CASES)
def test_saturation_pressure_at_vapor_and_liquid_densities(eos_class, data_file, a, b, extra_parameters, benchmark_atol, dtype, rtol, atol):
    """Compare vapor and liquid EOS pressures with tabulated saturation data."""
    saturation_data = np.loadtxt(EOS_DATA_DIRECTORY / data_file)
    temperature = jnp.asarray(saturation_data[:, 1], dtype=dtype)
    vapor_density = [jnp.asarray(saturation_data[:, 2], dtype=dtype)]
    liquid_density = [jnp.asarray(saturation_data[:, 3], dtype=dtype)]
    expected_pressure = saturation_data[:, 4]

    eos = eos_class(a=[a], b=[b], R=[1.0], temperature_field_type="thermal", **extra_parameters)

    vapor_pressure = np.asarray(eos.EOS_thermal(vapor_density, temperature)[0])
    liquid_pressure = np.asarray(eos.EOS_thermal(liquid_density, temperature)[0])

    atol = max(atol, benchmark_atol)
    np.testing.assert_allclose(vapor_pressure, expected_pressure, rtol=rtol, atol=atol)
    np.testing.assert_allclose(liquid_pressure, expected_pressure, rtol=rtol, atol=atol)
