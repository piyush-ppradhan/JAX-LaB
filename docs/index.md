# JAX-LaB Documentation

JAX-LaB provides differentiable lattice Boltzmann solvers for single-phase,
multiphase, multicomponent, and thermal simulations on JAX-supported hardware.

## Run a first simulation

Install the package from a source checkout, including optional visualization
dependencies:

```bash
pip install -e ".[io]"
```

Run the two-dimensional lid-driven cavity example:

```bash
python examples/isothermal/singlephase/cavity2d.py
```

The example defines the lattice, solver, initial fields, boundary conditions,
and output callback in one file. Use it as a compact template for a new
single-phase simulation.

## Choose an example

- Start with `examples/isothermal/singlephase/` for single-phase flows.
- Use `examples/isothermal/multiphase/` for multiphase and multicomponent flows.
- Use `examples/thermal/` for coupled fluid-temperature simulations.
- Use `examples/isothermal/differentiable/` for inverse and differentiable workflows.
- Use `examples/rendering/` for simulations with JAX-native in-situ rendering.

## Package layout

The simulation code is organized under `jax_lab.core`. Import lattices, collision
models, boundary conditions, equations of state, thermal solvers, and utilities
from their corresponding core modules:

```python
from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.models import BGKSim
```

Rendering is an independent package under `jax_lab.render`:

```python
from jax_lab.render import Scene, SurfaceRendering
```

The most commonly used solver classes remain available directly from `jax_lab`
for concise application code. The API reference uses the explicit module paths so
that each class's implementation location is clear.

## Validate a source checkout

Install the development and documentation dependencies, then run the tests and
build the Zensical site:

```bash
pip install -e ".[dev,docs,io]"
pytest
zensical build --clean
```

## Project overview

--8<-- "README.md"
