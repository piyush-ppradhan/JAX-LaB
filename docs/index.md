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

## Project overview

--8<-- "README.md"
