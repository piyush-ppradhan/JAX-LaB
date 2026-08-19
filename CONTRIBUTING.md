# Contributing to JAX-LaB

Thank you for your interest in improving JAX-LaB. Contributions can include bug reports, feature requests, documentation, examples, tests, and code.

## Ask Questions and Propose Changes

Use [GitHub Issues](https://github.com/piyush-ppradhan/JAX-LaB/issues) to report a bug or propose an enhancement. Before opening a new issue, check whether a
related issue already exists.

For bugs, include:

- A minimal, reproducible example
- The expected and actual behavior
- Your Python, JAX, and JAX-LaB versions
- Your operating system and compute backend (CPU, CUDA, ROCm, or TPU)
- The complete error message or traceback

For substantial features or changes to public APIs, open an issue before starting implementation. Early discussion helps establish the scope and avoids duplicated work.

## Development Setup

JAX-LaB requires Python 3.12 or newer. Clone your fork and install the package in editable mode with the development and I/O dependencies:

```bash
git clone https://github.com/<your-username>/JAX-LaB.git
cd JAX-LaB
python -m pip install -e ".[dev,io]"
```

This installs the CPU version of JAX. To develop against an accelerator, add the
appropriate extra from `pyproject.toml`, for example:

```bash
python -m pip install -e ".[dev,io,cuda13]"
```

## Making a Change

Create a focused branch from the `multiphase` branch, which is JAX-LaB's default development branch:

```bash
git switch multiphase
git pull --ff-only
git switch -c descriptive-branch-name
```

Keep each pull request limited to one logical change. Follow the existing project structure:

- Library code belongs in `jax_lab/`.
- Tests belong in `tests/` and should use the `test_*.py` naming convention.
- Runnable simulations and demonstrations belong in `examples/`.
- Documentation belongs in `docs/`.

Add or update tests for behavior changes and bug fixes. If numerical results are affected, explain the expected change and choose tolerances appropriate for the
supported precision policies.

## Coding Guidelines

Follow the established style in the surrounding code and these project rules:

- Use descriptive `snake_case` names for functions and `PascalCase` names for classes.
- Reuse existing implementations instead of duplicating functionality.
- Keep JIT-compiled functions functionally pure.
- Mark arguments as static only when required. In particular, avoid making JAX arrays static unless they are class members.
- Avoid unnecessary host-device transfers and Python-side work in performance-critical paths.
- Remember that out-of-bounds JAX indexing may not raise an error; validate indexing logic carefully.
- Keep docstrings concise, document inputs, outputs, types, and assumptions, and cite papers when an implementation relies on them.

Performance is a core requirement. Changes to solver kernels should preserve vectorization and accelerator compatibility. Include benchmark evidence in the
pull request when a change may materially affect runtime, memory use, or JIT compilation behavior.

## Tests, Linting, and Formatting

Run the same core checks used by continuous integration before opening a pull
request:

```bash
ruff check .
pytest
```

Format modified Python files with Ruff:

```bash
ruff format .
```

When practical, run the full test suite on each backend affected by the change. At minimum, make sure the CPU test suite passes. Tests should be deterministic,
small enough for continuous integration, and cover both 2D and 3D behavior when the implementation supports both.

To build and preview documentation changes locally, install the documentation extra and run Zensical:

```bash
python -m pip install -e ".[docs]"
zensical serve
```

## Submitting a Pull Request

Open a pull request against the `multiphase` branch in [JAX-LaB](https://github.com/piyush-ppradhan/JAX-LaB). In the description:

- Explain the problem and the approach taken.
- Link the related issue, if one exists.
- Describe how the change was tested, including hardware or accelerator details
  when relevant.
- Call out API changes, numerical differences, performance effects, and known
  limitations.
- Update documentation and examples when user-facing behavior changes.

Ensure all continuous integration checks pass and address review feedback with additional commits. Maintainers may ask for changes to keep the implementation
consistent with JAX-LaB's API, numerical methods, and performance goals.

JAX-LaB is distributed under the [Apache License 2.0](LICENSE). Contributions must be compatible with that license and must not include code or data that you
do not have permission to contribute.
