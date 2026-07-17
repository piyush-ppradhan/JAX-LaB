[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
# JAX-LaB: A Python-based, Accelerated, Differentiable Massively Parallel Lattice Boltzmann Library for Modeling Multiphase and Multiphysics Flows & Physics-Based Machine Learning

<div align="center">
  <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/3D_evaporation_fontainebleau.gif" alt="" width="49%" title="Evaporation in Fontainebleau sandstone">
  <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/drainage.gif" alt="" width="49%" title="Drainage Simulation">
</div>
<div align="center">
  <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/rayleigh_taylor_2d.gif" alt="" width="49%" title="Rayleigh-Taylor instability">
  <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/2d_pool_boiling.gif" alt="" width="49%" title="Two-dimensional pool boiling">
</div>
<div align="center">
  <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/contact_angle_hysteresis.gif" alt="" width="49%" title="Contact Angle Hysteresis">
  <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/droplet_evap_hysteresis.gif" alt="" width="49%" title="Droplet Evaporation">
</div>
<div align="center">
  <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/predicted.png" alt="" width="98.5%">
</div>

<div align="center">
  Temporal evolution of the density field determined using neural network for the inverse multiphase flow control problem of forming a droplet at t = 900. The MLP output is used as the initial condition for LBM and the backpropagation step during training leverages the auto-differentiation capabilities of JAX-LaB (see <a href="https://doi.org/10.1029/2025MS005313">paper</a> for details).
</div>
<!-- <p align="center">
  On GPU in-situ rendering using <a href="https://github.com/loliverhennigh/PhantomGaze">PhantomGaze</a> library (no I/O). Droplet impact on dry surface using MRT collision model with ~16 million cells.
  (single component, multiphase simulation, density ratio: 350, fluid modeled using Peng-Robinson EOS).
</p> -->
<!-- <p align="center">
  In-situ GPU rendering of drainage in a porous geometry. BGK collision model, 110 million cells.
</p> -->
<!-- <p align="center">
    Contact angle hysteresis: Left: droplet impinging on inclined surface (MRT collision model). Right: Droplet undergoing evaporation (Cascaded collision model). Simulated using Peng-Robinson EOS, geometric wetting.
</p> -->
<!-- <p align="center">
    Time evolution of liquid distribution in a Fontainebleau sandstone during evaporation simulated using the Cascaded (central-moment) collision model.
</p> -->
<!-- <p align="center">
    Vapor generation and departure during a two-dimensional pool-boiling simulation.
</p> -->

## Key Features
- **JAX Ecosystem Integration:** Works with machine learning libraries such as [Equinox](https://github.com/patrick-kidger/equinox), [Flax](https://github.com/google/flax), [Haiku](https://github.com/deepmind/dm-haiku), and [Optax](https://github.com/google-deepmind/optax).
- **Differentiable LBM:** Provides differentiable kernels for physics and deep learning applications.
- **Scalable and Portable:** Runs on multi-core CPUs, GPUs, and TPUs, with distributed support for simulations spanning hundreds of GPUs and billions of cells.
- **Broad LBM Support:** Includes several boundary conditions and collision kernels, along with Shan-Chen multiphase, multiphysics, and multicomponent flow modeling.
- **User-Friendly Python Interface:** Makes simulations easy to configure and the library straightforward to extend.
- **JAX Array and Shardmap:** Offers a NumPy-like interface while leaving performance optimization to the compiler.
- **Visualization:** Supports multiple output options, including in-situ GPU rendering with [PhantomGaze](https://github.com/loliverhennigh/PhantomGaze).

## Capabilities
### Multiphase Flow Modeling
**Shan-Chen** pseudopotential method with various modifications:
- Support for **high density ratio flows** (tested for density ratios > 10<sup>8</sup>) using improved forcing scheme.
- Incorporates **Equation of State (EOS)** to model multiphase flows. Currently implemented EOS include **Carnahan-Starling**, **Peng-Robinson**, **Redlich-Kwong**, **Redlich-Kwong-Soave**
and **VanderWaals**.
- **Density ratio independent surface tension** control by directly modifying pressure tensor (MRT collision model only).

### Multicomponent Flow Support
Computations use *pytrees* to **model any number of components**, each with its own equation of state, initial condition, and boundary conditions, without requiring library modifications.

### Thermal Flow Modeling
- **Hybrid thermal LBM solver** for two- and three-dimensional **single-phase, multiphase, and multicomponent** flows.
- Thermal equation is solved using lattice-based finite-difference stencils and **fourth-order Runge-Kutta** time integration.

### Wetting model
- [Geometric wetting scheme](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.87.013301)
- [Improved virtual density scheme](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.053313)


### Collision Models
- **BGK**
- **Multi-Relaxation Time (MRT)**
- **Cascaded (Central Moment)**
- **KBC**

### Lattice
- D2Q9
- D3Q19
- D3Q27

### Machine Learning
- Easy integration with JAX's ecosystem of machine learning libraries
- Differentiable LBM kernels both for single and multiphase flows
- Differentiable boundary conditions

### Compute Capabilities
- Distributed Multi-GPU support
- Mixed-Precision support (store vs compute)

### Output
- Binary and ASCII VTK output (based on [PyVista](https://docs.pyvista.org/) library)
- HDF5/XDMF output (based on [h5py](https://docs.h5py.org/)) to maximize I/O speed and minimize storage requirement
- In-situ rendering using [PhantomGaze](https://github.com/loliverhennigh/PhantomGaze) library
- [Orbax](https://github.com/google/orbax)-based distributed asynchronous checkpointing
- Image Output
- 3D mesh voxelizer using [trimesh](https://trimesh.org/)

### Boundary Conditions
- **Equilibrium:** Sets prescribed velocity or pressure using equilibrium populations.
- **Full-Way Bounceback:** Reflects populations to impose a stationary, no-slip wall.
- **Half-Way Bounceback:** Imposes a no-slip wall halfway between fluid and solid nodes.
- **Do Nothing:** Allows populations to pass through unmodified.
- **Zou-He:** Imposes a prescribed velocity or pressure profile.
- **Regularized:** Provides a more stable, but more expensive, alternative to Zou-He.
- **Extrapolation Outflow:** Reduces wave reflections using extrapolation.
- **Interpolated Bounceback:** Applies the Bouzidi scheme to curved or off-lattice walls.
- **Convective Outflow:** Supports outflow in applications such as porous media flow.
- **Dirichlet:** Prescribes temperature at the boundary.
- **Neumann:** Prescribes the normal temperature gradient.

## Accompanying Paper

The accompanying paper, published in Journal of Advances in Modeling Earth Systems (JAMES), is available [here](https://doi.org/10.1029/2025MS005313).

## Installation Guide

JAX-LaB is distributed as the `jax-lab` package (import name `jax_lab`). The default install targets CPU:
```bash
pip install jax-lab
```

### Accelerator support
Hardware acceleration is selected through dependency extras, which delegate the compiled backend packages to [JAX's own extras](https://docs.jax.dev/en/latest/installation.html):
```bash
pip install "jax-lab[cuda13]"   # NVIDIA GPU (CUDA 13, bundled)
pip install "jax-lab[cuda12]"   # NVIDIA GPU (CUDA 12, bundled)
pip install "jax-lab[tpu]"      # Google TPU
pip install "jax-lab[rocm]"     # AMD GPU (ROCm, local toolkit)
```
Use `cuda13-local`/`cuda12-local` instead if you manage the CUDA toolkit yourself.

### Optional I/O and visualization dependencies

The I/O and visualization utilities load their dependencies lazily (at call time, not at import time), so the core solver runs without them. The following packages are only needed if you call the corresponding functions:

| Package | Required by |
|---|---|
| [PyVista](https://docs.pyvista.org/) | `save_fields_vtk`, `save_BCs_vtk`, `live_volume_rendering` |
| [h5py](https://docs.h5py.org/) | `save_fields_hdf5_xdmf` |
| [matplotlib](https://matplotlib.org/) | `save_image`, `live_volume_rendering` |
| [trimesh](https://trimesh.org/) + Rtree | `voxelize_stl` |

Calling one of these functions without its dependency installed raises an `ImportError` naming the missing package. The `io` extra installs all of them at once (recommended for running the examples, most of which write VTK or image output):
```bash
pip install "jax-lab[io]"
```
Extras can be combined, e.g. `pip install "jax-lab[cuda13,io]"`.

### Development install

To work on JAX-LaB itself or run the bundled examples, install from source in editable mode:
```bash
git clone https://github.com/piyush-ppradhan/JAX-LaB
cd JAX-LaB
pip install -e ".[dev,io]"
```

**Note:** We encountered challenges when executing JAX-LaB on Apple GPUs due to the lack of support for certain operations in the Metal backend. We advise using the CPU backend on Mac OS. We will be testing JAX-LaB on Apple's GPUs in the future and will update this section accordingly.

Run an example:
```bash
python3 examples/singlephase/cavity2d.py
```
