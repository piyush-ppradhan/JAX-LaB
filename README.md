<h1 align="center">JAX-LaB</h1>

<div align="center">
<h4>

[Documentation](https://piyush-ppradhan.github.io/JAX-LaB/)  |  [Paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS005313?af=R)
</h4>
</div>

A Python-based, differentiable, massively parallel lattice Boltzmann library for modeling multiphase and multiphysics flows & physics-based machine learning

<table width="100%" cellspacing="0" cellpadding="0">
  <tr>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/3D_evaporation_fontainebleau.gif" alt="Evaporation in Fontainebleau sandstone" width="105%">
      <br>
      Evaporation in Fontainebleau sandstone.
    </td>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/drainage.gif" alt="Drainage through a porous geometry" width="105%">
      <br>
      Drainage through a beadpack geometry.
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/rayleigh_taylor_2d.gif" alt="Rayleigh-Taylor instability" width="105%">
      <br>
      Rayleigh-Taylor instability.
    </td>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/pool_boiling_3D.gif" alt="Two-dimensional pool boiling" width="105%">
      <br>
      Three-dimensional pool boiling.
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/contact_angle_hysteresis.gif" alt="Droplet impingement on an inclined surface" width="105%">
      <br>
      Droplet impingement on an inclined surface.
    </td>
    <td width="50%" align="center">
      <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/droplet_funnel.gif" alt="Droplet evaporation with contact-angle hysteresis" width="105%">
      <br>
      Droplet growth from a capillary (in situ render).
    </td>
  </tr>
</table>
<div align="center">
  <img src="https://raw.githubusercontent.com/piyush-ppradhan/JAX-LaB/multiphase/assets/predicted.png" alt="" width="98.5%">
</div>

<div align="center">
  Temporal evolution of the density field determined using neural network for the inverse multiphase flow control problem of forming a droplet at t = 900. The MLP output is used as the initial condition for LBM and the backpropagation step during training leverages the auto-differentiation capabilities of JAX-LaB (see <a href="https://doi.org/10.1029/2025MS005313">paper</a> for details).
</div>

## Key Features
- **JAX Ecosystem Integration:** Works with machine learning libraries such as [Equinox](https://github.com/patrick-kidger/equinox), [Flax](https://github.com/google/flax), [Haiku](https://github.com/deepmind/dm-haiku), and [Optax](https://github.com/google-deepmind/optax).
- **Differentiable LBM:** Provides differentiable kernels for physics and deep learning applications.
- **Scalable and Portable:** Runs on multi-core CPUs, GPUs, and TPUs, with distributed support for simulations spanning hundreds of GPUs and billions of cells.
- **Broad LBM Support:** Includes several boundary conditions and collision kernels, along with Shan-Chen multiphase, multiphysics, and multicomponent flow modeling.
- **User-Friendly Python Interface:** Written entirely in Python, simplifying simulation setup and making library easy to extend.
- **JAX Array and Shardmap:** Offers a NumPy-like interface while leaving performance optimization to the compiler.
- **Visualization:** Supports multiple output options, including JAX-native ray tracer for in situ surface, volume, and vector-field rendering of GPU/TPU arrays.

## Capabilities
### Multiphase Flow Modeling
**Shan-Chen** pseudopotential method with various modifications:
- Support for **high density ratio flows** (tested for density ratios > 10<sup>8</sup>) using improved forcing scheme.
- Incorporates **Equation of State (EOS)** to model multiphase flows. Currently implemented EOS include **Carnahan-Starling**, **Peng-Robinson**, **Redlich-Kwong**, **Redlich-Kwong-Soave**
and **VanderWaals**.
- **Density ratio independent surface tension** control by directly modifying pressure tensor (MRT model).

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
- Binary and ASCII VTK output using [PyVista](https://docs.pyvista.org/)
- HDF5/XDMF output using [h5py](https://docs.h5py.org/)
- JAX-native in-situ surface, refractive volume, and vector-field rendering and image output
- Distributed asynchronous checkpointing using [orbax](https://github.com/google/orbax) 
- 3D mesh voxelizer using [trimesh](https://trimesh.org/)

### Boundary Conditions
- **Equilibrium:** Sets prescribed velocity or pressure using equilibrium populations.
- **Full-Way Bounceback:** Reflects populations to impose a stationary, no-slip wall.
- **Half-Way Bounceback:** Imposes a no-slip wall halfway between fluid and solid nodes.
- **Do Nothing:** Allows populations to pass through unmodified.
- **Zou-He:** Imposes a prescribed velocity or pressure profile.
- **Regularized:** Provides a more stable, but more expensive, alternative to Zou-He.
- **Extrapolation Outflow:** Reduces wave reflections using extrapolation.
- **Non-Equilibrium Extrapolation:** Open boundary condition with prescribed density.
- **Exact Non-Equilibrium Extrapolation:** Mass-corrected open boundary condition with prescribed density.
- **Interpolated Bounceback:** Applies the Bouzidi scheme to curved or off-lattice walls.
- **Convective Outflow:** Supports outflow in applications such as porous media flow.
- **Dirichlet:** Prescribes temperature at the boundary.
- **Neumann:** Prescribes the normal temperature gradient.

## Accompanying Paper

The accompanying paper, published in Journal of Advances in Modeling Earth Systems (JAMES), is available [here](https://doi.org/10.1029/2025MS005313).

## Documentation

Complete API documentation is available [here](https://piyush-ppradhan.github.io/JAX-LaB/), or you can build and preview it locally:
```bash
pip install -e ".[docs]"
zensical serve
```

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
| [trimesh](https://trimesh.org/) + [Rtree](https://rtree.readthedocs.io/en/stable/) | `voxelize_stl` |

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

> [!NOTE]
> On macOS, please use the standard CPU installation, as JAX does not support GPU acceleration on this platform.

Run an example:
```bash
python3 examples/isothermal/singlephase/cavity2d.py
```

Solver components live under `jax_lab.core`, while the JAX-native rendering API
lives under `jax_lab.render`. For example:

```python
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.models import BGKSim
from jax_lab.render import Scene, SurfaceRendering
```

## Citation
If you use this software, please cite it as follows:
```bibtex
@article{pradhan_jax-lab_2026,
    title = {{JAX}-{LaB}: {A} {High}-{Performance}, {Differentiable} {Lattice} {Boltzmann} {Library} for {Modeling} {Multiphase} {Fluid} {Dynamics} in {Geosciences} and {Engineering}},
    volume = {18},
    copyright = {© 2026 The Author(s). Journal of Advances in Modeling Earth Systems published by Wiley Periodicals LLC on behalf of American Geophysical Union.},
    issn = {1942-2466},
    shorttitle = {{JAX}-{LaB}},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1029/2025MS005313},
    doi = {10.1029/2025MS005313},
    language = {en},
    number = {2},
    urldate = {2026-02-20},
    journal = {Journal of Advances in Modeling Earth Systems},
    author = {Pradhan, Piyush and Gentine, Pierre and Kelly, Shaina},
    year = {2026},
    keywords = {GPU, JAX, Lattice Boltzmann method, Python, Shan-Chen method, multiphase flow},
    pages = {e2025MS005313},
}
```
