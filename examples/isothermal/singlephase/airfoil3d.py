"""
This is a example for simulating fluid flow around a NACA airfoil using the lattice Boltzmann method (LBM).
The LBM is a computational fluid dynamics method for simulating fluid flow and is particularly effective
for complex geometries and multiphase flow.

In this example you'll be introduced to the following concepts:

1. Lattice: The example uses a D3Q27 lattice, which is a three-dimensional lattice model that considers
    27 discrete velocity directions. This allows for a more accurate representation of the fluid flow
    in three dimensions.

2. NACA Airfoil Generation: The example includes a function to generate a NACA airfoil shape, which is
    common in aerodynamics. The function allows for customization of the length, thickness, and angle
    of the airfoil.

3. Boundary Conditions: The example includes several boundary conditions. These include a "bounce back"
    condition on the airfoil surface and the top and bottom of the domain, a "do nothing" condition
    at the outlet (right side of the domain), and an "equilibrium" condition at the inlet
    (left side of the domain) to simulate a uniform flow.

4. Simulation Parameters: The example allows for the setting of various simulation parameters,
    including the Reynolds number, inlet velocity, and characteristic length.

5. In-situ visualization: The example renders q-criterion surfaces with JAX-LaB's
    JAX-native renderer while the field is still on the accelerator.
"""

import numpy as np

# from IPython import display
from jax_lab.core.models import BGKSim, KBCSim
from jax_lab.core.lattice import LatticeD3Q19, LatticeD3Q27
from jax_lab.core.boundary_conditions import DoNothing, BounceBack, EquilibriumBC
from jax_lab.core.utils import save_fields_vtk, q_criterion
from jax_lab.render import Light, Scene, SurfaceRendering
from jax import config
import jax.numpy as jnp
import subprocess

# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax
import scipy

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# config.update("jax_default_matmul_precision", "float32")


def makeNacaAirfoil(length, thickness=30, angle=0):
    def nacaAirfoil(x, thickness, chordLength):
        coeffs = [0.2969, -0.1260, -0.3516, 0.2843, -0.1015]
        exponents = [0.5, 1, 2, 3, 4]
        yt = [coeff * (x / chordLength) ** exp for coeff, exp in zip(coeffs, exponents)]
        yt = 5.0 * thickness / 100 * chordLength * np.sum(yt)

        return yt

    x = np.linspace(0, length, num=length)
    yt = np.array([nacaAirfoil(xi, thickness, length) for xi in x])

    y_max = int(np.max(yt)) + 1
    domain = np.zeros((2 * y_max, len(x)), dtype=int)

    for i, xi in enumerate(x):
        upper_bound = int(y_max + yt[i])
        lower_bound = int(y_max - yt[i])
        domain[lower_bound:upper_bound, i] = 1

    domain = scipy.ndimage.rotate(domain, angle, reshape=True)
    domain = np.where(domain > 0.5, 1, 0)

    return domain


class Airfoil(KBCSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        tx, ty = np.array([self.nx, self.ny], dtype=int) - airfoil.shape

        airfoil_mask = np.pad(airfoil, ((tx // 3, tx - tx // 3), (ty // 2, ty - ty // 2)), "constant", constant_values=False)
        airfoil_mask = np.repeat(airfoil_mask[:, :, np.newaxis], self.nz, axis=2)

        airfoil_indices = np.argwhere(airfoil_mask)
        wall = np.concatenate((airfoil_indices, self.bounding_box_indices["bottom"], self.bounding_box_indices["top"]))
        self.BCs.append(BounceBack(tuple(wall.T), self.grid_info, self.precision_policy))

        # Store airfoil boundary for visualization
        self.visualization_bc = jnp.zeros((self.nx, self.ny, self.nz), dtype=jnp.float32)
        self.visualization_bc = self.visualization_bc.at[tuple(airfoil_indices.T)].set(1.0)

        doNothing = self.bounding_box_indices["right"]
        self.BCs.append(DoNothing(tuple(doNothing.T), self.grid_info, self.precision_policy))

        inlet = self.bounding_box_indices["left"]
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_inlet = np.zeros((inlet.shape), dtype=self.precision_policy.compute_dtype)

        vel_inlet[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_inlet, vel_inlet))

    def output_data(self, **kwargs):
        # Compute q-criterion and vorticity using finite differences
        # Get velocity field
        u = kwargs["u"][..., 1:-1, :]
        # vorticity and q-criterion
        norm_mu, q = q_criterion(u)

        dx = 0.01
        focal_point = (self.visualization_bc.shape[0] * dx / 2, self.visualization_bc.shape[1] * dx / 2, self.visualization_bc.shape[2] * dx / 2)
        radius = 5.0
        angle = kwargs["timestep"] * 0.0001
        camera_position = (focal_point[0] + radius * np.sin(angle), focal_point[1], focal_point[2] + radius * np.cos(angle))
        scene = Scene(
            {
                "q_criterion": SurfaceRendering(
                    value_range=(0.00003, 1.0),
                    color=(0.1, 0.45, 1.0),
                    metallic=0.15,
                    roughness=0.3,
                    spacing=(dx, dx, dx),
                    color_field="vorticity_magnitude",
                    color_range=(0.0, 0.05),
                    colormap="jet",
                ),
                "airfoil": SurfaceRendering(
                    value_range=(0.95, 1.0),
                    color=(0.8, 0.82, 0.86),
                    metallic=0.5,
                    roughness=0.25,
                    spacing=(dx, dx, dx),
                ),
            },
            position=camera_position,
            target=focal_point,
            up=(0.0, 1.0, 0.0),
            resolution=(1920, 1080),
            background_color=(0.0, 0.0, 0.0),
            lights=(Light(position=(0.0, 4.0, -2.0), intensity=12.0),),
            output_dir=".",
        )

        # HDF5/XDMF output option:
        # from jax_lab.core.utils import save_fields_hdf5_xdmf
        # fields = {"q": np.array(q), "vorticity_magnitude": np.array(norm_mu)}
        # static_fields = {"flag": np.array(self.visualization_bc)}
        # save_fields_hdf5_xdmf(kwargs["timestep"], fields, "output", "airfoil", static_fields=static_fields)

        scene.render(
            {
                "q_criterion": q,
                "vorticity_magnitude": norm_mu,
                "airfoil": self.visualization_bc,
            },
            timestep=kwargs["timestep"],
            filename=f"q_criterion_{kwargs['timestep']:07d}.png",
        )


if __name__ == "__main__":
    airfoil_length = 101
    airfoil_thickness = 30
    airfoil_angle = 20
    airfoil = makeNacaAirfoil(length=airfoil_length, thickness=airfoil_thickness, angle=airfoil_angle).T
    precision = "f32/f32"

    lattice = LatticeD3Q27(precision)

    nx = airfoil.shape[0]
    ny = airfoil.shape[1]

    ny = 3 * ny
    nx = 5 * nx
    nz = 101

    Re = 30000.0
    prescribed_vel = 0.1
    clength = airfoil_length

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    subprocess.run("rm -rf ./*.vtk && rm -rf ./*.png", shell=True, check=True)

    # Set the parameters for the simulation
    kwargs = {
        "lattice": lattice,
        "omega": omega,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": precision,
        "io_rate": 100,
        "print_info_rate": 100,
    }

    sim = Airfoil(**kwargs)
    sim.run(20000)
