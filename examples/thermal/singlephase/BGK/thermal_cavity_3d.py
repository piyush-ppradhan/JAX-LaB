"""
3D thermal lid-driven cavity using the hybrid thermal LBM solver with the BGK
collision model for the fluid on a D3Q19 lattice.

The fluid is a standard lid-driven cavity (moving top lid at z = nz-1, no-slip
walls elsewhere). The temperature field is advanced with the finite difference
solver: the bottom wall is held hot (T_bottom) and the moving lid cold (T_top)
with Dirichlet conditions, while the four side walls are adiabatic (zero
gradient Neumann).
"""

import os

import numpy as np

from jax_lab.boundary_conditions import BounceBackHalfway, DirichletTemperature, EquilibriumBC, NeumannTemperature
from jax_lab.lattice import LatticeD3Q19
from jax_lab.models import BGKSim
from jax_lab.thermal import Thermal
from jax_lab.utils import save_fields_vtk

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_3d")


class Cavity(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # No-slip walls (all faces except the moving lid at the top, z = nz-1)
        walls = np.concatenate((
            self.bounding_box_indices["left"],
            self.bounding_box_indices["right"],
            self.bounding_box_indices["front"],
            self.bounding_box_indices["back"],
            self.bounding_box_indices["bottom"],
        ))
        self.BCs.append(BounceBackHalfway(tuple(walls.T), self.grid_info, self.precision_policy))

        # Moving lid (top)
        moving_wall = self.bounding_box_indices["top"]
        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precision_policy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.grid_info, self.precision_policy, rho_wall, vel_wall))


class ThermalCavity(Thermal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_temperature_field(self):
        return 0.5 * (T_bottom + T_top) * np.ones((nx, ny, nz, 1))

    def set_thermal_boundary_conditions(self):
        self.thermal_BCs = []
        bbox = self.fluid_solver.bounding_box_indices
        # Adiabatic side walls (applied first so the Dirichlet edges win)
        self.thermal_BCs.append(NeumannTemperature(tuple(bbox["left"].T), normal=(-1, 0, 0)))
        self.thermal_BCs.append(NeumannTemperature(tuple(bbox["right"].T), normal=(1, 0, 0)))
        self.thermal_BCs.append(NeumannTemperature(tuple(bbox["front"].T), normal=(0, -1, 0)))
        self.thermal_BCs.append(NeumannTemperature(tuple(bbox["back"].T), normal=(0, 1, 0)))
        # Hot bottom wall (z = 0), cold moving lid (z = nz-1)
        self.thermal_BCs.append(DirichletTemperature(tuple(bbox["bottom"].T), prescribed=T_bottom))
        self.thermal_BCs.append(DirichletTemperature(tuple(bbox["top"].T), prescribed=T_top))

    def output_data(self, **kwargs):
        timestep = kwargs["timestep"]
        rho = np.array(kwargs["rho"][0, ..., 0])
        u = np.array(kwargs["u"][0, ...])
        T = np.array(kwargs["T"][0, ..., 0])
        fields = {
            "rho": rho,
            "u_x": u[..., 0],
            "u_y": u[..., 1],
            "u_z": u[..., 2],
            "T": T,
        }
        save_fields_vtk(timestep, fields, output_dir)
        print(f"T min/max: {T.min():.4f} / {T.max():.4f}")


if __name__ == "__main__":
    precision = "f32/f32"
    nx = 100
    ny = 100
    nz = 100

    prescribed_vel = 0.1
    T_bottom = 1.0
    T_top = 0.0

    kwargs = {
        "lattice": LatticeD3Q19(precision),
        "omega": 1.0,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": precision,
        "io_rate": 500,
        "print_info_rate": 100,
    }
    fluid = Cavity(**kwargs)

    thermal_kwargs = {
        "fluid_solver": fluid,
        "specific_heat": 1.0,
        "thermal_conductivity": 0.05,
        "apply_buoyancy": True,
        "gravity": np.array([0.0, -1e-4]),
    }
    sim = ThermalCavity(**thermal_kwargs)
    sim.run(1000)
