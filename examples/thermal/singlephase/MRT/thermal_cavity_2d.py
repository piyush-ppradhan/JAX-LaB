"""
2D thermal lid-driven cavity using the hybrid thermal LBM solver with the MRT
(multi-relaxation time) collision model for the fluid.

The fluid is a standard lid-driven cavity (moving top lid, no-slip walls) with
the D2Q9 orthogonal moment basis of Lallemand & Luo, matching the collision
matrix used in the isothermal MRT droplet examples. The temperature field is
advanced with the finite difference solver: hot bottom wall and cold lid
(Dirichlet), adiabatic side walls (Neumann).
"""

import os

import numpy as np

from jax_lab.boundary_conditions import BounceBackHalfway, EquilibriumBC
from jax_lab.lattice import LatticeD2Q9
from jax_lab.models import MRTSim
from jax_lab.thermal import DirichletTemperature, NeumannTemperature, Thermal
from jax_lab.utils import save_fields_vtk

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_2d")


class Cavity(MRTSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # No-slip walls (left, right and bottom)
        walls = np.concatenate((self.boundingBoxIndices["left"], self.boundingBoxIndices["right"], self.boundingBoxIndices["bottom"]))
        self.BCs.append(BounceBackHalfway(tuple(walls.T), self.gridInfo, self.precisionPolicy))

        # Moving lid (top)
        moving_wall = self.boundingBoxIndices["top"]
        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, rho_wall, vel_wall))


class ThermalCavity(Thermal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_temperature_field(self):
        return 0.5 * (T_bottom + T_top) * np.ones((nx, ny, 1))

    def set_thermal_boundary_conditions(self):
        self.thermal_BCs = []
        bbox = self.fluid_solver.boundingBoxIndices
        # Adiabatic side walls (applied first so the Dirichlet corners win)
        self.thermal_BCs.append(NeumannTemperature(tuple(bbox["left"].T), normal=(-1, 0)))
        self.thermal_BCs.append(NeumannTemperature(tuple(bbox["right"].T), normal=(1, 0)))
        # Hot bottom wall, cold moving lid
        self.thermal_BCs.append(DirichletTemperature(tuple(bbox["bottom"].T), prescribed=T_bottom))
        self.thermal_BCs.append(DirichletTemperature(tuple(bbox["top"].T), prescribed=T_top))

    def output_data(self, **kwargs):
        timestep = kwargs["timestep"]
        rho = np.array(kwargs["rho"][0, ..., 0])
        u = np.array(kwargs["u"][0, ...])
        T = np.array(kwargs["T"][0, ..., 0])
        fields = {"rho": rho, "u_x": u[..., 0], "u_y": u[..., 1], "T": T}
        save_fields_vtk(timestep, fields, output_dir)
        print(f"T min/max: {T.min():.4f} / {T.max():.4f}")


if __name__ == "__main__":
    precision = "f32/f32"
    nx = 100
    ny = 100

    prescribed_vel = 0.1
    T_bottom = 1.0
    T_top = 0.0

    # D2Q9 orthogonal moment basis (same as the isothermal MRT droplet_2d example)
    e = LatticeD2Q9().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((9, 9))
    M[0, :] = en**0
    M[1, :] = -4 * en**0 + 3 * en**2
    M[2, :] = 4 * en**0 - (21 / 2) * en**2 + (9 / 2) * en**4
    M[3, :] = e[:, 0]
    M[4, :] = (-5 * en**0 + 3 * en**2) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (-5 * en**0 + 3 * en**2) * e[:, 1]
    M[7, :] = e[:, 0] ** 2 - e[:, 1] ** 2
    M[8, :] = e[:, 0] * e[:, 1]

    kwargs = {
        "lattice": LatticeD2Q9(precision),
        "nx": nx,
        "ny": ny,
        "nz": 0,
        "precision": precision,
        "M": M,
        "s_rho": 0.0,
        "s_e": 1.2,
        "s_eta": 1.0,
        "s_j": 0.0,
        "s_q": 1.0,
        "io_rate": 2500,
        "print_info_rate": 500,
    }
    fluid = Cavity(**kwargs)

    thermal_kwargs = {
        "fluid_solver": fluid,
        "specific_heat": 1.0,
        "thermal_conductivity": 0.05,
    }
    sim = ThermalCavity(**thermal_kwargs)
    sim.run(5000)
