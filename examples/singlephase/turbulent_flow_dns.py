"""
This example performs DNS (Direct Numerical Simulation) of turbulent flow past the rectangle obstacle using Cascaded LBM model.

1. Lattice: D3Q27
2. Re: 1,400,000
"""

import subprocess
import jax
import numpy as np
import jax.numpy as jnp
from jax import config

from src.utils import save_fields_vtk
from src.models import CLBMSim
from src.lattice import LatticeD3Q19
from src.boundary_conditions import EquilibriumBC, DoNothing, BounceBack

# config.update("jax_default_matmul_precision", "float32")


class Rectangle(CLBMSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        rectangle = np.array(
            [
                [100, j, k]
                for j in range(self.ny // 2 - length // 2, self.ny // 2 + length // 2 + 1)
                for k in range(self.nz // 2 - width // 2, self.nz // 2 + width // 2 + 1)
            ],
            dtype=int,
        )

        wall = np.concatenate((
            rectangle,
            self.boundingBoxIndices["bottom"],
            self.boundingBoxIndices["top"],
            self.boundingBoxIndices["front"],
            self.boundingBoxIndices["back"],
        ))
        self.BCs.append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))

        doNothing = self.boundingBoxIndices["right"]
        self.BCs.append(DoNothing(tuple(doNothing.T), self.gridInfo, self.precisionPolicy))
        self.BCs[-1].implementationStep = "PostCollision"

        inlet = self.boundingBoxIndices["left"]
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)

        vel_inlet[:, 0] = prescribed_vel
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel_inlet))

    def output_data(self, **kwargs):
        timestep = kwargs["timestep"]
        rho = np.array(kwargs["rho"][0, ...])
        u = np.array(kwargs["u"][0, ...])

        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        # HDF5/XDMF output option:
        # from src.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, "output")
        save_fields_vtk(timestep, fields, "output")


if __name__ == "__main__":
    precision = "f32/f32"
    lattice = LatticeD3Q19(precision)

    nx = 600
    ny = 350
    nz = 250

    e = LatticeD3Q19().c.T
    ex = e[:, 0]
    ey = e[:, 1]
    ez = e[:, 2]
    M = np.zeros((19, 19))
    M[0, :] = ex**0
    M[1, :] = ex
    M[2, :] = ey
    M[3, :] = ez
    M[4, :] = ex * ey
    M[5, :] = ex * ez
    M[6, :] = ey * ez
    M[7, :] = ex * ex
    M[8, :] = ey * ey
    M[9, :] = ez * ez
    M[10, :] = ex * ey * ey
    M[11, :] = ex * ez * ez
    M[12, :] = ey * ex * ex
    M[13, :] = ez * ex * ex
    M[14, :] = ey * ez * ez
    M[15, :] = ez * ey * ey
    M[16, :] = ex * ex * ey * ey
    M[17, :] = ex * ex * ez * ez
    M[18, :] = ey * ey * ez * ez

    s2 = 1.4
    s_0 = 1.0  # Conserved
    s_1 = 1.0  # Conserved
    s_2 = s2
    s_b = s2
    s_3 = 1.0
    s_4 = 1.0
    # s_3b = 1.0
    # s_4b = 1.0
    # s_5 = 1.0
    # s_6 = 1.0

    length = 80
    width = 80

    Re = 1.4e4
    prescribed_vel = 0.05
    clength = nx - 1

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    subprocess.run("rm -rf ./*.vtk && rm -rf ./*.png", shell=True, check=True)

    kwargs = {
        "lattice": lattice,
        "omega": omega,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": precision,
        "M": M,
        "s_0": s_0,
        "s_1": s_1,
        "s_b": s_b,
        "s_2": s_2,
        "s_3": s_3,
        "s_4": s_4,
        "s_v": omega,
        "io_rate": 100,
        "print_info_rate": 100,
    }
    sim = Rectangle(**kwargs)
    sim.run(200000)
