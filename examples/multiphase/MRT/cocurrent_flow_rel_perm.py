"""
Analytical benchmark for rel perm in a 2D channel. The wetting fluid flows along the walls (y > H/2 - a/2 or y < H/2 + a/2) while the non-wetting phase
flows along the center (y in [H/2 - a/2, H/2 + a/2]). The wetting phase saturation is defined as a/b.
"""

import os
import operator
import numpy as np

from src.lattice import LatticeD2Q9
from src.utils import save_fields_vtk
from src.multiphase import MultiphaseMRT
from src.boundary_conditions import BounceBack

from functools import partial
from jax import jit, vmap, config
from jax.tree import map, reduce
import jax.numpy as jnp

# config.update("jax_default_matmul_precision", "float32")


class Channel2D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []

        rho = rho_c * np.ones((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u]

        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        walls = tuple(walls.T)
        # apply bounce back boundary condition to the walls
        self.BCs[0].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta[walls], phi[walls], delta_rho[walls]))

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree):
        def f(g_kk):
            return reduce(operator.add, map(lambda _gkk, psi: _gkk * psi, list(g_kk), psi_tree))

        return map(lambda rho, psi, nt: rho / 3 + 1.5 * psi * nt, rho_tree, psi_tree, list(vmap(f, in_axes=(0,))(self.g_kkprime)))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho_total = np.array(kwargs.get("rho_total")[0, :, 1:-1, :])
        rho = np.array(kwargs["rho_tree"][0][0, :, 1:-1, :])
        p = np.array(kwargs["p"][0, 1:-1, 1:-1])
        u = np.array(kwargs["u_tree"][0][0, :, 1:-1, :])
        u_total = np.array(kwargs["u_total"][0, :, 1:-1, :])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "rho": rho[..., 0],
            "rho_total": rho_total[..., 0],
            "ux": u[..., 0],
            "uy": u[..., 1],
            "ux_total": u_total[..., 0],
            "uy_total": u_total[..., 1],
        }
        # HDF5/XDMF output option:
        # from src.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, f"output_channel_{wetting_type}_visc_ratio_{visc_ratio}", "data")
        save_fields_vtk(timestep, fields, f"output_channel_{wetting_type}_visc_ratio_{visc_ratio}", "data")
        if timestep == 20000:
            Q = np.sum(u[self.nx // 2, :, 0])
            file.write(f"{wetting_type},{visc_ratio},{Q}\n")


class Channel2DGeometric(Channel2D):
    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        walls = tuple(walls.T)
        # apply bounce back boundary condition to the walls
        self.BCs[0].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta[walls]))


class CocurrentFlow(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y)

        rho_tree = []

        # Wetting fluid
        rho = fraction * rho_t * np.ones((self.nx, self.ny, 1))
        rho[:, self.ny // 2 - a : self.ny // 2 + a] = (1 - fraction) * rho_t
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        # Non-wetting fluid
        rho = (1 - fraction) * rho_t * np.ones((self.nx, self.ny, 1))
        rho[:, self.ny // 2 - a : self.ny // 2 + a] = fraction * rho_t
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        rho_tree.append(rho)

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        u_tree = [u, u]
        return rho_tree, u_tree

    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        walls = tuple(walls.T)
        # apply bounce back boundary condition to the walls
        self.BCs[0].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta_w[walls], phi_w[walls], delta_rho_w[walls]))
        self.BCs[1].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta_nw[walls], phi_nw[walls], delta_rho_nw[walls]))

    @partial(jit, static_argnums=(0,))
    def compute_potential(self, rho_tree):
        U_tree = map(lambda rho: jnp.zeros_like(rho), rho_tree)
        return rho_tree, U_tree

    @partial(jit, static_argnums=(0,))
    def compute_pressure(self, rho_tree, psi_tree):
        def f(g_kk):
            return reduce(operator.add, map(lambda _gkk, psi: _gkk * psi, list(g_kk), psi_tree))

        return map(lambda rho, psi, nt: rho / 3 + 1.5 * psi * nt, rho_tree, psi_tree, list(vmap(f, in_axes=(0,))(self.g_kkprime)))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs.get("rho_total")[0, ...])
        rho_w = np.array(kwargs["rho_tree"][0][0, ...])
        rho_nw = np.array(kwargs["rho_tree"][1][0, ...])
        p_w = np.array(kwargs["p_tree"][0][...])
        p_nw = np.array(kwargs["p_tree"][1][...])
        p = np.array(kwargs["p"][0, :, :])
        u_w = np.array(kwargs["u_tree"][0][0, ...])
        u_nw = np.array(kwargs["u_tree"][1][0, ...])
        u = np.array(kwargs["u_total"][0, ...])
        timestep = kwargs["timestep"]
        fields = {
            "p": p[..., 0],
            "p_wetting": p_w[..., 0],
            "p_non_wetting": p_nw[..., 0],
            "rho": rho[..., 0],
            "rho_wetting": rho_w[..., 0],
            "rho_non_wetting": rho_nw[..., 0],
            "ux_wetting": u_w[..., 0],
            "uy_wetting": u_w[..., 1],
            "ux_non_wetting": u_nw[..., 0],
            "uy_non_wetting": u_nw[..., 1],
            "ux": u[..., 0],
            "uy": u[..., 1],
        }
        # HDF5/XDMF output option:
        # from src.utils import save_fields_hdf5_xdmf
        # save_fields_hdf5_xdmf(timestep, fields, f"output_cocurrent_flow_a_{a}_visc_ratio_{visc_ratio}", "data")
        save_fields_vtk(timestep, fields, f"output_cocurrent_flow_a_{a}_visc_ratio_{visc_ratio}", "data")

        S_nw = 2 * a / self.ny
        S_w = 1 - S_nw

        Q_w = np.sum(u[self.nx // 2, :, 0][self.ny // 2 + a : self.ny]) + np.sum(u[self.nx // 2, :, 0][0 : self.ny // 2 - a + 1])
        Q_nw = np.sum(u[self.nx // 2, :, 0][self.ny // 2 - a + 1 : self.ny // 2 + a])

        # Compute the flow rate at the final timestep
        if timestep == 30000:
            k_rw = Q_w / single_Q_w
            k_rnw = Q_nw / single_Q_nw
            # Write saturation and relative permeability to file
            file.write(f"{S_w},{S_nw},{k_rw},{k_rnw}\n")


class CocurrentFlowGeometric(CocurrentFlow):
    def set_boundary_conditions(self):
        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.boundingBoxIndices["top"], self.boundingBoxIndices["bottom"]))
        walls = tuple(walls.T)
        # apply bounce back boundary condition to the walls
        self.BCs[0].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta_w[walls]))
        self.BCs[1].append(BounceBack(walls, self.gridInfo, self.precisionPolicy, theta_nw[walls]))


if __name__ == "__main__":
    precision = "f32/f32"
    nx = 150
    ny = 50
    rho_NW = 1.0  # Displaced fluid
    rho_W = 1.0  # Invading fluid
    rho_t = rho_W + rho_NW
    fraction = 0.97

    g_kkprime = 0 * np.ones((2, 2))
    g = 0.025  # 0.025
    g_kkprime[0, 1] = g
    g_kkprime[1, 0] = g
    fx = 1e-5

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

    Visc_ratio = [1.0, 10.0, 0.1]
    Wetting = ["wetting", "non-wetting"]
    Rho = [rho_W, rho_NW]
    Theta = [np.pi / 6, np.pi / 2]
    Phi = [1.4, 1.0]
    Delta_Rho = [0.0, 0.0]

    theta_ = np.ones((nx, ny, 1))
    phi_ = np.ones((nx, ny, 1))
    delta_rho_ = np.ones((nx, ny, 1))

    # 2D channel simulation to get flow rate
    os.system("rm -rf Flow*")
    # Simulate for a range of viscosity ratio and wetting conditions
    file = open("Flow_rate_data.csv", "w")
    file.write("Wetting_type,Viscosity Ratio,Flow Rate(LU)\n")
    # for visc_ratio in [0.1, 1.0, 10]:
    for visc_ratio in [1.0]:
        tau_w = 5.0
        v_w = (tau_w - 0.5) / 3

        v_nw = v_w / visc_ratio
        tau_nw = 3 * v_nw + 0.5

        Tau = [tau_w, tau_nw]
        for i in range(2):
            tau = Tau[i]
            wetting_type = Wetting[i]
            rho_c = Rho[i]
            theta = Theta[i] * theta_
            phi = Phi[i] * phi_
            delta_rho = Delta_Rho[i] * delta_rho_
            s_rho = [0.0]  # Mass
            s_e = [0.4]
            s_eta = [1.0]
            s_j = [0.0]  # Momentum
            s_q = [1.0]
            s_v = [1 / tau]
            kwargs = {
                "n_components": 1,
                "lattice": LatticeD2Q9(precision),
                "nx": nx,
                "ny": ny,
                "nz": 0,
                "g_kkprime": g_kkprime,
                "body_force": [fx, 0.0],
                "precision": precision,
                "M": [M],
                "s_rho": s_rho,
                "s_e": s_e,
                "s_eta": s_eta,
                "s_j": s_j,
                "s_q": s_q,
                "s_v": s_v,
                "k": [0],
                "A": np.zeros((1, 1)),
                "kappa": [0.0],
                "io_rate": 20000,
                "compute_MLUPS": False,
                "print_info_rate": 20000,
                "checkpoint_rate": -1,
                "checkpoint_dir": os.path.abspath("./checkpoints_"),
                "restore_checkpoint": False,
            }
            sim = Channel2D(**kwargs)
            sim.run(20000)
    file.close()

    # Cocurrent flow simulation
    # Initial
    S_w = 0.0
    S_nw = 1.0
    # Flow rate obtained from singlephase simulation for wetting and non-wetting channel obtained from Channel2D, see Flow_rate_data.csv
    Single_Q_w = [0.06191386282444, 0.06191386282444, 0.06191386282444]
    Single_Q_nw = [0.06191386282444, 0.6143753528594971, 0.006619240622967482]

    theta_w = (np.pi / 6) * np.ones((nx, ny, 1))
    theta_nw = (np.pi / 2) * np.ones((nx, ny, 1))
    phi_w = 1.4 * np.ones((nx, ny, 1))
    phi_nw = np.ones((nx, ny, 1))
    delta_rho_w = np.zeros((nx, ny, 1))
    delta_rho_nw = np.zeros((nx, ny, 1))

    os.system("rm -rf Relative* output*")
    for i in range(3):
        visc_ratio = Visc_ratio[i]

        tau_w = 5.0
        v_w = (tau_w - 0.5) / 3

        v_nw = v_w / visc_ratio
        tau_nw = 3 * v_nw + 0.5

        s_rho = [0.0, 0.0]  # Mass
        s_e = [0.4, 0.4]
        s_eta = [1.0, 1.0]  # 1.5
        s_j = [0.0, 0.0]  # Momentum
        s_q = [1.0, 1.0]
        s_v = [1 / tau_nw, 1 / tau_w]

        file = open(f"Relative_permeability_channel_visc_ratio_{visc_ratio}.csv", "w")
        file.write("Wetting phase saturation,Non-wettting phase saturation,Relative permeability (wetting),Relative Permeability (non-wetting)\n")
        single_Q_nw = Single_Q_nw[i]
        single_Q_w = Single_Q_w[i]
        for a in [0, 5, 10, 15, 20, 25]:
            kwargs = {
                "n_components": 2,
                "lattice": LatticeD2Q9(precision),
                "nx": nx,
                "ny": ny,
                "nz": 0,
                "g_kkprime": g_kkprime,
                "body_force": [fx, 0.0],
                "precision": precision,
                "M": [M, M],
                "s_rho": s_rho,
                "s_e": s_e,
                "s_eta": s_eta,
                "s_j": s_j,
                "s_q": s_q,
                "s_v": s_v,
                "k": [0, 0],
                "A": np.zeros((2, 2)),
                "kappa": [0.0, 0.0],
                "io_rate": 30000,
                "compute_MLUPS": False,
                "print_info_rate": 30000,
                "checkpoint_rate": -1,
                "checkpoint_dir": os.path.abspath("./checkpoints_"),
                "restore_checkpoint": False,
            }
            sim = CocurrentFlow(**kwargs)
            sim.run(30000)
        file.close()
