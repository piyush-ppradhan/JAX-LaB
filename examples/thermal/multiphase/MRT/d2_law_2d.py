"""
Finite-size 2D droplet evaporation benchmark using the multiphase hybrid thermal LBM solver with the MRT collision model.

A liquid droplet (VanderWaals EOS, parameters from the isothermal MRT droplet_2d example) is initialized in a periodic flow domain.
The droplet is at the saturation temperature T_liq = 0.8 Tc while the surrounding vapor is superheated at the critical temperature Tc,
with a smooth tanh interpolation across the interface. Fixed-temperature boundaries keep the far field hot while heat diffusing into the
droplet drives evaporation.

The simulation is run for each thermal conductivity in K_VALUES. output_data records the droplet diameter every io_rate steps and stops
before the diffuse interface becomes comparable to the droplet diameter. Each history is written to a CSV file and a combined plot is saved,
along with HDF5/XDMF snapshots at regular intervals.
"""

import glob
import os
import sys

import numpy as np
from jax import config

from jax_lab.boundary_conditions import DirichletTemperature
from jax_lab.eos import VanderWaals
from jax_lab.lattice import LatticeD2Q9
from jax_lab.multiphase import MultiphaseMRT
from jax_lab.thermal import MultiphaseThermal
from jax_lab.utils import save_fields_hdf5_xdmf

# from jax_lab.utils import save_fields_vtk

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_d2_law_2d")

# config.update("jax_enable_x64", True)


def vapor_density_at_pressure(T, pressure):
    """
    Compute the stable vapor density at a prescribed VanderWaals pressure.

    Parameters
    ----------
    T (numpy.ndarray or float): Temperature in lattice units.
    pressure (float): Target thermodynamic pressure in lattice units.

    Returns
    -------
    numpy.ndarray: Vapor density satisfying p_EOS(rho, T) = pressure.
    """
    T = np.asarray(T, dtype=np.float64)
    rho = rho_g * T_liq / T
    for _ in range(12):
        residual = rho * R * T / (1.0 - b * rho) - a * rho**2 - pressure
        derivative = R * T / (1.0 - b * rho) ** 2 - 2.0 * a * rho
        rho = rho - residual / derivative
    return rho


class EvaporatingDroplet(MultiphaseMRT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y, indexing="ij")

        center_x = 0.5 * (self.nx - 1)
        center_y = 0.5 * (self.ny - 1)
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        liquid_fraction = 0.5 - 0.5 * np.tanh(2 * (dist - r) / width)
        T = T_liq + (T_vap - T_liq) * (1.0 - liquid_fraction)
        rho_vapor = vapor_density_at_pressure(T, p_sat)
        rho = liquid_fraction * rho_l + (1.0 - liquid_fraction) * rho_vapor
        rho = rho.reshape((self.nx, self.ny, 1))
        rho = self.distributed_array_init((self.nx, self.ny, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho_tree = [self.precision_policy.cast_to_output(rho)]

        u = np.zeros((self.nx, self.ny, 2))
        u = self.distributed_array_init((self.nx, self.ny, 2), self.precision_policy.compute_dtype, init_val=u)
        u_tree = [self.precision_policy.cast_to_output(u)]
        return rho_tree, u_tree


class DropletThermal(MultiphaseThermal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xdmf_prefix = kwargs.get("xdmf_prefix", "fields")
        self.xdmf_rate = kwargs.get("xdmf_rate", 10000)
        # self.vtk_prefix = kwargs.get("vtk_prefix", "fields")
        # self.vtk_rate = kwargs.get("vtk_rate", 10000)
        self.minimum_diameter = kwargs.get("minimum_diameter", 0.0)
        self.measurement_start = kwargs.get("measurement_start", 0)
        self.D0 = None
        self.history = []

    def initialize_temperature_field(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        x, y = np.meshgrid(x, y, indexing="ij")

        center_x = 0.5 * (self.nx - 1)
        center_y = 0.5 * (self.ny - 1)
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # Liquid at T_liq inside, superheated vapor at Tc outside (same smooth profile as density)
        T = 0.5 * (T_liq + T_vap) - 0.5 * (T_liq - T_vap) * np.tanh(2 * (dist - r) / width)
        return T.reshape((self.nx, self.ny, 1))

    def set_thermal_boundary_conditions(self):
        faces = ("left", "right", "bottom", "top")
        bbox = self.fluid_solver.bounding_box_indices
        far_field = np.unique(np.concatenate([bbox[face] for face in faces]), axis=0)
        self.thermal_BCs = [DirichletTemperature(tuple(far_field.T), prescribed=T_vap)]

    def output_data(self, **kwargs):
        timestep = kwargs["timestep"]
        rho = np.array(kwargs["rho"][0][0, ..., 0])
        T = np.array(kwargs["T"][0, ..., 0])

        if not np.isfinite(rho).all():
            print(f"Simulation diverged (non-finite density) at timestep {timestep}; increase rk_substeps.")
            self.stop_simulation = True
            return

        D, liquid_volume, liquid_fraction = droplet_diameter(rho)
        if timestep % self.xdmf_rate == 0:
            save_fields_hdf5_xdmf(timestep, {"rho": rho, "T": T}, output_dir, prefix=self.xdmf_prefix)
            # save_fields_vtk(timestep, {"rho": rho, "T": T}, output_dir, prefix=self.vtk_prefix)
        if liquid_volume <= 1.0 or D <= self.minimum_diameter:
            print(f"Stopping at timestep {timestep} before the droplet enters the diffuse-interface shrinkage regime.")
            self.stop_simulation = True
            return
        if timestep < self.measurement_start:
            return

        if self.D0 is None:
            self.D0 = D
        coordinates = np.indices(rho.shape)
        centroid = np.array([(liquid_fraction * coordinate).sum() / liquid_volume for coordinate in coordinates])
        self.history.append((timestep, D, centroid[0], centroid[1]))

        if timestep % self.xdmf_rate == 0:
            print(
                f"timestep {timestep}: (D/D0)^2 = {(D / self.D0) ** 2:.4f}, "
                # f"centroid = ({centroid[0]:.3f}, {centroid[1]:.3f}), liquid volume = {liquid_volume:.1f}"
            )


def droplet_diameter(rho):
    """
    Measure the subgrid droplet diameter from the phase-weighted area.

    Parameters
    ----------
    rho (numpy.ndarray): Density field of shape (nx, ny).

    Returns
    -------
    tuple (float, float, numpy.ndarray): Effective diameter, liquid area and liquid fraction.

    Notes
    -----
    The bulk phase densities are measured from the current field because vapor
    density increases as mass evaporates into the periodic domain.
    """
    rho_vapor, rho_liquid = np.quantile(rho, (0.001, 0.999))
    density_span = rho_liquid - rho_vapor
    if density_span <= 1.0e-12:
        return 0.0, 0.0, np.zeros_like(rho)
    liquid_fraction = np.clip((rho - rho_vapor) / density_span, 0.0, 1.0)
    liquid_volume = float(liquid_fraction.sum())
    return np.sqrt(4.0 * liquid_volume / np.pi), liquid_volume, liquid_fraction


if __name__ == "__main__":
    precision = "f32/f32"
    nx = 200
    ny = 200

    r = 30
    width = 3

    # VanderWaals EOS parameters from the isothermal MRT droplet_2d example
    a = 9 / 49
    b = 2 / 21
    R = 1.0
    Tc = 0.5714285714
    T_liq = 0.8 * Tc
    T_vap = Tc
    specific_heat = 12.0

    # Maxwell construction densities at T = 0.8 Tc
    rho_l = 6.764470400
    rho_g = 0.838834226
    p_sat = rho_g * R * T_liq / (1.0 - b * rho_g) - a * rho_g**2
    rho_vapor_far = float(vapor_density_at_pressure(T_vap, p_sat))
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

    # Thermal conductivities can also be passed on the command line to run a subset
    K_VALUES = [float(v) for v in sys.argv[1:]] or [1 / 3, 2 / 3]
    io_rate = 10000  # min(1000, max(1, t_max // 20))

    # Largest eigenvalue magnitude of the isotropic laplacian stencil; the
    # explicit RK4 update is stable for alpha * lambda_max * dt < 2.785 with
    # alpha = K / (rho c_v), worst in the low density vapor phase.
    lambda_max = 16.0 / 3.0

    for K in K_VALUES:
        t_max = 270000 if (np.abs(K - np.float32(1 / 3)) <= 1e-4) else 140000
        os.makedirs(output_dir, exist_ok=True)
        stale_outputs = glob.glob(os.path.join(output_dir, f"fields_K{K:.3f}_*.hdf5"))
        stale_outputs.extend(glob.glob(os.path.join(output_dir, f"fields_K{K:.3f}_*.xdmf")))
        # stale_outputs = glob.glob(os.path.join(output_dir, f"fields_K{K:.3f}_*.vtk"))
        stale_outputs.append(os.path.join(output_dir, f"d2_law_K{K:.3f}.csv"))
        for stale_output in stale_outputs:
            if os.path.exists(stale_output):
                os.remove(stale_output)

        rk_substeps = max(1, int(np.ceil((K / (rho_vapor_far * specific_heat)) * lambda_max / 2.5)))
        eos = VanderWaals(a=[a], b=[b], R=[R], temperature_field_type="thermal")
        kwargs = {
            "n_components": 1,
            "lattice": LatticeD2Q9(precision),
            "nx": nx,
            "ny": ny,
            "nz": 0,
            "g_kkprime": -1.0 * np.ones((1, 1)),
            "EOS": eos,
            "body_force": [0.0, 0.0],
            "k": [0.16],
            "A": -0.032 * np.ones((1, 1)),
            "M": [M],
            "s_rho": [0.0],
            "s_e": [0.1],
            "s_eta": [1.0],
            "s_j": [0.0],
            "s_q": [1.0],
            "s_v": [1.0],
            "kappa": [0.0],
            "precision": precision,
            "io_rate": io_rate,
            "print_info_rate": 10000,
            "checkpoint_rate": 0,
        }
        fluid = EvaporatingDroplet(**kwargs)
        sim = DropletThermal(
            fluid_solver=fluid,
            specific_heat=specific_heat,
            thermal_conductivity=float(K),
            rk_substeps=rk_substeps,
            xdmf_prefix=f"fields_K{K:.3f}",
            # vtk_prefix=f"fields_K{K:.3f}",
            minimum_diameter=4.0 * width,
            measurement_start=min(1000, t_max // 2),
        )
        print(f"K = {K:.4f}: rk_substeps = {rk_substeps}")
        sim.run(t_max)

        history = np.array(sim.history)
        diameter = history[:, 1]
        diameter_squared_normalized = (diameter / diameter[0]) ** 2
        kinematic_viscosity = (1.0 / 1.0 - 0.5) / 3.0
        t_star = (history[:, 0] - history[0, 0]) * kinematic_viscosity / diameter[0] ** 2
        curve = np.column_stack([
            history[:, 0],
            t_star,
            diameter,
            diameter_squared_normalized,
            history[:, 2],
            history[:, 3],
        ])

        csv_path = os.path.join(output_dir, f"d2_law_K{K:.3f}.csv")
        np.savetxt(csv_path, curve, delimiter=",", header="timestep,t_star,D,(D/D0)^2,centroid_x,centroid_y", comments="")
        print(f"K = {K:.4f}: saved {csv_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(4.5, 4.5))
    # Plot every curve present in the output directory (including previous runs)
    for csv_path in sorted(glob.glob(os.path.join(output_dir, "d2_law_K*.csv"))):
        curve = np.atleast_2d(np.loadtxt(csv_path, delimiter=",", skiprows=1))
        if curve.shape[0] < 2 or curve.shape[1] != 6 or np.unique(curve[:, 1]).size < 2:
            continue
        label = os.path.basename(csv_path).removeprefix("d2_law_K").removesuffix(".csv")
        fit_coefficients = np.polyfit(curve[:, 1], curve[:, 3], 1)
        linear_fit = np.polyval(fit_coefficients, curve[:, 1])
        residual = np.sum((curve[:, 3] - linear_fit) ** 2)
        total = np.sum((curve[:, 3] - curve[:, 3].mean()) ** 2)
        r_squared = 1.0 - residual / total if total > 0.0 else 1.0
        displacement = np.linalg.norm(curve[:, 4:6] - curve[0, 4:6], axis=1)
        axes.plot(curve[:, 1], curve[:, 3], label=rf"K = {label}, $R^2={r_squared:.4f}$")
    axes.set(xlabel=r"$t^* = t\nu/D_0^2$", ylabel=r"$(D/D_0)^2$", title=r"$D^2$ law")
    axes.legend()
    axes.grid(True, alpha=0.3)
    fig.suptitle("2D droplet evaporation (MRT, VanderWaals)")
    fig.savefig(os.path.join(output_dir, "d2_law_2d.png"), dpi=200, bbox_inches="tight")
    print(f"Saved {os.path.join(output_dir, 'd2_law_2d.png')}")
