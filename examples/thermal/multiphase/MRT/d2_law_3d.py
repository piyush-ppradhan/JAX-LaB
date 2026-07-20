"""Three-dimensional D^2-law droplet evaporation benchmark.

The setup follows Huang et al., "An efficient thermal lattice Boltzmann method for simulating three-dimensional liquid-vapor phase change", arXiv:2206.00946:
a radius-25 droplet at 0.86 Tc evaporates in a 100^3 domain whose far-field temperature is Tc. The flow uses a D3Q19 MRT model with a temperature-coupled
Peng-Robinson EOS, while temperature is advanced by the hybrid finite-difference RK4 solver.

The vapor density is adjusted across the initial thermal interface to keep the Peng-Robinson pressure equal to the saturation pressure. This avoids a non-physical
pressure impulse when temperature coupling is enabled. Full double precision is required to conserve mass at this density ratio. The default conductivity is increased
to K = 2 with automatic RK substepping so the benchmark completes faster than the published K = 1/3 reference case.
"""

import glob
import os
import sys

import numpy as np
from jax import config

from jax_lab.boundary_conditions import DirichletTemperature
from jax_lab.eos import PengRobinson
from jax_lab.lattice import LatticeD3Q19
from jax_lab.multiphase import MultiphaseMRT
from jax_lab.thermal import MultiphaseThermal
from jax_lab.utils import save_fields_hdf5_xdmf

# from jax_lab.utils import save_fields_vtk

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_d2_law_3d")

config.update("jax_enable_x64", True)


def peng_robinson_pressure(rho, T):
    """Evaluate the Peng-Robinson pressure.

    Parameters
    ----------
    rho (numpy.ndarray or float): Density in lattice units.
    T (numpy.ndarray or float): Temperature in lattice units.

    Returns
    -------
    numpy.ndarray or float: Pressure in lattice units.
    """
    alpha = (1.0 + (0.37464 + 1.54226 * pr_omega - 0.26992 * pr_omega**2) * (1.0 - np.sqrt(T / Tc))) ** 2
    denominator = 1.0 + 2.0 * b * rho - b**2 * rho**2
    return rho * R * T / (1.0 - b * rho) - a * alpha * rho**2 / denominator


def vapor_density_at_pressure(T, pressure):
    """Compute the stable vapor density at a prescribed pressure.

    Parameters
    ----------
    T (numpy.ndarray or float): Positive temperature in lattice units.
    pressure (float): Target Peng-Robinson pressure in lattice units.

    Returns
    -------
    numpy.ndarray: Low-density Peng-Robinson root at each temperature.
    """
    T = np.asarray(T, dtype=np.float64)
    rho = rho_g * T_liq / T
    for _ in range(12):
        alpha = (1.0 + (0.37464 + 1.54226 * pr_omega - 0.26992 * pr_omega**2) * (1.0 - np.sqrt(T / Tc))) ** 2
        denominator = 1.0 + 2.0 * b * rho - b**2 * rho**2
        denominator_derivative = 2.0 * b - 2.0 * b**2 * rho
        attraction_derivative = a * alpha * (2.0 * rho * denominator - rho**2 * denominator_derivative) / denominator**2
        pressure_derivative = R * T / (1.0 - b * rho) ** 2 - attraction_derivative
        rho = rho - (peng_robinson_pressure(rho, T) - pressure) / pressure_derivative
    return rho


class EvaporatingDroplet(MultiphaseMRT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_macroscopic_fields(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        z = np.linspace(0, self.nz - 1, self.nz, dtype=int)
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        center_x = 0.5 * (self.nx - 1)
        center_y = 0.5 * (self.ny - 1)
        center_z = 0.5 * (self.nz - 1)
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2)
        liquid_fraction = 0.5 - 0.5 * np.tanh(2 * (dist - r) / width)
        T = T_liq + (T_vap - T_liq) * (1.0 - liquid_fraction)
        rho_vapor = vapor_density_at_pressure(T, p_sat)
        rho = liquid_fraction * rho_l + (1.0 - liquid_fraction) * rho_vapor
        rho = rho.reshape((self.nx, self.ny, self.nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho_tree = [self.precision_policy.cast_to_output(rho)]

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype, init_val=u)
        u_tree = [self.precision_policy.cast_to_output(u)]
        return rho_tree, u_tree


class DropletThermal(MultiphaseThermal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.xdmf_prefix = kwargs.get("xdmf_prefix", "fields")
        # self.vtk_prefix = kwargs.get("vtk_prefix", "fields")
        self.minimum_diameter = kwargs.get("minimum_diameter", 0.0)
        self.measurement_start = kwargs.get("measurement_start", 0)
        self.D0 = None
        self.initial_mass = None
        self.history = []

    def initialize_temperature_field(self):
        x = np.linspace(0, self.nx - 1, self.nx, dtype=int)
        y = np.linspace(0, self.ny - 1, self.ny, dtype=int)
        z = np.linspace(0, self.nz - 1, self.nz, dtype=int)
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        center_x = 0.5 * (self.nx - 1)
        center_y = 0.5 * (self.ny - 1)
        center_z = 0.5 * (self.nz - 1)
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2)
        # Liquid at T_liq inside, superheated vapor at Tc outside (same smooth profile as density)
        T = 0.5 * (T_liq + T_vap) - 0.5 * (T_liq - T_vap) * np.tanh(2 * (dist - r) / width)
        return T.reshape((self.nx, self.ny, self.nz, 1))

    def set_thermal_boundary_conditions(self):
        faces = ("left", "right", "front", "back", "bottom", "top")
        bbox = self.fluid_solver.bounding_box_indices
        far_field = np.unique(np.concatenate([bbox[face] for face in faces]), axis=0)
        self.thermal_BCs = [DirichletTemperature(tuple(far_field.T), prescribed=T_vap)]

    def output_data(self, **kwargs):
        timestep = kwargs["timestep"]
        rho = np.array(kwargs["rho"][0][0, ..., 0])
        T = np.array(kwargs["T"][0, ..., 0])

        if not np.isfinite(rho).all() or not np.isfinite(T).all():
            print(f"Simulation diverged (non-finite density or temperature) at timestep {timestep}.")
            self.stop_simulation = True
            return

        mass = float(rho.sum(dtype=np.float64))
        if self.initial_mass is None:
            self.initial_mass = mass
        relative_mass_error = abs(mass - self.initial_mass) / self.initial_mass

        D, liquid_volume, liquid_fraction = droplet_diameter(rho)
        if timestep % self.io_rate == 0:
            save_fields_hdf5_xdmf(timestep, {"rho": rho, "T": T}, output_dir, prefix=self.xdmf_prefix)
            # save_fields_vtk(timestep, {"rho": rho, "T": T}, output_dir, prefix=self.vtk_prefix)
        if liquid_volume <= 1.0 or D <= self.minimum_diameter:
            print(f"Stopping at timestep {timestep} before the droplet enters the diffuse-interface shrinkage regime.")
            self.stop_simulation = True
            return
        # Always retain timestep zero as D0. Without this baseline, a run that
        # stops before the first post-equilibration I/O leaves history empty.
        if timestep < self.measurement_start and timestep != 0:
            return

        if self.D0 is None:
            self.D0 = D
        coordinates = np.indices(rho.shape)
        centroid = np.array([(liquid_fraction * coordinate).sum() / liquid_volume for coordinate in coordinates])
        self.history.append((timestep, D, centroid[0], centroid[1], centroid[2]))

        if timestep % self.io_rate == 0:
            print(
                f"timestep {timestep}: (D/D0)^2 = {(D / self.D0) ** 2:.4f}, "
                # f"rho = [{rho.min():.4f}, {rho.max():.4f}], T/Tc = [{T.min() / Tc:.4f}, {T.max() / Tc:.4f}], "
                # f"mass error = {relative_mass_error:.2e}, centroid = ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})"
            )


def droplet_diameter(rho):
    """
    Measure the subgrid droplet diameter from the phase-weighted volume.

    Parameters
    ----------
    rho (numpy.ndarray): Density field of shape (nx, ny, nz).

    Returns
    -------
    tuple (float, float, numpy.ndarray): Effective diameter, liquid volume and liquid fraction.

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
    return (6.0 * liquid_volume / np.pi) ** (1.0 / 3.0), liquid_volume, liquid_fraction


if __name__ == "__main__":
    precision = "f64/f64"
    nx = 100
    ny = 100
    nz = 100

    r = 25
    width = 5

    # Parameters used by Huang et al. for the 3D D^2-law benchmark.
    a = 3 / 49
    b = 2 / 21
    R = 1.0
    pr_omega = 0.344
    Tc = 0.1093785558
    T_liq = 0.86 * Tc
    T_vap = Tc
    specific_heat = 5.0

    # Maxwell construction densities at T = 0.86 Tc.
    rho_l = 6.499210784
    rho_g = 0.379598891
    p_sat = float(peng_robinson_pressure(rho_g, T_liq))
    # D3Q19 orthogonal moment basis (same as the isothermal MRT droplet_3d example)
    e = LatticeD3Q19().c.T
    en = np.linalg.norm(e, axis=1)

    M = np.zeros((19, 19))
    M[0, :] = en**0
    M[1, :] = 19 * en**2 - 30
    M[2, :] = (21 * en**4 - 53 * en**2 + 24) / 2
    M[3, :] = e[:, 0]
    M[4, :] = (5 * en**2 - 9) * e[:, 0]
    M[5, :] = e[:, 1]
    M[6, :] = (5 * en**2 - 9) * e[:, 1]
    M[7, :] = e[:, 2]
    M[8, :] = (5 * en**2 - 9) * e[:, 2]
    M[9, :] = 3 * e[:, 0] ** 2 - en**2
    M[10, :] = (3 * en**2 - 5) * (3 * e[:, 0] ** 2 - en**2)
    M[11, :] = e[:, 1] ** 2 - e[:, 2] ** 2
    M[12, :] = (3 * en**2 - 5) * (e[:, 1] ** 2 - e[:, 2] ** 2)
    M[13, :] = e[:, 0] * e[:, 1]
    M[14, :] = e[:, 1] * e[:, 2]
    M[15, :] = e[:, 0] * e[:, 2]
    M[16, :] = (e[:, 1] ** 2 - e[:, 2] ** 2) * e[:, 0]
    M[17, :] = (e[:, 2] ** 2 - e[:, 0] ** 2) * e[:, 1]
    M[18, :] = (e[:, 0] ** 2 - e[:, 1] ** 2) * e[:, 2]

    # Thermal conductivities can also be passed on the command line to run a subset
    K_VALUES = [1.8]
    io_rate = 1000
    measurement_start = 1000

    # Largest eigenvalue magnitude of the isotropic laplacian stencil (worst mode at k = (pi, pi, 0) for D3Q19, same 16/3 as D2Q9); the explicit RK4
    # update is stable for alpha * lambda_max * dt < 2.785 with alpha = K / (rho c_v), worst in the low density vapor phase.
    lambda_max = 16.0 / 3.0

    for K in K_VALUES:
        t_max = 24000
        os.makedirs(output_dir, exist_ok=True)
        stale_outputs = glob.glob(os.path.join(output_dir, f"fields_K{K:.3f}_*.hdf5"))
        stale_outputs.extend(glob.glob(os.path.join(output_dir, f"fields_K{K:.3f}_*.xdmf")))
        # stale_outputs = glob.glob(os.path.join(output_dir, f"fields_K{K:.3f}_*.vtk"))
        stale_outputs.append(os.path.join(output_dir, f"d2_law_K{K:.3f}.csv"))
        for stale_output in stale_outputs:
            if os.path.exists(stale_output):
                os.remove(stale_output)

        rho_vapor_far = float(vapor_density_at_pressure(T_vap, p_sat))
        rk_substeps = max(1, int(np.ceil((K / (rho_vapor_far * specific_heat)) * lambda_max / 2.5)))
        eos = PengRobinson(a=[a], b=[b], R=[R], pr_omega=[pr_omega], temperature_field_type="thermal")
        kwargs = {
            "n_components": 1,
            "lattice": LatticeD3Q19(precision),
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "g_kkprime": -1.0 * np.ones((1, 1)),
            "EOS": eos,
            "body_force": [0.0, 0.0, 0.0],
            "k": [0.27],
            "A": 0.01 * np.ones((1, 1)),
            "M": [M],
            "s_rho": [0.0],
            "s_e": [0.8],
            "s_eta": [1.0],
            "s_j": [0.0],
            "s_q": [1.0],
            "s_pi": [1.0],
            "s_m": [1.0],
            "s_v": [1.25],
            "kappa": [0.0],
            "precision": precision,
            "io_rate": io_rate,
            "print_info_rate": 1000,
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
            measurement_start=min(measurement_start, t_max // 2),
        )
        print(f"K = {K:.4f}: rk_substeps = {rk_substeps}")
        sim.run(t_max)

        history = np.asarray(sim.history, dtype=np.float64).reshape((-1, 5))
        if history.shape[0] == 0:
            print(f"K = {K:.4f}: no valid diameter samples; skipping curve output.")
            continue
        diameter = history[:, 1]
        diameter_squared_normalized = (diameter / diameter[0]) ** 2
        kinematic_viscosity = (1.0 / 1.25 - 0.5) / 3.0
        t_star = (history[:, 0] - history[0, 0]) * kinematic_viscosity / diameter[0] ** 2
        curve = np.column_stack([history[:, 0], t_star, diameter, diameter_squared_normalized, history[:, 2], history[:, 3], history[:, 4]])

        csv_path = os.path.join(output_dir, f"d2_law_K{K:.3f}.csv")
        np.savetxt(csv_path, curve, delimiter=",", header="timestep,t_star,D,(D/D0)^2,centroid_x,centroid_y,centroid_z", comments="")
        print(f"K = {K:.4f}: saved {csv_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(4.5, 4.5))
    # Plot every curve present in the output directory (including previous runs)
    for csv_path in sorted(glob.glob(os.path.join(output_dir, "d2_law_K*.csv"))):
        curve = np.atleast_2d(np.loadtxt(csv_path, delimiter=",", skiprows=1))
        if curve.shape[0] < 2 or curve.shape[1] != 7 or np.unique(curve[:, 1]).size < 2:
            continue
        label = os.path.basename(csv_path).removeprefix("d2_law_K").removesuffix(".csv")
        fit_curve = curve[curve[:, 0] >= measurement_start]
        if fit_curve.shape[0] < 2:
            continue
        fit_coefficients = np.polyfit(fit_curve[:, 1], fit_curve[:, 3], 1)
        linear_fit = np.polyval(fit_coefficients, fit_curve[:, 1])
        residual = np.sum((fit_curve[:, 3] - linear_fit) ** 2)
        total = np.sum((fit_curve[:, 3] - fit_curve[:, 3].mean()) ** 2)
        r_squared = 1.0 - residual / total if total > 0.0 else 1.0
        axes.plot(curve[:, 1], curve[:, 3], label=rf"K = {label}, $R^2={r_squared:.4f}$")
    axes.set(xlabel=r"$t^* = t\nu/D_0^2$", ylabel=r"$(D/D_0)^2$", title=r"$D^2$ law")
    axes.grid(True, alpha=0.3)
    if axes.lines:
        axes.legend(loc="best", frameon=True, fontsize="small")
    fig.suptitle("Finite-size 3D droplet evaporation (MRT, Peng-Robinson)")
    fig.savefig(os.path.join(output_dir, "d2_law_3d.png"), dpi=200, bbox_inches="tight")
    print(f"Saved {os.path.join(output_dir, 'd2_law_3d.png')}")
