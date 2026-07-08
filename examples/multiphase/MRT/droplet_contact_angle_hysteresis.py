"""
Spherical droplet impinging and sliding on an inclined wall with geometric wetting.

The collision matrix is based on:
1. Coveney, P. V. et al. Multiple-relaxation-time lattice Boltzmann models in three dimensions.
Philosophical Transactions of the Royal Society of London.
Series A: Mathematical, Physical and Engineering Sciences 360, 437-451 (2002).
"""

import csv
import os
from pathlib import Path

import numpy as np

from src.boundary_conditions import BounceBack
from src.eos import Peng_Robinson
from src.lattice import LatticeD3Q19
from src.multiphase import MultiphaseMRT
from src.utils import save_fields_hdf5_xdmf


class DropletContactAngleHysteresis3D(MultiphaseMRT):
    def initialize_macroscopic_fields(self):
        dist = np.sqrt((x - droplet_center[0]) ** 2 + (y - droplet_center[1]) ** 2 + (z - droplet_center[2]) ** 2)
        rho = 0.5 * (rho_l + rho_g) - 0.5 * (rho_l - rho_g) * np.tanh(2 * (dist - droplet_radius) / interface_width)
        rho = rho.reshape((nx, ny, nz, 1))
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precisionPolicy.compute_dtype, init_val=rho)
        rho = self.precisionPolicy.cast_to_output(rho)
        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precisionPolicy.compute_dtype, init_val=u)
        u = self.precisionPolicy.cast_to_output(u)
        return [rho], [u]

    def set_boundary_conditions(self):
        self.BCs[0].append(BounceBack(wall_indices, self.gridInfo, self.precisionPolicy, theta[wall_indices]))

    def output_data(self, **kwargs):
        rho = np.array(kwargs["rho_tree"][0][0, ...])
        u = np.array(kwargs["u_tree"][0][0, ...])
        timestep = kwargs["timestep"]
        rho_scalar = rho[..., 0]
        # fields = {"rho": rho_scalar, "ux": u[..., 0], "uy": u[..., 1], "uz": u[..., 2]}
        fields = {"rho": rho_scalar}
        mask = np.array(self.solid_mask_streamed[0][..., 0])
        save_fields_hdf5_xdmf(timestep, fields, output_dir, "data", static_fields={"mask": mask})

        angles = measure_leading_lagging_contact_angles(rho_scalar)
        if angles is not None:
            contact_angle_writer.writerow([timestep, *angles])
            contact_angle_file.flush()


def measure_leading_lagging_contact_angles(rho_scalar):
    liquid = rho_scalar >= interface_level
    near_wall = (plane_distance > 0.0) & (plane_distance <= contact_search_distance)
    contact_points = np.argwhere(liquid & near_wall & interior_mask)
    if contact_points.shape[0] < min_contact_points:
        return None

    tangent_coordinate = coordinates_tangent[tuple(contact_points.T)]
    leading_cutoff = np.percentile(tangent_coordinate, 90.0)
    lagging_cutoff = np.percentile(tangent_coordinate, 10.0)
    leading_points = contact_points[tangent_coordinate >= leading_cutoff]
    lagging_points = contact_points[tangent_coordinate <= lagging_cutoff]
    if leading_points.shape[0] < min_fit_points or lagging_points.shape[0] < min_fit_points:
        return None

    leading_angle = fit_local_contact_angle(rho_scalar, leading_points)
    lagging_angle = fit_local_contact_angle(rho_scalar, lagging_points)
    if leading_angle is None or lagging_angle is None:
        return None
    return leading_angle, lagging_angle


def fit_local_contact_angle(rho_scalar, points):
    center = np.mean(points, axis=0)
    window = (
        (np.abs(x - center[0]) <= contact_fit_radius)
        & (np.abs(y - center[1]) <= contact_fit_radius)
        & (np.abs(z - center[2]) <= contact_fit_radius)
        & (plane_distance > 0.0)
        & (plane_distance <= contact_fit_distance)
        & interior_mask
    )
    liquid = rho_scalar >= interface_level
    interface_band = np.abs(rho_scalar - interface_level) <= interface_band_width
    samples = np.argwhere(window & interface_band)
    if samples.shape[0] < min_fit_points:
        samples = np.argwhere(window & liquid)
    if samples.shape[0] < min_fit_points:
        return None

    sample_coordinates = np.column_stack((
        coordinates_tangent[tuple(samples.T)],
        plane_distance[tuple(samples.T)],
    ))
    circle = fit_circle(sample_coordinates)
    if circle is None:
        return None
    center_2d, radius = circle
    return float(np.degrees(np.arccos(np.clip(-center_2d[1] / radius, -1.0, 1.0))))


def fit_circle(points):
    if points.shape[0] < 3:
        return None
    points = np.asarray(points, dtype=np.float64)
    A = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(points.shape[0])))
    b = np.sum(points * points, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    radius_sq = sol[2] + np.dot(sol[:2], sol[:2])
    if radius_sq <= 0.0:
        return None
    return sol[:2], np.sqrt(radius_sq)


if __name__ == "__main__":
    precision = "f32/f32"

    nx = 350
    ny = 300
    nz = 300
    buffer_layers = 6
    wall_angle = 60.0 * np.pi / 180.0
    wall_thickness = 4.0
    droplet_radius = 20.0
    interface_width = 4.0
    output_dir = "output_hysteresis"

    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    x, y, z = np.meshgrid(x, y, z, indexing="ij")
    interior_mask = (
        (x >= buffer_layers)
        & (x < nx - buffer_layers)
        & (y >= buffer_layers)
        & (y < ny - buffer_layers)
        & (z >= buffer_layers)
        & (z < nz - buffer_layers)
    )

    wall_normal = np.array([np.cos(wall_angle), 0.0, np.sin(wall_angle)], dtype=np.float64)
    downslope_tangent = np.array([np.sin(wall_angle), 0.0, -np.cos(wall_angle)], dtype=np.float64)
    wall_point = np.array([115.0, ny / 2, 155.0], dtype=np.float64)
    plane_distance = wall_normal[0] * (x - wall_point[0]) + wall_normal[2] * (z - wall_point[2])
    coordinates_tangent = downslope_tangent[0] * (x - wall_point[0]) + downslope_tangent[2] * (z - wall_point[2])
    wall = (plane_distance <= 0.0) & (plane_distance > -wall_thickness) & interior_mask
    wall_indices = tuple(np.where(wall))

    droplet_gap = 14.0
    droplet_center = wall_point + (droplet_radius + droplet_gap) * wall_normal - 70.0 * downslope_tangent

    visc = 0.15
    tau = 3 * visc + 0.5

    a = 3 / 49
    b = 2 / 21
    Tc = 0.1093785558
    T = 0.86 * Tc
    eos = Peng_Robinson(a=a, b=b, pr_omega=0.344, R=1.0, T=T)
    rho_g = 0.379598891
    rho_l = 6.499210784
    interface_level = 0.5 * (rho_l + rho_g)
    interface_band_width = 0.05 * (rho_l - rho_g)

    contact_search_distance = 6.0
    contact_fit_distance = 20.0
    contact_fit_radius = 24.0
    min_contact_points = 80
    min_fit_points = 30

    g_kkprime = -1 * np.ones((1, 1))

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

    s_rho = [0.0]
    s_e = [1.0]
    s_eta = [1.0]
    s_j = [0.0]
    s_q = [1.0]
    s_v = [1 / tau]
    s_m = [1.0]
    s_pi = [1.0]

    theta = 25.0 * np.pi / 180.0 * np.ones((nx, ny, nz, 1))

    os.system(f"rm -rf {output_dir}/")
    Path(output_dir).mkdir(exist_ok=True)
    contact_angle_path = Path(output_dir) / "contact_angles.csv"
    contact_angle_file = contact_angle_path.open("w", newline="", encoding="utf-8")
    contact_angle_writer = csv.writer(contact_angle_file)
    contact_angle_writer.writerow(["timestep", "leading_angle_deg", "lagging_angle_deg"])

    kwargs = {
        "lattice": LatticeD3Q19(precision),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_components": 1,
        "body_force": [0.0, 0.0, -2.5e-5],
        "g_kkprime": g_kkprime,
        "M": [M],
        "s_rho": s_rho,
        "s_e": s_e,
        "s_eta": s_eta,
        "s_j": s_j,
        "s_q": s_q,
        "s_pi": s_pi,
        "s_m": s_m,
        "s_v": s_v,
        "EOS": eos,
        "kappa": [0],
        "k": [1.0],
        "A": 0.0 * np.zeros((1, 1)),
        "precision": precision,
        "io_rate": 1000,
        "print_info_rate": 1000,
        "checkpoint_rate": -1,
        "checkpoint_dir": os.path.abspath("./checkpoints_"),
        "restore_checkpoint": False,
        "wetting_formulation": "geometric",
    }
    try:
        sim = DropletContactAngleHysteresis3D(**kwargs)
        sim.run(35000)
    finally:
        contact_angle_file.close()
