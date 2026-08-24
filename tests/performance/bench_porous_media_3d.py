"""A/B performance benchmark for the optimizations in performance_optimization.md.

Reproduces the physics setup of porous_media_evaporation_3D_mrt_low_memory.py (single-component MultiphaseMRT,
same collision matrix, EOS, relaxation values and wetting parameters) on a smaller, reproducible 64^3 randomly
generated porous medium, so the same script can be re-run before/after each optimization lands and compared.

Run directly for the full A/B numbers (requires a GPU):
    python tests/performance/bench_porous_media_3d.py [--nx 64] [--steps 200] [--devices <n>]

Collected by pytest as a fast GPU-gated smoke test (skipped automatically without a GPU).
"""

import os

# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
# os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer=").strip()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_lab.core.boundary_conditions import BounceBack, ExactNonEquilibriumExtrapolation  # noqa: E402
from jax_lab.core.eos import PengRobinson  # noqa: E402
from jax_lab.core.lattice import LatticeD3Q19  # noqa: E402
from jax_lab.core.multiphase import MultiphaseMRT  # noqa: E402

RHO_L = 6.499210784
RHO_G = 0.379598891
S2 = 0.8
THETA_W = np.pi / 3
PHI_W = 1.15
DELTA_RHO_W = 0.0
PRECISION = "f32/f32"


def require_gpu():
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU test required, found: {devices}")


def _mrt_matrix_d3q19():
    e = np.asarray(LatticeD3Q19().c).T
    ex, ey, ez = e[:, 0], e[:, 1], e[:, 2]
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
    return M


def _random_porous_mask(nx, ny, nz, seed, solid_fraction=0.35):
    """Reproducible random porous medium: independent per-voxel noise, blurred and thresholded so solid
    forms connected blobs rather than isolated single voxels."""
    rng = np.random.default_rng(seed)
    noise = rng.random((nx, ny, nz))
    # Cheap separable box blur (3 passes) instead of a scipy dependency, to correlate neighboring voxels.
    for axis in range(3):
        noise = (noise + np.roll(noise, 1, axis=axis) + np.roll(noise, -1, axis=axis)) / 3.0
    threshold = np.quantile(noise, solid_fraction)
    mask = noise < threshold
    mask[:, :, :2] = False  # keep a clear inlet/outlet slab at each z end
    mask[:, :, -2:] = False
    return mask


class PorousMediaBenchmark(MultiphaseMRT):
    def __init__(self, mask, **kwargs):
        self._mask = mask
        super().__init__(**kwargs)

    def initialize_macroscopic_fields(self):
        z = np.arange(self.nz).reshape(1, 1, self.nz)
        L = self.nz // 3
        width = 3
        rho = 0.5 * (RHO_L + RHO_G) - 0.5 * (RHO_L - RHO_G) * np.tanh(2 * (z - L) / width)
        rho = np.broadcast_to(rho, (self.nx, self.ny, self.nz)).copy().reshape(self.nx, self.ny, self.nz, 1)
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
        return [rho], [u]

    def set_boundary_conditions(self):
        bottom = self.bounding_box_indices["bottom"]
        self.BCs[0].append(BounceBack(tuple(bottom.T), self.grid_info, self.precision_policy))

        top = self.bounding_box_indices["top"]
        # 0.75*RHO_G (a stronger evaporation-like forcing, matching the reference script) goes unstable by
        # ~step 30-40 on this synthetic random geometry; 0.93*RHO_G stays finite past 100 steps at every size
        # tested (64^3-192^3), so it's used here purely to give this benchmark a stable multi-hundred-step
        # window - not a claim about what's stable for a real porous-media study.
        prescribed_rho = 0.93 * RHO_G * jnp.ones((top.shape[0], 1))
        self.BCs[0].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.grid_info, self.precision_policy, prescribed_rho, "density"))

        porous_indices = np.array(np.where(self._mask)).T
        self.BCs[0].append(BounceBack(tuple(porous_indices.T), self.grid_info, self.precision_policy, THETA_W, PHI_W, DELTA_RHO_W))


def build_simulation(nx, ny, nz, seed):
    n_devices = jax.device_count()
    if nx % n_devices:
        nx += n_devices - nx % n_devices
    mask = _random_porous_mask(nx, ny, nz, seed)

    s_0, s_1, s_b, s_2, s_4 = [1.0], [1.0], [0.8], [S2], [1.0]
    s_3 = [(16 - 8 * S2) / (8 - S2)]
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(PRECISION),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": -np.ones((1, 1)),
        "k": [1.0],
        "A": np.zeros((1, 1)),
        "EOS": PengRobinson(a=[3 / 49], b=[2 / 21], pr_omega=[0.344], R=[1.0], T=0.86 * 0.1093785558),
        "M": [_mrt_matrix_d3q19()],
        "s_0": s_0,
        "s_1": s_1,
        "s_b": s_b,
        "s_2": s_2,
        "s_3": s_3,
        "s_4": s_4,
        "kappa": [0.0],
        "wetting_formulation": "improved_virtual_density",
        "s_rho": s_0,
        "s_e": s_b,
        "s_eta": s_b,
        "s_j": s_1,
        "s_q": s_3,
        "s_v": s_2,
        "s_pi": s_3,
        "s_m": s_4,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "compute_MLUPS": True,
        "checkpoint_rate": 0,
    }
    return PorousMediaBenchmark(mask, **kwargs)


def run_benchmark(nx=64, ny=64, nz=64, steps=200, seed=0, sample_memory=False):
    """
    sample_memory (bool): When True, block and read device.memory_stats()['bytes_in_use'] after every step
    instead of only before/after the loop, to see the live allocator footprint fluctuate step to step (the
    cudamallocasync pool grows/shrinks as per-step temporaries are allocated and freed). This forces a device
    sync every step, so elapsed/MLUPS from a sample_memory=True run is conservative relative to a normal run -
    it is a memory-fluctuation diagnostic, not the throughput number.
    """
    require_gpu()
    sim = build_simulation(nx, ny, nz, seed)

    f_tree = sim.assign_fields_sharded()
    f_tree, _ = sim.step(f_tree, 0)  # compile, excluded from timing
    jax.block_until_ready(f_tree)

    device = jax.local_devices()[0]
    memory_before = device.memory_stats() or {}
    memory_trace = []

    start = time.time()
    for timestep in range(1, steps + 1):
        f_tree, _ = sim.step(f_tree, timestep)
        if sample_memory:
            jax.block_until_ready(f_tree)
            memory_trace.append((device.memory_stats() or {}).get("bytes_in_use"))
    jax.block_until_ready(f_tree)
    elapsed = time.time() - start

    memory_after = device.memory_stats() or {}
    rho_tree, _ = sim.update_macroscopic(f_tree)
    force_tree = sim.compute_force(rho_tree)

    rho = np.asarray(rho_tree[0])
    voxels = sim.nx * sim.ny * sim.nz
    result = {
        "nx": sim.nx,
        "ny": sim.ny,
        "nz": sim.nz,
        "n_devices": jax.device_count(),
        "steps": steps,
        "elapsed_seconds": elapsed,
        "mlups": voxels * steps / elapsed / 1e6,
        "bytes_in_use": memory_after.get("bytes_in_use"),
        "peak_bytes_in_use": memory_after.get("peak_bytes_in_use", memory_before.get("peak_bytes_in_use")),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "rho_total_mass": float(rho.sum()),
        "rho_finite": bool(np.isfinite(rho).all()),
        "population": np.asarray(f_tree[0]),
        "force": np.asarray(force_tree[0]),
    }
    if sample_memory:
        result["bytes_in_use_min"] = min(memory_trace)
        result["bytes_in_use_max"] = max(memory_trace)
        result["bytes_in_use_mean"] = sum(memory_trace) / len(memory_trace)
        result["bytes_in_use_trace"] = memory_trace
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="bench_result")
    parser.add_argument("--sample-memory", action="store_true", help="Trace bytes_in_use every step (see run_benchmark)")
    args = parser.parse_args()

    result = run_benchmark(args.nx, args.ny, args.nz, args.steps, args.seed, sample_memory=args.sample_memory)
    population, force = result.pop("population"), result.pop("force")
    trace = result.pop("bytes_in_use_trace", None)

    print(json.dumps(result, indent=2))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    np.save(out_dir / "population.npy", population)
    np.save(out_dir / "force.npy", force)
    if trace is not None:
        np.save(out_dir / "bytes_in_use_trace.npy", np.asarray(trace))
    print(f"Saved arrays and summary to {out_dir}/")


@pytest.mark.skipif(
    not jax.devices() or any(device.platform != "gpu" for device in jax.devices()),
    reason="GPU required for the porous-media benchmark",
)
def test_benchmark_smoke():
    """Fast smoke run (small domain, few steps) so this file stays pytest-collectible without taking minutes.
    Use `python bench_porous_media_3d.py` directly for the real A/B numbers."""
    result = run_benchmark(nx=16, ny=16, nz=16, steps=3, seed=0)
    assert result["rho_finite"]
    assert result["mlups"] > 0


if __name__ == "__main__":
    main()
