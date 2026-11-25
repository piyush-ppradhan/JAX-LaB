"""Bounce-back correction of geometric wetting: exact wall populations for a planar interface."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_lab.core.boundary_conditions import BounceBack
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.shanchen.eos import PengRobinson

from test_wetting_denominator import _mrt_matrix

N = 64
WALL = 3  # solid rows at the bottom and the top: the two-cell force stencil from the first fluid row stays in the wall


# Liquid band LEFT < x - SHIFT - y < RIGHT (periodic), interfaces at 45 degrees to the walls. The bottom contact lines sit on
# the periodic seam x = 0 (characteristics traced across the domain face) and on the shard face x = N / 2 (cross-shard reads
# with 2 devices)
LEFT, RIGHT, SHIFT = 16, 48, 14


def _band_density(rho_liquid=6.499, rho_vapor=0.3796):  # Peng-Robinson coexistence at 0.86 T_c
    x, y = np.indices((N, N)).astype(np.float64)
    u = (x - SHIFT - y) % N
    distance = np.minimum(u - LEFT, RIGHT - u) / np.sqrt(2.0)
    return 0.5 * (rho_liquid + rho_vapor) + 0.5 * (rho_liquid - rho_vapor) * np.tanh(distance / 2.0)  # width 4


class _PeriodicBand:
    def initialize_macroscopic_fields(self):
        rho = _band_density()[..., None]
        return [self.distributed_array_init(rho.shape, self.precision_policy.compute_dtype, init_val=rho)], [
            self.distributed_array_init((N, N, 2), self.precision_policy.compute_dtype, init_val=0.0)
        ]


def _simulation(walls, wall_density=None):
    from jax_lab.core.shanchen.multiphase import MultiphaseMRT

    class Band(_PeriodicBand, MultiphaseMRT):
        def apply_contact_angle(self, rho_tree):
            # Optionally the exact (periodic) density at the wall nodes instead of the geometric one, which differs from it
            # slightly where the profile overshoots next to the interface: isolates the populations the wall sends
            if wall_density is None:
                return super().apply_contact_angle(rho_tree)
            rows = np.r_[0:WALL, N - WALL : N]
            return [rho_tree[0].at[:, rows].set(jnp.asarray(wall_density[:, rows, None], dtype=rho_tree[0].dtype))]

        def set_boundary_conditions(self):
            if not walls:
                return
            mask = np.zeros((N, N), dtype=bool)
            mask[:, :WALL] = mask[:, N - WALL :] = True
            nodes = np.array(np.nonzero(mask)).T
            x, y = nodes[:, 0].astype(float), nodes[:, 1].astype(float)
            bottom = y < WALL
            plane = np.where(bottom, WALL - 0.5, N - WALL - 0.5)
            # Contact lines of the band on each wall plane: the right interface (liquid on its left at the bottom) meets the
            # bottom wall at 135 degrees through the liquid and the top wall at 45, the left interface the other way round
            right, left = (RIGHT + SHIFT + plane) % N, (LEFT + SHIFT + plane) % N
            near_right = np.abs((x - right + N / 2) % N - N / 2) < np.abs((x - left + N / 2) % N - N / 2)
            theta = np.where(bottom == near_right, 3 * np.pi / 4, np.pi / 4)
            self.BCs[0].append(BounceBack(tuple(nodes.T), self.grid_info, self.precision_policy, theta))

    lattice = LatticeD2Q9("f32/f32")
    tau = 0.95  # viscosity 0.15; with s_q = 1.2 the magic parameter is 0.15, where bounce-back's defect is large
    kwargs = dict(
        lattice=lattice,
        nx=N,
        ny=N,
        nz=0,
        precision="f32/f32",
        io_rate=0,
        print_info_rate=0,
        checkpoint_rate=0,
        n_components=1,
        g_kkprime=-np.ones((1, 1)),
        lambda_=[1.0],
        kappa=[0.0],
        body_force=[0.0, 0.0],
        EOS=PengRobinson(a=3 / 49, b=2 / 21, pr_omega=0.344, R=1.0, T=0.86 * 0.1093785558),
        M=[_mrt_matrix(lattice)],
        s_rho=[0.0],
        s_j=[0.0],
        s_e=[1.0],
        s_eta=[1.0],
        s_q=[1.2],
        s_v=[1.0 / tau],
    )
    if walls:
        kwargs.update(wetting_formulation="geometric")
    return Band(**kwargs)


@pytest.fixture(scope="module")
def periodic_band():
    sim = _simulation(walls=False)
    state = sim.advance(sim.state_from_populations(sim.assign_fields_sharded()), 0, 2000)
    fields = np.stack([np.asarray(f) for f in state[0].fields], axis=-1)  # post-collision populations
    return np.asarray(sim.poststreaming(state)[0]), fields


def _one_step(f_periodic):
    sim = _simulation(walls=True, wall_density=f_periodic.sum(axis=-1))
    state = sim.step(sim.state_from_populations([f_periodic]), 0)[0]
    return sim, np.stack([np.asarray(f) for f in state[0].fields], axis=-1)


def _reference_correction(sim, f, g_previous, g):
    """NumPy reference of the corrected wall populations (see Multiphase._create_geometric_wall_correction_data): bounce-back
    plus D_i = G_i(s) - H_i(s + c_i) at the nearest node s of the selected characteristic sample."""
    c = np.asarray(sim.lattice.c).T
    w = np.asarray(sim.lattice.w)
    opposite = np.asarray(sim.lattice.opp_indices)
    rho = f.sum(axis=-1)
    j = 0.5 * (np.einsum("xyq,qd->xyd", f, c) + np.einsum("xyq,qd->xyd", g, c))  # physical momentum
    solid = np.zeros((N, N), dtype=bool)
    solid[:, :WALL] = solid[:, N - WALL :] = True
    expected = {}
    for walls, theta, points in sim._geometric_wall_samples[0]:
        samples = np.stack([sim._interpolation_points(point) for point in points], axis=1)  # (n, K, 2)
        for b, th, sample_points in zip(walls, theta, samples, strict=True):
            nodes = np.mod(np.round(sample_points).astype(int), N)
            along = lambda t: np.mod(np.round(b + t * (sample_points - b)).astype(int), N)
            clear = np.all([~solid[along(t)[:, 0], along(t)[:, 1]] for t in (1.25, 1.5, 1.75, 2.0)], axis=0)
            far = np.where(clear[:, None], along(2.0), nodes)  # the far node only along a clear path
            score = rho[nodes[:, 0], nodes[:, 1]] + rho[far[:, 0], far[:, 1]]
            if np.ptp(rho[nodes[:, 0], nodes[:, 1]]) <= sim._GEOMETRIC_WETTING_ACTIVITY * np.ptp(rho[~solid]):
                continue  # uniform phase: plain bounce-back
            s = nodes[np.argmax(score) if th <= np.pi / 2 else np.argmin(score)]
            for i in range(sim.q):
                x = np.mod(b + c[i], N)
                if not c[i].any() or solid[x[0], x[1]]:
                    continue
                t = np.mod(s + c[i], N)
                o = opposite[i]
                defect = g[s[0], s[1], i] - 3 * w[i] * c[i] @ j[s[0], s[1]] - g[t[0], t[1], o] - 3 * w[i] * c[i] @ j[t[0], t[1]]
                expected[(b[0], b[1], i)] = g_previous[x[0], x[1], o] + defect
    return expected


def test_correction_matches_reference(periodic_band):
    f_periodic, g_periodic = periodic_band
    sim, g = _one_step(f_periodic)
    expected = _reference_correction(sim, f_periodic, g_periodic, g)
    assert len(expected) > 0
    error = max(abs(g[x, y, i] - value) for (x, y, i), value in expected.items())
    assert error < 1e-5
    # Everywhere else on the walls: plain bounce-back (the populations the fluid sent, reflected)
    c = np.asarray(sim.lattice.c).T
    opposite = np.asarray(sim.lattice.opp_indices)
    for y, sign in ((WALL - 1, 1), (N - WALL, -1)):
        for i in range(sim.q):
            if c[i, 1] != sign:
                continue
            plain = [x for x in range(N) if (x, y, i) not in expected]
            bounced = g_periodic[(np.array(plain) + c[i, 0]) % N, y + sign, opposite[i]]
            np.testing.assert_allclose(g[plain, y, i], bounced, atol=1e-6)


def test_correction_restores_the_populations_of_a_planar_contact_line(periodic_band):
    """At the contact lines of the planar band the populations the wall sends into the fluid must be those of the periodic
    band. Bounce-back misses them by a kinetic error; with the correction only the momentum of the band's slow drift (which
    the no-slip correction removes) is left."""
    f_periodic, g_periodic = periodic_band
    sim, g = _one_step(f_periodic)
    c = np.asarray(sim.lattice.c).T
    opposite = np.asarray(sim.lattice.opp_indices)
    x = np.arange(N)
    errors = {"bounceback": 0.0, "corrected": 0.0}
    for y, sign in ((WALL - 1, 1), (N - WALL, -1)):
        for i in range(sim.q):
            if c[i, 1] != sign:
                continue
            exact = g_periodic[:, y, i]
            # Plain bounce-back: the population the fluid node sent, reflected
            bounced = g_periodic[(x + c[i, 0]) % N, y + sign, opposite[i]]
            errors["bounceback"] = max(errors["bounceback"], np.abs(bounced - exact).max())
            errors["corrected"] = max(errors["corrected"], np.abs(g[:, y, i] - exact).max())
    bounceback, corrected = errors["bounceback"], errors["corrected"]
    assert bounceback > 5e-3
    assert corrected < bounceback / 5


def test_correction_uses_the_samples_geometric_wetting_chose(periodic_band):
    """The correction takes the characteristic samples geometric wetting chose (mapped from its shard-local rows); choosing
    them itself must give the same populations."""
    f_periodic, _ = periodic_band

    def run(own_choice):
        sim = _simulation(walls=True)
        if own_choice:
            # Hide the wetting's choices from the step: the correction falls back to choosing the samples itself
            original = type(sim).apply_contact_angle

            def apply_contact_angle(self, rho_tree):
                export, self._wetting_choice_export = self._wetting_choice_export, None
                try:
                    return original(self, rho_tree)
                finally:
                    self._wetting_choice_export = export

            sim.apply_contact_angle = apply_contact_angle.__get__(sim)
        state = sim.state_from_populations([f_periodic])
        for timestep in range(4):
            state = sim.step(state, timestep)[0]
        return np.stack([np.asarray(f) for f in state[0].fields], axis=-1)

    np.testing.assert_allclose(run(own_choice=False), run(own_choice=True), atol=1e-6)


def test_step_keeps_the_compute_precision_in_64_bit_mode():
    """With 64-bit mode enabled and f32/f32 precision, no operation of the time step (geometric wetting and its bounce-back
    correction included) may compute in float64: float64 constants (numpy scalars) would promote it. Traced in a subprocess,
    since 64-bit mode is global."""
    import os
    import subprocess
    import sys

    script = (
        "import re, jax, test_geometric_wall_correction as t\n"
        "sim = t._simulation(walls=True)\n"
        "state = sim.state_from_populations(sim.assign_fields_sharded())\n"
        "text = str(jax.make_jaxpr(lambda s: sim.step(s, 0)[0])(state))\n"
        "print(sum(bool(re.search(r'^\\s*\\w+:f64\\[\\d', line)) for line in text.splitlines()))\n"
    )
    # On the CPU: tracing needs no GPU, and the test process already holds the GPU memory
    environment = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(__file__),
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    assert result.stdout.strip().splitlines()[-1] == "0"
