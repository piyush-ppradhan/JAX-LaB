"""Verify fused multiphase MRT collision against an independent unfused reference.

Uses nonzero surface tension and body force because Taylor-Green tests do not exercise those paths.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_lab.core.eos import VanderWaals
from jax_lab.core.lattice import LatticeD3Q19
from jax_lab.core.multiphase import MultiphaseMRT

DOMAIN = (10, 10, 10)
PRECISION = "f64/f64"
SEED = 0


def _mrt_matrix(lattice):
    e = np.asarray(lattice.c).T
    en = np.linalg.norm(e, axis=1)
    x, y, z = e.T
    matrix = np.zeros((19, 19))
    matrix[0, :] = en**0
    matrix[1, :] = 19 * en**2 - 30
    matrix[2, :] = (21 * en**4 - 53 * en**2 + 24) / 2
    matrix[3, :] = x
    matrix[4, :] = (5 * en**2 - 9) * x
    matrix[5, :] = y
    matrix[6, :] = (5 * en**2 - 9) * y
    matrix[7, :] = z
    matrix[8, :] = (5 * en**2 - 9) * z
    matrix[9, :] = 3 * x**2 - en**2
    matrix[10, :] = (3 * en**2 - 5) * (3 * x**2 - en**2)
    matrix[11, :] = y**2 - z**2
    matrix[12, :] = (3 * en**2 - 5) * (y**2 - z**2)
    matrix[13, :] = x * y
    matrix[14, :] = y * z
    matrix[15, :] = x * z
    matrix[16, :] = (y**2 - z**2) * x
    matrix[17, :] = (z**2 - x**2) * y
    matrix[18, :] = (x**2 - y**2) * z
    return matrix


def _build_sim(kappa):
    nx, ny, nz = DOMAIN
    lattice = LatticeD3Q19(PRECISION)
    s = [0.8]
    kwargs = {
        "lattice": lattice,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "checkpoint_rate": 0,
        "n_components": 1,
        "omega": [1.0],
        "g_kkprime": -np.ones((1, 1)),
        "k": [1.0],
        "A": np.zeros((1, 1)),
        "body_force": [1e-5, -2e-5, 0.0],
        "EOS": VanderWaals(a=[9.0 / 49.0], b=[2.0 / 21.0], R=[1.0], T=0.8 * 0.5714285714),
        "M": [_mrt_matrix(lattice)],
        "s_rho": [1.0],
        "s_e": s,
        "s_eta": s,
        "s_j": [1.0],
        "s_q": s,
        "s_v": s,
        "s_pi": s,
        "s_m": [1.0],
        "kappa": [kappa],
    }
    return MultiphaseMRT(**kwargs)


def _random_fin(sim):
    # Small perturbation: VanderWaals below the critical temperature has a spinodal region where
    # p - cs2*rho goes negative (making the pseudopotential sqrt undefined) - a large random per-channel
    # perturbation can push the summed density into it. Stay close to the uniform rho=1 initial state, matching
    # how the Taylor-Green regression test in test_collision.py only perturbs density by a tiny amount.
    fin = sim.assign_fields_sharded()
    rng = np.random.default_rng(SEED)
    return [f + jnp.asarray(rng.uniform(-0.005, 0.005, size=f.shape)) for f in fin]


def _reference_collision(sim, fin_tree, T=None):
    """Independent unfused reference: m = f @ M; mout = m - (m - meq) @ S + delta_meq + C; fout = mout @ M_inv."""
    fin_tree = [sim.precision_policy.cast_to_compute(f) for f in fin_tree]
    rho_tree, u_tree = sim.update_macroscopic(fin_tree)
    feq_tree = sim.equilibrium(rho_tree, u_tree, cast_output=False)
    delta_feq_tree = sim._compute_force_delta_feq(rho_tree, u_tree, T=T)
    psi_tree, _ = sim.compute_potential(rho_tree, T=T)
    C_tree = sim.adjust_surface_tension(psi_tree)

    fout_tree = []
    for f, feq, delta_feq, C, M, S, M_inv in zip(fin_tree, feq_tree, delta_feq_tree, C_tree, sim.M, sim.S, sim.M_inv, strict=True):
        m = jnp.dot(f, M)
        meq = jnp.dot(feq, M)
        delta_meq = jnp.dot(delta_feq, M)
        mout = m - jnp.dot(m - meq, S) + delta_meq
        fout_tree.append(jnp.dot(mout + C, M_inv))
    return [sim.precision_policy.cast_to_output(fout) for fout in fout_tree]


def _reference_surface_tension(sim, psi_tree):
    """Original q-channel streamed formulation, retained only as an independent regression reference."""
    psi_s_tree = [sim.streaming(jnp.repeat(psi, axis=-1, repeats=sim.q)) for psi in psi_tree]
    c = jnp.transpose(sim.c)
    output = []
    for kappa, A, s_v, s_e, psi, psi_s in zip(
        sim.kappa,
        sim.A.diagonal(),
        sim.s_v,
        sim.s_e,
        psi_tree,
        psi_s_tree,
        strict=True,
    ):
        tm1 = lambda i, j: psi[..., 0] * jnp.dot(sim.G_ff * (psi_s - psi), c[:, i] * c[:, j])
        tm2 = lambda i, j: jnp.dot(sim.G_ff * (psi_s**2 - psi**2), c[:, i] * c[:, j])
        qxx = -kappa * ((1.0 - A) * tm1(0, 0) + 0.5 * A * tm2(0, 0))
        qxy = -kappa * ((1.0 - A) * tm1(0, 1) + 0.5 * A * tm2(0, 1))
        qyy = -kappa * ((1.0 - A) * tm1(1, 1) + 0.5 * A * tm2(1, 1))
        C = jnp.zeros_like(psi_s, dtype=sim.precision_policy.compute_dtype)
        if sim.dim == 2:
            C = C.at[..., 1].set(1.5 * s_e * (qxx + qyy))
            C = C.at[..., 2].set(-1.5 * sim.s_eta[len(output)] * (qxx + qyy))
            C = C.at[..., 7].set(-s_v * (qxx - qyy))
            C = C.at[..., 8].set(-s_v * qxy)
        else:
            qxz = -kappa * ((1.0 - A) * tm1(0, 2) + 0.5 * A * tm2(0, 2))
            qyz = -kappa * ((1.0 - A) * tm1(1, 2) + 0.5 * A * tm2(1, 2))
            qzz = -kappa * ((1.0 - A) * tm1(2, 2) + 0.5 * A * tm2(2, 2))
            C = C.at[..., 1].set((2.0 / 5.0) * s_e * (qxx + qyy + qzz))
            C = C.at[..., 9].set(-s_v * (2.0 * qxx - qyy - qzz))
            C = C.at[..., 11].set(-s_v * (qyy - qzz))
            C = C.at[..., 13].set(-s_v * qxy)
            C = C.at[..., 14].set(-s_v * qyz)
            C = C.at[..., 15].set(-s_v * qxz)
        output.append(C)
    return output


def _check(kappa):
    sim = _build_sim(kappa)
    fin_tree = _random_fin(sim)

    actual = sim.collision(fin_tree)
    expected = _reference_collision(sim, fin_tree)

    for a, e in zip(actual, expected, strict=True):
        assert np.max(np.abs(np.asarray(a) - np.asarray(e))) < 1e-9


def test_fused_symbolic_collision_matches_reference_without_surface_tension():
    sim = _build_sim(kappa=0.0)
    assert sim.scalar_surface_stencil is None
    _check(kappa=0.0)


def test_fused_symbolic_collision_matches_reference_with_surface_tension():
    _check(kappa=0.05)


def test_scalar_surface_stencil_matches_q_channel_reference():
    sim = _build_sim(kappa=0.05)
    rng = np.random.default_rng(SEED + 2)
    psi_tree = [jnp.asarray(rng.uniform(0.1, 1.0, size=(sim.nx, sim.ny, sim.nz, 1)), dtype=jnp.float64)]
    actual = sim.adjust_surface_tension(psi_tree)
    expected = _reference_surface_tension(sim, psi_tree)
    assert np.max(np.abs(np.asarray(actual[0]) - np.asarray(expected[0]))) < 1e-12
