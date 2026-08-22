"""Verify the fused/symbolic MultiphaseMRT.collision() (K = M @ S @ M_inv, sparse per-direction columns) against
an independent reference built from the unfused three-matmul formula (m = f @ M; relax; + delta_meq + C;
@ M_inv), and the dense (difference @ K) form against the symbolic (per-column sum) form directly. Uses nonzero
kappa (surface tension) and a nonzero body force, since the Taylor-Green regression test in test_collision.py
always runs with both zero and would not exercise either path.
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


def _check(kappa):
    sim = _build_sim(kappa)
    fin_tree = _random_fin(sim)

    actual = sim.collision(fin_tree)
    expected = _reference_collision(sim, fin_tree)

    for a, e in zip(actual, expected, strict=True):
        assert np.max(np.abs(np.asarray(a) - np.asarray(e))) < 1e-9


def test_fused_symbolic_collision_matches_reference_without_surface_tension():
    _check(kappa=0.0)


def test_fused_symbolic_collision_matches_reference_with_surface_tension():
    _check(kappa=0.05)


def test_dense_and_symbolic_collision_matrix_agree():
    """Guide's explicit acceptance check: dense (difference @ K) and symbolic (per-column sum of nonzero
    terms) forms of the fused collision matrix must agree, with matching finite/NaN masks."""
    sim = _build_sim(kappa=0.0)
    fin_tree = _random_fin(sim)
    fin_tree = [sim.precision_policy.cast_to_compute(f) for f in fin_tree]
    rho_tree, u_tree = sim.update_macroscopic(fin_tree)
    feq_tree = sim.equilibrium(rho_tree, u_tree, cast_output=False)

    for f, feq, K, columns in zip(fin_tree, feq_tree, sim.collision_matrix, sim.collision_terms, strict=True):
        difference = f - feq
        dense = jnp.dot(difference, K)
        symbolic = jnp.stack(
            [sum(difference[..., i] * coefficient for i, coefficient in terms) for terms in columns],
            axis=-1,
        )
        dense, symbolic = np.asarray(dense), np.asarray(symbolic)
        assert np.max(np.abs(dense - symbolic)) < 1e-6
        assert np.array_equal(np.isfinite(dense), np.isfinite(symbolic))


if __name__ == "__main__":
    test_fused_symbolic_collision_matches_reference_without_surface_tension()
    test_fused_symbolic_collision_matches_reference_with_surface_tension()
    test_dense_and_symbolic_collision_matrix_agree()
    print("fused/symbolic MRT collision matches the unfused reference and the dense collision matrix")
