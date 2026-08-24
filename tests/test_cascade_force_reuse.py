"""Regression coverage for the fused force/potential evaluation in MultiphaseCascade.collision."""

import jax.numpy as jnp
import numpy as np
from jax.tree import map as tree_map

from jax_lab.core.eos import VanderWaals
from jax_lab.core.lattice import LatticeD2Q9
from jax_lab.core.multiphase import MultiphaseCascade

DOMAIN = (12, 10)
PRECISION = "f32/f32"


def _build_sim():
    matrix = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, -1, 0, 1, -1, -1, 1],
        [0, 0, 1, 0, -1, 1, 1, -1, -1],
        [0, 1, 1, 1, 1, 2, 2, 2, 2],
        [0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 1, 1, -1, -1],
        [0, 0, 0, 0, 0, 1, -1, -1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
    ])
    return MultiphaseCascade(
        n_components=1,
        lattice=LatticeD2Q9(PRECISION),
        nx=DOMAIN[0],
        ny=DOMAIN[1],
        nz=0,
        g_kkprime=-np.ones((1, 1)),
        body_force=[1e-6, -2e-6],
        k=[1.0],
        A=np.zeros((1, 1)),
        EOS=VanderWaals(a=[9.0 / 49.0], b=[2.0 / 21.0], R=[1.0], T=0.8 * 0.5714285714),
        M=[matrix],
        s_0=[1.0],
        s_1=[1.0],
        s_b=[0.8],
        s_2=[0.8],
        s_3=[1.0],
        s_4=[1.0],
        sigma=[0.05],
        precision=PRECISION,
        io_rate=0,
        print_info_rate=0,
        checkpoint_rate=0,
    )


def _reference_collision(sim, fin_tree):
    """Previous collision dataflow, which evaluated force once for velocity and again in apply_force."""
    fin_tree = tree_map(lambda f: sim.precision_policy.cast_to_compute(f), fin_tree)
    rho_tree, _ = sim.update_macroscopic(fin_tree)
    u_tree = sim.macroscopic_velocity(fin_tree, rho_tree)
    moment_tree = tree_map(lambda f, M: jnp.dot(f, M), fin_tree, sim.M)
    central_tree = sim.compute_central_moment(moment_tree, u_tree)
    equilibrium_tree = sim.compute_eq_central_moments(rho_tree)
    relaxed_tree = tree_map(
        lambda central, equilibrium, S: jnp.dot(central, jnp.eye(sim.q) - S) + jnp.dot(equilibrium, S),
        central_tree,
        equilibrium_tree,
        sim.S,
    )
    forced_tree = sim.apply_force(relaxed_tree, rho_tree, u_tree)
    shifted_tree = sim.compute_central_moment_inverse(forced_tree, u_tree)
    return tree_map(lambda shifted, inverse: jnp.dot(shifted, inverse), shifted_tree, sim.M_inv)


def test_cascade_collision_matches_previous_force_dataflow():
    sim = _build_sim()
    rng = np.random.default_rng(0)
    fin_tree = [sim.assign_fields_sharded()[0] + jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(*DOMAIN, sim.q)), dtype=jnp.float32)]
    actual = sim.collision(fin_tree)
    expected = _reference_collision(sim, fin_tree)
    assert np.max(np.abs(np.asarray(actual[0]) - np.asarray(expected[0]))) < 2e-6
