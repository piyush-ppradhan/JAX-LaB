import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial
from .base import LBMBase
from .lattice import LatticeD2Q9, LatticeD3Q19, LatticeD3Q27

"""
Collision operators are defined in this file for different models.
"""


class BGKSim(LBMBase):
    """
    BGK simulation class.

    This class implements the Bhatnagar-Gross-Krook (BGK) approximation for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        f = self.precision_policy.cast_to_compute(f)
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precision_policy.cast_to_output(fout)


class KBCSim(LBMBase):
    """
    KBC simulation class.

    This class implements the Karlin-Bösch-Chikatamarla (KBC) model for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(self, **kwargs):
        if kwargs.get("lattice").name != "D3Q27" and kwargs.get("nz") > 0:
            raise ValueError("KBC collision operator in 3D must only be used with D3Q27 lattice.")
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        KBC collision step for lattice.
        """
        f = self.precision_policy.cast_to_compute(f)
        tiny = 1e-32
        beta = self.omega * 0.5
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        if self.dim == 2:
            deltaS = self.fdecompose_shear_d2q9(fneq) * rho / 4.0
        else:
            deltaS = self.fdecompose_shear_d3q27(fneq) * rho
        deltaH = fneq - deltaS
        invBeta = 1.0 / beta
        gamma = invBeta - (2.0 - invBeta) * self.entropic_scalar_product(deltaS, deltaH, feq) / (
            tiny + self.entropic_scalar_product(deltaH, deltaH, feq)
        )

        fout = f - beta * (2.0 * deltaS + gamma[..., None] * deltaH)

        # add external force
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precision_policy.cast_to_output(fout)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision_modified(self, f):
        """
        Alternative KBC collision step for lattice.
        Note:
        At low Reynolds number the orignal KBC collision above produces inaccurate results because
        it does not check for the entropy increase/decrease. The KBC stabalizations should only be
        applied in principle to cells whose entropy decrease after a regular BGK collision. This is
        the case in most cells at higher Reynolds numbers and hence a check may not be needed.
        Overall the following alternative collision is more reliable and may replace the original
        implementation. The issue at the moment is that it is about 60-80% slower than the above method.
        """
        f = self.precision_policy.cast_to_compute(f)
        tiny = 1e-32
        beta = self.omega * 0.5
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, castOutput=False)

        # Alternative KBC: only stabalizes for voxels whose entropy decreases after BGK collision.
        f_bgk = f - self.omega * (f - feq)
        H_fin = jnp.sum(f * jnp.log(f / self.w), axis=-1, keepdims=True)
        H_fout = jnp.sum(f_bgk * jnp.log(f_bgk / self.w), axis=-1, keepdims=True)

        # the rest is identical to collision_deprecated
        fneq = f - feq
        if self.dim == 2:
            deltaS = self.fdecompose_shear_d2q9(fneq) * rho / 4.0
        else:
            deltaS = self.fdecompose_shear_d3q27(fneq) * rho
        deltaH = fneq - deltaS
        invBeta = 1.0 / beta
        gamma = invBeta - (2.0 - invBeta) * self.entropic_scalar_product(deltaS, deltaH, feq) / (
            tiny + self.entropic_scalar_product(deltaH, deltaH, feq)
        )

        f_kbc = f - beta * (2.0 * deltaS + gamma[..., None] * deltaH)
        fout = jnp.where(H_fout > H_fin, f_kbc, f_bgk)

        # add external force
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precision_policy.cast_to_output(fout)

    @partial(jit, static_argnums=(0,), inline=True)
    def entropic_scalar_product(self, x, y, feq):
        """
        Compute the entropic scalar product of x and y to approximate gamma in KBC.

        Returns
        -------
        jax.numpy.array
            Entropic scalar product of x, y, and feq.
        """
        return jnp.sum(x * y / feq, axis=-1)

    @partial(jit, static_argnums=(0,), inline=True)
    def fdecompose_shear_d2q9(self, fneq):
        """
        Decompose fneq into shear components for D2Q9 lattice.

        Parameters
        ----------
        fneq : jax.numpy.array
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.array
            Shear components of fneq.
        """
        Pi = self.momentum_flux(fneq)
        N = Pi[..., 0] - Pi[..., 2]
        s = jnp.zeros_like(fneq)
        s = s.at[..., 6].set(N)
        s = s.at[..., 3].set(N)
        s = s.at[..., 2].set(-N)
        s = s.at[..., 1].set(-N)
        s = s.at[..., 8].set(Pi[..., 1])
        s = s.at[..., 4].set(-Pi[..., 1])
        s = s.at[..., 5].set(-Pi[..., 1])
        s = s.at[..., 7].set(Pi[..., 1])

        return s

    @partial(jit, static_argnums=(0,), inline=True)
    def fdecompose_shear_d3q27(self, fneq):
        """
        Decompose fneq into shear components for D3Q27 lattice.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components of fneq.
        """
        # if self.grid.dim == 3:
        #     diagonal    = (0, 3, 5)
        #     offdiagonal = (1, 2, 4)
        # elif self.grid.dim == 2:
        #     diagonal    = (0, 2)
        #     offdiagonal = (1,)

        # c=
        # array([[0, 0, 0],-----0
        #        [0, 0, -1],----1
        #        [0, 0, 1],-----2
        #        [0, -1, 0],----3
        #        [0, -1, -1],---4
        #        [0, -1, 1],----5
        #        [0, 1, 0],-----6
        #        [0, 1, -1],----7
        #        [0, 1, 1],-----8
        #        [-1, 0, 0],----9
        #        [-1, 0, -1],--10
        #        [-1, 0, 1],---11
        #        [-1, -1, 0],--12
        #        [-1, -1, -1],-13
        #        [-1, -1, 1],--14
        #        [-1, 1, 0],---15
        #        [-1, 1, -1],--16
        #        [-1, 1, 1],---17
        #        [1, 0, 0],----18
        #        [1, 0, -1],---19
        #        [1, 0, 1],----20
        #        [1, -1, 0],---21
        #        [1, -1, -1],--22
        #        [1, -1, 1],---23
        #        [1, 1, 0],----24
        #        [1, 1, -1],---25
        #        [1, 1, 1]])---26
        Pi = self.momentum_flux(fneq)
        Nxz = Pi[..., 0] - Pi[..., 5]
        Nyz = Pi[..., 3] - Pi[..., 5]

        # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
        s = jnp.zeros_like(fneq)
        s = s.at[..., 9].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[..., 18].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[..., 3].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[..., 6].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[..., 1].set((-Nxz - Nyz) / 6.0)
        s = s.at[..., 2].set((-Nxz - Nyz) / 6.0)

        # For c = (i, j, 0)
        s = s.at[..., 12].set(Pi[..., 1] / 4.0)
        s = s.at[..., 24].set(Pi[..., 1] / 4.0)
        s = s.at[..., 21].set(-Pi[..., 1] / 4.0)
        s = s.at[..., 15].set(-Pi[..., 1] / 4.0)

        # For c = (i, 0, k)
        s = s.at[..., 10].set(Pi[..., 2] / 4.0)
        s = s.at[..., 20].set(Pi[..., 2] / 4.0)
        s = s.at[..., 19].set(-Pi[..., 2] / 4.0)
        s = s.at[..., 11].set(-Pi[..., 2] / 4.0)

        # For c = (0, j, k)
        s = s.at[..., 8].set(Pi[..., 4] / 4.0)
        s = s.at[..., 4].set(Pi[..., 4] / 4.0)
        s = s.at[..., 7].set(-Pi[..., 4] / 4.0)
        s = s.at[..., 5].set(-Pi[..., 4] / 4.0)

        return s


class AdvectionDiffusionBGK(LBMBase):
    """
    Advection Diffusion Model based on the BGK model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vel = kwargs.get("vel", None)
        if self.vel is None:
            raise ValueError("Velocity must be specified for AdvectionDiffusionBGK.")

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice.
        """
        f = self.precision_policy.cast_to_compute(f)
        rho = jnp.sum(f, axis=-1, keepdims=True)
        feq = self.equilibrium(rho, self.vel, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        return self.precision_policy.cast_to_output(fout)


class MRTSim(LBMBase):
    """
    Multi-relaxation time model.
    """

    def __init__(self, **kwargs):
        kwargs.update({"omega": 1.0})
        super().__init__(**kwargs)
        self.s_rho = kwargs.get("s_rho")
        self.s_e = kwargs.get("s_e")
        self.s_eta = kwargs.get("s_eta")
        self.s_j = kwargs.get("s_j")
        self.s_q = kwargs.get("s_q")
        self.s_v = self.omega
        self.M_inv = jnp.array(
            np.transpose(np.linalg.inv(kwargs.get("M"))),
            dtype=self.precision_policy.compute_dtype,
        )
        self.M = jnp.array(np.transpose(kwargs.get("M")), dtype=self.precision_policy.compute_dtype)
        if isinstance(self.lattice, LatticeD2Q9):
            self.S = jnp.array(
                np.diag([self.s_rho, self.s_e, self.s_eta, self.s_j, self.s_q, self.s_j, self.s_q, self.s_v, self.s_v]),
                dtype=self.precision_policy.compute_dtype,
            )
        elif isinstance(self.lattice, LatticeD3Q19):
            self.s_pi = kwargs.get("s_pi")
            self.s_m = kwargs.get("s_m")
            self.S = jnp.array(
                np.diag([
                    self.s_rho,
                    self.s_e,
                    self.s_eta,
                    self.s_j,
                    self.s_q,
                    self.s_j,
                    self.s_q,
                    self.s_j,
                    self.s_q,
                    self.s_v,
                    self.s_pi,
                    self.s_v,
                    self.s_pi,
                    self.s_v,
                    self.s_v,
                    self.s_v,
                    self.s_m,
                    self.s_m,
                    self.s_m,
                ]),
                dtype=self.precision_policy.compute_dtype,
            )
        else:
            NotImplementedError(f"Lattice type {self.lattice.name} has not been implemented")

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        MRT collision step for lattice.
        """
        f = self.precision_policy.cast_to_compute(f)
        m = jnp.dot(f, self.M)
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u)
        meq = jnp.dot(feq, self.M)
        mout = -jnp.dot(m - meq, self.S)
        if self.force is not None:
            mout = self.apply_force(mout, meq, rho, u)
        return self.precision_policy.cast_to_output(f + jnp.dot(mout, self.M_inv))


class CLBMSim(LBMBase):
    """
    Central moment (cascaded) collision model.
    """

    def __init__(self, **kwargs):
        kwargs.update({"omega": 1.0})
        super().__init__(**kwargs)
        self.M_inv = jnp.array(
            np.transpose(np.linalg.inv(kwargs.get("M"))),
            dtype=self.precision_policy.compute_dtype,
        )
        self.M = jnp.array(np.transpose(kwargs.get("M")), dtype=self.precision_policy.compute_dtype)
        self.s_0 = kwargs.get("s_0")
        self.s_1 = kwargs.get("s_1")
        self.s_b = kwargs.get("s_b")
        self.s_2 = kwargs.get("s_2")
        self.s_3 = kwargs.get("s_3")
        self.s_4 = kwargs.get("s_4")
        self.s_v = self.omega
        if isinstance(self.lattice, LatticeD2Q9):
            self.S = jnp.array(
                np.diag([self.s_0, self.s_1, self.s_1, self.s_b, self.s_2, self.s_2, self.s_3, self.s_3, self.s_4]),
                dtype=self.precision_policy.compute_dtype,
            )
        elif isinstance(self.lattice, LatticeD3Q19):
            self.s_plus = (self.s_b + 2 * self.s_2) / 3
            self.s_minus = (self.s_b - self.s_2) / 3

            S = np.diag([
                self.s_0,
                self.s_1,
                self.s_1,
                self.s_1,
                self.s_v,
                self.s_v,
                self.s_v,
                self.s_plus,
                self.s_plus,
                self.s_plus,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_4,
                self.s_4,
                self.s_4,
            ])
            S[7, 8] = self.s_minus
            S[7, 9] = self.s_minus
            S[8, 7] = self.s_minus
            S[8, 9] = self.s_minus
            S[9, 7] = self.s_minus
            S[9, 8] = self.s_minus
            self.S = jnp.array(S, dtype=self.precision_policy.compute_dtype)

        elif isinstance(self.lattice, LatticeD3Q27):
            self.s_plus = (self.s_b + 2 * self.s_2) / 3
            self.s_minus = (self.s_b - self.s_2) / 3
            self.s_3b = kwargs.get("s_3b")
            self.s_4b = kwargs.get("s_4b")
            self.s_5 = kwargs.get("s_5")
            self.s_6 = kwargs.get("s_6")

            S = np.diag([
                self.s_0,
                self.s_1,
                self.s_1,
                self.s_1,
                self.s_v,
                self.s_v,
                self.s_v,
                self.s_plus,
                self.s_plus,
                self.s_plus,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3b,
                self.s_4,
                self.s_4,
                self.s_4,
                self.s_4b,
                self.s_4b,
                self.s_4b,
                self.s_5,
                self.s_5,
                self.s_5,
                self.s_6,
            ])
            S[7, 8] = self.s_minus
            S[7, 9] = self.s_minus
            S[8, 7] = self.s_minus
            S[8, 9] = self.s_minus
            S[9, 7] = self.s_minus
            S[9, 8] = self.s_minus
            self.S = jnp.array(S, dtype=self.precision_policy.compute_dtype)

    @partial(jit, static_argnums=(0,), inline=True)
    def macroscopic_velocity(self, f, rho):
        """
        macroscopic_velocity computes the velocity and incorporates forces into velocity for Exact Difference Method (EDM) (used for SRT and MRT collision) models
        and the consistent forcing scheme developed by LinLin Fei et. al (for Cascaded LBM). This is used for post-processing only and not for equilibrium distribution computation.

        Parameters
        ----------
        f: jax.numpy.ndarray
            Distribution arrays.
        rho: jax.numpy.ndarray
            Density fields.

        Returns
        -------
        u: jax.numpy.ndarray
            Velocity fields.
        """
        # rho_tree = map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype).T
        u = jnp.dot(f, c) / rho
        if self.force is not None:
            return u + 0.5 * self.force / rho
        else:
            return u

    @partial(jit, static_argnums=(0,))
    def compute_central_moment(self, m, u):
        if isinstance(self.lattice, LatticeD2Q9):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                usq = ux**2 + uy**2
                udiff = ux**2 - uy**2
                T = jnp.zeros_like(m)
                T = T.at[..., 0].set(m[..., 0])
                T = T.at[..., 1].set(-ux * m[..., 0] + m[..., 1])
                T = T.at[..., 2].set(-uy * m[..., 0] + m[..., 2])
                T = T.at[..., 3].set(usq * m[..., 0] - 2 * ux * m[..., 1] - 2 * uy * m[..., 2] + m[..., 3])
                T = T.at[..., 4].set(udiff * m[..., 0] - 2 * ux * m[..., 1] + 2 * uy * m[..., 2] + m[..., 4])
                T = T.at[..., 5].set(ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 5])
                T = T.at[..., 6].set(
                    -(ux**2) * uy * m[..., 0]
                    + 2 * ux * uy * m[..., 1]
                    + ux**2 * m[..., 2]
                    - 0.5 * uy * m[..., 3]
                    - 0.5 * uy * m[..., 4]
                    - 2 * ux * m[..., 5]
                    + m[..., 6]
                )
                T = T.at[..., 7].set(
                    -(uy**2) * ux * m[..., 0]
                    + uy**2 * m[..., 1]
                    + 2 * ux * uy * m[..., 2]
                    - 0.5 * ux * m[..., 3]
                    + 0.5 * ux * m[..., 4]
                    - 2 * uy * m[..., 5]
                    + m[..., 7]
                )
                T = T.at[..., 8].set(
                    (uy**2 * ux**2) * m[..., 0]
                    - 2 * ux * uy**2 * m[..., 1]
                    - 2 * uy * ux**2 * m[..., 2]
                    + 0.5 * usq * m[..., 3]
                    - 0.5 * udiff * m[..., 4]
                    + 4 * ux * uy * m[..., 5]
                    - 2 * uy * m[..., 6]
                    - 2 * ux * m[..., 7]
                    + m[..., 8]
                )
                return T

            return shift(m, u)

        elif isinstance(self.lattice, LatticeD3Q19):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                T = jnp.zeros_like(m)
                T = T.at[..., 0].set(m[..., 0])
                T = T.at[..., 1].set(-ux * m[..., 0] + m[..., 1])
                T = T.at[..., 2].set(-uy * m[..., 0] + m[..., 2])
                T = T.at[..., 3].set(-uz * m[..., 0] + m[..., 3])
                T = T.at[..., 4].set(ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 4])
                T = T.at[..., 5].set(ux * uz * m[..., 0] - uz * m[..., 1] - ux * m[..., 3] + m[..., 5])
                T = T.at[..., 6].set(uy * uz * m[..., 0] - uz * m[..., 2] - uy * m[..., 3] + m[..., 6])
                T = T.at[..., 7].set((ux**2) * m[..., 0] - 2 * ux * m[..., 1] + m[..., 7])
                T = T.at[..., 8].set((uy**2) * m[..., 0] - 2 * uy * m[..., 2] + m[..., 8])
                T = T.at[..., 9].set((uz**2) * m[..., 0] - 2 * uz * m[..., 3] + m[..., 9])
                T = T.at[..., 10].set(
                    -ux * (uy**2) * m[..., 0] + (uy**2) * m[..., 1] + 2 * ux * uy * m[..., 2] - 2 * uy * m[..., 4] - ux * m[..., 8] + m[..., 10]
                )
                T = T.at[..., 11].set(
                    -ux * (uz**2) * m[..., 0] + (uz**2) * m[..., 1] + 2 * ux * uz * m[..., 3] - 2 * uz * m[..., 5] - ux * m[..., 9] + m[..., 11]
                )
                T = T.at[..., 12].set(
                    -(ux**2) * uy * m[..., 0] + 2 * ux * uy * m[..., 1] + (ux**2) * m[..., 2] - 2 * ux * m[..., 4] - uy * m[..., 7] + m[..., 12]
                )
                T = T.at[..., 13].set(
                    -(ux**2) * uz * m[..., 0] + 2 * ux * uz * m[..., 1] + (ux**2) * m[..., 3] - 2 * ux * m[..., 5] - uz * m[..., 7] + m[..., 13]
                )
                T = T.at[..., 14].set(
                    -uy * (uz**2) * m[..., 0] + (uz**2) * m[..., 2] + 2 * uy * uz * m[..., 3] - 2 * uz * m[..., 6] - uy * m[..., 9] + m[..., 14]
                )
                T = T.at[..., 15].set(
                    -(uy**2) * uz * m[..., 0] + 2 * uy * uz * m[..., 2] + (uy**2) * m[..., 3] - 2 * uy * m[..., 6] - uz * m[..., 8] + m[..., 15]
                )
                T = T.at[..., 16].set(
                    (ux**2) * (uy**2) * m[..., 0]
                    - 2 * ux * (uy**2) * m[..., 1]
                    - 2 * uy * (ux**2) * m[..., 2]
                    + 4 * ux * uy * m[..., 4]
                    + (uy**2) * m[..., 7]
                    + (ux**2) * m[..., 8]
                    - 2 * ux * m[..., 10]
                    - 2 * uy * m[..., 12]
                    + m[..., 16]
                )
                T = T.at[..., 17].set(
                    (ux**2) * (uz**2) * m[..., 0]
                    - 2 * ux * (uz**2) * m[..., 1]
                    - 2 * uz * (ux**2) * m[..., 3]
                    + 4 * ux * uz * m[..., 5]
                    + (uz**2) * m[..., 7]
                    + (ux**2) * m[..., 9]
                    - 2 * ux * m[..., 11]
                    - 2 * uz * m[..., 13]
                    + m[..., 17]
                )
                T = T.at[..., 18].set(
                    (uy**2) * (uz**2) * m[..., 0]
                    - 2 * uy * (uz**2) * m[..., 2]
                    - 2 * uz * (uy**2) * m[..., 3]
                    + 4 * uy * uz * m[..., 6]
                    + (uz**2) * m[..., 8]
                    + (uy**2) * m[..., 9]
                    - 2 * uy * m[..., 14]
                    - 2 * uz * m[..., 15]
                    + m[..., 18]
                )
                return T

            return shift(m, u)

        elif isinstance(self.lattice, LatticeD3Q27):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                T = jnp.zeros_like(m)
                T = T.at[..., 0].set(m[..., 0])
                T = T.at[..., 1].set(m[..., 1] - m[..., 0] * ux)
                T = T.at[..., 2].set(m[..., 2] - m[..., 0] * uy)
                T = T.at[..., 3].set(m[..., 3] - m[..., 0] * uz)
                T = T.at[..., 4].set(m[..., 4] - m[..., 2] * ux - m[..., 1] * uy + m[..., 0] * ux * uy)
                T = T.at[..., 5].set(m[..., 5] - m[..., 3] * ux - m[..., 1] * uz + m[..., 0] * ux * uz)
                T = T.at[..., 6].set(m[..., 6] - m[..., 3] * uy - m[..., 2] * uz + m[..., 0] * uy * uz)
                T = T.at[..., 7].set(m[..., 0] * ux**2 - 2 * m[..., 1] * ux + m[..., 7])
                T = T.at[..., 8].set(m[..., 0] * uy**2 - 2 * m[..., 2] * uy + m[..., 8])
                T = T.at[..., 9].set(m[..., 0] * uz**2 - 2 * m[..., 3] * uz + m[..., 9])
                T = T.at[..., 10].set(
                    m[..., 10] - m[..., 8] * ux - 2 * m[..., 4] * uy + m[..., 1] * uy**2 - m[..., 0] * ux * uy**2 + 2 * m[..., 2] * ux * uy
                )
                T = T.at[..., 11].set(
                    m[..., 11] - m[..., 9] * ux - 2 * m[..., 5] * uz + m[..., 1] * uz**2 - m[..., 0] * ux * uz**2 + 2 * m[..., 3] * ux * uz
                )
                T = T.at[..., 12].set(
                    m[..., 12] - 2 * m[..., 4] * ux - m[..., 7] * uy + m[..., 2] * ux**2 - m[..., 0] * ux**2 * uy + 2 * m[..., 1] * ux * uy
                )
                T = T.at[..., 13].set(
                    m[..., 13] - 2 * m[..., 5] * ux - m[..., 7] * uz + m[..., 3] * ux**2 - m[..., 0] * ux**2 * uz + 2 * m[..., 1] * ux * uz
                )
                T = T.at[..., 14].set(
                    m[..., 14] - m[..., 9] * uy - 2 * m[..., 6] * uz + m[..., 2] * uz**2 - m[..., 0] * uy * uz**2 + 2 * m[..., 3] * uy * uz
                )
                T = T.at[..., 15].set(
                    m[..., 15] - 2 * m[..., 6] * uy - m[..., 8] * uz + m[..., 3] * uy**2 - m[..., 0] * uy**2 * uz + 2 * m[..., 2] * uy * uz
                )
                T = T.at[..., 16].set(
                    m[..., 16]
                    - m[..., 6] * ux
                    - m[..., 5] * uy
                    - m[..., 4] * uz
                    + m[..., 3] * ux * uy
                    + m[..., 2] * ux * uz
                    + m[..., 1] * uy * uz
                    - m[..., 0] * ux * uy * uz
                )
                T = T.at[..., 17].set(
                    m[..., 0] * ux**2 * uy**2
                    - 2 * m[..., 2] * ux**2 * uy
                    + m[..., 8] * ux**2
                    - 2 * m[..., 1] * ux * uy**2
                    + 4 * m[..., 4] * ux * uy
                    - 2 * m[..., 10] * ux
                    + m[..., 7] * uy**2
                    - 2 * m[..., 12] * uy
                    + m[..., 17]
                )
                T = T.at[..., 18].set(
                    m[..., 0] * ux**2 * uz**2
                    - 2 * m[..., 3] * ux**2 * uz
                    + m[..., 9] * ux**2
                    - 2 * m[..., 1] * ux * uz**2
                    + 4 * m[..., 5] * ux * uz
                    - 2 * m[..., 11] * ux
                    + m[..., 7] * uz**2
                    - 2 * m[..., 13] * uz
                    + m[..., 18]
                )
                T = T.at[..., 19].set(
                    m[..., 0] * uy**2 * uz**2
                    - 2 * m[..., 3] * uy**2 * uz
                    + m[..., 9] * uy**2
                    - 2 * m[..., 2] * uy * uz**2
                    + 4 * m[..., 6] * uy * uz
                    - 2 * m[..., 14] * uy
                    + m[..., 8] * uz**2
                    - 2 * m[..., 15] * uz
                    + m[..., 19]
                )
                T = T.at[..., 20].set(
                    m[..., 20]
                    - 2 * m[..., 16] * ux
                    - m[..., 13] * uy
                    - m[..., 12] * uz
                    + m[..., 6] * ux**2
                    - m[..., 3] * ux**2 * uy
                    - m[..., 2] * ux**2 * uz
                    + 2 * m[..., 5] * ux * uy
                    + 2 * m[..., 4] * ux * uz
                    + m[..., 7] * uy * uz
                    - 2 * m[..., 1] * ux * uy * uz
                    + m[..., 0] * ux**2 * uy * uz
                )
                T = T.at[..., 21].set(
                    m[..., 21]
                    - m[..., 15] * ux
                    - 2 * m[..., 16] * uy
                    - m[..., 10] * uz
                    + m[..., 5] * uy**2
                    - m[..., 3] * ux * uy**2
                    - m[..., 1] * uy**2 * uz
                    + 2 * m[..., 6] * ux * uy
                    + m[..., 8] * ux * uz
                    + 2 * m[..., 4] * uy * uz
                    - 2 * m[..., 2] * ux * uy * uz
                    + m[..., 0] * ux * uy**2 * uz
                )
                T = T.at[..., 22].set(
                    m[..., 22]
                    - m[..., 14] * ux
                    - m[..., 11] * uy
                    - 2 * m[..., 16] * uz
                    + m[..., 4] * uz**2
                    - m[..., 2] * ux * uz**2
                    - m[..., 1] * uy * uz**2
                    + m[..., 9] * ux * uy
                    + 2 * m[..., 6] * ux * uz
                    + 2 * m[..., 5] * uy * uz
                    - 2 * m[..., 3] * ux * uy * uz
                    + m[..., 0] * ux * uy * uz**2
                )
                T = T.at[..., 23].set(
                    m[..., 23]
                    - m[..., 19] * ux
                    - 2 * m[..., 22] * uy
                    - 2 * m[..., 21] * uz
                    + m[..., 11] * uy**2
                    + m[..., 10] * uz**2
                    - m[..., 9] * ux * uy**2
                    - m[..., 8] * ux * uz**2
                    - 2 * m[..., 4] * uy * uz**2
                    - 2 * m[..., 5] * uy**2 * uz
                    + m[..., 1] * uy**2 * uz**2
                    + 2 * m[..., 14] * ux * uy
                    + 2 * m[..., 15] * ux * uz
                    + 4 * m[..., 16] * uy * uz
                    - 4 * m[..., 6] * ux * uy * uz
                    + 2 * m[..., 2] * ux * uy * uz**2
                    + 2 * m[..., 3] * ux * uy**2 * uz
                    - m[..., 0] * ux * uy**2 * uz**2
                )
                T = T.at[..., 24].set(
                    m[..., 24]
                    - 2 * m[..., 22] * ux
                    - m[..., 18] * uy
                    - 2 * m[..., 20] * uz
                    + m[..., 14] * ux**2
                    + m[..., 12] * uz**2
                    - m[..., 9] * ux**2 * uy
                    - 2 * m[..., 4] * ux * uz**2
                    - 2 * m[..., 6] * ux**2 * uz
                    - m[..., 7] * uy * uz**2
                    + m[..., 2] * ux**2 * uz**2
                    + 2 * m[..., 11] * ux * uy
                    + 4 * m[..., 16] * ux * uz
                    + 2 * m[..., 13] * uy * uz
                    - 4 * m[..., 5] * ux * uy * uz
                    + 2 * m[..., 1] * ux * uy * uz**2
                    + 2 * m[..., 3] * ux**2 * uy * uz
                    - m[..., 0] * ux**2 * uy * uz**2
                )
                T = T.at[..., 25].set(
                    m[..., 25]
                    - 2 * m[..., 21] * ux
                    - 2 * m[..., 20] * uy
                    - m[..., 17] * uz
                    + m[..., 15] * ux**2
                    + m[..., 13] * uy**2
                    - 2 * m[..., 5] * ux * uy**2
                    - 2 * m[..., 6] * ux**2 * uy
                    - m[..., 8] * ux**2 * uz
                    - m[..., 7] * uy**2 * uz
                    + m[..., 3] * ux**2 * uy**2
                    + 4 * m[..., 16] * ux * uy
                    + 2 * m[..., 10] * ux * uz
                    + 2 * m[..., 12] * uy * uz
                    - 4 * m[..., 4] * ux * uy * uz
                    + 2 * m[..., 1] * ux * uy**2 * uz
                    + 2 * m[..., 2] * ux**2 * uy * uz
                    - m[..., 0] * ux**2 * uy**2 * uz
                )
                T = T.at[..., 26].set(
                    m[..., 0] * ux**2 * uy**2 * uz**2
                    - 2 * m[..., 3] * ux**2 * uy**2 * uz
                    + m[..., 9] * ux**2 * uy**2
                    - 2 * m[..., 2] * ux**2 * uy * uz**2
                    + 4 * m[..., 6] * ux**2 * uy * uz
                    - 2 * m[..., 14] * ux**2 * uy
                    + m[..., 8] * ux**2 * uz**2
                    - 2 * m[..., 15] * ux**2 * uz
                    + m[..., 19] * ux**2
                    - 2 * m[..., 1] * ux * uy**2 * uz**2
                    + 4 * m[..., 5] * ux * uy**2 * uz
                    - 2 * m[..., 11] * ux * uy**2
                    + 4 * m[..., 4] * ux * uy * uz**2
                    - 8 * m[..., 16] * ux * uy * uz
                    + 4 * m[..., 22] * ux * uy
                    - 2 * m[..., 10] * ux * uz**2
                    + 4 * m[..., 21] * ux * uz
                    - 2 * m[..., 23] * ux
                    + m[..., 7] * uy**2 * uz**2
                    - 2 * m[..., 13] * uy**2 * uz
                    + m[..., 18] * uy**2
                    - 2 * m[..., 12] * uy * uz**2
                    + 4 * m[..., 20] * uy * uz
                    - 2 * m[..., 24] * uy
                    + m[..., 17] * uz**2
                    - 2 * m[..., 25] * uz
                    + m[..., 26]
                )

                return T

            return shift(m, u)

    @partial(jit, static_argnums=(0,))
    def compute_central_moment_inverse(self, T, u):
        if isinstance(self.lattice, LatticeD2Q9):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                usq = ux**2 + uy**2
                udiff = ux**2 - uy**2
                m = jnp.zeros_like(T)
                m = m.at[..., 0].set(T[..., 0])
                m = m.at[..., 1].set(ux * T[..., 0] + T[..., 1])
                m = m.at[..., 2].set(uy * T[..., 0] + T[..., 2])
                m = m.at[..., 3].set(usq * T[..., 0] + 2 * ux * T[..., 1] + 2 * uy * T[..., 2] + T[..., 3])
                m = m.at[..., 4].set(udiff * T[..., 0] + 2 * ux * T[..., 1] - 2 * uy * T[..., 2] + T[..., 4])
                m = m.at[..., 5].set(ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 5])
                m = m.at[..., 6].set(
                    (ux**2) * uy * T[..., 0]
                    + 2 * ux * uy * T[..., 1]
                    + ux**2 * T[..., 2]
                    + 0.5 * uy * T[..., 3]
                    + 0.5 * uy * T[..., 4]
                    + 2 * ux * T[..., 5]
                    + T[..., 6]
                )
                m = m.at[..., 7].set(
                    (uy**2) * ux * T[..., 0]
                    + uy**2 * T[..., 1]
                    + 2 * ux * uy * T[..., 2]
                    + 0.5 * ux * T[..., 3]
                    - 0.5 * ux * T[..., 4]
                    + 2 * uy * T[..., 5]
                    + T[..., 7]
                )
                m = m.at[..., 8].set(
                    (uy**2 * ux**2) * T[..., 0]
                    + 2 * ux * uy**2 * T[..., 1]
                    + 2 * uy * ux**2 * T[..., 2]
                    + 0.5 * usq * T[..., 3]
                    - 0.5 * udiff * T[..., 4]
                    + 4 * ux * uy * T[..., 5]
                    + 2 * uy * T[..., 6]
                    + 2 * ux * T[..., 7]
                    + T[..., 8]
                )
                return m

            return shift_inverse(T, u)

        elif isinstance(self.lattice, LatticeD3Q19):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                m = jnp.zeros_like(T)
                m = m.at[..., 0].set(T[..., 0])
                m = m.at[..., 1].set(ux * T[..., 0] + T[..., 1])
                m = m.at[..., 2].set(uy * T[..., 0] + T[..., 2])
                m = m.at[..., 3].set(uz * T[..., 0] + T[..., 3])
                m = m.at[..., 4].set(ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 4])
                m = m.at[..., 5].set(ux * uz * T[..., 0] + uz * T[..., 1] + ux * T[..., 3] + T[..., 5])
                m = m.at[..., 6].set(uy * uz * T[..., 0] + uz * T[..., 2] + uy * T[..., 3] + T[..., 6])
                m = m.at[..., 7].set((ux**2) * T[..., 0] + 2 * ux * T[..., 1] + T[..., 7])
                m = m.at[..., 8].set((uy**2) * T[..., 0] + 2 * uy * T[..., 2] + T[..., 8])
                m = m.at[..., 9].set((uz**2) * T[..., 0] + 2 * uz * T[..., 3] + T[..., 9])
                m = m.at[..., 10].set(
                    ux * (uy**2) * T[..., 0] + (uy**2) * T[..., 1] + 2 * ux * uy * T[..., 2] + 2 * uy * T[..., 4] + ux * T[..., 8] + T[..., 10]
                )
                m = m.at[..., 11].set(
                    ux * (uz**2) * T[..., 0] + (uz**2) * T[..., 1] + 2 * ux * uz * T[..., 3] + 2 * uz * T[..., 5] + ux * T[..., 9] + T[..., 11]
                )
                m = m.at[..., 12].set(
                    (ux**2) * uy * T[..., 0] + 2 * ux * uy * T[..., 1] + (ux**2) * T[..., 2] + 2 * ux * T[..., 4] + uy * T[..., 7] + T[..., 12]
                )
                m = m.at[..., 13].set(
                    (ux**2) * uz * T[..., 0] + 2 * ux * uz * T[..., 1] + (ux**2) * T[..., 3] + 2 * ux * T[..., 5] + uz * T[..., 7] + T[..., 13]
                )
                m = m.at[..., 14].set(
                    uy * (uz**2) * T[..., 0] + (uz**2) * T[..., 2] + 2 * uy * uz * T[..., 3] + 2 * uz * T[..., 6] + uy * T[..., 9] + T[..., 14]
                )
                m = m.at[..., 15].set(
                    (uy**2) * uz * T[..., 0] + 2 * uy * uz * T[..., 2] + (uy**2) * T[..., 3] + 2 * uy * T[..., 6] + uz * T[..., 8] + T[..., 15]
                )
                m = m.at[..., 16].set(
                    (ux**2) * (uy**2) * T[..., 0]
                    + 2 * ux * (uy**2) * T[..., 1]
                    + 2 * uy * (ux**2) * T[..., 2]
                    + 4 * ux * uy * T[..., 4]
                    + (uy**2) * T[..., 7]
                    + (ux**2) * T[..., 8]
                    + 2 * ux * T[..., 10]
                    + 2 * uy * T[..., 12]
                    + T[..., 16]
                )
                m = m.at[..., 17].set(
                    (ux**2) * (uz**2) * T[..., 0]
                    + 2 * ux * (uz**2) * T[..., 1]
                    + 2 * uz * (ux**2) * T[..., 3]
                    + 4 * ux * uz * T[..., 5]
                    + (uz**2) * T[..., 7]
                    + (ux**2) * T[..., 9]
                    + 2 * ux * T[..., 11]
                    + 2 * uz * T[..., 13]
                    + T[..., 17]
                )
                m = m.at[..., 18].set(
                    (uy**2) * (uz**2) * T[..., 0]
                    + 2 * uy * (uz**2) * T[..., 2]
                    + 2 * uz * (uy**2) * T[..., 3]
                    + 4 * uy * uz * T[..., 6]
                    + (uz**2) * T[..., 8]
                    + (uy**2) * T[..., 9]
                    + 2 * uy * T[..., 14]
                    + 2 * uz * T[..., 15]
                    + T[..., 18]
                )
                return m

            return shift_inverse(T, u)

        elif isinstance(self.lattice, LatticeD3Q27):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                m = jnp.zeros_like(T)
                m = m.at[..., 0].set(T[..., 0])
                m = m.at[..., 1].set(T[..., 1] + T[..., 0] * ux)
                m = m.at[..., 2].set(T[..., 2] + T[..., 0] * uy)
                m = m.at[..., 3].set(T[..., 3] + T[..., 0] * uz)
                m = m.at[..., 4].set(T[..., 4] + T[..., 2] * ux + T[..., 1] * uy + T[..., 0] * ux * uy)
                m = m.at[..., 5].set(T[..., 5] + T[..., 3] * ux + T[..., 1] * uz + T[..., 0] * ux * uz)
                m = m.at[..., 6].set(T[..., 6] + T[..., 3] * uy + T[..., 2] * uz + T[..., 0] * uy * uz)
                m = m.at[..., 7].set(T[..., 0] * ux**2 + 2 * T[..., 1] * ux + T[..., 7])
                m = m.at[..., 8].set(T[..., 0] * uy**2 + 2 * T[..., 2] * uy + T[..., 8])
                m = m.at[..., 9].set(T[..., 0] * uz**2 + 2 * T[..., 3] * uz + T[..., 9])
                m = m.at[..., 10].set(
                    T[..., 10] + T[..., 8] * ux + 2 * T[..., 4] * uy + T[..., 1] * uy**2 + T[..., 0] * ux * uy**2 + 2 * T[..., 2] * ux * uy
                )
                m = m.at[..., 11].set(
                    T[..., 11] + T[..., 9] * ux + 2 * T[..., 5] * uz + T[..., 1] * uz**2 + T[..., 0] * ux * uz**2 + 2 * T[..., 3] * ux * uz
                )
                m = m.at[..., 12].set(
                    T[..., 12] + 2 * T[..., 4] * ux + T[..., 7] * uy + T[..., 2] * ux**2 + T[..., 0] * ux**2 * uy + 2 * T[..., 1] * ux * uy
                )
                m = m.at[..., 13].set(
                    T[..., 13] + 2 * T[..., 5] * ux + T[..., 7] * uz + T[..., 3] * ux**2 + T[..., 0] * ux**2 * uz + 2 * T[..., 1] * ux * uz
                )
                m = m.at[..., 14].set(
                    T[..., 14] + T[..., 9] * uy + 2 * T[..., 6] * uz + T[..., 2] * uz**2 + T[..., 0] * uy * uz**2 + 2 * T[..., 3] * uy * uz
                )
                m = m.at[..., 15].set(
                    T[..., 15] + 2 * T[..., 6] * uy + T[..., 8] * uz + T[..., 3] * uy**2 + T[..., 0] * uy**2 * uz + 2 * T[..., 2] * uy * uz
                )
                m = m.at[..., 16].set(
                    T[..., 16]
                    + T[..., 6] * ux
                    + T[..., 5] * uy
                    + T[..., 4] * uz
                    + T[..., 3] * ux * uy
                    + T[..., 2] * ux * uz
                    + T[..., 1] * uy * uz
                    + T[..., 0] * ux * uy * uz
                )
                m = m.at[..., 17].set(
                    T[..., 0] * ux**2 * uy**2
                    + 2 * T[..., 2] * ux**2 * uy
                    + T[..., 8] * ux**2
                    + 2 * T[..., 1] * ux * uy**2
                    + 4 * T[..., 4] * ux * uy
                    + 2 * T[..., 10] * ux
                    + T[..., 7] * uy**2
                    + 2 * T[..., 12] * uy
                    + T[..., 17]
                )
                m = m.at[..., 18].set(
                    T[..., 0] * ux**2 * uz**2
                    + 2 * T[..., 3] * ux**2 * uz
                    + T[..., 9] * ux**2
                    + 2 * T[..., 1] * ux * uz**2
                    + 4 * T[..., 5] * ux * uz
                    + 2 * T[..., 11] * ux
                    + T[..., 7] * uz**2
                    + 2 * T[..., 13] * uz
                    + T[..., 18]
                )
                m = m.at[..., 19].set(
                    T[..., 0] * uy**2 * uz**2
                    + 2 * T[..., 3] * uy**2 * uz
                    + T[..., 9] * uy**2
                    + 2 * T[..., 2] * uy * uz**2
                    + 4 * T[..., 6] * uy * uz
                    + 2 * T[..., 14] * uy
                    + T[..., 8] * uz**2
                    + 2 * T[..., 15] * uz
                    + T[..., 19]
                )
                m = m.at[..., 20].set(
                    T[..., 20]
                    + 2 * T[..., 16] * ux
                    + T[..., 13] * uy
                    + T[..., 12] * uz
                    + T[..., 6] * ux**2
                    + T[..., 3] * ux**2 * uy
                    + T[..., 2] * ux**2 * uz
                    + 2 * T[..., 5] * ux * uy
                    + 2 * T[..., 4] * ux * uz
                    + T[..., 7] * uy * uz
                    + 2 * T[..., 1] * ux * uy * uz
                    + T[..., 0] * ux**2 * uy * uz
                )
                m = m.at[..., 21].set(
                    T[..., 21]
                    + T[..., 15] * ux
                    + 2 * T[..., 16] * uy
                    + T[..., 10] * uz
                    + T[..., 5] * uy**2
                    + T[..., 3] * ux * uy**2
                    + T[..., 1] * uy**2 * uz
                    + 2 * T[..., 6] * ux * uy
                    + T[..., 8] * ux * uz
                    + 2 * T[..., 4] * uy * uz
                    + 2 * T[..., 2] * ux * uy * uz
                    + T[..., 0] * ux * uy**2 * uz
                )
                m = m.at[..., 22].set(
                    T[..., 22]
                    + T[..., 14] * ux
                    + T[..., 11] * uy
                    + 2 * T[..., 16] * uz
                    + T[..., 4] * uz**2
                    + T[..., 2] * ux * uz**2
                    + T[..., 1] * uy * uz**2
                    + T[..., 9] * ux * uy
                    + 2 * T[..., 6] * ux * uz
                    + 2 * T[..., 5] * uy * uz
                    + 2 * T[..., 3] * ux * uy * uz
                    + T[..., 0] * ux * uy * uz**2
                )
                m = m.at[..., 23].set(
                    T[..., 23]
                    + T[..., 19] * ux
                    + 2 * T[..., 22] * uy
                    + 2 * T[..., 21] * uz
                    + T[..., 11] * uy**2
                    + T[..., 10] * uz**2
                    + T[..., 9] * ux * uy**2
                    + T[..., 8] * ux * uz**2
                    + 2 * T[..., 4] * uy * uz**2
                    + 2 * T[..., 5] * uy**2 * uz
                    + T[..., 1] * uy**2 * uz**2
                    + 2 * T[..., 14] * ux * uy
                    + 2 * T[..., 15] * ux * uz
                    + 4 * T[..., 16] * uy * uz
                    + 4 * T[..., 6] * ux * uy * uz
                    + 2 * T[..., 2] * ux * uy * uz**2
                    + 2 * T[..., 3] * ux * uy**2 * uz
                    + T[..., 0] * ux * uy**2 * uz**2
                )
                m = m.at[..., 24].set(
                    T[..., 24]
                    + 2 * T[..., 22] * ux
                    + T[..., 18] * uy
                    + 2 * T[..., 20] * uz
                    + T[..., 14] * ux**2
                    + T[..., 12] * uz**2
                    + T[..., 9] * ux**2 * uy
                    + 2 * T[..., 4] * ux * uz**2
                    + 2 * T[..., 6] * ux**2 * uz
                    + T[..., 7] * uy * uz**2
                    + T[..., 2] * ux**2 * uz**2
                    + 2 * T[..., 11] * ux * uy
                    + 4 * T[..., 16] * ux * uz
                    + 2 * T[..., 13] * uy * uz
                    + 4 * T[..., 5] * ux * uy * uz
                    + 2 * T[..., 1] * ux * uy * uz**2
                    + 2 * T[..., 3] * ux**2 * uy * uz
                    + T[..., 0] * ux**2 * uy * uz**2
                )
                m = m.at[..., 25].set(
                    T[..., 25]
                    + 2 * T[..., 21] * ux
                    + 2 * T[..., 20] * uy
                    + T[..., 17] * uz
                    + T[..., 15] * ux**2
                    + T[..., 13] * uy**2
                    + 2 * T[..., 5] * ux * uy**2
                    + 2 * T[..., 6] * ux**2 * uy
                    + T[..., 8] * ux**2 * uz
                    + T[..., 7] * uy**2 * uz
                    + T[..., 3] * ux**2 * uy**2
                    + 4 * T[..., 16] * ux * uy
                    + 2 * T[..., 10] * ux * uz
                    + 2 * T[..., 12] * uy * uz
                    + 4 * T[..., 4] * ux * uy * uz
                    + 2 * T[..., 1] * ux * uy**2 * uz
                    + 2 * T[..., 2] * ux**2 * uy * uz
                    + T[..., 0] * ux**2 * uy**2 * uz
                )
                m = m.at[..., 26].set(
                    T[..., 0] * ux**2 * uy**2 * uz**2
                    + 2 * T[..., 3] * ux**2 * uy**2 * uz
                    + T[..., 9] * ux**2 * uy**2
                    + 2 * T[..., 2] * ux**2 * uy * uz**2
                    + 4 * T[..., 6] * ux**2 * uy * uz
                    + 2 * T[..., 14] * ux**2 * uy
                    + T[..., 8] * ux**2 * uz**2
                    + 2 * T[..., 15] * ux**2 * uz
                    + T[..., 19] * ux**2
                    + 2 * T[..., 1] * ux * uy**2 * uz**2
                    + 4 * T[..., 5] * ux * uy**2 * uz
                    + 2 * T[..., 11] * ux * uy**2
                    + 4 * T[..., 4] * ux * uy * uz**2
                    + 8 * T[..., 16] * ux * uy * uz
                    + 4 * T[..., 22] * ux * uy
                    + 2 * T[..., 10] * ux * uz**2
                    + 4 * T[..., 21] * ux * uz
                    + 2 * T[..., 23] * ux
                    + T[..., 7] * uy**2 * uz**2
                    + 2 * T[..., 13] * uy**2 * uz
                    + T[..., 18] * uy**2
                    + 2 * T[..., 12] * uy * uz**2
                    + 4 * T[..., 20] * uy * uz
                    + 2 * T[..., 24] * uy
                    + T[..., 17] * uz**2
                    + 2 * T[..., 25] * uz
                    + T[..., 26]
                )
                return m

            return shift_inverse(T, u)

    @partial(jit, static_argnums=(0,))
    def compute_eq_central_moments(self, rho):
        """
        Calculate the central moments of the equilibrium distribution.

        Parameters
        ----------
        rho : jax.numpy.ndarray
            Density field.

        Returns
        -------
        T_eq : jax.numpy.ndarray
            Central moments of the equilibrium distribution.
        """

        if isinstance(self.lattice, LatticeD2Q9):
            T_eq = jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precision_policy.compute_dtype)
            T_eq = T_eq.at[..., 0].set(rho[..., 0])
            T_eq = T_eq.at[..., 3].set(2 * rho[..., 0] * self.lattice.cs2)
            T_eq = T_eq.at[..., 8].set(rho[..., 0] * self.lattice.cs**4)

            return T_eq

        elif isinstance(self.lattice, LatticeD3Q19):
            T_eq = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype)
            T_eq = T_eq.at[..., 0].set(rho[..., 0])
            T_eq = T_eq.at[..., 7].set(rho[..., 0] * self.lattice.cs2)
            T_eq = T_eq.at[..., 8].set(rho[..., 0] * self.lattice.cs2)
            T_eq = T_eq.at[..., 9].set(rho[..., 0] * self.lattice.cs2)
            T_eq = T_eq.at[..., 16].set(rho[..., 0] * self.lattice.cs**4)
            T_eq = T_eq.at[..., 17].set(rho[..., 0] * self.lattice.cs**4)
            T_eq = T_eq.at[..., 18].set(rho[..., 0] * self.lattice.cs**4)

            return T_eq

        elif isinstance(self.lattice, LatticeD3Q27):
            T_eq = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype)
            T_eq = T_eq.at[..., 0].set(rho[..., 0])
            T_eq = T_eq.at[..., 7].set(rho[..., 0] * self.lattice.cs2)
            T_eq = T_eq.at[..., 8].set(rho[..., 0] * self.lattice.cs2)
            T_eq = T_eq.at[..., 9].set(rho[..., 0] * self.lattice.cs2)
            T_eq = T_eq.at[..., 17].set(rho[..., 0] * self.lattice.cs**4)
            T_eq = T_eq.at[..., 18].set(rho[..., 0] * self.lattice.cs**4)
            T_eq = T_eq.at[..., 19].set(rho[..., 0] * self.lattice.cs**4)
            T_eq = T_eq.at[..., 26].set(rho[..., 0] * self.lattice.cs**6)

            return T_eq

    @partial(jit, static_argnums=(0,))
    def compute_force_central_moments(self, F):
        """
        Calculate the central moments of the force distribution. Includes modification to accurately replicate mechanical stability conditions.

        Parameters
        ----------
        F : pytree of jax.numpy.ndarray
            Force field.

        Returns
        -------
        C : pytree of jax.numpy.ndarray
            Central moments of the force distribution.
        """

        if isinstance(self.lattice, LatticeD2Q9):
            C = jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precision_policy.compute_dtype)
            Fx = F[..., 0]
            Fy = F[..., 1]
            C = C.at[..., 1].set(Fx)
            C = C.at[..., 2].set(Fy)
            C = C.at[..., 6].set(Fy * self.lattice.cs2)
            C = C.at[..., 7].set(Fx * self.lattice.cs2)

            return C
        elif isinstance(self.lattice, LatticeD3Q19):
            C = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype)
            Fx = F[..., 0]
            Fy = F[..., 1]
            Fz = F[..., 2]
            C = C.at[..., 1].set(Fx)
            C = C.at[..., 2].set(Fy)
            C = C.at[..., 3].set(Fz)
            C = C.at[..., 10].set(Fx * self.lattice.cs2)
            C = C.at[..., 11].set(Fx * self.lattice.cs2)
            C = C.at[..., 12].set(Fy * self.lattice.cs2)
            C = C.at[..., 13].set(Fz * self.lattice.cs2)
            C = C.at[..., 14].set(Fy * self.lattice.cs2)
            C = C.at[..., 15].set(Fz * self.lattice.cs2)

            return C
        elif isinstance(self.lattice, LatticeD3Q27):
            C = jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype)
            Fx = F[..., 0]
            Fy = F[..., 1]
            Fz = F[..., 2]
            C = C.at[..., 1].set(Fx)
            C = C.at[..., 2].set(Fy)
            C = C.at[..., 3].set(Fz)
            C = C.at[..., 10].set(Fx * self.lattice.cs2)
            C = C.at[..., 11].set(Fx * self.lattice.cs2)
            C = C.at[..., 12].set(Fy * self.lattice.cs2)
            C = C.at[..., 13].set(Fz * self.lattice.cs2)
            C = C.at[..., 14].set(Fy * self.lattice.cs2)
            C = C.at[..., 15].set(Fz * self.lattice.cs2)
            C = C.at[..., 23].set(Fx * self.lattice.cs**4)
            C = C.at[..., 24].set(Fy * self.lattice.cs**4)
            C = C.at[..., 25].set(Fz * self.lattice.cs**4)

            return C

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, Tdash, rho, u):
        """
        Modified version of the apply_force defined in LBMBase to account for modified force.

        Parameters
        ----------
            Tdash (jax.numpy.ndarray): Central moments post-collision distribution functions.
            rho (jax.numpy.ndarray): Density field.
            u (jax.numpy.ndarray): Velocity field.

        Returns
        -------
            f_postcollision (jax.numpy.ndarray): Post-collision distribution functions with the force applied.
        """
        F = self.get_force()
        if F is None:
            F = jnp.zeros_like(u)
        C = self.compute_force_central_moments(F)
        Tf = jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * self.S)
        return Tdash + Tf

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin):
        """
        Cascaded LBM collision step for lattice.
        """
        fin = self.precision_policy.cast_to_compute(fin)
        rho, _ = self.update_macroscopic(fin)
        u = self.macroscopic_velocity(fin, rho)
        T = jnp.dot(fin, self.M)
        Tdash = self.compute_central_moment(T, u)
        Tdash_eq = self.compute_eq_central_moments(rho)
        Tout = jnp.dot(Tdash, jnp.eye(self.lattice.q) - self.S) + jnp.dot(Tdash_eq, self.S)
        Tout = self.apply_force(Tout, rho, u)
        Tout = self.compute_central_moment_inverse(Tout, u)
        fout = jnp.dot(T, self.M_inv)
        return self.precision_policy.cast_to_output(fout)
