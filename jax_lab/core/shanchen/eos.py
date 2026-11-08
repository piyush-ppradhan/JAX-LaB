from functools import partial

from jax import jit
from jax.tree import map as tree_map

import jax.numpy as jnp


class EOS:
    """
    Base class for equations of state.

    Parameters
    ----------
    temperature_field_type (str, optional): ``"isothermal"`` for a fixed temperature or ``"thermal"`` for a temperature
        field. Defaults to ``"isothermal"``.

    a, b (float or list of float): Equation-specific parameters for each component.

    R (float or list of float): Gas constant for each component.

    T (float, optional): Fixed temperature required for isothermal calculations.
    """

    def __init__(self, temperature_field_type="isothermal", **kwargs):
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.R = kwargs.get("R")
        self.temperature_field_type = temperature_field_type
        if self.temperature_field_type == "isothermal":
            self.T = kwargs.get("T")

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        if value is None:
            raise ValueError("Gas constant value must be provided")
        if isinstance(value, float) or isinstance(value, int):
            self._R = [value]
        elif isinstance(value, list):
            self._R = value
        else:
            raise ValueError("Gas constant must be an int, float, or list for multicomponent flows")

    @property
    def temperature_field_type(self):
        return self._temperature_field_type

    @temperature_field_type.setter
    def temperature_field_type(self, value):
        if value in ["isothermal", "thermal"]:
            self._temperature_field_type = value
        else:
            raise ValueError("Invalid temperature_field_type, it can only be 'isothermal' or 'thermal'")

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if value is None and self.temperature_field_type == "isothermal":
            raise ValueError("Temperature value must be provided for isothermal case")
        if self.temperature_field_type == "isothermal" and value < 0:
            raise ValueError("Temperature cannot be negative")
        self._T = value

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("EOS parameter a must be provided")
        if isinstance(value, float) or isinstance(value, int):
            self._a = [value]
        elif isinstance(value, list):
            self._a = value
        else:
            raise ValueError("EOS parameter a must be int, float or a list (for multi-component flows)")

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("EOS parameter b must be provided")
        if isinstance(value, float) or isinstance(value, int):
            self._b = [value]
        elif isinstance(value, list):
            self._b = value
        else:
            raise ValueError("EOS parameter b must be int, float or a list (for multi-component flows)")

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        """
        Evaluate the isothermal equation of state.

        Parameters
        ----------
        rho_tree (pytree of jax.Array): Component density fields.

        Returns
        -------
        pytree of jax.Array
            Component pressure fields.
        """
        pass

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        """
        Evaluate the equation of state with a temperature field.

        Parameters
        ----------
        rho_tree (pytree of jax.Array): Component density fields.

        T (jax.Array): Temperature field.

        Returns
        -------
        pytree of jax.Array
            Component pressure fields.
        """
        pass

    @partial(jit, static_argnums=(0,), inline=True)
    def dp_eos_dT(self, rho_tree, T):
        """
        Evaluate the pressure derivative with respect to temperature.

        Parameters
        ----------
        rho_tree (pytree of jax.Array): Component density fields.

        T (jax.Array): Temperature field.

        Returns
        -------
        pytree of jax.Array
            Component pressure derivatives.
        """
        pass


class VanderWaals(EOS):
    """
    Define a multiphase model using the VanderWaals EOS.

    Parameters
    ----------
    a (list of float): Attraction parameter for each component.

    b (list of float): Excluded-volume parameter for each component.

    R (list of float): Gas constant for each component.

    T (float or jax.Array): Temperature used by the isothermal equation of state.

    References
    ----------
    1. Reprint of: The Equation of State for Gases and Liquids. The Journal of Supercritical Fluids,
    100th year Anniversary of van der Waals' Nobel Lecture, 55, no. 2 (2010): 403-14. https://doi.org/10.1016/j.supflu.2010.11.001.

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - a*rho^2
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        eos = lambda a, b, R, rho: (rho * R * self.T) / (1.0 - b * rho) - a * rho**2
        return tree_map(eos, self.a, self.b, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        eos = lambda a, b, R, rho: (rho * R * T) / (1.0 - b * rho) - a * rho**2
        return tree_map(eos, self.a, self.b, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def dp_eos_dT(self, rho_tree, T):
        dpeos_dT = lambda b, R, rho: (rho * R) / (1.0 - b * rho)
        return tree_map(lambda b, R, rho: dpeos_dT(b, R, rho), self.b, self.R, rho_tree)


class RedlichKwong(EOS):
    """
    Define multiphase model using the Redlich-Kwong EOS.

    Parameters
    ----------
    a, b, R (list of float): Equation parameters for each component.

    T (float or jax.Array): Temperature used by the isothermal equation of state.

    References
    ----------
    1. Redlich O., Kwong JN., "On the thermodynamics of solutions; an equation of state; fugacities of gaseous solutions."
    Chem Rev. 1949 Feb;44(1):233-44. https://doi.org/10.1021/cr60137a013.

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - (a*rho^2)/(sqrt(T) * (1 + b*rho))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        eos = lambda a, b, R, rho: (rho * R * self.T) / (1.0 - b * rho) - (a * rho**2) / (jnp.sqrt(self.T) * (1.0 + b * rho))
        return tree_map(eos, self.a, self.b, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        eos = lambda a, b, R, rho: (rho * R * T) / (1.0 - b * rho) - (a * rho**2) / (jnp.sqrt(T) * (1.0 + b * rho))
        return tree_map(eos, self.a, self.b, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def dp_eos_dT(self, rho_tree, T):
        dpeos_dT = lambda a, b, R, rho: (rho * R) / (1.0 - b * rho) + (0.5 * a * rho**2) / ((T**1.5) * (1.0 + b * rho))
        return tree_map(lambda a, b, R, rho: dpeos_dT(a, b, R, rho), self.a, self.b, self.R, rho_tree)


class RedlichKwongSoave(EOS):
    """
    Define multiphase model using the Redlich-Kwong-Soave EOS.

    Parameters
    ----------
    a, b, R (list of float): Equation parameters for each component.

    T (float or jax.Array): Temperature used by the isothermal equation of state.

    RKS_omega (list of float): Acentric factor for each component.

    References
    ----------
    1. Giorgio Soave, "Equilibrium constants from a modified Redlich-Kwong equation of state",
    Chemical Engineering Science 27, no. 6(1972), 1197-1203, https://doi.org/10.1016/0009-2509(72)80096-4.

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + b*rho)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rks_omega = kwargs.get("RKS_omega")
        self.Tc_tree = tree_map(lambda a, b, R: (a / b) * (0.08664 / 0.42784) / R, self.a, self.b, self.R)
        self.set_alpha()

    @property
    def rks_omega(self):
        return self._rks_omega

    @rks_omega.setter
    def rks_omega(self, value):
        if value is None:
            raise ValueError("rks_omega value must be provided for using Redlich-Kwong EOS")
        self._rks_omega = value

    def set_alpha(self):
        if self.temperature_field_type == "isothermal":
            self.alpha = tree_map(
                lambda rks_omega, Tc: (1 + (0.480 + 1.574 * rks_omega - 0.176 * rks_omega**2) * (1 - jnp.sqrt(self.T / Tc))) ** 2,
                self.rks_omega,
                self.Tc_tree,
            )

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        eos = lambda a, b, alpha, R, rho: (rho * R * self.T) / (1.0 - b * rho) - (a * alpha * rho**2) / (1.0 + b * rho)
        return tree_map(eos, self.a, self.b, self.alpha, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        alpha_tree = tree_map(
            lambda rks_omega, Tc: (1 + (0.480 + 1.574 * rks_omega - 0.176 * rks_omega**2) * (1 - jnp.sqrt(T / Tc))) ** 2, self.rks_omega, self.Tc_tree
        )
        eos = lambda a, b, alpha, R, rho: (rho * R * T) / (1.0 - b * rho) - (a * alpha * rho**2) / (1.0 + b * rho)
        return tree_map(eos, self.a, self.b, alpha_tree, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def dp_eos_dT(self, rho_tree, T):
        dalpha_dT = lambda rks_omega, Tc: (
            -1.0
            * (T / Tc) ** 0.5
            * ((1 - (T / Tc) ** 0.5) * (-0.176 * rks_omega**2 + 1.574 * rks_omega + 0.48) + 1)
            * (-0.176 * rks_omega**2 + 1.574 * rks_omega + 0.48)
            / T
        )
        dpeos_dT = lambda a, b, R, rho, rks_omega, Tc: (rho * R) / (1.0 - b * rho) - (a * dalpha_dT(rks_omega, Tc) * rho**2) / (1.0 + b * rho)
        return tree_map(
            lambda a, b, R, rho, rks_omega, Tc: dpeos_dT(a, b, R, rho, rks_omega, Tc), self.a, self.b, self.R, rho_tree, self.rks_omega, self.Tc_tree
        )


class PengRobinson(EOS):
    """
    Define multiphase model using the Peng-Robinson EOS.

    Parameters
    ----------
    a, b, R (list of float): Equation parameters for each component.

    T (float or jax.Array): Temperature used by the isothermal equation of state.

    pr_omega (float or list of float): Acentric factor for each component.

    References
    ----------
    1. Peng, Ding-Yu, and Donald B. Robinson. "A new two-constant equation of state."
    Industrial & Engineering Chemistry Fundamentals 15, no. 1 (1976): 59-64. https://doi.org/10.1021/i160057a011

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + 2*b*rho - (b*rho)**2)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pr_omega = kwargs.get("pr_omega")
        self.Tc_tree = tree_map(lambda a, b, R: (a / b) * (0.0778 / 0.45724) / R, self.a, self.b, self.R)
        self.set_alpha()

    @property
    def pr_omega(self):
        return self._pr_omega

    @pr_omega.setter
    def pr_omega(self, value):
        if value is None:
            raise ValueError("pr_omega value must be provided for using Peng-Robinson EOS")
        if isinstance(value, int) or isinstance(value, float):
            self._pr_omega = [value]
        if isinstance(value, list):
            self._pr_omega = value

    def set_alpha(self):
        if self.temperature_field_type == "isothermal":
            self.alpha = tree_map(
                lambda pr_omega, Tc: (1 + (0.37464 + 1.54226 * pr_omega - 0.26992 * pr_omega**2) * (1 - jnp.sqrt(self.T / Tc))) ** 2,
                self.pr_omega,
                self.Tc_tree,
            )

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        eos = lambda a, b, alpha, R, rho: (rho * R * self.T) / (1.0 - b * rho) - (a * alpha * rho**2) / (1.0 + 2 * b * rho - b**2 * rho**2)
        return tree_map(eos, self.a, self.b, self.alpha, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        alpha_tree = tree_map(
            lambda pr_omega, Tc: (1 + (0.37464 + 1.54226 * pr_omega - 0.26992 * pr_omega**2) * (1 - jnp.sqrt(T / Tc))) ** 2,
            self.pr_omega,
            self.Tc_tree,
        )
        eos = lambda a, b, alpha, R, rho: (rho * R * T) / (1.0 - b * rho) - (a * alpha * rho**2) / (1.0 + 2 * b * rho - b**2 * rho**2)
        return tree_map(eos, self.a, self.b, alpha_tree, self.R, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def dp_eos_dT(self, rho_tree, T):
        dalpha_dT = lambda pr_omega, Tc: (
            -1.0
            * (T / Tc) ** 0.5
            * ((1 - (T / Tc) ** 0.5) * (-0.26992 * pr_omega**2 + 1.54226 * pr_omega + 0.37464) + 1)
            * (-0.26992 * pr_omega**2 + 1.54226 * pr_omega + 0.37464)
            / T
        )
        dpeos_dT = lambda a, b, R, rho, pr_omega, Tc: (
            (rho * R) / (1.0 - b * rho) - (a * dalpha_dT(pr_omega, Tc) * rho**2) / (1.0 + 2 * b * rho - b**2 * rho**2)
        )
        return tree_map(
            lambda a, b, R, rho, pr_omega, Tc: dpeos_dT(a, b, R, rho, pr_omega, Tc), self.a, self.b, self.R, rho_tree, self.pr_omega, self.Tc_tree
        )


class CarnahanStarling(EOS):
    """
    Define multiphase model using the Carnahan-Starling EOS.

    Parameters
    ----------
    a, b, R (list of float): Equation parameters for each component.

    T (float or jax.Array): Temperature used by the isothermal equation of state.

    References
    ----------
    1.  Carnahan, Norman F., and Kenneth E. Starling. "Equation of state for nonattracting rigid spheres."
    The Journal of chemical physics 51, no. 2 (1969): 635-636. https://doi.org/10.1063/1.1672048

    Notes
    -----
    EOS is given by:
        p = (rho*R*T)*(1 + (0.25*b*rho) + (0.25*b*rho)^2 - (0.25*b*rho)^3)/(1 - b*rho)^3 - (a*rho^2)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS(self, rho_tree):
        x_tree = tree_map(lambda b, rho: 0.25 * b * rho, self.b, rho_tree)
        eos = lambda a, R, rho, x: (rho * R * self.T * (1.0 + x + x**2 - x**3) / ((1.0 - x) ** 3)) - (a * rho**2)
        return tree_map(eos, self.a, self.R, rho_tree, x_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def EOS_thermal(self, rho_tree, T):
        x_tree = tree_map(lambda b, rho: 0.25 * b * rho, self.b, rho_tree)
        eos = lambda a, R, rho, x: (rho * R * T * (1.0 + x + x**2 - x**3) / ((1.0 - x) ** 3)) - (a * rho**2)
        return tree_map(eos, self.a, self.R, rho_tree, x_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def dp_eos_dT(self, rho_tree, T):
        x_tree = tree_map(lambda b, rho: 0.25 * b * rho, self.b, rho_tree)
        dpeos_dT = lambda x, R, rho: (rho * R) * (1.0 + x + x**2 - x**3) / ((1.0 - x) ** 3)
        return tree_map(lambda x, R, rho: dpeos_dT(x, R, rho), x_tree, self.R, rho_tree)
