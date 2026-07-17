"""
Utility class to convert between LBM (lattice) units and physical SI units for single and
multiphase, thermal and isothermal flows.

References
----------
1. Krüger, Timm, Halim Kusumaatmaja, Alexandr Kuzmin, Orest Shardt, Goncalo Silva, and Erlend Magnus Viggen. The Lattice Boltzmann Method: Principles and Practice.
Graduate Texts in Physics. Springer International Publishing, 2017. https://doi.org/10.1007/978-3-319-44649-3.
"""

from jax.tree import map as tree_map


class Unit:
    """
    Convert quantities between LBM (lattice) units and SI units.

    The conversion is defined by four independent scale factors (mass, length, time and
    temperature), each expressed as SI units per lattice unit. All derived quantities
    (density, velocity, pressure, force, viscosity) are converted using combinations of
    these factors. Conversion methods accept scalars, JAX/numpy arrays or pytrees of
    arrays, so both single phase fields and multiphase/multicomponent field trees
    (as used by the Multiphase class) are supported. The temperature scale is only
    relevant for thermal flows; isothermal flows can leave it at its default.

    Parameters
    ----------
    lbm_mass : float, optional
        Mass of one lattice mass unit in kg. Defaults to 1.0.
    lbm_length : float, optional
        Length of one lattice spacing in m. Defaults to 1.0.
    lbm_time : float, optional
        Duration of one lattice time step in s. Defaults to 1.0.
    lbm_temperature : float, optional
        Temperature of one lattice temperature unit in K. Defaults to 1.0.
    """

    def __init__(self, **kwargs):
        # LBM -> Physical scaling parameters; the Physical -> LBM parameters (si_mass,
        # si_length, si_time, si_temperature) are exposed as computed properties so they
        # can never go stale when a scale is updated.
        self.lbm_mass = kwargs.get("lbm_mass", 1.0)
        self.lbm_length = kwargs.get("lbm_length", 1.0)
        self.lbm_time = kwargs.get("lbm_time", 1.0)
        self.lbm_temperature = kwargs.get("lbm_temperature", 1.0)

    @property
    def lbm_mass(self):
        return self._lbm_mass

    @lbm_mass.setter
    def lbm_mass(self, value):
        if value <= 0:
            raise ValueError("lbm mass scale must be positive.")
        self._lbm_mass = value

    @property
    def lbm_length(self):
        return self._lbm_length

    @lbm_length.setter
    def lbm_length(self, value):
        if value <= 0:
            raise ValueError("lbm length scale must be positive.")
        self._lbm_length = value

    @property
    def lbm_time(self):
        return self._lbm_time

    @lbm_time.setter
    def lbm_time(self, value):
        if value <= 0:
            raise ValueError("lbm time scale must be positive.")
        self._lbm_time = value

    @property
    def lbm_temperature(self):
        return self._lbm_temperature

    @lbm_temperature.setter
    def lbm_temperature(self, value):
        if value <= 0:
            raise ValueError("lbm temperature scale must be positive.")
        self._lbm_temperature = value

    @property
    def si_mass(self):
        return 1.0 / self.lbm_mass

    @property
    def si_length(self):
        return 1.0 / self.lbm_length

    @property
    def si_time(self):
        return 1.0 / self.lbm_time

    @property
    def si_temperature(self):
        return 1.0 / self.lbm_temperature

    def _scale(self, value, factor):
        """
        Multiply a scalar, array or pytree of arrays by a scalar conversion factor.

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Quantity to be scaled.
        factor : float
            Conversion factor.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Scaled quantity with the same structure as value.
        """
        return tree_map(lambda leaf: leaf * factor, value)

    def to_si_length(self, value):
        """
        Convert a length from LBM units to SI units (m).

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Length in LBM units.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Length in SI units.
        """
        return self._scale(value, self.lbm_length)

    def to_lbm_length(self, value):
        """
        Convert a length from SI units (m) to LBM units.

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Length in SI units.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Length in LBM units.
        """
        return self._scale(value, 1.0 / self.lbm_length)

    def to_si_time(self, value):
        """
        Convert a time from LBM units to SI units (s).

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Time in LBM units.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Time in SI units.
        """
        return self._scale(value, self.lbm_time)

    def to_lbm_time(self, value):
        """
        Convert a time from SI units (s) to LBM units.

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Time in SI units.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Time in LBM units.
        """
        return self._scale(value, 1.0 / self.lbm_time)

    def to_si_density(self, value):
        """
        Convert a density field from LBM units to SI units (kg/m^3).

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Density field in LBM units; a pytree for multiphase flows.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Density field in SI units.
        """
        return self._scale(value, self.lbm_mass / self.lbm_length**3)

    def to_lbm_density(self, value):
        """
        Convert a density field from SI units (kg/m^3) to LBM units.

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Density field in SI units; a pytree for multiphase flows.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Density field in LBM units.
        """
        return self._scale(value, self.lbm_length**3 / self.lbm_mass)

    def to_si_velocity(self, value):
        """
        Convert a velocity field from LBM units to SI units (m/s).

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Velocity field in LBM units; a pytree for multiphase flows.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Velocity field in SI units.
        """
        return self._scale(value, self.lbm_length / self.lbm_time)

    def to_lbm_velocity(self, value):
        """
        Convert a velocity field from SI units (m/s) to LBM units.

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Velocity field in SI units; a pytree for multiphase flows.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Velocity field in LBM units.
        """
        return self._scale(value, self.lbm_time / self.lbm_length)

    def to_si_kinematic_viscosity(self, value):
        """
        Convert a kinematic viscosity from LBM units to SI units (m^2/s).

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Kinematic viscosity in LBM units.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Kinematic viscosity in SI units.
        """
        return self._scale(value, self.lbm_length**2 / self.lbm_time)

    def to_lbm_kinematic_viscosity(self, value):
        """
        Convert a kinematic viscosity from SI units (m^2/s) to LBM units.

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Kinematic viscosity in SI units.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Kinematic viscosity in LBM units.
        """
        return self._scale(value, self.lbm_time / self.lbm_length**2)

    def to_si_pressure(self, value):
        """
        Convert a pressure field from LBM units to SI units (Pa).

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Pressure field in LBM units; a pytree for multiphase flows.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Pressure field in SI units.
        """
        return self._scale(value, self.lbm_mass / (self.lbm_length * self.lbm_time**2))

    def to_lbm_pressure(self, value):
        """
        Convert a pressure field from SI units (Pa) to LBM units.

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Pressure field in SI units; a pytree for multiphase flows.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Pressure field in LBM units.
        """
        return self._scale(value, self.lbm_length * self.lbm_time**2 / self.lbm_mass)

    def to_si_force(self, value):
        """
        Convert a force field from LBM units to SI units (N).

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Force field in LBM units; a pytree for multiphase flows.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Force field in SI units.
        """
        return self._scale(value, self.lbm_mass * self.lbm_length / self.lbm_time**2)

    def to_lbm_force(self, value):
        """
        Convert a force field from SI units (N) to LBM units.

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Force field in SI units; a pytree for multiphase flows.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Force field in LBM units.
        """
        return self._scale(value, self.lbm_time**2 / (self.lbm_mass * self.lbm_length))

    def to_si_temperature(self, value):
        """
        Convert a temperature field from LBM units to SI units (K).

        Assumes the temperature scale was set (thermal flows); with the default scale
        of 1.0 the value is returned unchanged.

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Temperature field in LBM units.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Temperature field in SI units.
        """
        return self._scale(value, self.lbm_temperature)

    def to_lbm_temperature(self, value):
        """
        Convert a temperature field from SI units (K) to LBM units.

        Assumes the temperature scale was set (thermal flows); with the default scale
        of 1.0 the value is returned unchanged.

        Parameters
        ----------
        value : float or jax.Array or pytree of jax.Array
            Temperature field in SI units.

        Returns
        -------
        float or jax.Array or pytree of jax.Array
            Temperature field in LBM units.
        """
        return self._scale(value, 1.0 / self.lbm_temperature)

    def determine_lbm_scale_singlephase(
        self, si_kinematic_visc, si_density, lbm_kinematic_visc, lbm_density=1.0, lbm_length=None, lbm_time=None, lbm_mass=None
    ):
        """
        Determine the mass, length and time scales for a single phase flow.

        Matching kinematic viscosity and density between SI and LBM units fixes two of
        the three mechanical scales, so exactly one of lbm_length, lbm_time or lbm_mass
        must be provided to close the system. Assumes the LBM viscosity was chosen for
        the target relaxation time, nu = cs^2 * (tau - 1/2) (see Krüger et al., 2017).

        Parameters
        ----------
        si_kinematic_visc : float
            Kinematic viscosity in SI units (m^2/s).
        si_density : float
            Reference density in SI units (kg/m^3).
        lbm_kinematic_visc : float
            Kinematic viscosity in LBM units.
        lbm_density : float, optional
            Reference density in LBM units. Defaults to 1.0.
        lbm_length : float, optional
            Length of one lattice spacing in m.
        lbm_time : float, optional
            Duration of one lattice time step in s.
        lbm_mass : float, optional
            Mass of one lattice mass unit in kg.

        Returns
        -------
        None
            The mass, length and time scales of this instance are updated in place.
        """
        if sum(scale is not None for scale in (lbm_length, lbm_time, lbm_mass)) != 1:
            raise ValueError("Exactly one of lbm_length, lbm_time or lbm_mass must be provided.")

        visc_scale = si_kinematic_visc / lbm_kinematic_visc
        density_scale = si_density / lbm_density
        if lbm_length is not None:
            self.lbm_length = lbm_length
            self.lbm_time = lbm_length**2 / visc_scale
        elif lbm_time is not None:
            self.lbm_time = lbm_time
            self.lbm_length = (visc_scale * lbm_time) ** 0.5
        else:
            self.lbm_length = (lbm_mass / density_scale) ** (1.0 / 3.0)
            self.lbm_time = self.lbm_length**2 / visc_scale
        self.lbm_mass = density_scale * self.lbm_length**3

    def determine_lbm_scale_multiphase(
        self,
        si_kinematic_visc,
        si_density,
        lbm_kinematic_visc,
        lbm_density,
        si_temperature=None,
        lbm_temperature=None,
        lbm_length=None,
        lbm_time=None,
        lbm_mass=None,
    ):
        """
        Determine the mass, length, time and temperature scales for a multiphase flow.

        Same closure as determine_lbm_scale_singlephase, but the LBM reference density
        is fixed by the equation of state (typically the critical or coexistence liquid
        density) and must be provided. For thermal flows, matching a reference
        temperature (typically the critical temperature of the EOS) sets the
        temperature scale; isothermal flows can omit it (see Yuan and Schaefer, 2006,
        https://doi.org/10.1063/1.2187070).

        Parameters
        ----------
        si_kinematic_visc : float
            Kinematic viscosity in SI units (m^2/s).
        si_density : float
            Reference density in SI units (kg/m^3).
        lbm_kinematic_visc : float
            Kinematic viscosity in LBM units.
        lbm_density : float
            Reference density in LBM units, as given by the EOS.
        si_temperature : float, optional
            Reference temperature in SI units (K); required together with
            lbm_temperature for thermal flows.
        lbm_temperature : float, optional
            Reference temperature in LBM units, as given by the EOS.
        lbm_length : float, optional
            Length of one lattice spacing in m.
        lbm_time : float, optional
            Duration of one lattice time step in s.
        lbm_mass : float, optional
            Mass of one lattice mass unit in kg.

        Returns
        -------
        None
            The mass, length, time and (for thermal flows) temperature scales of this
            instance are updated in place.
        """
        self.determine_lbm_scale_singlephase(
            si_kinematic_visc, si_density, lbm_kinematic_visc, lbm_density, lbm_length=lbm_length, lbm_time=lbm_time, lbm_mass=lbm_mass
        )
        if (si_temperature is None) != (lbm_temperature is None):
            raise ValueError("si_temperature and lbm_temperature must be provided together.")
        if si_temperature is not None:
            self.lbm_temperature = si_temperature / lbm_temperature
