"""
An example density profile: a time-dependent Plummer profile for 47 Tuc
"""

from __future__ import annotations

import numpy as np

from hjmodel.clusters import DensityProfile, LocalEnvironment
from hjmodel.config import AU_PER_PSC, G

__all__ = ["Plummer"]

_N0: float = 2.0e6
_R0: float = 1.91
_A: float = 6.991e-4
_M_AVG: float = 0.8
_M0: float = 1.64e6
_M1: float = 0.9e6
_T_AGE: float = 12000.0


_A_OVER_RH = 0.766


class Plummer(DensityProfile):
    """
    Time-dependent Plummer density profile for globular cluster 47 Tucanae.

    Implements a Plummer sphere model with time-evolving half-mass radius
    and total mass, calibrated to observations of 47 Tuc.

    Attributes:
        N0: Initial number of stars.
        R0: Initial half-mass radius (pc).
        A: Expansion rate coefficient (pc^1.5 / Myr).
        M_avg: Average stellar mass (M_sun).
        M0: Initial total mass (M_sun).
        M1: Final total mass at T_AGE (M_sun).
        T_AGE: Age of the cluster (Myr).
        M_fixed: Fixed mass for density calculations (M_sun).
    """

    def __init__(
        self,
        N0: float = _N0,
        R0: float = _R0,
        A: float = _A,
        M_avg: float = _M_AVG,
        M0: float = _M0,
        M1: float = _M1,
        T_AGE: float = _T_AGE,
    ):
        """
        Initialize the Plummer profile with 47 Tuc parameters.

        Args:
            N0: Initial number of stars. Default: 2.0e6.
            R0: Initial half-mass radius in pc. Default: 1.91.
            A: Expansion rate coefficient in pc^1.5/Myr. Default: 6.991e-4.
            M_avg: Average stellar mass in M_sun. Default: 0.8.
            M0: Initial total mass in M_sun. Default: 1.64e6.
            M1: Final total mass at T_AGE in M_sun. Default: 0.9e6.
            T_AGE: Age of the cluster in Myr. Default: 12000.
        """
        self.N0 = N0
        self.R0 = R0
        self.A = A
        self.M_avg = M_avg
        self.M0 = M0
        self.M1 = M1
        self.T_AGE = T_AGE
        self.M_fixed = N0 * M_avg

    def rh(self, t: float) -> float:
        """
        Compute the half-mass radius at a given time.

        Args:
            t: Time since cluster formation (Myr).

        Returns:
            Half-mass radius (pc).
        """
        return (self.R0**1.5 + self.A * t) ** (2 / 3)

    def a(self, t: float) -> float:
        """
        Compute the Plummer scale radius at a given time.

        Args:
            t: Time since cluster formation (Myr).

        Returns:
            Scale radius (pc).
        """
        return _A_OVER_RH * self.rh(t)

    def M_t(self, t: float) -> float:
        """
        Compute the total cluster mass at a given time.

        Mass evolves linearly from M0 to M1 over T_AGE.

        Args:
            t: Time since cluster formation (Myr).

        Returns:
            Total mass (M_sun).
        """
        return self.M0 + (self.M1 - self.M0) * (t / self.T_AGE)

    def get_number_density(self, r: float, t: float) -> float:
        """
        Compute the stellar number density at a given radius and time.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            Number density (stars per pc^3 per 10^6).
        """
        a_t = self.a(t)
        rho = (3 * self.M_fixed) / (4 * np.pi * a_t**3) * (1 + (r / a_t) ** 2) ** -2.5
        return rho / self.M_avg / 1e6

    def get_isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        """
        Compute the isotropic velocity dispersion at a given radius and time.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            Velocity dispersion (au/yr).
        """
        r_scaled = r * AU_PER_PSC
        a_scaled = self.a(t) * AU_PER_PSC
        return np.sqrt(G * self.M_t(t) / (6 * np.sqrt(r_scaled**2 + a_scaled**2)))

    def get_mass_fraction_within_radius(self, r: float, t: float) -> float:
        """
        Compute the mass fraction enclosed within a given radius.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            Mass fraction enclosed (0 to 1).
        """
        a_t = self.a(t)
        r_scaled = r / a_t
        return (r_scaled**3) / ((1 + r_scaled**2) ** 1.5)

    def get_radius(self, lagrange: float, t: float) -> float:
        """
        Compute the radius enclosing a given Lagrangian mass fraction.

        Args:
            lagrange: Lagrangian mass fraction (0 < lagrange < 1).
            t: Time since cluster formation (Myr).

        Returns:
            Radius enclosing the specified mass fraction (pc).
        """
        a_t = self.a(t)
        r_scaled = (lagrange ** (2 / 3)) / (1 - lagrange ** (2 / 3))
        r_scaled = r_scaled**0.5
        return r_scaled * a_t

    def get_local_environment(self, r: float, t: float) -> LocalEnvironment:
        """
        Get the local environment properties at a given radius and time.

        Optimized version that computes density and velocity dispersion
        together to avoid redundant calculations.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            LocalEnvironment containing number density and velocity dispersion.
        """
        a_t = self.a(t)

        rho = (3 * self.M_fixed) / (4 * np.pi * a_t**3) * (1 + (r / a_t) ** 2) ** -2.5
        n_tot = rho / self.M_avg / 1e6

        r_scaled = r * AU_PER_PSC
        a_scaled = a_t * AU_PER_PSC
        sigma_v = np.sqrt(G * self.M_t(t) / (6 * np.sqrt(r_scaled**2 + a_scaled**2)))

        return LocalEnvironment(n_tot=n_tot, sigma_v=sigma_v)
