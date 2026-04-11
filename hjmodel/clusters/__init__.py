from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np

__all__ = ["LocalEnvironment", "DensityProfile", "Cluster"]


LocalEnvironment = namedtuple("LocalEnvironment", ["n_tot", "sigma_v"])
"""
Container for local cluster environment properties.

Attributes:
    n_tot: Total number density at the given position (stars per pc^3 per 10^6).
    sigma_v: Isotropic velocity dispersion at the given position (au/yr).
"""


class DensityProfile(ABC):
    """
    Abstract base class for cluster density profiles.

    Subclasses must implement methods for computing number density,
    velocity dispersion, radius from Lagrangian mass fraction, and
    mass fraction within a given radius.
    """

    @abstractmethod
    def get_number_density(self, r: float, t: float) -> float:
        """
        Compute the stellar number density at a given radius and time.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            Number density (stars per pc^3 per 10^6).
        """
        pass

    @abstractmethod
    def get_isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        """
        Compute the isotropic velocity dispersion at a given radius and time.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            Velocity dispersion (au/yr).
        """
        pass

    @abstractmethod
    def get_radius(self, lagrange: float, t: float) -> float:
        """
        Compute the radius enclosing a given Lagrangian mass fraction.

        Args:
            lagrange: Lagrangian mass fraction (0 < lagrange < 1).
            t: Time since cluster formation (Myr).

        Returns:
            Radius enclosing the specified mass fraction (pc).
        """
        pass

    @abstractmethod
    def get_mass_fraction_within_radius(self, r: float, t: float) -> float:
        """
        Compute the mass fraction enclosed within a given radius.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            Mass fraction enclosed (0 to 1).
        """
        pass


class Cluster:
    """
    Wrapper class providing a unified interface to cluster density profiles.

    Delegates density profile computations to the underlying DensityProfile
    implementation while adding validation and convenience methods.

    Attributes:
        profile: The underlying density profile implementation.
        r_max: Maximum radius for sampling (pc).
    """

    def __init__(self, profile: DensityProfile, r_max: float = 100):
        """
        Initialize a Cluster with a density profile.

        Args:
            profile: A DensityProfile implementation defining the cluster structure.
            r_max: Maximum radius to consider for sampling distributions (pc).
        """
        self.profile = profile
        self.r_max = r_max

    def get_number_density(self, r: float, t: float) -> float:
        """
        Get the stellar number density at a given radius and time.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            Number density (stars per pc^3 per 10^6).
        """
        return self.profile.get_number_density(r, t)

    def get_isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        """
        Get the isotropic velocity dispersion at a given radius and time.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            Velocity dispersion (au/yr).
        """
        return self.profile.get_isotropic_velocity_dispersion(r, t)

    def get_mass_fraction_within_radius(self, r: float, t: float) -> float:
        """
        Get the mass fraction enclosed within a given radius.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            Mass fraction enclosed (0 to 1).
        """
        return self.profile.get_mass_fraction_within_radius(r, t)

    def get_radius(self, lagrange: float, t: float) -> float:
        """
        Get the radius enclosing a given Lagrangian mass fraction.

        Args:
            lagrange: Lagrangian mass fraction (0 < lagrange < 1).
            t: Time since cluster formation (Myr).

        Returns:
            Radius enclosing the specified mass fraction (pc).

        Raises:
            ValueError: If lagrange is not in the range (0, 1).
        """
        if not (0 < lagrange < 1):
            raise ValueError(
                "Lagrange mass fraction must be between 0 and 1 (exclusive)"
            )
        return self.profile.get_radius(lagrange, t)

    def get_local_environment(self, r: float, t: float) -> LocalEnvironment:
        """
        Get the local environment properties at a given radius and time.

        Args:
            r: Radial distance from cluster center (pc).
            t: Time since cluster formation (Myr).

        Returns:
            LocalEnvironment containing number density and velocity dispersion.
        """
        if hasattr(self.profile, "get_local_environment"):
            return self.profile.get_local_environment(r, t)
        return LocalEnvironment(
            n_tot=self.get_number_density(r, t),
            sigma_v=self.get_isotropic_velocity_dispersion(r, t),
        )

    def get_lagrange_distribution(
        self, n_samples: int, t: float, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """
        Sample Lagrangian mass fractions uniformly across the cluster.

        Generates a stratified sample of Lagrangian radii by dividing
        the cumulative mass distribution into equal bins and sampling
        uniformly within each bin.

        Args:
            n_samples: Number of samples to generate.
            t: Time since cluster formation (Myr).
            rng: Random number generator. If None, creates a new default generator.

        Returns:
            Array of Lagrangian mass fractions.

        Raises:
            ValueError: If the CDF at r_max is invalid.
        """
        if rng is None:
            rng = np.random.default_rng()

        cdf_max = self.profile.get_mass_fraction_within_radius(self.r_max, t)
        if not (0 < cdf_max <= 1):
            raise ValueError(
                f"Invalid CDF value {cdf_max:.3f} returned for r_max={self.r_max} at time t={t}"
            )

        bins = np.linspace(0, cdf_max, n_samples + 1)
        return rng.uniform(bins[:-1], bins[1:])
