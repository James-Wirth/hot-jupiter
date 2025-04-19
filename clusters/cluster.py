import numpy as np

class Cluster:
    """
    Abstract base class for star cluster models.
    """
    def __init__(self):
        pass

    def number_density(self, r: float, t: float) -> float:
        """Number density profile at radius r and time t."""
        raise NotImplementedError

    def isotropic_velocity_dispersion(self, r: float, t: float) -> float:
        """Isotropic velocity dispersion at radius r and time t."""
        raise NotImplementedError

    def env_vars(self, r: float, t: float) -> dict[str, float]:
        """Returns environmental variables like density and velocity dispersion."""
        raise NotImplementedError

    def get_lagrange_distribution(self, n_samples: int, t: float, seed=None) -> np.ndarray:
        """Returns uniform samples in CDF space."""
        raise NotImplementedError

    def get_radius(self, lagrange: float, t: float) -> float:
        """Inverse CDF to convert lagrange multiplier to radius."""
        raise NotImplementedError
