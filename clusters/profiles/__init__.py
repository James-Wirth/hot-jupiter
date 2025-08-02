from abc import ABC, abstractmethod

class DensityProfile(ABC):
    @abstractmethod
    def get_number_density(self, r: float, t: float) -> float: pass

    @abstractmethod
    def get_isotropic_velocity_dispersion(self, r: float, t: float) -> float: pass

    @abstractmethod
    def get_radius(self, lagrange: float, t: float) -> float: pass

    @abstractmethod
    def get_mass_fraction_within_radius(self, r: float, t: float) -> float: pass
