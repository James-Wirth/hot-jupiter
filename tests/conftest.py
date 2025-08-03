import numpy as np
import pytest


class DummyCluster:
    def get_lagrange_distribution(self, n_samples: int, t: float):
        return np.ones(n_samples) * 1.0

    def get_radius(self, lagrange: float, t: float):
        return 1.0

    def get_local_environment(self, r: float, t: float):
        return {
            "sigma_v": 1.0,
            "local_n_tot": 1.0,
            "local_sigma_v": 1.0,
        }


@pytest.fixture
def dummy_cluster():
    return DummyCluster()


@pytest.fixture
def rng():
    return np.random.default_rng(42)
