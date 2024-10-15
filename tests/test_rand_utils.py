from hjmodel import rand_utils as ru
from hjmodel.config import *
import matplotlib.pyplot as plt
import numpy as np

NUM_SAMPLES = 10000

def test_b():
    b_sample = [ru.rand_b() for _ in range(NUM_SAMPLES)]
    plt.hist(b_sample, bins=int(np.sqrt(NUM_SAMPLES)), density=True)

    b_lin = np.linspace(0, B_MAX, 100)
    plt.plot(b_lin, 2 * b_lin / B_MAX**2)
    plt.show()