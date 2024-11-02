from hjmodel.model_utils import get_true_anomaly, true_anomaly_approximation
import numpy as np
import matplotlib.pyplot as plt

def test_kepler_approx():
    e = np.linspace(0, 1, 500)
    M = 0.424
    model_vals = np.vectorize(get_true_anomaly)(M, e)
    approx_vals = np.vectorize(true_anomaly_approximation)(M, e)

    plt.plot(e, model_vals)
    plt.plot(e, approx_vals)

    plt.show()
