def test_a_init():
    import numpy as np
    import matplotlib.pyplot as plt

    import numpy as np
    import matplotlib.pyplot as plt

    # Constants for the broken power-law
    A_MIN = 1.0
    A_BR = 2.5
    A_MAX = 30.0
    s1 = 0.8
    s2 = -1.83

    def sample_a_init():
        def sample_segment(u, a1, a2, alpha):
            if alpha == -1:
                return a1 * (a2 / a1) ** u
            else:
                p = alpha + 1
                return (a1 ** p + u * (a2 ** p - a1 ** p)) ** (1 / p)

        def integral(a1, a2, alpha):
            return np.log(a2 / a1) if alpha == -1 else (a2 ** (alpha + 1) - a1 ** (alpha + 1)) / (alpha + 1)

        # Continuity scaling
        c2_over_c1 = A_BR ** (s1 - s2)

        # Weighted integrals for segment probabilities
        I1 = integral(A_MIN, A_BR, s1)
        I2 = c2_over_c1 * integral(A_BR, A_MAX, s2)
        prob1 = I1 / (I1 + I2)

        # Sample
        u = np.random.rand()
        if u < prob1:
            return sample_segment(np.random.rand(), A_MIN, A_BR, s1)
        else:
            return sample_segment(np.random.rand(), A_BR, A_MAX, s2)

    # Generate samples
    samples = np.array([sample_a_init() for _ in range(100000)])

    # Plot histogram in log-space
    log_bins = np.logspace(np.log10(A_MIN), np.log10(A_MAX), 100)
    hist, bins = np.histogram(samples, bins=log_bins, density=True)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.loglog(bin_centers, hist, label='Sampled Histogram', drawstyle='steps-mid')

    # Overlay expected shape
    norm = hist[0] / (bin_centers[0] ** s1)  # Match normalization to first bin
    expected = np.piecewise(
        bin_centers,
        [bin_centers <= A_BR, bin_centers > A_BR],
        [lambda a: norm * a ** s1,
         lambda a: norm * A_BR ** (s1 - s2) * a ** s2]
    )
    plt.loglog(bin_centers, expected, '--', label='Expected dN/dlog(a) ∝ a^s', linewidth=2)

    plt.xlabel('a')
    plt.ylabel('dN / dlog(a)')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.title('Broken Power-Law Sampling Verification')
    plt.show()



if __name__ == "__main__":
    test_a_init()