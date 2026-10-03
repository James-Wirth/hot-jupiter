# hot-jupiter

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
![CI](https://github.com/James-Wirth/hot-jupiter/actions/workflows/ci.yml/badge.svg)

> This repository accompanies the paper:  
> **"Hot Jupiter formation in dense stellar clusters: A Monte Carlo model applied to 47 Tucanae"**  
> James A. Wirth, Cathie J. Clarke, Andrew J. Winter. *Monthly Notices of the Royal Astronomical Society, 2025.*<br>
> [DOI](https://doi.org/10.1093/mnras/staf1325) | [arXiv](https://arxiv.org/abs/2508.08406)

**hot-jupiter** is a Monte Carlo simulation package for studying Hot Jupiter formation in dense globular clusters via high-eccentricity migration. We use the [REBOUND](https://github.com/hannorein/rebound) code with the IAS15 numerical integrator to follow planetary systems perturbed by stellar encounters over Gyr timescales. The paper applies this model to 47 Tucanae, asking how efficiently stellar encounters can turn cold Jupiter progenitors into Hot Jupiters and whether the resulting occurrence rate is consistent with transit-survey non-detections.

The eccentricity of a planetary system can be perturbed by stellar flybys. At high eccentricity, the planet may pass close enough to its host star for tidal torques to rapidly circularise the orbit, leading to the formation of a Hot Jupiter. Analytic expressions for the eccentricity excitation were derived by [Heggie & Rasio (1996), *The effect of encounters on the eccentricity of binaries in clusters*](https://doi.org/10.1093/mnras/282.3.1064), but these hold only in the tidal and slow regime, neglecting terms higher than quadrupole in the multipole expansion for the perturbing force. We have introduced an efficient hybrid scheme for computing the eccentricity diffusion, whereby direct N-body simulations of the encounter are performed only in regimes where the analytic expressions are invalid.

Applied to cold Jupiter progenitors at initial separations of 1–30 au over 12 Gyr, the hybrid model in the [paper](https://arxiv.org/html/2508.08406v1#S4) gives an HJ occurrence rate of approximately $5.9 \times 10^{-4}$ per cluster star, assuming a 10% initial cold Jupiter occurrence rate. This is a 51% enhancement relative to the analytic Monte Carlo baseline, while remaining consistent with the observational upper limits discussed in the paper. HJ formation is concentrated towards the cluster core and falls steeply beyond a few parsecs.

<br>
<p align="center">
  <img src="https://github.com/user-attachments/files/21801314/show_paths_mnras.pdf" alt="Simulation Results" width="80%" />
  <br />
  <em>Example: The phase-space paths for a sample of Hot Jupiter (HJ), Warm Jupiter (WJ), Tidal Disruption (TD) and No-Migration (NM) outcomes. </em>
</p>
<br>

The final states of planetary systems are categorised into five unique outcomes: Ionisation (ION), Tidal Disruption (TD), Hot Jupiter formation (HJ), Warm Jupiter formation (WJ) and No Migration (NM).

## Installation

Use Python 3.11, as in the CI environment. Clone the repository, install the pinned dependencies, and install the package locally (the Python import name is `hj`):

```bash
git clone https://github.com/James-Wirth/hot-jupiter.git
cd hot-jupiter
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

The supplied `Plummer` profile describes the time-dependent cluster background used for 47 Tuc. The following example evolves 1,000 planetary systems for 12 Gyr using the hybrid encounter model:

```python
from hj import HJModel, Plummer

cluster = Plummer()
model = HJModel(name="47tuc_hybrid", base_dir="data")
model.run(
    time=12_000.0,  # Myr
    num_systems=1_000,
    cluster=cluster,
    hybrid_switch=True,
    seed=42,
    n_jobs=4,
)
```

`num_systems` is the number of sampled star–planet systems; `n_jobs` controls parallel encounter integrations (`-1` uses all available CPUs). This is an illustrative population size: precise estimates of rare HJ outcomes require much larger samples. Initial orbital distributions and tidal parameters are defined in [hj/config.py](hj/config.py); integration settings can be supplied through the optional `numerics=NumericalSettings(...)` argument (see [hj/numerics.py](hj/numerics.py)).

Setting `hybrid_switch=False` selects the analytic-only comparison. This applies perturbative kicks even outside their validity domain and can produce negative eccentricities that are classified as NM. Such outcomes should not be interpreted as physical circularisation; `model.results.compute_outcome_diagnostics()` identifies them.

Each call to `run` creates a new directory, beginning with `data/47tuc_hybrid/run_000/`, containing:

- `results.parquet`: initial and final orbital elements, cluster radii, outcome codes and stopping times for each system.
- `metadata.json`: the seed, model configuration, numerical settings, source revision, dependency versions and completion status.

Completed runs with the same experiment name are automatically combined by `model.results`. Keep different configurations under separate names, and use distinct seeds for independent batches. To reopen saved results in a later session, construct `HJModel` with the same `name` and `base_dir`; there is no need to call `run` again.

## Results

Compute outcome probabilities for the whole cluster or a radial selection. Radii are measured from the cluster centre in parsecs at the end of the run:

```python
probabilities = model.results.compute_outcome_probabilities()
core_probabilities = model.results.compute_outcome_probabilities(r_range=(0.0, 0.5))
print(f"HJ fraction: {probabilities['HJ']:.3%}")
```

The default selection includes systems within 100 pc. These probabilities are fractions of the simulated progenitor population; comparison with the paper's per-star occurrence rates requires the assumed cold Jupiter abundance. Individual system records are also available as a pandas DataFrame through `model.df`.

The results object also provides plotting methods for the final orbital distribution, stopping times and cluster radii. For example, to plot the final states in the phase plane:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
model.results.plot_phase_plane(ax)
ax.set_xlabel(r"$a$ / au")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
```
