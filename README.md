# HotJupiter

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
![CI](https://github.com/James-Wirth/hot-jupiter/actions/workflows/ci.yml/badge.svg)

> This repository accompanies the paper:  
> **"Hot Jupiter formation in dense stellar clusters: A Monte Carlo model applied to 47 Tucanae"**  
> James A. Wirth, Cathie C. Clarke, Andrew J. Winter. *Monthly Notices of the Royal Astronomical Society, 2025.*  
> [DOI](https://doi.org/10.1093/mnras/staf1325) | [arXiv](https://arxiv.org/abs/2508.08406)

**HotJupiter** is a Monte-Carlo simulation package for studying Hot Jupiter formation in dense globular clusters via high-eccentricity migration. We have used the [REBOUND](https://github.com/hannorein/rebound) code with the IAS15 numerical integrator to simulate the dynamical evolution of planetary systems due to stellar perturbation over Gyr timescales.

The eccentricity of a planetary system can be perturbed by stellar flybys. At sufficiently high eccentricity, the planet may pass sufficiently close to its host star for tidal torques to rapidly circularize the orbit, leading to the formation of a Hot Jupiter. Analytic expressions for the eccentricity excitation were derived by [Heggie & Rasio (1996), *The effect of encounters on the eccentricity of binaries in clusters*](https://doi.org/10.1093/mnras/282.3.1064), but these hold only in the tidal and slow regime, neglecting terms higher than quadrupole in the multipole expansion for the perturbing force. We have introduced an efficient hybrid scheme for computing the eccentricity diffusion, whereby direct N-body simulations of the encounter are performed only in regimes where the analytic expressions are invalid. 

<br>
<p align="center">
  <img src="https://github.com/user-attachments/files/21801314/show_paths_mnras.pdf" alt="Simulation Results" width="80%" />
  <br />
  <em>Example: The phase-space paths for a sample of Hot Jupiter (HJ), Warm Jupiter (WJ), Tidal Disruption (TD) and No-Migration (NM) outcomes. </em>
</p>
<br>


The final states of planetary systems are categorized into five unique outcomes: Ionisation (I), Tidal Disruption (TD), Hot Jupiter formation (HJ), Warm Jupiter formation (WJ) and No Migration (NM).

## Installation

To get started with HotJupiter, clone the repository and install the dependencies:

```bash
git clone https://github.com/James-Wirth/hot-jupiter
cd hot-jupiter
pip install -r requirements.txt
```

## Usage

Create a cluster instance to serve as the background for the simulation. We have provided a time-dependent Plummer profile for 47-Tuc:

```python

from hjmodel.clusters import Cluster
from hjmodel.clusters.profiles.plummer import Plummer

cluster = Cluster(
    profile=Plummer(N0=2e6, R0=1.91, A=6.99e-4),
    r_max=100
)
```

For guidance on running a simulation, see the provided example. You can customize the run parameters:

```python
from hjmodel import HJModel

model = HJModel(name=NAME, base_dir=BASE_DIR)
model.run(
    time=TIME,
    num_systems=NUM_SYSTEMS,
    cluster=cluster,
    hybrid_switch=HYBRID_SWITCH,
    seed=SEED
)
```

where `TIME` is the simulation duration in Megayears, `NUM_SYSTEMS` is the number of Monte-Carlo ensembles, `cluster` is the `Cluster` instance defined above, `hybrid_switch` is the boolean flag that turns on/off the hybrid mode, and `seed` is the seed for random number generation (included for reproducibility). 

This generates:

- A results file: `<BASE_DIR>/NAME/run_XXX/results.parquet`
- A summary report `<BASE_DIR>/NAME/run_XXX/summary.txt`

Multiple runs with the same name are stored under auto-incrementing `run_XXX` identifiers, for optional manual batching. 
When acessing `model.results` (see next section), data from all runs are automatically aggregated.

## Results

The outcome probabilities can be accessed via:

```python
model.results.compute_outcome_probabilities(r_range=(R_MIN, R_MAX))
```

The `models.results` object includes built-in methods for generating key plots, e.g.:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
model.results.plot_phase_plane(ax)
plt.show()
```
