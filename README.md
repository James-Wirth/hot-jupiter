# HotJupiter

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**HotJupiter** is a Monte-Carlo simulation package for studying Hot Jupiter formation in dense globular clusters via high-e migration. We have used the [REBOUND](https://github.com/hannorein/rebound) code with the IAS15 numerical integrator to simulate the dynamical evolution of planetary systems due to stellar perturbation over Gyr timescales.

The final states of planetary systems are categorized into five unique outcomes:

| Category            | Description                                                     |
|---------------------|-----------------------------------------------------------------|
| **Ionisation**      | Unbound systems with $e > 1$                                   |
| **Tidal Disruption**| Systems disrupted if $r(t) < R_{\mathrm{td}}$ at any time $t$  |
| **Hot Jupiter**     | Systems with orbital periods $T < 10 \ \mathrm{days}$          |
| **Warm Jupiter**    | Systems with orbital periods $10 < T < 100 \ \mathrm{days}$    |
| **No Migration**    | Systems that do not fall into the above categories             |


## Installation

To get started with HotJupiter, clone the repository and install the dependencies:

```bash
git clone https://github.com/James-Wirth/HotJupiter
cd HotJupiter
pip install -r requirements.txt
```

## Usage

Create a cluster instance to serve as the background for the simulation. We have provided a time-dependent Plummer profile for 47-Tuc:

```python

from clusters import Cluster
from clusters.profiles.plummer import Plummer

cluster = Cluster(
    profile=Plummer(N0=2e6, R0=1.91, A=6.99e-4),
    r_max=100
)
```

To run a simulation, use the provided driver script in `main.py`. You can customize the run directly:

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

This generates:

- A results file: `<BASE_DIR>/NAME/run_XXX/results.parquet`
- A summary report `<BASE_DIR>/NAME/run_XXX/summary.txt`

Multiple runs with the same name are stored under auto-incrementing run_XXX identifiers. 
When acessing `model.results` (see next section), data from all runs are automatically aggregated.

## Results

Access results via:

```python
results = model.results
```

The outcome probabilities can be accessed via:

```python
results.compute_outcome_probabilities(r_range=(R_MIN, R_MAX))
```

The `results` object includes built-in methods for generating key plots:

| Method                       | Description                            |
|-----------------------------|----------------------------------------|
| `plot_phase_plane(ax)`      | Phase space: $a$ vs $1/(1-e)$          |
| `plot_stopping_cdf(ax)`     | CDF of stopping times                  |
| `plot_sma_distribution(ax)` | Final $a$ distribution                 |
| `plot_sma_scatter(ax)`      | Final $a$ vs cluster radius            |
| `plot_projected_probability(ax)` | Projected radial outcome probabilities |

**Usage pattern:**

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
results.plot_phase_plane(ax)
plt.show()
