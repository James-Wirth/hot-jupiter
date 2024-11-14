# HotJupiter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**HotJupiter** is a Monte-Carlo model for studying planet formation in dense globular clusters, with a focus on high-eccentricity migration. Utilizing the N-body [REBOUND](https://github.com/hannorein/rebound) code, HotJupiter simulates the dynamical evolution of planetary systems under the influence of stellar perturbations.

Through these simulations, the final states of planetary systems are categorized into five unique outcomes:
- **Ionisation**: Unbound systems with $e > 1$
- **Tidal Disruption**: Systems disrupted when $r(t) < R_{\mathrm{td}}$ at any time $t$
- **Hot Jupiter**: Systems with orbital periods $T < 10 \ \mathrm{days}$
- **Warm Jupiter**: Systems with orbital periods $10 < T < 100 \ \mathrm{days}$
- **No Migration**: Systems that do not fall into the above categories

## Installation

To get started with HotJupiter, clone the repository and install the dependencies:

```bash
git clone https://github.com/James-Wirth/HotJupiter
cd HotJupiter
pip install -r requirements.txt
```

## Usage

### Quickstart

The DynamicPlummer class initializes a simple time-interpolated Plummer model for a globular cluster with specified parameters. For example, for the 47 Tuc cluster (following Giersz & Heggie, 2011), we can set up a 12 Gyr-evolved instance as follows:

```python
from hjmodel.cluster import DynamicPlummer

plummer = DynamicPlummer(
    m0=(1.64E6, 0.9E6),        # Mass at t=0 and t=total_time
    rt=(86, 70),               # Tidal radius at t=0 and t=total_time
    rh=(1.91, 4.96),           # Half-mass radius at t=0 and t=total_time
    n=(2E6, 1.85E6),           # Number of stars/binaries at t=0 and t=total_time
    total_time=12000           # Evolution duration in Myr
)
```

### Run the Monte-Carlo simulation

With the cluster instance created, you can now initialize an HJModel instance and run the Monte-Carlo simulation.

```python
from hjmodel import HJModel

model = HJModel(res_path=PATH)   # Specify output path

model.run_dynamic(
    time=12000,                  # Total simulation time in Myr
    num_systems=500000,          # Number of planetary systems to simulate
    cluster=plummer              # Cluster instance
)
```

The results are saved to the specified `res_path`. You can access the full data frame of results with `model.df`, or use built-in methods for a summary.

### Analyze the results

1. Formation rates:

Retrieve the formation rates of different planetary system outcomes:

```python
outcome_probs = model.get_outcome_probabilities()
# Returns: {'I': f_i, 'TD': f_td, 'HJ': f_hj, 'WJ': f_wj, 'NM': f_nm}
```

2. System feature statistics

Generate histogram data for system features across specified outcomes:

```python
stats = model.get_stats_for_outcome(outcomes=["HJ", "WJ"], feature="stopping_time")
# Returns: list of values for the stopping time feature in Hot and Warm Jupiters
```

3. Summary figures:

HotJupiter includes several visual summary functions for analyzing and presenting simulation results:

```python
fig1 = model.get_summary_figure()
fig2 = model.get_projected_probability_figure()
```

## License
