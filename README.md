# HotJupiter

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**HotJupiter** is a Monte-Carlo simulation package for studying Hot Jupiter formation in dense globular clusters via high-e migration. We have used the [REBOUND](https://github.com/hannorein/rebound) code with the IAS15 numerical integrator to simulate the dynamical evolution of planetary systems due to stellar perturbation.

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

To run a simulation, you must provide a cluster instance to serve as the background. We have provided a pre-built, time-dependent Plummer profile:

```python
cluster = Cluster(
  profile=Plummer(N0=2e6, R0=1.91, A=6.99e-4),
  r_max=100
)
```

Create a `HJModel` instance and run the simulation using your desired parameters:

```python
model = HJModel(res_path="YOUR_OUTPUT_PATH.parquet")

model.run_dynamic(
  time=12000,
  num_systems=1000,
  cluster=cluster,
  hybrid_switch=True
)
```

The last parameter `hybrid_switch` controls whether to apply the hybrid model (with direct N-body simulations) or the pure analytic model. 

## Results

To explore the results, create a `Processor` instance:

```python
processor = Processor(model=model)
```

To view summary stistics:
```python
# outcome probabilities
processor.compute_outcome_probabilities(
  r_range=(0, 100)
)

# statistics for specified outcomes
processor.get_statistics_for_outcomes(
  outcomes=["HJ"],
  feature="final_e"
)
```

To view plots:
```python
# outcome probability vs projected radius
processor.plot_projected_probability()

# outcomes in a vs 1/(1-e) phase-space:
processor.plot_phase_plane()

# stopping time distribution
processor.plot_stopping_cdf()

# semi-major axis distribution
processor.plot_sma_distribution()
```
