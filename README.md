# HotJupiter

HotJupiter is a Monte-Carlo model for planet formation in dense globular cluster environments by high-eccentricity migration. The dynamical evolution of planetary systems due to stellar perturbations are simulated with the N-body [REBOUND](https://github.com/hannorein/rebound) code. The final states of the planetary system are categorised into five distinct outcomes:

Outcome | Description | 
--- | --- 
Ionisation | $e > 1$ |
Tidal disruption | $r(t) < R_{\mathrm{td}}$ at any $t$ |
Hot Jupiter | $T < 10 \ \mathrm{days}$ |
Warm Jupiter | $10 < T < 100 \ \mathrm{days}$ |
No Migration | None of the above |

## Usage

Create a `DynamicPlummer` instance with the values of the cluster parameters $M_0$ (total mass), $r_t$ (tidal radius), $r_h$ (half-mass radius), $N$ (number of stars and binaries) evaluated at $t=0$ and $t=$`total_time`. For 47 Tuc (c.f. Giersz and Heggie, 2011) evolved for 12 Gyr:

```python
plummer = DynamicPlummer(m0 = (1.64E6, 0.9E6),
                         rt = (86, 70),
                         rh = (1.91, 4.96),
                         n  = (2E6, 1.85E6),
                         total_time = 12000)
```

Run the model:

```python
model = HJModel(res_path = RES_PATH)

model.run_dynamic(time        = 12000,
                  num_systems = 500000,
                  cluster     = plummer)
```

The output is saved to the path `res_path`. The results of a HJModel instance can be accessed by `model.df`, although we provide a few in-built methods for summary statistics. 

1. Formation rates:

```python
outcome_probs = model.get_outcome_probabilities()
"""
Returns: dict[str, float]    {'I': f_i, 'TD': f_td, 'HJ': f_hj, 'WJ': f_wj, 'NM': f_nm}
"""
```

2. Histogram data for system features:

```python
stats = model.get_stats_for_outcome(outcomes=OUTCOMES, feature=FEATURE)
"""
outcomes: list[str]          subset of ['I', 'TD', 'HJ', 'WJ', 'NM']
feature:  str                in ['r', 'e_init', 'a_init', 'stopping_time', 'final_e', 'final_e', 'm1']
--------
Returns: list[float]
"""
```

3. Summary figures:

```python
fig = model.get_summary_figure()
"""
CDFs for radius and stopping time
and outcome distribution in a, 1/(1-e) space
"""

fig2 = model.get_projected_probability_figure()
"""
outcome probability against projected radius
"""
```

