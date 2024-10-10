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

```python
from hjmodel import HJModel

model = HJModel(res_path=RES_PATH, res_name=RES_NAME)
model.run(time=TIME_IN_MYR, num_systems=NUM_SYSTEMS)
```
The output is saved to the file `RES_NAME.pq` in the directory specified by `RES_PATH`. The results of a HJModel instance can be accessed by `model.df`, although we provide a few in-built methods for summary statistics:

```python
outcome_probs = model.get_outcome_probabilities()
"""
Returns: dict[str, float]    {'I': f_i, 'TD': f_td, 'HJ': f_hj, 'WJ': f_wj, 'NM': f_nm}
"""

stats = get_stats_for_outcome(outcomes=OUTCOMES, feature=FEATURE)
"""
outcomes: list[str]          subset of ['I', 'TD', 'HJ', 'WJ', 'NM']
feature:  str                in ['r', 'e_init', 'a_init', 'stopping_time', 'final_e', 'final_e', 'm1']
--------
Returns: list[float]
"""   
```

