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

model = HJModel(time=TIME_IN_MYR, num_systems=NUM_SYSTEMS, res_path=RES_PATH, res_name=RES_NAME)
model.run()
```
The output is saved to the file `RES_NAME.pq` in the directory specified by `RES_PATH`. 
