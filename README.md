# HotJupiter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
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
