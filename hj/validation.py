from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import binomtest

from hj import core
from hj.config import ETA, R_P, XI, G
from hj.provenance import runtime_metadata


@dataclass(frozen=True)
class Encounter:
    name: str
    v_infty: float
    b: float
    lan: float
    inc: float
    aop: float
    e: float
    a: float
    m1: float
    m2: float
    m3: float
    mean_anomaly_at_pericentre: float

    def __post_init__(self):
        values = asdict(self)
        values.pop("name")
        if not self.name or not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Encounter requires a name and finite parameters")
        if min(self.v_infty, self.b, self.a, self.m1, self.m2, self.m3) <= 0:
            raise ValueError(
                "Speed, impact parameter, semimajor axis and masses must be positive"
            )
        if not 0 <= self.e < 1 or not 0 <= self.inc <= math.pi:
            raise ValueError("Require 0 <= eccentricity < 1 and 0 <= inclination <= pi")

    def arguments(self):
        return (
            self.v_infty,
            self.b,
            self.lan,
            self.inc,
            self.aop,
            self.e,
            self.a,
            self.m1,
            self.m2,
            self.m3,
        )


@dataclass(frozen=True)
class EncounterSettings:
    xi: float = XI
    epsilon: float = 1e-9

    def __post_init__(self):
        if not 0 < self.xi < 1 or not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Require 0 < xi < 1 and finite epsilon > 0")


def _cartesian_state(sim):
    masses = np.array([p.m for p in sim.particles])
    positions = np.array([p.xyz for p in sim.particles])
    velocities = np.array([p.vxyz for p in sim.particles])
    return masses, positions, velocities


def _invariants(masses, positions, velocities):
    kinetic = float(np.sum(masses[:, None] * velocities**2) / 2)
    potential = sum(
        -G * masses[i] * masses[j] / np.linalg.norm(positions[j] - positions[i])
        for i in range(len(masses))
        for j in range(i + 1, len(masses))
    )
    angular_momenta = masses[:, None] * np.cross(positions, velocities)
    return (
        kinetic + potential,
        kinetic + abs(potential),
        angular_momenta.sum(axis=0),
        np.linalg.norm(angular_momenta, axis=1).sum(),
    )


def _integrate_dop853(sim, duration, rtol, atol):
    masses, positions, velocities = _cartesian_state(sim)

    def derivative(time, state):
        positions = state[:9].reshape(3, 3)
        displacement = positions[None, :, :] - positions[:, None, :]
        distance_squared = np.sum(displacement**2, axis=2)
        np.fill_diagonal(distance_squared, np.inf)
        acceleration = G * np.sum(
            displacement * (masses[None, :] / distance_squared**1.5)[:, :, None],
            axis=1,
        )
        return np.concatenate((state[9:], acceleration.ravel()))

    solution = solve_ivp(
        derivative,
        (0.0, duration),
        np.concatenate((positions.ravel(), velocities.ravel())),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    final = solution.y[:, -1]
    for i, particle in enumerate(sim.particles):
        particle.xyz = final[3 * i : 3 * i + 3]
        particle.vxyz = final[9 + 3 * i : 12 + 3 * i]
    sim.t = duration


def integrate_encounter(
    encounter: Encounter,
    settings: EncounterSettings | None = None,
    *,
    integrator: str = "ias15",
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> dict:
    settings = EncounterSettings() if settings is None else settings
    if integrator not in {"ias15", "dop853"}:
        raise ValueError("Integrator must be ias15 or dop853")
    if not all(math.isfinite(v) and v > 0 for v in (rtol, atol)):
        raise ValueError("DOP853 tolerances must be finite and positive")
    start = perf_counter()
    sim, duration = core._create_encounter_simulation(
        *encounter.arguments(),
        encounter.mean_anomaly_at_pericentre,
        xi=settings.xi,
        epsilon=settings.epsilon,
        phase_at_pericentre=True,
    )
    initial_energy, energy_scale, initial_angular, angular_scale = _invariants(
        *_cartesian_state(sim)
    )
    if integrator == "ias15":
        sim.integrate(duration)
    else:
        _integrate_dop853(sim, duration, rtol, atol)
    final_energy, _, final_angular, _ = _invariants(*_cartesian_state(sim))
    orbit = sim.particles[1].orbit(primary=sim.particles[0])
    other_orbit = sim.particles[1].orbit(primary=sim.particles[2])
    analytic_valid, analytic_delta_e = core._analytic_encounter_de(
        *encounter.arguments()
    )
    a_pert, e_pert, r_p = core._perturber_orbit(
        encounter.v_infty,
        encounter.b,
        encounter.m1,
        encounter.m2,
        encounter.m3,
    )
    baseline_duration = core._integration_time(
        a_pert,
        e_pert,
        r_p,
        encounter.m1,
        encounter.m2,
        encounter.m3,
    )
    q = orbit.a * (1.0 - orbit.e)
    host_bound = orbit.a > 0 and orbit.e < 1
    tidal_radius = ETA * R_P * (encounter.m1 / encounter.m2) ** (1.0 / 3.0)
    endpoint_class = "ION" if not host_bound else "TD" if q < tidal_radius else "BOUND"
    return {
        **asdict(encounter),
        **asdict(settings),
        "integrator": integrator,
        "dop853_rtol": rtol if integrator == "dop853" else None,
        "dop853_atol": atol if integrator == "dop853" else None,
        "duration_years": duration,
        "runtime_seconds": perf_counter() - start,
        "tidal_ratio": r_p / encounter.a,
        "slowness_at_default_xi": baseline_duration
        / math.sqrt(encounter.a**3 / (encounter.m1 + encounter.m2)),
        "analytic_valid": bool(analytic_valid),
        "analytic_delta_e": analytic_delta_e,
        "delta_e": orbit.e - encounter.e,
        "delta_a": orbit.a - encounter.a,
        "final_e": orbit.e,
        "final_a": orbit.a,
        "final_q": q,
        "host_bound": host_bound,
        "endpoint_class": endpoint_class,
        "perturber_bound_at_endpoint": other_orbit.a > 0 and other_orbit.e < 1,
        "endpoint_energy_error_scaled": abs(final_energy - initial_energy)
        / energy_scale,
        "endpoint_angular_momentum_error_scaled": float(
            np.linalg.norm(final_angular - initial_angular) / angular_scale
        ),
    }


def benchmark_encounters(seed: int = 42, phases: int = 2) -> list[Encounter]:
    if phases < 1:
        raise ValueError("phases must be positive")
    regimes = [
        ("weak_slow", 2.0, 60.0, 0.3, 1.0),
        ("fast", 12.0, 15.0, 0.3, 1.0),
        ("penetrating", 3.0, 8.0, 0.3, 5.0),
        ("high_e", 5.0, 20.0, 0.99, 1.0),
        ("tidal_boundary", 3.0, 27.0, 0.6, 1.5),
        ("wide_planet", 6.0, 25.0, 0.6, 20.0),
    ]
    generators = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(len(regimes))
    ]
    return [
        Encounter(
            f"{name}_{phase:03d}",
            speed,
            impact,
            1.0,
            1.0,
            1.0,
            eccentricity,
            semimajor,
            0.5,
            0.001,
            0.5,
            float(rng.uniform(-math.pi, math.pi)),
        )
        for (name, speed, impact, eccentricity, semimajor), rng in zip(
            regimes, generators
        )
        for phase in range(phases)
    ]


def encounter_convergence(
    encounters: list[Encounter],
    settings: list[EncounterSettings],
    reference: EncounterSettings | None = None,
) -> pd.DataFrame:
    reference = EncounterSettings(1e-6, 1e-12) if reference is None else reference
    if not encounters or not settings:
        raise ValueError("Provide encounters and numerical settings")
    if len({e.name for e in encounters}) != len(encounters):
        raise ValueError("Encounter names must be unique")
    if any(reference.xi > s.xi or reference.epsilon > s.epsilon for s in settings):
        raise ValueError(
            "Reference must use at least as small xi and epsilon as every trial"
        )
    rows = []
    for encounter in encounters:
        reference_result = integrate_encounter(encounter, reference)
        for setting in settings:
            result = integrate_encounter(encounter, setting)
            result.update(
                {
                    "reference_xi": reference.xi,
                    "reference_epsilon": reference.epsilon,
                    "reference_delta_e": reference_result["delta_e"],
                    "reference_delta_a": reference_result["delta_a"],
                    "reference_final_q": reference_result["final_q"],
                    "reference_host_bound": reference_result["host_bound"],
                    "reference_endpoint_class": reference_result["endpoint_class"],
                    "endpoint_class_disagreement": result["endpoint_class"]
                    != reference_result["endpoint_class"],
                    "reference_energy_error_scaled": reference_result[
                        "endpoint_energy_error_scaled"
                    ],
                    "reference_angular_momentum_error_scaled": reference_result[
                        "endpoint_angular_momentum_error_scaled"
                    ],
                    "reference_runtime_seconds": reference_result["runtime_seconds"],
                    "abs_delta_e_error": abs(
                        result["delta_e"] - reference_result["delta_e"]
                    ),
                    "abs_delta_a_error_over_initial_a": abs(
                        result["delta_a"] - reference_result["delta_a"]
                    )
                    / encounter.a,
                    "abs_q_error_over_initial_q": abs(
                        result["final_q"] - reference_result["final_q"]
                    )
                    / (encounter.a * (1.0 - encounter.e)),
                    "host_bound_disagreement": result["host_bound"]
                    != reference_result["host_bound"],
                }
            )
            rows.append(result)
    return pd.DataFrame(rows)


def summarize_convergence(
    results: pd.DataFrame, confidence: float = 0.95
) -> pd.DataFrame:
    if results.empty or not 0 < confidence < 1:
        raise ValueError("Provide nonempty results and 0 < confidence < 1")
    data = results.assign(regime=results.name.str.rsplit("_", n=1).str[0])
    rows = []
    for (regime, xi, epsilon), group in data.groupby(["regime", "xi", "epsilon"]):
        if group.name.duplicated().any():
            raise ValueError("Duplicate encounter within a numerical setting")
        n = len(group)
        disagreements = int(group.endpoint_class_disagreement.sum())
        interval = binomtest(disagreements, n).proportion_ci(confidence, method="exact")
        row = {
            "regime": regime,
            "xi": xi,
            "epsilon": epsilon,
            "phases": n,
            "class_disagreements": disagreements,
            "class_disagreement_fraction": disagreements / n,
            "class_disagreement_lower": interval.low,
            "class_disagreement_upper": interval.high,
            "confidence": confidence,
            "host_bound_disagreements": int(group.host_bound_disagreement.sum()),
            "energy_error_max": group.endpoint_energy_error_scaled.max(),
        }
        surviving = group[group.host_bound & group.reference_host_bound]
        row["surviving_pairs"] = len(surviving)
        row["ionised_pairs"] = int(
            (~group.host_bound & ~group.reference_host_bound).sum()
        )
        for column in ("abs_delta_e_error", "abs_q_error_over_initial_q"):
            row[f"{column}_median"] = group[column].median()
            row[f"{column}_p95"] = group[column].quantile(0.95)
            row[f"{column}_max"] = group[column].max()
            row[f"surviving_{column}_p95"] = surviving[column].quantile(0.95)
            row[f"surviving_{column}_max"] = surviving[column].max()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phases", type=int, default=2)
    parser.add_argument("--xi", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])
    parser.add_argument("--epsilon", type=float, nargs="+", default=[1e-9, 1e-11])
    parser.add_argument("--reference-xi", type=float, default=1e-6)
    parser.add_argument("--reference-epsilon", type=float, default=1e-12)
    parser.add_argument("--regime", action="append")
    parser.add_argument("--independent-check", action="store_true")
    args = parser.parse_args()
    encounters = benchmark_encounters(args.seed, args.phases)
    if args.regime:
        available = {e.name.rsplit("_", 1)[0] for e in encounters}
        if not set(args.regime) <= available:
            parser.error(f"Regimes must be selected from {sorted(available)}")
        encounters = [e for e in encounters if e.name.rsplit("_", 1)[0] in args.regime]
    settings = [
        EncounterSettings(xi, epsilon) for xi in args.xi for epsilon in args.epsilon
    ]
    reference = EncounterSettings(args.reference_xi, args.reference_epsilon)
    if any(reference.xi > s.xi or reference.epsilon > s.epsilon for s in settings):
        parser.error("Reference xi and epsilon must be no larger than the trial values")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        **runtime_metadata(),
        "seed": args.seed,
        "phase_reference": "unperturbed planetary mean anomaly at perturber pericentre",
        "phase_sampling_scheme": "per_regime_seed_sequence_v1",
        "encounters": [asdict(e) for e in encounters],
        "settings": [asdict(s) for s in settings],
        "reference": asdict(reference),
        "independent_check": args.independent_check,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = encounter_convergence(encounters, settings, reference)
    results.to_parquet(args.output / "convergence.parquet", index=False)
    summary = summarize_convergence(results)
    summary.to_csv(args.output / "summary.csv", index=False)
    print(summary.to_string(index=False))
    if args.independent_check:
        rows = []
        for encounter in encounters:
            if encounter.name.startswith("fast_"):
                for method in ("ias15", "dop853"):
                    rows.append(integrate_encounter(encounter, integrator=method))
        pd.DataFrame(rows).to_parquet(args.output / "independent.parquet", index=False)


if __name__ == "__main__":
    main()
