from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from hj import HJModel, NumericalSettings, Plummer
from hj.diagnostics import outcome_diagnostics
from hj.provenance import runtime_metadata
from hj.statistics import (
    outcome_intervals,
    paired_outcome_transitions,
    paired_replicate_comparison,
    replicate_intervals,
)


def _write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _completed_seeds(model, expected, allowed_seeds, source_hash):
    completed = set()
    for directory in sorted(model.exp_path.glob("run_*")):
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Missing metadata in {directory}")
        metadata = json.loads(metadata_path.read_text())
        seed = metadata["seed_entropy"]
        if seed not in allowed_seeds:
            raise ValueError(f"Unexpected replicate seed in {directory}")
        if metadata.get("source_sha256") != source_hash:
            raise ValueError(f"Source differs in {directory}")
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Configuration differs in {directory}")
        if metadata["status"] != "complete":
            if (directory / "results.parquet").exists():
                raise ValueError(f"Incomplete run has results in {directory}")
            continue
        if not (directory / "results.parquet").exists():
            raise ValueError(f"Completed run has no results in {directory}")
        if seed in completed:
            raise ValueError(f"Duplicate completed replicate seed in {model.exp_path}")
        completed.add(seed)
    return completed


def _summarize(output, models, completed_pairs, r_max):
    paired_ids = {str(seed) for seed in completed_pairs}
    frames = {}
    initial_columns = [
        "replicate_id",
        "system_id",
        "e_init",
        "a_init",
        "m1",
        "m2",
        "lagrange",
    ]
    initial = None
    for label, model in models.items():
        model.invalidate_cache()
        df = model.df
        df = df[df.replicate_id.isin(paired_ids)].sort_values(
            ["replicate_id", "system_id"]
        )
        if initial is None:
            initial = df[initial_columns].reset_index(drop=True)
        elif not initial.equals(df[initial_columns].reset_index(drop=True)):
            raise ValueError("Paired treatments have different initial populations")
        frames[label] = df[df.r <= r_max]
        if set(frames[label].replicate_id) != paired_ids:
            raise ValueError("Radius selection removed an entire replicate")
        outcome_intervals(frames[label].stop_code, method="exact").to_csv(
            output / f"{label}_outcomes.csv"
        )
        outcome_diagnostics(frames[label]).to_csv(output / f"{label}_diagnostics.csv")
        if len(paired_ids) >= 2:
            replicate_intervals(frames[label]).to_csv(
                output / f"{label}_replicates.csv"
            )
    paired_outcome_transitions(frames["treatment"], frames["baseline"]).to_csv(
        output / "transition_counts.csv"
    )
    negative = frames["baseline"][frames["baseline"].e < 0]
    keys = ["replicate_id", "system_id"]
    treatment_negative = frames["treatment"].merge(
        negative[keys], on=keys, validate="one_to_one"
    )
    paired_outcome_transitions(treatment_negative, negative).to_csv(
        output / "negative_e_transition_counts.csv"
    )
    if len(paired_ids) >= 2:
        comparison = paired_replicate_comparison(
            frames["treatment"], frames["baseline"]
        )
        comparison.to_csv(output / "paired_comparison.csv")
        return comparison
    return None


def run_paired_experiment(
    output: Path,
    *,
    num_replicates: int,
    num_systems: int,
    time_myr: float = 12000.0,
    seed: int = 42,
    cluster: Plummer | None = None,
    treatment_numerics: NumericalSettings | None = None,
    baseline_numerics: NumericalSettings | None = None,
    baseline_hybrid: bool = False,
    r_max: float = 100.0,
    n_jobs: int = 1,
    resume: bool = False,
) -> pd.DataFrame:
    if num_replicates < 2 or num_systems < 1:
        raise ValueError("Require at least two replicates and one system per replicate")
    if (
        not np.isfinite(time_myr)
        or time_myr < 0
        or not np.isfinite(r_max)
        or r_max <= 0
    ):
        raise ValueError("Require finite time >= 0 and finite radius > 0")
    cluster = Plummer() if cluster is None else cluster
    treatment_numerics = (
        NumericalSettings(phase_at_pericentre=True)
        if treatment_numerics is None
        else treatment_numerics
    )
    baseline_numerics = (
        treatment_numerics if baseline_numerics is None else baseline_numerics
    )
    if treatment_numerics.phase_at_pericentre != baseline_numerics.phase_at_pericentre:
        raise ValueError("Paired treatments must use the same phase reference")
    if (
        treatment_numerics.encounter_xi != baseline_numerics.encounter_xi
        and not treatment_numerics.phase_at_pericentre
    ):
        raise ValueError("Cutoff comparisons must align phases at pericentre")
    provenance = runtime_metadata()
    configuration = {
        "num_systems": num_systems,
        "time_myr": float(time_myr),
        "seed": seed,
        "cluster": vars(cluster),
        "cluster_type": type(cluster).__name__,
        "r_max": r_max,
        "treatment_numerics": asdict(treatment_numerics),
        "baseline_numerics": asdict(baseline_numerics),
        "baseline_hybrid": baseline_hybrid,
        "source_sha256": provenance["source_sha256"],
        "packages": provenance["packages"],
        "python": provenance["python"],
        "platform": provenance["platform"],
    }
    output = Path(output)
    manifest_path = output / "manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(
                "Output exists; use resume with the same configuration"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest["configuration"] != configuration:
            raise ValueError(
                "Cannot resume with a different configuration, source or environment"
            )
        if num_replicates < manifest["num_replicates"]:
            raise ValueError("Cannot reduce the replicate count when resuming")
    else:
        output.mkdir(parents=True)
        manifest = {"configuration": configuration, "provenance": provenance}
    seeds = [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(seed).spawn(num_replicates)
    ]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Replicate seeds collided")
    manifest.update(
        {
            "num_replicates": num_replicates,
            "replicate_seeds": seeds,
            "status": "running",
            "n_jobs": n_jobs,
        }
    )
    models = {label: HJModel(label, output) for label in ("baseline", "treatment")}
    modes = {"baseline": baseline_hybrid, "treatment": True}
    settings = {"baseline": baseline_numerics, "treatment": treatment_numerics}
    completed = {}
    for label, model in models.items():
        expected = {
            "num_systems": num_systems,
            "time_myr": float(time_myr),
            "cluster": vars(cluster),
            "cluster_type": type(cluster).__name__,
            "hybrid_switch": modes[label],
            "numerics": asdict(settings[label]),
        }
        completed[label] = _completed_seeds(
            model, expected, set(seeds), provenance["source_sha256"]
        )
    _write_json(manifest_path, manifest)
    try:
        for index, replicate_seed in enumerate(seeds):
            for label, model in models.items():
                if replicate_seed in completed[label]:
                    continue
                model.run(
                    time_myr,
                    num_systems,
                    cluster,
                    hybrid_switch=modes[label],
                    seed=replicate_seed,
                    n_jobs=n_jobs,
                    numerics=settings[label],
                )
                completed[label].add(replicate_seed)
            paired = completed["baseline"] & completed["treatment"]
            manifest["completed_pairs"] = len(paired)
            _write_json(manifest_path, manifest)
            _summarize(output, models, paired, r_max)
            print(f"Completed pair {index + 1}/{num_replicates}", flush=True)
        result = _summarize(output, models, set(seeds), r_max)
        manifest["status"] = "complete"
        manifest.pop("error", None)
        _write_json(manifest_path, manifest)
        return result
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--systems", type=int, default=256)
    parser.add_argument("--time", type=float, default=12000.0)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--xi", type=float, default=1e-4)
    parser.add_argument("--baseline-xi", type=float)
    parser.add_argument("--epsilon", type=float, default=1e-9)
    parser.add_argument("--tidal-step", type=float, default=0.05)
    parser.add_argument("--baseline-hybrid", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    treatment = NumericalSettings(
        encounter_xi=args.xi,
        ias15_epsilon=args.epsilon,
        tidal_step_fraction=args.tidal_step,
        phase_at_pericentre=True,
    )
    baseline = NumericalSettings(
        encounter_xi=args.baseline_xi if args.baseline_xi is not None else args.xi,
        ias15_epsilon=args.epsilon,
        tidal_step_fraction=args.tidal_step,
        phase_at_pericentre=True,
    )
    result = run_paired_experiment(
        args.output,
        num_replicates=args.replicates,
        num_systems=args.systems,
        time_myr=args.time,
        seed=args.seed,
        n_jobs=args.jobs,
        cluster=Plummer(N0=2e6, R0=1.91, A=6.99e-4, r_max=100),
        treatment_numerics=treatment,
        baseline_numerics=baseline,
        baseline_hybrid=args.baseline_hybrid,
        resume=args.resume,
    )
    print(result.to_string())


if __name__ == "__main__":
    main()
