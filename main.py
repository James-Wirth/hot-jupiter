import logging
from pathlib import Path

from hjmodel import HJModel
from hjmodel.clusters import Cluster
from hjmodel.clusters.profiles.plummer import Plummer

NAME = "EXAMPLE"
TIME = 12000  # in Myr
NUM_SYSTEMS = 10000
HYBRID_SWITCH = True
SEED = 42
R_MAX = 100  # in parsecs

# directory where data/<<RESULTS_PATH>>/results.parquet will be stored
BASE_DIR = Path(__file__).resolve().parent / "data"


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting experiment: '%s'", NAME)

    cluster = Cluster(profile=Plummer(N0=2e6, R0=1.91, A=6.991e-4), r_max=R_MAX)

    model = HJModel(name=NAME, base_dir=BASE_DIR)
    logger.info(
        "Running simulation: name=%s time=%s num_systems=%s seed=%s hybrid_switch=%s",
        NAME,
        TIME,
        NUM_SYSTEMS,
        SEED,
        HYBRID_SWITCH,
    )
    model.run(
        time=TIME,
        num_systems=NUM_SYSTEMS,
        cluster=cluster,
        hybrid_switch=HYBRID_SWITCH,
        seed=SEED,
    )

    results = model.results
    probs = results.compute_outcome_probabilities()
    logger.info("Outcome probabilities for experiment: '%s':", NAME)
    for label, p in probs.items():
        logger.info("  %s: %.4f", label, p)

    if model.path is None:
        logger.error("Model path not set. Cannot write summary.")
        return

    run_dir = Path(model.path).parent
    summary_path = run_dir / "summary.txt"
    run_label = run_dir.name

    with open(summary_path, "w") as f:
        f.write(f"name={NAME}\n")
        f.write(f"run_label={run_label}\n")
        f.write(f"time={TIME}\n")
        f.write(f"num_systems={NUM_SYSTEMS}\n")
        f.write(f"hybrid_switch={HYBRID_SWITCH}\n")
        f.write(f"seed={SEED}\n")
        f.write("outcome_probabilities:\n")
        for label, p in probs.items():
            f.write(f"  {label}: {p:.6f}\n")

    logger.info("Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
