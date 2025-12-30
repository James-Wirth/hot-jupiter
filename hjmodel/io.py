from __future__ import annotations

import logging
import shutil
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

logger = logging.getLogger(__name__)


class DaskProcessor:
    """
    Manages partitioned parquet file writing and merging using Dask.

    Handles intermediate storage of simulation results during parallel
    processing, then combines all partitions into a single output file.

    Attributes:
        output_path: Final output path for the merged parquet file.
        partition_dir: Directory for storing intermediate partition files.
    """

    def __init__(self, output_path: Path):
        """
        Initialize the DaskProcessor with an output path.

        Args:
            output_path: Path where the final merged parquet file will be saved.
        """
        self.output_path = Path(output_path)
        self.partition_dir = self.output_path.parent / "parquet_batches"
        self._partition_count = 0

    def prepare_dir(self) -> None:
        """
        Prepare the partition directory for writing.

        Removes any existing partition directory and creates a fresh one.
        Resets the partition counter to zero.
        """
        if self.partition_dir.exists():
            logger.info("Cleaning existing directory: %s", self.partition_dir)
            shutil.rmtree(self.partition_dir)
        self.partition_dir.mkdir(parents=True, exist_ok=True)
        self._partition_count = 0

    def write_partition(self, df: pd.DataFrame) -> Path:
        """
        Write a DataFrame as a parquet partition file.

        Args:
            df: DataFrame to write as a partition.

        Returns:
            Path to the written partition file.
        """
        self._partition_count += 1
        partition_path = (
            self.partition_dir / f"partition_{self._partition_count}.parquet"
        )
        df.to_parquet(partition_path, engine="pyarrow", compression="snappy")
        logger.debug("Wrote partition %d to %s", self._partition_count, partition_path)
        return partition_path

    def save_all_partitions(self) -> None:
        """
        Combine all partition files into a single parquet file using Dask.

        Reads all partition files from the partition directory, merges them,
        and writes the result to the output path.
        """
        logger.info("Combining %d partitions with Dask...", self._partition_count)
        ddf = dd.read_parquet(str(self.partition_dir / "*.parquet"))

        logger.info("Saving combined dataset to %s", self.output_path)
        parent = self.output_path.parent
        parent.mkdir(parents=True, exist_ok=True)

        ddf.to_parquet(str(self.output_path), engine="pyarrow", write_index=False)
        logger.info("Successfully saved combined results")

    def clean_partitions(self) -> None:
        """
        Remove all partition files and the partition directory.

        Attempts to delete each partition file and then the directory itself.
        Logs warnings if files cannot be deleted but does not raise exceptions.
        """
        for partition_file in self.partition_dir.glob("partition_*.parquet"):
            try:
                partition_file.unlink()
            except Exception:
                logger.warning("Couldn't delete partition file %s", partition_file)
        try:
            self.partition_dir.rmdir()
        except Exception:
            logger.debug(
                "Couldn't remove partition directory: %s (might not be empty)",
                self.partition_dir,
            )
