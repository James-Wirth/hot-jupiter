import os
import glob
import dask.dataframe as dd
from pathlib import Path
import gc
import pandas as pd


def merge_parquet_batches(output_path: str, batch_folder: str = "data/parquet_batches"):
    """
    Merges existing Parquet batch files in the given batch_folder and saves the result to output_path.

    Parameters:
    - output_path: str -> Path to save the merged Parquet file
    - batch_folder: str -> Folder containing Parquet batch files
    """
    batch_dir = Path(batch_folder)
    if not batch_dir.exists():
        print(f"Error: The batch directory {batch_folder} does not exist.")
        return

    # Find all Parquet batch files
    parquet_files = list(batch_dir.glob("partition_*.parquet"))
    if not parquet_files:
        print("No Parquet batch files found. Exiting.")
        return

    print(f"Found {len(parquet_files)} batch files. Merging...")

    # Load all batch files with Dask
    ddf = dd.read_parquet(str(batch_dir / "*.parquet"))

    print(f"Saving merged dataset to {output_path}")
    ddf.to_parquet(output_path, engine="pyarrow", write_index=False)

    # Optional: Clean up individual partition files
    for partition_file in parquet_files:
        partition_file.unlink()
    batch_dir.rmdir()

    # Load into memory for verification (optional, but useful)
    df = pd.read_parquet(output_path)
    print(f"Final dataset saved with {df.shape[0]} rows.")
    gc.collect()


if __name__ == "__main__":
    # Set output path where the merged Parquet file should be saved
    output_parquet_path = "data/exp_data_47tuc_FINAL_BATCHED_ANALYTIC_RUN2.pq"
    merge_parquet_batches(output_path=output_parquet_path)