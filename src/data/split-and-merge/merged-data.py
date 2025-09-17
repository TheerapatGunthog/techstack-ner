"""
Module for merging multiple CSV files into a single dataset.
"""

import sys
from pathlib import Path
from typing import Optional
import pandas as pd

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data.raw import RAW_DATA_PATH


def read_csv_files_from_folder(folder_path: Path) -> Optional[pd.DataFrame]:
    """
    Read all CSV files from a folder and combine them into a single DataFrame.

    Args:
        folder_path: Path to folder containing CSV files

    Returns:
        Combined DataFrame or None if no files found
    """
    csv_files = list(folder_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return None

    return pd.concat(
        [
            pd.read_csv(file, usecols=["Topic", "Position", "Qualification"])
            for file in csv_files
        ],
        ignore_index=True,
    )


def main():
    """Main function to process and merge CSV files."""
    # Define the folder paths
    fd_paths = [
        RAW_DATA_PATH / "first-data/csv1/",
        RAW_DATA_PATH / "first-data/csv2/",
        RAW_DATA_PATH / "first-data/csv3/",
    ]

    # Process all folders and combine into one DataFrame
    dataframes = [
        df for path in fd_paths if (df := read_csv_files_from_folder(path)) is not None
    ]

    if not dataframes:
        print("No data found in any folder.")
        return

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Show sum of rows and columns
    print(f"Merged data shape: {merged_df.shape}")

    # Save the merged DataFrame to a CSV file
    output_path = RAW_DATA_PATH / "merged.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Successfully saved merged data to {output_path}")


if __name__ == "__main__":
    main()
