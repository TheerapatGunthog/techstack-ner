import sys
from pathlib import Path
import pandas as pd

# Add parent directory to sys.path (assuming this is for importing RAW_DATA_PATH)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data.raw import RAW_DATA_PATH

# Define the folder path
fd_path0 = Path(RAW_DATA_PATH / "first-data/csv1/")
fd_path1 = Path(RAW_DATA_PATH / "first-data/csv2/")

# List to store DataFrames
df_list = []

# Iterate over all CSV files in the folder
for csv_file in fd_path0.glob("*.csv"):
    # Read each CSV file, selecting only the desired columns
    df = pd.read_csv(csv_file, usecols=["Topic", "Position", "Qualification"])
    df_list.append(df)

for csv_file in fd_path1.glob("*.csv"):
    # Read each CSV file, selecting only the desired columns
    df = pd.read_csv(csv_file, usecols=["Topic", "Position", "Qualification"])
    df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(df_list, ignore_index=True)

# Show sum of rows and columns
print(merged_df.shape)

# Save the merged DataFrame to a CSV file
merged_df.to_csv(RAW_DATA_PATH / "merged.csv", index=False)
