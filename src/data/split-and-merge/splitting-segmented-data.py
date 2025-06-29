import pandas as pd
from pathlib import Path

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

# Input and output paths
input_csv = PROJECT_PATH / "data/interim/preprocessed-data/scraping_data.csv"
output_dir = PROJECT_PATH / "data/interim/splited-scraping-data/"
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

# Import the data from CSV file
df = pd.read_csv(input_csv)
print(df.head())

# Display the shape of the dataframe
print(f"Dataframe shape: {df.shape}")

x = int(df.shape[0])

# Split the dataframe into two parts
df1 = df.iloc[0 : int(x / 2)]
df2 = df.iloc[int(x / 2) : x]

# Save the split dataframes to CSV files
for i, df_part in enumerate([df1, df2], start=1):
    output_path = output_dir / f"splitted-scraping-data-{i}.csv"
    df_part.to_csv(output_path, index=False)
