import pandas as pd

# Import the data from CSV file
df = pd.read_csv(
    "/home/whilebell/Code/Project/TechStack-NER/data/interim/segmented-data/adjoining-word-data/scraping-segmented-data.csv"
)
print(df.head())

# Display the shape of the dataframe
print(f"Dataframe shape: {df.shape}")

x = int(df.shape[0])

# Split the dataframe into six parts
df1 = df.iloc[0 : int(x / 6)]
df2 = df.iloc[int(x / 6) : int(2 * x / 6)]
df3 = df.iloc[int(2 * x / 6) : int(3 * x / 6)]
df4 = df.iloc[int(3 * x / 6) : int(4 * x / 6)]
df5 = df.iloc[int(4 * x / 6) : int(5 * x / 6)]
df6 = df.iloc[int(5 * x / 6) : int(x)]

# Save the split dataframes to CSV files
for i, df_part in enumerate([df1, df2, df3, df4, df5, df6], start=1):
    df_part.to_csv(
        f"/home/whilebell/Code/Project/TechStack-NER/data/interim/splited-data/splitted-segmented-suspect-data-{i}.csv",
        index=False,
    )
