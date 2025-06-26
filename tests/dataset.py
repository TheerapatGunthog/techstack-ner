from pathlib import Path
import pandas as pd

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

df = pd.read_csv(PROJECT_PATH / "data/raw/summarize_text/kaggle_data.csv")

df.head()

# show all cloumn names
print(df.columns.tolist())

# Select specific columns
df[["Topic", "Qualification_Summary"]]

# Shave to csv with no index
df[["Topic", "Qualification_Summary"]].to_csv(
    PROJECT_PATH / "data/raw/summarize_text/kaggle_data.csv", index=False
)
