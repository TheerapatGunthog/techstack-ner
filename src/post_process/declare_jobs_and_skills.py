import pandas as pd
import json
from pathlib import Path

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

DATA_PATH = PROJECT_PATH / "data/interim/summarize_text/kaggle_data.csv"
SKILL_PATH = PROJECT_PATH / "data/keywords/canonical_dictionary_v2.json"
POST_PROCESSED_PATH = PROJECT_PATH / "data/post_processed/"


df = pd.read_csv(DATA_PATH)

# Print all unique values in column "Topic"
unique_topics = df["Topic"].unique()
unique_topics = sorted(unique_topics)

position_df = pd.DataFrame(
    {"id": range(1, len(unique_topics) + 1), "name": unique_topics}
)

# Open the JSON file and load its content
with open(SKILL_PATH, "r") as file:
    skill_data = json.load(file)

skill_df = pd.DataFrame.from_dict(skill_data, orient="index")

skill_df.reset_index(inplace=True)

skill_df.rename(columns={"index": "name", 0: "group"}, inplace=True)

skill_df.insert(0, "id", range(1, len(skill_df) + 1))

# Save the DataFrames to CSV files

skill_df.to_csv(POST_PROCESSED_PATH / "skills.csv", index=False)

position_df.to_csv(POST_PROCESSED_PATH / "position.csv", index=False)
