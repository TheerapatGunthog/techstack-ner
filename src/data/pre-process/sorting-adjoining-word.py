import pandas as pd
import re

input_csv = "/home/whilebell/Code/Project/TechStack-NER/data/interim/segmented-data/scraping-segmented-data.csv"
output_suspect = "/home/whilebell/Code/Project/TechStack-NER/data/interim/segmented-data/adjoining-word-data/scraping-segmented-suspect-data.csv"
output_clean = "/home/whilebell/Code/Project/TechStack-NER/data/interim/segmented-data/adjoining-word-data/scraping-segmented-data.csv"

# The pattern to detect abnormal sentences
stuck_pattern = re.compile(
    r"[a-zA-Z0-9][A-Z][a-z]|[a-z][A-Z]|[a-zA-Z][0-9][a-zA-Z]|#([a-zA-Z])|\+\+([a-zA-Z])"
)


def has_stuck_sentence(text):
    """Check if the text in Qualification has any abnormal pattern."""
    if pd.isnull(text):
        return False
    return bool(stuck_pattern.search(str(text)))


df = pd.read_csv(input_csv, encoding="utf-8")

# Create a mask for rows that have suspicious sentences in the Qualification column
mask_suspect = df["Segmented_Qualification"].apply(has_stuck_sentence)

# Split into two datasets
df_suspect = df[mask_suspect].copy()
df_clean = df[~mask_suspect].copy()

# Export to CSV files
df_suspect.to_csv(output_suspect, index=False, encoding="utf-8")
df_clean.to_csv(output_clean, index=False, encoding="utf-8")

print(f"Suspicious sentences saved at: {output_suspect}")
print(f"Normal sentences saved at: {output_clean}")
