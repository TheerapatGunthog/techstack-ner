import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path
import os

PROJECT_ROOT = Path(os.getcwd())

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

INPUT_DS_PATH = PROJECT_ROOT / "data/raw/scraping-data/merged_product.csv"
OUTPUT_NONTECH_PATH = PROJECT_ROOT / "data/raw/classified/non_tech_jobs.csv"
OUTPUT_TECH_PATH = PROJECT_ROOT / "data/raw/classified/tech_jobs.csv"

# Read CSV file
df = pd.read_csv(INPUT_DS_PATH)

# Check the DataFrame structure
print(f"Total number of entries: {len(df)}")
print(df.head())

# Define candidate labels for classification
candidate_labels = ["Technology job", "Non-technology job"]


def classify_job(topic, Qualification, threshold=0.6):
    text_to_classify = f"Job title: {topic}. Job requirements: {Qualification}"

    # Perform classification
    result = classifier(text_to_classify, candidate_labels)

    # Check if it is a tech job
    is_tech = (
        result["labels"][0] == "Technology job" and result["scores"][0] >= threshold
    )

    return {
        "is_tech": is_tech,
        "tech_score": result["scores"][0]
        if result["labels"][0] == "Technology job"
        else result["scores"][1],
        "confidence": result["scores"][0],
        "predicted_label": result["labels"][0],
    }


# Process all data
# Iterate through each row and classify
# Show progress by using a tqdm
results = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Classifying jobs"):
    try:
        result = classify_job(row["Topic"], row["Qualification"])
        results.append(result)
    except Exception as e:
        print(f"Error on entry {index}: {e}")
        results.append(
            {
                "is_tech": False,
                "tech_score": 0.0,
                "confidence": 0.0,
                "predicted_label": "Error",
            }
        )

# Add results back to DataFrame
df["is_tech"] = [r["is_tech"] for r in results]
df["tech_score"] = [r["tech_score"] for r in results]
df["confidence"] = [r["confidence"] for r in results]
df["predicted_label"] = [r["predicted_label"] for r in results]

# Filter for tech jobs only
tech_jobs = df[df["is_tech"]].copy()

# Summarize results
print(f"Total jobs: {len(df)}")
print(f"Technology jobs: {len(tech_jobs)}")
print(f"Non-technology jobs: {len(df) - len(tech_jobs)}")
print(f"Percentage of technology jobs: {len(tech_jobs) / len(df) * 100:.1f}%")

# Show sample tech jobs
for idx, row in tech_jobs.head().iterrows():
    print(f"\nJob title: {row['Topic']}")
    print(f"Tech score: {row['tech_score']:.3f}")
    print(f"Qualification: {row['Qualification'][:100]}...")

# Show tech job with highest score
top_tech_job = tech_jobs.loc[tech_jobs["tech_score"].idxmax()]
print(f"Job title: {top_tech_job['Topic']}")
print(f"Tech score: {top_tech_job['tech_score']:.3f}")
print(f"Qualification: {top_tech_job['Qualification']}")

tech_jobs = df[df["is_tech"]].copy()
non_tech_jobs = df[~df["is_tech"]].copy()

# Save only tech jobs
tech_jobs.to_csv(
    OUTPUT_TECH_PATH,
    index=False,
    encoding="utf-8-sig",
)

# Save only non-tech jobs
non_tech_jobs.to_csv(
    OUTPUT_NONTECH_PATH,
    index=False,
    encoding="utf-8-sig",
)

# Additional statistics
print(f"Average tech job score: {tech_jobs['tech_score'].mean():.3f}")
print(f"Minimum tech job score: {tech_jobs['tech_score'].min():.3f}")
print(f"Maximum tech job score: {tech_jobs['tech_score'].max():.3f}")

# Show tech job score distribution
print(tech_jobs["tech_score"].describe())
