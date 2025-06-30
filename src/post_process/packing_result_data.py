import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    logging,
)
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress informational messages from transformers
logging.set_verbosity_error()

# --- Configuration ---
PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")
MODEL_PATH = PROJECT_PATH / "models/bootstrapping01/best_model/"
DATA_PATH = PROJECT_PATH / "data/interim/summarize_text/kaggle_data.csv"
POSITION_PATH = PROJECT_PATH / "data/post_processed/position.csv"
SKILL_PATH = PROJECT_PATH / "data/post_processed/skills.csv"
OUTPUT_PATH = PROJECT_PATH / "data/post_processed/packing_data.csv"
SCORE_THRESHOLD = 0.65

# --- Data Loading ---
try:
    df = pd.read_csv(DATA_PATH)
    # Rename columns on load to be specific and avoid name clashes
    position_id_df = pd.read_csv(POSITION_PATH).rename(
        columns={"id": "job_id", "name": "pos_name"}
    )
    skill_id_df = pd.read_csv(SKILL_PATH).rename(
        columns={"id": "skill_id", "name": "skill_name"}
    )
    # --- CHANGE: Convert skill names to lowercase for consistent matching ---
    skill_id_df["skill_name"] = skill_id_df["skill_name"].str.lower()

except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    exit()

# Create an empty DataFrame for packing results
df_packing = pd.DataFrame(columns=["pos_name", "skill_name"])

# --- NER Pipeline Initialization ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
except OSError:
    print(f"Error: Model or tokenizer not found at '{MODEL_PATH}'.")
    ner_pipeline = None


# ===================================================================
def group_ner_entities(ner_results):
    """
    Groups B- and I- tagged tokens into complete entities.
    """
    grouped_entities = []
    current_entity_tokens = []
    current_entity_scores = []
    current_entity_label = ""
    current_entity_start = None
    current_entity_end = None

    for entity in ner_results:
        label = entity["entity"]
        if label.startswith("B-"):
            if current_entity_tokens:
                clean_word = "".join(current_entity_tokens).replace("Ġ", " ").strip()
                grouped_entities.append(
                    {
                        "word": clean_word,
                        "label": current_entity_label,
                        "score": float(np.mean(current_entity_scores)),
                        "start": current_entity_start,
                        "end": current_entity_end,
                    }
                )
            current_entity_tokens = [entity["word"]]
            current_entity_scores = [entity["score"]]
            current_entity_label = label.split("-")[1]
            current_entity_start = entity["start"]
            current_entity_end = entity["end"]
        elif label.startswith("I-") and current_entity_label == label.split("-")[1]:
            current_entity_tokens.append(entity["word"])
            current_entity_scores.append(entity["score"])
            current_entity_end = entity["end"]
        else:
            if current_entity_tokens:
                clean_word = "".join(current_entity_tokens).replace("Ġ", " ").strip()
                grouped_entities.append(
                    {
                        "word": clean_word,
                        "label": current_entity_label,
                        "score": float(np.mean(current_entity_scores)),
                        "start": current_entity_start,
                        "end": current_entity_end,
                    }
                )
            current_entity_tokens = []
            current_entity_scores = []
            current_entity_label = ""
            current_entity_start = None
            current_entity_end = None
    if current_entity_tokens:
        clean_word = "".join(current_entity_tokens).replace("Ġ", " ").strip()
        grouped_entities.append(
            {
                "word": clean_word,
                "label": current_entity_label,
                "score": float(np.mean(current_entity_scores)),
                "start": current_entity_start,
                "end": current_entity_end,
            }
        )
    return grouped_entities


def run_ner_prediction_and_group(topic, text, threshold):
    """
    Runs prediction and groups the results, adding unique skills to the DataFrame.
    """
    raw_results = ner_pipeline(text)
    grouped_results = group_ner_entities(raw_results)
    added_skills_for_this_index = set()
    filtered_results_count = 0

    for entity in grouped_results:
        if entity["score"] >= threshold:
            verified_word = text[entity["start"] : entity["end"]].strip()
            if (
                verified_word
                and verified_word.lower() not in added_skills_for_this_index
            ):
                filtered_results_count += 1
                df_packing.loc[len(df_packing)] = [topic, verified_word]
                added_skills_for_this_index.add(verified_word.lower())


def counting_skills_scores(df_to_count):
    """Counts occurrences of each skill in each topic."""
    if df_to_count.empty:
        return df_to_count
    skill_counts = (
        df_to_count.groupby(["pos_name", "skill_name"]).size().reset_index(name="score")
    )
    skill_counts = skill_counts.sort_values(by="score", ascending=False)
    skill_counts.reset_index(drop=True, inplace=True)
    return skill_counts


def mapping_id(df_to_map):
    """Maps position and skill names to their IDs, keeping the score."""
    if df_to_map.empty:
        return pd.DataFrame(columns=["job_id", "skill_id", "score"])

    # Merge with position IDs to get job_id
    mapped_df = df_to_map.merge(position_id_df, on="pos_name", how="left")

    # Merge with skill IDs to get skill_id
    mapped_df = mapped_df.merge(skill_id_df, on="skill_name", how="left")

    # Drop rows where a match was not found for either position or skill
    mapped_df.dropna(subset=["job_id", "skill_id"], inplace=True)

    # Select and reorder the final columns we want: job_id, skill_id, and score
    final_df = mapped_df[["job_id", "skill_id", "score"]].copy()

    # Convert ID columns to integer type
    final_df = final_df.astype({"job_id": int, "skill_id": int})

    return final_df


# --- Main Execution ---
if __name__ == "__main__":
    if ner_pipeline:
        print("--- Starting NER Skill Extraction ---")
        for index in tqdm(
            range(df.shape[0]),
            desc="Extracting Skills",
            bar_format="{desc}: {n_fmt}/{total_fmt}",
        ):
            run_ner_prediction_and_group(
                df["Topic"][index],
                df["Qualification"][index],
                SCORE_THRESHOLD,
            )

        # Check if any skills were found before proceeding
        if not df_packing.empty:
            print("\n--- Counting and Mapping Skills ---")

            # --- CHANGE: Convert extracted skill names to lowercase before counting ---
            df_packing["skill_name"] = df_packing["skill_name"].str.lower()

            df_processed = counting_skills_scores(df_packing)

            print("\n--- Unmapped Data (Skills in lowercase) ---")
            print(df_processed)

            df_final = mapping_id(df_processed)

            print("\n--- Final Mapped Data ---")
            print(df_final)

            # Save the final data
            df_final.to_csv(OUTPUT_PATH, index=False)
            print(f"\nSuccessfully saved final data to {OUTPUT_PATH}")
        else:
            print("\n--- No skills met the threshold. Final DataFrame is empty. ---")
