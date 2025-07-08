import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pathlib import Path
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# Project path
PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

# Models path
MODEL_PATH = PROJECT_PATH / "models/bootstrapping01/best_model/"

# Data path
DATA_PATH = PROJECT_PATH / "data/interim/summarize_text/scraping-data-1.csv"

# Output file path
OUTPUT_FILENAME = (
    PROJECT_PATH
    / "data/interim/bootstrapping/002/labels_by_boostrapping_models_one_filtered_rows.json"
)

# Read data from the CSV file and handle empty values (NaN)
df = pd.read_csv(DATA_PATH).fillna({"Qualification_Summary": ""})

# --- ✨ Configuration ---
SCORE_THRESHOLD = 0.65
# Define the labels that qualify a row for inclusion
DESIRED_LABELS = {"PSML", "DB", "CP", "ET"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)


# ===================================================================
def group_ner_entities(ner_results):
    """
    Groups B- and I- labeled tokens into a single entity.
    """
    grouped_entities = []
    current_entity_tokens, current_entity_scores, current_entity_label = [], [], ""
    current_entity_start, current_entity_end = None, None

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
            current_entity_start, current_entity_end = entity["start"], entity["end"]
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
            current_entity_tokens, current_entity_scores, current_entity_label = (
                [],
                [],
                "",
            )
            current_entity_start, current_entity_end = None, None
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


# --- Main ---
if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"Error: The specified model path does not exist: '{MODEL_PATH}'")
    else:
        texts_to_process = [
            str(text).replace("\n", " ").replace("(", "").replace(")", "")
            for text in df["Qualification_Summary"]
        ]

        all_processed_data = []

        print("Running NER prediction on the dataset...")
        all_ner_results = ner_pipeline(texts_to_process, batch_size=8)
        print("NER prediction complete. Processing results...")

        for original_text, raw_results in tqdm(
            zip(texts_to_process, all_ner_results),
            total=len(texts_to_process),
            desc="Grouping and filtering entities",
        ):
            grouped_results = group_ner_entities(raw_results)

            # --- ✨ MODIFIED LOGIC ---

            # 1. First, collect all entities that meet the score threshold.
            high_score_labels = []
            if grouped_results:
                for entity in grouped_results:
                    if entity["score"] >= SCORE_THRESHOLD:
                        high_score_labels.append(
                            {
                                "text": original_text[entity["start"] : entity["end"]],
                                "label": entity["label"],
                            }
                        )

            # 2. Proceed only if we found any high-score labels.
            if high_score_labels:
                # 3. Check if any of the found labels are in our desired set.
                has_desired_label = any(
                    entity["label"] in DESIRED_LABELS for entity in high_score_labels
                )

                # 4. If the condition is met, add the original text and the *complete*
                #    list of high-score labels to our final data.
                if has_desired_label:
                    all_processed_data.append(
                        {
                            "original_text": original_text,
                            "ner_labels": high_score_labels,
                        }
                    )

        # --- Custom JSON Formatting ---
        placeholder_map = {}
        data_with_placeholders = []

        for i, item in enumerate(all_processed_data):
            new_labels = []
            for j, label in enumerate(item["ner_labels"]):
                placeholder = f"__PLACEHOLDER_{i}_{j}__"
                compact_label_str = json.dumps(label, ensure_ascii=False)
                placeholder_map[placeholder] = compact_label_str
                new_labels.append(placeholder)

            new_item = item.copy()
            new_item["ner_labels"] = new_labels
            data_with_placeholders.append(new_item)

        pretty_json_with_placeholders = json.dumps(
            data_with_placeholders, indent=2, ensure_ascii=False
        )

        final_json_string = pretty_json_with_placeholders
        for placeholder, compact_str in placeholder_map.items():
            final_json_string = final_json_string.replace(
                f'"{placeholder}"', compact_str
            )

        print(f"\nSaving results to {OUTPUT_FILENAME}...")
        OUTPUT_FILENAME.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            f.write(final_json_string)

        print("Process completed successfully!")
