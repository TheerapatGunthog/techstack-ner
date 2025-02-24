import sys
import pandas as pd
import yaml
from pathlib import Path
from transformers import pipeline
from collections import Counter
import json
import re
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from data.interim import INTERIM_DATA_PATH
from data.keywords import KEYWORDS_DATA_PATH

# Load the model
model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
ner = pipeline("ner", model=model_name, tokenizer=model_name, device=0)

df = pd.read_csv(INTERIM_DATA_PATH / "segmented_data.csv").iloc[:500]

# Load keywords from YAML file
with open(KEYWORDS_DATA_PATH / "classification-keyword.yaml", "r") as file:
    classification_keywords = yaml.safe_load(file)


def flatten_and_lower(keyword_list):
    flattened_list = []
    for item in keyword_list:
        if isinstance(item, list):
            flattened_list.extend([sub_item.lower() for sub_item in item])
        else:
            flattened_list.append(item.lower())
    return set(flattened_list)


# Extract keyword categories
programming_languages = flatten_and_lower(
    classification_keywords["keywords"]["Programming_Scripting_and_Markup_languages"]
)
cloud_platforms = flatten_and_lower(
    classification_keywords["keywords"]["Cloud_platforms"]
)
databases = flatten_and_lower(classification_keywords["keywords"]["Database"])
web_frameworks_and_technologies = flatten_and_lower(
    classification_keywords["keywords"]["Web_Framework_and_Technologies"]
)
frameworks_and_libraries = flatten_and_lower(
    classification_keywords["keywords"]["Other_Framework_and_libraries"]
)
embedded_technologies = flatten_and_lower(
    classification_keywords["keywords"]["Embedded_Technologies"]
)

# Debug: Print keyword lists
print("Programming Languages:", sorted(programming_languages))


def refine_labels(ner_results, text):
    """
    Refine labels using NER results and keyword matching, case-insensitive.
    """
    refined_labels = []
    all_keywords = (
        programming_languages
        | cloud_platforms
        | databases
        | web_frameworks_and_technologies
        | frameworks_and_libraries
        | embedded_technologies
    )

    # Step 1: Process NER results
    for entity in ner_results:
        word = entity.get("word", "").strip()
        word_lower = word.lower()
        start = entity.get("start")
        end = entity.get("end")
        entity_type = entity.get("entity", "")

        # Debug: Print every entity
        print(f"NER Entity: {word}, Type: {entity_type}, Start: {start}, End: {end}")

        # Only process if we have valid start and end
        if start is None or end is None:
            continue

        label = None
        if word_lower in programming_languages:
            label = "B-PROGRAMMINGLANG"
        elif word_lower in cloud_platforms:
            label = "B-CLOUDPLATFORM"
        elif word_lower in databases:
            label = "B-DATABASE"
        elif word_lower in web_frameworks_and_technologies:
            label = "B-WEBFRAMEWORK_TECH"
        elif word_lower in frameworks_and_libraries:
            label = "B-FRAMEWORK_LIB"
        elif word_lower in embedded_technologies:
            label = "B-EMBEDDEDTECH"

        if label:
            words = word.split()
            if len(words) > 1:
                current_pos = start
                for i, w in enumerate(words):
                    w_start = text.find(w, current_pos, end)
                    w_end = w_start + len(w)
                    w_label = (
                        f"B-{label.split('-')[1]}"
                        if i == 0
                        else f"I-{label.split('-')[1]}"
                    )
                    refined_labels.append(
                        {
                            "entity": w_label,
                            "start": w_start,
                            "end": w_end,
                            "text": text[w_start:w_end],
                        }
                    )
                    current_pos = w_end
            else:
                refined_labels.append(
                    {
                        "entity": label,
                        "start": start,
                        "end": end,
                        "text": word,
                    }
                )

    # Step 2: Direct keyword matching for missed entities
    for keyword in all_keywords:
        for match in re.finditer(
            r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE
        ):
            start, end = match.span()
            # Avoid duplicates
            if not any(
                label["start"] == start and label["end"] == end
                for label in refined_labels
            ):
                word = text[start:end]
                word_lower = word.lower()
                label = None
                if word_lower in programming_languages:
                    label = "B-PROGRAMMINGLANG"
                elif word_lower in cloud_platforms:
                    label = "B-CLOUDPLATFORM"
                elif word_lower in databases:
                    label = "B-DATABASE"
                elif word_lower in web_frameworks_and_technologies:
                    label = "B-WEBFRAMEWORK_TECH"
                elif word_lower in frameworks_and_libraries:
                    label = "B-FRAMEWORK_LIB"
                elif word_lower in embedded_technologies:
                    label = "B-EMBEDDEDTECH"

                if label:
                    refined_labels.append(
                        {
                            "entity": label,
                            "start": start,
                            "end": end,
                            "text": word,
                        }
                    )

    refined_labels = sorted(refined_labels, key=lambda x: x["start"])
    return refined_labels


# Create a list to store the results in Label Studio format
label_studio_data = []
all_entities = []
idcount = 0

# Process each row using Segmented_Qualification column
for index, row in tqdm(
    df.iterrows(), total=len(df), desc="Processing Segmented_Qualification"
):
    text = row["Segmented_Qualification"]
    if pd.notna(text) and text.strip():  # Check if text is not NaN or empty
        print(
            f"Row {index} Text: {text[:100]}..."
        )  # Print first 100 chars for debugging
        ner_results = ner(text)
        print(f"Row {index} NER Results: {ner_results}")
        refined_labels = refine_labels(ner_results, text)

        if refined_labels:
            label_studio_data.append(
                {
                    "id": str(idcount),
                    "data": {"text": text},
                    "annotations": [
                        {
                            "id": idcount,
                            "result": [
                                {
                                    "value": {
                                        "start": label["start"],
                                        "end": label["end"],
                                        "text": label["text"],
                                        "labels": [label["entity"]],
                                    },
                                    "id": f"result-{idcount}-{i}",
                                    "from_name": "label",
                                    "to_name": "text",
                                    "type": "labels",
                                }
                                for i, label in enumerate(refined_labels)
                            ],
                        }
                    ],
                }
            )
            idcount += 1

        all_entities.extend([label["entity"] for label in refined_labels])

# Save the results as JSON
output_path = INTERIM_DATA_PATH / "labeled_by_code_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(label_studio_data, f, ensure_ascii=False, indent=4)

print(f"Data saved to {output_path}")

# Count the frequency of each unique entity
entity_counts = Counter(all_entities)

# Display unique entities and their counts
print("\nUnique entities and their counts:")
for entity, count in entity_counts.items():
    print(f"{entity}: {count}")
