import sys
import pandas as pd
import yaml
from pathlib import Path
from transformers import pipeline, AutoTokenizer
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
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_csv(INTERIM_DATA_PATH / "segmented_data.csv").iloc[:1000]

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


def refine_labels(ner_results, text):
    """
    Refine labels using NER results and keyword matching, but only for MISC entities.
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

    tokenized = tokenizer.encode_plus(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    tokens = tokenized.tokens()
    offset_mapping = tokenized["offset_mapping"]

    # Step 1: Process NER results (only MISC entities)
    for entity in ner_results:
        word = entity.get("word", "").strip()
        word_lower = word.lower()
        start = entity.get("start")
        end = entity.get("end")
        entity_label = entity.get("entity")

        if not entity_label or "MISC" not in entity_label:
            continue

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
            refined_labels.append(
                {
                    "entity": label,
                    "start": start,
                    "end": end,
                    "text": word,
                }
            )

    # Step 2: use offset_mapping to find start-end positions
    for keyword in all_keywords:
        pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.span()

            token_start, token_end = None, None
            for i, (token_text, (s, e)) in enumerate(zip(tokens, offset_mapping)):
                if s == start and e == end:
                    token_start, token_end = s, e
                    break

            if token_start is None:
                for i, (s, e) in enumerate(offset_mapping):
                    if s <= start < e or s < end <= e:
                        token_start = s if token_start is None else min(token_start, s)
                        token_end = e if token_end is None else max(token_end, e)

            if not any(
                label["start"] == token_start and label["end"] == token_end
                for label in refined_labels
            ):
                word = text[token_start:token_end]
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
                            "start": token_start,
                            "end": token_end,
                            "text": word,
                        }
                    )

    return sorted(refined_labels, key=lambda x: x["start"])


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
        ner_results = ner(text)
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
