import pandas as pd
import yaml
from pathlib import Path
from transformers import pipeline, AutoTokenizer
import json
import re
from tqdm import tqdm

PROJECT_ROOT = Path("/home/whilebell/Code/techstack-ner/")

# Load the model
model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
ner = pipeline("ner", model=model_name, tokenizer=model_name, device=0)
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_csv(PROJECT_ROOT / "data/interim/segmented-data/kaggle-segmented-data.csv")

print(df)

# Load keywords from YAML file
with open(PROJECT_ROOT / "data/keywords/classification-keyword.yaml", "r") as file:
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
# PSML
programming_languages = flatten_and_lower(
    classification_keywords["keywords"]["Programming_Scripting_and_Markup_languages"]
)
# CP
cloud_platforms = flatten_and_lower(
    classification_keywords["keywords"]["Cloud_platforms"]
)
# DB
databases = flatten_and_lower(classification_keywords["keywords"]["Database"])

web_frameworks_and_technologies = flatten_and_lower(
    classification_keywords["keywords"]["Web_Framework_and_Technologies"]
)
frameworks_and_libraries = flatten_and_lower(
    classification_keywords["keywords"]["Other_Framework_and_libraries"]
)
# Merge a Web_Framework and Technologies with Other Framework and Libraries
# TFL
tech_frameworks_and_libraries = (
    web_frameworks_and_technologies | frameworks_and_libraries
)

# ET
embedded_technologies = flatten_and_lower(
    classification_keywords["keywords"]["Embedded_Technologies"]
)


def clean_word(word):
    """
    Clean subword artifacts like ' ', '##', '@@' etc.
    """
    return re.sub(r"^[ #Ġ@]+", "", word).strip().lower()


def refine_labels(ner_results, text):
    """
    Refine labels using NER results and keyword matching, but only for MISC entities.
    Ensure that overlapping entities keep only the longest span.
    """
    refined_labels = []
    all_keywords = (
        programming_languages
        | cloud_platforms
        | databases
        | tech_frameworks_and_libraries
        | embedded_technologies
    )

    tokenized = tokenizer.encode_plus(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    tokens = tokenized.tokens()
    offset_mapping = tokenized["offset_mapping"]

    temp_labels = []

    # Step 1: Process NER results (only MISC entities)
    for entity in ner_results:
        start = entity.get("start")
        end = entity.get("end")
        entity_label = entity.get("entity")

        if not entity_label or "MISC" not in entity_label:
            continue

        if start is None or end is None:
            continue

        raw_word = text[start:end]
        word_clean = clean_word(raw_word)

        label = None
        if word_clean in programming_languages:
            label = "PSML"
        elif word_clean in cloud_platforms:
            label = "CP"
        elif word_clean in databases:
            label = "DB"
        elif word_clean in tech_frameworks_and_libraries:
            label = "TFL"
        elif word_clean in embedded_technologies:
            label = "ET"
        else:
            label = "OTHER"

        if label:
            temp_labels.append(
                {"entity": label, "start": start, "end": end, "text": raw_word}
            )

    # Step 2: Use regex to match any keyword in original text
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
                for s, e in offset_mapping:
                    if s <= start < e or s < end <= e:
                        token_start = s if token_start is None else min(token_start, s)
                        token_end = e if token_end is None else max(token_end, e)

            if not any(
                label["start"] == token_start and label["end"] == token_end
                for label in temp_labels
            ):
                word = text[token_start:token_end]
                word_clean = clean_word(word)
                label = None

                if word_clean in programming_languages:
                    label = "PSML"
                elif word_clean in cloud_platforms:
                    label = "CP"
                elif word_clean in databases:
                    label = "DB"
                elif word_clean in tech_frameworks_and_libraries:
                    label = "TFL"
                elif word_clean in embedded_technologies:
                    label = "ET"

                if label:
                    temp_labels.append(
                        {
                            "entity": label,
                            "start": token_start,
                            "end": token_end,
                            "text": word,
                        }
                    )

    # Step 3: Remove overlapping entities, keeping the longest
    temp_labels.sort(key=lambda x: (x["start"], -x["end"]))
    prev_start, prev_end = -1, -1
    for label in temp_labels:
        if label["start"] >= prev_end:
            refined_labels.append(label)
            prev_start, prev_end = label["start"], label["end"]
        elif label["end"] - label["start"] > prev_end - prev_start:
            refined_labels[-1] = label
            prev_start, prev_end = label["start"], label["end"]

    return refined_labels


# Create a list to store only the texts that have identified labels
texts_with_labels = []

# Process each row using Segmented_Qualification column
for index, row in tqdm(
    df.iterrows(), total=len(df), desc="Processing Segmented_Qualification"
):
    text = row["Segmented_Qualification"]
    if pd.notna(text) and text.strip():  # Check if text is not NaN or empty
        ner_results = ner(text)
        refined_labels = refine_labels(ner_results, text)

        # If any labels were identified, add the original text to our list
        if refined_labels:
            texts_with_labels.append(text)

# Save the filtered texts as JSON
output_path = PROJECT_ROOT / "data/interim/gemini/filtered_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(texts_with_labels, f, ensure_ascii=False, indent=4)

print(f"Data saved to {output_path}")
