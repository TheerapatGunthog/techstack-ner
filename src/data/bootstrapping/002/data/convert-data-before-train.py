import sys
from pathlib import Path

# Add path to import data.interim and data.processed
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

from data.interim import INTERIM_DATA_PATH
import random
from typing import List, Dict
import json

# Define mapping for entity types
ENTITY_MAPPING = {
    "CLOUDPLATFORM": (1, 2),
    "PROGRAMMINGLANG": (3, 4),
    "FRAMEWORK_LIB": (5, 6),
    "WEBFRAMEWORK_TECH": (7, 8),
    "DATABASE": (9, 10),
    "EMBEDDEDTECH": (11, 12),
}


def process_labelstudio_to_ner_format(raw_data: str) -> List[Dict]:
    lines = raw_data.strip().split("\n")

    # Variable to store results
    sentences = []
    current_id = 0
    current_tokens = []
    current_tags = []

    for line in lines:
        if line.strip() == "":
            # End of a sentence, create data
            if current_tokens:
                sentences.append(
                    {
                        "id": str(current_id),
                        "tokens": current_tokens,
                        "ner_tags": current_tags,
                    }
                )
                current_id += 1
                current_tokens = []
                current_tags = []
            continue

        # Split token and label
        parts = line.split("-X-")
        if len(parts) < 2:
            continue
        token = parts[0].strip()
        label_part = parts[1].strip().split()
        if len(label_part) < 2:
            continue
        label = label_part[1]

        # Add token
        current_tokens.append(token)

        # Define ner_tag
        if label == "O":
            current_tags.append(0)
        else:
            prefix, entity_type = label.split("-", 1)
            if entity_type in ENTITY_MAPPING:
                tag_b, tag_i = ENTITY_MAPPING[entity_type]
                current_tags.append(tag_b if prefix == "B" else tag_i)
            else:
                current_tags.append(0)  # If an unknown entity is found, set it to O

    # Add the last sentence if any
    if current_tokens:
        sentences.append(
            {"id": str(current_id), "tokens": current_tokens, "ner_tags": current_tags}
        )

    return sentences


# Define the path of the data file
input_file = (
    INTERIM_DATA_PATH / "bootstrapping/001/project-6-at-2025-03-03-12-04-d2bbce64.conll"
)

# Check if the file exists
if not input_file.exists():
    raise FileNotFoundError(f"File not found: {input_file}")

# Read data from the file
with open(input_file, "r", encoding="utf-8") as f:
    raw_data = f.read()

# Convert data
processed_data = process_labelstudio_to_ner_format(raw_data)

# Display sample results
for sentence in processed_data[:2]:
    print(f"ID: {sentence['id']}")
    print(f"Tokens: {sentence['tokens']}")
    print(f"NER Tags: {sentence['ner_tags']}")
    print()

# Split Train and Validate
random.shuffle(processed_data)  # Shuffle data
train_size = int(0.8 * len(processed_data))  # 80% for Train
train_data = processed_data[:train_size]
validate_data = processed_data[train_size:]

# Display data size
print(f"Train data size: {len(train_data)}")
print(f"Validate data size: {len(validate_data)}")

# Create directories for saving data if not exist
INTERIM_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Save data as JSON
with open(
    INTERIM_DATA_PATH / "./bootstrapping/train-001/train_data.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
with open(
    INTERIM_DATA_PATH / "./bootstrapping/train-001/validate_data.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(validate_data, f, ensure_ascii=False, indent=2)

print("Data processing completed and saved successfully!")
