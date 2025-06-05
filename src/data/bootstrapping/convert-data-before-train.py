import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from data.interim import INTERIM_DATA_PATH
from data.processed import PROCESS_DATA_PATH
import random
from typing import List, Dict
import json
from transformers import AutoTokenizer

# Initialize RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

# Define mapping for entity types (using standard BIO tagging)
ENTITY_MAPPING = {
    "CLOUDPLATFORM": ("B-CLOUDPLATFORM", "I-CLOUDPLATFORM"),
    "PROGRAMMINGLANG": ("B-PROGRAMMINGLANG", "I-PROGRAMMINGLANG"),
    "FRAMEWORK_LIB": ("B-FRAMEWORK_LIB", "I-FRAMEWORK_LIB"),
    "WEBFRAMEWORK_TECH": ("B-WEBFRAMEWORK_TECH", "I-WEBFRAMEWORK_TECH"),
    "DATABASE": ("B-DATABASE", "I-DATABASE"),
    "EMBEDDEDTECH": ("B-EMBEDDEDTECH", "I-EMBEDDEDTECH"),
}

# Create label to id mapping
label_list = ["O"]
for entity_type in ENTITY_MAPPING:
    b_tag, i_tag = ENTITY_MAPPING[entity_type]
    label_list.extend([b_tag, i_tag])

label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}


def align_labels_with_tokens(tokens, labels, tokenizer):
    """
    Align labels with tokenized inputs for RoBERTa
    """
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        padding=False,
        return_offsets_mapping=True,
    )

    word_ids = tokenized_inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens (CLS, SEP, PAD)
            aligned_labels.append(-100)  # Ignore index for loss calculation
        elif word_idx != previous_word_idx:
            # First subword of a word
            aligned_labels.append(labels[word_idx])
        else:
            # Subsequent subwords of the same word
            # Use -100 to ignore or use the same label
            aligned_labels.append(-100)
        previous_word_idx = word_idx

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": aligned_labels,
    }


def process_labelstudio_to_roberta_format(raw_data: str) -> List[Dict]:
    lines = raw_data.strip().split("\n")

    sentences = []
    current_id = 0
    current_tokens = []
    current_labels = []

    for line in lines:
        if line.strip() == "":
            # End of a sentence
            if current_tokens:
                # Align labels with RoBERTa tokenization
                aligned_data = align_labels_with_tokens(
                    current_tokens, current_labels, tokenizer
                )

                sentences.append(
                    {
                        "id": current_id,
                        "tokens": current_tokens,
                        "input_ids": aligned_data["input_ids"],
                        "attention_mask": aligned_data["attention_mask"],
                        "labels": aligned_data["labels"],
                    }
                )

                current_id += 1
                current_tokens = []
                current_labels = []
            continue

        # Parse token and label
        parts = line.split("-X-")
        if len(parts) < 2:
            continue
        token = parts[0].strip()
        label_part = parts[1].strip().split()
        if len(label_part) < 2:
            continue
        label = label_part[1]

        current_tokens.append(token)

        # Convert label to ID
        if label == "O":
            current_labels.append(label_to_id["O"])
        else:
            prefix, entity_type = label.split("-", 1)
            if entity_type in ENTITY_MAPPING:
                b_tag, i_tag = ENTITY_MAPPING[entity_type]
                full_label = b_tag if prefix == "B" else i_tag
                current_labels.append(label_to_id[full_label])
            else:
                current_labels.append(label_to_id["O"])  # Unknown entity → O

    # Process last sentence
    if current_tokens:
        aligned_data = align_labels_with_tokens(
            current_tokens, current_labels, tokenizer
        )

        sentences.append(
            {
                "id": current_id,
                "tokens": current_tokens,
                "input_ids": aligned_data["input_ids"],
                "attention_mask": aligned_data["attention_mask"],
                "labels": aligned_data["labels"],
            }
        )

    return sentences


def save_dataset_for_roberta(data: List[Dict], output_path: Path):
    """
    Save dataset in format optimized for RoBERTa training
    """
    # Format for Hugging Face datasets
    formatted_data = {
        "input_ids": [item["input_ids"] for item in data],
        "attention_mask": [item["attention_mask"] for item in data],
        "labels": [item["labels"] for item in data],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)


# Define input file path
input_file = (
    INTERIM_DATA_PATH / "bootstrapping/001/project-6-at-2025-03-03-12-04-d2bbce64.conll"
)

# Verify file exists
if not input_file.exists():
    raise FileNotFoundError(f"File not found: {input_file}")

# Read and process data
with open(input_file, "r", encoding="utf-8") as f:
    raw_data = f.read()

processed_data = process_labelstudio_to_roberta_format(raw_data)

# Display sample results
print("Sample processed data:")
for i, sentence in enumerate(processed_data[:2]):
    print(f"\nSentence {i}:")
    print(f"Original tokens: {sentence['tokens']}")
    print(f"Input IDs length: {len(sentence['input_ids'])}")
    print(f"Labels length: {len(sentence['labels'])}")
    print(f"First 10 input_ids: {sentence['input_ids'][:10]}")
    print(f"First 10 labels: {sentence['labels'][:10]}")

# Split data (70% train, 15% validation, 15% test)
random.shuffle(processed_data)
total_size = len(processed_data)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)

train_data = processed_data[:train_size]
val_data = processed_data[train_size : train_size + val_size]
test_data = processed_data[train_size + val_size :]

print("\nDataset sizes:")
print(f"Train: {len(train_data)}")
print(f"Validation: {len(val_data)}")
print(f"Test: {len(test_data)}")

# Create output directories
output_dir = PROCESS_DATA_PATH / "bootstrapping/001"
output_dir.mkdir(parents=True, exist_ok=True)

# Save datasets in RoBERTa-optimized format
save_dataset_for_roberta(train_data, output_dir / "train.json")
save_dataset_for_roberta(val_data, output_dir / "validation.json")
save_dataset_for_roberta(test_data, output_dir / "test.json")

# Save label mappings for later use
label_info = {
    "label_to_id": label_to_id,
    "id_to_label": id_to_label,
    "num_labels": len(label_list),
}

with open(output_dir / "label_mappings.json", "w", encoding="utf-8") as f:
    json.dump(label_info, f, ensure_ascii=False, indent=2)

# Save tokenizer configuration
tokenizer.save_pretrained(output_dir / "tokenizer")

print("\nData processing completed!")
print(f"Files saved to: {output_dir}")
print("- train.json, validation.json, test.json")
print("- label_mappings.json")
print("- tokenizer/ (RoBERTa tokenizer config)")
print("\nLabel mappings:")
for label, idx in label_to_id.items():
    print(f"  {label}: {idx}")
