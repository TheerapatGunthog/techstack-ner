import pandas as pd
from collections import Counter
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from transformers import AutoTokenizer
import os
import json
from pathlib import Path

# Step 1: Loading single data file and preparing labels

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

# Specify the path to your single CoNLL file here
single_file_path = (
    PROJECT_PATH
    / "data/interim/bootstrapping/001/project-13-at-2025-06-29-03-58-c919e3bc.conll"
)


def parse_conll_file(file_path):
    """Parse CoNLL format file directly"""
    sentences = []
    current_tokens = []
    current_pos_tags = []
    current_chunk_tags = []
    current_ner_tags = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:  # Empty line indicates end of sentence
                if current_tokens:
                    sentences.append(
                        {
                            "id": str(len(sentences)),
                            "tokens": current_tokens.copy(),
                            "pos_tags": current_pos_tags.copy(),
                            "chunk_tags": current_chunk_tags.copy(),
                            "ner_tags": current_ner_tags.copy(),
                        }
                    )
                    current_tokens.clear()
                    current_pos_tags.clear()
                    current_chunk_tags.clear()
                    current_ner_tags.clear()
            elif not line.startswith("#"):  # Skip comments
                parts = line.split()
                if len(parts) >= 4:
                    current_tokens.append(parts[0])
                    current_pos_tags.append(parts[1])
                    current_chunk_tags.append(parts[2])
                    current_ner_tags.append(parts[3])
                else:
                    print(f"Warning: Line {line_num} has insufficient columns: {line}")

    # Add the last sentence if file doesn't end with empty line
    if current_tokens:
        sentences.append(
            {
                "id": str(len(sentences)),
                "tokens": current_tokens.copy(),
                "pos_tags": current_pos_tags.copy(),
                "chunk_tags": current_chunk_tags.copy(),
                "ner_tags": current_ner_tags.copy(),
            }
        )

    return sentences


def extract_unique_labels(sentences):
    """Extract all unique NER labels from parsed sentences"""
    labels = set()
    for sentence in sentences:
        labels.update(sentence["ner_tags"])
    return sorted(list(labels))


print(f"Parsing CoNLL file: {single_file_path}")
sentences = parse_conll_file(single_file_path)
print(f"Found {len(sentences)} sentences")

# Extract actual labels from the file
label_names = extract_unique_labels(sentences)
print(f"Found labels: {label_names}")

# Create features structure using the actual labels
custom_features = Features(
    {
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "pos_tags": Sequence(Value("string")),
        "chunk_tags": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_names)),
    }
)

# Create dataset from parsed data
dataset = Dataset.from_list(sentences, features=custom_features)
raw_datasets = DatasetDict({"train": dataset})

# Create dictionaries for label-name to ID conversion and back
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

print("Labels loaded successfully.")
print(f"Label names: {label_names}")
print("-" * 50)

# Step 2: Preprocess Data (Tokenization and Label Alignment)

tokenizer = AutoTokenizer.from_pretrained("roberta-large", add_prefix_space=True)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


processed_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

print("Data tokenized and labels aligned.")
print("-" * 50)

# Step 3: Split Dataset into Train/Validation

train_val_split = processed_datasets["train"].train_test_split(test_size=0.1, seed=42)

final_datasets = DatasetDict(
    {"train": train_val_split["train"], "validation": train_val_split["test"]}
)

print("Split created successfully.")
print(f"\nFinal dataset structure:\n{final_datasets}")
print("-" * 50)

# Step 4: Display Key Dataset Statistics

# 1. Show sample counts
print("\n[Sample Count]")
print(f"Training examples: {len(final_datasets['train'])}")
print(f"Validation examples: {len(final_datasets['validation'])}")

# 2. Show class information
num_labels = len(label_names)
print("\n[Class Info]")
print(f"Total number of classes: {num_labels}")
print(f"Class names: {label_names}")

# 3. Show class distribution
print("\n[Class Distribution in Training Set (excluding 'O')]")

# Collect and count all labels
all_labels = [
    label for example in final_datasets["train"] for label in example["labels"]
]
filtered_labels = [label for label in all_labels if label != -100]
label_counts = Counter(filtered_labels)

# Create DataFrame for display
df = pd.DataFrame(
    {
        "Label Name": [id2label[id] for id in label_counts.keys()],
        "Count": label_counts.values(),
    }
)

# Filter out 'O' and sort
df_sorted = (
    df[df["Label Name"] != "O"]
    .sort_values("Count", ascending=False)
    .reset_index(drop=True)
)

if not df_sorted.empty:
    print(df_sorted.to_string())
else:
    print("No entities other than 'O' found in the training set.")

# Show all labels with counts (including O)
print("\n[Complete Label Distribution]")
df_complete = df.sort_values("Count", ascending=False).reset_index(drop=True)
print(df_complete.to_string())

print("\n" + "=" * 50)

# Step 5: Save Processed Dataset to Disk

# Specify the folder name to save the dataset
output_dir = PROJECT_PATH / "data/processed/bootstrapping/001/"

# Create the folder if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Use the .save_to_disk() method
final_datasets.save_to_disk(str(output_dir))

print(f"\nDataset saved successfully to: '{output_dir}'")
print("You can now zip this folder and upload it to Kaggle. 🎉")

# Save label mappings for future reference
mappings = {"id2label": id2label, "label2id": label2id, "label_names": label_names}

with open(output_dir / "label_mappings.json", "w") as f:
    json.dump(mappings, f, indent=2, ensure_ascii=False)

print(f"Label mappings saved to: {output_dir / 'label_mappings.json'}")

# Show some example sentences for verification
print("\n[Sample Data Verification]")
print("First 3 sentences from the dataset:")
for i, sentence in enumerate(sentences[:3]):
    print(f"\nSentence {i + 1}:")
    print(f"Tokens: {sentence['tokens'][:10]}...")  # Show first 10 tokens
    print(f"NER Tags: {sentence['ner_tags'][:10]}...")  # Show first 10 tags
