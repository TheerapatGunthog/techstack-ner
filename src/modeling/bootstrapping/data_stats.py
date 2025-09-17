import pandas as pd
from collections import Counter
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
)
import json
from pathlib import Path

# --- Simulate loading previously saved data (or you can modify to load from another source) ---
# For statistics only, we assume the dataset and label mappings have already been created and saved,
# and we will load them for analysis.

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")
processed_data_dir = PROJECT_PATH / "data/processed/bootstrapping/002/"

print(f"Loading processed dataset from: '{processed_data_dir}'")

# Load the saved Dataset
try:
    final_datasets = DatasetDict.load_from_disk(str(processed_data_dir))
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure the dataset has been saved to the specified path.")
    exit()

# Load label mappings
try:
    with open(processed_data_dir / "label_mappings.json", "r") as f:
        mappings = json.load(f)
    id2label = mappings["id2label"]
    label2id = mappings["label2id"]
    label_names = mappings["label_names"]
    print(f"Label mappings loaded successfully. Found {len(label_names)} labels.")
except Exception as e:
    print(f"Error loading label mappings: {e}")
    print("Please ensure 'label_mappings.json' exists in the processed data directory.")
    exit()

# Load tokenizer to convert token IDs back to words
# Must ensure the tokenizer used for pre-processing is the same
tokenizer = AutoTokenizer.from_pretrained("roberta-large", add_prefix_space=True)
print("Tokenizer loaded.")

print("\n" + "=" * 50)
print("DATASET STATISTICS")
print("=" * 50)

# 1. Show sample counts
print("\n[Sample Count]")
print(f"Training examples: {len(final_datasets['train'])}")
print(f"Validation examples: {len(final_datasets['validation'])}")

# 2. Show class information
num_labels = len(label_names)
print("\n[Class Info]")
print(f"Total number of classes: {num_labels}")
print(f"Class names: {label_names}")

# 3. Show class distribution in Training Set
print("\n[Complete Label Distribution in Training Set]")

# Collect all label ids and map them back to names for counting
all_labels_id = [
    label_id for example in final_datasets["train"] for label_id in example["labels"]
]
# Filter out -100 (special token for subword tokenization) before counting
valid_labels_id = [label for label in all_labels_id if label != -100]

# Convert label IDs back to their names for meaningful display
label_counts_named = Counter(
    [id2label[str(label_id)] for label_id in valid_labels_id]
)  # use str(label_id) because keys in id2label from json are strings

df_complete = pd.DataFrame(
    {
        "Label Name": label_counts_named.keys(),
        "Count": label_counts_named.values(),
    }
)

df_complete_sorted = df_complete.sort_values("Count", ascending=False).reset_index(
    drop=True
)
print(df_complete_sorted.to_string())

# 4. Show sentence length distribution (Sentence length distribution) in Training Set
# For sentence length, we count the number of token IDs that are not -100 (i.e., real words)
print("\n[Sentence Length Distribution (Tokenized Length) in Training Set]")
tokenized_lengths = [
    sum(1 for label_id in example["labels"] if label_id != -100)
    for example in final_datasets["train"]
]

if tokenized_lengths:
    min_len = min(tokenized_lengths)
    max_len = max(tokenized_lengths)
    avg_len = sum(tokenized_lengths) / len(tokenized_lengths)
    print(f"Minimum tokenized length: {min_len} tokens")
    print(f"Maximum tokenized length: {max_len} tokens")
    print(f"Average tokenized length: {avg_len:.2f} tokens")
else:
    print("No examples found in the training set to calculate length statistics.")

print("\n" + "=" * 50)

# 5. Show examples of preprocessed data (Sample Data Verification)
print("\n[Sample Data Verification - First 3 Processed Examples from Training Set]")
print("=" * 50)
for i in range(min(3, len(final_datasets["train"]))):  # show at most 3 examples
    example = final_datasets["train"][i]
    print(f"\nProcessed Example {i + 1}:")
    # Decode token_ids back to tokens for better readability
    decoded_tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    print(f"Tokenized Tokens (first 15): {decoded_tokens[:15]}...")
    # Map label IDs back to names for better readability, skipping -100
    decoded_labels = [
        id2label[str(lbl)] if lbl != -100 else "IGNORE" for lbl in example["labels"]
    ]  # use str(lbl)
    print(f"Aligned Labels (first 15): {decoded_labels[:15]}...")

print("\nFinished displaying dataset statistics.")
