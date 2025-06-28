import pandas as pd
from collections import Counter
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import os

# Step 1: Loading single data file and preparing labels

# Specify the path to your single CoNLL file here
single_file_path = "path/to/your_data.txt"

# Load all data into the "train" split first
raw_datasets = load_dataset("conll2003", data_files={"train": single_file_path})

# Get feature and label names
ner_feature = raw_datasets["train"].features["ner_tags"]
label_names = ner_feature.feature.names

# Create dictionaries for label-to-id and id-to-label mapping
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

print("Labels loaded successfully.")
print(f"Label names: {label_names}")
print("-" * 50)


# Step 2: Preprocess Data (Tokenization and Label Alignment)

tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)


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

print("\n" + "=" * 50)


# Step 5: Save Processed Dataset to Disk

# Specify the folder name to save the dataset
output_dir = "processed_ner_dataset"

# Create the folder if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Use the .save_to_disk() method
final_datasets.save_to_disk(output_dir)

print(f"\nDataset saved successfully to: '{output_dir}'")
print("You can now zip this folder and upload it to Kaggle. 🎉")
