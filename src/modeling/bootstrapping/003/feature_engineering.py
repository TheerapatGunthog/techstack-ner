import os
import json
from pathlib import Path
from collections import Counter
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from transformers import AutoTokenizer

# ===============================
# Config
# ===============================
PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")  # Changeable
# Use files uploaded in this session
TRAIN_FILE = PROJECT_PATH / (
    "data/interim/bootstrapping/003/03/train_augmented_v6_fixed.pos.relabeled.conll"
)
DEV_FILE = PROJECT_PATH / (
    "data/interim/bootstrapping/003/03/dev_large.pos.relabeled.conll"
)
TEST_FILE = PROJECT_PATH / (
    "data/interim/bootstrapping/003/03/test_large.pos.relabeled.conll"
)

MODEL_NAME = "FacebookAI/roberta-base"
OUTPUT_DIR = PROJECT_PATH / "data/processed/bootstrapping/003/"


# ===============================
# Utils: CoNLL Parser (auto-detect columns)
# ===============================
def parse_conll_file(file_path):
    """
    Supports two formats:
      - 2 columns: token, ner
      - 4+ columns: token, pos, chunk, ner (ner = last column)
    Skips empty lines, comments (#...), and -DOCSTART-
    """
    sentences = []
    cur_tok, cur_pos, cur_chunk, cur_ner = [], [], [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            s = line.strip()

            if not s:
                if cur_tok:
                    sentences.append(
                        {
                            "id": str(len(sentences)),
                            "tokens": cur_tok.copy(),
                            # May be empty if not present in the file
                            "pos_tags": cur_pos.copy() if cur_pos else None,
                            "chunk_tags": cur_chunk.copy() if cur_chunk else None,
                            "ner_tags": cur_ner.copy(),
                        }
                    )
                    cur_tok.clear()
                    cur_pos.clear()
                    cur_chunk.clear()
                    cur_ner.clear()
                continue

            if s.startswith("#") or s.upper().startswith("-DOCSTART-"):
                continue

            # Split by general whitespace
            parts = s.split()
            if len(parts) < 2:
                # Skip malformed lines
                continue

            token = parts[0]
            if len(parts) == 2:
                # token, ner
                ner = parts[1]
                cur_tok.append(token)
                cur_ner.append(ner)
            else:
                # token, pos, chunk, ..., ner (last column)
                ner = parts[-1]
                pos = parts[1]
                chunk = parts[2]
                cur_tok.append(token)
                cur_pos.append(pos)
                cur_chunk.append(chunk)
                cur_ner.append(ner)

    # Finalize file if it doesn't end with an empty line
    if cur_tok:
        sentences.append(
            {
                "id": str(len(sentences)),
                "tokens": cur_tok.copy(),
                "pos_tags": cur_pos.copy() if cur_pos else None,
                "chunk_tags": cur_chunk.copy() if cur_chunk else None,
                "ner_tags": cur_ner.copy(),
            }
        )

    return sentences


def coalesce_fields(sentences_list):
    """
    Normalize the schema of every sentence:
    - If at least one file has pos/chunk, create that field in every example
    - If missing, fill with an empty list of the same length as tokens
    """
    any_pos = any(s["pos_tags"] is not None for s in sentences_list)
    any_chunk = any(s["chunk_tags"] is not None for s in sentences_list)
    for s in sentences_list:
        n = len(s["tokens"])
        if any_pos and s["pos_tags"] is None:
            s["pos_tags"] = [""] * n
        if any_chunk and s["chunk_tags"] is None:
            s["chunk_tags"] = [""] * n
    return any_pos, any_chunk


def extract_unique_labels(*sentences_groups):
    labels = set()
    for group in sentences_groups:
        for s in group:
            labels.update(s["ner_tags"])
    return sorted(labels)


# ===============================
# Load all splits
# ===============================
print(f"Parsing: {TRAIN_FILE}")
train_sents = parse_conll_file(TRAIN_FILE)
print(f"Parsing: {DEV_FILE}")
dev_sents = parse_conll_file(DEV_FILE)
print(f"Parsing: {TEST_FILE}")
test_sents = parse_conll_file(TEST_FILE)

# Combine to normalize schema
all_sents = train_sents + dev_sents + test_sents
has_pos, has_chunk = coalesce_fields(all_sents)

# Split back
train_sents = all_sents[: len(train_sents)]
dev_sents = all_sents[len(train_sents) : len(train_sents) + len(dev_sents)]
test_sents = all_sents[len(train_sents) + len(dev_sents) :]

# Labels
label_names = extract_unique_labels(train_sents, dev_sents, test_sents)
id2label = {i: lab for i, lab in enumerate(label_names)}
label2id = {lab: i for i, lab in enumerate(label_names)}
print(f"Labels: {label_names}")

# ===============================
# Build Features
# ===============================
feature_dict = {
    "id": Value("string"),
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(ClassLabel(names=label_names)),
}
if has_pos:
    feature_dict["pos_tags"] = Sequence(Value("string"))
if has_chunk:
    feature_dict["chunk_tags"] = Sequence(Value("string"))

features = Features(feature_dict)


# ===============================
# Create HF Datasets
# ===============================
def to_dataset(sentences):
    # HuggingFace will map string->int based on ClassLabel when creating the dataset
    return Dataset.from_list(sentences, features=features)


raw_datasets = DatasetDict(
    {
        "train": to_dataset(train_sents),
        "validation": to_dataset(dev_sents),
        "test": to_dataset(test_sents),
    }
)

# ===============================
# Tokenization + Label alignment
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)


def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, ner_ids in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        aligned = []
        for w in word_ids:
            if w is None:
                aligned.append(-100)
            elif w != prev:
                aligned.append(ner_ids[w])  # ner_ids are already integers
            else:
                aligned.append(-100)
            prev = w
        labels.append(aligned)
    tokenized["labels"] = labels
    return tokenized


processed = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

print("Tokenization done.")


# ===============================
# Stats
# ===============================
def label_stats(ds_split, name):
    all_labels = []
    for ex in ds_split:
        all_labels.extend([x for x in ex["labels"] if x != -100])
    cnt = Counter(all_labels)
    if not cnt:
        print(f"[{name}] no labels")
        return
    df = (
        pd.DataFrame(
            {
                "Label Name": [id2label[i] for i in cnt.keys()],
                "Count": list(cnt.values()),
            }
        )
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n[{name}] Label Distribution (incl. 'O'):\n{df.to_string(index=False)}")
    df_wo_o = df[df["Label Name"] != "O"]
    if not df_wo_o.empty:
        print(f"\n[{name}] Entities only:\n{df_wo_o.to_string(index=False)}")
    else:
        print(f"\n[{name}] No entities other than 'O'.")


print(
    f"\nSamples: train={len(processed['train'])}  dev={len(processed['validation'])}  test={len(processed['test'])}"
)
label_stats(processed["train"], "Train")
label_stats(processed["validation"], "Dev")
label_stats(processed["test"], "Test")

# ===============================
# Save to disk
# ===============================
os.makedirs(OUTPUT_DIR, exist_ok=True)
processed.save_to_disk(str(OUTPUT_DIR))
with open(OUTPUT_DIR / "label_mappings.json", "w", encoding="utf-8") as f:
    json.dump(
        {"id2label": id2label, "label2id": label2id, "label_names": label_names},
        f,
        indent=2,
        ensure_ascii=False,
    )
print(f"\nSaved to: {OUTPUT_DIR}")
print("label_mappings.json written.")


# ===============================
# Quick sample view
# ===============================
def show_samples(sentences, k=3):
    print("\n[Sample Data Verification]")
    for i, s in enumerate(sentences[:k]):
        print(f"\nSentence {i+1}:")
        print(f"Tokens: {s['tokens'][:15]}...")
        print(f"NER   : {s['ner_tags'][:15]}...")


show_samples(train_sents, k=3)
