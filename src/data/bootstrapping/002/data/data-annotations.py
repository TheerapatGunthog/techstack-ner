import sys
import pandas as pd
from pathlib import Path
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
import torch
from tqdm import tqdm
import json
from collections import Counter

# Add parent directory to system path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

from data.interim import INTERIM_DATA_PATH
from models import MODELS_DATA_PATH

# Define the path to your custom trained model
final_model_path = MODELS_DATA_PATH / "bootstrapping001/final_model"

# Load tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained(final_model_path)
model = RobertaForTokenClassification.from_pretrained(final_model_path)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load data from CSV
df = pd.read_csv(INTERIM_DATA_PATH / "segmented_data.csv")


def predict_entities(text):
    """
    Predict entities, combine B- and I- labels into a single entity, and remove B-/I- prefixes.
    """
    try:
        # Step 1: Tokenize the input
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Store offset_mapping separately and remove it from inputs
        offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
        word_ids = inputs.word_ids(batch_index=0)

        # Move inputs to the same device as the model
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Convert logits to predictions
        predictions = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()

        # Align predictions with original words
        aligned_predictions = []
        prev_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:  # Skip special tokens like [CLS], [SEP]
                continue
            elif word_idx != prev_word_idx:  # Only take prediction for first subtoken
                aligned_predictions.append(predictions[i])
            prev_word_idx = word_idx

        # Convert predictions to label strings
        id2label = model.config.id2label
        predicted_labels = [id2label[pred] for pred in aligned_predictions]

        # Split text into words for alignment
        words = text.split()
        min_length = min(len(words), len(predicted_labels))
        words = words[:min_length]
        predicted_labels = predicted_labels[:min_length]

        # Combine B- and I- labels into single entities
        refined_labels = []
        current_entity = None
        current_start = None
        current_end = None
        current_text = []

        char_pos = 0
        for i, (word, label) in enumerate(zip(words, predicted_labels)):
            start = text.index(word, char_pos)
            end = start + len(word)
            char_pos = end

            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    refined_labels.append(
                        {
                            "entity": current_entity,
                            "start": current_start,
                            "end": current_end,
                            "text": " ".join(current_text),
                        }
                    )
                # Start new entity
                current_entity = label[2:]  # Remove "B-" prefix
                current_start = start
                current_end = end
                current_text = [word]
            elif label.startswith("I-") and current_entity == label[2:]:
                # Continue current entity
                current_end = end
                current_text.append(word)
            elif label == "O" or (
                label.startswith("I-") and current_entity != label[2:]
            ):
                # Save previous entity if exists and reset
                if current_entity:
                    refined_labels.append(
                        {
                            "entity": current_entity,
                            "start": current_start,
                            "end": current_end,
                            "text": " ".join(current_text),
                        }
                    )
                current_entity = None
                current_start = None
                current_end = None
                current_text = []

        # Save the last entity if exists
        if current_entity:
            refined_labels.append(
                {
                    "entity": current_entity,
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_text),
                }
            )

        return refined_labels
    except Exception as e:
        print(f"Error processing text: {text[:50]}... - {str(e)}")
        return []


# Initialize list to store results in Label Studio format
label_studio_data = []
all_entities = []
idcount = 0

# Process each row in the Segmented_Qualification column
for index, row in tqdm(
    df.iterrows(), total=len(df), desc="Processing Segmented_Qualification"
):
    text = row["Segmented_Qualification"]
    if pd.notna(text) and text.strip():
        refined_labels = predict_entities(text)
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

# Define output path and save results as JSON
output_path = INTERIM_DATA_PATH / "bootstrapping/002/labeled_by_custom_model.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(label_studio_data, f, ensure_ascii=False, indent=4)

print(f"Data saved to {output_path}")

# Count the frequency of each unique entity
entity_counts = Counter(all_entities)

# Display unique entities and their counts
print("\nUnique entities and their counts:")
for entity, count in entity_counts.items():
    print(f"{entity}: {count}")
