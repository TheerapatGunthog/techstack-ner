from transformers import RobertaTokenizerFast, RobertaForTokenClassification
import torch
import sys
from pathlib import Path

# Add path to parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

from models import MODELS_DATA_PATH

# Define model path
final_model_path = MODELS_DATA_PATH / "bootstrapping001/final_model"

# Load tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained(final_model_path)
model = RobertaForTokenClassification.from_pretrained(final_model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Sample data for prediction
new_sentence = "Core Values We are looking for candidates who align with our core values Integrity Be honest and ethical Do what is rightExcellence Hold ourselves to high standards Be thoughtful, thorough, and disciplinedCare Be respectful and inclusive Look after each other Contribute to the wellbeing of our communities and the environmentCourage Take initiative and make a difference Think boldly and act with conviction Take personal ownershipResilience Be determined and persevere Be purposeful and steadfast in our principles What you will do You will be responsible for providing administrative support to a department or individual."

# Step 1: Tokenize the input
inputs = tokenizer(
    new_sentence,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
    return_offsets_mapping=True,  # For mapping tokens back to words
)

# Store offset_mapping separately and remove it from inputs
offset_mapping = inputs.pop("offset_mapping")
word_ids = inputs.word_ids(batch_index=0)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

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
    elif (
        word_idx != prev_word_idx
    ):  # Only take prediction for first subtoken of each word
        aligned_predictions.append(predictions[i])
    prev_word_idx = word_idx

# Convert predictions to label strings
id2label = model.config.id2label
predicted_labels = [id2label[pred] for pred in aligned_predictions]

# Get the original words by splitting the sentence
words = new_sentence.split()

# Ensure lengths match (truncate if necessary)
min_length = min(len(words), len(predicted_labels))
words = words[:min_length]
predicted_labels = predicted_labels[:min_length]

# Display results
print("\nPrediction Results:")
print("-" * 50)
for word, label in zip(words, predicted_labels):
    print(f"{word}: {label}")
