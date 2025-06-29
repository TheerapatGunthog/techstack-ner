import sys
import pandas as pd
from pathlib import Path
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
import torch
from tqdm import tqdm
import json
from collections import Counter
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("ner_labeling.log")],
)
logger = logging.getLogger(__name__)

# Add parent directory to system path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

try:
    from data.interim import INTERIM_DATA_PATH
    from models import MODELS_DATA_PATH
except ImportError:
    logger.warning("Could not import paths from project. Using relative paths.")
    # Fallback paths if imports fail
    INTERIM_DATA_PATH = Path("./data/interim")
    MODELS_DATA_PATH = Path("./models")

# Configuration
CONFIG = {
    "model_path": MODELS_DATA_PATH / "bootstrapping001/final_model",
    "input_csv": INTERIM_DATA_PATH / "segmented_data.csv",
    "output_path": INTERIM_DATA_PATH
    / "bootstrapping/test-002/labeled_by_custom_model.json",
    "batch_size": 8,  # Process texts in batches for better performance
    "max_length": 512,
    "text_column": "Segmented_Qualification",
    "confidence_threshold": 0.7,  # Minimum confidence score to accept a prediction
}


def setup_model_and_tokenizer() -> Tuple[
    RobertaForTokenClassification, RobertaTokenizerFast, torch.device
]:
    """
    Set up the model, tokenizer and determine the device.

    Returns:
        Tuple containing model, tokenizer and device
    """
    logger.info(f"Loading model from {CONFIG['model_path']}")

    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(CONFIG["model_path"])
        model = RobertaForTokenClassification.from_pretrained(CONFIG["model_path"])

        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model.to(device)
        model.eval()

        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def load_data() -> pd.DataFrame:
    """
    Load and validate input data.

    Returns:
        DataFrame containing input data
    """
    logger.info(f"Loading data from {CONFIG['input_csv']}")

    try:
        df = pd.read_csv(CONFIG["input_csv"]).iloc[:1000]

        # Validate that required column exists
        if CONFIG["text_column"] not in df.columns:
            raise ValueError(
                f"Required column '{CONFIG['text_column']}' not found in input CSV"
            )

        # Remove rows with empty text
        original_count = len(df)
        df = df.dropna(subset=[CONFIG["text_column"]])
        df = df[df[CONFIG["text_column"]].str.strip().astype(bool)]

        logger.info(
            f"Loaded {len(df)} valid records (removed {original_count - len(df)} empty records)"
        )
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def batch_tokenize(
    texts: List[str], tokenizer, device
) -> List[Dict[str, torch.Tensor]]:
    """
    Tokenize a batch of texts efficiently.

    Args:
        texts: List of text strings to tokenize
        tokenizer: The tokenizer to use
        device: The device to place tensors on

    Returns:
        List of dictionaries containing tokenized inputs
    """
    batched_inputs = []

    for text in texts:
        # Tokenize individual text
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Store offset_mapping separately and remove from inputs that go to the model
        offset_mapping = inputs.pop("offset_mapping")[0].cpu()
        word_ids = inputs.word_ids(batch_index=0)

        # Move inputs to device
        device_inputs = {key: val.to(device) for key, val in inputs.items()}

        batched_inputs.append(
            {
                "text": text,
                "inputs": device_inputs,
                "offset_mapping": offset_mapping,
                "word_ids": word_ids,
            }
        )

    return batched_inputs


def predict_entities_batch(
    batched_inputs: List[Dict], model, id2label: Dict[int, str]
) -> List[List[Dict]]:
    """
    Process a batch of texts and extract entities with confidence scores.

    Args:
        batched_inputs: List of dictionaries with tokenized inputs
        model: The NER model
        id2label: Mapping from label IDs to label strings

    Returns:
        List of lists of entity dictionaries
    """
    batch_entities = []

    with torch.no_grad():
        for item in batched_inputs:
            text = item["text"]
            inputs = item["inputs"]
            offset_mapping = item["offset_mapping"]
            word_ids = item["word_ids"]

            try:
                # Get model outputs
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                # Get predictions and confidence scores
                predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
                confidence_scores = torch.max(probabilities, dim=2)[0][0].cpu().numpy()

                # Process predictions to extract entities
                entities = extract_entities_from_predictions(
                    text,
                    predictions,
                    confidence_scores,
                    word_ids,
                    offset_mapping,
                    id2label,
                )

                batch_entities.append(entities)
            except Exception as e:
                logger.error(f"Error processing text: {text[:50]}... - {str(e)}")
                batch_entities.append([])

    return batch_entities


def extract_entities_from_predictions(
    text: str,
    predictions: List[int],
    confidence_scores: List[float],
    word_ids: List[Optional[int]],
    offset_mapping: torch.Tensor,
    id2label: Dict[int, str],
) -> List[Dict[str, Any]]:
    """
    Extract entity spans from token-level predictions.

    Args:
        text: Original text string
        predictions: Model predictions (label IDs)
        confidence_scores: Confidence scores for each prediction
        word_ids: Mapping from tokens to words
        offset_mapping: Character offsets for each token
        id2label: Mapping from label IDs to label strings

    Returns:
        List of entity dictionaries
    """
    # Words and their corresponding predictions and confidence scores
    word_preds = []
    word_confs = []
    word_offsets = []

    prev_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:  # Skip special tokens
            continue
        elif word_idx != prev_word_idx:  # Only take first subtoken
            word_preds.append(predictions[i])
            word_confs.append(confidence_scores[i])
            word_offsets.append(
                (offset_mapping[i][0].item(), offset_mapping[i][1].item())
            )
        prev_word_idx = word_idx

    # Convert to label strings
    predicted_labels = [id2label[pred] for pred in word_preds]

    # Split text into words for reference
    # Using regex to handle whitespace better
    words = re.findall(r"\S+", text)

    # Make sure our arrays have the same length
    min_length = min(
        len(words), len(predicted_labels), len(word_confs), len(word_offsets)
    )
    words = words[:min_length]
    predicted_labels = predicted_labels[:min_length]
    word_confs = word_confs[:min_length]
    word_offsets = word_offsets[:min_length]

    # Extract entity spans
    entities = []
    current_entity = None
    current_start = None
    current_end = None
    current_text = []
    current_conf_sum = 0
    current_token_count = 0

    for i, (word, label, conf, offset) in enumerate(
        zip(words, predicted_labels, word_confs, word_offsets)
    ):
        char_start, char_end = offset

        if label.startswith("B-"):
            # Save previous entity if it exists
            if current_entity and current_token_count > 0:
                avg_conf = current_conf_sum / current_token_count
                if avg_conf >= CONFIG["confidence_threshold"]:
                    entities.append(
                        {
                            "entity": current_entity,
                            "start": current_start,
                            "end": current_end,
                            "text": " ".join(current_text),
                            "confidence": avg_conf,
                        }
                    )

            # Start new entity
            current_entity = label[2:]  # Remove "B-" prefix
            current_start = char_start
            current_end = char_end
            current_text = [word]
            current_conf_sum = conf
            current_token_count = 1

        elif label.startswith("I-") and current_entity == label[2:]:
            # Continue current entity
            current_end = char_end
            current_text.append(word)
            current_conf_sum += conf
            current_token_count += 1

        elif label == "O" or (label.startswith("I-") and current_entity != label[2:]):
            # Save previous entity if it exists
            if current_entity and current_token_count > 0:
                avg_conf = current_conf_sum / current_token_count
                if avg_conf >= CONFIG["confidence_threshold"]:
                    entities.append(
                        {
                            "entity": current_entity,
                            "start": current_start,
                            "end": current_end,
                            "text": " ".join(current_text),
                            "confidence": avg_conf,
                        }
                    )
            # Reset
            current_entity = None
            current_start = None
            current_end = None
            current_text = []
            current_conf_sum = 0
            current_token_count = 0

    # Save the last entity if it exists
    if current_entity and current_token_count > 0:
        avg_conf = current_conf_sum / current_token_count
        if avg_conf >= CONFIG["confidence_threshold"]:
            entities.append(
                {
                    "entity": current_entity,
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_text),
                    "confidence": avg_conf,
                }
            )

    return entities


def convert_to_label_studio_format(
    df: pd.DataFrame, all_entities: List[List[Dict]]
) -> List[Dict]:
    """
    Convert entities to Label Studio format.

    Args:
        df: DataFrame with original texts
        all_entities: List of entity lists for each text

    Returns:
        List of Label Studio format dictionaries
    """
    label_studio_data = []
    idcount = 0

    for (_, row), entities in zip(df.iterrows(), all_entities):
        text = row[CONFIG["text_column"]]

        if entities:
            # Create annotation in Label Studio format
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
                                        "start": entity["start"],
                                        "end": entity["end"],
                                        "text": entity["text"],
                                        "labels": [entity["entity"]],
                                    },
                                    "id": f"result-{idcount}-{i}",
                                    "from_name": "label",
                                    "to_name": "text",
                                    "type": "labels",
                                    "score": {
                                        "confidence": float(entity["confidence"])
                                    },
                                }
                                for i, entity in enumerate(entities)
                            ],
                        }
                    ],
                }
            )
            idcount += 1

    return label_studio_data


def save_output(label_studio_data: List[Dict], all_entities: List[Dict]) -> None:
    """
    Save results to disk and print statistics.

    Args:
        label_studio_data: Data in Label Studio format
        all_entities: Flattened list of all entities
    """
    # Create directory if it doesn't exist
    output_path = Path(CONFIG["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save Label Studio format data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(label_studio_data, f, ensure_ascii=False, indent=2)

    # Save statistics
    entity_counts = Counter([entity["entity"] for entity in all_entities])
    stats_path = output_path.parent / "entity_statistics.json"

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "entity_counts": {k: v for k, v in entity_counts.most_common()},
                "total_entities": len(all_entities),
                "total_documents": len(label_studio_data),
                "entities_per_document": len(all_entities) / len(label_studio_data)
                if label_studio_data
                else 0,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Print summary
    logger.info(f"Data saved to {output_path}")
    logger.info(f"Statistics saved to {stats_path}")
    logger.info(
        f"Found {len(all_entities)} entities across {len(label_studio_data)} documents"
    )

    logger.info("\nTop 10 entities and their counts:")
    for entity, count in entity_counts.most_common(10):
        logger.info(f"{entity}: {count}")


def main() -> None:
    """Main function to process data and apply NER labeling."""
    try:
        # Setup
        model, tokenizer, device = setup_model_and_tokenizer()
        id2label = model.config.id2label

        # Load data
        df = load_data()

        # Process in batches
        all_entities_per_text = []
        all_entities_flat = []

        # Create batches of texts
        texts = df[CONFIG["text_column"]].tolist()

        # Process in batches
        for i in tqdm(
            range(0, len(texts), CONFIG["batch_size"]), desc="Processing batches"
        ):
            batch_texts = texts[i : i + CONFIG["batch_size"]]
            batched_inputs = batch_tokenize(batch_texts, tokenizer, device)
            batch_entities = predict_entities_batch(batched_inputs, model, id2label)

            all_entities_per_text.extend(batch_entities)
            all_entities_flat.extend(
                [entity for entities in batch_entities for entity in entities]
            )

        # Convert to Label Studio format and save
        label_studio_data = convert_to_label_studio_format(df, all_entities_per_text)
        save_output(label_studio_data, all_entities_flat)

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
