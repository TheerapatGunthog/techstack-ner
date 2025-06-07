import pandas as pd
from pathlib import Path
from transformers import pipeline
from collections import Counter
import json
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import logging

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# KEYWORDS_DATA_PATH = Path("/home/whilebell/Code/Project/TechStack-NER/data/keywords")
INTERIM_DATA_PATH = Path("/home/whilebell/Code/Project/TechStack-NER/data/interim")

# Load the zero-shot classification model
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=0,
)

# Define candidate labels for classification
candidate_labels = [
    "Programming_Scripting_and_Markup_languages",
    "Cloud_platforms",
    "Database",
    "Tech_Framework_and_libraries",
    "Embedded_Technologies",
    "Other_Technology_tools",  # for other technology tools and software
    "other_word",  # for non-technology related words
]

# Mapping to shorter labels
label_mapping = {
    "Programming_Scripting_and_Markup_languages": "PSML",
    "Cloud_platforms": "CP",
    "Database": "DB",
    "Tech_Framework_and_libraries": "TFL",
    "Embedded_Technologies": "ET",
    "Other_Technology_tools": "OTT",  # New category
    "other_word": "OTHER",  # Non-tech words
}

df = pd.read_csv(INTERIM_DATA_PATH / "segmented-data/kaggle-segmented-data.csv")

print(df)

# Setup logging for medium confidence predictions (0.48 - 0.6)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            INTERIM_DATA_PATH / "medium_confidence_predictions_0.48-0.6.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def is_word_worth_classifying(word):
    """Filter out only obvious non-tech words, let model decide the rest"""
    word_lower = word.lower()

    # Skip very short words
    if len(word) < 2:
        return False

    # Skip common stop words and articles
    common_words = {
        # Articles and basic words
        "the",
        "and",
        "or",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "here",
        "there",
        "where",
        "when",
        "why",
        "how",
        "but",
        "if",
        "because",
        "so",
        "than",
        "as",
        "very",
        "just",
        "only",
        "also",
        "can",
        "may",
        "must",
        "shall",
        "might",
        # Job qualification common words
        "experience",
        "experienced",
        "work",
        "working",
        "worked",
        "job",
        "role",
        "position",
        "years",
        "year",
        "month",
        "months",
        "day",
        "days",
        "time",
        "times",
        "knowledge",
        "skill",
        "skills",
        "ability",
        "abilities",
        "competency",
        "competencies",
        "proficient",
        "proficiency",
        "familiar",
        "familiarity",
        "understanding",
        "know",
        "knows",
        "good",
        "excellent",
        "strong",
        "solid",
        "deep",
        "extensive",
        "comprehensive",
        "basic",
        "advanced",
        "intermediate",
        "beginner",
        "expert",
        "senior",
        "junior",
        "required",
        "prefer",
        "preferred",
        "desirable",
        "essential",
        "must",
        "should",
        "including",
        "include",
        "includes",
        "such",
        "like",
        "similar",
        "related",
        "plus",
        "bonus",
        "advantage",
        "asset",
        "benefit",
        "nice",
        "ideal",
        # Business/organizational words
        "team",
        "teams",
        "project",
        "projects",
        "client",
        "clients",
        "customer",
        "customers",
        "business",
        "company",
        "organization",
        "department",
        "stakeholder",
        "stakeholders",
        "environment",
        "environments",
        "solution",
        "solutions",
        "system",
        "systems",
        "process",
        "processes",
        "methodology",
        "methodologies",
        "practice",
        "practices",
        # Education/certification words
        "degree",
        "bachelor",
        "master",
        "phd",
        "diploma",
        "certificate",
        "certification",
        "course",
        "courses",
        "training",
        "education",
        "university",
        "college",
        "school",
        "graduate",
        "graduated",
        "study",
        "studied",
        "learn",
        "learned",
        "learning",
        # General descriptors
        "new",
        "old",
        "latest",
        "current",
        "modern",
        "traditional",
        "standard",
        "custom",
        "large",
        "small",
        "big",
        "little",
        "high",
        "low",
        "fast",
        "slow",
        "quick",
        "easy",
        "difficult",
        "hard",
        "simple",
        "complex",
        "complicated",
        "important",
        "critical",
        "key",
        "main",
        "primary",
        "secondary",
        "additional",
        # Common conjunctions and transitions
        "however",
        "therefore",
        "moreover",
        "furthermore",
        "additionally",
        "meanwhile",
        "otherwise",
        "nevertheless",
        "consequently",
        "accordingly",
        "thus",
        "hence",
        "although",
        "though",
        "while",
        "whereas",
        "unless",
        "until",
        "since",
        "before",
        "after",
    }

    if word_lower in common_words:
        return False

    # Skip pure numbers
    if word.isdigit():
        return False

    # Skip pure punctuation
    if not any(c.isalnum() for c in word):
        return False

    # Let model decide everything else - including potential tech terms
    return True


def classify_words_in_text(text, row_id):
    """Classify words with minimal filtering - let model decide tech relevance"""
    words = word_tokenize(text)
    classified_entities = []

    for i, word in enumerate(words):
        # Only filter out obvious non-tech words
        if not is_word_worth_classifying(word):
            continue

        try:
            # Classify the word - let model decide if it's tech or not
            result = classifier(word, candidate_labels)

            # Get the top prediction
            top_label = result["labels"][0]
            confidence = result["scores"][0]

            # Log medium confidence predictions (0.48 - 0.6 range)
            if 0.48 <= confidence <= 0.6 and top_label != "other_word":
                short_label_for_log = label_mapping.get(top_label, top_label)
                logger.info(
                    f"Word: '{word}', Confidence: {confidence:.3f}, Class: {short_label_for_log}, Row ID: {row_id}"
                )

            # Only keep predictions for tech categories (not 'other_word') with confidence >= 0.48
            if top_label != "other_word" and confidence >= 0.48:
                # Map to shorter label
                short_label = label_mapping.get(top_label, top_label)

                # Find word position in original text
                word_start = text.find(word)
                if word_start != -1:
                    word_end = word_start + len(word)

                    classified_entities.append(
                        {
                            "entity": short_label,
                            "start": word_start,
                            "end": word_end,
                            "text": word,
                            "confidence": confidence,
                        }
                    )

        except Exception as e:
            logger.error(f"Error classifying word '{word}' in row {row_id}: {str(e)}")
            continue

    return classified_entities


def remove_overlapping_entities(entities):
    """Remove overlapping entities, keeping the one with higher confidence"""
    if not entities:
        return entities

    # Sort by start position, then by confidence (descending)
    entities.sort(key=lambda x: (x["start"], -x["confidence"]))

    filtered_entities = []
    for entity in entities:
        # Check if this entity overlaps with any already added entity
        overlaps = False
        for existing in filtered_entities:
            if entity["start"] < existing["end"] and entity["end"] > existing["start"]:
                # There's an overlap, keep the one with higher confidence
                if entity["confidence"] > existing["confidence"]:
                    # Remove the existing entity and add the new one
                    filtered_entities.remove(existing)
                    break
                else:
                    # Keep the existing entity, skip the new one
                    overlaps = True
                    break

        if not overlaps:
            filtered_entities.append(entity)

    return filtered_entities


# Create a list to store the results in Label Studio format
label_studio_data = []
all_entities = []
idcount = 0

# Process each row using Segmented_Qualification column1
for index, row in tqdm(
    df.iterrows(), total=len(df), desc="Processing Segmented_Qualification"
):
    text = row["Segmented_Qualification"]
    if pd.notna(text) and text.strip():  # Check if text is not NaN or empty
        # Classify words in the text
        classified_entities = classify_words_in_text(text, str(idcount))

        # Remove overlapping entities
        refined_labels = remove_overlapping_entities(classified_entities)

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

# Save the results as JSON
output_path = (
    INTERIM_DATA_PATH / "./bootstrapping/001/kaggle-data/labels-by-code-001.json"
)
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

print(
    f"\nMedium confidence predictions (0.48-0.6) logged to: {INTERIM_DATA_PATH / 'medium_confidence_predictions_0.48-0.6.log'}"
)
