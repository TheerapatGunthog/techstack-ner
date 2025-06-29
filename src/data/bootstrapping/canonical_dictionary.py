import json
from pathlib import Path

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")


INPUT_FILENAME = (
    PROJECT_PATH
    / "data/interim/bootstrapping/001/labeled-by-code-data/labels-by-gemini-001.json"
)


OUTPUT_FILENAME = (
    PROJECT_PATH / "data/keywords/canonoical_dictionary_gemini_labels.json"
)


def create_canonical_dictionary(data):
    """
    Scans through the dataset to create a canonical dictionary of entities.
    It normalizes entity text to lowercase to ensure consistency.
    """
    canonical_dict = {}

    # Loop through each item in the dataset
    for item in data:
        if "ner_labels" not in item:
            continue

        # Loop through each label in the item
        for label_info in item["ner_labels"]:
            # Normalize to lowercase
            entity_text = label_info["text"].lower()
            entity_label = label_info["label"]

            # If the term already exists with a different label, issue a warning
            # so you can decide which label to use
            if (
                entity_text in canonical_dict
                and canonical_dict[entity_text] != entity_label
            ):
                print(
                    f"⚠️  Warning: Inconsistent label for '{entity_text}'. "
                    f"Found '{entity_label}', but already have '{canonical_dict[entity_text]}'. "
                    "Overwriting with the new one."
                )

            # Add/update the entry in the dictionary
            canonical_dict[entity_text] = entity_label

    # Sort for readability
    sorted_dict = dict(sorted(canonical_dict.items()))
    return sorted_dict


# --- Main ---
if __name__ == "__main__":
    print("▶️ Starting file conversion process...")

    # 1. Read data from input file
    try:
        with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
            print(f"🔄 Reading data from '{INPUT_FILENAME}'...")
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File '{INPUT_FILENAME}' not found.")
        exit()
    except json.JSONDecodeError:
        print(f"❌ Error: File '{INPUT_FILENAME}' is not a valid JSON format.")
        exit()

    # Build the dictionary from the data
    canonical_dictionary = create_canonical_dictionary(input_data)

    print("\n✅  Canonical dictionary created successfully!")
    print("=" * 50)

    # Print the result as pretty JSON
    if canonical_dictionary is not None:
        try:
            with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
                json.dump(canonical_dictionary, f, ensure_ascii=False, indent=2)
            print(f"✅ Conversion successful! Result saved to '{OUTPUT_FILENAME}'")
        except IOError as e:
            print(f"❌ Error: Could not write file '{OUTPUT_FILENAME}': {e}")
