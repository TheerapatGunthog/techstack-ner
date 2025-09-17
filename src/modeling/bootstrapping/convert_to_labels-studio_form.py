import json
from pathlib import Path

# ===================================================================
# Section to modify: Please specify your filenames here
# ===================================================================

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

# 1. Raw data file to be converted
INPUT_FILENAME = PROJECT_PATH / "data/interim/bootstrapping/002/clean_label.json"

# 2. [NEW] Main canonical dictionary file after filtering
CANONICAL_DICT_PATH = (
    PROJECT_PATH
    / "data/interim/bootstrapping/002/dictionary/dictionary.json"  # Edit Path here
)

# 3. Output file to be generated
OUTPUT_FILENAME = PROJECT_PATH / "data/interim/bootstrapping/002/studio_form.json"

# ===================================================================


# [MODIFIED] Function updated to accept additional `canonical_rules`
def convert_to_label_studio_format(data, canonical_rules):
    """
    Converts data to Label Studio format, enforcing consistency using a
    canonical dictionary.
    """
    label_studio_tasks = []

    if not isinstance(data, list):
        print("Error: Input data must be a List (JSON array)")
        return None

    for item_index, item in enumerate(data):
        results = []

        if "original_text" not in item or "ner_labels" not in item:
            print(f"Warning: Skipping item {item_index} due to missing keys.")
            continue

        original_text = item["original_text"]
        is_char_labeled = [False] * len(original_text)
        sorted_labels = sorted(
            item["ner_labels"], key=lambda x: len(x.get("text", "")), reverse=True
        )

        for label_info in sorted_labels:
            if not label_info.get("text"):
                continue

            label_text = label_info["text"]

            # Logic to enforce canonical labels
            normalized_text = label_text.lower()

            if normalized_text in canonical_rules:
                # If found in the dictionary, use the correct label
                correct_label = canonical_rules[normalized_text]
            else:
                # If not found, skip and notify
                # So we can later add this word to the canonical dictionary
                print(
                    f"Info: Skipping entity '{label_text}' as it's not in the canonical dictionary."
                )
                continue
            # End of new logic

            # Original logic for finding positions (unchanged)
            search_start_pos = 0
            new_start = -1
            while True:
                found_pos = original_text.find(label_text, search_start_pos)
                if found_pos == -1:
                    break
                if any(is_char_labeled[found_pos : found_pos + len(label_text)]):
                    search_start_pos = found_pos + 1
                    continue
                else:
                    new_start = found_pos
                    break

            if new_start == -1:
                continue

            new_end = new_start + len(label_text)

            results.append(
                {
                    "value": {
                        "start": new_start,
                        "end": new_end,
                        "text": label_text,
                        # Always use the correct label from the canonical dictionary
                        "labels": [correct_label],
                    },
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                }
            )

            for i in range(new_start, new_end):
                is_char_labeled[i] = True

        task = {
            "data": {"text": original_text},
            "annotations": [{"result": results}],
        }
        label_studio_tasks.append(task)

    return label_studio_tasks


# Main program section
if __name__ == "__main__":
    print("Starting data cleaning and conversion process...")

    # 1. Load canonical dictionary
    try:
        with open(CANONICAL_DICT_PATH, "r", encoding="utf-8") as f:
            print(f"Reading canonical rules from '{CANONICAL_DICT_PATH}'...")
            canonical_rules = json.load(f)
    except FileNotFoundError:
        print(f"Error: Canonical dictionary not found: '{CANONICAL_DICT_PATH}'")
        exit()
    except json.JSONDecodeError:
        print(f"Error: File '{CANONICAL_DICT_PATH}' is not a valid JSON.")
        exit()

    # 2. Load raw input data
    try:
        with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
            print(f"Reading raw data from '{INPUT_FILENAME}'...")
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: '{INPUT_FILENAME}'")
        exit()
    except json.JSONDecodeError:
        print(f"Error: File is not a valid JSON: '{INPUT_FILENAME}'")
        exit()

    # 3. Convert data using rules from the canonical dictionary
    converted_data = convert_to_label_studio_format(input_data, canonical_rules)

    # 4. Save output file
    if converted_data is not None:
        try:
            with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            print(f"Data cleaning successful! Result saved to '{OUTPUT_FILENAME}'")
        except IOError as e:
            print(f"Error: Could not write file '{OUTPUT_FILENAME}': {e}")
