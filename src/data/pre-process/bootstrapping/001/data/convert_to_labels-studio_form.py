import json
from pathlib import Path

# ===================================================================
# 👉 Section to modify: Please specify your filename here 👈
# ===================================================================

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

# Specify the input data filename (the file you want to convert)
INPUT_FILENAME = (
    PROJECT_PATH
    / "data/interim/bootstrapping/001/labeled-by-code-data/labels-by-gemini-001.json"
)

# Specify the desired output filename (the file to be created)
OUTPUT_FILENAME = (
    PROJECT_PATH
    / "data/interim/bootstrapping/001/labeled-by-code-data/gemini_001_labels-studio_form.json"
)

# ===================================================================


def convert_to_label_studio_format(data):
    """
    Convert the specific data format to the JSON format required by Label Studio
    for Named Entity Recognition (NER) tasks.

    This version recalculates indices, prevents overlaps, and handles
    multiple occurrences of the same text (e.g., "Java" appearing twice).
    """
    label_studio_tasks = []

    if not isinstance(data, list):
        print(
            f"❌ Error: The data in file '{INPUT_FILENAME}' must be a List (JSON array)"
        )
        return None

    for item_index, item in enumerate(data):
        results = []

        if "original_text" not in item or "ner_labels" not in item:
            print(
                f"⚠️ Warning: Skipping item {item_index} because it lacks 'original_text' or 'ner_labels'"
            )
            continue

        original_text = item["original_text"]

        is_char_labeled = [False] * len(original_text)

        sorted_labels = sorted(
            item["ner_labels"], key=lambda x: len(x.get("text", "")), reverse=True
        )

        for label in sorted_labels:
            if not label.get("text"):
                continue

            label_text = label["text"]

            # --- New logic to find the NEXT AVAILABLE spot for a label ---
            search_start_pos = 0
            new_start = -1

            while True:
                # Find the next occurrence from the search_start_pos
                found_pos = original_text.find(label_text, search_start_pos)

                if found_pos == -1:
                    # No more occurrences of this text in the string
                    break

                # Check if this specific occurrence is already covered by another label
                if any(is_char_labeled[found_pos : found_pos + len(label_text)]):
                    # This spot is taken, so for the next search, start after this found spot
                    search_start_pos = found_pos + 1
                    continue
                else:
                    # Found a valid, available spot. Lock it in and stop searching for this label.
                    new_start = found_pos
                    break
            # --- End of new search logic ---

            if new_start == -1:
                # If after all searching, no available spot was found, it means
                # it was genuinely a substring of an already-processed longer label.
                # This log is helpful for debugging.
                # print(f"ℹ️ Info: Skipping nested label '{label_text}' as no free slot was found.")
                continue

            new_end = new_start + len(label_text)

            results.append(
                {
                    "value": {
                        "start": new_start,
                        "end": new_end,
                        "text": label_text,
                        "labels": [label["label"]],
                    },
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                }
            )

            # Mark this span as labeled in our character map
            for i in range(new_start, new_end):
                is_char_labeled[i] = True

        task = {
            "data": {"text": original_text},
            "annotations": [{"result": results}],
        }
        label_studio_tasks.append(task)

    return label_studio_tasks


# --- Main program section (unchanged) ---
if __name__ == "__main__":
    print(f"▶️ Starting file conversion process...")

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

    # 2. Perform data conversion
    converted_data = convert_to_label_studio_format(input_data)

    # 3. Save the result as the output file (if conversion is successful)
    if converted_data is not None:
        try:
            with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Conversion successful! Result saved to '{OUTPUT_FILENAME}'")
        except IOError as e:
            print(f"❌ Error: Could not write file '{OUTPUT_FILENAME}': {e}")
