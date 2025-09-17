import json
from pathlib import Path


# --- Main workflow section ---

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

# Input file names
DICTIONARY = PROJECT_PATH / "data/keywords/canonical_dictionary_v2.json"
LABEL_DICTIONARY = (
    PROJECT_PATH / "data/interim/bootstrapping/002/dictionary/labels_final.json"
)
LABEL_CONFLICTS = (
    PROJECT_PATH / "data/interim/bootstrapping/002/dictionary/labels_conflicts.json"
)

OUTPUT_DICTIONARY = (
    PROJECT_PATH / "data/interim/bootstrapping/002/dictionary/dictionary.json"
)


def merge_new_labels(target_dict, file_path):
    """
    Read a JSON file, convert keys from that file to lowercase, and merge them with the target dictionary.
    """
    print(f"Processing file: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        new_entries = json.load(f)

    added_count = 0
    # 1. Convert the keys of the new data to lowercase first
    new_entries_lower = {key.lower(): value for key, value in new_entries.items()}

    # 2. Loop to add only keys that are not already in the target dictionary
    for key, value in new_entries_lower.items():
        if key not in target_dict:
            target_dict[key] = value
            added_count += 1
        else:
            print(f"    - Skipping '{key}' because it already exists")

    print(f"Added {added_count} new entries from {file_path.name}")
    return target_dict


# --- Start Execution ---
if __name__ == "__main__":
    # 1. Load the main dictionary (no need to convert keys since they are already in lowercase)
    print(f"Loading main dictionary: {DICTIONARY}")
    with open(DICTIONARY, "r", encoding="utf-8") as f:
        main_dict = json.load(f)
    print(f"The main dictionary contains {len(main_dict)} entries")
    print("-" * 30)

    # 2. Merge data from LABEL_DICTIONARY
    main_dict = merge_new_labels(main_dict, LABEL_DICTIONARY)
    print("-" * 30)

    # 3. Merge data from LABEL_CONFLICTS
    main_dict = merge_new_labels(main_dict, LABEL_CONFLICTS)
    print("-" * 30)

    # 4. Save the result to a file
    print(f"Saving final dictionary to: {OUTPUT_DICTIONARY}")
    with open(OUTPUT_DICTIONARY, "w", encoding="utf-8") as f:
        json.dump(main_dict, f, indent=4, ensure_ascii=False)

    print("\nExecution complete")
    print(f"The new dictionary contains {len(main_dict)} entries")
