import csv
import json
from pathlib import Path
from collections import defaultdict


def process_labels_with_conflict_detection(file_path, json_column_name):
    """
    Processes a CSV file to extract labels, detecting and separating entries
    where the same term is given conflicting labels.

    Args:
        file_path (str): The path to the CSV file.
        json_column_name (str): The name of the column containing the JSON string.

    Returns:
        tuple: A tuple containing two dictionaries:
               - clean_labels (dict): Terms with a single, consistent label.
               - conflicts (dict): Terms with multiple, different labels,
                                   with the value being a list of all found labels.
    """
    # final_labels will store terms that haven't encountered conflicts yet
    final_labels = {}
    # conflicts will store terms that have already encountered conflicts
    # defaultdict(set) will automatically create a set when encountering a new key
    conflicts = defaultdict(set)

    try:
        with open(file_path, mode="r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            print("Starting processing... This may take a moment for large files.")
            for i, row in enumerate(reader):
                if json_column_name in row and row[json_column_name]:
                    try:
                        json_data = json.loads(row[json_column_name])
                        for annotation in json_data:
                            if (
                                "text" in annotation
                                and "labels" in annotation
                                and annotation["labels"]
                            ):
                                term = annotation["text"].strip()
                                label = annotation["labels"][0].strip()

                                if not term:  # Skip if term is empty
                                    continue

                                # 1. Check if this term is already in the conflicts list
                                if term in conflicts:
                                    conflicts[term].add(label)
                                    continue

                                # 2. Check if this term is a new term
                                if term not in final_labels:
                                    final_labels[term] = label
                                # 3. If this term already exists, check if the label is the same as before
                                elif final_labels[term] != label:
                                    # If not the same -> conflict occurs!
                                    # Move from final_labels to conflicts
                                    print(
                                        f"  - Conflict found for '{term}': Was '{final_labels[term]}', now found '{label}'. Moved to conflicts."
                                    )

                                    # Add both labels (old and new) to the set
                                    conflicts[term].add(final_labels[term])
                                    conflicts[term].add(label)

                                    # Remove from the conflict-free list
                                    del final_labels[term]

                    except json.JSONDecodeError:
                        print(f"Warning: JSON decode error in row {i + 2}. Skipping.")
                    except (TypeError, KeyError) as e:
                        print(
                            f"Warning: Data structure error in row {i + 2}: {e}. Skipping."
                        )

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return {}, {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}, {}

    # Convert sets in conflicts to lists so they can be saved as JSON
    conflicts_as_list = {key: sorted(list(value)) for key, value in conflicts.items()}

    return final_labels, conflicts_as_list


def save_dict_to_json(data_dict, json_file_path):
    """Saves a dictionary to a JSON file."""
    if not data_dict:
        print(f"No data to save for '{json_file_path}'. File not created.")
        return
    try:
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(data_dict, json_file, ensure_ascii=False, indent=2)
        print(f"Successfully saved {len(data_dict)} items to '{json_file_path}'")
    except Exception as e:
        print(f"An error occurred while saving to JSON '{json_file_path}': {e}")


# --- Main workflow section ---

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

JSON_DATA_COLUMN = "label"
CSV_FILE_NAME = (
    PROJECT_PATH
    / "data/interim/bootstrapping/002/dictionary/project-35-at-2025-07-16-05-59-61deef20.csv"
)

# Output file names
CLEAN_JSON_OUTPUT = (
    PROJECT_PATH / "data/interim/bootstrapping/002/dictionary/labels_final.json"
)
CONFLICT_JSON_OUTPUT = (
    PROJECT_PATH / "data/interim/bootstrapping/002/dictionary/labels_conflicts.json"
)

# Call processfunctions
clean_data, conflict_data = process_labels_with_conflict_detection(
    CSV_FILE_NAME, JSON_DATA_COLUMN
)

print("\n--- Processing Complete ---")

# Save
save_dict_to_json(clean_data, CLEAN_JSON_OUTPUT)
save_dict_to_json(conflict_data, CONFLICT_JSON_OUTPUT)
