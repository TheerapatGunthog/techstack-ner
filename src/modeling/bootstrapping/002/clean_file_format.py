import pandas as pd
import json
from pathlib import Path


def remove_unused_rows(df):
    """
    Filter and remove unwanted rows from DataFrame:
    - Remove rows where the 'label' column is empty.
    - Remove rows where the 'label' column contains only 'TAS'.
    """

    def is_row_valid(label_str):
        """
        Helper function to check if each row should be kept or removed.
        Returns True if the row should be kept.
        """
        try:
            # Convert the JSON string in the 'label' column to a list
            labels_data = json.loads(label_str)

            # 1. Check if the label is empty
            if not labels_data:
                return False  # If no label exists -> remove

            # 2. Check if the label contains only 'TAS'
            # Create a set of all labels in that row
            unique_labels = {
                item["labels"][0]
                for item in labels_data
                if "labels" in item and item["labels"]
            }

            if unique_labels == {"TAS"}:
                return False  # If it contains only 'TAS' -> remove

        except (json.JSONDecodeError, TypeError, AttributeError):
            # If the data in the label column is not JSON or is empty (NaN)
            return False  # Remove

        # If none of the above conditions are met -> keep the row
        return True

    # Create a boolean mask to select which rows to keep
    keep_mask = df["label"].apply(is_row_valid)

    # Return the filtered DataFrame and reset the index
    return df[keep_mask].reset_index(drop=True)


def convert_csv_to_json(csv_file_path, json_file_path):
    """
    Read a CSV file, remove unused rows, convert to JSON, and save the file.
    """
    try:
        df = pd.read_csv(csv_file_path)

        # --- Call the function to filter data ---
        df_filtered = remove_unused_rows(df)

        print(f"Original rows: {len(df)}, Rows after filtering: {len(df_filtered)}")

        output_list = []
        # Loop through the filtered DataFrame
        for index, row in df_filtered.iterrows():
            labels_str = row.get("label", "[]")
            try:
                labels = json.loads(labels_str)
            except (json.JSONDecodeError, TypeError):
                labels = []

            ner_labels = []

            if isinstance(labels, list):
                for label in labels:
                    if isinstance(label, dict):
                        ner_labels.append(
                            {
                                "text": label.get("text"),
                                "label": label.get("labels", [None])[0],
                                "start": label.get("start"),
                                "end": label.get("end"),
                            }
                        )

            output_list.append(
                {
                    "original_text": row.get("text"),
                    "ner_labels": ner_labels,
                    "row_index": index,
                }
            )

        # Write the list of dictionaries to a JSON file
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(output_list, f, ensure_ascii=False, indent=4)

        print(f"File '{json_file_path}' has been successfully created! ✅")

    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found. Please check the path again.")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Program entry point ---
if __name__ == "__main__":
    # **Please replace 'your_file.csv' with your file path**
    PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")
    input_csv_path = (
        PROJECT_PATH
        / "data/interim/bootstrapping/002/dictionary/project-35-at-2025-07-16-05-59-61deef20.csv"
    )
    output_json_path = PROJECT_PATH / "data/interim/bootstrapping/002/clean_label.json"

    convert_csv_to_json(input_csv_path, output_json_path)
