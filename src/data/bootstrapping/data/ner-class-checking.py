import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from data.interim import INTERIM_DATA_PATH


class NERClassChecking:
    def __init__(self, json_data=None, json_path=None):
        """
        Initialize NERClassChecking class

        Args:
            json_data (list): List of JSON data containing NER annotations
            json_path (str or Path): Path to JSON file if loading from file
        """
        self.json_data = json_data
        self.json_path = (
            json_path if json_path is None else Path(json_path)
        )  # Convert to Path object if provided
        self.processed_data = None
        self.entities_df = None

        # If json_path is provided but no json_data, load the data
        if self.json_path is not None and self.json_data is None:
            self.load_json_data()

    def load_json_data(self):
        """Load JSON data from file"""
        if not self.json_path:
            raise ValueError("No JSON path provided")

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.json_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            raise

    def process_annotations(self):
        """Process NER annotations into structured format"""
        if not self.json_data:
            raise ValueError(
                "No JSON data available to process. Please provide json_data or valid json_path"
            )

        # Ensure json_data is iterable
        if not isinstance(self.json_data, (list, tuple)):
            raise TypeError(
                f"json_data must be a list or tuple, got {type(self.json_data)}"
            )

        processed_results = []

        for item in tqdm(self.json_data, desc="Processing annotations"):
            task_id = item["id"]
            text = item["data"]["text"]

            # Extract annotations
            for annotation in item["annotations"]:
                for result in annotation["result"]:
                    entity = {
                        "task_id": task_id,
                        "text": text,
                        "entity_text": result["value"]["text"],
                        "label": result["value"]["labels"][0],
                        "start": result["value"]["start"],
                        "end": result["value"]["end"],
                        "annotation_id": annotation["id"],
                    }
                    processed_results.append(entity)

        self.processed_data = processed_results
        self.entities_df = pd.DataFrame(processed_results)

        return self.entities_df

    # คงฟังก์ชันอื่นๆ ไว้เหมือนเดิม
    def get_entities_by_label(self, label):
        if self.entities_df is None:
            self.process_annotations()
        return self.entities_df[self.entities_df["label"] == label]

    def get_statistics(self):
        if self.entities_df is None:
            self.process_annotations()
        stats = {
            "total_tasks": len(set(self.entities_df["task_id"])),
            "total_entities": len(self.entities_df),
            "unique_labels": self.entities_df["label"].unique().tolist(),
            "label_counts": self.entities_df["label"].value_counts().to_dict(),
            "average_entities_per_task": self.entities_df.groupby("task_id")
            .size()
            .mean(),
        }
        return stats

    def save_processed_data(self, output_path):
        if self.entities_df is None:
            self.process_annotations()
        try:
            self.entities_df.to_csv(output_path, index=False)
            print(f"Data saved successfully to {output_path}")
        except Exception as e:
            print(f"Error saving data: {e}")
            raise

    def validate_entities(self):
        if self.entities_df is None:
            self.process_annotations()
        validation_results = []
        for _, row in self.entities_df.iterrows():
            actual_text = row["text"][row["start"] : row["end"]]
            is_valid = actual_text == row["entity_text"]
            validation_results.append(
                {
                    "task_id": row["task_id"],
                    "entity_text": row["entity_text"],
                    "label": row["label"],
                    "start": row["start"],
                    "end": row["end"],
                    "is_valid": is_valid,
                    "actual_text": actual_text,
                }
            )
        return pd.DataFrame(validation_results)


# Example usage
if __name__ == "__main__":
    # ตัวอย่างการใช้งานเมื่อมี path
    json_file_path = (
        INTERIM_DATA_PATH / "project-58-at-2025-02-28-14-33-5fa3cffa (1).json"
    )  # ปรับ path ตามจริง
    ner_checker = NERClassChecking(json_path=json_file_path)

    # Process the annotations
    entities_df = ner_checker.process_annotations()
    print("Processed entities:")
    print(entities_df.head())

    # Get statistics
    stats = ner_checker.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
