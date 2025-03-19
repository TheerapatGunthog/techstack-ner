import sys
from pathlib import Path
from tqdm import tqdm
import json
import random

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

from data.interim import INTERIM_DATA_PATH
from data.raw import RAW_DATA_PATH

# Keywords dictionary (เหมือนเดิม)
KEYWORDS = {
    "Programming_Scripting_and_Markup_languages": [
        "JavaScript",
        "HTML",
        "CSS",
        "Python",
        "SQL",
        "typescript",
        "Bash",
        "Shell",
        "Java",
        "C#",
        "C++",
        "C",
        "php",
        "PowerShell",
        "Go",
        "Rust",
        "Kotlin",
        "Lua",
        "Dart",
        "Assembly",
        "Ruby",
        "Swift",
        "R",
        "Visual Basic",
        "MATLAB",
        "VBA",
        "Groovy",
        "Scala",
        "Perl",
        "GDScript",
        "Objective-C",
        "Elixir",
        "Haskell",
        "Delphi",
        "MicroPython",
        "Lisp",
        "Clojure",
        "Julia",
        "Zig",
        "Fortran",
        "Solidity",
        "Ada",
        "Erlang",
        "F#",
        "Apex",
        "Prolog",
        "OCaml",
        "Cobol",
        "Crystal",
        "Nim",
        "Zephyr",
    ],
    "Cloud_platforms": [
        "AWS",
        "Amazon Web Services",
        "Azure",
        "Microsoft Azure",
        "Google Cloud",
        "Cloudflare",
        "Firebase",
        "Vercel",
        "Digital Ocean",
        "Heroku",
        "Netlify",
        "VMware",
        "Hetzner",
        "Supabase",
        "Linode",
        "Akamai",
        "OVH",
        "Managed Hosting",
        "COI",
        "Render",
        "Fly.io",
        "OpenShift",
        "Databricks",
        "PythonAnywhere",
        "Vultr",
        "OpenStack",
        "Alibaba Cloud",
        "IBM Cloud",
        "Watson",
        "Scaleway",
        "Colocation",
    ],
    "Database": [
        "PostgreSQL",
        "MySQL",
        "SQLite",
        "Microsoft SQL Server",
        "MongoDB",
        "Redis",
        "Mariadb",
        "Elasticsearch",
        "Oracle",
        "Dynamodb",
        "Firebase Realtime Database",
        "Cloud Firestore",
        "BigQuery",
        "Microsoft Access",
        "Supabase",
        "H2",
        "Cosmos DB",
        "Snowflake",
        "influxDB",
        "Cassandra",
        "Databricks SQL",
        "Neo4J",
        "IBM DB2",
        "Clickhouse",
        "Solr",
        "DuckDB",
        "Firebird",
        "Couch DB",
        "Cockroachdb",
        "Couchbase",
        "Presto",
        "Datamic",
        "EventStoreDB",
        "RavenDB",
        "TiDB",
    ],
    "Web_Framework_and_Technologies": [
        "Nodejs",
        "React",
        "jQuery",
        "Nextjs",
        "Expressjs",
        "Angular",
        "ASP.NET CORE",
        "Vuejs",
        "ASP.NET",
        "Flask",
        "Spring Boot",
        "Django",
        "WordPress",
        "FastAPI",
        "Laravel",
        "Angularjs",
        "Svelte",
        "NestJS",
        "Blazor",
        "Ruby",
        "Rails",
        "Nuxtjs",
        "Htmx",
        "Symfony",
        "Astro",
        "Fastify",
        "Deno",
        "Phoenix",
        "Drupal",
        "Strapi",
        "Codelgniter",
        "Gatsby",
        "Remix",
        "Solidjs",
        "Yii 2",
        "Play Framework",
        "Elm",
    ],
    "Other_Framework_and_libraries": [
        ".NET",
        "NumPy",
        "Pandas",
        ".NET Framework",
        "Spring Framework",
        "RabbitMQ",
        "Scikit-Learn",
        "Torch",
        "PyTorch",
        "TensorFlow",
        "Apache Kafka",
        "Flutter",
        "Opencv",
        "React Native",
        "Qt",
        "Opengl",
        "Electron",
        "CUDA",
        "Hugging Face Transformers",
        "Apache Spark",
        "SwiftUI",
        "Keras",
        ".NET MAUI",
        "Ruff",
        "Xamarin",
        "GTK",
        "Inonic",
        "Tauri",
        "Hadoop",
        "Cordova",
        "DirectX",
        "Capacitor",
        "OpenCL",
        "Tidyverse",
        "Roslyn",
        "Quarkus",
        "Ktor",
        "mlflow",
        "JAX",
        "MFC",
    ],
    "Embedded_Technologies": [
        "Respberry Pi",
        "Arduino",
        "GNU",
        "GCC",
        "CMake",
        "LLVM Clang",
        "Cargo",
        "MSVC",
        "Ninja",
        "PlatformIO",
        "Meson",
        "QMake",
        "Catch2",
        "cppunit",
        "doctest",
        "SCons",
        "ZMK",
        "Micronaut",
        "build2",
        "CUTE",
    ],
}

LABEL_TO_KEYWORD = {
    "PROGRAMMINGLANG": "Programming_Scripting_and_Markup_languages",
    "CLOUDPLATFORM": "Cloud_platforms",
    "DATABASE": "Database",
    "WEBFRAMEWORK_TECH": "Web_Framework_and_Technologies",
    "FRAMEWORK_LIB": "Other_Framework_and_libraries",
    "EMBEDDEDTECH": "Embedded_Technologies",
}


class NERDataAugmenter:
    def __init__(self, json_path=None, json_data=None, frequency_threshold=1):
        self.json_path = json_path if json_path is None else Path(json_path)
        self.json_data = json_data
        self.converted_data = None
        self.entity_freq = {}  # Store entity frequencies
        self.rare_entities_list = []  # Store list of rare or missing entities
        self.frequency_threshold = (
            frequency_threshold  # Threshold for "frequent" entities
        )
        if self.json_path and not self.json_data:
            self.load_json_data()

    def load_json_data(self):
        """Load JSON data from file"""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.json_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            raise

    def convert_format(self):
        """Convert Label Studio format to desired format and calculate entity frequencies for annotated entities only"""
        if not self.json_data:
            raise ValueError("No JSON data available to process")

        converted = []
        entity_counts = {}

        for item in tqdm(self.json_data, desc="Converting format"):
            new_item = {
                "id": str(item["id"]),
                "data": {"text": item["data"]["text"]},
                "annotations": [],
            }

            for ann in item["annotations"]:
                new_ann = {
                    "id": ann["id"],
                    "result": [
                        {
                            "value": result["value"],
                            "id": result["id"],
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                        }
                        for result in ann["result"]
                    ],
                }
                new_item["annotations"].append(new_ann)

                # Count only annotated entities (from annotations), case-insensitive
                for result in ann["result"]:
                    entity_text = result["value"]["text"].lower().strip()
                    label = result["value"]["labels"][0]
                    if label not in entity_counts:
                        entity_counts[label] = {}
                    entity_counts[label][entity_text] = (
                        entity_counts[label].get(entity_text, 0) + 1
                    )

            converted.append(new_item)

        self.converted_data = converted
        self.entity_freq = entity_counts
        # Generate and store rare entities list for annotated entities only
        self.rare_entities_list = self.get_rare_entities()
        return self.converted_data

    def get_rare_entities(self):
        """Get all rare or missing entities from all keyword categories, comparing only annotated entities"""
        all_rare_entities = []
        for category, keywords in KEYWORDS.items():
            # Get frequent entities only from annotated entities in the dataset
            frequent_entities = set()
            for label, entities in self.entity_freq.items():
                if LABEL_TO_KEYWORD.get(label) == category:
                    frequent_entities.update(
                        entity
                        for entity, count in entities.items()
                        if count > self.frequency_threshold
                    )
            # Compare only with keywords (case-insensitive) to find rare or missing entities
            rare_entities = [k for k in keywords if k.lower() not in frequent_entities]
            all_rare_entities.extend(rare_entities)

        return all_rare_entities

    def print_entity_frequencies(self):
        """Print entity frequencies and rare entities list for debugging"""
        print("\nEntity Frequencies by Label (Annotated Entities Only):")
        for label, entities in self.entity_freq.items():
            print(f"\n{label}:")
            for entity, count in sorted(
                entities.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {entity}: {count}")

        print("\nRare or Missing Entities (Comparing Only Annotated Entities):")
        for entity in self.rare_entities_list:
            print(f"  {entity}")


class NERDataAugmenterWithReplacement(NERDataAugmenter):
    def __init__(self, json_path=None, json_data=None, frequency_threshold=1):
        super().__init__(json_path, json_data, frequency_threshold)
        self.augmented_data = []

    def get_rare_entities(self):
        """Get all rare or missing entities with their class labels."""
        all_rare_entities = []
        for category, keywords in KEYWORDS.items():
            frequent_entities = set()
            for label, entities in self.entity_freq.items():
                if LABEL_TO_KEYWORD.get(label) == category:
                    frequent_entities.update(
                        entity
                        for entity, count in entities.items()
                        if count > self.frequency_threshold
                    )

            rare_entities = [
                (k, category) for k in keywords if k.lower() not in frequent_entities
            ]
            all_rare_entities.extend(rare_entities)

        return all_rare_entities

    def mention_replacement(self, sentences, num_augmented=10):
        """Perform mention replacement on multiple sentence templates using rare entities."""
        if not self.rare_entities_list:
            raise ValueError(
                "Rare entities list is empty. Please run convert_format() first."
            )

        augmented_sentences = []

        for _ in range(num_augmented):
            for sentence in sentences:
                augmented_sentence = sentence
                annotations = []

                placeholders = [
                    placeholder
                    for placeholder in LABEL_TO_KEYWORD.keys()
                    if f"[{placeholder}]" in sentence
                ]

                for placeholder in placeholders:
                    relevant_entities = [
                        entity
                        for entity, category in self.rare_entities_list
                        if LABEL_TO_KEYWORD.get(placeholder) == category
                    ]

                    if relevant_entities:
                        chosen_entity = random.choice(relevant_entities)
                        augmented_sentence = augmented_sentence.replace(
                            f"[{placeholder}]", chosen_entity, 1
                        )
                        start_idx = augmented_sentence.find(chosen_entity)
                        end_idx = start_idx + len(chosen_entity)

                        annotation = {
                            "value": {
                                "start": start_idx,
                                "end": end_idx,
                                "text": chosen_entity,
                                "labels": [placeholder],
                            },
                            "id": str(random.randint(1000, 9999)),
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                        }
                        annotations.append(annotation)

                augmented_item = {
                    "data": {"text": augmented_sentence},
                    "annotations": [{"result": annotations}],
                }
                augmented_sentences.append(augmented_item)
                self.augmented_data.append(augmented_item)

        return augmented_sentences

    def get_label_from_keyword(self, entity):
        """Return label based on entity keyword category."""
        entity_lower = entity.lower()
        for label, category in LABEL_TO_KEYWORD.items():
            if entity_lower in (k.lower() for k in KEYWORDS[category]):
                return label
        return "UNKNOWN"

    def save_augmented_data(self, output_path):
        """Save the augmented dataset to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.augmented_data, f, ensure_ascii=False, indent=2)
        print(f"Augmented data saved to {output_path}")


# Function to convert original data format
def convert_original_format(original_data):
    """Convert original dataset to the specified format."""
    converted_data = []

    for item in tqdm(original_data, desc="Converting original format"):
        new_item = {
            "id": str(item["id"]),
            "data": {"text": item["data"]["text"]},
            "annotations": [],
        }

        for ann in item["annotations"]:
            new_ann = {
                "id": int(ann["id"]),  # Convert annotation ID to integer
                "result": [
                    {
                        "value": result["value"],
                        "id": result["id"],
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                    }
                    for result in ann["result"]
                ],
            }
            new_item["annotations"].append(new_ann)

        converted_data.append(new_item)

    return converted_data


# Function to merge original and augmented datasets
def merge_datasets(original_data, augmented_data):
    """Merge converted original dataset with augmented dataset, continuing IDs sequentially."""
    combined_data = []

    # Add converted original data first
    max_id = -1
    for item in original_data:
        combined_data.append(item)
        item_id = int(item["id"])
        if item_id > max_id:
            max_id = item_id

    # Start new IDs from max_id + 1
    next_id = max_id + 1

    # Add augmented data with new sequential IDs
    for aug_item in augmented_data:
        new_item = {
            "id": str(next_id),
            "data": {"text": aug_item["data"]["text"]},
            "annotations": [
                {
                    "id": next_id,  # Use the same ID for annotation as the item
                    "result": aug_item["annotations"][0]["result"],
                }
            ],
        }
        combined_data.append(new_item)
        next_id += 1

    return combined_data


# Example Usage
if __name__ == "__main__":
    json_file_path = (
        INTERIM_DATA_PATH
        / ".bootstrapping/001/project-3-at-2025-03-01-22-23-fd6863ce.json"
    )
    augmenter = NERDataAugmenterWithReplacement(
        json_path=json_file_path, frequency_threshold=1
    )
    augmenter.convert_format()

    # Read sentence templates from the txt file
    sentence_file_path = RAW_DATA_PATH / "sentence_templates.txt"
    with open(sentence_file_path, "r", encoding="utf-8") as f:
        sentence_templates = [
            line.strip() for line in f if line.strip()
        ]  # Read each line as a sentence

    # Perform mention replacement using the list of sentences
    augmented_sentences = augmenter.mention_replacement(
        sentence_templates, num_augmented=2
    )

    # Load original dataset
    with open(json_file_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    # Convert original dataset to the desired format
    converted_original_data = convert_original_format(original_data)

    # Merge converted original and augmented datasets
    combined_dataset = merge_datasets(converted_original_data, augmenter.augmented_data)

    # Save the combined dataset
    output_path = INTERIM_DATA_PATH / "augmented_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=2)
    print(f"Combined dataset saved to {output_path}")
