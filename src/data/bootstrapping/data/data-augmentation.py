import sys
from pathlib import Path
from tqdm import tqdm
import json
import random

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from data.interim import INTERIM_DATA_PATH

# Keywords dictionary (เหมือนเดิม)
KEYWORDS = {
    "Programming_Scripting_and_Markup_languages": [
        "JavaScript",
        "HTML",
        "CSS",
        "python",
        "sql",
        "typescript",
        "bash",
        "shell",
        "java",
        "c#",
        "c++",
        "c",
        "php",
        "powershell",
        "golang",
        "go",
        "rust",
        "kotlin",
        "lua",
        "dart",
        "assembly",
        "ruby",
        "swift",
        "r",
        "visual basic",
        "matlab",
        "vba",
        "groovy",
        "scala",
        "perl",
        "gdscript",
        "objective-c",
        "elixir",
        "haskell",
        "delphi",
        "micropython",
        "lisp",
        "clojure",
        "julia",
        "zig",
        "fortran",
        "solidity",
        "ada",
        "erlang",
        "f#",
        "apex",
        "prolog",
        "ocaml",
        "cobol",
        "crystal",
        "nim",
        "zephyr",
    ],
    "Cloud_platforms": [
        "aws",
        "aws cloud",
        "azure",
        "google cloud",
        "cloudflare",
        "firebase",
        "vercel",
        "digital ocean",
        "heroku",
        "netlify",
        "vmware",
        "hetzner",
        "supabase",
        "linode",
        "akamai",
        "ovh",
        "managed hosting",
        "coi",
        "render",
        "fly.io",
        "openshift",
        "databricks",
        "pythonanywhere",
        "vultr",
        "openstack",
        "alibaba cloud",
        "ibm cloud",
        "watson",
        "scaleway",
        "colocation",
    ],
    "Database": [
        "postgresql",
        "mysql",
        "sqlite",
        "microsoft sql server",
        "mongodb",
        "redis",
        "mariadb",
        "elasticsearch",
        "oracle",
        "dynamodb",
        "firebase realtime database",
        "cloud firestore",
        "bigquery",
        "microsoft access",
        "supabase",
        "h2",
        "cosmos db",
        "snowflake",
        "influxdb",
        "cassandra",
        "databricks sql",
        "neo4j",
        "ibm db2",
        "clickhouse",
        "solr",
        "duckdb",
        "firebird",
        "couch db",
        "cockroachdb",
        "couchbase",
        "presto",
        "datamic",
        "eventstoredb",
        "ravendb",
        "tidb",
    ],
    "Web_Framework_and_Technologies": [
        "nodejs",
        "react",
        "jquery",
        "nextjs",
        "expressjs",
        "angular",
        "asp.net core",
        "vuejs",
        "asp.net",
        "flask",
        "spring boot",
        "django",
        "wordpress",
        "fastapi",
        "laravel",
        "angularjs",
        "svelte",
        "nestjs",
        "blazor",
        "ruby",
        "rails",
        "nextjs",
        "htmx",
        "symfony",
        "astro",
        "fastify",
        "deno",
        "phoenix",
        "drupal",
        "strapi",
        "codelgniter",
        "gatsby",
        "remix",
        "solidjs",
        "yii 2",
        "play framework",
        "elm",
    ],
    "Other_Framework_and_libraries": [
        ".net",
        "numpy",
        "pandas",
        ".ent framework",
        "spring framework",
        "rabbitmq",
        "scikit learn",
        "torch",
        "pytorch",
        "tensorflow",
        "apache kafka",
        "flutter",
        "opencv",
        "react native",
        "qt",
        "opengl",
        "electron",
        "cuda",
        "hugging face transformers",
        "apache spark",
        "swiftui",
        "keras",
        ".net maui",
        "ruff",
        "xamarin",
        "gtk",
        "inonic",
        "tauri",
        "hadoop",
        "cordova",
        "directx",
        "capacitor",
        "opencl",
        "tidyverse",
        "roslyn",
        "quarkus",
        "ktor",
        "mlflow",
        "jax",
        "mfc",
    ],
    "Embedded_Technologies": [
        "respberry pi",
        "arduino",
        "gnu gcc",
        "cmake",
        "llvm clang",
        "cargo",
        "msvc",
        "ninja",
        "platformio",
        "meson",
        "qmake",
        "catch2",
        "cppunit",
        "doctest",
        "scons",
        "zmk",
        "micronaut",
        "build2",
        "cute",
    ],
}

LABEL_TO_KEYWORD = {
    "PROGRAMMINGLANG": "Programming_Scripting_and_Markup_languages",
    "CLOUDPLATFORM": "Cloud_platforms",
    "DATABASE": "Database",
    "WEBFRAMEWORK": "Web_Framework_and_Technologies",
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
        INTERIM_DATA_PATH / "project-58-at-2025-02-28-14-33-5fa3cffa (1).json"
    )
    augmenter = NERDataAugmenterWithReplacement(
        json_path=json_file_path, frequency_threshold=1
    )
    augmenter.convert_format()

    sentence = [
        "Experience in developing applications using [PROGRAMMINGLANG] and [WEBFRAMEWORK].",
        "Hands-on knowledge of cloud infrastructure on [CLOUDPLATFORM].",
        "Proficiency in working with [DATABASE] and writing optimized queries.",
        "Familiarity with modern web technologies like [WEBFRAMEWORK] and [FRAMEWORK_LIB].",
        "Experience in developing microservices using [PROGRAMMINGLANG] with [CLOUDPLATFORM].",
        "Strong understanding of embedded systems using [EMBEDDEDTECH].",
        "Experience in API development with [WEBFRAMEWORK] and [PROGRAMMINGLANG].",
        "Knowledge of cloud services such as [CLOUDPLATFORM] for infrastructure management.",
        "Hands-on experience with [DATABASE] and data modeling.",
        "Proficiency in using [FRAMEWORK_LIB] for software development.",
        "Experience in working with [EMBEDDEDTECH] for IoT applications.",
        "Familiarity with containerization tools integrated with [CLOUDPLATFORM].",
        "Strong background in writing scalable applications using [PROGRAMMINGLANG].",
        "Experience in integrating backend services using [WEBFRAMEWORK] and [DATABASE].",
        "Knowledge of cloud security best practices on [CLOUDPLATFORM].",
        "Experience in developing front-end applications with [WEBFRAMEWORK].",
        "Proficiency in using [FRAMEWORK_LIB] for unit testing and automation.",
        "Knowledge of embedded protocols and communication interfaces with [EMBEDDEDTECH].",
        "Experience in deploying applications on [CLOUDPLATFORM] using CI/CD pipelines.",
        "Hands-on development of RESTful services using [PROGRAMMINGLANG] and [WEBFRAMEWORK].",
    ]
    augmented_sentences = augmenter.mention_replacement(sentence, num_augmented=10)

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
