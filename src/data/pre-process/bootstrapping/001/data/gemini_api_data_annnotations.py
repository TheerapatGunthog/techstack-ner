import google.generativeai as genai
import pandas as pd
from pathlib import Path
import time
import json
import os
import re
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
# Gemini API key and project path
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables. Please set it in your .env file."
    )

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")
# Path to the input CSV data
DATASET_PATH = PROJECT_PATH / "data/interim/preprocessed-data/kaggle_data.csv"
# The name of the column in your CSV that contains the job descriptions
JOB_DESCRIPTION_COLUMN = "Segmented_Qualification"
# Gemini API model to be used
GEMINI_API_MODEL = "gemini-2.5-flash"
# Path for saving interim (checkpoint) results
CHECKPOINT_PATH = (
    PROJECT_PATH / "data/interim/bootstrapping/001/gemini_api_checkpoint.json"
)
# Path for the final output file
RESULTS_PATH = (
    PROJECT_PATH
    / "data/interim/bootstrapping/001/labeled-by-code-data/labels-by-gemini-001.json"
)

# --- API & Processing Settings ---
MAX_RETRIES = 10  # Maximum number of retries for API calls
RETRY_DELAY_SECONDS = (
    5  # Initial delay in seconds before retrying (increased for robustness)
)
CHECKPOINT_INTERVAL = 10  # Save results every X rows

# --- Gemini API Setup ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_API_MODEL)

# --- Prompt Engineering ---
# Definition of NER classes for the prompt
NER_CLASSES_DEFINITION = """
- PSML: Programming languages, scripting languages, or markup languages (e.g., Python, Java, C++, JavaScript, HTML, CSS, SQL, Shell Script, Go, Ruby, Swift, Kotlin, PHP, C#)
- DB: Database systems or related data storage technologies (e.g., MySQL, PostgreSQL, MongoDB, Redis, Oracle, SQL Server, Cassandra, DynamoDB, Firebase, SQLite)
- CP: Cloud computing platforms and related services/technologies (e.g., AWS, Azure, Google Cloud Platform (GCP), Kubernetes, Docker, Serverless, Terraform, OpenStack)
- FAL: Software development frameworks, libraries, SDKs, or widely used tools (e.g., React, Angular, Vue.js, Spring Boot, Node.js, Django, .NET, Laravel, Flask, TensorFlow, PyTorch, Keras, NumPy, Pandas, Scikit-learn, Express.js, Bootstrap)
- ET: Technologies specifically related to embedded systems, IoT, hardware programming, or real-time operating systems (e.g., RTOS, Microcontroller, FPGA, ARM, Raspberry Pi, Arduino, C for embedded, bare-metal programming)
- TAS: General software development skills, methodologies, techniques, or non-platform/language-specific abilities (e.g., Agile, Scrum, CI/CD, Unit Testing, Data Structures, Algorithms, Machine Learning, Deep Learning, RESTful API, OOP, TDD, Version Control, Git, Problem Solving, Communication, Design Patterns)
"""

# Prompt template for Named Entity Recognition (NER)
PROMPT_TEMPLATE = """
Extract technology-related entities **strictly and directly** from the provided job description below. Categorize each entity according to the provided Labels.
**Only extract entities explicitly mentioned within the text.** Do not infer or assume.
For each entity, provide its exact `text`, its `label`, and its `start` and `end` character indices from the original "Job Description to Analyze".

Output results as a JSON array of objects.
Return an empty array `[]` if no relevant entities are found.

---
**Labels and Definitions:**
{ner_classes_placeholder}
---

**Example Input:**
We are looking for a Software Engineer with strong experience in Java, Spring Boot, and AWS. Knowledge of PostgreSQL, Kafka, and Docker would be highly beneficial. Experience with Agile methodologies and CI/CD pipelines is also a plus.

**Expected Output Example:**
```json
[
  {{"text": "Java", "label": "PSML", "start": 59, "end": 63}},
  {{"text": "Spring Boot", "label": "FAL", "start": 65, "end": 76}},
  {{"text": "AWS", "label": "CP", "start": 82, "end": 85}},
  {{"text": "PostgreSQL", "label": "DB", "start": 100, "end": 110}},
  {{"text": "Kafka", "label": "FAL", "start": 112, "end": 117}},
  {{"text": "Docker", "label": "CP", "start": 123, "end": 129}},
  {{"text": "Agile", "label": "TAS", "start": 164, "end": 169}},
  {{"text": "CI/CD", "label": "TAS", "start": 188, "end": 193}}
]
Job Description to Analyze:
{job_desc_content}
"""


def load_dataframe(path):
    """Loads a CSV file into a pandas DataFrame."""
    if not path.exists():
        print(f"Error: Dataset file not found at {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"Successfully loaded {len(df)} rows from {path}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def load_checkpoint(path):
    """Loads previously processed results from a checkpoint file."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} results from checkpoint {path}.")
            return results
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"Warning: Could not load checkpoint from {path}: {e}. Starting fresh."
            )
    return []


def save_checkpoint(data, path):
    """Saves results to a checkpoint file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # print(f"\nCheckpoint: Saved {len(data)} results to {path}")
    except IOError as e:
        print(f"Error: Could not save checkpoint to {path}: {e}")


def extract_json_from_text(text):
    """Extracts a JSON object from a string, even if it's in a markdown block."""
    # Look for a JSON markdown block first
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if json_match:
        return json_match.group(1).strip()
    # If no block, assume the whole string might be JSON
    return text.strip()


def main():
    """Main function to run the NER processing script."""
    # --- Load Data ---
    df = load_dataframe(DATASET_PATH)
    if df is None:
        return

    if JOB_DESCRIPTION_COLUMN not in df.columns:
        print(f"Error: Column '{JOB_DESCRIPTION_COLUMN}' not found in the DataFrame.")
        print(f"Available columns are: {list(df.columns)}")
        return

    # --- Load Checkpoint & Prepare for Processing ---
    results_ner = load_checkpoint(CHECKPOINT_PATH)
    start_index = len(results_ner)

    if start_index >= len(df):
        print("All rows have already been processed. Nothing to do.")
        return

    print(f"Resuming processing from row {start_index} of {len(df)}.")

    # --- Main Processing Loop ---
    process_df = df.iloc[start_index:]

    for index, row in tqdm(
        process_df.iterrows(), total=len(process_df), desc="Processing Job Descriptions"
    ):
        job_description_text = row.get(JOB_DESCRIPTION_COLUMN)

        # Skip rows with empty or invalid job description text
        if (
            not isinstance(job_description_text, str)
            or not job_description_text.strip()
        ):
            print(
                f"Warning: Skipping row {index} due to empty or invalid job description."
            )
            results_ner.append(
                {
                    "original_text": "",
                    "ner_labels": "Skipped - Empty Input",
                    "row_index": index,
                }
            )
            continue

        full_prompt = PROMPT_TEMPLATE.format(
            ner_classes_placeholder=NER_CLASSES_DEFINITION.strip(),
            job_desc_content=job_description_text,
        )

        retries = 0
        success = False
        while retries < MAX_RETRIES and not success:
            try:
                # --- API Call ---
                response = model.generate_content(full_prompt)

                if response.candidates:
                    generated_text = response.candidates[0].content.parts[0].text
                    json_str = extract_json_from_text(generated_text)

                    try:
                        ner_output = json.loads(json_str)
                        results_ner.append(
                            {
                                "original_text": job_description_text,
                                "ner_labels": ner_output,
                                "row_index": index,
                            }
                        )
                    except json.JSONDecodeError:
                        print(
                            f"Warning: JSON parsing failed for row {index}. Raw output saved."
                        )
                        results_ner.append(
                            {
                                "original_text": job_description_text,
                                "ner_labels": "Parsing Error",
                                "raw_output": generated_text,
                                "row_index": index,
                            }
                        )
                else:
                    # Handle cases where the prompt was blocked or response was empty
                    block_reason = (
                        response.prompt_feedback.block_reason
                        if response.prompt_feedback
                        else "Unknown"
                    )
                    print(
                        f"Warning: No candidates for row {index}. Reason: {block_reason}"
                    )
                    results_ner.append(
                        {
                            "original_text": job_description_text,
                            "ner_labels": f"Blocked or Empty Response - Reason: {block_reason}",
                            "row_index": index,
                        }
                    )

                success = True  # Mark as successful to exit retry loop

            except Exception as e:
                retries += 1
                print(f"Error on row {index} (Attempt {retries}/{MAX_RETRIES}): {e}")
                if retries >= MAX_RETRIES:
                    print(f"Max retries reached for row {index}. Skipping.")
                    results_ner.append(
                        {
                            "original_text": job_description_text,
                            "ner_labels": "API Error after retries",
                            "row_index": index,
                        }
                    )
                else:
                    # Exponential backoff
                    time.sleep(RETRY_DELAY_SECONDS * (2 ** (retries - 1)))

        # --- Checkpointing ---
        if len(results_ner) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(results_ner, CHECKPOINT_PATH)
            tqdm.write(f"Checkpoint: Saved {len(results_ner)} total results.")

    # --- Final Save ---
    print("\nProcessing complete.")
    save_checkpoint(results_ner, RESULTS_PATH)
    print(f"All {len(results_ner)} NER results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
