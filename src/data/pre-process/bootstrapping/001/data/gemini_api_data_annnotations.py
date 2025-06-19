import google.generativeai as genai
import pandas as pd
from pathlib import Path
import time
import json
import os
import re
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

# Gemini API key and project path
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")


# Load raw JSON data from the specified path
json_file_path = PROJECT_PATH / "data/interim/gemini/filtered_data.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    raw_data_list = json.load(f)

# Declare column name for job descriptions
MY_JOB_DESCRIPTION_COLUMN = "Job_Description"

# Create DataFrame from the loaded data
df = pd.DataFrame(raw_data_list, columns=[MY_JOB_DESCRIPTION_COLUMN])

# Display head and shape of the DataFrame for verification
print(f"DataFrame Head:\n{df.head()}")
print(f"Total Rows in DataFrame: {df.shape[0]}\n")

# Gemini API model to be used
GEMINI_API_MODEL = "gemini-2.5-flash"

# --- Configure Gemini API Key ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_API_MODEL)


# Definition of NER classes for the prompt
NER_CLASSES_DEFINITION = """
- Programming_Scripting_and_Markup_languages: Programming languages, scripting languages, or markup languages (e.g., Python, Java, C++, JavaScript, HTML, CSS, SQL, Shell Script, Go, Ruby, Swift, Kotlin, PHP, C#)
- Database: Database systems or related data storage technologies (e.g., MySQL, PostgreSQL, MongoDB, Redis, Oracle, SQL Server, Cassandra, DynamoDB, Firebase, SQLite)
- Cloud_platforms: Cloud computing platforms and related services/technologies (e.g., AWS, Azure, Google Cloud Platform (GCP), Kubernetes, Docker, Serverless, Terraform, OpenStack)
- Framework_and_Libraries: Software development frameworks, libraries, SDKs, or widely used tools (e.g., React, Angular, Vue.js, Spring Boot, Node.js, Django, .NET, Laravel, Flask, TensorFlow, PyTorch, Keras, NumPy, Pandas, Scikit-learn, Express.js, Bootstrap)
- Embedded_Technologies: Technologies specifically related to embedded systems, IoT, hardware programming, or real-time operating systems (e.g., RTOS, Microcontroller, FPGA, ARM, Raspberry Pi, Arduino, C for embedded, bare-metal programming)
- Technique_and_Skill: General software development skills, methodologies, techniques, or non-platform/language-specific abilities (e.g., Agile, Scrum, CI/CD, Unit Testing, Data Structures, Algorithms, Machine Learning, Deep Learning, RESTful API, OOP, TDD, Version Control, Git, Problem Solving, Communication, Design Patterns)
"""

# Prompt template for Named Entity Recognition (NER)
PROMPT_TEMPLATE = """
Extract technology-related entities **strictly and directly** from the provided job description below. Categorize each entity according to the provided Labels.
**Only extract entities explicitly mentioned within the text.** Do not infer or assume.
Output results as a JSON array, where each object contains the exact `text` and its `label`.
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
  {{"text": "Java", "label": "Programming_Scripting_and_Markup_languages"}},
  {{"text": "Spring Boot", "label": "Framework_and_Libraries"}},
  {{"text": "AWS", "label": "Cloud_platforms"}},
  {{"text": "PostgreSQL", "label": "Database"}},
  {{"text": "Kafka", "label": "Technique_and_Skill"}},
  {{"text": "Docker", "label": "Cloud_platforms"}},
  {{"text": "Agile", "label": "Technique_and_Skill"}},
  {{"text": "CI/CD", "label": "Technique_and_Skill"}}
]

Job Description to Analyze:
{job_desc_content}
"""

# --- Configuration for robust API calls ---
MAX_RETRIES = 10  # Maximum number of retries for API calls
RETRY_DELAY_SECONDS = 1  # Initial delay in seconds before retrying
# Path for saving interim (checkpoint) results
INTERIM_RESULTS_PATH = (
    PROJECT_PATH / "data/interim/bootstrapping/001/gemini_ner_interim_results.json"
)
CHECKPOINT_INTERVAL = 10  # Save results every X rows

# --- Load previous results if available to resume processing ---
results_ner = []
start_index = 0
if INTERIM_RESULTS_PATH.exists():
    try:
        with open(INTERIM_RESULTS_PATH, "r", encoding="utf-8") as f:
            results_ner = json.load(f)
        start_index = len(results_ner)
        print(
            f"Loaded {start_index} previously processed results from {INTERIM_RESULTS_PATH}."
        )
        print("Continuing processing from where it left off.")
    except json.JSONDecodeError as e:
        print(
            f"Error loading interim results from {INTERIM_RESULTS_PATH}: {e}. Starting fresh."
        )
        results_ner = []
        start_index = 0
    except Exception as e:
        print(
            f"An unexpected error occurred while loading interim results: {e}. Starting fresh."
        )
        results_ner = []
        start_index = 0

# Total number of rows in the DataFrame for token estimation and processing
SAMPLE_SIZE = df.shape[0]
total_input_tokens_sample = 0
total_output_tokens_sample = 0
processed_rows_count_current_run = (
    0  # Counter for successfully processed rows in the current run
)

print(
    f"---Processing from row {start_index} to {SAMPLE_SIZE - 1} for Token Estimation and NER ---"
)

# Iterate through the DataFrame with tqdm for a progress bar, starting from start_index
for index, row in tqdm(
    df.iloc[start_index:SAMPLE_SIZE].iterrows(),
    total=SAMPLE_SIZE - start_index,
    initial=start_index,
    desc="Processing Job Descriptions",
):
    JOB_DESCRIPTION_COLUMN_NAME = MY_JOB_DESCRIPTION_COLUMN

    if JOB_DESCRIPTION_COLUMN_NAME not in row:
        print(
            f"Error: Column '{JOB_DESCRIPTION_COLUMN_NAME}' not found in DataFrame row {index}. Please adjust 'MY_JOB_DESCRIPTION_COLUMN'. Skipping row."
        )
        continue

    job_description_text = row[JOB_DESCRIPTION_COLUMN_NAME]

    full_prompt = PROMPT_TEMPLATE.format(
        ner_classes_placeholder=NER_CLASSES_DEFINITION.strip(),
        job_desc_content=job_description_text,
    )

    retries = 0
    while retries < MAX_RETRIES:
        try:
            # Count input tokens for the current prompt
            input_token_response = model.count_tokens(full_prompt)
            input_tokens = input_token_response.total_tokens
            total_input_tokens_sample += input_tokens

            # Make the API call to generate content
            response = model.generate_content(full_prompt)

            generated_text = ""
            if response.candidates:
                generated_text = response.candidates[0].content.parts[0].text

                # --- Robust JSON parsing ---
                try:
                    # Use regex to extract JSON block, if present
                    json_match = re.search(
                        r"```json\s*([\s\S]*?)\s*```", generated_text
                    )
                    if json_match:
                        json_str = json_match.group(1).strip()
                    else:
                        # Attempt to parse the entire text as JSON if no code block is found
                        json_str = generated_text.strip()

                    ner_output = json.loads(json_str)
                    results_ner.append(
                        {
                            "original_text": job_description_text,
                            "ner_labels": ner_output,
                        }
                    )

                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Could not parse JSON for row {index}. Error: {e}. Raw output (first 300 chars): {generated_text[:300]}..."
                    )
                    ner_output = []
                    results_ner.append(
                        {
                            "original_text": job_description_text,
                            "ner_labels": "Parsing Error",
                            "raw_output": generated_text,
                        }
                    )
                except Exception as e:
                    print(
                        f"Warning: Unexpected error during JSON parsing for row {index}: {e}. Raw output (first 300 chars): {generated_text[:300]}..."
                    )
                    ner_output = []
                    results_ner.append(
                        {
                            "original_text": job_description_text,
                            "ner_labels": "Parsing Error",
                            "raw_output": generated_text,
                        }
                    )
            else:
                print(
                    f"Warning: No candidates (response content) found for row {index}. This could be due to safety filters or empty generation."
                )
                generated_text = ""
                ner_output = []
                results_ner.append(
                    {
                        "original_text": job_description_text,
                        "ner_labels": "No Candidates",
                    }
                )

            # Get output tokens from usage_metadata
            output_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                output_tokens = response.usage_metadata.candidates_token_count
                total_output_tokens_sample += output_tokens
            else:
                print(
                    f"Warning: usage_metadata not found for row {index}. Output tokens assumed 0 for cost calculation."
                )

            processed_rows_count_current_run += (
                1  # Increment counter for rows processed in the *current* run
            )

            # --- Save interim results (checkpointing) ---
            # Save based on total accumulated results, not just current run's count
            if (
                start_index + processed_rows_count_current_run
            ) % CHECKPOINT_INTERVAL == 0:
                try:
                    INTERIM_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(INTERIM_RESULTS_PATH, "w", encoding="utf-8") as f:
                        json.dump(results_ner, f, ensure_ascii=False, indent=2)
                    print(
                        f"\nCheckpoint: Saved {len(results_ner)} results to {INTERIM_RESULTS_PATH}"
                    )
                except Exception as e:
                    print(f"Error saving interim results: {e}")

            time.sleep(0.05)  # Small delay to mitigate rate limiting
            break  # Break out of the retry loop on successful API call

        except genai.types.BlockedPromptException as e:
            print(f"Error processing row {index} (Blocked by Safety Settings): {e}")
            results_ner.append(
                {
                    "original_text": job_description_text,
                    "ner_labels": "Blocked by Safety",
                }
            )
            break  # No point in retrying if content is blocked by safety settings
        except Exception as e:
            retries += 1
            print(
                f"Error processing row {index} (API Error, attempt {retries}/{MAX_RETRIES}): {e}"
            )
            if retries < MAX_RETRIES:
                time.sleep(
                    RETRY_DELAY_SECONDS * (2 ** (retries - 1))
                )  # Exponential backoff
            else:
                print(f"Max retries reached for row {index}. Skipping.")
                results_ner.append(
                    {
                        "original_text": job_description_text,
                        "ner_labels": "API Error after retries",
                    }
                )
            time.sleep(0.1)  # Add a small delay even on error

# --- Final Calculation and Extrapolation ---
# Use the total number of items in results_ner for final count, as it includes loaded items
final_total_processed_count = len(results_ner)
if final_total_processed_count == 0:
    print(
        "\n--- ERROR: No rows were successfully processed for token estimation. Cannot compute budget. ---"
    )
else:
    # These averages are for the *current run's* token usage
    if (
        processed_rows_count_current_run > 0
    ):  # Check if any new rows were processed in this run
        avg_input_tokens_per_row_current_run = (
            total_input_tokens_sample / processed_rows_count_current_run
        )
        avg_output_tokens_per_row_current_run = (
            total_output_tokens_sample / processed_rows_count_current_run
        )
    else:
        avg_input_tokens_per_row_current_run = 0
        avg_output_tokens_per_row_current_run = 0

    print(
        f"\n--- Token Averages from {processed_rows_count_current_run} Rows Processed in This Run ---"
    )
    print(
        f"Average Input Tokens per Row (including prompt template): {avg_input_tokens_per_row_current_run:.2f}"
    )
    print(
        f"Average Output Tokens per Row (NER JSON output): {avg_output_tokens_per_row_current_run:.2f}\n"
    )

# Save all results (including loaded ones) to the final output file
output_results_path_json = (
    PROJECT_PATH
    / "data/interim/bootstrapping/001/labels-by-gemini-api.json"  # Changed filename for clarity
)
try:
    output_results_path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_results_path_json, "w", encoding="utf-8") as f:
        json.dump(results_ner, f, ensure_ascii=False, indent=2)
    print(
        f"\nAll {len(results_ner)} NER results (including loaded ones) saved to: {output_results_path_json}"
    )
except Exception as e:
    print(f"Error saving final NER results: {e}")
