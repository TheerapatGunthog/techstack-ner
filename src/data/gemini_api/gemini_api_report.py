import google.generativeai as genai
import pandas as pd
from pathlib import Path
import time  # For potential rate limiting
import json  # To handle JSON output from Gemini
import os
import sys  # Add this for stdout redirection
import re  # Import re for robust JSON extraction
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Output File Configuration ---
# Define the directory and filename for your output text file
# All subsequent print() statements after redirection will go to this file
OUTPUT_DIR = Path(
    "/home/whilebell/Code/techstack-ner/reports/"
)  # Directory to save the output
OUTPUT_FILENAME = "gemini_budget_report.txt"
OUTPUT_FILE_PATH = OUTPUT_DIR / OUTPUT_FILENAME

# Ensure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Store the original stdout (console)
original_stdout = sys.stdout

# Open the output file in write mode ('w') and redirect stdout
# All print() calls from this point will write to the file
sys.stdout = open(OUTPUT_FILE_PATH, "w", encoding="utf-8")

# Initial print to the file to confirm redirection
print("--- Gemini API Budget Report ---")
print(f"Report generated on: {time.ctime()}")
print(f"Saving all console output to: {OUTPUT_FILE_PATH}\n")


# --- Project Configuration ---
# Project path directory where your data is located
PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

# Add a column to JSON data
json_file_path = PROJECT_PATH / "data/interim/gemini/filtered_data.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    raw_data_list = json.load(f)

# Declare column name
MY_JOB_DESCRIPTION_COLUMN = "Job_Description"

# Removed: print(raw_data_list) - to prevent printing potentially large raw data list to file

# Load dataset from the specified path
df = pd.DataFrame(raw_data_list, columns=[MY_JOB_DESCRIPTION_COLUMN])

# Display head and shape of the DataFrame for verification
print(f"DataFrame Head:\n{df.head()}")
print(
    f"Total Rows in DataFrame: {df.shape[0]}\n"
)  # Also added total rows count back here for consistency


# --- Gemini API Model Configuration ---
# The model name you intend to use.
GEMINI_API_MODEL = "gemini-2.5-flash"

# --- Configure Gemini API Key ---
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_API_MODEL)

# --- Gemini API Pricing Variables (per 1M tokens in USD, from your provided image) ---
INPUT_PRICE_PER_1M_TOKENS = 0.30  # USD for text/image/video input
OUTPUT_PRICE_PER_1M_TOKENS = 2.50  # USD for output

# Context caching prices (from your image, not directly used in this basic token calculation)
CONTEXT_CACH_PRICE = 0.075  # USD (e.g., per 1K tokens for cache write)
CONTEXT_CACH_PRICE_1M_HR_TOKEN_STORAGE = (
    1.0  # USD (e.g., per 1M tokens stored per hour)
)

# Display all prices for user reference
print("--- Gemini API Pricing (Per 1M Tokens) ---")
print(f"Input price (text/image/video): ${INPUT_PRICE_PER_1M_TOKENS:.2f} USD")
print(f"Output price: ${OUTPUT_PRICE_PER_1M_TOKENS:.2f} USD")
print(f"Context cache write price (example): ${CONTEXT_CACH_PRICE:.3f} USD")
print(
    f"Context cache storage price (per 1M token/hr): ${CONTEXT_CACH_PRICE_1M_HR_TOKEN_STORAGE:.1f} USD\n"
)

# --- NER Prompt Template Definition ---
# This defines the task for Gemini API for Named Entity Recognition.
# It includes clear instructions, label definitions, an example, and the desired JSON output format.

# Define your NER classes and their descriptions for the prompt in English.
# Ensure these descriptions are clear and distinct for each category.
NER_CLASSES_DEFINITION = """
- Programming_Scripting_and_Markup_languages: Programming languages, scripting languages, or markup languages (e.g., Python, Java, C++, JavaScript, HTML, CSS, SQL, Shell Script, Go, Ruby, Swift, Kotlin, PHP, C#)
- Database: Database systems or related data storage technologies (e.g., MySQL, PostgreSQL, MongoDB, Redis, Oracle, SQL Server, Cassandra, DynamoDB, Firebase, SQLite)
- Cloud_platforms: Cloud computing platforms and related services/technologies (e.g., AWS, Azure, Google Cloud Platform (GCP), Kubernetes, Docker, Serverless, Terraform, OpenStack)
- Framework_and_Libraries: Software development frameworks, libraries, SDKs, or widely used tools (e.g., React, Angular, Vue.js, Spring Boot, Node.js, Django, .NET, Laravel, Flask, TensorFlow, PyTorch, Keras, NumPy, Pandas, Scikit-learn, Express.js, Bootstrap)
- Embedded_Technologies: Technologies specifically related to embedded systems, IoT, hardware programming, or real-time operating systems (e.g., RTOS, Microcontroller, FPGA, ARM, Raspberry Pi, Arduino, C for embedded, bare-metal programming)
- Technique_and_Skill: General software development skills, methodologies, techniques, or non-platform/language-specific abilities (e.g., Agile, Scrum, CI/CD, Unit Testing, Data Structures, Algorithms, Machine Learning, Deep Learning, RESTful API, OOP, TDD, Version Control, Git, Problem Solving, Communication, Design Patterns)
"""
# --- IMPORTANT: If you have a 7th class, please add it above with its definition! ---

# The complete prompt structure that will be sent to the Gemini API for each job description.
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
"
"""

# --- Step 1: Calculate average tokens from a sample of actual rows ---
# We'll process a sample of 100 rows to get an accurate average token count.
SAMPLE_SIZE = 100
total_input_tokens_sample = 0
total_output_tokens_sample = 0
processed_rows_count = 0  # Counter for successfully processed rows
results_ner = []  # List to store the NER results for analysis (optional)

print(f"--- Processing {SAMPLE_SIZE} Sample Rows for Token Estimation and NER ---")

# Iterate through the first 'SAMPLE_SIZE' rows of the DataFrame
for index, row in df.head(SAMPLE_SIZE).iterrows():
    JOB_DESCRIPTION_COLUMN_NAME = (
        MY_JOB_DESCRIPTION_COLUMN  # Using the variable defined earlier
    )

    if JOB_DESCRIPTION_COLUMN_NAME not in row:
        print(
            f"Error: Column '{JOB_DESCRIPTION_COLUMN_NAME}' not found in DataFrame row {index}. Please adjust 'MY_JOB_DESCRIPTION_COLUMN'. Skipping row."
        )
        continue  # Skip to the next row if column is missing

    job_description_text = row[JOB_DESCRIPTION_COLUMN_NAME]

    # Construct the full prompt using .format()
    # MODIFIED: Use .format() to fill the placeholders
    full_prompt = PROMPT_TEMPLATE.format(
        ner_classes_placeholder=NER_CLASSES_DEFINITION.strip(),
        job_desc_content=job_description_text,
    )

    try:
        # Get input tokens count using model.count_tokens() - this is highly accurate for input
        input_token_response = model.count_tokens(full_prompt)
        input_tokens = input_token_response.total_tokens
        total_input_tokens_sample += input_tokens

        # Call generate_content to get the NER output and measure actual output tokens
        # IMPORTANT: This is an actual API call and will incur costs and time.
        # Consider adding safety_settings if your content might be flagged.
        response = model.generate_content(full_prompt)

        # Extract generated text from the response
        generated_text = ""
        if response.candidates:
            generated_text = response.candidates[0].content.parts[0].text

            # --- Attempt to parse the JSON output more robustly ---
            try:
                # Use regex to find the JSON block, even if it's wrapped in other text
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", generated_text)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    # If no ```json``` block, try to parse the entire text as JSON
                    json_str = generated_text.strip()

                ner_output = json.loads(json_str)
                results_ner.append(
                    {"original_text": job_description_text, "ner_labels": ner_output}
                )

            except json.JSONDecodeError as e:
                print(
                    f"Warning: Could not parse JSON for row {index}. Error: {e}. Raw output: {generated_text[:300]}..."
                )
                ner_output = []  # Indicate failure to parse for this row
                results_ner.append(
                    {
                        "original_text": job_description_text,
                        "ner_labels": "Parsing Error",
                        "raw_output": generated_text,
                    }
                )
            except Exception as e:
                print(
                    f"Warning: Unexpected error during JSON parsing for row {index}: {e}. Raw output: {generated_text[:300]}..."
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
                {"original_text": job_description_text, "ner_labels": "No Candidates"}
            )

        # Get output tokens from usage_metadata (most accurate for output tokens)
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            output_tokens = response.usage_metadata.candidates_token_count
            total_output_tokens_sample += output_tokens
        else:
            # This case should ideally not happen often unless API response is malformed
            print(
                f"Warning: usage_metadata not found for row {index}. Output tokens assumed 0 for cost calculation."
            )

        processed_rows_count += 1
        print(
            f"Row {index} (OK): Input Tokens={input_tokens}, Output Tokens={output_tokens}"
        )

    except genai.types.BlockedPromptException as e:
        print(f"Error processing row {index} (Blocked by Safety Settings): {e}")
        # Log or handle rows blocked by safety filters
        results_ner.append(
            {"original_text": job_description_text, "ner_labels": "Blocked by Safety"}
        )
    except Exception as e:
        print(f"Error processing row {index} (API Error): {e}")
        results_ner.append(
            {"original_text": job_description_text, "ner_labels": "API Error"}
        )
        # Add a short delay to avoid hitting rate limits if many errors occur consecutively
        time.sleep(1)

# --- Final Calculation and Extrapolation ---
if processed_rows_count == 0:
    print(
        "\n--- ERROR: No rows were successfully processed for token estimation. Cannot compute budget. ---"
    )
else:
    # Calculate average tokens per row from the processed sample
    avg_input_tokens_per_row = total_input_tokens_sample / processed_rows_count
    avg_output_tokens_per_row = total_output_tokens_sample / processed_rows_count

    print(f"\n--- Token Averages from {processed_rows_count} Sample Rows ---")
    print(
        f"Average Input Tokens per Row (including prompt template): {avg_input_tokens_per_row:.2f}"
    )
    print(
        f"Average Output Tokens per Row (NER JSON output): {avg_output_tokens_per_row:.2f}\n"
    )

    # Calculate estimated cost for the processed sample size (e.g., 100 rows)
    cost_sample_input = (
        total_input_tokens_sample / 1_000_000
    ) * INPUT_PRICE_PER_1M_TOKENS
    cost_sample_output = (
        total_output_tokens_sample / 1_000_000
    ) * OUTPUT_PRICE_PER_1M_TOKENS
    total_cost_sample = cost_sample_input + cost_sample_output

    print(f"--- Estimated Cost for {processed_rows_count} Sample Rows ---")
    print(f"Input Cost ({processed_rows_count} rows): ${cost_sample_input:.6f}")
    print(f"Output Cost ({processed_rows_count} rows): ${cost_sample_output:.6f}")
    print(f"Total Cost ({processed_rows_count} rows): ${total_cost_sample:.6f}\n")

    # Extrapolate the cost for the total target dataset size (8559 rows)
    TARGET_ROWS = 8559
    scaling_factor = TARGET_ROWS / processed_rows_count
    estimated_total_cost_8559_rows = total_cost_sample * scaling_factor

    print(f"--- Estimated Total Cost for {TARGET_ROWS} Rows ---")
    print(f"Scaling Factor (Target Rows / Sample Rows): {scaling_factor:.2f}")
    print(
        f"Estimated Total Cost for {TARGET_ROWS} Rows: ${estimated_total_cost_8559_rows:.6f}"
    )
    print(
        f"This is an estimation based on the average token usage from {processed_rows_count} sample rows."
    )

# --- Restore stdout to its original value ---
sys.stdout.close()  # Close the file
sys.stdout = original_stdout  # Restore console output

# This final print will show on the console, not in the file
print(f"\nReport generation complete. Check the output file at: {OUTPUT_FILE_PATH}")

# Optional: Save NER results for the sample rows to a JSON file
# Corrected: Removed leading slash for relative path concatenation
output_results_path_json = (
    PROJECT_PATH / "data/processed/gemini_ner_sample_results.json"
)
try:
    # Ensure the directory for processed data exists
    output_results_path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_results_path_json, "w", encoding="utf-8") as f:
        json.dump(results_ner, f, ensure_ascii=False, indent=2)
    print(f"\nNER results for sample rows also saved to: {output_results_path_json}")
except Exception as e:
    print(f"Error saving NER results: {e}")
