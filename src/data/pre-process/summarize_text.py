import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # For a progress bar

# --- Setup ---
# Ensure you have tqdm installed: pip install tqdm

# Model name
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Add a padding token if it doesn't exist. This is good practice for batching.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.resize_token_embeddings(len(tokenizer))


# --- File Paths ---
# Declaration of the project root directory
# Please ensure this path is correct for your system
PROJECT_PATH = Path(
    "/home/whilebell/Code/techstack-ner/"
)  # Assuming the script runs in the project root
DATA_PATH = (
    PROJECT_PATH / "data/interim/preprocessed-data/scraping-segmented-data.csv"
)  # Adjusted for local testing
OUTPUT_PATH = PROJECT_PATH / "data/interim/summarize_text/scraping-summarize-data.csv"

# --- Load the dataset ---
print(f"Loading dataset from: {DATA_PATH}")
try:
    # We load the full dataframe and will process the first 10 rows as specified
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: The file was not found at {DATA_PATH}.")
    print("Please make sure the DATA_PATH variable is set correctly.")
    exit()

# Define the columns you need
TOPIC_COLUMN = "Topic"
QUALIFICATION_COLUMN = "Segmented_Qualification"

# Ensure the columns exist in the DataFrame
if TOPIC_COLUMN not in df.columns or QUALIFICATION_COLUMN not in df.columns:
    print(
        f"Error: One of the required columns ('{TOPIC_COLUMN}', '{QUALIFICATION_COLUMN}') not found in the CSV."
    )
    exit()


# --- Improved Prompt Engineering ---
def create_summary_prompt(topic: str, qualification: str) -> str:
    """Creates a structured prompt for qualification summarization."""
    return f"""
[INST]
Your task is to act as an expert in recruitment and talent acquisition. Given the job title and qualifications below, extract a list of only the essential skills. **Your primary goal is to make the list as brief and concise as possible, using keywords or short phrases instead of full sentences.** Remove all non-essential information, marketing language, and generic soft skills. The final output must be a clean list of keywords suitable for training a machine learning model.

**Job Title:** "{topic}"
**Qualifications:** "{qualification}"
[/INST]
"""


# --- Summarization Function ---
# ***** FIX 1: The function now accepts both topic and qualification *****
def summarize_text(topic: str, qualification: str) -> str:
    """
    Generates a summary for a given text using the pre-loaded model.
    Returns an empty string if the input text is invalid.
    """
    if not isinstance(qualification, str) or not qualification.strip():
        return ""  # Return empty if the cell is empty or not a string

    # The prompt now receives both arguments
    prompt = create_summary_prompt(topic, qualification)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the summary
    try:
        output = model.generate(
            **inputs,
            max_new_tokens=384,  # Increased slightly for bullet points
            do_sample=True,
            top_k=10,
            top_p=0.95,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode and extract the summary
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        summary = full_response.split("[/INST]")[-1].strip()
        return summary

    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return "[ERROR: Could not generate summary]"


# --- Main Processing Loop ---
# ***** FIX 2: The loop now iterates over rows to access both columns *****
print(f"Starting summarization for {len(df)} rows...")
summaries = []
# Use tqdm to iterate over DataFrame rows for a live progress bar
for index, row in tqdm(
    df.iterrows(), total=df.shape[0], desc="Summarizing Qualifications"
):
    topic_to_summarize = row[TOPIC_COLUMN]
    qualification_to_summarize = row[QUALIFICATION_COLUMN]

    # Pass both topic and qualification to the summarization function
    summary = summarize_text(topic_to_summarize, qualification_to_summarize)
    summaries.append(summary)

# --- Save Results ---
# Add the summaries as a new column to the DataFrame
df["Qualification_Summary"] = summaries

# Keep only the relevant columns for the final output
df_output = df[[TOPIC_COLUMN, "Qualification_Summary"]]

# Save the updated DataFrame to a new CSV file
print(f"\nSummarization complete. Saving results to: {OUTPUT_PATH}")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
df_output.to_csv(OUTPUT_PATH, index=False)

print("Process finished successfully.")
