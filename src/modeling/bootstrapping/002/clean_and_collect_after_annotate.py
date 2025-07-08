from pathlib import Path

# --- Step 1: Define file paths ---
PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

# Path to your original CoNLL file
input_file_path = (
    PROJECT_PATH
    / "data/interim/bootstrapping/002/project-24-at-2025-07-08-08-49-71020547.conll"
)

# Path for the new cleaned file
output_file_path = input_file_path.with_name("cleaned_" + input_file_path.name)


# --- Step 2: Read and process the file ---
try:
    with open(input_file_path, "r", encoding="utf-8") as f:
        # Split the file content into sentences. Sentences in CoNLL are separated by a blank line.
        sentences = f.read().strip().split("\n\n")

    kept_sentences = []

    for sentence in sentences:
        # Skip if the sentence block is empty
        if not sentence.strip():
            continue

        lines = sentence.split("\n")

        # Extract all tags from the last column of each line
        tags = [line.split()[-1] for line in lines if line.strip()]

        # Create a set of unique entity tags, excluding the 'O' (Outside) tag
        # The set will only contain actual labels like 'B-TAS', 'I-TEC', etc.
        unique_labels = set(tag.split("-")[-1] for tag in tags if tag != "O")

        # --- Filtering Logic ---
        # Keep the sentence if:
        # 1. `unique_labels` is not empty (i.e., there is at least one label other than 'O').
        # 2. `unique_labels` is not just {'TAS'}.
        if unique_labels and unique_labels != {"TAS"}:
            kept_sentences.append(sentence)

    # --- Step 3: Write the cleaned data to a new file ---
    if kept_sentences:
        with open(output_file_path, "w", encoding="utf-8") as f:
            # Join the kept sentences back together with double newlines
            f.write("\n\n".join(kept_sentences))
            f.write("\n")  # Add a final newline for good practice

        print("Processing complete!")
        print(f"Original sentences: {len(sentences)}")
        print(f"Sentences after cleaning: {len(kept_sentences)}")
        print(f"Cleaned file saved to: {output_file_path}")
    else:
        print("No sentences met the criteria. The output file was not created.")

except FileNotFoundError:
    print(f"Error: The file was not found at {input_file_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
