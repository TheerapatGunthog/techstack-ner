import json
from pathlib import Path
import random
import re
import requests

# from collections import Counter
# import pandas as pd

# Path to your canonical dictionary
PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")
canonical_dict_path = PROJECT_PATH / "data/keywords/canonical_dictionary_v2.json"


# 1. Load Canonical Dictionary (same as before)
def load_canonical_dictionary(file_path):
    """
    Loads the canonical dictionary from a JSON file.
    Assumes the dictionary maps "term": "ENTITY_TYPE" (e.g., "Python": "PSML").
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entity_types_map = {}
    all_short_entity_types = set()

    for term, entity_type_short in data.items():
        entity_types_map[term.lower()] = entity_type_short
        all_short_entity_types.add(entity_type_short)

    grouped_entities = {etype: [] for etype in all_short_entity_types}
    for term, etype in data.items():
        grouped_entities[etype].append(term)

    return grouped_entities, entity_types_map, sorted(list(all_short_entity_types))


# 2. Function to call Ollama API (same as before)
def call_ollama_api(prompt: str, model_name: str = "phi3") -> str:
    """
    Calls the Ollama API to generate text.
    """
    ollama_api_url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 700},
    }

    try:
        response = requests.post(
            ollama_api_url, headers=headers, json=payload, timeout=300
        )
        response.raise_for_status()

        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        print("Please ensure Ollama server is running and 'phi3' model is downloaded.")
        return ""


# 3. Generate Prompt for LLM (updated, does not require NER tags)
def generate_llm_prompt_sentences_only(
    grouped_entities, num_sentences=3, target_entity_types=["ET", "DB", "CP", "PSML"]
):
    """
    Generates a prompt for the LLM to create sentences using specific entity types,
    without requiring NER tags in the output.
    """
    prompt_parts = [
        f"Generate {num_sentences} diverse and grammatically correct sentences about technology or software development.",
        "Each sentence MUST subtly incorporate and use terms that belong to the following technology categories.",
        "Ensure the incorporated terms are taken from the provided dictionary snippets or are very closely related to them.",
        "Do NOT provide any NER tags (like B-TAG, I-TAG, O) in the output. Just provide the sentences.",
        "\n**Dictionary Snippet (for reference, use these types and similar terms):**",
    ]

    for entity_type in target_entity_types:
        if entity_type in grouped_entities:
            sample_entities = random.sample(
                grouped_entities[entity_type],
                min(5, len(grouped_entities[entity_type])),
            )
            prompt_parts.append(
                f"- {entity_type} (e.g., {' '.join(sample_entities)}): Technologies, Devices, Platforms, Programming Languages."
            )
        else:
            print(
                f"Warning: Entity type '{entity_type}' not found in grouped entities from dictionary."
            )

    prompt_parts.append(
        "\n**Output Format (STRICTLY ADHERE TO THIS FORMAT - Sentences only):**"
    )
    for i in range(1, num_sentences + 1):
        prompt_parts.append(f"Sentence {i}: [Sentence text]")

    # Add a clearer example output
    prompt_parts.append("\n**Example Output (CRITICAL for format guidance):**")
    prompt_parts.append(
        "Sentence 1: Our team prefers working with MongoDB on AWS services."
    )
    prompt_parts.append(
        "Sentence 2: The Arduino board collects data for PostgreSQL analysis."
    )
    prompt_parts.append(
        "Sentence 3: We built a new application using Java and deployed it on Google Cloud."
    )
    prompt_parts.append(
        "Sentence 4: Migrating from Oracle to Cassandra requires careful planning and execution."
    )

    return "\n".join(prompt_parts)


# 4. Process LLM output (updated, does not parse NER tags)
def parse_llm_output_sentences_only(llm_output: str):
    """
    Parses the LLM output (expected to be sentences only).
    Returns a list of dictionaries, each containing:
    - 'original_text': The generated sentence.
    - 'source': Origin of the data.
    """
    parsed_sentences = []

    # Use regex to split each sentence by "Sentence X: " pattern
    # re.DOTALL: Make '.' match also newlines.
    # re.MULTILINE: Make '^' and '$' match the start/end of each line.

    # Method 1: Separate each sentence using "Sentence X:" as a separator
    sentence_matches = re.finditer(
        r"Sentence \d+:\s*(.*?)(?=\nSentence \d+:|\Z)", llm_output, re.DOTALL
    )

    for i, match in enumerate(sentence_matches):
        sentence_text = re.sub(
            r"\*\*|\*", "", match.group(1)
        ).strip()  # Remove ** or * that LLM may include

        if sentence_text:
            parsed_sentences.append(
                {
                    "original_text": sentence_text,
                    "source": "LLM_Augmentation_Sentences_Only",
                }
            )
        else:
            print(f"Warning: Empty sentence parsed at index {i}. Skipping.")

    return parsed_sentences


# --- Main Execution Flow ---
if __name__ == "__main__":
    print(f"Loading canonical dictionary from: {canonical_dict_path}")
    grouped_entities, entity_types_map, all_short_entity_types = (
        load_canonical_dictionary(canonical_dict_path)
    )

    print("Sample Loaded Canonical Entities (Grouped):")
    for etype, terms in list(grouped_entities.items())[:5]:
        print(f"  {etype}: {len(terms)} terms (e.g., {terms[:3]})")
    print(f"All recognized short entity types: {all_short_entity_types}")

    target_low_frequency_entity_types = ["ET", "DB", "CP", "PSML"]

    valid_target_entity_types = [
        t for t in target_low_frequency_entity_types if t in all_short_entity_types
    ]
    if not valid_target_entity_types:
        print(
            "\nError: None of the specified target entity types found in the canonical dictionary. Please check your dictionary file and target types."
        )
        exit()

    print(f"\nTargeting entity types for augmentation: {valid_target_entity_types}")

    # 1. Generate Prompt for LLM to create sentences only
    llm_prompt = generate_llm_prompt_sentences_only(
        grouped_entities,
        num_sentences=5,  # Request 5 sentences
        target_entity_types=valid_target_entity_types,
    )

    print("\n" + "=" * 50)
    print("GENERATED LLM PROMPT (truncated for display):")
    print("=" * 50)
    print(llm_prompt[:1000] + "..." if len(llm_prompt) > 1000 else llm_prompt)
    print("=" * 50)

    # 2. Call Ollama API
    print("\nAttempting to call Ollama API with phi3 model...")
    llm_raw_output = call_ollama_api(llm_prompt, model_name="phi3")

    if not llm_raw_output:
        print("\nFailed to get response from Ollama. Exiting.")
        exit()

    print("\n" + "=" * 50)
    print("OLLAMA RAW OUTPUT (Sentences Only):")
    print("=" * 50)
    print(llm_raw_output)
    print("=" * 50)

    # 3. Process LLM output
    augmented_sentences_only = parse_llm_output_sentences_only(llm_raw_output)

    print("\n" + "=" * 50)
    print("PARSED AUGMENTED SENTENCES (No NER Tags from LLM):")
    print("=" * 50)

    if augmented_sentences_only:
        # Print the result as JSON Array of Object
        print(json.dumps(augmented_sentences_only, indent=2, ensure_ascii=False))
    else:
        print("No sentences were generated from LLM output.")
