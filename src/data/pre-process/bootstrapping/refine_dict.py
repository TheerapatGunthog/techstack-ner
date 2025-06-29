import json
from pathlib import Path

PROJECT_PATH = Path("/home/whilebell/Code/techstack-ner/")

# Specify the path of the original dictionary file
INPUT_DICT_PATH = (
    PROJECT_PATH / "data/keywords/canonoical_dictionary_gemini_labels.json"
)

# Specify the path for the new file to be saved
OUTPUT_DICT_PATH = PROJECT_PATH / "data/keywords/canonical_dictionary_v2.json"

# ===================================================================


def refine_dictionary(input_path, output_path):
    """
    Loads the canonical dictionary, refines the labels for specific
    groups of words, and saves the result to a new file.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"✅ Loaded {len(data)} words from '{input_path}'")
    except FileNotFoundError:
        print(f"❌ Error: File '{input_path}' not found.")
        return

    # Define word groups to move and new Labels
    # You can add or remove words in these lists as needed
    refinement_rules = {
        # "CERT": ["a+", "ccie", "ccna", "ccnp", "mcsa", "rhce"],
        "FAL": [
            "ajax",
            "api",
            "basecamp",
            "bitbucket",
            "clearcase",
            "git",
            "github",
            "gitlab",
            "http",
            "https",
            "jira",
            "kanban",
            "ldap",
            "mercurial",
            "perforce",
            "rest",
            "scrum",
            "soap",
            "svn",
            "tcp/ip",
            "tdd",
            "uml",
            "waterfall",
            "xp",
            "agile",
            "bdd",
            "ci/cd",
            "devops",
            "lean",
            "travisci",
            "nexus",
            "restful",
        ],
        "ET": ["access point"],
    }

    refined_count = 0

    # Loop through each rule
    for new_label, words_to_move in refinement_rules.items():
        for word in words_to_move:
            # Search for words that start with 'word' (e.g., 'api', 'api development')
            for key, current_label in data.items():
                if key.startswith(word):
                    if current_label != new_label:
                        # print(f"🔄 Changing '{key}': from '{current_label}' -> '{new_label}'")
                        data[key] = new_label
                        refined_count += 1

    print(f"✨ Refined {refined_count} labels.")

    # Save the result to a new file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully saved refined dictionary to '{output_path}'")
    except IOError as e:
        print(f"❌ Error writing file: {e}")


# --- Main ---
if __name__ == "__main__":
    refine_dictionary(INPUT_DICT_PATH, OUTPUT_DICT_PATH)
