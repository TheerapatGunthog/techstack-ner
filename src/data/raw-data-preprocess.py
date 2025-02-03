import sys
import yaml
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import spacy

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data.keywords import KEYWORDS_DATA_PATH
from data.raw import RAW_DATA_PATH
from data.interim import INTERIM_DATA_PATH

tqdm.pandas()


class MergeDataset:
    def __init__(self, raw_data_path):
        self.data_directory = Path(raw_data_path) / "csv1"

    def merge_csv_files(self, output_file):
        all_files = list(self.data_directory.glob("*.csv"))
        df_list = [pd.read_csv(file) for file in all_files]
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(output_file, index=False)


def load_keywords(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data.get("keywords", [])


def escape_regex(keyword):
    return re.escape(keyword).replace(r"\ ", " ")


class DataSelection:
    def __init__(self, raw_data_path, keywords_data_path):
        self.dataset = Path(raw_data_path) / "merged.csv"
        self.df = pd.read_csv(self.dataset)
        self.keywords = load_keywords(
            Path(keywords_data_path) / "jobs-title-keyword-copy.yaml"
        )

    def clean_unused_columns(self):
        self.df = self.df[["Topic", "Qualification"]]
        self.df.dropna(subset=["Topic"], inplace=True)

        # Remove rows that contain Thai characters
        self.df = self.df[~self.df["Topic"].str.contains(r"[\u0E00-\u0E7F]", na=False)]

    def jobs_title_filter(self):
        self.df["Topic"] = self.df["Topic"].str.lower().fillna("")

        escaped_keywords = [escape_regex(keyword) for keyword in self.keywords]
        pattern = r"\b(" + "|".join(escaped_keywords) + r")\b"

        regex_filtered_df = self.df[
            self.df["Topic"].str.contains(pattern, case=False, na=False, regex=True)
        ]

        self.df = pd.concat([regex_filtered_df]).reset_index(drop=True)

    def jobs_description_filter(self):
        # Remove rows that contain Thai characters
        self.df = self.df[
            ~self.df["Qualification"].str.contains(r"[\u0E00-\u0E7F]", na=False)
        ]

        self.df = self.df.drop_duplicates(subset=["Qualification"]).reset_index(
            drop=True
        )
        return self.df


class DataCleaning:
    def __init__(self, df):
        self.df = df
        self.html_tags_pattern = re.compile(r"<.*?>")
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        self.phone_pattern = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
        self.css_pattern = re.compile(r"<style.*?>.*?</style>", re.DOTALL)
        self.url_pattern = re.compile(r"http\S+|www\S+")
        self.css_inline_pattern = re.compile(r"\.[a-zA-Z0-9_-]+\s*\{.*?\}", re.DOTALL)
        self.bullet_point_pattern = re.compile(r"[-•*]")
        self.non_alphanumeric_pattern = re.compile(r"[^a-zA-Z0-9.'\"#\+\-\._ ]+")

    def clean(self, text):
        # Remove CSS styles
        text = self.css_pattern.sub("", text)
        text = self.css_inline_pattern.sub("", text)
        # Remove HTML tags
        text = self.html_tags_pattern.sub("", text)
        # Remove emails
        text = self.email_pattern.sub("", text)
        # Remove phone numbers
        text = self.phone_pattern.sub("", text)
        # Remove URLs
        text = self.url_pattern.sub("", text)
        # Remove bullet points
        text = self.bullet_point_pattern.sub("", text)
        # Remove non-alphanumeric characters
        text = self.non_alphanumeric_pattern.sub(" ", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


class DataSegmentation:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def split_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]


if __name__ == "__main__":
    raw_data_path = RAW_DATA_PATH
    interim_data_path = INTERIM_DATA_PATH
    keywords_data_path = KEYWORDS_DATA_PATH

    # Data Selection
    data = DataSelection(raw_data_path, keywords_data_path)
    data.clean_unused_columns()
    data.jobs_title_filter()
    selectiondata = data.jobs_description_filter()

    # Data Cleaning
    cleaningdata = selectiondata
    cleaningdata["Qualification"] = cleaningdata["Qualification"].apply(
        DataCleaning(cleaningdata).clean
    )

    print(cleaningdata)

    # Data Segmentation
    segmenter = DataSegmentation()
    segmentationdata = cleaningdata.copy()
    segmentationdata["Sentence_Index"] = segmentationdata.index

    segmentationdata["Segmented_Qualification"] = segmentationdata[
        "Qualification"
    ].progress_apply(segmenter.split_sentences)

    segmentationdata = segmentationdata.explode("Segmented_Qualification").reset_index(
        drop=True
    )

    segmentationdata = segmentationdata[
        ["Topic", "Sentence_Index", "Segmented_Qualification"]
    ]

    segmentationdata.to_csv(interim_data_path / "segmented_data.csv", index=False)

    print(segmentationdata)
