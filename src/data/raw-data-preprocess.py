import sys
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import spacy

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data.raw import RAW_DATA_PATH
from data.interim import INTERIM_DATA_PATH

tqdm.pandas()


class DataCleaning:
    """Cleaning and filtering"""

    def __init__(self):
        self.html_tags_pattern = re.compile(r"<.*?>")
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        self.phone_pattern = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
        self.css_pattern = re.compile(r"<style.*?>.*?</style>", re.DOTALL)
        self.url_pattern = re.compile(r"http\S+|www\S+")
        self.css_inline_pattern = re.compile(r"\.[a-zA-Z0-9_-]+\s*\{.*?\}", re.DOTALL)
        self.bullet_point_pattern = re.compile(r"[-•*]")
        self.non_alphanumeric_pattern = re.compile(r"[^a-zA-Z0-9.'\"#\+\-\._, /\\&]+")
        self.hashtag = re.compile(r"#\w+")

    def filter_non_thai(self, text):
        """Removes Thai characters"""
        if isinstance(text, str) and not re.search(r"[\u0E00-\u0E7F]", text):
            return text
        return None

    def clean_text(self, text):
        """Cleans text"""
        if not isinstance(text, str):
            return ""
        text = self.css_pattern.sub("", text)
        text = self.css_inline_pattern.sub("", text)
        text = self.html_tags_pattern.sub("", text)
        text = self.email_pattern.sub("", text)
        text = self.phone_pattern.sub("", text)
        text = self.url_pattern.sub("", text)
        text = self.bullet_point_pattern.sub("", text)
        text = self.hashtag.sub("", text)
        text = self.non_alphanumeric_pattern.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class DataSegmentation:
    """Segments text data into sentences using Spacy."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def split_sentences(self, text):
        """Splits text into sentences using Spacy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]


class DataQualityCheck:
    """Quality checks on dataset."""

    def __init__(self, df):
        self.df = df

    def check_missing_values(self):
        """Checks for missing values"""
        missing_values = self.df.isnull().sum()
        print("\n Missing Values:")
        print(missing_values)

    def check_duplicates(self):
        """Checks for duplicate values"""
        duplicated_values = self.df.duplicated().sum()
        print("\n Duplicated Values : ", duplicated_values)

    def run_checks(self):
        """Runs data quality checks."""
        print("\n Running Data Quality Checks...")
        self.check_missing_values()
        self.check_duplicates()


class JobDataProcessor:
    """Main class"""

    def __init__(self, raw_data_path, interim_data_path):
        self.raw_data_path = raw_data_path
        self.interim_data_path = interim_data_path
        self.cleaner = DataCleaning()
        self.segmenter = DataSegmentation()

    def process(self):
        """segmenting, and checking quality."""
        df = pd.read_csv(self.raw_data_path / "merged.csv").iloc[0:1000]

        # Drop duplicate qualifications
        df.drop_duplicates(subset=["Qualification"], inplace=True)

        # Filter non-Thai topics and qualifications
        df["Topic"] = df["Topic"].progress_apply(self.cleaner.filter_non_thai)
        df["Qualification"] = df["Qualification"].progress_apply(
            self.cleaner.filter_non_thai
        )
        df.dropna(subset=["Topic", "Qualification"], inplace=True)

        # Clean qualification text
        df["Qualification"] = df["Qualification"].apply(self.cleaner.clean_text)

        # Segment qualification text into sentences
        df["Sentence_Index"] = df.index
        df["Segmented_Qualification"] = df["Qualification"].progress_apply(
            self.segmenter.split_sentences
        )
        df = df.explode("Segmented_Qualification").reset_index(drop=True)
        df = df[["Topic", "Sentence_Index", "Segmented_Qualification"]]

        df.drop_duplicates(subset=["Segmented_Qualification"], inplace=True)

        # Save processed data
        df.to_csv(self.interim_data_path / "segmented_data.csv", index=False)

        # Run data quality checks
        data_quality = DataQualityCheck(df)
        data_quality.run_checks()


if __name__ == "__main__":
    processor = JobDataProcessor(RAW_DATA_PATH, INTERIM_DATA_PATH)
    processor.process()
