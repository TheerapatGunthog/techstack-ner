import re
from pathlib import Path
import pandas as pd
import logging
from typing import Optional, List, Dict, Union
import os

# ---- Paths ----
PROJECT_PATH = Path(os.getcwd())

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---- Cleaning ----
class DataCleaning:
    def __init__(self, config: Optional[Dict] = None):
        # Default configuration
        self.config = {
            "remove_emails": True,
            "remove_phones": True,
            "remove_urls": True,
            "remove_html": True,
            "remove_css": True,
            "normalize_whitespace": True,
            "remove_special_chars": True,
            "lowercase": False,
            "min_text_length": 10,
            # version and numbers policy
            "remove_versions": True,  # remove version numbers from tech tokens completely
            "remove_standalone_numbers": True,  # remove numbers not tied to letter-led tokens
            # comma handling (applied at the very end of clean_text)
            "comma_handling": "normalize",  # "keep" | "drop" | "space" | "normalize"
        }
        if config:
            self.config.update(config)

        # Regex patterns
        self.pat = {
            "html_tags": re.compile(r"<[^>]+>"),
            "css_block": re.compile(r"<style.*?>.*?</style>", re.DOTALL),
            "css_inline": re.compile(r"\.[a-zA-Z0-9_-]+\s*\{.*?\}", re.DOTALL),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "phone": re.compile(r"(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}"),
            "url": re.compile(r"(http|https)://\S+|www\.\S+"),
            "bullet": re.compile(r"[-•*]"),
            "hashtag": re.compile(r"#\w+"),
            # allow typical tech chars (c++, node-js, next.js, c#, lists with commas)
            "non_alnum": re.compile(r"[^a-zA-Z0-9.'\"#\+\-\._, /\\&]+"),
            "spaces": re.compile(r"\s+"),
            "thai": re.compile(r"[\u0E00-\u0E7F]"),
            # version removal patterns
            # 1) "tool v1.2.3" or "tool version 2" or "tool 2.0" -> capture tool only
            "tech_space_version": re.compile(
                r"\b([A-Za-z][A-Za-z0-9\+\-#\._]{1,})\s*(?:v(?:ersion)?\s*)?\d+(?:\.\d+){0,3}\b",
                re.IGNORECASE,
            ),
            # 2) attached versions: "mysql5.7", "react18", "cuda11.8", "ES6" -> capture tool only
            "tech_attached_version": re.compile(
                r"\b([A-Za-z][A-Za-z0-9\+\-#\._]{1,}?)(\d+(?:\.\d+){0,3})\b"
            ),
            # standalone numbers like 2020, 100%, 1.2 (not preceded by letters)
            "standalone_numbers": re.compile(r"(?<![A-Za-z])\b\d+(?:\.\d+)*%?\b"),
        }

    def filter_non_thai(self, text: Optional[str]) -> Optional[str]:
        """Filter out text containing Thai characters (project focus is English)."""
        if not isinstance(text, str):
            return None
        return None if self.pat["thai"].search(text) else text

    # ---- Version removal ----
    def _remove_versions(self, text: str) -> str:
        """
        Remove version numbers from technology tokens entirely.
        Examples:
          'python v3.10' -> 'python'
          'react18' -> 'react'
          'mysql5.7' -> 'mysql'
          'cuda11.8' -> 'cuda'
        """
        # remove spaced versions: "tool v1.2" / "tool version 2" / "tool 2.0"
        text = self.pat["tech_space_version"].sub(r"\1", text)
        # remove attached versions: "tool12.3" / "tool18"
        text = self.pat["tech_attached_version"].sub(r"\1", text)
        return text

    # ---- Number policy ----
    def _remove_standalone_numbers(self, text: str) -> str:
        """Remove numbers not tied to a preceding letter token."""
        return self.pat["standalone_numbers"].sub(" ", text)

    # ---- Final comma normalization (run at the very end) ----
    def _handle_commas(self, text: str) -> str:
        """
        Final pass for commas:
        - keep: keep commas as-is
        - space/drop: replace any comma group with a single space
        - normalize (default): squeeze any sequence of commas (with/without spaces) into a single ','
          examples:
            'python,,java'   -> 'python,java'
            'spark, ,hadoop' -> 'spark,hadoop'
            ' , , '          -> ','
        """
        mode = self.config.get("comma_handling", "normalize")

        if mode == "keep":
            return text

        if mode in ("space", "drop"):
            return re.sub(r"\s*,\s*", " ", text)

        # normalize mode (default)
        text = re.sub(r"\s*,\s*", ",", text)  # collapse space around commas
        text = re.sub(r",+", ",", text)  # reduce runs of commas to a single comma
        return text

    # ---- Main clean ----
    def clean_text(self, text: Optional[str]) -> str:
        if not isinstance(text, str):
            return ""

        # 1) structural noise
        if self.config["remove_css"]:
            text = self.pat["css_block"].sub(" ", text)
            text = self.pat["css_inline"].sub(" ", text)

        if self.config["remove_html"]:
            text = self.pat["html_tags"].sub(" ", text)

        if self.config["remove_emails"]:
            text = self.pat["email"].sub(" ", text)

        if self.config["remove_phones"]:
            text = self.pat["phone"].sub(" ", text)

        if self.config["remove_urls"]:
            text = self.pat["url"].sub(" ", text)

        # 2) bullets and hashtags
        text = self.pat["bullet"].sub(" ", text)
        text = self.pat["hashtag"].sub(" ", text)

        # 3) remove version numbers completely
        if self.config.get("remove_versions", True):
            text = self._remove_versions(text)

        # 4) special char cleanup (commas preserved by pattern)
        if self.config["remove_special_chars"]:
            text = self.pat["non_alnum"].sub(" ", text)

        # 5) remove standalone numbers (years, percents, list indices)
        if self.config.get("remove_standalone_numbers", True):
            text = self._remove_standalone_numbers(text)

        # 6) lowercase if requested
        if self.config["lowercase"]:
            text = text.lower()

        # 7) whitespace normalization
        if self.config["normalize_whitespace"]:
            text = self.pat["spaces"].sub(" ", text).strip()

        # 8) FINAL: comma normalization
        text = self._handle_commas(text)

        return text


# ---- HF sentence segmenter ----
class HFSentenceSegmenter:
    def __init__(
        self,
        model_name: str = "igorsterner/xlmr-multilingual-sentence-segmentation",
        aggregation_strategy: str = "simple",
        device: int = -1,
    ):
        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy
        self.device = device
        self.pipe = None
        self._init_pipeline()

    def _init_pipeline(self):
        """Try to load Hugging Face segmentation model, fallback to regex."""
        try:
            from transformers import pipeline

            self.pipe = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy=self.aggregation_strategy,
                device=self.device,
            )
            logger.info(f"Loaded HF segmenter: {self.model_name}")
        except Exception as e:
            self.pipe = None
            logger.warning(
                f"Cannot init HF segmenter ({self.model_name}): {e}. Using regex fallback."
            )

    def split(self, text: str) -> List[str]:
        """Split text into sentences using HF model or regex fallback."""
        if not isinstance(text, str) or not text.strip():
            return []
        if self.pipe is None:
            return self._regex_fallback(text)
        try:
            preds = self.pipe(text)
            boundaries = []
            for p in preds:
                label = (p.get("entity_group") or p.get("entity") or "").upper()
                if (
                    "SEG" in label
                    or "SENT" in label
                    or label.endswith("-E")
                    or label.startswith("B-SEG")
                    or label.startswith("E-SEG")
                ):
                    boundaries.append(int(p["end"]))
            if not boundaries:
                return self._regex_fallback(text)

            boundaries = sorted(set(boundaries))
            sents, prev = [], 0
            n = len(text)
            for b in boundaries:
                b = max(prev, min(b, n))
                chunk = text[prev:b].strip()
                if chunk:
                    sents.append(chunk)
                prev = b
            tail = text[prev:].strip()
            if tail:
                sents.append(tail)
            return [s for s in sents if s]
        except Exception as e:
            logger.warning(f"HF segmentation failed: {e}")
            return self._regex_fallback(text)

    @staticmethod
    def _regex_fallback(text: str) -> List[str]:
        """Simple regex-based sentence splitter."""
        parts = re.split(r"[.!?]+", text)
        return [p.strip() for p in parts if p and p.strip()]


# ---- Processor ----
class JobDataProcessor:
    def __init__(
        self,
        raw_data_path: Union[str, Path],
        interim_data_path: Union[str, Path],
        config: Optional[Dict] = None,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.interim_data_path = Path(interim_data_path)
        self.config = {
            "input_file": "training-data/job_title_des.csv",
            "output_file": "segmented_data.csv",
            "min_sentence_length": 5,
            "segment_sentences": True,
            "sample_fraction": 1.0,
            "random_seed": 42,
            "save_intermediate": False,
            "use_gpu": False,
        }
        if config:
            self.config.update(config)

        self.cleaner = DataCleaning(config)
        device = 0 if self.config.get("use_gpu") else -1
        self.segmenter = HFSentenceSegmenter(device=device)

        self.interim_data_path.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        fp = self.raw_data_path / self.config["input_file"]
        logger.info(f"Loading {fp}")
        df = pd.read_csv(fp)
        return df

    def process(self) -> pd.DataFrame:
        df = self.load_data()

        # drop duplicates by Qualification
        # before = len(df)
        # df = df.drop_duplicates(subset=["Qualification"]).reset_index(drop=True)
        # logger.info(f"Removed {before - len(df)} duplicate qualifications")

        # filter out Thai text
        df["Topic"] = df["Topic"].apply(self.cleaner.filter_non_thai)
        df["Qualification"] = df["Qualification"].apply(self.cleaner.filter_non_thai)
        df = df.dropna(subset=["Topic", "Qualification"]).reset_index(drop=True)

        # clean Qualification text
        df["Qualification"] = df["Qualification"].apply(self.cleaner.clean_text)

        # sentence segmentation
        df["Sentence_Index"] = df.index
        if self.config["segment_sentences"]:
            df["Segmented_Qualification"] = df["Qualification"].apply(
                self.segmenter.split
            )
            df = df.explode("Segmented_Qualification").reset_index(drop=True)
            df = df[
                df["Segmented_Qualification"].astype(str).str.len()
                >= self.config["min_sentence_length"]
            ]
        else:
            df = df.rename(columns={"Qualification": "Segmented_Qualification"})

        # drop duplicates again
        # before = len(df)
        # df = df.drop_duplicates(subset=["Segmented_Qualification"]).reset_index(
        #     drop=True
        # )
        # logger.info(f"Removed {before - len(df)} duplicate segmented qualifications")

        df = df[["Topic", "Position", "Sentence_Index", "Segmented_Qualification"]]

        # save output
        out = self.interim_data_path / self.config["output_file"]
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Saved to {out}")
        return df


if __name__ == "__main__":
    config = {
        "input_file": "data/raw/classified/tech_jobs.csv",
        "output_file": "data/interim/preprocessed-data/product_data.csv",
        "segment_sentences": True,
        "min_sentence_length": 10,
        "use_gpu": True,
    }
    processor = JobDataProcessor(PROJECT_PATH, PROJECT_PATH, config)
    _ = processor.process()
    print("Done.")
