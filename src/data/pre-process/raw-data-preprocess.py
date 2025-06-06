import sys
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import spacy
import logging
from typing import List, Dict, Optional, Union, Set
import multiprocessing as mp
import string
import nltk
from nltk.corpus import stopwords
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


try:
    import wordninja

    WORDNINJA_AVAILABLE = True
except ImportError:
    WORDNINJA_AVAILABLE = False

try:
    import wordsegment

    wordsegment.load()
    WORDSEGMENT_AVAILABLE = True
except ImportError:
    WORDSEGMENT_AVAILABLE = False

# Add path to the project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

try:
    from data.raw import RAW_DATA_PATH
    from data.interim import INTERIM_DATA_PATH
except ImportError:
    # Fallback paths
    RAW_DATA_PATH = Path("./data/raw")
    INTERIM_DATA_PATH = Path("./data/interim")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("data_preprocessing.log")],
)
logger = logging.getLogger(__name__)

# Configure pandas to display more info
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Initialize progress bar for pandas operations
tqdm.pandas()

# Try to download NLTK resources if not already downloaded
try:
    nltk.download("stopwords", quiet=True)
except:
    logger.warning("Could not download NLTK stopwords. Will proceed without them.")


class DataCleaning:
    """Class for text cleaning and filtering operations."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with regex patterns and configuration.

        Args:
            config: Optional configuration dictionary to override defaults
        """
        # Default configuration
        self.config = {
            "remove_emails": True,
            "remove_phones": True,
            "remove_urls": True,
            "remove_html": True,
            "remove_css": True,
            "normalize_whitespace": True,
            "remove_special_chars": True,
            "min_text_length": 10,
            "remove_stopwords": False,  # Default to False to preserve sentence structure
            "lowercase": False,  # Default to False to preserve case information
        }

        # Override with provided config if any
        if config:
            self.config.update(config)

        # Compile regex patterns for better performance
        self.patterns = {
            "html_tags": re.compile(r"<[^>]+>"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}"),
            "css": re.compile(r"<style.*?>.*?</style>", re.DOTALL),
            "url": re.compile(r"(http|https)://[^\s]+|www\.[^\s]+"),
            "css_inline": re.compile(r"\.[a-zA-Z0-9_-]+\s*\{.*?\}", re.DOTALL),
            "bullet_point": re.compile(r"[-•*]"),
            "non_alphanumeric": re.compile(r"[^a-zA-Z0-9.'\"#\+\-\._, /\\&]+"),
            "hashtag": re.compile(r"#\w+"),
            "excessive_whitespace": re.compile(r"\s+"),
            "punctuation": re.compile(f"[{re.escape(string.punctuation)}]"),
        }

        # Initialize stopwords if needed
        self.stopwords = set()
        if self.config["remove_stopwords"]:
            try:
                self.stopwords = set(stopwords.words("english"))
            except:
                logger.warning(
                    "NLTK stopwords not available. Stopword removal disabled."
                )
                self.config["remove_stopwords"] = False

    def filter_non_thai(self, text: Optional[str]) -> Optional[str]:
        """
        Keep only text that doesn't contain Thai characters.

        Args:
            text: Input text

        Returns:
            Text if no Thai characters present, None otherwise
        """
        if not isinstance(text, str):
            return None

        # Check for Thai Unicode range
        if re.search(r"[\u0E00-\u0E7F]", text):
            return None
        return text

    def clean_text(self, text: Optional[str]) -> str:
        """
        Apply a series of cleaning operations to text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Apply configured cleaning operations
        if self.config["remove_css"]:
            text = self.patterns["css"].sub(" ", text)
            text = self.patterns["css_inline"].sub(" ", text)

        if self.config["remove_html"]:
            text = self.patterns["html_tags"].sub(" ", text)

        if self.config["remove_emails"]:
            text = self.patterns["email"].sub(" ", text)

        if self.config["remove_phones"]:
            text = self.patterns["phone"].sub(" ", text)

        if self.config["remove_urls"]:
            text = self.patterns["url"].sub(" ", text)

        # Replace bullet points with space
        text = self.patterns["bullet_point"].sub(" ", text)

        # Remove hashtags
        text = self.patterns["hashtag"].sub(" ", text)

        if self.config["remove_special_chars"]:
            text = self.patterns["non_alphanumeric"].sub(" ", text)

        if self.config["normalize_whitespace"]:
            text = self.patterns["excessive_whitespace"].sub(" ", text).strip()

        if self.config["lowercase"]:
            text = text.lower()

        if self.config["remove_stopwords"] and self.stopwords:
            words = text.split()
            words = [word for word in words if word.lower() not in self.stopwords]
            text = " ".join(words)

        return text

    def clean_text_batch(self, texts: List[Optional[str]]) -> List[str]:
        """
        Clean multiple texts in parallel.

        Args:
            texts: List of input texts

        Returns:
            List of cleaned texts
        """
        # Use multiprocessing if the batch is large enough
        if len(texts) > 1000:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = list(pool.map(self.clean_text, texts))
            return results
        else:
            return [self.clean_text(text) for text in texts]


class DataSegmentation:
    """Class for segmenting text into sentences."""

    def __init__(self, model_name: str = "en_core_web_sm", batch_size: int = 1000):
        """
        Initialize with spaCy model.

        Args:
            model_name: Name of spaCy model to use
            batch_size: Size of batches for processing
        """
        try:
            self.nlp = spacy.load(model_name)
            # ใช้ sentencizer แทน senter
            self.nlp.add_pipe("sentencizer")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            logger.info("Trying to download the model...")
            try:
                # Try to download the model
                import subprocess

                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name], check=True
                )
                self.nlp = spacy.load(model_name)
                self.nlp.add_pipe("sentencizer")
            except Exception as download_error:
                logger.error(f"Failed to download spaCy model: {str(download_error)}")
                raise

        self.batch_size = batch_size

    def split_sentences(self, text: Optional[str]) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        if not isinstance(text, str) or not text.strip():
            return []

        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            return sentences
        except Exception as e:
            logger.warning(f"Error in sentence splitting: {str(e)}")
            # Fallback to a simple sentence splitter
            simple_sentences = re.split(r"[.!?]+", text)
            return [sent.strip() for sent in simple_sentences if sent.strip()]

    def split_sentences_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Process multiple texts with spaCy's efficient batch processing.

        Args:
            texts: List of input texts

        Returns:
            List of lists of sentences
        """
        if not texts:
            return []

        # Filter out None and empty strings
        valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]

        if not valid_texts:
            return [[] for _ in texts]

        try:
            # Process in batches for better performance
            all_sentences = []
            for i in range(0, len(valid_texts), self.batch_size):
                batch = valid_texts[i : i + self.batch_size]
                docs = list(self.nlp.pipe(batch))
                batch_sentences = [
                    [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                    for doc in docs
                ]
                all_sentences.extend(batch_sentences)

            # Create result list with correct alignment to input texts
            result = []
            valid_idx = 0
            for text in texts:
                if isinstance(text, str) and text.strip():
                    result.append(all_sentences[valid_idx])
                    valid_idx += 1
                else:
                    result.append([])
            return result

        except Exception as e:
            logger.error(f"Batch sentence splitting failed: {str(e)}")
            # Fall back to individual processing
            return [self.split_sentences(text) for text in texts]


class SimilarSentenceRemover:
    """Class for detecting and removing similar sentences."""

    def __init__(self, similarity_threshold=0.85, chunk_size=1000, n_jobs=None):
        """
        Initialize with similarity threshold and processing parameters.

        Args:
            similarity_threshold: Threshold above which sentences are considered similar (0.0-1.0)
            chunk_size: Size of chunks for processing large datasets
            n_jobs: Number of processes to use (None = use all available)
        """
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs if n_jobs is not None else max(1, mp.cpu_count() - 1)
        self.logger = logging.getLogger(__name__)

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=10000,
            min_df=1,  # ลดลงจาก 2 เป็น 1 (รับคำที่ปรากฏเพียงครั้งเดียว)
            max_df=1.0,  # เพิ่มจาก 0.95 เป็น 1.0 (รับคำที่ปรากฏในทุกเอกสาร)
            stop_words="english",
            ngram_range=(1, 2),
            strip_accents="unicode",
        )

    def normalize_text(self, text):
        """
        Perform basic text normalization to improve similarity detection.

        Args:
            text: Input text string

        Returns:
            Normalized text string
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)

        # Remove topic prefix if present (e.g., "[Software Engineer] ")
        text = re.sub(r"^\[.*?\]\s*", "", text)

        # Remove punctuation that doesn't change meaning significantly
        text = re.sub(r'[,;:.!?()"\']', "", text)

        # Replace numbers with <NUM> token
        text = re.sub(r"\b\d+\b", "<NUM>", text)

        return text.strip()

    def find_similar_sentences(self, sentences):
        """
        Find groups of similar sentences using TF-IDF and cosine similarity.

        Args:
            sentences: List of sentences to process

        Returns:
            Dictionary mapping sentence indices to cluster IDs
        """
        if len(sentences) <= 1:
            return {}

        # Create normalized versions of sentences for similarity detection
        normalized_sentences = [self.normalize_text(sent) for sent in sentences]

        # Filter out empty strings after normalization
        valid_indices = [i for i, sent in enumerate(normalized_sentences) if sent]
        if len(valid_indices) <= 1:
            return {}

        valid_sentences = [normalized_sentences[i] for i in valid_indices]

        # Calculate TF-IDF matrix
        try:
            # หากล้มเหลวด้วย default parameters ให้ลองใช้ค่าที่ยืดหยุ่นมากขึ้น
            try:
                tfidf_matrix = self.vectorizer.fit_transform(valid_sentences)
            except ValueError as e:
                self.logger.warning(
                    f"Trying with more flexible TF-IDF parameters: {str(e)}"
                )
                backup_vectorizer = TfidfVectorizer(
                    lowercase=True, min_df=1, max_df=1.0
                )
                tfidf_matrix = backup_vectorizer.fit_transform(valid_sentences)
        except Exception as e:
            self.logger.warning(f"Failed to vectorize sentences: {str(e)}")
            return {}

        # Use NetworkX to find clusters of similar sentences
        G = nx.Graph()

        # Add all sentence indices as nodes
        for i in range(len(valid_indices)):
            G.add_node(valid_indices[i])

        # Compute similarities and add edges for similar pairs
        batch_size = 100  # Process similarity in batches to save memory
        for i in range(0, len(valid_indices), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(valid_indices))))
            batch_tfidf = tfidf_matrix[batch_indices]

            # Calculate similarity between this batch and all sentences
            similarities = cosine_similarity(batch_tfidf, tfidf_matrix)

            # Add edges for similar sentence pairs
            for batch_idx, full_similarities in enumerate(similarities):
                src_idx = valid_indices[i + batch_idx]

                # Find indices with similarity above threshold
                # Avoid self-comparison and only process half the matrix
                for j, sim in enumerate(full_similarities):
                    if j > i + batch_idx and sim >= self.similarity_threshold:
                        tgt_idx = valid_indices[j]
                        G.add_edge(src_idx, tgt_idx, weight=sim)

        # Find connected components (clusters of similar sentences)
        clusters = list(nx.connected_components(G))

        # Create mapping from sentence index to cluster ID
        cluster_mapping = {}
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                cluster_mapping[idx] = cluster_id

        return cluster_mapping

    def _process_chunk(self, df_chunk):
        """
        Process a chunk of the DataFrame to find similar sentences.

        Args:
            df_chunk: DataFrame chunk to process

        Returns:
            Tuple of (sentence indices to keep, mapping of sentence index to cluster)
        """
        sentences = df_chunk["Segmented_Qualification"].tolist()
        cluster_mapping = self.find_similar_sentences(sentences)

        # Find the best representative for each cluster
        # (we'll keep the longest sentence in each cluster)
        sentences_to_keep = set()
        cluster_to_best_idx = {}

        for idx, cluster_id in cluster_mapping.items():
            real_idx = df_chunk.index[idx]
            if cluster_id not in cluster_to_best_idx:
                cluster_to_best_idx[cluster_id] = real_idx
                sentences_to_keep.add(real_idx)
            else:
                # Compare sentence lengths to find the best representative
                current_best = cluster_to_best_idx[cluster_id]
                current_sent = df_chunk.loc[current_best, "Segmented_Qualification"]
                new_sent = df_chunk.loc[real_idx, "Segmented_Qualification"]

                # Keep the longer, more informative sentence
                if len(new_sent) > len(current_sent):
                    sentences_to_keep.discard(current_best)
                    sentences_to_keep.add(real_idx)
                    cluster_to_best_idx[cluster_id] = real_idx

        # Add singleton sentences (those without similar matches)
        for i in range(len(df_chunk)):
            if i not in cluster_mapping:
                sentences_to_keep.add(df_chunk.index[i])

        return sentences_to_keep, cluster_mapping

    def remove_similar_sentences(self, df):
        """
        Remove similar sentences from the DataFrame, keeping the best representative from each cluster.

        Args:
            df: DataFrame containing sentences in 'Segmented_Qualification' column

        Returns:
            DataFrame with similar sentences removed
        """
        if len(df) <= 1:
            return df

        self.logger.info(
            f"Finding and removing similar sentences with threshold {self.similarity_threshold}..."
        )
        original_count = len(df)

        # Process in chunks to handle large datasets
        chunks = [
            df.iloc[i : i + self.chunk_size] for i in range(0, len(df), self.chunk_size)
        ]

        all_sentences_to_keep = set()
        total_clusters = 0

        # Process each chunk
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            sentences_to_keep, cluster_mapping = self._process_chunk(chunk)
            all_sentences_to_keep.update(sentences_to_keep)

            # Count clusters in this chunk
            if cluster_mapping:
                total_clusters += len(set(cluster_mapping.values()))

        # Keep only the selected sentences
        df_filtered = df.loc[list(all_sentences_to_keep)].reset_index(drop=True)

        removed_count = original_count - len(df_filtered)
        self.logger.info(
            f"Removed {removed_count} similar sentences ({removed_count / original_count:.2%})"
        )
        self.logger.info(f"Found {total_clusters} clusters of similar sentences")

        return df_filtered

    def remove_similar_sentences_parallel(self, df):
        """
        Remove similar sentences using parallel processing for larger datasets.

        Args:
            df: DataFrame containing sentences in 'Segmented_Qualification' column

        Returns:
            DataFrame with similar sentences removed
        """
        if len(df) <= self.chunk_size or self.n_jobs <= 1:
            return self.remove_similar_sentences(df)

        self.logger.info(
            f"Finding and removing similar sentences in parallel with {self.n_jobs} processes..."
        )
        original_count = len(df)

        # Split data into larger chunks for parallel processing
        parallel_chunk_size = max(self.chunk_size, len(df) // self.n_jobs)
        chunks = [
            df.iloc[i : i + parallel_chunk_size].copy()
            for i in range(0, len(df), parallel_chunk_size)
        ]

        # Process each large chunk in a separate process
        with mp.Pool(processes=self.n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(self.remove_similar_sentences, chunks),
                    total=len(chunks),
                    desc="Processing in parallel",
                )
            )

        # Combine results
        df_filtered = pd.concat(results, ignore_index=True)

        # Perform a final pass to catch similar sentences across chunk boundaries
        if len(chunks) > 1:
            self.logger.info(
                "Performing final pass to catch cross-boundary similarities..."
            )
            df_filtered = self.remove_similar_sentences(df_filtered)

        removed_count = original_count - len(df_filtered)
        self.logger.info(
            f"Removed {removed_count} similar sentences ({removed_count / original_count:.2%})"
        )

        return df_filtered


class SmartWordSeparator:
    """Smart word separator using statistical models with tech-aware whitelist."""

    def __init__(self, use_wordninja: bool = True):
        """
        Initialize with word segmentation library preference.

        Args:
            use_wordninja: Prefer wordninja over wordsegment if both available
        """
        self.segmentation_method = self._init_segmentation_method(use_wordninja)
        self.tech_whitelist = self._create_tech_whitelist()
        self.protected_patterns = self._create_protected_patterns()

        logger.info(
            f"SmartWordSeparator initialized with method: {self.segmentation_method}"
        )

    def _init_segmentation_method(self, prefer_wordninja: bool) -> str:
        """Initialize the best available segmentation method."""
        if prefer_wordninja and WORDNINJA_AVAILABLE:
            return "wordninja"
        elif WORDSEGMENT_AVAILABLE:
            return "wordsegment"
        elif WORDNINJA_AVAILABLE:
            return "wordninja"
        else:
            logger.warning(
                "No word segmentation library available. Install 'wordninja' or 'wordsegment'"
            )
            return "fallback"

    def _create_tech_whitelist(self) -> Set[str]:
        """Create comprehensive whitelist of tech terms that should not be segmented."""

        tech_terms = {
            # Programming Languages & Runtimes
            "javascript",
            "typescript",
            "coffeescript",
            "actionscript",
            "python",
            "cpython",
            "jython",
            "ironpython",
            "micropython",
            "java",
            "openjdk",
            "oraclejdk",
            "javase",
            "javaee",
            "javafx",
            "csharp",
            "fsharp",
            "visualbasic",
            "dotnet",
            "netcore",
            "netframework",
            "cplusplus",
            "objectivec",
            "swift",
            "swiftui",
            "uikit",
            "golang",
            "rustlang",
            "kotlinlang",
            "scala",
            "clojure",
            "groovy",
            "ruby",
            "rubyonrails",
            "rubygems",
            "bundler",
            "php",
            "hack",
            "hhvm",
            "composer",
            "perl",
            "python",
            "haskell",
            "erlang",
            "elixir",
            "matlab",
            "octave",
            "rlang",
            "julia",
            "powershell",
            "bash",
            "zsh",
            "fish",
            # JavaScript Frameworks & Libraries
            "react",
            "reactjs",
            "reactnative",
            "redux",
            "mobx",
            "recoil",
            "angular",
            "angularjs",
            "rxjs",
            "ngrx",
            "vue",
            "vuejs",
            "vuex",
            "nuxtjs",
            "vuepress",
            "svelte",
            "sveltekit",
            "ember",
            "emberjs",
            "backbone",
            "backbonejs",
            "marionette",
            "jquery",
            "jqueryui",
            "zepto",
            "lodash",
            "underscore",
            "ramda",
            "momentjs",
            "dayjs",
            "datejs",
            "d3js",
            "chartjs",
            "highcharts",
            "plotly",
            "threejs",
            "babylonjs",
            "aframe",
            # Node.js & Backend
            "nodejs",
            "nodemon",
            "npm",
            "yarn",
            "pnpm",
            "express",
            "expressjs",
            "fastify",
            "koajs",
            "nestjs",
            "nextjs",
            "gatsby",
            "gridsome",
            "vuepress",
            "webpack",
            "rollup",
            "parcel",
            "vite",
            "snowpack",
            "babel",
            "typescript",
            "eslint",
            "prettier",
            "husky",
            "jest",
            "mocha",
            "jasmine",
            "cypress",
            "playwright",
            "puppeteer",
            # Python Frameworks & Libraries
            "django",
            "djangorest",
            "flask",
            "fastapi",
            "tornado",
            "pyramid",
            "bottle",
            "cherrypy",
            "falcon",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "plotly",
            "scipy",
            "scikit",
            "scikitlearn",
            "tensorflow",
            "pytorch",
            "keras",
            "theano",
            "caffe",
            "mxnet",
            "jupyter",
            "ipython",
            "anaconda",
            "miniconda",
            "requests",
            "urllib",
            "httpx",
            "aiohttp",
            "sqlalchemy",
            "peewee",
            "tortoise",
            "celery",
            "redis",
            "rabbitmq",
            "pytest",
            "unittest",
            "nose",
            "tox",
            # Java Frameworks & Libraries
            "spring",
            "springboot",
            "springframework",
            "springmvc",
            "hibernate",
            "mybatis",
            "jpa",
            "jdbc",
            "junit",
            "testng",
            "mockito",
            "easymock",
            "maven",
            "gradle",
            "ant",
            "sbt",
            "tomcat",
            "jetty",
            "undertow",
            "netty",
            "jackson",
            "gson",
            "fastjson",
            "log4j",
            "logback",
            "slf4j",
            # Databases
            "mysql",
            "mariadb",
            "percona",
            "postgresql",
            "postgis",
            "timescaledb",
            "mongodb",
            "mongoose",
            "mongoengine",
            "redis",
            "memcached",
            "hazelcast",
            "elasticsearch",
            "opensearch",
            "solr",
            "lucene",
            "cassandra",
            "scylladb",
            "datastax",
            "dynamodb",
            "cosmosdb",
            "firestore",
            "firebase",
            "neo4j",
            "arangodb",
            "orientdb",
            "influxdb",
            "prometheus",
            "grafana",
            "clickhouse",
            "bigquery",
            "snowflake",
            "redshift",
            "sqlite",
            "leveldb",
            "rocksdb",
            "oracle",
            "oracledb",
            "sqlserver",
            "mssql",
            # Cloud Platforms & Services
            "aws",
            "amazonwebservices",
            "awslambda",
            "awsec2",
            "awss3",
            "azure",
            "azuredevops",
            "azuread",
            "azuresql",
            "gcp",
            "googlecloud",
            "googlecloudplatform",
            "firebase",
            "heroku",
            "netlify",
            "vercel",
            "digitalocean",
            "linode",
            "vultr",
            "hetzner",
            "ovh",
            "cloudflare",
            "fastly",
            "maxcdn",
            # DevOps & Infrastructure
            "docker",
            "dockerfile",
            "dockercompose",
            "podman",
            "kubernetes",
            "k8s",
            "kubectl",
            "helm",
            "istio",
            "terraform",
            "terragrunt",
            "pulumi",
            "cloudformation",
            "ansible",
            "puppet",
            "chef",
            "saltstack",
            "vagrant",
            "packer",
            "consul",
            "vault",
            "jenkins",
            "jenkinsx",
            "bamboo",
            "teamcity",
            "gitlab",
            "gitlabci",
            "github",
            "githubactions",
            "travis",
            "travisci",
            "circleci",
            "appveyor",
            "nginx",
            "apache",
            "httpd",
            "haproxy",
            "envoy",
            "prometheus",
            "grafana",
            "jaeger",
            "zipkin",
            "fluentd",
            "logstash",
            "filebeat",
            "metricbeat",
            # Mobile Development
            "android",
            "androidx",
            "androidstudio",
            "ios",
            "xcode",
            "swift",
            "swiftui",
            "objectivec",
            "flutter",
            "dart",
            "flutterflow",
            "reactnative",
            "expo",
            "metro",
            "ionic",
            "cordova",
            "phonegap",
            "capacitor",
            "xamarin",
            "nativescript",
            "titanium",
            "unity",
            "unreal",
            "godot",
            "cocos2d",
            # Frontend & CSS
            "html",
            "html5",
            "xhtml",
            "xml",
            "svg",
            "css",
            "css3",
            "sass",
            "scss",
            "less",
            "stylus",
            "postcss",
            "autoprefixer",
            "cssnano",
            "bootstrap",
            "tailwindcss",
            "bulma",
            "foundation",
            "materialui",
            "antdesign",
            "chakraui",
            "semanticui",
            "styledcomponents",
            "emotion",
            "stitches",
            # Version Control & Collaboration
            "git",
            "github",
            "gitlab",
            "bitbucket",
            "sourceforge",
            "svn",
            "subversion",
            "mercurial",
            "bazaar",
            "gitflow",
            "githubflow",
            "gitlabflow",
            "slack",
            "discord",
            "teams",
            "zoom",
            "jira",
            "confluence",
            "notion",
            "miro",
            "figma",
            "sketch",
            # Data & Analytics
            "apache",
            "spark",
            "hadoop",
            "hdfs",
            "yarn",
            "kafka",
            "pulsar",
            "nats",
            "rabbitmq",
            "airflow",
            "luigi",
            "prefect",
            "dagster",
            "dbt",
            "fivetran",
            "stitch",
            "airbyte",
            "tableau",
            "powerbi",
            "looker",
            "metabase",
            "superset",
            "grafana",
            "kibana",
            "datadog",
            "snowflake",
            "databricks",
            "palantir",
            # Security & Authentication
            "oauth",
            "oauth2",
            "openid",
            "saml",
            "ldap",
            "jwt",
            "json",
            "jsonwebtoken",
            "passport",
            "auth0",
            "okta",
            "keycloak",
            "firebase",
            "ssl",
            "tls",
            "https",
            "certificates",
            "vault",
            "secrets",
            "kubernetes",
            "helm",
            # API & Communication
            "rest",
            "restful",
            "restapi",
            "graphql",
            "grpc",
            "protobuf",
            "thrift",
            "avro",
            "soap",
            "wsdl",
            "xml",
            "json",
            "yaml",
            "websocket",
            "socketio",
            "webrtc",
            "sse",
            "swagger",
            "openapi",
            "postman",
            "insomnia",
            # Operating Systems & Platforms
            "linux",
            "ubuntu",
            "debian",
            "centos",
            "redhat",
            "fedora",
            "opensuse",
            "archlinux",
            "gentoo",
            "alpine",
            "busybox",
            "scratch",
            "windows",
            "windowsserver",
            "powershell",
            "macos",
            "osx",
            "homebrew",
            "macports",
            "freebsd",
            "openbsd",
            "netbsd",
            "solaris",
            # Protocols & Standards
            "http",
            "https",
            "http2",
            "http3",
            "quic",
            "tcp",
            "udp",
            "ip",
            "ipv4",
            "ipv6",
            "dns",
            "dhcp",
            "ntp",
            "snmp",
            "ssh",
            "ftp",
            "sftp",
            "smtp",
            "pop3",
            "imap",
            "mime",
            "cors",
            "csrf",
            "xss",
            "owasp",
            # Design & UX Tools
            "figma",
            "sketch",
            "adobexd",
            "invision",
            "zeplin",
            "photoshop",
            "illustrator",
            "aftereffects",
            "premiere",
            "blender",
            "maya",
            "cinema4d",
            "unity",
            "unreal",
            "framer",
            "principle",
            "protopie",
            "marvel",
            # Certifications & Methodologies
            "scrum",
            "agile",
            "kanban",
            "devops",
            "sre",
            "itil",
            "prince2",
            "pmp",
            "csm",
            "psm",
            "aws",
            "azure",
            "gcp",
            "cissp",
            "cisa",
            "cism",
            # File Formats & Extensions
            "json",
            "xml",
            "yaml",
            "toml",
            "ini",
            "csv",
            "tsv",
            "pdf",
            "docx",
            "xlsx",
            "pptx",
            "odt",
            "ods",
            "odp",
            "jpg",
            "jpeg",
            "png",
            "gif",
            "svg",
            "webp",
            "avif",
            "mp4",
            "webm",
            "avi",
            "mov",
            "wmv",
            "flv",
            "mp3",
            "wav",
            "flac",
            "ogg",
            "aac",
            "m4a",
            # Version Numbers & Patterns
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "v7",
            "v8",
            "v9",
            "v10",
            "es6",
            "es2015",
            "es2016",
            "es2017",
            "es2018",
            "es2019",
            "es2020",
            "python2",
            "python3",
            "node12",
            "node14",
            "node16",
            "node18",
            "java8",
            "java11",
            "java17",
            "java19",
            # Common Compound Terms
            "fullstack",
            "frontend",
            "backend",
            "devops",
            "sre",
            "microservices",
            "serverless",
            "cloudnative",
            "containerization",
            "machinelearning",
            "artificialintelligence",
            "deeplearning",
            "blockchain",
            "cryptocurrency",
            "fintech",
            "healthtech",
            "edtech",
            "adtech",
            "martech",
            "proptech",
            "iot",
            "ar",
            "vr",
            "xr",
            "metaverse",
            "webrtc",
            "websocket",
            "sse",
            "pwa",
            "spa",
            "ssr",
            "ssg",
            "jamstack",
            "headless",
            "cms",
            "ecommerce",
            "saas",
            "paas",
            "iaas",
            "faas",
            "baas",
            "cicd",
            "gitops",
            "infrastructure",
            "monitoring",
            "observability",
            "telemetry",
            "logging",
            "tracing",
        }

        # Add variations (uppercase, mixed case)
        extended_terms = set()
        for term in tech_terms:
            extended_terms.add(term.lower())
            extended_terms.add(term.upper())
            extended_terms.add(term.title())
            extended_terms.add(term.capitalize())

        return extended_terms

    def _create_protected_patterns(self) -> List[str]:
        """Create regex patterns for terms that should be protected from segmentation."""
        return [
            # Version patterns
            r"\b[a-zA-Z]+\d+(\.\d+)*\b",  # e.g., Python3.9, Node16.14
            r"\bv\d+(\.\d+)*\b",  # e.g., v1.0, v2.3.1
            r"\bes\d+\b",  # e.g., ES6, ES2020
            # Tech with numbers
            r"\b[a-zA-Z]+\d+[a-zA-Z]*\b",  # e.g., HTML5, CSS3, HTTP2
            # Camel/Pascal case tech terms
            r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b",  # e.g., JavaScript, TypeScript
            # Acronyms
            r"\b[A-Z]{2,}\b",  # e.g., API, REST, JWT, SQL
            # Domain extensions and URLs
            r"\b\w+\.(com|org|net|io|dev|tech|app)\b",
            # File extensions
            r"\b\w+\.(js|ts|py|java|cpp|cs|rb|php|go|rs)\b",
        ]

    def _is_protected_term(self, word: str) -> bool:
        """Check if a word should be protected from segmentation."""
        word_lower = word.lower()

        # Check whitelist
        if word_lower in self.tech_whitelist:
            return True

        # Check protected patterns
        for pattern in self.protected_patterns:
            if re.match(pattern, word, re.IGNORECASE):
                return True

        return False

    def _segment_word(self, word: str) -> List[str]:
        """Segment a single word using the available method."""
        if self.segmentation_method == "wordninja":
            return wordninja.split(word)
        elif self.segmentation_method == "wordsegment":
            return wordsegment.segment(word)
        else:
            # Fallback: simple heuristic segmentation
            return self._fallback_segment(word)

    def _fallback_segment(self, word: str) -> List[str]:
        """Fallback segmentation using simple heuristics."""
        # Basic camelCase/PascalCase splitting
        segments = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|$)|\d+", word)
        return segments if segments else [word]

    def separate_concatenated_words(self, text: str) -> str:
        """
        Separate concatenated words while preserving tech terms.

        Args:
            text: Input text with potentially concatenated words

        Returns:
            Text with intelligently separated words
        """
        if not isinstance(text, str) or not text.strip():
            return text

        # Tokenize while preserving spaces and punctuation
        tokens = re.findall(r"\S+|\s+", text)
        result_tokens = []

        for token in tokens:
            # Skip whitespace and punctuation-only tokens
            if re.match(r"\s+$", token) or re.match(r"^[^\w]+$", token):
                result_tokens.append(token)
                continue

            # Clean token for processing (remove leading/trailing punctuation)
            clean_token = re.sub(r"^[^\w]+|[^\w]+$", "", token)
            prefix = token[
                : len(token) - len(token.lstrip("".join(re.findall(r"^[^\w]+", token))))
            ]
            suffix = token[
                len(token) - len(token.rstrip("".join(re.findall(r"[^\w]+$", token)))) :
            ]

            if not clean_token:
                result_tokens.append(token)
                continue

            # Check if the clean token should be protected
            if self._is_protected_term(clean_token):
                result_tokens.append(token)
                continue

            # Skip if token is too short or already well-separated
            if len(clean_token) <= 4:
                result_tokens.append(token)
                continue

            # Segment the clean token
            try:
                segments = self._segment_word(clean_token)

                # Filter out single characters and very short segments
                # except for meaningful ones
                meaningful_segments = []
                for seg in segments:
                    if len(seg) >= 2 or seg.lower() in {"i", "a", "x", "y", "z"}:
                        meaningful_segments.append(seg)

                if len(meaningful_segments) > 1:
                    # Reconstruct with spaces, preserving original prefix/suffix
                    segmented = prefix + " ".join(meaningful_segments) + suffix
                    result_tokens.append(segmented)
                else:
                    result_tokens.append(token)

            except Exception as e:
                logger.debug(f"Error segmenting '{clean_token}': {e}")
                result_tokens.append(token)

        # Join tokens and clean up excessive spaces
        result = "".join(result_tokens)
        result = re.sub(r"\s+", " ", result).strip()

        return result


class EnhancedDataCleaning:
    """Enhanced DataCleaning class with smart word separation."""

    def __init__(self, config=None):
        """
        Initialize with configuration.

        Args:
            config: Optional configuration dictionary
        """
        # Default configuration
        self.config = {
            "remove_emails": True,
            "remove_phones": True,
            "remove_urls": True,
            "remove_html": True,
            "remove_css": True,
            "normalize_whitespace": True,
            "remove_special_chars": True,
            "min_text_length": 10,
            "remove_stopwords": False,
            "lowercase": False,
            "separate_concatenated_words": True,
            "use_wordninja": True,  # Prefer wordninja over wordsegment
        }

        # Override with provided config
        if config:
            self.config.update(config)

        # Initialize word separator if enabled
        self.word_separator = None
        if self.config["separate_concatenated_words"]:
            self.word_separator = SmartWordSeparator(
                use_wordninja=self.config["use_wordninja"]
            )

        # Compile regex patterns for better performance
        self.patterns = {
            "html_tags": re.compile(r"<[^>]+>"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}"),
            "css": re.compile(r"<style.*?>.*?</style>", re.DOTALL),
            "url": re.compile(r"(http|https)://[^\s]+|www\.[^\s]+"),
            "css_inline": re.compile(r"\.[a-zA-Z0-9_-]+\s*\{.*?\}", re.DOTALL),
            "bullet_point": re.compile(r"[-•*]"),
            "non_alphanumeric": re.compile(r"[^a-zA-Z0-9.'\"#\+\-\._, /\\&]+"),
            "hashtag": re.compile(r"#\w+"),
            "excessive_whitespace": re.compile(r"\s+"),
        }

    def clean_text(self, text):
        """
        Enhanced text cleaning with smart word separation.

        Args:
            text: Input text

        Returns:
            Cleaned text with separated words
        """
        if not isinstance(text, str):
            return ""

        # Apply word separation first (before other cleaning that might affect segmentation)
        if self.config["separate_concatenated_words"] and self.word_separator:
            text = self.word_separator.separate_concatenated_words(text)

        # Apply cleaning operations
        if self.config["remove_css"]:
            text = self.patterns["css"].sub(" ", text)
            text = self.patterns["css_inline"].sub(" ", text)

        if self.config["remove_html"]:
            text = self.patterns["html_tags"].sub(" ", text)

        if self.config["remove_emails"]:
            text = self.patterns["email"].sub(" ", text)

        if self.config["remove_phones"]:
            text = self.patterns["phone"].sub(" ", text)

        if self.config["remove_urls"]:
            text = self.patterns["url"].sub(" ", text)

        # Replace bullet points with space
        text = self.patterns["bullet_point"].sub(" ", text)

        # Remove hashtags
        text = self.patterns["hashtag"].sub(" ", text)

        if self.config["remove_special_chars"]:
            text = self.patterns["non_alphanumeric"].sub(" ", text)

        if self.config["normalize_whitespace"]:
            text = self.patterns["excessive_whitespace"].sub(" ", text).strip()

        if self.config["lowercase"]:
            text = text.lower()

        return text

    def filter_non_thai(self, text: Optional[str]) -> Optional[str]:
        """
        Keep only text that doesn't contain Thai characters.

        Args:
            text: Input text

        Returns:
            Text if no Thai characters present, None otherwise
        """
        if not isinstance(text, str):
            return None

        # Check for Thai Unicode range
        if re.search(r"[\u0E00-\u0E7F]", text):
            return None
        return text


class DataQualityCheck:
    """Class for data quality checking."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with DataFrame.

        Args:
            df: DataFrame to check
        """
        self.df = df
        self.quality_stats = {}

    def check_data_shape(self) -> Dict:
        """
        Check shape of the DataFrame.

        Returns:
            Dictionary with shape information
        """
        rows, cols = self.df.shape
        logger.info(f"Data Shape: {rows} rows, {cols} columns")
        return {"rows": rows, "columns": cols}

    def check_missing_values(self) -> Dict:
        """
        Check missing values in the DataFrame.

        Returns:
            Dictionary with missing value counts by column
        """
        missing_values = self.df.isnull().sum().to_dict()
        total_missing = sum(missing_values.values())
        logger.info(f"Total missing values: {total_missing}")
        for col, count in missing_values.items():
            if count > 0:
                pct = (count / len(self.df)) * 100
                logger.info(f"- {col}: {count} missing values ({pct:.2f}%)")
        return {"missing_counts": missing_values, "total_missing": total_missing}

    def check_duplicates(self) -> Dict:
        """
        Check duplicates in the DataFrame.

        Returns:
            Dictionary with duplicate information
        """
        duplicate_count = self.df.duplicated().sum()
        duplicate_pct = (
            (duplicate_count / len(self.df)) * 100 if len(self.df) > 0 else 0
        )
        logger.info(f"Duplicated rows: {duplicate_count} ({duplicate_pct:.2f}%)")

        # Check duplicates by column
        column_duplicates = {}
        for col in self.df.columns:
            if self.df[col].dtype == "object":  # Only check object/string columns
                dup_count = self.df.duplicated(subset=[col]).sum()
                if dup_count > 0:
                    dup_pct = (dup_count / len(self.df)) * 100
                    column_duplicates[col] = {
                        "count": int(dup_count),
                        "percentage": float(f"{dup_pct:.2f}"),
                    }
                    logger.info(
                        f"- Column '{col}' has {dup_count} duplicate values ({dup_pct:.2f}%)"
                    )

        return {
            "total_duplicates": int(duplicate_count),
            "duplicate_percentage": float(f"{duplicate_pct:.2f}"),
            "column_duplicates": column_duplicates,
        }

    def check_text_quality(self, text_column: str) -> Dict:
        """
        Check text quality for a specific column.

        Args:
            text_column: Column name containing text

        Returns:
            Dictionary with text quality metrics
        """
        if text_column not in self.df.columns:
            logger.warning(f"Column {text_column} not found in DataFrame")
            return {}

        # Get text length statistics
        self.df["text_length"] = self.df[text_column].fillna("").astype(str).apply(len)
        length_stats = self.df["text_length"].describe().to_dict()

        # Convert numpy values to Python native types for JSON serialization
        length_stats = {k: float(v) for k, v in length_stats.items()}

        # Count empty strings
        empty_count = (self.df[text_column].fillna("") == "").sum()
        empty_pct = (empty_count / len(self.df)) * 100

        # Count very short texts (less than 10 chars)
        short_count = (
            (self.df["text_length"] > 0) & (self.df["text_length"] < 10)
        ).sum()
        short_pct = (short_count / len(self.df)) * 100

        logger.info(f"Text quality for column '{text_column}':")
        logger.info(f"- Empty strings: {empty_count} ({empty_pct:.2f}%)")
        logger.info(f"- Very short texts (<10 chars): {short_count} ({short_pct:.2f}%)")
        logger.info(
            f"- Length statistics: min={length_stats['min']}, max={length_stats['max']}, avg={length_stats['mean']:.2f}"
        )

        # Drop temporary column
        self.df.drop("text_length", axis=1, inplace=True)

        return {
            "empty_strings": int(empty_count),
            "empty_percentage": float(f"{empty_pct:.2f}"),
            "short_texts": int(short_count),
            "short_percentage": float(f"{short_pct:.2f}"),
            "length_stats": length_stats,
        }

    def run_checks(self, text_columns: Optional[List[str]] = None) -> Dict:
        """
        Run all data quality checks.

        Args:
            text_columns: Optional list of columns to check for text quality

        Returns:
            Dictionary with all quality check results
        """
        logger.info("Running data quality checks...")

        self.quality_stats = {
            "shape": self.check_data_shape(),
            "missing_values": self.check_missing_values(),
            "duplicates": self.check_duplicates(),
            "text_quality": {},
        }

        if text_columns:
            for col in text_columns:
                self.quality_stats["text_quality"][col] = self.check_text_quality(col)

        return self.quality_stats

    def save_report(self, output_path: Union[str, Path]) -> None:
        """
        Save quality report to JSON file.

        Args:
            output_path: Path to save the report
        """
        if not self.quality_stats:
            logger.warning("No quality checks have been run yet")
            return

        # Ensure directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.quality_stats, f, indent=2)

        logger.info(f"Quality report saved to {output_path}")


class JobDataProcessor:
    """Main class for processing job data."""

    def __init__(
        self,
        raw_data_path: Union[str, Path],
        interim_data_path: Union[str, Path],
        config: Optional[Dict] = None,
    ):
        """
        Initialize with paths and configuration.

        Args:
            raw_data_path: Path to raw data
            interim_data_path: Path to save processed data
            config: Optional configuration dictionary
        """
        self.raw_data_path = Path(raw_data_path)
        self.interim_data_path = Path(interim_data_path)

        # Default configuration
        self.config = {
            "input_file": "training-data/job_title_des.csv",
            "output_file": "segmented_data.csv",
            "quality_report_file": "data_quality_report.json",
            "separate_concatenated_words": True,
            "sample_fraction": 0.2,  # Use 20% of data by default
            "random_seed": 42,
            "min_sentence_length": 5,  # Minimum characters for a valid sentence
            "max_tokens_per_sentence": 100,  # For limiting excessive text
            "parallelize": True if mp.cpu_count() > 2 else False,
            "batch_size": 1000,
            "save_intermediate": True,
            "keep_original_columns": False,  # Whether to keep all original columns
            "similarity_threshold": 0.85,  # Threshold for SimilarSentenceRemover
        }

        # Override with provided config
        if config:
            self.config.update(config)

        # Initialize components
        self.cleaner = EnhancedDataCleaning(config)
        self.segmenter = DataSegmentation(batch_size=self.config["batch_size"])
        self.similar_remover = SimilarSentenceRemover(
            similarity_threshold=self.config["similarity_threshold"],
            chunk_size=self.config["batch_size"],
            n_jobs=mp.cpu_count() - 1 if self.config["parallelize"] else 1,
        )

        # Create directories if needed
        self.interim_data_path.mkdir(parents=True, exist_ok=True)

        # Log configuration
        logger.info(f"JobDataProcessor initialized with config: {self.config}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the input file.

        Returns:
            Loaded DataFrame
        """
        input_path = self.raw_data_path / self.config["input_file"]
        print("input_path", input_path)
        logger.info(f"Loading data from {input_path}")

        try:
            df = pd.read_csv(input_path)
            logger.info(f"Loaded data with shape: {df.shape}")

            # Take a sample if configured
            if self.config["sample_fraction"] < 1.0:
                original_size = len(df)
                df = df.sample(
                    frac=self.config["sample_fraction"],
                    random_state=self.config["random_seed"],
                ).reset_index(drop=True)
                logger.info(f"Sampled {len(df)} records from {original_size} total")

            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def process(self) -> pd.DataFrame:
        """
        Process the data: filtering, cleaning, segmenting, removing similar sentences, and checking quality.

        Returns:
            Processed DataFrame
        """
        # Load data
        df_full = self.load_data()

        # Drop duplicate qualifications
        original_count = len(df_full)
        df_full.drop_duplicates(subset=["Qualification"], inplace=True)
        logger.info(f"Removed {original_count - len(df_full)} duplicate qualifications")

        # Filter non-Thai topics and qualifications
        logger.info("Filtering non-Thai content...")
        with tqdm(total=len(df_full)) as pbar:
            if self.config["parallelize"]:
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    df_full["Topic"] = pool.map(
                        self.cleaner.filter_non_thai, df_full["Topic"].tolist()
                    )
                    pbar.update(len(df_full) // 2)
                    df_full["Qualification"] = pool.map(
                        self.cleaner.filter_non_thai, df_full["Qualification"].tolist()
                    )
                    pbar.update(len(df_full) // 2)
            else:
                df_full["Topic"] = df_full["Topic"].progress_apply(
                    self.cleaner.filter_non_thai
                )
                df_full["Qualification"] = df_full["Qualification"].progress_apply(
                    self.cleaner.filter_non_thai
                )

        # Drop rows with Thai content
        before_drop = len(df_full)
        df_full.dropna(subset=["Topic", "Qualification"], inplace=True)
        logger.info(f"Removed {before_drop - len(df_full)} rows with Thai content")

        # Clean qualification text
        logger.info("Cleaning qualification text...")
        if self.config["parallelize"]:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                df_full["Qualification"] = pool.map(
                    self.cleaner.clean_text, df_full["Qualification"].tolist()
                )
        else:
            df_full["Qualification"] = df_full["Qualification"].progress_apply(
                self.cleaner.clean_text
            )

        # Save intermediate DataFrame if configured
        if self.config["save_intermediate"]:
            interim_path = self.interim_data_path / "cleaned_data.csv"
            df_full.to_csv(interim_path, index=False)
            logger.info(f"Saved cleaned data to {interim_path}")

        # Segment qualification text into sentences
        logger.info("Segmenting qualification text into sentences...")
        df_full["Sentence_Index"] = df_full.index

        if self.config["parallelize"]:
            # Process in batches for better memory management
            df_full["Segmented_Qualification"] = self.segmenter.split_sentences_batch(
                df_full["Qualification"].tolist()
            )
        else:
            df_full["Segmented_Qualification"] = df_full[
                "Qualification"
            ].progress_apply(self.segmenter.split_sentences)

        # Explode the list of sentences to separate rows
        logger.info("Exploding sentences to individual rows...")
        df = df_full.explode("Segmented_Qualification").reset_index(drop=True)

        # Filter out None, empty sentences and sentences that are too short
        before_filter = len(df)
        df = df[df["Segmented_Qualification"].notna()]
        df = df[df["Segmented_Qualification"].str.strip().astype(bool)]
        df = df[
            df["Segmented_Qualification"].str.len()
            >= self.config["min_sentence_length"]
        ]
        logger.info(f"Removed {before_filter - len(df)} invalid or too short sentences")

        # # Combine Topic and Segmented_Qualification
        # logger.info("Combining Topic with Segmented_Qualification...")
        # df["Segmented_Qualification"] = df.apply(
        #     lambda row: f"[{row['Topic']}] {row['Segmented_Qualification']}",
        #     axis=1,
        # )

        # Remove duplicate segmented qualifications
        before_dedup = len(df)
        df.drop_duplicates(subset=["Segmented_Qualification"], inplace=True)
        logger.info(
            f"Removed {before_dedup - len(df)} duplicate segmented qualifications"
        )

        # Remove similar sentences
        logger.info("Removing similar sentences...")
        df = self.similar_remover.remove_similar_sentences_parallel(df)

        # Run data quality checks
        data_quality = DataQualityCheck(df)
        quality_stats = data_quality.run_checks(
            text_columns=["Segmented_Qualification"]
        )

        # Save quality report
        quality_report_path = (
            self.interim_data_path / self.config["quality_report_file"]
        )
        data_quality.save_report(quality_report_path)

        # Save only Sentence_Index and Segmented_Qualification
        df = df[["Topic", "Position", "Sentence_Index", "Segmented_Qualification"]]

        # Save processed data
        output_path = self.interim_data_path / self.config["output_file"]
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

        return df


if __name__ == "__main__":
    config = {
        "input_file": "classified/classified_jobs.csv",
        "output_file": "segmented-data/scraping-segmented-data.csv",
        "quality_report_file": "data_quality_report.json",
        "separate_concatenated_words": False,
        "use_wordninja": True,
        "sample_fraction": 1,
        "min_sentence_length": 10,
        "parallelize": True,
        "batch_size": 1000,
        "save_intermediate": True,
        "keep_original_columns": False,
        "similarity_threshold": 1.00,
    }

    processor = JobDataProcessor(RAW_DATA_PATH, INTERIM_DATA_PATH, config)
    df = processor.process()

    # Print summary statistics
    print("\nProcessing Complete!")
    print(f"Final dataset has {len(df)} rows with segmented qualifications")
