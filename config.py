"""Configuration module for AI Terminal Assistant.

Defines directory paths, application settings, NLP thresholds, and execution safety rules.
"""

from pathlib import Path
import os
from typing import List
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CORE_DIR = BASE_DIR / "core"
UTILS_DIR = BASE_DIR / "utils"
SCRIPTS_DIR = BASE_DIR / "scripts"
TESTS_DIR = BASE_DIR / "tests"
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", str(BASE_DIR / "workspace"))).resolve()

# Specific file paths
INTENTS_DATASET_PATH = DATA_DIR / "intents_dataset.csv"
COMMAND_HISTORY_PATH = DATA_DIR / "command_history.json"
NLTK_DATA_DIR = DATA_DIR / "nltk_data"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation_report.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
REFERENCE_EMBEDDINGS_PATH = MODELS_DIR / "reference_embeddings.joblib"
SEMANTIC_REPORT_PATH = MODELS_DIR / "semantic_evaluation_report.json"
SEMANTIC_COMPARISON_PATH = MODELS_DIR / "semantic_comparison_report.json"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Register local NLTK data path if available
import nltk
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

# Load environment variables if .env exists
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# Application & NLP Settings
APP_ENV: str = os.getenv("APP_ENV", "development")
DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Intent Classification Thresholds
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
SEMANTIC_SIMILARITY_THRESHOLD: float = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.55"))
SENTENCE_TRANSFORMER_MODEL: str = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
USE_SEMANTIC_FALLBACK: bool = os.getenv("USE_SEMANTIC_FALLBACK", "True").lower() in ("true", "1", "yes")

# Execution & Safety
COMMAND_TIMEOUT_SECONDS: int = int(os.getenv("COMMAND_TIMEOUT_SECONDS", "15"))
REQUIRE_CONFIRMATION: bool = os.getenv("REQUIRE_CONFIRMATION", "True").lower() in ("true", "1", "yes")
SAFE_MODE: bool = os.getenv("SAFE_MODE", "True").lower() in ("true", "1", "yes")

# Protected Windows system paths (Blocked from modification/deletion)
PROTECTED_SYSTEM_DIRS: List[Path] = [
    Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve(),
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")).resolve(),
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")).resolve(),
    Path(os.environ.get("ProgramData", r"C:\ProgramData")).resolve(),
]

PROTECTED_ROOT_DIRS: List[Path] = [
    Path("C:/").resolve(),
    Path("D:/").resolve(),
    Path.home().resolve(),
    Path.home().parent.resolve() if Path.home().parent else Path(r"C:\Users").resolve(),
]

PROTECTED_SYSTEM_PATHS: List[Path] = PROTECTED_SYSTEM_DIRS + PROTECTED_ROOT_DIRS

# Supported Intent Categories
SUPPORTED_INTENTS: List[str] = [
    "LIST_FILES",
    "CREATE_DIRECTORY",
    "DELETE_FILE",
    "SEARCH_FILES",
    "CHECK_VERSION",
    "CURRENT_DIRECTORY",
    "MOVE_FILE",
    "COPY_FILE",
    "SYSTEM_INFO",
    "NETWORK_INFO",
    "CREATE_FILE",
    "UNKNOWN",
]
