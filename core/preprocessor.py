"""Text Preprocessor module for AI Terminal Assistant.

Handles text normalization, tokenization, stopword removal, lemmatization,
and preservation of filenames, extensions, numbers, Windows paths, and action words.
"""

import re
import logging
from pathlib import Path
from typing import List, Set, Optional, Tuple

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import config

logger = logging.getLogger(__name__)

# Register local project NLTK data directory if it exists
if hasattr(config, "NLTK_DATA_DIR") and config.NLTK_DATA_DIR.exists():
    if str(config.NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(config.NLTK_DATA_DIR))


def check_nltk_resources() -> Tuple[bool, List[str]]:
    """Checks whether all required NLTK resources are available.

    Returns:
        (all_available, list_of_missing_resources)
    """
    required_resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    missing = []
    for path_key, package_name in required_resources:
        try:
            nltk.data.find(path_key)
        except LookupError:
            missing.append(package_name)
    return len(missing) == 0, missing


def ensure_nltk_resources(download_if_missing: bool = False) -> None:
    """Ensures required NLTK resources exist, raising a descriptive error or downloading.

    Args:
        download_if_missing: If True, attempts to download missing resources.
    """
    all_available, missing = check_nltk_resources()
    if not all_available:
        if download_if_missing:
            logger.info("Downloading missing NLTK resources: %s", missing)
            for pkg in missing:
                nltk.download(pkg, quiet=True)
        else:
            missing_str = " ".join(missing)
            raise LookupError(
                f"Missing required NLTK resource(s): {missing}. "
                f"Please run: python -m nltk.downloader {missing_str}"
            )


class TextPreprocessor:
    """Preprocesses natural language queries for intent classification and entity parsing.

    Features:
    - Normalizes lowercase and repeated whitespace.
    - Preserves filenames (e.g., 'notes.txt'), extensions (e.g., '.pdf'),
      Windows absolute paths ('C:\\Projects'), relative paths ('.\\subfolder'),
      and numbers ('2026').
    - Preserves negations (not, don't, never, etc.) and action verbs (delete, move, etc.).
    - Robust lemmatization using WordNetLemmatizer.
    """

    # Comprehensive regex capturing contractions, Windows paths, relative paths, filenames, extensions, words, and symbols
    TOKEN_PATTERN = (
        r"\w+['’]\w+|"                    # Contractions: don't, can't, won't
        r"[a-zA-Z]:\\[^\s\"'<>|]+|"       # Windows absolute path: C:\dir\file.txt
        r"\.{1,2}[/\\][^\s\"'<>|]+|"      # Relative path: .\subfolder or ../dir
        r"[\w.-]+\.[a-zA-Z0-9]+|"         # Filenames: file.txt, data_v1.0.csv
        r"\.[a-zA-Z0-9]+|"                # Extensions: .py, .pdf, .json
        r"\w+|"                           # Words, numbers, identifiers
        r"[^\w\s]"                        # Standalone punctuation
    )

    NEGATIONS: Set[str] = {
        "not",
        "no",
        "nor",
        "dont",
        "don't",
        "never",
        "without",
        "neither",
        "cannot",
        "cant",
        "can't",
    }

    ACTION_WORDS: Set[str] = {
        "show",
        "list",
        "make",
        "create",
        "delete",
        "remove",
        "erase",
        "trash",
        "purge",
        "find",
        "search",
        "locate",
        "check",
        "get",
        "move",
        "transfer",
        "shift",
        "relocate",
        "copy",
        "duplicate",
        "clone",
        "where",
        "who",
        "what",
        "which",
        "how",
        "to",
        "from",
        "into",
    }

    def __init__(self) -> None:
        """Initializes tokenizer, stopword sets, and lemmatizer with resource checking."""
        self.tokenizer = RegexpTokenizer(self.TOKEN_PATTERN)

        # Ensure resources are available
        all_available, missing = check_nltk_resources()
        if not all_available:
            # Fallback check before raising
            try:
                self.stopwords = set(stopwords.words("english"))
            except Exception:
                raise LookupError(
                    f"Missing NLTK resources: {missing}. Run: python -m nltk.downloader {' '.join(missing)}"
                )
        else:
            self.stopwords = set(stopwords.words("english"))

        # Remove negations and crucial actions from stopwords so they are preserved
        self.effective_stopwords = self.stopwords - self.NEGATIONS - self.ACTION_WORDS

        try:
            self.lemmatizer = WordNetLemmatizer()
            # Test wordnet load
            _ = self.lemmatizer.lemmatize("files")
        except Exception as e:
            raise LookupError(
                f"Failed to initialize WordNetLemmatizer: {e}. Run: python -m nltk.downloader wordnet omw-1.4"
            )

    def normalize_text(self, text: Optional[str]) -> str:
        """Normalizes casing, repeated spaces, and extraneous punctuation.

        Preserves internal punctuation in filenames (notes.txt), extensions (.py),
        and paths (C:\\Users).

        Args:
            text: Raw input string.

        Returns:
            Normalized lowercase text string with trimmed whitespace.
        """
        if text is None:
            return ""

        # Normalize whitespace (tabs, newlines, multiple spaces)
        cleaned = re.sub(r"\s+", " ", text.strip())
        if not cleaned:
            return ""

        # Lowercase conversion
        cleaned = cleaned.lower()

        # Strip enclosing single/double quotes around user targets e.g. "my folder" -> my folder
        # while keeping valid quote characters inside tokens if needed
        return cleaned

    def tokenize(self, text: Optional[str]) -> List[str]:
        """Tokenizes text using NLTK RegexpTokenizer preserving paths, extensions, and filenames.

        Args:
            text: Input string.

        Returns:
            List of token strings.
        """
        if not text:
            return []

        tokens = self.tokenizer.tokenize(text)
        return tokens

    @staticmethod
    def is_protected_token(token: str) -> bool:
        """Determines if a token is a filename, extension, path, or number that must not be stripped."""
        if not token:
            return False
        # Windows path e.g. C:\... or relative path .\...
        if re.match(r"^[a-zA-Z]:\\", token) or token.startswith((".\\", "./", "..\\", "../")):
            return True
        # Filename with extension or extension itself e.g. notes.txt, .pdf
        if "." in token and len(token) > 1 and not re.match(r"^\.+$", token):
            return True
        # Numbers e.g. 2026, 45, 10
        if token.isdigit():
            return True
        return False

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Removes non-essential stopwords while preserving negations, action verbs, paths, and files.

        Args:
            tokens: List of token strings.

        Returns:
            Filtered list of tokens.
        """
        filtered: List[str] = []
        for t in tokens:
            t_clean = t.strip()
            if not t_clean:
                continue

            # Protected tokens (paths, files, extensions, numbers) are never removed
            if self.is_protected_token(t_clean):
                filtered.append(t_clean)
                continue

            # Filter standalone punctuation marks (e.g. '?', '!', ',', ';', "'", '"')
            if len(t_clean) == 1 and not t_clean.isalnum():
                continue

            # Check against effective stopwords
            if t_clean.lower() in self.effective_stopwords:
                continue

            filtered.append(t_clean)
        return filtered

    def lemmatize_single(self, token: str) -> str:
        """Lemmatizes an individual token, preserving non-word structures."""
        if self.is_protected_token(token) or not token.isalpha():
            return token

        # Attempt verb lemmatization first (e.g. created -> create, moving -> move)
        verb_lemma = self.lemmatizer.lemmatize(token, pos="v")
        if verb_lemma != token:
            return verb_lemma

        # Fallback to noun lemmatization (e.g. files -> file, directories -> directory)
        return self.lemmatizer.lemmatize(token, pos="n")

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Applies WordNet lemmatization across a list of tokens.

        Args:
            tokens: List of token strings.

        Returns:
            List of lemmatized tokens.
        """
        return [self.lemmatize_single(t) for t in tokens]

    def preprocess(self, text: Optional[str]) -> str:
        """Full preprocessing pipeline: normalize -> tokenize -> remove_stopwords -> lemmatize.

        Args:
            text: Raw input query.

        Returns:
            Preprocessed tokens joined into a single clean string.
        """
        if text is None:
            return ""

        normalized = self.normalize_text(text)
        if not normalized:
            return ""

        tokens = self.tokenize(normalized)
        filtered = self.remove_stopwords(tokens)
        lemmatized = self.lemmatize(filtered)

        return " ".join(lemmatized)
