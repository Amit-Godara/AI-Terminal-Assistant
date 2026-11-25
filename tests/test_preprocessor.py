"""Unit tests for core.preprocessor.TextPreprocessor."""

import pytest
from core.preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Fixture providing a TextPreprocessor instance."""
    return TextPreprocessor()


def test_lowercase_normalization(preprocessor: TextPreprocessor) -> None:
    """1. Test lowercase conversion."""
    text = "SHOW ALL FILES AND DIRECTORIES"
    normalized = preprocessor.normalize_text(text)
    assert normalized == "show all files and directories"
    assert "SHOW" not in normalized


def test_whitespace_normalization(preprocessor: TextPreprocessor) -> None:
    """2. Test normalization of tabs, newlines, and multiple spaces."""
    text = "   find   all   python \t files \n in   folder   "
    normalized = preprocessor.normalize_text(text)
    assert normalized == "find all python files in folder"
    assert "  " not in normalized


def test_tokenization(preprocessor: TextPreprocessor) -> None:
    """3. Test tokenization into meaningful units."""
    text = "make a directory named Projects"
    tokens = preprocessor.tokenize(text)
    assert isinstance(tokens, list)
    assert tokens == ["make", "a", "directory", "named", "Projects"]


def test_stopword_handling(preprocessor: TextPreprocessor) -> None:
    """4. Test that non-essential stopwords are removed while preserving key words."""
    tokens = ["find", "all", "the", "files", "in", "this", "folder"]
    filtered = preprocessor.remove_stopwords(tokens)
    # 'the', 'in', 'this' should be removed
    assert "the" not in filtered
    assert "this" not in filtered
    assert "find" in filtered
    assert "files" in filtered


def test_negation_preservation(preprocessor: TextPreprocessor) -> None:
    """5. Test that negations (not, don't, never, etc.) are strictly preserved."""
    test_cases = [
        "do not delete notes.txt",
        "never remove report.pdf",
        "don't erase system.log",
    ]
    for tc in test_cases:
        res = preprocessor.preprocess(tc)
        assert any(neg in res for neg in ["not", "never", "don't", "dont"])


def test_lemmatization(preprocessor: TextPreprocessor) -> None:
    """6. Test WordNet lemmatization of nouns and verbs."""
    tokens = ["files", "directories", "created", "moving", "copied"]
    lemmatized = preprocessor.lemmatize(tokens)
    assert lemmatized == ["file", "directory", "create", "move", "copy"]


def test_filename_preservation(preprocessor: TextPreprocessor) -> None:
    """7. Test preservation of filenames with extensions."""
    samples = ["notes.txt", "annual_report.pdf", "data-v1.csv", "archive.zip"]
    for fname in samples:
        res = preprocessor.preprocess(f"delete file {fname}")
        assert fname.lower() in res


def test_file_extension_preservation(preprocessor: TextPreprocessor) -> None:
    """8. Test preservation of standalone file extensions."""
    extensions = [".py", ".pdf", ".json", ".csv", ".log", ".docx"]
    for ext in extensions:
        res = preprocessor.preprocess(f"search for {ext} files")
        assert ext in res


def test_windows_path_preservation(preprocessor: TextPreprocessor) -> None:
    """9. Test preservation of Windows absolute and relative paths."""
    paths = [
        r"C:\Users\Amit\Documents",
        r"D:\Projects\data",
        r".\logs\output.log",
        r"..\backup",
    ]
    for p in paths:
        res = preprocessor.preprocess(f"move file to {p}")
        assert p.lower() in res


def test_number_preservation(preprocessor: TextPreprocessor) -> None:
    """10. Test that numbers and years are not stripped."""
    text = "Financial Reports 2026 and 42 items"
    res = preprocessor.preprocess(text)
    assert "2026" in res
    assert "42" in res


def test_empty_input_handling(preprocessor: TextPreprocessor) -> None:
    """11. Test that None, empty string, and whitespace-only string return empty string."""
    assert preprocessor.normalize_text("") == ""
    assert preprocessor.normalize_text(None) == ""
    assert preprocessor.normalize_text("   \n\t  ") == ""
    assert preprocessor.tokenize("") == []
    assert preprocessor.tokenize(None) == []
    assert preprocessor.preprocess("") == ""
    assert preprocessor.preprocess(None) == ""
    assert preprocessor.preprocess("    ") == ""


def test_punctuation_handling(preprocessor: TextPreprocessor) -> None:
    """12. Test that trailing and standalone punctuation is cleaned without harming filenames."""
    text = "Find ALL Python files in this directory?!?"
    res = preprocessor.preprocess(text)
    assert "?" not in res
    assert "!" not in res
    assert res == "find python file directory"
