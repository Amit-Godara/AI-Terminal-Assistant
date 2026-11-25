"""Unit tests for dataset loading, class distribution, duplicates, and missing values."""

import pytest
import pandas as pd
from pathlib import Path
import config
from data.validator import validate_dataset


def test_dataset_loading() -> None:
    """13. Test that the dataset exists and loads into a non-empty DataFrame."""
    assert config.INTENTS_DATASET_PATH.exists(), f"Missing dataset file at {config.INTENTS_DATASET_PATH}"
    df = pd.read_csv(config.INTENTS_DATASET_PATH)
    assert not df.empty
    assert "text" in df.columns
    assert "intent" in df.columns
    assert len(df) >= 200, f"Expected at least 200 examples, got {len(df)}"


def test_dataset_class_validation() -> None:
    """14. Test that all 12 supported intents are present with balanced distribution."""
    report = validate_dataset(config.INTENTS_DATASET_PATH)
    assert report["is_valid"], f"Validation failed with errors: {report['errors']}"
    assert report["num_classes"] == 12, f"Expected 12 classes, got {report['num_classes']}"

    # Ensure every supported intent is in the dataset
    for expected_intent in config.SUPPORTED_INTENTS:
        assert expected_intent in report["class_distribution"], f"Missing intent: {expected_intent}"
        count = report["class_distribution"][expected_intent]
        assert count >= 15, f"Class {expected_intent} has only {count} examples (expected ~20)"


def test_duplicate_detection(tmp_path: Path) -> None:
    """15. Test duplicate detection mechanism on valid dataset and on artificial duplicate."""
    # Test on real dataset: should have zero duplicates
    real_report = validate_dataset(config.INTENTS_DATASET_PATH)
    assert real_report["duplicates_count"] == 0, f"Found unexpected duplicates: {real_report['duplicate_examples']}"

    # Test validator catches introduced duplicate
    dup_csv = tmp_path / "dup_test.csv"
    dup_df = pd.DataFrame([
        {"text": "show all files", "intent": "LIST_FILES"},
        {"text": "SHOW ALL FILES", "intent": "LIST_FILES"},
    ])
    dup_df.to_csv(dup_csv, index=False)
    dup_report = validate_dataset(dup_csv)
    assert not dup_report["is_valid"]
    assert dup_report["duplicates_count"] == 1


def test_missing_value_detection(tmp_path: Path) -> None:
    """16. Test missing value detection on valid dataset and on artificial missing values."""
    # Test on real dataset: should have zero missing values
    real_report = validate_dataset(config.INTENTS_DATASET_PATH)
    assert real_report["missing_values_count"] == 0, "Real dataset contains missing values"

    # Test validator catches missing text or intent
    missing_csv = tmp_path / "missing_test.csv"
    missing_df = pd.DataFrame([
        {"text": "show all files", "intent": "LIST_FILES"},
        {"text": None, "intent": "LIST_FILES"},
        {"text": "make folder test", "intent": None},
    ])
    missing_df.to_csv(missing_csv, index=False)
    missing_report = validate_dataset(missing_csv)
    assert not missing_report["is_valid"]
    assert missing_report["missing_values_count"] == 2
