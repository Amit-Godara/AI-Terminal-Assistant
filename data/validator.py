"""Dataset validation utility for checking format, completeness, duplicates, and class distribution."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import config


def validate_dataset(csv_path: Path = config.INTENTS_DATASET_PATH) -> Dict[str, Any]:
    """Validates an intents dataset CSV file.

    Checks:
    - File exists
    - Required columns ('text', 'intent') exist
    - Missing values in either column
    - Class distribution and supported intent labels
    - Duplicate text entries (case-insensitive)

    Returns a detailed validation report dictionary.
    """
    report: Dict[str, Any] = {
        "is_valid": True,
        "csv_path": str(csv_path),
        "total_examples": 0,
        "num_classes": 0,
        "class_distribution": {},
        "missing_values_count": 0,
        "duplicates_count": 0,
        "duplicate_examples": [],
        "invalid_labels": [],
        "errors": [],
    }

    if not csv_path.exists():
        report["is_valid"] = False
        report["errors"].append(f"Dataset file does not exist at {csv_path}")
        return report

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        report["is_valid"] = False
        report["errors"].append(f"Failed to read CSV: {e}")
        return report

    # Check required columns
    required_cols = {"text", "intent"}
    if not required_cols.issubset(df.columns):
        report["is_valid"] = False
        missing_cols = list(required_cols - set(df.columns))
        report["errors"].append(f"Missing required columns: {missing_cols}")
        return report

    report["total_examples"] = len(df)

    # Check for missing values
    missing_count = int(df[["text", "intent"]].isna().sum().sum())
    report["missing_values_count"] = missing_count
    if missing_count > 0:
        report["is_valid"] = False
        report["errors"].append(f"Found {missing_count} missing values in dataset")

    # Drop missing for further checks to avoid type errors
    clean_df = df.dropna(subset=["text", "intent"])

    # Check invalid intent labels
    valid_intents = set(config.SUPPORTED_INTENTS)
    present_intents = set(clean_df["intent"].unique())
    invalid_labels = list(present_intents - valid_intents)
    report["invalid_labels"] = invalid_labels
    if invalid_labels:
        report["is_valid"] = False
        report["errors"].append(f"Found invalid intent labels: {invalid_labels}")

    # Class distribution
    distribution = clean_df["intent"].value_counts().to_dict()
    report["class_distribution"] = distribution
    report["num_classes"] = len(distribution)

    # Check duplicates (case-insensitive)
    lower_texts = clean_df["text"].str.strip().str.lower()
    duplicates = clean_df[lower_texts.duplicated(keep=False)]["text"].tolist()
    report["duplicates_count"] = len(clean_df[lower_texts.duplicated(keep="first")])
    report["duplicate_examples"] = duplicates
    if report["duplicates_count"] > 0:
        report["is_valid"] = False
        report["errors"].append(f"Found {report['duplicates_count']} duplicate text examples")

    return report


if __name__ == "__main__":
    rep = validate_dataset()
    print("=== DATASET VALIDATION REPORT ===")
    print(f"Valid: {rep['is_valid']}")
    print(f"Total Examples: {rep['total_examples']}")
    print(f"Total Classes: {rep['num_classes']}")
    print("Class Distribution:")
    for intent, count in rep["class_distribution"].items():
        print(f"  - {intent}: {count}")
    print(f"Missing Values: {rep['missing_values_count']}")
    print(f"Duplicates: {rep['duplicates_count']}")
    if rep["errors"]:
        print("Errors:")
        for err in rep["errors"]:
            print(f"  ! {err}")
