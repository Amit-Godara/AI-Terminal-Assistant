"""Script to train, evaluate, and save the Intent Classification model."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config
from core.intent_classifier import IntentClassifier

console = Console()

# 24 unseen test cases (2 per intent) for interactive prediction verification
UNSEEN_TEST_CASES = [
    # 1. LIST_FILES
    ("can you show me what's inside this directory?", "LIST_FILES"),
    ("display all folders and files here", "LIST_FILES"),

    # 2. CREATE_DIRECTORY
    ("make a directory called WorkDocs", "CREATE_DIRECTORY"),
    ("create a new folder for projects", "CREATE_DIRECTORY"),

    # 3. DELETE_FILE
    ("please discard the file old_draft.txt", "DELETE_FILE"),
    ("remove file obsolete_data.csv", "DELETE_FILE"),

    # 4. SEARCH_FILES
    ("locate all spreadsheets with .xlsx format", "SEARCH_FILES"),
    ("find files with pattern config_*.json", "SEARCH_FILES"),

    # 5. CHECK_VERSION
    ("what version of python is installed on my system?", "CHECK_VERSION"),
    ("show git build version", "CHECK_VERSION"),

    # 6. CURRENT_DIRECTORY
    ("what is the path of this folder?", "CURRENT_DIRECTORY"),
    ("where am I right now?", "CURRENT_DIRECTORY"),

    # 7. MOVE_FILE
    ("relocate report.pdf to C:\\Archive", "MOVE_FILE"),
    ("transfer image.png into the assets folder", "MOVE_FILE"),

    # 8. COPY_FILE
    ("make a duplicate copy of summary.docx", "COPY_FILE"),
    ("replicate dataset.csv into C:\\DataBackup", "COPY_FILE"),

    # 9. SYSTEM_INFO
    ("what are the hardware specifications of this machine?", "SYSTEM_INFO"),
    ("display processor and operating system information", "SYSTEM_INFO"),

    # 10. NETWORK_INFO
    ("display all active network adapters and ip addresses", "NETWORK_INFO"),
    ("check my internet configuration and gateway", "NETWORK_INFO"),

    # 11. CREATE_FILE
    ("make a brand new file named setup.py", "CREATE_FILE"),
    ("generate an empty file called notes.md", "CREATE_FILE"),

    # 12. UNKNOWN
    ("how far is the moon from earth?", "UNKNOWN"),
    ("can you brew a cup of coffee?", "UNKNOWN"),
]


def run_training() -> None:
    """Executes model training pipeline and runs interactive prediction tests."""
    console.print(
        Panel.fit(
            "[bold cyan]AI TERMINAL ASSISTANT - MODEL TRAINING PIPELINE[/bold cyan]\n"
            "[dim]TF-IDF Feature Extraction + Logistic Regression Classifier[/dim]",
            border_style="cyan",
        )
    )

    classifier = IntentClassifier(confidence_threshold=config.CONFIDENCE_THRESHOLD)

    console.print(f"[bold yellow]1. Loading dataset from:[/] {config.INTENTS_DATASET_PATH}")
    eval_results = classifier.train(
        dataset_path=config.INTENTS_DATASET_PATH,
        test_size=0.20,
        random_state=42,
        save_artifacts=True,
    )

    console.print("[bold green]✓ Training and evaluation complete![/bold green]\n")

    # Display overall metrics table
    metrics_table = Table(title="Model Evaluation Summary (Held-Out Test Set: 20%)", border_style="green")
    metrics_table.add_column("Metric", style="cyan", no_wrap=True)
    metrics_table.add_column("Value", style="bold white")

    metrics_table.add_row("Total Samples", str(eval_results["total_samples"]))
    metrics_table.add_row("Training Samples", str(eval_results["train_samples"]))
    metrics_table.add_row("Test Samples", str(eval_results["test_samples"]))
    metrics_table.add_row("Train Accuracy", f"{eval_results['train_accuracy'] * 100:.2f}%")
    metrics_table.add_row("Test Accuracy", f"{eval_results['accuracy'] * 100:.2f}%")
    metrics_table.add_row("Macro F1-Score", f"{eval_results['macro_f1'] * 100:.2f}%")
    metrics_table.add_row("Weighted F1-Score", f"{eval_results['weighted_f1'] * 100:.2f}%")
    metrics_table.add_row("Macro Precision", f"{eval_results['macro_precision'] * 100:.2f}%")
    metrics_table.add_row("Macro Recall", f"{eval_results['macro_recall'] * 100:.2f}%")
    metrics_table.add_row("Confidence Threshold", f"{classifier.confidence_threshold:.2f}")

    console.print(metrics_table)
    console.print()

    # Display classification report
    console.print("[bold cyan]Classification Report:[/bold cyan]")
    console.print(eval_results["classification_report_str"])

    # Artifact persistence confirmation
    console.print("\n[bold cyan]Saved Artifacts:[/bold cyan]")
    console.print(f"  • Classifier Model: [green]{config.CLASSIFIER_MODEL_PATH}[/green]")
    console.print(f"  • TF-IDF Vectorizer: [green]{config.VECTORIZER_PATH}[/green]")
    console.print(f"  • Model Metadata:   [green]{config.MODEL_METADATA_PATH}[/green]")
    console.print(f"  • Evaluation Report: [green]{config.EVALUATION_REPORT_PATH}[/green]")
    console.print(f"  • Confusion Matrix:  [green]{config.CONFUSION_MATRIX_PATH}[/green]\n")

    # 24 Unseen Sample Predictions
    console.print("[bold yellow]2. Running Interactive Prediction Verification (24 Unseen Queries):[/bold yellow]")
    test_table = Table(title="Unseen Paraphrase Predictions", border_style="cyan")
    test_table.add_column("Query", style="white", max_width=45)
    test_table.add_column("Expected", style="cyan")
    test_table.add_column("Predicted", style="bold")
    test_table.add_column("Confidence", style="yellow")
    test_table.add_column("Status", justify="center")

    passed_count = 0
    for query, expected in UNSEEN_TEST_CASES:
        pred = classifier.predict(query)
        is_pass = (pred.predicted_intent == expected)
        if is_pass:
            passed_count += 1
            status_str = "[bold green]PASS[/bold green]"
            pred_style = f"[green]{pred.predicted_intent}[/green]"
        else:
            status_str = "[bold red]FAIL[/bold red]"
            pred_style = f"[red]{pred.predicted_intent}[/red]"

        test_table.add_row(
            query,
            expected,
            pred_style,
            f"{pred.confidence * 100:.1f}%",
            status_str,
        )

    console.print(test_table)
    console.print(
        f"\n[bold white]Verification Result:[/] [bold green]{passed_count}/{len(UNSEEN_TEST_CASES)} passed[/bold green] "
        f"({passed_count / len(UNSEEN_TEST_CASES) * 100:.1f}% accuracy on unseen queries)\n"
    )


if __name__ == "__main__":
    run_training()
