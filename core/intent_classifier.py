"""Intent Classifier module using TF-IDF feature extraction and Logistic Regression.

Supports training, evaluation, saving/loading artifacts, probabilistic predictions,
confidence gating, and structured prediction outputs.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

# Non-interactive matplotlib backend
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "data" / ".mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from core.preprocessor import TextPreprocessor
from data.validator import validate_dataset

logger = logging.getLogger(__name__)


class IntentPrediction(BaseModel):
    """Structured result of intent classification."""
    original_text: str
    predicted_intent: str
    raw_intent: str
    confidence: float
    top_alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    status: str  # "CONFIDENT" or "LOW_CONFIDENCE"
    is_fallback: bool = False


class IntentClassifier:
    """Supervised Intent Classifier combining TF-IDF and Logistic Regression."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        """Initializes the IntentClassifier.

        Args:
            model_path: Path to serialized classifier (.joblib).
            vectorizer_path: Path to serialized vectorizer (.joblib).
            confidence_threshold: Minimum probability required to accept prediction.
        """
        self.model_path = model_path or config.CLASSIFIER_MODEL_PATH
        self.vectorizer_path = vectorizer_path or config.VECTORIZER_PATH
        self.metadata_path = config.MODEL_METADATA_PATH
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.CONFIDENCE_THRESHOLD
        )

        self.preprocessor = TextPreprocessor()
        self.model: Optional[LogisticRegression] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classes_: Optional[List[str]] = None
        self.metadata: Dict[str, Any] = {}

        # Auto-load model if artifacts exist
        if self.model_path.exists() and self.vectorizer_path.exists():
            try:
                self.load_model()
            except Exception as e:
                logger.warning("Could not auto-load existing model artifacts: %s", e)

    def is_trained(self) -> bool:
        """Returns True if the model and vectorizer are fitted and ready."""
        return self.model is not None and self.vectorizer is not None

    def train(
        self,
        dataset_path: Path = config.INTENTS_DATASET_PATH,
        test_size: float = 0.20,
        random_state: int = 42,
        save_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """Trains the TF-IDF + Logistic Regression model on the specified dataset.

        Args:
            dataset_path: Path to the CSV dataset.
            test_size: Proportion of the dataset to include in the test split.
            random_state: Seed used by the random number generator.
            save_artifacts: Whether to persist artifacts to disk.

        Returns:
            Dictionary containing evaluation metrics and training summary.
        """
        # 1. Validate dataset
        validation_report = validate_dataset(dataset_path)
        if not validation_report["is_valid"]:
            raise ValueError(f"Dataset validation failed: {validation_report['errors']}")

        df = pd.read_csv(dataset_path)
        X = df["text"].tolist()
        y = df["intent"].tolist()

        # 2. Train/Test Stratified Split (Avoid Data Leakage)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        # 3. Preprocess Text
        X_train_clean = [self.preprocessor.preprocess(t) for t in X_train_raw]
        X_test_clean = [self.preprocessor.preprocess(t) for t in X_test_raw]

        # 4. TF-IDF Feature Extraction
        # token_pattern=r'\S+' preserves filenames, extensions, and numbers without splitting
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=False,  # TextPreprocessor already handles case normalization
            token_pattern=r"\S+",
            min_df=1,
        )

        # Fit vectorizer ONLY on training set
        X_train_vec = self.vectorizer.fit_transform(X_train_clean)
        # Transform test set using the fitted vectorizer
        X_test_vec = self.vectorizer.transform(X_test_clean)

        # 5. Logistic Regression Classifier
        self.model = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
            solver="lbfgs",
        )
        self.model.fit(X_train_vec, y_train)
        self.classes_ = list(self.model.classes_)

        # 6. Model Evaluation on Unseen Test Set
        eval_results = self.evaluate(X_test_vec, y_test, X_test_raw)

        # Add train evaluation for sanity check against overfitting
        y_train_pred = self.model.predict(X_train_vec)
        eval_results["train_accuracy"] = float(accuracy_score(y_train, y_train_pred))
        eval_results["train_samples"] = len(y_train)
        eval_results["test_samples"] = len(y_test)
        eval_results["total_samples"] = len(X)
        eval_results["classes"] = self.classes_

        # 7. Metadata creation
        self.metadata = {
            "model_name": "TF-IDF + Logistic Regression Intent Classifier",
            "algorithm": "LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced')",
            "dataset_path": str(dataset_path),
            "dataset_size": len(X),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "intent_classes": self.classes_,
            "num_classes": len(self.classes_),
            "training_date": datetime.now().isoformat(),
            "random_state": random_state,
            "test_size": test_size,
            "vectorizer_config": {
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "lowercase": False,
                "token_pattern": r"\S+",
                "min_df": 1,
                "vocabulary_size": len(self.vectorizer.vocabulary_),
            },
            "metrics": {
                "test_accuracy": eval_results["accuracy"],
                "macro_f1": eval_results["macro_f1"],
                "weighted_f1": eval_results["weighted_f1"],
                "macro_precision": eval_results["macro_precision"],
                "macro_recall": eval_results["macro_recall"],
            },
            "confidence_threshold": self.confidence_threshold,
        }

        # 8. Persist Artifacts
        if save_artifacts:
            self.save_model()
            self._save_evaluation_report(eval_results)
            self._save_confusion_matrix(eval_results["confusion_matrix"], self.classes_)

        return eval_results

    def evaluate(
        self,
        X_test_vec: Any,
        y_test: List[str],
        X_test_raw: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluates model performance against held-out ground truth.

        Args:
            X_test_vec: Vectorized test features.
            y_test: Ground truth labels.
            X_test_raw: Raw text samples for error inspection.

        Returns:
            Dictionary containing metrics, classification report, and confusion matrix.
        """
        if not self.is_trained():
            raise RuntimeError("Model is not trained. Call train() or load_model() first.")

        y_pred = self.model.predict(X_test_vec)

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_str = classification_report(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)

        return {
            "accuracy": acc,
            "macro_precision": prec_macro,
            "macro_recall": rec_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "classification_report_dict": report_dict,
            "classification_report_str": report_str,
            "confusion_matrix": cm.tolist(),
        }

    def predict_proba(self, text: Optional[str]) -> Dict[str, float]:
        """Calculates class probabilities for a given natural-language input query.

        Args:
            text: Raw input query.

        Returns:
            Dictionary mapping intent labels to probability floats, sorted descending.
        """
        if not self.is_trained():
            raise RuntimeError("Model is not trained or loaded. Call train() or load_model() first.")

        if text is None or not text.strip():
            # If text is empty, return uniform or zero probabilities with UNKNOWN preferred
            uniform_prob = 1.0 / len(self.classes_)
            return {c: uniform_prob for c in self.classes_}

        clean_text = self.preprocessor.preprocess(text)
        if not clean_text:
            uniform_prob = 1.0 / len(self.classes_)
            return {c: uniform_prob for c in self.classes_}

        vec = self.vectorizer.transform([clean_text])
        probabilities = self.model.predict_proba(vec)[0]

        prob_dict = {
            cls_name: float(prob)
            for cls_name, prob in zip(self.model.classes_, probabilities)
        }
        # Sort descending by probability
        return dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    def predict(self, text: Optional[str]) -> IntentPrediction:
        """Predicts the user intent from natural-language text with confidence threshold gating.

        Args:
            text: Natural language user query.

        Returns:
            Structured IntentPrediction object.
        """
        if text is None or not text.strip():
            return IntentPrediction(
                original_text=text or "",
                predicted_intent="UNKNOWN",
                raw_intent="UNKNOWN",
                confidence=0.0,
                top_alternatives=[],
                status="LOW_CONFIDENCE",
            )

        prob_dict = self.predict_proba(text)
        sorted_probs = list(prob_dict.items())

        top_class, top_confidence = sorted_probs[0]
        alternatives = [
            {"intent": intent, "confidence": round(prob, 4)}
            for intent, prob in sorted_probs[1:4]
        ]

        if top_confidence < self.confidence_threshold:
            return IntentPrediction(
                original_text=text,
                predicted_intent="UNKNOWN",
                raw_intent=top_class,
                confidence=round(top_confidence, 4),
                top_alternatives=alternatives,
                status="LOW_CONFIDENCE",
            )

        return IntentPrediction(
            original_text=text,
            predicted_intent=top_class,
            raw_intent=top_class,
            confidence=round(top_confidence, 4),
            top_alternatives=alternatives,
            status="CONFIDENT",
        )

    def save_model(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        """Persists trained model, vectorizer, and metadata to disk."""
        if not self.is_trained():
            raise RuntimeError("Cannot save an untrained model.")

        m_path = model_path or self.model_path
        v_path = vectorizer_path or self.vectorizer_path
        meta_path = metadata_path or self.metadata_path

        m_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, m_path)
        joblib.dump(self.vectorizer, v_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("Saved model artifacts to %s, %s, and %s", m_path, v_path, meta_path)

    def load_model(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        """Loads model artifacts from disk."""
        m_path = model_path or self.model_path
        v_path = vectorizer_path or self.vectorizer_path
        meta_path = metadata_path or self.metadata_path

        if not m_path.exists():
            raise FileNotFoundError(f"Model file not found at {m_path}")
        if not v_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found at {v_path}")

        self.model = joblib.load(m_path)
        self.vectorizer = joblib.load(v_path)
        self.classes_ = list(self.model.classes_)

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                if "confidence_threshold" in self.metadata:
                    self.confidence_threshold = self.metadata["confidence_threshold"]

        logger.info("Successfully loaded model artifacts from %s", m_path.parent)

    def _save_evaluation_report(self, eval_results: Dict[str, Any]) -> None:
        """Saves evaluation metrics report to models/evaluation_report.json."""
        report_path = config.EVALUATION_REPORT_PATH
        report_data = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "accuracy": eval_results["accuracy"],
            "macro_precision": eval_results["macro_precision"],
            "macro_recall": eval_results["macro_recall"],
            "macro_f1": eval_results["macro_f1"],
            "weighted_f1": eval_results["weighted_f1"],
            "train_accuracy": eval_results.get("train_accuracy"),
            "train_samples": eval_results.get("train_samples"),
            "test_samples": eval_results.get("test_samples"),
            "classification_report": eval_results["classification_report_dict"],
            "confusion_matrix": eval_results["confusion_matrix"],
            "classes": eval_results.get("classes"),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    def _save_confusion_matrix(self, cm: List[List[int]], classes: List[str]) -> None:
        """Saves confusion matrix visualization to models/confusion_matrix.png."""
        matrix_path = config.CONFUSION_MATRIX_PATH
        cm_array = np.array(cm)

        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, colorbar=True)
        plt.title("Intent Classification Confusion Matrix (Held-out Test Set)")
        plt.tight_layout()
        plt.savefig(matrix_path, dpi=300)
        plt.close(fig)
