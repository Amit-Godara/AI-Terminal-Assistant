"""Unit tests for core.intent_classifier.IntentClassifier."""

import json
from pathlib import Path
import pytest
import config
from core.intent_classifier import IntentClassifier, IntentPrediction


@pytest.fixture(scope="module")
def trained_classifier() -> IntentClassifier:
    """Fixture providing a trained and persistent IntentClassifier instance."""
    clf = IntentClassifier(confidence_threshold=0.45)
    if not clf.is_trained():
        clf.train(dataset_path=config.INTENTS_DATASET_PATH, test_size=0.20, random_state=42)
    return clf


def test_1_dataset_loading(trained_classifier: IntentClassifier) -> None:
    """1. Test dataset loading during training."""
    assert trained_classifier.metadata is not None
    assert trained_classifier.metadata.get("dataset_size") == 240
    assert trained_classifier.metadata.get("train_samples") == 192
    assert trained_classifier.metadata.get("test_samples") == 48


def test_2_training(trained_classifier: IntentClassifier) -> None:
    """2. Test model training status and metrics validity."""
    assert trained_classifier.is_trained()
    assert trained_classifier.model is not None
    assert trained_classifier.vectorizer is not None
    metrics = trained_classifier.metadata.get("metrics", {})
    assert "test_accuracy" in metrics
    assert metrics["test_accuracy"] > 0.70, f"Expected >70% accuracy, got {metrics['test_accuracy']}"


def test_3_prediction(trained_classifier: IntentClassifier) -> None:
    """3. Test prediction returns structured IntentPrediction object."""
    pred = trained_classifier.predict("list all files in this directory")
    assert isinstance(pred, IntentPrediction)
    assert pred.predicted_intent == "LIST_FILES"
    assert pred.confidence > 0.0
    assert pred.status in ("CONFIDENT", "LOW_CONFIDENCE")


def test_4_probability_output(trained_classifier: IntentClassifier) -> None:
    """4. Test that predict_proba returns sorted valid probabilities summing to ~1.0."""
    probs = trained_classifier.predict_proba("show directory contents")
    assert isinstance(probs, dict)
    assert len(probs) == 12
    total_prob = sum(probs.values())
    assert pytest.approx(total_prob, rel=1e-3) == 1.0

    # Ensure probabilities are sorted descending
    values = list(probs.values())
    assert values == sorted(values, reverse=True)


def test_5_model_saving(trained_classifier: IntentClassifier, tmp_path: Path) -> None:
    """5. Test model artifact saving to custom directory."""
    m_path = tmp_path / "test_model.joblib"
    v_path = tmp_path / "test_vec.joblib"
    meta_path = tmp_path / "test_meta.json"

    trained_classifier.save_model(model_path=m_path, vectorizer_path=v_path, metadata_path=meta_path)
    assert m_path.exists()
    assert v_path.exists()
    assert meta_path.exists()


def test_6_model_loading(trained_classifier: IntentClassifier, tmp_path: Path) -> None:
    """6. Test model artifact loading into a fresh instance."""
    m_path = tmp_path / "test_model.joblib"
    v_path = tmp_path / "test_vec.joblib"
    meta_path = tmp_path / "test_meta.json"

    trained_classifier.save_model(model_path=m_path, vectorizer_path=v_path, metadata_path=meta_path)

    # Instance with non-existent paths starts untrained
    fresh_clf = IntentClassifier(
        model_path=tmp_path / "none_model.joblib",
        vectorizer_path=tmp_path / "none_vec.joblib",
    )
    assert not fresh_clf.is_trained()

    # Manual load succeeds
    fresh_clf.load_model(model_path=m_path, vectorizer_path=v_path, metadata_path=meta_path)
    assert fresh_clf.is_trained()
    assert len(fresh_clf.classes_) == 12

    # Auto-load on init succeeds when paths point to existing files
    auto_clf = IntentClassifier(model_path=m_path, vectorizer_path=v_path)
    assert auto_clf.is_trained()
    assert len(auto_clf.classes_) == 12


def test_7_prediction_after_reloading(trained_classifier: IntentClassifier, tmp_path: Path) -> None:
    """7. Test predictions from a reloaded model match the original."""
    m_path = tmp_path / "test_model.joblib"
    v_path = tmp_path / "test_vec.joblib"
    meta_path = tmp_path / "test_meta.json"

    trained_classifier.save_model(model_path=m_path, vectorizer_path=v_path, metadata_path=meta_path)

    reloaded = IntentClassifier(model_path=m_path, vectorizer_path=v_path)
    reloaded.load_model(model_path=m_path, vectorizer_path=v_path, metadata_path=meta_path)

    query = "check installed python version"
    pred_orig = trained_classifier.predict(query)
    pred_reloaded = reloaded.predict(query)

    assert pred_orig.predicted_intent == pred_reloaded.predicted_intent
    assert pytest.approx(pred_orig.confidence, rel=1e-3) == pred_reloaded.confidence


def test_8_unknown_and_low_confidence_handling(trained_classifier: IntentClassifier) -> None:
    """8. Test that low-confidence inputs fall back to UNKNOWN with LOW_CONFIDENCE status."""
    # A complete out-of-domain query with random nonsense words
    gibberish = "xylophone quantum banana astrophysics kaleidoscope"
    pred = trained_classifier.predict(gibberish)
    assert pred.predicted_intent == "UNKNOWN"
    assert pred.status == "LOW_CONFIDENCE"


def test_9_all_12_classes(trained_classifier: IntentClassifier) -> None:
    """9. Test that all 12 classes are recognized in classifier model."""
    assert set(trained_classifier.classes_) == set(config.SUPPORTED_INTENTS)
    assert len(trained_classifier.classes_) == 12


def test_10_invalid_input_handling(trained_classifier: IntentClassifier) -> None:
    """10. Test handling of None, empty string, and whitespace queries."""
    for empty_input in [None, "", "   ", "\t\n  "]:
        pred = trained_classifier.predict(empty_input)
        assert pred.predicted_intent == "UNKNOWN"
        assert pred.confidence == 0.0
        assert pred.status == "LOW_CONFIDENCE"


def test_11_reproducibility() -> None:
    """11. Test that training with the same random_state produces identical metrics."""
    clf1 = IntentClassifier()
    eval1 = clf1.train(save_artifacts=False, random_state=42)

    clf2 = IntentClassifier()
    eval2 = clf2.train(save_artifacts=False, random_state=42)

    assert eval1["accuracy"] == eval2["accuracy"]
    assert eval1["macro_f1"] == eval2["macro_f1"]


def test_12_no_test_data_leakage() -> None:
    """12. Test that vectorizer vocabulary is formed strictly from training split."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from core.preprocessor import TextPreprocessor

    df = pd.read_csv(config.INTENTS_DATASET_PATH)
    X = df["text"].tolist()
    y = df["intent"].tolist()

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    assert len(X_train) == 192
    assert len(X_test) == 48

    prep = TextPreprocessor()
    train_clean = [prep.preprocess(t) for t in X_train]

    # Verify that vectorizer fitted during training only contains vocabulary from train_clean
    clf = IntentClassifier()
    clf.train(save_artifacts=False, random_state=42)

    train_tokens = set()
    for doc in train_clean:
        train_tokens.update(doc.split())

    vocab_unigrams = {k for k in clf.vectorizer.vocabulary_.keys() if " " not in k}
    # Every unigram in vectorizer vocabulary must have originated from training split
    assert vocab_unigrams.issubset(train_tokens)
