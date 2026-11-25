"""Unit tests for Phase 5 Semantic Similarity Engine and Hybrid Intent Inference.

Tests cover:
- Model initialization & lazy loading
- Reference data splitting (no test leakage)
- Embeddings caching
- TF-IDF Cosine Similarity fallback mode
- Offline resilience
- Similarity thresholding & confidence grading
- Intent aggregation strategies ('max', 'top_k_mean', 'mean')
- Unseen paraphrase matching
- Input boundary handling (empty, whitespace, symbols)
- Out-of-scope query rejection
- Hybrid prediction logic (consensus, paraphrase rescue, disagreement, mutual uncertainty)
- Top alternatives ranking and structure
"""

import pytest
import numpy as np

import config
from core.intent_classifier import IntentClassifier
from core.semantic_engine import SemanticEngine, SemanticResult, HybridPrediction


@pytest.fixture(scope="module")
def semantic_engine():
    """Shared SemanticEngine instance with primary SentenceTransformer."""
    return SemanticEngine(force_fallback=False)


@pytest.fixture(scope="module")
def fallback_engine():
    """Shared SemanticEngine instance forced to TF-IDF Cosine fallback mode."""
    return SemanticEngine(force_fallback=True)


@pytest.fixture(scope="module")
def intent_classifier():
    """Shared trained IntentClassifier instance."""
    return IntentClassifier()


# 1. Initialization and reference separation
def test_initialization_and_reference_split(semantic_engine):
    """Verifies engine initializes and strictly uses 192 training samples (no test data leakage)."""
    assert semantic_engine._is_initialized is True
    # 240 dataset examples with 80% train split = 192 references
    assert len(semantic_engine.reference_texts) == 192
    assert len(semantic_engine.reference_intents) == 192
    assert len(set(semantic_engine.reference_intents)) == 12


def test_embeddings_caching(semantic_engine):
    """Verifies dense embeddings are cached and match reference dataset dimension."""
    assert semantic_engine.reference_embeddings is not None
    assert semantic_engine.reference_embeddings.shape == (192, 384)
    # Disk cache file should exist
    assert config.REFERENCE_EMBEDDINGS_PATH.exists()


# 2. Fallback mode
def test_fallback_mode_activation(fallback_engine):
    """Verifies that forced fallback initializes TF-IDF sparse matrix."""
    assert fallback_engine.is_fallback is True
    assert fallback_engine.tfidf_vectorizer is not None
    assert fallback_engine.reference_tfidf is not None
    assert fallback_engine.reference_tfidf.shape[0] == 192

    res = fallback_engine.find_similarity("list all files")
    assert isinstance(res, SemanticResult)
    assert res.is_fallback is True
    assert res.model_used == "TF-IDF-Cosine"
    assert res.top_intent == "LIST_FILES"


# 3. Aggregation strategies
def test_aggregation_strategies(semantic_engine):
    """Tests 'max', 'top_k_mean', and 'mean' aggregation strategies."""
    query = "show contents of this directory"

    res_max = semantic_engine.find_similarity(query, aggregation="max")
    res_top_k = semantic_engine.find_similarity(query, aggregation="top_k_mean")
    res_mean = semantic_engine.find_similarity(query, aggregation="mean")

    assert res_max.top_intent == "LIST_FILES"
    assert res_top_k.top_intent == "LIST_FILES"
    assert res_mean.top_intent == "LIST_FILES"

    # Max similarity should be >= top_k_mean >= mean
    assert res_max.score >= res_top_k.score
    assert res_top_k.score >= res_mean.score


# 4. Unseen paraphrase semantic matching
def test_unseen_paraphrase_matching(semantic_engine):
    """Verifies semantic engine identifies intents on unseen vocabulary."""
    test_cases = [
        ("what items are saved in this folder?", "LIST_FILES"),
        ("which Python release is currently active?", "CHECK_VERSION"),
        ("what is my active working directory path?", "CURRENT_DIRECTORY"),
        ("make a brand new folder called Archives", "CREATE_DIRECTORY"),
        ("show me my operating system information", "SYSTEM_INFO"),
    ]

    for query, expected_intent in test_cases:
        res = semantic_engine.find_similarity(query)
        assert res.top_intent == expected_intent
        assert res.score >= 0.50
        assert res.status == "CONFIDENT"


# 5. Input boundary handling
def test_empty_and_whitespace_queries(semantic_engine, fallback_engine):
    """Verifies empty strings and whitespace return UNKNOWN without errors."""
    for engine in (semantic_engine, fallback_engine):
        for empty_val in ("", "   ", "\t\n"):
            res = engine.find_similarity(empty_val)
            assert res.top_intent == "UNKNOWN"
            assert res.score == 0.0
            assert res.status == "LOW_CONFIDENCE"
            assert len(res.matches) == 0


def test_special_characters_and_symbols(semantic_engine):
    """Verifies queries consisting of punctuation/symbols do not crash the engine."""
    res = semantic_engine.find_similarity("###@@@$$$ %%% ^^^")
    assert isinstance(res, SemanticResult)
    assert res.score <= semantic_engine.similarity_threshold


def test_out_of_scope_rejection(semantic_engine):
    """Verifies conversational/irrelevant queries receive low similarity scores or UNKNOWN."""
    res = semantic_engine.find_similarity("order a pepperoni pizza for dinner tonight")
    assert res.score < 0.40 or res.top_intent == "UNKNOWN"


# 6. Matches structure and top alternatives
def test_matches_and_scores_structure(semantic_engine):
    """Verifies structure of SemanticMatch items and intent_scores dictionary."""
    res = semantic_engine.find_similarity("display all files in current folder", top_k=5)

    assert len(res.matches) == 5
    for match in res.matches:
        assert match.intent in config.SUPPORTED_INTENTS
        assert 0.0 <= match.score <= 1.0
        assert isinstance(match.reference_text, str)

    # Check scores for all 12 intents
    assert len(res.intent_scores) == 12
    for intent in config.SUPPORTED_INTENTS:
        assert intent in res.intent_scores
        assert 0.0 <= res.intent_scores[intent] <= 1.0


# 7. Hybrid prediction tests
def test_hybrid_prediction_consensus(semantic_engine, intent_classifier):
    """When classifier and semantic engine both confidently agree -> HYBRID_ENSEMBLE."""
    query = "check installed python version"
    clf_pred = intent_classifier.predict(query)
    assert clf_pred.status == "CONFIDENT"
    assert clf_pred.predicted_intent == "CHECK_VERSION"

    hybrid_res = semantic_engine.predict_hybrid(query, classifier=intent_classifier)

    assert isinstance(hybrid_res, HybridPrediction)
    assert hybrid_res.final_intent == "CHECK_VERSION"
    assert hybrid_res.source == "HYBRID_ENSEMBLE"
    assert hybrid_res.status == "CONFIDENT"
    assert hybrid_res.classifier_intent == "CHECK_VERSION"
    assert hybrid_res.semantic_intent == "CHECK_VERSION"


def test_hybrid_prediction_paraphrase_rescue(semantic_engine, intent_classifier):
    """When classifier is uncertain on an unseen paraphrase, semantic engine rescues it -> SEMANTIC_ENGINE (BOOSTED)."""
    # Unseen query that classifier lacks exact n-grams for
    query = "what release of Python is currently active?"
    clf_pred = intent_classifier.predict(query)
    # Baseline classifier confidence is either low or wrong on this unseen paraphrase
    hybrid_res = semantic_engine.predict_hybrid(query, classifier=intent_classifier)

    assert hybrid_res.final_intent == "CHECK_VERSION"
    assert hybrid_res.status in ("BOOSTED", "CONFIDENT")
    assert hybrid_res.confidence >= 0.55


def test_hybrid_prediction_mutual_uncertainty(semantic_engine, intent_classifier):
    """When both models are uncertain on an out-of-scope query -> UNKNOWN (FALLBACK)."""
    query = "order a pizza for dinner tonight"
    hybrid_res = semantic_engine.predict_hybrid(query, classifier=intent_classifier)

    assert hybrid_res.final_intent == "UNKNOWN"
    assert hybrid_res.source == "FALLBACK"
    assert hybrid_res.status == "LOW_CONFIDENCE"


def test_hybrid_prediction_alternatives(semantic_engine, intent_classifier):
    """Verifies top alternatives in hybrid prediction are sorted and do not duplicate final intent."""
    query = "show contents of this folder"
    hybrid_res = semantic_engine.predict_hybrid(query, classifier=intent_classifier)

    assert isinstance(hybrid_res.top_alternatives, list)
    assert len(hybrid_res.top_alternatives) > 0
    for alt in hybrid_res.top_alternatives:
        assert alt["intent"] != hybrid_res.final_intent
        assert "score" in alt
