"""Semantic Similarity Engine using Sentence Transformers with TF-IDF fallback.

Provides semantic similarity matching against reference intent utterances,
handles paraphrasing and vocabulary mismatches, supports lazy loading and offline
caching, and integrates with the supervised IntentClassifier via hybrid inference.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Configure offline-first HuggingFace operations to prevent remote network hangs
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import config
from core.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class SemanticMatch(BaseModel):
    """Represents a match against an individual reference utterance."""
    intent: str
    score: float
    reference_text: str


class SemanticResult(BaseModel):
    """Structured result of semantic similarity analysis for a query."""
    query: str
    top_intent: str
    score: float
    status: str  # "CONFIDENT" or "LOW_CONFIDENCE"
    matches: List[SemanticMatch] = Field(default_factory=list)
    intent_scores: Dict[str, float] = Field(default_factory=dict)
    is_fallback: bool = False
    model_used: str = ""


class HybridPrediction(BaseModel):
    """Structured result of hybrid (Classifier + Semantic Engine) intent inference."""
    query: str
    final_intent: str
    source: str  # "CLASSIFIER", "SEMANTIC_ENGINE", "HYBRID_ENSEMBLE", "FALLBACK"
    confidence: float
    classifier_intent: Optional[str] = None
    classifier_confidence: Optional[float] = None
    semantic_intent: Optional[str] = None
    semantic_score: Optional[float] = None
    status: str = "CONFIDENT"  # "CONFIDENT", "LOW_CONFIDENCE", "BOOSTED", "DISAGREEMENT"
    top_alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    is_fallback: bool = False


class SemanticEngine:
    """Computes semantic similarity against reference intents with hybrid fallback."""

    def __init__(
        self,
        model_name: str = config.SENTENCE_TRANSFORMER_MODEL,
        similarity_threshold: float = config.SEMANTIC_SIMILARITY_THRESHOLD,
        dataset_path: Path = config.INTENTS_DATASET_PATH,
        embeddings_cache_path: Path = config.REFERENCE_EMBEDDINGS_PATH,
        use_training_split_only: bool = True,
        auto_initialize: bool = True,
        force_fallback: bool = False,
    ) -> None:
        """Initializes the SemanticEngine.

        Args:
            model_name: Name of SentenceTransformer model (default: all-MiniLM-L6-v2).
            similarity_threshold: Minimum cosine similarity score for confident match.
            dataset_path: Path to intents dataset CSV.
            embeddings_cache_path: Path to disk cache for reference embeddings.
            use_training_split_only: If True, uses stratified 80% split to prevent test data leakage.
            auto_initialize: Whether to load references and model upon instantiation.
            force_fallback: If True, forces TF-IDF cosine similarity fallback mode.
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.dataset_path = dataset_path
        self.embeddings_cache_path = embeddings_cache_path
        self.use_training_split_only = use_training_split_only
        self.force_fallback = force_fallback

        self.preprocessor = TextPreprocessor()
        self.is_fallback: bool = force_fallback
        self._model: Any = None
        self._is_initialized: bool = False

        # Reference data structures
        self.reference_texts: List[str] = []
        self.reference_intents: List[str] = []
        self.reference_embeddings: Optional[np.ndarray] = None

        # Fallback TF-IDF structures
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.reference_tfidf: Optional[Any] = None

        if auto_initialize:
            self.initialize()

    def initialize(self) -> None:
        """Loads reference dataset, initializes primary or fallback model, and builds reference vectors."""
        if self._is_initialized:
            return

        # 1. Load reference dataset
        self._load_reference_data()

        # 2. Try loading primary SentenceTransformer model unless force_fallback is set
        if not self.force_fallback:
            self._load_sentence_transformer()
        else:
            self.is_fallback = True

        # 3. Setup embeddings / vectors
        if not self.is_fallback and self._model is not None:
            self._build_or_load_dense_embeddings()
        else:
            self._build_tfidf_fallback()

        self._is_initialized = True

    def _load_reference_data(self) -> None:
        """Loads reference utterances and labels from dataset, preserving test split separation."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Intents dataset not found at {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)
        if "text" not in df.columns or "intent" not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'intent' columns.")

        X = df["text"].astype(str).tolist()
        y = df["intent"].astype(str).tolist()

        if self.use_training_split_only:
            # Reproduce exact stratified split used in Phase 3 training (random_state=42, test_size=0.20)
            X_train, _, y_train, _ = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )
            self.reference_texts = X_train
            self.reference_intents = y_train
        else:
            self.reference_texts = X
            self.reference_intents = y

        logger.info(
            "Loaded %d reference utterances across %d intents (training_split_only=%s).",
            len(self.reference_texts),
            len(set(self.reference_intents)),
            self.use_training_split_only,
        )

    def _load_sentence_transformer(self) -> None:
        """Attempts to load SentenceTransformer lazily, falling back on any failure."""
        try:
            from sentence_transformers import SentenceTransformer

            # Attempt local cache load without remote network requirement
            self._model = SentenceTransformer(self.model_name)
            self.is_fallback = False
            logger.info("Successfully loaded SentenceTransformer model: %s", self.model_name)
        except Exception as exc:
            logger.warning(
                "Failed to load SentenceTransformer ('%s'): %s. Switching to TF-IDF cosine similarity fallback.",
                self.model_name,
                exc,
            )
            self._model = None
            self.is_fallback = True

    def _build_or_load_dense_embeddings(self) -> None:
        """Loads dense embeddings from cache or computes and caches them."""
        # Check cache on disk
        if self.embeddings_cache_path.exists():
            try:
                cached_data = joblib.load(self.embeddings_cache_path)
                if (
                    isinstance(cached_data, dict)
                    and cached_data.get("model_name") == self.model_name
                    and cached_data.get("texts") == self.reference_texts
                    and "embeddings" in cached_data
                ):
                    self.reference_embeddings = cached_data["embeddings"]
                    logger.info("Loaded cached reference embeddings from %s", self.embeddings_cache_path)
                    return
            except Exception as e:
                logger.warning("Could not read cached embeddings (%s). Recomputing...", e)

        # Compute embeddings
        logger.info("Computing dense reference embeddings for %d utterances...", len(self.reference_texts))
        try:
            embeddings = self._model.encode(
                self.reference_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self.reference_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

            # Save to disk cache
            try:
                cache_payload = {
                    "model_name": self.model_name,
                    "texts": self.reference_texts,
                    "intents": self.reference_intents,
                    "embeddings": self.reference_embeddings,
                }
                joblib.dump(cache_payload, self.embeddings_cache_path)
                logger.info("Persisted reference embeddings to %s", self.embeddings_cache_path)
            except Exception as cache_err:
                logger.warning("Failed to persist embeddings cache: %s", cache_err)
        except Exception as encode_err:
            logger.error("Failed to compute dense embeddings: %s. Switching to fallback.", encode_err)
            self.is_fallback = True
            self._build_tfidf_fallback()

    def _build_tfidf_fallback(self) -> None:
        """Sets up TF-IDF cosine similarity fallback representations."""
        logger.info("Configuring TF-IDF fallback representations...")
        cleaned_references = [self.preprocessor.preprocess(t) for t in self.reference_texts]

        # Use unigram + bigram representation preserving special tokens
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"\S+",
            sublinear_tf=True,
        )
        self.reference_tfidf = self.tfidf_vectorizer.fit_transform(cleaned_references)
        self.is_fallback = True

    def find_similarity(
        self,
        query: str,
        top_k: int = 5,
        aggregation: str = "top_k_mean",
    ) -> SemanticResult:
        """Finds closest matching intents and reference utterances for a query.

        Args:
            query: Natural language user query.
            top_k: Number of individual reference matches to return.
            aggregation: Aggregation strategy per intent ('max', 'top_k_mean', 'mean').

        Returns:
            SemanticResult containing top intent, score, matches, and intent rankings.
        """
        if not self._is_initialized:
            self.initialize()

        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return SemanticResult(
                query=query,
                top_intent="UNKNOWN",
                score=0.0,
                status="LOW_CONFIDENCE",
                matches=[],
                intent_scores={intent: 0.0 for intent in config.SUPPORTED_INTENTS},
                is_fallback=self.is_fallback,
                model_used="TF-IDF-Cosine" if self.is_fallback else self.model_name,
            )

        # Compute raw similarity scores against all reference utterances
        if not self.is_fallback and self._model is not None and self.reference_embeddings is not None:
            try:
                query_embedding = self._model.encode(
                    [cleaned_query],
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                # Since embeddings are normalized, dot product equals cosine similarity
                raw_scores = np.dot(self.reference_embeddings, query_embedding.T).flatten()
            except Exception as e:
                logger.warning("Dense embedding failed on query (%s), executing fallback: %s", cleaned_query, e)
                if self.tfidf_vectorizer is None:
                    self._build_tfidf_fallback()
                raw_scores = self._compute_tfidf_scores(cleaned_query)
        else:
            if self.tfidf_vectorizer is None:
                self._build_tfidf_fallback()
            raw_scores = self._compute_tfidf_scores(cleaned_query)

        # Aggregate similarity scores by intent
        intent_scores: Dict[str, float] = {}
        unique_intents = list(set(self.reference_intents))

        for intent in unique_intents:
            indices = [i for i, lbl in enumerate(self.reference_intents) if lbl == intent]
            candidate_scores = raw_scores[indices]
            if len(candidate_scores) == 0:
                intent_scores[intent] = 0.0
                continue

            if aggregation == "max":
                score = float(np.max(candidate_scores))
            elif aggregation == "top_k_mean":
                # Average of top 2 matches for this intent to balance peak similarity and consistency
                k_val = min(2, len(candidate_scores))
                score = float(np.mean(np.partition(candidate_scores, -k_val)[-k_val:]))
            elif aggregation == "mean":
                score = float(np.mean(candidate_scores))
            else:
                score = float(np.max(candidate_scores))

            # Round to 4 decimal places
            intent_scores[intent] = round(score, 4)

        # Ensure all supported intents are represented
        for intent in config.SUPPORTED_INTENTS:
            if intent not in intent_scores:
                intent_scores[intent] = 0.0

        # Sort intents by aggregated score
        sorted_intents = sorted(intent_scores.items(), key=lambda item: item[1], reverse=True)
        top_intent, top_score = sorted_intents[0] if sorted_intents else ("UNKNOWN", 0.0)

        # Individual utterance matches
        top_indices = np.argsort(raw_scores)[::-1][:top_k]
        matches: List[SemanticMatch] = [
            SemanticMatch(
                intent=self.reference_intents[idx],
                score=round(float(raw_scores[idx]), 4),
                reference_text=self.reference_texts[idx],
            )
            for idx in top_indices
        ]

        # Determine confidence status
        status = "CONFIDENT" if top_score >= self.similarity_threshold else "LOW_CONFIDENCE"
        if top_score < 0.35:
            # Score is too low to represent any known command pattern
            top_intent = "UNKNOWN"

        return SemanticResult(
            query=query,
            top_intent=top_intent,
            score=round(top_score, 4),
            status=status,
            matches=matches,
            intent_scores=intent_scores,
            is_fallback=self.is_fallback,
            model_used="TF-IDF-Cosine" if self.is_fallback else self.model_name,
        )

    def _compute_tfidf_scores(self, query: str) -> np.ndarray:
        """Computes TF-IDF cosine similarity vector against reference texts."""
        clean_query = self.preprocessor.preprocess(query)
        if not clean_query.strip():
            return np.zeros(len(self.reference_texts), dtype=np.float32)

        query_vec = self.tfidf_vectorizer.transform([clean_query])
        scores = cosine_similarity(query_vec, self.reference_tfidf).flatten()
        return scores

    def predict_hybrid(
        self,
        query: str,
        classifier: Optional[Any] = None,
        aggregation: str = "top_k_mean",
    ) -> HybridPrediction:
        """Performs hybrid prediction integrating IntentClassifier and SemanticEngine.

        Decision Logic:
        1. Query classifier and semantic similarity engine.
        2. If classifier is confident (>= 0.45) and not UNKNOWN:
           - If semantic engine agrees or has moderate support: Final intent = classifier intent (Source: CLASSIFIER).
           - If semantic engine strongly disagrees with high similarity (>= 0.85) while classifier is borderline (< 0.55):
             Semantic engine resolves ambiguity (Source: SEMANTIC_ENGINE, Status: DISAGREEMENT).
        3. If classifier is low confidence (< 0.45) or UNKNOWN:
           - If semantic engine has confident match (>= similarity_threshold):
             Semantic engine rescues prediction (Source: SEMANTIC_ENGINE, Status: BOOSTED).
           - Else:
             Both models uncertain -> Final intent = UNKNOWN (Source: FALLBACK, Status: LOW_CONFIDENCE).

        Args:
            query: User's natural language command string.
            classifier: Optional IntentClassifier instance. If None, instantiates one.
            aggregation: Semantic aggregation method ('top_k_mean', 'max', 'mean').

        Returns:
            HybridPrediction with final intent, confidence, sources, and alternatives.
        """
        # Lazy import of IntentClassifier to avoid circular dependency
        if classifier is None:
            from core.intent_classifier import IntentClassifier

            classifier = IntentClassifier()

        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return HybridPrediction(
                query=query,
                final_intent="UNKNOWN",
                source="FALLBACK",
                confidence=0.0,
                classifier_intent="UNKNOWN",
                classifier_confidence=0.0,
                semantic_intent="UNKNOWN",
                semantic_score=0.0,
                status="LOW_CONFIDENCE",
                top_alternatives=[],
                is_fallback=self.is_fallback,
            )

        # 1. Obtain individual model inferences
        clf_pred = classifier.predict(cleaned_query)
        sem_res = self.find_similarity(cleaned_query, aggregation=aggregation)

        clf_intent = clf_pred.predicted_intent
        clf_conf = clf_pred.confidence
        clf_is_confident = (clf_pred.status == "CONFIDENT") and (clf_intent != "UNKNOWN")

        sem_intent = sem_res.top_intent
        sem_score = sem_res.score
        sem_is_confident = (sem_res.status == "CONFIDENT") and (sem_intent != "UNKNOWN")

        # 2. Hybrid Decision Rules
        final_intent: str
        source: str
        confidence: float
        status: str

        if clf_is_confident and sem_is_confident:
            if clf_intent == sem_intent:
                # Strong consensus between both models
                final_intent = clf_intent
                source = "HYBRID_ENSEMBLE"
                confidence = round(max(clf_conf, sem_score), 4)
                status = "CONFIDENT"
            elif sem_score >= 0.82 and clf_conf < 0.60:
                # Semantic match is overwhelmingly strong on a paraphrase, classifier is borderline
                final_intent = sem_intent
                source = "SEMANTIC_ENGINE"
                confidence = round(sem_score, 4)
                status = "BOOSTED"
            else:
                # Classifier takes precedence on learned discriminative boundaries
                final_intent = clf_intent
                source = "CLASSIFIER"
                confidence = round(clf_conf, 4)
                status = "DISAGREEMENT"

        elif clf_is_confident and not sem_is_confident:
            # Classifier is confident, semantic similarity didn't cross threshold
            final_intent = clf_intent
            source = "CLASSIFIER"
            confidence = round(clf_conf, 4)
            status = "CONFIDENT"

        elif not clf_is_confident and sem_is_confident:
            # Paraphrase rescue: Classifier lacked exact n-grams, semantic engine rescues intent!
            final_intent = sem_intent
            source = "SEMANTIC_ENGINE"
            confidence = round(sem_score, 4)
            status = "BOOSTED"

        else:
            # Both models are uncertain
            final_intent = "UNKNOWN"
            source = "FALLBACK"
            confidence = round(max(clf_conf, sem_score), 4)
            status = "LOW_CONFIDENCE"

        # Combine alternatives from both engines
        alternatives: List[Dict[str, Any]] = []
        seen_intents = {final_intent}

        # Add top alternatives from semantic scores
        sorted_sem_intents = sorted(sem_res.intent_scores.items(), key=lambda x: x[1], reverse=True)
        for alt_intent, alt_score in sorted_sem_intents:
            if alt_intent not in seen_intents and alt_intent != "UNKNOWN":
                alternatives.append({"intent": alt_intent, "score": alt_score, "origin": "semantic"})
                seen_intents.add(alt_intent)
            if len(alternatives) >= 3:
                break

        return HybridPrediction(
            query=query,
            final_intent=final_intent,
            source=source,
            confidence=confidence,
            classifier_intent=clf_intent,
            classifier_confidence=clf_conf,
            semantic_intent=sem_intent,
            semantic_score=sem_score,
            status=status,
            top_alternatives=alternatives,
            is_fallback=self.is_fallback,
        )
