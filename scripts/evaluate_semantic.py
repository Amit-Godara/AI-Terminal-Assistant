"""Evaluation script for Semantic Similarity Engine and Hybrid Intent Classifier.

Benchmarks unseen paraphrased queries across all 12 intent classes against:
1. Dense SentenceTransformer Semantic Engine (all-MiniLM-L6-v2)
2. TF-IDF Cosine Similarity Fallback Engine
3. Baseline Supervised Classifier (TF-IDF + LogisticRegression)
4. Hybrid Ensemble Model (predict_hybrid)

Saves results to:
- models/semantic_evaluation_report.json
- models/semantic_comparison_report.json
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from core.intent_classifier import IntentClassifier
from core.semantic_engine import SemanticEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 36 Unseen evaluation queries (3 distinct, realistic paraphrases for each of the 12 intents)
# Completely separate from the 240 examples in data/intents_dataset.csv
UNSEEN_EVAL_BENCHMARK: List[Tuple[str, str]] = [
    # 1. LIST_FILES
    ("display everything in the present directory", "LIST_FILES"),
    ("what items are saved in this folder?", "LIST_FILES"),
    ("show all contents of this directory", "LIST_FILES"),
    # 2. CREATE_DIRECTORY
    ("make a brand new folder called Archives", "CREATE_DIRECTORY"),
    ("build a new directory named Assets", "CREATE_DIRECTORY"),
    ("set up a new folder for logs", "CREATE_DIRECTORY"),
    # 3. DELETE_FILE
    ("erase test.log immediately", "DELETE_FILE"),
    ("get rid of the file summary.docx", "DELETE_FILE"),
    ("discard the document memo.txt", "DELETE_FILE"),
    # 4. SEARCH_FILES
    ("locate every text file inside here", "SEARCH_FILES"),
    ("find where report.pdf is saved", "SEARCH_FILES"),
    ("hunt down any python scripts in this project", "SEARCH_FILES"),
    # 5. CHECK_VERSION
    ("what release of Python is currently active?", "CHECK_VERSION"),
    ("which git build is running?", "CHECK_VERSION"),
    ("display installed node runtime version", "CHECK_VERSION"),
    # 6. CURRENT_DIRECTORY
    ("which folder am I currently standing in?", "CURRENT_DIRECTORY"),
    ("what is my active working directory path?", "CURRENT_DIRECTORY"),
    ("tell me the path I am currently inside", "CURRENT_DIRECTORY"),
    # 7. MOVE_FILE
    ("transfer data.csv into the backup folder", "MOVE_FILE"),
    ("relocate document.pdf to C:\\Archive", "MOVE_FILE"),
    ("shift project.zip over to the external drive", "MOVE_FILE"),
    # 8. COPY_FILE
    ("make a duplicate of config.json in backups", "COPY_FILE"),
    ("clone test.py into the temp folder", "COPY_FILE"),
    ("replicate notes.txt to old_notes.txt", "COPY_FILE"),
    # 9. SYSTEM_INFO
    ("show me my operating system information", "SYSTEM_INFO"),
    ("display hardware resources and memory stats", "SYSTEM_INFO"),
    ("how much RAM and CPU is currently active?", "SYSTEM_INFO"),
    # 10. NETWORK_INFO
    ("what is my local network IP configuration?", "NETWORK_INFO"),
    ("display active network adapters and addresses", "NETWORK_INFO"),
    ("show open network sockets and connections", "NETWORK_INFO"),
    # 11. CREATE_FILE
    ("generate an empty file named output.log", "CREATE_FILE"),
    ("touch a new file called notes.txt", "CREATE_FILE"),
    ("craft a new blank document script.py", "CREATE_FILE"),
    # 12. UNKNOWN
    ("tell me a funny programming joke", "UNKNOWN"),
    ("play some relaxing background music", "UNKNOWN"),
    ("order a pepperoni pizza for dinner tonight", "UNKNOWN"),
]


def run_evaluation() -> Dict[str, Any]:
    """Runs full comparative evaluation on unseen paraphrase benchmark."""
    logger.info("Initializing models for benchmark evaluation...")

    # 1. Supervised baseline classifier
    classifier = IntentClassifier()

    # 2. Dense Semantic Engine (Sentence Transformers)
    dense_engine = SemanticEngine(force_fallback=False)

    # 3. TF-IDF Fallback Semantic Engine
    fallback_engine = SemanticEngine(force_fallback=True)

    logger.info("Running evaluation on %d unseen queries...", len(UNSEEN_EVAL_BENCHMARK))

    results = {
        "dense_st": {"correct_top1": 0, "correct_top3": 0, "latencies": [], "predictions": []},
        "tfidf_fallback": {"correct_top1": 0, "correct_top3": 0, "latencies": [], "predictions": []},
        "classifier_baseline": {"correct_top1": 0, "correct_top3": 0, "latencies": [], "predictions": []},
        "hybrid_ensemble": {"correct_top1": 0, "correct_top3": 0, "latencies": [], "predictions": []},
    }

    per_intent_breakdown: Dict[str, Dict[str, int]] = {
        intent: {"total": 0, "dense_correct": 0, "fallback_correct": 0, "classifier_correct": 0, "hybrid_correct": 0}
        for intent in config.SUPPORTED_INTENTS
    }

    for query, true_intent in UNSEEN_EVAL_BENCHMARK:
        per_intent_breakdown[true_intent]["total"] += 1

        # Evaluate Dense SentenceTransformer
        t0 = time.perf_counter()
        res_dense = dense_engine.find_similarity(query)
        dt_dense = time.perf_counter() - t0
        results["dense_st"]["latencies"].append(dt_dense)
        sorted_dense_intents = [
            item[0]
            for item in sorted(res_dense.intent_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        top1_dense = res_dense.top_intent
        top3_dense = sorted_dense_intents[:3]
        is_dense_top1 = top1_dense == true_intent
        is_dense_top3 = true_intent in top3_dense
        if is_dense_top1:
            results["dense_st"]["correct_top1"] += 1
            per_intent_breakdown[true_intent]["dense_correct"] += 1
        if is_dense_top3:
            results["dense_st"]["correct_top3"] += 1
        results["dense_st"]["predictions"].append({
            "query": query,
            "true_intent": true_intent,
            "predicted": top1_dense,
            "score": res_dense.score,
            "is_correct": is_dense_top1,
        })

        # Evaluate TF-IDF Fallback Engine
        t0 = time.perf_counter()
        res_fallback = fallback_engine.find_similarity(query)
        dt_fallback = time.perf_counter() - t0
        results["tfidf_fallback"]["latencies"].append(dt_fallback)
        sorted_fallback_intents = [
            item[0]
            for item in sorted(res_fallback.intent_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        top1_fallback = res_fallback.top_intent
        top3_fallback = sorted_fallback_intents[:3]
        is_fallback_top1 = top1_fallback == true_intent
        is_fallback_top3 = true_intent in top3_fallback
        if is_fallback_top1:
            results["tfidf_fallback"]["correct_top1"] += 1
            per_intent_breakdown[true_intent]["fallback_correct"] += 1
        if is_fallback_top3:
            results["tfidf_fallback"]["correct_top3"] += 1
        results["tfidf_fallback"]["predictions"].append({
            "query": query,
            "true_intent": true_intent,
            "predicted": top1_fallback,
            "score": res_fallback.score,
            "is_correct": is_fallback_top1,
        })

        # Evaluate Classifier Baseline
        t0 = time.perf_counter()
        res_clf = classifier.predict(query)
        dt_clf = time.perf_counter() - t0
        results["classifier_baseline"]["latencies"].append(dt_clf)
        top1_clf = res_clf.predicted_intent
        top3_clf = [res_clf.predicted_intent] + [alt["intent"] for alt in res_clf.top_alternatives[:2]]
        is_clf_top1 = top1_clf == true_intent
        is_clf_top3 = true_intent in top3_clf
        if is_clf_top1:
            results["classifier_baseline"]["correct_top1"] += 1
            per_intent_breakdown[true_intent]["classifier_correct"] += 1
        if is_clf_top3:
            results["classifier_baseline"]["correct_top3"] += 1
        results["classifier_baseline"]["predictions"].append({
            "query": query,
            "true_intent": true_intent,
            "predicted": top1_clf,
            "confidence": res_clf.confidence,
            "status": res_clf.status,
            "is_correct": is_clf_top1,
        })

        # Evaluate Hybrid Ensemble
        t0 = time.perf_counter()
        res_hybrid = dense_engine.predict_hybrid(query, classifier=classifier)
        dt_hybrid = time.perf_counter() - t0
        results["hybrid_ensemble"]["latencies"].append(dt_hybrid)
        top1_hybrid = res_hybrid.final_intent
        top3_hybrid = [res_hybrid.final_intent] + [alt["intent"] for alt in res_hybrid.top_alternatives[:2]]
        is_hybrid_top1 = top1_hybrid == true_intent
        is_hybrid_top3 = true_intent in top3_hybrid
        if is_hybrid_top1:
            results["hybrid_ensemble"]["correct_top1"] += 1
            per_intent_breakdown[true_intent]["hybrid_correct"] += 1
        if is_hybrid_top3:
            results["hybrid_ensemble"]["correct_top3"] += 1
        results["hybrid_ensemble"]["predictions"].append({
            "query": query,
            "true_intent": true_intent,
            "predicted": top1_hybrid,
            "confidence": res_hybrid.confidence,
            "source": res_hybrid.source,
            "status": res_hybrid.status,
            "is_correct": is_hybrid_top1,
        })

    n_samples = len(UNSEEN_EVAL_BENCHMARK)

    # Summary calculation
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_samples,
        "n_classes": len(config.SUPPORTED_INTENTS),
        "models": {
            "SentenceTransformer (Dense)": {
                "top1_accuracy": round(results["dense_st"]["correct_top1"] / n_samples, 4),
                "top3_accuracy": round(results["dense_st"]["correct_top3"] / n_samples, 4),
                "mean_latency_ms": round(float(np.mean(results["dense_st"]["latencies"]) * 1000), 2),
                "p95_latency_ms": round(float(np.percentile(results["dense_st"]["latencies"], 95) * 1000), 2),
            },
            "TF-IDF Cosine (Fallback)": {
                "top1_accuracy": round(results["tfidf_fallback"]["correct_top1"] / n_samples, 4),
                "top3_accuracy": round(results["tfidf_fallback"]["correct_top3"] / n_samples, 4),
                "mean_latency_ms": round(float(np.mean(results["tfidf_fallback"]["latencies"]) * 1000), 2),
                "p95_latency_ms": round(float(np.percentile(results["tfidf_fallback"]["latencies"], 95) * 1000), 2),
            },
            "Logistic Regression (Baseline Classifier)": {
                "top1_accuracy": round(results["classifier_baseline"]["correct_top1"] / n_samples, 4),
                "top3_accuracy": round(results["classifier_baseline"]["correct_top3"] / n_samples, 4),
                "mean_latency_ms": round(float(np.mean(results["classifier_baseline"]["latencies"]) * 1000), 2),
                "p95_latency_ms": round(float(np.percentile(results["classifier_baseline"]["latencies"], 95) * 1000), 2),
            },
            "Hybrid Model (Classifier + Semantic Engine)": {
                "top1_accuracy": round(results["hybrid_ensemble"]["correct_top1"] / n_samples, 4),
                "top3_accuracy": round(results["hybrid_ensemble"]["correct_top3"] / n_samples, 4),
                "mean_latency_ms": round(float(np.mean(results["hybrid_ensemble"]["latencies"]) * 1000), 2),
                "p95_latency_ms": round(float(np.percentile(results["hybrid_ensemble"]["latencies"], 95) * 1000), 2),
            },
        },
        "per_intent_breakdown": per_intent_breakdown,
    }

    # Save detailed evaluation report
    eval_report = {
        "summary": summary,
        "dense_predictions": results["dense_st"]["predictions"],
        "fallback_predictions": results["tfidf_fallback"]["predictions"],
        "classifier_predictions": results["classifier_baseline"]["predictions"],
        "hybrid_predictions": results["hybrid_ensemble"]["predictions"],
    }

    config.SEMANTIC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SEMANTIC_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Saved semantic evaluation report to %s", config.SEMANTIC_REPORT_PATH)

    with open(config.SEMANTIC_COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved semantic comparison report to %s", config.SEMANTIC_COMPARISON_PATH)

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 5 SEMANTIC SIMILARITY & HYBRID EVALUATION REPORT")
    print("=" * 80)
    print(f"{'Model Architecture':<36} | {'Top-1 Acc':<10} | {'Top-3 Acc':<10} | {'Avg Latency':<12}")
    print("-" * 80)
    for model_name, metrics in summary["models"].items():
        top1_pct = f"{metrics['top1_accuracy'] * 100:.2f}%"
        top3_pct = f"{metrics['top3_accuracy'] * 100:.2f}%"
        latency_str = f"{metrics['mean_latency_ms']:.1f} ms"
        print(f"{model_name:<36} | {top1_pct:<10} | {top3_pct:<10} | {latency_str:<12}")
    print("=" * 80 + "\n")

    return summary


if __name__ == "__main__":
    run_evaluation()
