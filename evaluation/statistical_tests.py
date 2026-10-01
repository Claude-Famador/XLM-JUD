"""
Statistical Hypothesis Testing (Section 3.3.3 & 3.3.4).
"""

import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def paired_t_test(scores_a, scores_b, name_a="Model A", name_b="Model B"):
    """Paired t-test on k-fold results. Guards against degenerate (all-identical) scores."""
    scores_a, scores_b = np.array(scores_a), np.array(scores_b)
    
    # Guard: if all scores are identical (mock CV), the test is meaningless
    if np.std(scores_a) == 0 and np.std(scores_b) == 0:
        print(f"  WARNING: Skipping paired t-test ({name_a} vs {name_b}) — "
              f"all fold scores are identical (likely mock CV). Run with --cv for valid results.")
        return {
            "test": "Paired t-test (SKIPPED — degenerate scores)",
            "model_a": name_a, "model_b": name_b,
            "mean_a": float(np.mean(scores_a)), "mean_b": float(np.mean(scores_b)),
            "p_value": None, "significant": None,
            "note": "All fold scores identical — run proper k-fold CV for valid statistical test"
        }
    
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    is_significant = p_value < config.SIGNIFICANCE_THRESHOLD
    
    return {
        "test": "Paired t-test",
        "model_a": name_a, "model_b": name_b,
        "mean_a": float(np.mean(scores_a)), "mean_b": float(np.mean(scores_b)),
        "p_value": float(p_value), "significant": bool(is_significant)
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b, name_a="Model A", name_b="Model B"):
    """McNemar's test on prediction vectors."""
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    
    chi2 = ((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) > 0 else 0.0
    p_value = 1 - stats.chi2.cdf(chi2, df=1) if (b + c) > 0 else 1.0
    is_significant = p_value < config.SIGNIFICANCE_THRESHOLD
    
    return {
        "test": "McNemar's test",
        "model_a": name_a, "model_b": name_b,
        "p_value": float(p_value), "significant": bool(is_significant)
    }


def run_all_statistical_tests(results, output_dir=None):
    """Run all statistical tests across all model pairs."""
    output_dir = output_dir or config.LOG_DIR
    all_tests = []

    # --- Paired t-tests on k-fold CV scores (same family, different strategy) ---
    t_test_pairs = [
        ("XLM_original_cv", "XLM_augmented_cv", "XLM-R (Original)", "XLM-R (Augmented)"),
        ("SVM_original_cv", "SVM_smote_cv", "SVM (Original)", "SVM (SMOTE)"),
        ("LR_original_cv", "LR_smote_cv", "LR (Original)", "LR (SMOTE)"),
    ]
    for key_a, key_b, name_a, name_b in t_test_pairs:
        if key_a in results and key_b in results:
            all_tests.append(paired_t_test(
                results[key_a]["fold_scores"],
                results[key_b]["fold_scores"],
                name_a, name_b
            ))

    # --- McNemar's tests on per-sample predictions (all pairwise comparisons) ---
    test_labels = results.get("test_labels")
    if test_labels:
        y_true = np.array(test_labels)

        # Collect all models that have predictions
        model_keys = [
            ("XLM_original", "XLM-RoBERTa (Original)"),
            ("XLM_augmented", "XLM-RoBERTa (Augmented)"),
            ("SVM_original", "SVM (Original)"),
            ("SVM_smote", "SVM (SMOTE)"),
            ("LR_original", "LR (Original)"),
            ("LR_smote", "LR (SMOTE)"),
        ]

        available = []
        for key, name in model_keys:
            preds = results.get(key, {}).get("predictions", [])
            if len(preds) == len(y_true):
                available.append((key, name, np.array(preds)))

        # Run McNemar's for every unique pair
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                _, name_a, preds_a = available[i]
                _, name_b, preds_b = available[j]
                all_tests.append(mcnemar_test(y_true, preds_a, preds_b, name_a, name_b))

    if all_tests:
        with open(os.path.join(output_dir, "statistical_tests.json"), "w") as f:
            json.dump(all_tests, f, indent=2)

    return all_tests
