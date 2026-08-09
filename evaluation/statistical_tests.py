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
    """Paired t-test on k-fold results."""
    scores_a, scores_b = np.array(scores_a), np.array(scores_b)
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
    """Run all statistical tests."""
    output_dir = output_dir or config.LOG_DIR
    all_tests = []

    if "XLM_original_cv" in results and "XLM_augmented_cv" in results:
        all_tests.append(paired_t_test(
            results["XLM_original_cv"]["fold_scores"],
            results["XLM_augmented_cv"]["fold_scores"],
            "XLM-R (Original)", "XLM-R (Augmented)"
        ))

    if "SVM_original_cv" in results and "SVM_smote_cv" in results:
        all_tests.append(paired_t_test(
            results["SVM_original_cv"]["fold_scores"],
            results["SVM_smote_cv"]["fold_scores"],
            "SVM (Original)", "SVM (SMOTE)"
        ))

    test_labels = results.get("test_labels")
    if test_labels:
        y_true = np.array(test_labels)
        if "XLM_original" in results and "SVM_original" in results:
            all_tests.append(mcnemar_test(
                y_true, 
                np.array(results["XLM_original"].get("predictions", [])), 
                np.array(results["SVM_original"].get("predictions", [])),
                "XLM-RoBERTa", "SVM"
            ))

    if all_tests:
        with open(os.path.join(output_dir, "statistical_tests.json"), "w") as f:
            json.dump(all_tests, f, indent=2)

    return all_tests
