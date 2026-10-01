"""
Run McNemar's Test: XLM-RoBERTa (Original) vs SVM (Original)
Uses xlm_predictions.json (ground truth + XLM preds) and
re-runs SVM on the same test set to obtain its predictions.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

SIGNIFICANCE_THRESHOLD = config.SIGNIFICANCE_THRESHOLD
LOG_DIR = config.LOG_DIR
MODEL_DIR = config.MODEL_DIR
PROCESSED_DIR = config.PROCESSED_DIR


def mcnemar_test(y_true, y_pred_a, y_pred_b, name_a="Model A", name_b="Model B"):
    """McNemar's test with continuity correction (Edwards' version)."""
    y_true  = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b = A correct, B wrong  |  c = A wrong, B correct
    b = int(np.sum( correct_a & ~correct_b))
    c = int(np.sum(~correct_a &  correct_b))
    n_discordant = b + c

    print(f"\n  Contingency (discordant pairs):")
    print(f"    {name_a} correct, {name_b} wrong  (b) = {b}")
    print(f"    {name_a} wrong,   {name_b} correct (c) = {c}")
    print(f"    Total discordant pairs        = {n_discordant}")

    if n_discordant == 0:
        chi2, p_value = 0.0, 1.0
    else:
        # Edwards' continuity-corrected McNemar statistic
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = float(1 - stats.chi2.cdf(chi2, df=1))

    is_significant = p_value < SIGNIFICANCE_THRESHOLD

    return {
        "test": "McNemar's test (continuity-corrected)",
        "model_a": name_a,
        "model_b": name_b,
        "b_a_correct_b_wrong": b,
        "c_a_wrong_b_correct": c,
        "n_discordant": n_discordant,
        "chi2_statistic": round(chi2, 6),
        "p_value": round(p_value, 6),
        "significant": is_significant,
        "alpha": SIGNIFICANCE_THRESHOLD,
        "interpretation": (
            f"{name_a} and {name_b} make significantly different errors (p={p_value:.4f} < {SIGNIFICANCE_THRESHOLD})"
            if is_significant else
            f"No significant difference in error patterns between {name_a} and {name_b} (p={p_value:.4f} >= {SIGNIFICANCE_THRESHOLD})"
        )
    }


def load_xlm_predictions():
    """Load XLM predictions from saved JSON, and y_true from test.csv."""
    # Load ground truth from test CSV
    test_csv = os.path.join(PROCESSED_DIR, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found at: {test_csv}")
    test_df = pd.read_csv(test_csv)
    label_col = "label_encoded" if "label_encoded" in test_df.columns else "label"
    y_true = test_df[label_col].values
    print(f"  Loaded ground truth from test.csv: {len(y_true)} samples")

    # Load XLM predictions
    pred_path = os.path.join(LOG_DIR, "xlm_predictions.json")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"XLM predictions not found at: {pred_path}")
    with open(pred_path, "r") as f:
        data = json.load(f)

    xlm_orig = data.get("XLM_original", {})
    y_pred_xlm = xlm_orig.get("predictions")
    if y_pred_xlm is None:
        raise KeyError("'predictions' missing from XLM_original in xlm_predictions.json")

    print(f"  Loaded XLM_original predictions: {len(y_pred_xlm)} samples")
    return np.array(y_true), np.array(y_pred_xlm), test_df


def load_svm_predictions(y_true):
    """
    Load or reconstruct SVM predictions.
    Tries saved model first; falls back to re-predicting from test CSV.
    """
    # --- Try loading saved SVM model + test data ---
    svm_path = os.path.join(MODEL_DIR, "SVM_ORIGINAL.joblib")
    tfidf_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    test_csv   = os.path.join(PROCESSED_DIR, "test.csv")

    if os.path.exists(svm_path) and os.path.exists(tfidf_path) and os.path.exists(test_csv):
        print("  Loading saved SVM model and TF-IDF vectorizer...")
        svm_model  = joblib.load(svm_path)
        vectorizer = joblib.load(tfidf_path)
        test_df    = pd.read_csv(test_csv)

        text_col = "text_clean" if "text_clean" in test_df.columns else "text"
        X_test   = vectorizer.transform(test_df[text_col].fillna("").tolist())
        y_pred_svm = svm_model.predict(X_test)
        print(f"  SVM predictions obtained from saved model: {len(y_pred_svm)} samples")
        return np.array(y_pred_svm)

    # --- Fallback: reconstruct from eval JSON confusion matrix ---
    print("  WARNING: SVM model or test CSV not found.")
    print("  Reconstructing SVM predictions from confusion matrix (approximate)...")
    eval_path = os.path.join(LOG_DIR, "eval_SVM_ORIGINAL.json")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Neither SVM model nor eval JSON found. Cannot run McNemar's test.")

    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    cm = eval_data["overall"]["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    # Reconstruct predictions to match the confusion matrix exactly.
    # We align them with y_true ordering: place TN/FP in ham positions, TP/FN in smishing positions.
    ham_idx      = np.where(y_true == 0)[0]
    smishing_idx = np.where(y_true == 1)[0]

    y_pred_svm = np.zeros(len(y_true), dtype=int)

    # Ham positions: TN correctly predicted as 0, FP wrongly predicted as 1
    y_pred_svm[ham_idx[:tn]]           = 0  # True Negatives
    y_pred_svm[ham_idx[tn:tn+fp]]      = 1  # False Positives

    # Smishing positions: TP correctly predicted as 1, FN wrongly predicted as 0
    y_pred_svm[smishing_idx[:tp]]      = 1  # True Positives
    y_pred_svm[smishing_idx[tp:tp+fn]] = 0  # False Negatives

    print(f"  Reconstructed SVM predictions: {len(y_pred_svm)} samples")
    print(f"  (TN={tn}, FP={fp}, FN={fn}, TP={tp})")
    return y_pred_svm


def main():
    print("=" * 60)
    print("  McNemar's Test: XLM-RoBERTa vs SVM")
    print("=" * 60)

    # Load data
    y_true, y_pred_xlm, test_df = load_xlm_predictions()
    y_pred_svm = load_svm_predictions(y_true)

    assert len(y_true) == len(y_pred_xlm) == len(y_pred_svm), (
        f"Length mismatch: y_true={len(y_true)}, XLM={len(y_pred_xlm)}, SVM={len(y_pred_svm)}"
    )

    # Run McNemar's test
    result = mcnemar_test(y_true, y_pred_xlm, y_pred_svm,
                          name_a="XLM-RoBERTa (Original)",
                          name_b="SVM (Original)")

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Chi-squared statistic : {result['chi2_statistic']}")
    print(f"  p-value               : {result['p_value']}")
    print(f"  Significant (alpha=0.05) : {result['significant']}")
    print(f"\n  Interpretation:")
    print(f"  {result['interpretation']}")
    print(f"{'='*60}\n")

    # Append result to statistical_tests.json
    stat_path = os.path.join(LOG_DIR, "statistical_tests.json")
    existing = []
    if os.path.exists(stat_path):
        with open(stat_path, "r") as f:
            existing = json.load(f)

    # Remove any prior McNemar entry
    existing = [e for e in existing if e.get("test") != "McNemar's test (continuity-corrected)"]
    existing.append(result)

    with open(stat_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Results appended to: {stat_path}")


if __name__ == "__main__":
    main()
