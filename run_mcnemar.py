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


def load_baseline_predictions(model_name, y_true):
    """
    Load predictions for a baseline model (SVM or LR, original or SMOTE).
    Tries saved model first; falls back to confusion matrix reconstruction.
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    tfidf_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    test_csv   = os.path.join(PROCESSED_DIR, "test.csv")

    if os.path.exists(model_path) and os.path.exists(tfidf_path) and os.path.exists(test_csv):
        print(f"  Loading saved {model_name} model and TF-IDF vectorizer...")
        model      = joblib.load(model_path)
        vectorizer = joblib.load(tfidf_path)
        test_df    = pd.read_csv(test_csv)

        text_col = "text_clean" if "text_clean" in test_df.columns else "text"
        X_test   = vectorizer.transform(test_df[text_col].fillna("").tolist())
        y_pred   = model.predict(X_test)
        print(f"  {model_name} predictions obtained: {len(y_pred)} samples")
        return np.array(y_pred)

    # Fallback: reconstruct from eval JSON confusion matrix
    eval_name = model_name.upper().replace("_", "_")
    eval_path = os.path.join(LOG_DIR, f"eval_{eval_name}.json")
    if not os.path.exists(eval_path):
        print(f"  WARNING: Neither {model_name} model nor eval JSON found. Skipping.")
        return None

    print(f"  Reconstructing {model_name} predictions from confusion matrix (approximate)...")
    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    cm = eval_data["overall"]["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    ham_idx      = np.where(y_true == 0)[0]
    smishing_idx = np.where(y_true == 1)[0]

    y_pred = np.zeros(len(y_true), dtype=int)
    y_pred[ham_idx[:tn]]           = 0
    y_pred[ham_idx[tn:tn+fp]]      = 1
    y_pred[smishing_idx[:tp]]      = 1
    y_pred[smishing_idx[tp:tp+fn]] = 0

    print(f"  Reconstructed {model_name} predictions: {len(y_pred)} samples")
    print(f"  (TN={tn}, FP={fp}, FN={fn}, TP={tp})")
    return y_pred


def main():
    print("=" * 60)
    print("  McNemar's Test: All Pairwise Model Comparisons")
    print("=" * 60)

    # Load ground truth and XLM predictions
    y_true, y_pred_xlm_orig, test_df = load_xlm_predictions()

    # Load XLM augmented predictions
    pred_path = os.path.join(LOG_DIR, "xlm_predictions.json")
    with open(pred_path, "r") as f:
        xlm_data = json.load(f)

    y_pred_xlm_aug = None
    xlm_aug = xlm_data.get("XLM_augmented", {})
    if "predictions" in xlm_aug:
        y_pred_xlm_aug = np.array(xlm_aug["predictions"])
        print(f"  Loaded XLM_augmented predictions: {len(y_pred_xlm_aug)} samples")

    # Load all baseline predictions
    y_pred_svm_orig = load_baseline_predictions("svm_original", y_true)
    y_pred_svm_smote = load_baseline_predictions("svm_smote", y_true)
    y_pred_lr_orig = load_baseline_predictions("lr_original", y_true)
    y_pred_lr_smote = load_baseline_predictions("lr_smote", y_true)

    # Build list of available models
    all_models = [
        ("XLM-RoBERTa (Original)", y_pred_xlm_orig),
        ("XLM-RoBERTa (Augmented)", y_pred_xlm_aug),
        ("SVM (Original)", y_pred_svm_orig),
        ("SVM (SMOTE)", y_pred_svm_smote),
        ("LR (Original)", y_pred_lr_orig),
        ("LR (SMOTE)", y_pred_lr_smote),
    ]

    # Filter to models that have valid predictions
    available = [(name, preds) for name, preds in all_models
                 if preds is not None and len(preds) == len(y_true)]

    print(f"\n  Available models for comparison: {len(available)}")
    for name, preds in available:
        print(f"    - {name} ({len(preds)} predictions)")

    # Run McNemar's test for every unique pair
    mcnemar_results = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            name_a, preds_a = available[i]
            name_b, preds_b = available[j]

            result = mcnemar_test(y_true, preds_a, preds_b,
                                  name_a=name_a, name_b=name_b)
            mcnemar_results.append(result)

            # Print individual result
            print(f"\n  {name_a} vs {name_b}:")
            print(f"    Chi2 = {result['chi2_statistic']}, p = {result['p_value']}, "
                  f"significant = {result['significant']}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {len(mcnemar_results)} McNemar's Tests")
    print(f"{'='*60}")
    sig_count = sum(1 for r in mcnemar_results if r["significant"])
    print(f"  Significant: {sig_count}/{len(mcnemar_results)} (alpha={SIGNIFICANCE_THRESHOLD})")

    for r in mcnemar_results:
        marker = "***" if r["significant"] else "   "
        print(f"  {marker} {r['model_a']} vs {r['model_b']}: "
              f"p={r['p_value']:.6f}")
    print(f"{'='*60}\n")

    # Save results — merge with existing statistical_tests.json
    stat_path = os.path.join(LOG_DIR, "statistical_tests.json")
    existing = []
    if os.path.exists(stat_path):
        with open(stat_path, "r") as f:
            existing = json.load(f)

    # Remove any prior McNemar entries
    existing = [e for e in existing if e.get("test") != "McNemar's test (continuity-corrected)"]
    existing.extend(mcnemar_results)

    with open(stat_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Results saved to: {stat_path}")


if __name__ == "__main__":
    main()

