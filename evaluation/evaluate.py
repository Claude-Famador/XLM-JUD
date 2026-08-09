"""
Model Evaluation Module (Section 3.3).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_metrics(y_true, y_pred, y_probs=None, model_name="model"):
    """Compute primary and additional metrics."""
    metrics = {
        "model": model_name,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1_ham": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_smishing": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

    if y_probs is not None:
        try:
            auc = roc_auc_score(y_true, y_probs[:, 1]) if y_probs.ndim == 2 else roc_auc_score(y_true, y_probs)
            metrics["auc_roc"] = float(auc)
        except:
            metrics["auc_roc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def evaluate_dialect_awareness(test_df, y_pred, y_probs=None, model_name="model"):
    """Measure dialect-awareness through stratified subset evaluation."""
    dialect_results = {}
    if "language" not in test_df.columns:
        return dialect_results

    y_true = test_df["label_encoded"].values
    dialect_results["overall"] = compute_metrics(y_true, y_pred, y_probs, f"{model_name}_overall")

    for lang in test_df["language"].dropna().unique():
        lang_mask = test_df["language"] == lang
        if lang_mask.sum() == 0:
            continue

        lang_y_true = y_true[lang_mask]
        lang_y_pred = y_pred[lang_mask]
        lang_y_probs = y_probs[lang_mask] if y_probs is not None else None

        if len(np.unique(lang_y_true)) < 2:
            dialect_results[lang] = {
                "model": f"{model_name}_{lang}",
                "n_samples": int(lang_mask.sum()),
                "macro_f1": float(f1_score(lang_y_true, lang_y_pred, average="macro", zero_division=0)),
            }
        else:
            dialect_results[lang] = compute_metrics(lang_y_true, lang_y_pred, lang_y_probs, f"{model_name}_{lang}")
            dialect_results[lang]["n_samples"] = int(lang_mask.sum())

    return dialect_results


def evaluate_model_comprehensive(test_df, y_pred, y_probs=None, model_name="model", output_dir=None):
    """Run full evaluation pipeline for a single model."""
    output_dir = output_dir or config.LOG_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = test_df["label_encoded"].values
    overall_metrics = compute_metrics(y_true, y_pred, y_probs, model_name)
    report = classification_report(y_true, y_pred, output_dict=True)
    dialect_metrics = evaluate_dialect_awareness(test_df, y_pred, y_probs, model_name)

    results = {
        "model": model_name,
        "overall": overall_metrics,
        "classification_report": report,
        "dialect_results": dialect_metrics,
    }

    with open(os.path.join(output_dir, f"eval_{model_name}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def compare_models(results_list, output_dir=None):
    """Create a comparison table of all model results."""
    output_dir = output_dir or config.LOG_DIR
    rows = []
    
    for result in results_list:
        overall = result.get("overall", {})
        rows.append({
            "Model": result.get("model", "unknown"),
            "Macro-F1": overall.get("macro_f1"),
            "Precision": overall.get("precision_macro"),
            "Recall": overall.get("recall_macro"),
            "MCC": overall.get("mcc"),
            "AUC-ROC": overall.get("auc_roc"),
        })

    df = pd.DataFrame(rows).sort_values("Macro-F1", ascending=False)
    df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    return df
