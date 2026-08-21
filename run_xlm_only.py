"""
Run XLM-RoBERTa training only (5 epochs) — skips slow baseline models.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.set_seed()

print("=" * 70)
print("  XLM-RoBERTa TRAINING (5 EPOCHS)")
print(f"  Device: {config.DEVICE}")
print(f"  Epochs: {config.XLM_EPOCHS}")
print(f"  Batch size: {config.XLM_BATCH_SIZE}")
print(f"  Learning rate: {config.XLM_LEARNING_RATE}")
print("=" * 70)

start_time = time.time()

# 1. Load data
print("\n[1/4] Loading preprocessed data...")
from models.train import load_data
train_df, val_df, test_df = load_data()
print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 2. Train XLM-RoBERTa on original data
print("\n[2/4] Training XLM-RoBERTa on original data...")
from models.xlm_roberta_model import XLMRoBERTaTrainer
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, classification_report

trainer = XLMRoBERTaTrainer()
save_dir = os.path.join(config.MODEL_DIR, "xlm_original")
history = trainer.train(train_df, val_df, save_dir=save_dir)

# 3. Evaluate
print("\n[3/4] Evaluating on test set...")
trainer.load_best_model(os.path.join(save_dir, "xlm_roberta_best"))
test_texts = test_df["text_clean"].fillna("").tolist()
test_labels = test_df["label_encoded"].values

y_pred, y_probs = trainer.predict(test_texts)

macro_f1 = f1_score(test_labels, y_pred, average="macro")
precision = precision_score(test_labels, y_pred, average="macro")
recall = recall_score(test_labels, y_pred, average="macro")
mcc = matthews_corrcoef(test_labels, y_pred)

print(f"\n  Macro F1:    {macro_f1:.4f}")
print(f"  Precision:   {precision:.4f}")
print(f"  Recall:      {recall:.4f}")
print(f"  MCC:         {mcc:.4f}")

try:
    auc = roc_auc_score(test_labels, y_probs[:, 1])
    print(f"  AUC-ROC:     {auc:.4f}")
except:
    auc = None

print(f"\n  Classification Report:")
print(classification_report(test_labels, y_pred, target_names=["ham", "smishing"]))

# 4. Save results
print("[4/4] Saving results...")
results = {
    "XLM_original": {
        "model": "XLM-RoBERTa_original",
        "macro_f1": macro_f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "mcc": mcc,
        "epochs": config.XLM_EPOCHS,
        "history": {k: [float(x) for x in v] for k, v in history.items()},
    }
}
if auc is not None:
    results["XLM_original"]["auc_roc"] = auc

results_path = os.path.join(config.LOG_DIR, "xlm_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to: {results_path}")

elapsed = time.time() - start_time
print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
