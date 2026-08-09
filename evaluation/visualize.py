"""
Visualization Module (Section 3.3).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_training_history(history, model_name="XLM-RoBERTa", output_dir=None):
    output_dir = output_dir or config.FIGURE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if "train_loss" in history and history["train_loss"]:
        train_x = np.linspace(1, len(history.get("val_loss", [1])), len(history["train_loss"]))
        ax1.plot(train_x, history["train_loss"], label="Train Loss")
        
    if "val_loss" in history and history["val_loss"]:
        val_x = range(1, len(history["val_loss"]) + 1)
        ax1.plot(val_x, history["val_loss"], label="Val Loss")
        
    ax1.legend()
    ax1.set_title(f"{model_name} Loss")
    
    if "val_macro_f1" in history and history["val_macro_f1"]:
        val_x = range(1, len(history["val_macro_f1"]) + 1)
        ax2.plot(val_x, history["val_macro_f1"], label="Val Macro-F1")
        ax2.legend()
        
    ax2.set_title(f"{model_name} Macro-F1")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"history_{model_name}.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name="Model", output_dir=None):
    output_dir = output_dir or config.FIGURE_DIR
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(os.path.join(output_dir, f"cm_{model_name}.png"))
    plt.close()


def generate_all_visualizations(results, test_df=None, output_dir=None):
    output_dir = output_dir or config.FIGURE_DIR
    
    for key in ["XLM_original", "XLM_augmented"]:
        if key in results and "history" in results[key]:
            plot_training_history(results[key]["history"], results[key].get("model", key), output_dir)
            
    if test_df is not None:
        y_true = test_df["label_encoded"].values
        for key, res in results.items():
            if isinstance(res, dict) and "predictions" in res:
                plot_confusion_matrix(y_true, np.array(res["predictions"]), res.get("model", key), output_dir)
