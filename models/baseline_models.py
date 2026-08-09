"""
Baseline Models (Section 3.2.2).

Support Vector Machine (SVM) and Logistic Regression implementations.
Includes training, evaluation, cross-validation, and model persistence.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BaselineTrainer:
    def __init__(self):
        self.tfidf = None
        self.models = {}
        self.results = {}
        
    def fit_tfidf(self, texts: list[str]):
        """Fit TF-IDF vectorizer on training data."""
        self.tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        )
        self.tfidf.fit(texts)
        
        # Save vectorizer
        joblib.dump(self.tfidf, os.path.join(config.MODEL_DIR, "tfidf_vectorizer.joblib"))
        
    def train_svm(self, X_train, y_train, C=1.0, kernel="linear", dataset_name="original"):
        """Train SVM model."""
        print(f"  Training SVM ({kernel}, C={C}) on {dataset_name} data...")
        model = SVC(
            C=C, 
            kernel=kernel, 
            random_state=config.SEED,
            probability=True,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        
        key = f"svm_{dataset_name}"
        self.models[key] = model
        return model
        
    def train_logistic_regression(self, X_train, y_train, C=1.0, dataset_name="original"):
        """Train Logistic Regression model."""
        print(f"  Training Logistic Regression (C={C}) on {dataset_name} data...")
        model = LogisticRegression(
            C=C, 
            penalty="l2", 
            random_state=config.SEED,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        
        key = f"lr_{dataset_name}"
        self.models[key] = model
        return model

    def evaluate(self, model, X_test, y_test, model_name: str) -> dict:
        """Evaluate model on test set."""
        y_pred = model.predict(X_test)
        
        result = {
            "model": model_name,
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
            "predictions": y_pred.tolist()
        }
        
        self.results[model_name] = result
        return result
        
    def cross_validate(self, model_class, X, y, params: dict, model_name: str, k_folds=None):
        """Perform k-fold cross-validation."""
        k_folds = k_folds or config.K_FOLDS
        
        model = model_class(**params, random_state=config.SEED)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config.SEED)
        scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro", n_jobs=1)
        
        result = {
            "model": model_name,
            "cv_macro_f1_mean": float(scores.mean()),
            "cv_macro_f1_std": float(scores.std()),
            "fold_scores": scores.tolist()
        }
        
        self.results[f"{model_name}_cv"] = result
        return result
        
    def save_model(self, key: str):
        """Save a specific model."""
        if key in self.models:
            path = os.path.join(config.MODEL_DIR, f"{key}.joblib")
            joblib.dump(self.models[key], path)
            print(f"Saved model {key} -> {path}")
            
    def save_results(self):
        """Save evaluation results to CSV."""
        import pandas as pd
        rows = []
        for name, res in self.results.items():
            if "predictions" in res:
                rows.append({
                    "Model": res["model"],
                    "Macro-F1": res["macro_f1"],
                    "Precision": res["precision_macro"],
                    "Recall": res["recall_macro"],
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(config.LOG_DIR, "baseline_results.csv"), index=False)

if __name__ == "__main__":
    print("Baseline models module loaded.")
