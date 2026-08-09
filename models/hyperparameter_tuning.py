"""
Hyperparameter Tuning (Section 3.2.3).

Automated hyperparameter optimization using Optuna for XLM-RoBERTa
and grid search for baseline models (SVM, Logistic Regression).
"""

import os
import sys
import numpy as np
import pandas as pd
import optuna
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def tune_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 20,
    k_folds: int = None,
    seed: int = None,
) -> dict:
    """Tune SVM hyperparameters using Optuna."""
    k_folds = k_folds or config.K_FOLDS
    seed = seed or config.SEED

    def objective(trial):
        C = trial.suggest_float("C", 0.01, 100.0, log=True)
        kernel = trial.suggest_categorical("kernel", config.SVM_KERNEL_OPTIONS)

        model = SVC(
            C=C,
            kernel=kernel,
            random_state=seed,
            class_weight="balanced",
        )

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring="f1_macro",
            n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def tune_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 20,
    k_folds: int = None,
    seed: int = None,
) -> dict:
    """Tune Logistic Regression hyperparameters using Optuna."""
    k_folds = k_folds or config.K_FOLDS
    seed = seed or config.SEED

    def objective(trial):
        C = trial.suggest_float("C", 0.01, 100.0, log=True)

        model = LogisticRegression(
            C=C,
            penalty="l2",
            random_state=seed,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring="f1_macro",
            n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def tune_xlm_roberta(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 10,
    text_column: str = "text_clean",
    label_column: str = "label_encoded",
    seed: int = None,
) -> dict:
    """Tune XLM-RoBERTa hyperparameters using Optuna."""
    seed = seed or config.SEED

    def objective(trial):
        lr = trial.suggest_float("learning_rate", *config.XLM_LR_RANGE, log=True)
        batch_size = trial.suggest_categorical("batch_size", config.XLM_BATCH_SIZE_OPTIONS)
        weight_decay = trial.suggest_float("weight_decay", *config.XLM_WEIGHT_DECAY_RANGE)

        from models.xlm_roberta_model import XLMRoBERTaTrainer

        trainer = XLMRoBERTaTrainer(
            learning_rate=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            epochs=3,  # Short training for HPO
            patience=2,
            seed=seed,
        )

        history = trainer.train(
            train_df, val_df,
            save_dir=os.path.join(config.MODEL_DIR, f"hpo_trial_{trial.number}"),
        )

        best_f1 = max(history["val_macro_f1"]) if history["val_macro_f1"] else 0.0

        del trainer
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_f1

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    results_path = os.path.join(config.LOG_DIR, "hpo_results.csv")
    study.trials_dataframe().to_csv(results_path, index=False)
    
    return study.best_params


if __name__ == "__main__":
    config.set_seed()
    print("Hyperparameter Tuning Module")
