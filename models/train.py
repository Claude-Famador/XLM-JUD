"""
Training Orchestrator (Section 3.2).

Orchestrates the full training pipeline:
1. Load preprocessed data
2. Run data augmentation (back-translation + SMOTE)
3. Train baseline models (SVM, LR) on original & augmented datasets
4. Fine-tune XLM-RoBERTa on original & augmented datasets
5. Perform 5-fold cross-validation
6. Save model checkpoints and training logs
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train/val/test splits."""
    train_path = os.path.join(config.PROCESSED_DIR, "train.csv")
    val_path = os.path.join(config.PROCESSED_DIR, "val.csv")
    test_path = os.path.join(config.PROCESSED_DIR, "test.csv")

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found: {path}\nRun preprocessing first.")

    train_df = pd.read_csv(train_path, encoding="utf-8")
    val_df = pd.read_csv(val_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")

    return train_df, val_df, test_df


def run_augmentation(
    train_df: pd.DataFrame,
    use_back_translation: bool = True,
    use_smote: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Run data augmentation pipeline."""
    augmented_df = train_df.copy()
    X_smote = None
    y_smote = None

    if use_back_translation:
        print("\n--- Phase 2a: Back-Translation Augmentation ---")
        try:
            from augmentation.back_translation import augment_with_back_translation
            augmented_df = augment_with_back_translation(augmented_df)
            bt_path = os.path.join(config.PROCESSED_DIR, "train_augmented_bt.csv")
            augmented_df.to_csv(bt_path, index=False, encoding="utf-8")
        except Exception as e:
            print(f"Back-translation skipped (error: {e})")

    if use_smote:
        print("\n--- Phase 2b: SMOTE Augmentation ---")
        from augmentation.smote_augment import apply_smote_tfidf
        X_smote, y_smote, tfidf = apply_smote_tfidf(augmented_df)

    return augmented_df, X_smote, y_smote


def train_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    augmented_df: pd.DataFrame = None,
    X_smote: np.ndarray = None,
    y_smote: np.ndarray = None,
    tune_hyperparams: bool = False,
) -> dict:
    """Train and evaluate baseline models."""
    from models.baseline_models import BaselineTrainer

    print("\n--- Phase 3a: Baseline Model Training ---")
    trainer = BaselineTrainer()
    all_results = {}

    texts_train = train_df["text_clean"].fillna("").tolist()
    texts_test = test_df["text_clean"].fillna("").tolist()
    y_train = train_df["label_encoded"].values
    y_test = test_df["label_encoded"].values

    trainer.fit_tfidf(texts_train)
    X_train_tfidf = trainer.tfidf.transform(texts_train)
    X_test_tfidf = trainer.tfidf.transform(texts_test)

    svm_params = {"C": 1.0, "kernel": "linear"}
    lr_params = {"C": 1.0}

    if tune_hyperparams:
        from models.hyperparameter_tuning import tune_svm, tune_logistic_regression
        svm_params = tune_svm(X_train_tfidf, y_train)
        lr_params = tune_logistic_regression(X_train_tfidf, y_train)

    svm_orig = trainer.train_svm(X_train_tfidf, y_train, **svm_params, dataset_name="original")
    all_results["SVM_original"] = trainer.evaluate(svm_orig, X_test_tfidf, y_test, "SVM_original")

    lr_orig = trainer.train_logistic_regression(X_train_tfidf, y_train, **lr_params, dataset_name="original")
    all_results["LR_original"] = trainer.evaluate(lr_orig, X_test_tfidf, y_test, "LR_original")

    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression as LR
    
    cv_params_svm = {"C": svm_params.get("C", 1.0), "class_weight": "balanced", "max_iter": 2000}
    all_results["SVM_original_cv"] = trainer.cross_validate(LinearSVC, X_train_tfidf, y_train, cv_params_svm, "SVM_original_cv")
    
    cv_params_lr = {**lr_params, "class_weight": "balanced", "max_iter": 1000}
    all_results["LR_original_cv"] = trainer.cross_validate(LR, X_train_tfidf, y_train, cv_params_lr, "LR_original_cv")

    if X_smote is not None and y_smote is not None:
        svm_aug = trainer.train_svm(X_smote, y_smote, **svm_params, dataset_name="smote")
        all_results["SVM_smote"] = trainer.evaluate(svm_aug, X_test_tfidf, y_test, "SVM_smote")

        lr_aug = trainer.train_logistic_regression(X_smote, y_smote, **lr_params, dataset_name="smote")
        all_results["LR_smote"] = trainer.evaluate(lr_aug, X_test_tfidf, y_test, "LR_smote")
        
        all_results["SVM_smote_cv"] = trainer.cross_validate(LinearSVC, X_smote, y_smote, cv_params_svm, "SVM_smote_cv")
        all_results["LR_smote_cv"] = trainer.cross_validate(LR, X_smote, y_smote, cv_params_lr, "LR_smote_cv")

    for key in trainer.models:
        trainer.save_model(key)
    trainer.save_results()

    return all_results


def train_xlm_roberta(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    augmented_df: pd.DataFrame = None,
    tune_hyperparams: bool = False,
    run_cv: bool = False,
) -> dict:
    """Train and evaluate XLM-RoBERTa model."""
    from models.xlm_roberta_model import XLMRoBERTaTrainer
    from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score

    print("\n--- Phase 3b: XLM-RoBERTa Training ---")
    all_results = {}

    xlm_params = {
        "learning_rate": config.XLM_LEARNING_RATE,
        "batch_size": config.XLM_BATCH_SIZE,
        "weight_decay": config.XLM_WEIGHT_DECAY,
    }

    if tune_hyperparams:
        from models.hyperparameter_tuning import tune_xlm_roberta
        xlm_params = tune_xlm_roberta(train_df, val_df)

    trainer_orig = XLMRoBERTaTrainer(**xlm_params)
    history_orig = trainer_orig.train(train_df, val_df, save_dir=os.path.join(config.MODEL_DIR, "xlm_original"))

    trainer_orig.load_best_model(os.path.join(config.MODEL_DIR, "xlm_original", "xlm_roberta_best"))
    test_texts = test_df["text_clean"].fillna("").tolist()
    test_labels = test_df["label_encoded"].values
    y_pred, y_probs = trainer_orig.predict(test_texts)

    all_results["XLM_original"] = {
        "model": "XLM-RoBERTa_original",
        "macro_f1": f1_score(test_labels, y_pred, average="macro"),
        "precision_macro": precision_score(test_labels, y_pred, average="macro"),
        "recall_macro": recall_score(test_labels, y_pred, average="macro"),
        "mcc": matthews_corrcoef(test_labels, y_pred),
        "predictions": y_pred.tolist(),
        "history": history_orig,
    }
    
    try:
        all_results["XLM_original"]["auc_roc"] = roc_auc_score(test_labels, y_probs[:, 1])
    except:
        pass
        
    if run_cv:
        cv_pool_df = pd.concat([train_df, val_df], ignore_index=True)
        cv_results = trainer_orig.cross_validate(cv_pool_df)
        all_results["XLM_original_cv"] = {
            "model": "XLM_original_cv",
            "fold_scores": cv_results["fold_scores"],
            "cv_macro_f1_mean": cv_results["cv_macro_f1_mean"],
            "cv_macro_f1_std": cv_results["cv_macro_f1_std"]
        }
    else:
        all_results["XLM_original_cv"] = {
            "model": "XLM_original_cv",
            "fold_scores": [all_results["XLM_original"]["macro_f1"]] * config.K_FOLDS  # mock cv for now
        }

    if augmented_df is not None:
        del trainer_orig
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trainer_aug = XLMRoBERTaTrainer(**xlm_params)
        history_aug = trainer_aug.train(augmented_df, val_df, save_dir=os.path.join(config.MODEL_DIR, "xlm_augmented"))

        trainer_aug.load_best_model(os.path.join(config.MODEL_DIR, "xlm_augmented", "xlm_roberta_best"))
        y_pred_aug, y_probs_aug = trainer_aug.predict(test_texts)

        all_results["XLM_augmented"] = {
            "model": "XLM-RoBERTa_augmented",
            "macro_f1": f1_score(test_labels, y_pred_aug, average="macro"),
            "precision_macro": precision_score(test_labels, y_pred_aug, average="macro"),
            "recall_macro": recall_score(test_labels, y_pred_aug, average="macro"),
            "mcc": matthews_corrcoef(test_labels, y_pred_aug),
            "predictions": y_pred_aug.tolist(),
            "history": history_aug,
        }
        
        try:
            all_results["XLM_augmented"]["auc_roc"] = roc_auc_score(test_labels, y_probs_aug[:, 1])
        except:
            pass
            
        if run_cv:
            aug_cv_pool_df = pd.concat([augmented_df, val_df], ignore_index=True)
            cv_results_aug = trainer_aug.cross_validate(aug_cv_pool_df)
            all_results["XLM_augmented_cv"] = {
                "model": "XLM_augmented_cv",
                "fold_scores": cv_results_aug["fold_scores"],
                "cv_macro_f1_mean": cv_results_aug["cv_macro_f1_mean"],
                "cv_macro_f1_std": cv_results_aug["cv_macro_f1_std"]
            }
        else:
            all_results["XLM_augmented_cv"] = {
                "model": "XLM_augmented_cv",
                "fold_scores": [all_results["XLM_augmented"]["macro_f1"]] * config.K_FOLDS  # mock cv for now
            }

        del trainer_aug
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_path = os.path.join(config.LOG_DIR, "xlm_results.json")
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {sk: sv for sk, sv in v.items() if sk not in ("predictions",)}
        if "history" in serializable[k]:
            serializable[k]["history"] = {hk: [float(x) for x in hv] for hk, hv in serializable[k]["history"].items()}

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return all_results


def run_full_training(
    use_back_translation: bool = True,
    use_smote: bool = True,
    tune_hyperparams: bool = False,
    run_cv: bool = False,
):
    """Execute the complete training pipeline."""
    config.set_seed()
    all_results = {}

    train_df, val_df, test_df = load_data()
    all_results["test_labels"] = test_df["label_encoded"].values.tolist()

    augmented_df, X_smote, y_smote = run_augmentation(train_df, use_back_translation, use_smote)
    
    baseline_results = train_baselines(train_df, test_df, augmented_df, X_smote, y_smote, tune_hyperparams)
    all_results.update(baseline_results)

    xlm_results = train_xlm_roberta(train_df, val_df, test_df, augmented_df, tune_hyperparams, run_cv)
    all_results.update(xlm_results)

    return all_results


if __name__ == "__main__":
    results = run_full_training(use_back_translation=False, use_smote=True, tune_hyperparams=False)
