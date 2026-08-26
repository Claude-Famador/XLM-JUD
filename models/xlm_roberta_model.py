"""
XLM-RoBERTa Model Configuration (Section 3.2.1).

Implementation for fine-tuning XLM-RoBERTa on smishing datasets.
Includes custom trainer class with early stopping and PyTorch/HuggingFace integration.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_metrics(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions, average="macro")
    }


class XLMRoBERTaTrainer:
    def __init__(
        self,
        learning_rate: float = config.XLM_LEARNING_RATE,
        batch_size: int = config.XLM_BATCH_SIZE,
        weight_decay: float = config.XLM_WEIGHT_DECAY,
        epochs: int = config.XLM_EPOCHS,
        patience: int = config.XLM_EARLY_STOPPING_PATIENCE,
        label_smoothing: float = config.XLM_LABEL_SMOOTHING,
        dropout: float = config.XLM_DROPOUT,
        max_grad_norm: float = config.XLM_MAX_GRAD_NORM,
        warmup_ratio: float = config.XLM_WARMUP_RATIO,
        seed: int = config.SEED,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.seed = seed
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.XLM_MODEL_NAME)
        
        # Load model config and override dropout for stronger regularization
        model_config = AutoConfig.from_pretrained(
            config.XLM_MODEL_NAME,
            num_labels=2,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.XLM_MODEL_NAME,
            config=model_config,
        ).to(config.DEVICE)
        
        self.trainer = None
        
    def _prepare_dataset(self, df: pd.DataFrame, text_column="text_clean", label_column="label_encoded"):
        """Convert DataFrame to HF Dataset and tokenize."""
        dataset = Dataset.from_pandas(df[[text_column, label_column]])
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=config.XLM_MAX_LENGTH,
            )
            
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        # Rename label column for trainer
        tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")
        return tokenized_dataset
        
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, save_dir: str):
        """Train XLM-RoBERTa."""
        print(f"  Tokenizing datasets...")
        train_dataset = self._prepare_dataset(train_df)
        val_dataset = self._prepare_dataset(val_df)
        
        os.makedirs(save_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=save_dir,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            max_grad_norm=self.max_grad_norm,
            label_smoothing_factor=self.label_smoothing,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            seed=self.seed,
            report_to="none",  # disable wandb etc if not configured
            logging_dir=os.path.join(save_dir, "logs"),
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
        )
        
        print(f"  Starting training for {self.epochs} epochs...")
        train_result = self.trainer.train()
        
        # Save best model
        best_model_path = os.path.join(save_dir, "xlm_roberta_best")
        self.trainer.save_model(best_model_path)
        self.tokenizer.save_pretrained(best_model_path)
        print(f"  Best model saved to {best_model_path}")
        
        # Extract history
        history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
        for log in self.trainer.state.log_history:
            if "loss" in log and "eval_loss" not in log:
                history["train_loss"].append(log["loss"])
            if "eval_loss" in log:
                history["val_loss"].append(log["eval_loss"])
            if "eval_macro_f1" in log:
                history["val_macro_f1"].append(log["eval_macro_f1"])
                
        return history
        
    def load_best_model(self, model_path: str):
        """Load a saved model."""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(config.DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.trainer = Trainer(model=self.model)
        
    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Predict on raw texts."""
        dataset = Dataset.from_dict({"text": texts})
        
        def tokenize(examples):
            return self.tokenizer(examples["text"], padding=True, truncation=True, max_length=config.XLM_MAX_LENGTH)
            
        tokenized = dataset.map(tokenize, batched=True)
        predictions = self.trainer.predict(tokenized)
        
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = np.argmax(logits, axis=-1)
        
        return preds, probs

    def cross_validate(
        self,
        data_df: pd.DataFrame,
        k_folds: int = None,
        save_dir: str = None,
        cv_epochs: int = 3,
        cv_patience: int = 1,
    ) -> dict:
        """
        Perform stratified k-fold cross-validation (Section 3.3.3).

        Each fold re-initializes the model from the base checkpoint, trains on
        the fold's training portion, and evaluates Macro-F1 on the held-out
        portion.  Fold models are discarded after scoring.

        Args:
            data_df: DataFrame with 'text_clean' and 'label_encoded' columns
                     (typically train + val merged).
            k_folds: Number of folds (default: config.K_FOLDS).
            save_dir: Directory for temporary fold checkpoints.
            cv_epochs: Max epochs per fold (reduced for speed).
            cv_patience: Early-stopping patience per fold.

        Returns:
            dict with 'fold_scores', 'cv_macro_f1_mean', 'cv_macro_f1_std'.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score

        k_folds = k_folds or config.K_FOLDS
        save_dir = save_dir or os.path.join(config.MODEL_DIR, "cv_temp")
        os.makedirs(save_dir, exist_ok=True)

        labels = data_df["label_encoded"].values
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config.SEED)

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data_df, labels), 1):
            print(f"\n  === CV Fold {fold_idx}/{k_folds} ===")

            fold_train = data_df.iloc[train_idx].reset_index(drop=True)
            fold_val = data_df.iloc[val_idx].reset_index(drop=True)

            print(f"  Train: {len(fold_train)}, Val: {len(fold_val)}")

            # Re-initialize model from base checkpoint (fresh weights each fold)
            model_config = AutoConfig.from_pretrained(
                config.XLM_MODEL_NAME,
                num_labels=2,
                hidden_dropout_prob=self.dropout,
                attention_probs_dropout_prob=self.dropout,
            )
            fold_model = AutoModelForSequenceClassification.from_pretrained(
                config.XLM_MODEL_NAME,
                config=model_config,
            ).to(config.DEVICE)

            # Tokenize fold data
            fold_tokenizer = AutoTokenizer.from_pretrained(config.XLM_MODEL_NAME)
            
            train_dataset = Dataset.from_pandas(fold_train[["text_clean", "label_encoded"]])
            val_dataset = Dataset.from_pandas(fold_val[["text_clean", "label_encoded"]])
            
            def tokenize_fn(examples):
                return fold_tokenizer(
                    examples["text_clean"],
                    padding="max_length",
                    truncation=True,
                    max_length=config.XLM_MAX_LENGTH,
                )

            train_dataset = train_dataset.map(tokenize_fn, batched=True)
            train_dataset = train_dataset.rename_column("label_encoded", "labels")
            val_dataset = val_dataset.map(tokenize_fn, batched=True)
            val_dataset = val_dataset.rename_column("label_encoded", "labels")

            fold_dir = os.path.join(save_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=fold_dir,
                eval_strategy="steps",
                eval_steps=500,
                save_strategy="steps",
                save_steps=500,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=cv_epochs,
                weight_decay=self.weight_decay,
                warmup_ratio=self.warmup_ratio,
                max_grad_norm=self.max_grad_norm,
                label_smoothing_factor=self.label_smoothing,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=1,
                seed=self.seed,
                report_to="none",
                logging_dir=os.path.join(fold_dir, "logs"),
            )

            fold_trainer = Trainer(
                model=fold_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=cv_patience)],
            )

            fold_trainer.train()

            # Evaluate on fold validation set
            val_texts = fold_val["text_clean"].fillna("").tolist()
            val_labels = fold_val["label_encoded"].values

            pred_dataset = Dataset.from_dict({"text": val_texts})
            pred_dataset = pred_dataset.map(
                lambda ex: fold_tokenizer(ex["text"], padding=True, truncation=True, max_length=config.XLM_MAX_LENGTH),
                batched=True,
            )
            predictions = fold_trainer.predict(pred_dataset)
            y_pred = np.argmax(predictions.predictions, axis=-1)

            fold_f1 = f1_score(val_labels, y_pred, average="macro")
            fold_scores.append(fold_f1)
            print(f"  Fold {fold_idx} Macro-F1: {fold_f1:.4f}")

            # Cleanup GPU memory
            del fold_model, fold_trainer, fold_tokenizer
            del train_dataset, val_dataset, pred_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fold_scores = np.array(fold_scores)
        print(f"\n  CV Results: {fold_scores.mean():.4f} ± {fold_scores.std():.4f}")
        print(f"  Per-fold: {[f'{s:.4f}' for s in fold_scores]}")

        # Clean up temporary fold checkpoints
        import shutil
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)

        return {
            "fold_scores": fold_scores.tolist(),
            "cv_macro_f1_mean": float(fold_scores.mean()),
            "cv_macro_f1_std": float(fold_scores.std()),
        }
