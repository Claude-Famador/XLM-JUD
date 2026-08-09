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
        seed: int = config.SEED,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.XLM_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.XLM_MODEL_NAME, 
            num_labels=2
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
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
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
