"""
Central Configuration for Dialect-Aware Smishing Detection Pipeline.

All hyperparameters, paths, random seeds, and model settings are defined here
to ensure reproducibility and ease of experimentation.
"""

import os
import torch
import numpy as np
import random

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# Create directories if they don't exist
for d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, LOG_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# Random Seed — Fixed for Reproducibility (Section 3.2.4)
# =============================================================================
SEED = 42

# =============================================================================
# Data Configuration (Section 3.1)
# =============================================================================
# Dataset paths
RAW_DATASET_PATH = os.path.join(DATA_DIR, "sample_dataset.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Stratified split ratios (Section 3.1.3)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Class labels
LABEL_MAP = {"ham": 0, "smishing": 1}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

# FLORES-200 languages (Section 3.1.1)
FLORES_LANGUAGES = ["ceb_Latn", "ilo_Latn"]
FLORES_SPLIT = "devtest"  # 997 sentences per language

# =============================================================================
# Preprocessing Configuration (Section 3.1.2)
# =============================================================================
# Entity masking placeholder tokens
ENTITY_PLACEHOLDERS = {
    "PERSON": "<USER>",
    "GPE": "<LOC>",
    "LOC": "<LOC>",
    "ORG": "<ORG>",
    "FAC": "<LOC>",
}

# Conservative cleaning — patterns to REMOVE (non-linguistic noise)
HTML_TAG_PATTERN = r"<[^>]+>"
URL_PATTERN = r"https?://\S+|www\.\S+"
METADATA_PATTERN = r"\[.*?\]"

# =============================================================================
# Augmentation Configuration (Section 3.1.2 & 3.2.2)
# =============================================================================
# Back-translation model (NLLB-200)
BACKTRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
BACKTRANSLATION_NUM_VARIANTS = 1  # Number of back-translated variants per sample

# Target dialect codes for NLLB-200
DIALECT_TARGETS = ["ilo_Latn", "ceb_Latn", "tgl_Latn"]
ENGLISH_CODE = "eng_Latn"

# SMOTE configuration
SMOTE_SAMPLING_STRATEGY = "auto"  # Balance to ~1:1
SMOTE_K_NEIGHBORS = 5

# =============================================================================
# XLM-RoBERTa Configuration (Section 3.2.1 & 3.2.2)
# =============================================================================
XLM_MODEL_NAME = "xlm-roberta-base"
XLM_MAX_LENGTH = 128  # Max token length for SMS messages

# Training hyperparameters (Section 3.2.3)
XLM_EPOCHS = 10
XLM_BATCH_SIZE = 16
XLM_LEARNING_RATE = 2e-5
XLM_WEIGHT_DECAY = 0.01
XLM_WARMUP_RATIO = 0.1
XLM_EARLY_STOPPING_PATIENCE = 3  # Stop if val loss doesn't improve for 3 epochs

# Hyperparameter search ranges (Section 3.2.3)
XLM_LR_RANGE = (1e-5, 5e-5)
XLM_BATCH_SIZE_OPTIONS = [8, 16, 32]
XLM_WEIGHT_DECAY_RANGE = (0.0, 0.1)

# =============================================================================
# Baseline Model Configuration (Section 3.2.2)
# =============================================================================
# TF-IDF settings
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 3)  # Unigrams to trigrams
TFIDF_SUBLINEAR_TF = True

# SVM hyperparameter ranges
SVM_C_RANGE = [0.01, 0.1, 1.0, 10.0, 100.0]
SVM_KERNEL_OPTIONS = ["linear", "rbf"]

# Logistic Regression hyperparameter ranges
LR_C_RANGE = [0.01, 0.1, 1.0, 10.0, 100.0]
LR_PENALTY_OPTIONS = ["l2"]

# =============================================================================
# Cross-Validation (Section 3.3.3)
# =============================================================================
K_FOLDS = 5

# =============================================================================
# Statistical Testing (Section 3.3.4)
# =============================================================================
SIGNIFICANCE_THRESHOLD = 0.05  # α = 0.05

# =============================================================================
# Device Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED):
    """
    Set random seeds across all libraries for deterministic training.
    Enforces reproducibility as described in Section 3.2.4.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
