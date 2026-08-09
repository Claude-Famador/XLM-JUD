import os
import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def apply_smote_tfidf(
    train_df: pd.DataFrame,
    text_column: str = "text_clean",
    label_column: str = "label_encoded"
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Apply TF-IDF vectorization and SMOTE to the training data.
    
    Returns:
        X_resampled, y_resampled, fitted_tfidf_vectorizer
    """
    print(f"\nApplying SMOTE Augmentation")
    
    texts = train_df[text_column].fillna("").tolist()
    labels = train_df[label_column].values
    
    print(f"  Original class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 1. TF-IDF Vectorization
    print("  Fitting TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
    )
    X_tfidf = tfidf.fit_transform(texts)
    
    # 2. SMOTE Oversampling
    print("  Running SMOTE...")
    smote = SMOTE(
        sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
        k_neighbors=config.SMOTE_K_NEIGHBORS,
        random_state=config.SEED
    )
    
    X_res, y_res = smote.fit_resample(X_tfidf, labels)
    
    print(f"  SMOTE class distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    
    return X_res, y_res, tfidf


if __name__ == "__main__":
    config.set_seed()
    train_path = os.path.join(config.PROCESSED_DIR, "train.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path, encoding="utf-8")
        X, y, vec = apply_smote_tfidf(df)
        print(f"Resulting shape: X={X.shape}, y={y.shape}")
