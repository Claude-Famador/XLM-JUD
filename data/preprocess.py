"""
Data Preprocessing Module (Section 3.1.2).

Implements the "conservative cleaning approach" specified in the paper:
- Preserves code-switching and dialectal features
- Retains non-standard punctuation/casing often found in smishing
- Standardizes Unicode characters
- Masks named entities (URLs, emails, phone numbers)
- Normalizes excessive whitespace
"""

import os
import sys
import re
import pandas as pd
import unicodedata2 as unicodedata

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def normalize_unicode(text: str) -> str:
    """Standardize Unicode characters to NFKC form."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize('NFKC', text)


def mask_entities(text: str) -> str:
    """
    Mask specific entities (URLs, emails, phone numbers) 
    that might cause overfitting.
    """
    # Mask URLs
    text = re.sub(config.URL_PATTERN, "<URL>", text)
    
    # Mask emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', "<EMAIL>", text)
    
    # Mask phone numbers (simple pattern for numbers with 7+ digits)
    text = re.sub(r'\b(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){7,14}\b', "<PHONE>", text)
    
    return text


def clean_text(text: str) -> str:
    """
    Conservative cleaning pipeline.
    Does NOT lowercase or remove punctuation, as these are features of smishing.
    """
    text = normalize_unicode(text)
    
    # Remove HTML tags if any
    text = re.sub(config.HTML_TAG_PATTERN, " ", text)
    
    # Remove metadata brackets
    text = re.sub(config.METADATA_PATTERN, " ", text)
    
    # Mask entities
    text = mask_entities(text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_dataset(
    input_path: str = None,
    output_dir: str = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dataset, clean text, encode labels, and create stratified splits.
    
    Returns:
        train_df, val_df, test_df
    """
    input_path = input_path or config.RAW_DATASET_PATH
    output_dir = output_dir or config.PROCESSED_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nPreprocessing Dataset from: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
        
    df = pd.read_csv(input_path, encoding="utf-8")
    
    # Drop rows with missing text
    initial_len = len(df)
    df = df.dropna(subset=['text'])
    print(f"Dropped {initial_len - len(df)} rows with missing text.")
    
    # Apply conservative cleaning
    print("Applying conservative cleaning...")
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Filter out empty texts after cleaning
    df = df[df['text_clean'].str.len() > 0]
    
    # Encode labels
    df['label_encoded'] = df['label'].map(config.LABEL_MAP)
    
    if df['label_encoded'].isnull().any():
        print("Warning: Found labels not in LABEL_MAP. Dropping these rows.")
        df = df.dropna(subset=['label_encoded'])
        df['label_encoded'] = df['label_encoded'].astype(int)
        
    # Stratified Split (Train/Val/Test)
    from sklearn.model_selection import train_test_split
    
    print("Creating stratified splits...")
    
    # First split: Train vs Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        stratify=df['label_encoded'],
        random_state=config.SEED
    )
    
    # Second split: Val vs Test
    test_relative_ratio = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_relative_ratio,
        stratify=temp_df['label_encoded'],
        random_state=config.SEED
    )
    
    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"Data saved to {output_dir}:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    config.set_seed()
    preprocess_dataset()
