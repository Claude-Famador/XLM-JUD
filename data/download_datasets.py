"""
Dataset Downloader (Section 3.1.1).

Downloads the UCI SMS Spam Collection dataset as the primary English corpus,
and optionally downloads the FLORES-200 evaluation benchmark for Cebuano (ceb_Latn)
and Ilocano (ilo_Latn).
"""

import os
import sys
import pandas as pd
from datasets import load_dataset

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def download_uci_dataset(output_dir: str = None) -> pd.DataFrame:
    """
    Download the UCI SMS Spam Collection dataset using Hugging Face datasets.

    Returns:
        DataFrame containing English Ham and Spam messages.
    """
    output_dir = output_dir or config.DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading UCI SMS Spam Collection Dataset...")
    try:
        dataset = load_dataset("sms_spam", trust_remote_code=True)
        df = pd.DataFrame(dataset["train"])

        # Rename columns to match our expected format
        # sms_spam dataset usually has 'sms' and 'label' (0=ham, 1=spam)
        # Verify columns
        if 'sms' in df.columns and 'label' in df.columns:
            df = df.rename(columns={'sms': 'text'})
            # Map 0 -> ham, 1 -> smishing
            label_mapping = {0: 'ham', 1: 'smishing'}
            df['label'] = df['label'].map(label_mapping)
        else:
            print(f"  Unexpected columns in UCI dataset: {df.columns}")

        df['language'] = 'english'

        output_path = os.path.join(output_dir, "uci_spam.csv")
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"  Saved {len(df)} sentences to: {output_path}")

        return df
    except Exception as e:
        print(f"  Error downloading UCI dataset: {e}")
        return pd.DataFrame()


def download_flores(
    languages: list[str] = None,
    split: str = None,
    output_dir: str = None,
) -> dict[str, pd.DataFrame]:
    """
    Download FLORES-200 data for specified languages.
    """
    languages = languages or config.FLORES_LANGUAGES
    split = split or config.FLORES_SPLIT
    output_dir = output_dir or config.DATA_DIR

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print(f"\nDownloading FLORES-200 data for languages: {languages}")
    for lang in languages:
        try:
            dataset = load_dataset("facebook/flores", lang, split=split, trust_remote_code=True)
            df = pd.DataFrame({
                "text": dataset["sentence"],
                "language": lang,
                "label": "ham",
            })
            output_path = os.path.join(output_dir, f"flores_{lang}.csv")
            df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"  Saved {len(df)} {lang} sentences to: {output_path}")
            results[lang] = df
        except Exception as e:
            print(f"  Error downloading {lang}: {e}")

    return results


def merge_datasets(
    sample_path: str = None,
    data_dir: str = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Merge UCI Spam dataset, sample dataset, and optionally FLORES data.
    """
    sample_path = sample_path or config.RAW_DATASET_PATH
    data_dir = data_dir or config.DATA_DIR
    output_path = output_path or os.path.join(data_dir, "merged_dataset.csv")

    dfs = []

    # 1. UCI Dataset
    uci_path = os.path.join(data_dir, "uci_spam.csv")
    if os.path.exists(uci_path):
        dfs.append(pd.read_csv(uci_path, encoding="utf-8"))
        print(f"Added UCI dataset: {len(dfs[-1])} samples")

    # 2. Sample Dataset
    if os.path.exists(sample_path):
        dfs.append(pd.read_csv(sample_path, encoding="utf-8"))
        print(f"Added sample dataset: {len(dfs[-1])} samples")

    # 3. FLORES Data
    for lang in config.FLORES_LANGUAGES:
        flores_path = os.path.join(data_dir, f"flores_{lang}.csv")
        if os.path.exists(flores_path):
            dfs.append(pd.read_csv(flores_path, encoding="utf-8"))
            print(f"Added FLORES {lang}: {len(dfs[-1])} samples")

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\nMerged dataset saved: {len(merged)} total samples -> {output_path}")
        # Update config to point to merged dataset for preprocessing
        config.RAW_DATASET_PATH = output_path
        return merged
    else:
        print("No datasets found to merge.")
        return pd.DataFrame()


if __name__ == "__main__":
    config.set_seed()
    print("=" * 60)
    print("Dataset Downloader")
    print("=" * 60)
    download_uci_dataset()
    download_flores()
    merge_datasets()
