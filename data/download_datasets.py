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
        dataset = load_dataset("sms_spam")
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
            dataset = load_dataset("facebook/flores", lang, split=split)
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


def download_extra_datasets(output_dir: str = None) -> dict[str, pd.DataFrame]:
    """
    Download additional SMS/Phishing datasets defined in config.EXTRA_DATASETS.

    Each dataset is normalized to the standard schema: text, label, language.
    Returns a dict mapping dataset name to its DataFrame.
    """
    output_dir = output_dir or config.DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    if not hasattr(config, "EXTRA_DATASETS") or not config.EXTRA_DATASETS:
        print("\nNo extra datasets configured.")
        return results

    print(f"\nDownloading {len(config.EXTRA_DATASETS)} extra dataset(s)...")

    for name, cfg in config.EXTRA_DATASETS.items():
        try:
            print(f"\n  [{name}] Loading {cfg['hf_id']}...")

            # Load from HuggingFace
            if cfg.get("subset"):
                dataset = load_dataset(
                    cfg["hf_id"], cfg["subset"]
                )
            else:
                dataset = load_dataset(cfg["hf_id"])

            # Combine all splits into one DataFrame
            all_splits = []
            for split_name in dataset:
                all_splits.append(dataset[split_name].to_pandas())
            df = pd.concat(all_splits, ignore_index=True)

            # Apply row-level filter if configured (e.g., keep only SMS rows)
            if cfg.get("filter_col") and cfg.get("filter_value"):
                before_len = len(df)
                df = df[df[cfg["filter_col"]] == cfg["filter_value"]].copy()
                print(f"  [{name}] Filtered {cfg['filter_col']}=={cfg['filter_value']}: "
                      f"{before_len} -> {len(df)} rows")

            # Rename text column
            text_col = cfg["text_col"]
            label_col = cfg["label_col"]
            if text_col != "text":
                df = df.rename(columns={text_col: "text"})
            if label_col != "label":
                df = df.rename(columns={label_col: "label"})

            # Map labels to our standard (ham / smishing)
            label_map = cfg.get("label_map", {})
            if label_map:
                df["label"] = df["label"].map(label_map)
                # Drop rows whose labels didn't map (unexpected labels)
                unmapped = df["label"].isnull().sum()
                if unmapped > 0:
                    print(f"  [{name}] Dropped {unmapped} rows with unmapped labels")
                    df = df.dropna(subset=["label"])

            # Keep only text + label, add language tag
            df = df[["text", "label"]].copy()
            df["language"] = f"extra_{name}"

            # Drop empty texts
            df = df.dropna(subset=["text"])
            df = df[df["text"].str.strip().str.len() > 0]

            # Save
            output_path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"  [{name}] Saved {len(df)} samples to: {output_path}")

            # Print class distribution
            dist = df["label"].value_counts()
            for lbl, cnt in dist.items():
                print(f"    {lbl}: {cnt}")

            results[name] = df

        except Exception as e:
            print(f"  [{name}] Error downloading: {e}")

    return results


def merge_datasets(
    sample_path: str = None,
    data_dir: str = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Merge UCI Spam dataset, sample dataset, extra datasets, and FLORES data.
    Deduplicates by text content to avoid double-counting overlapping samples.
    """
    sample_path = sample_path or config.RAW_DATASET_PATH
    data_dir = data_dir or config.DATA_DIR
    output_path = output_path or os.path.join(data_dir, "merged_dataset.csv")

    dfs = []

    # 1. UCI Dataset (primary)
    uci_path = os.path.join(data_dir, "uci_spam.csv")
    if os.path.exists(uci_path):
        dfs.append(pd.read_csv(uci_path, encoding="utf-8"))
        print(f"Added UCI dataset: {len(dfs[-1])} samples")

    # 2. Sample Dataset
    if os.path.exists(sample_path):
        dfs.append(pd.read_csv(sample_path, encoding="utf-8"))
        print(f"Added sample dataset: {len(dfs[-1])} samples")

    # 3. Extra HuggingFace Datasets
    if hasattr(config, "EXTRA_DATASETS"):
        for name in config.EXTRA_DATASETS:
            extra_path = os.path.join(data_dir, f"{name}.csv")
            if os.path.exists(extra_path):
                dfs.append(pd.read_csv(extra_path, encoding="utf-8"))
                print(f"Added extra dataset [{name}]: {len(dfs[-1])} samples")

    # 4. FLORES Data
    for lang in config.FLORES_LANGUAGES:
        flores_path = os.path.join(data_dir, f"flores_{lang}.csv")
        if os.path.exists(flores_path):
            dfs.append(pd.read_csv(flores_path, encoding="utf-8"))
            print(f"Added FLORES {lang}: {len(dfs[-1])} samples")

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        before_dedup = len(merged)

        # Deduplicate by normalized text content
        merged["_text_norm"] = merged["text"].str.strip().str.lower()
        merged = merged.drop_duplicates(subset=["_text_norm"], keep="first")
        merged = merged.drop(columns=["_text_norm"])

        dedup_removed = before_dedup - len(merged)
        if dedup_removed > 0:
            print(f"\nDeduplication: removed {dedup_removed} duplicate samples")

        merged.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\nMerged dataset saved: {len(merged)} total samples -> {output_path}")

        # Print final class distribution
        print("\nFinal class distribution:")
        for lbl, cnt in merged["label"].value_counts().items():
            print(f"  {lbl}: {cnt}")

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
    download_extra_datasets()
    merge_datasets()
