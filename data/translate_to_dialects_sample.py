"""
Translate a SAMPLE of English rows to Cebuano & Ilocano for human validation.

This is a sampling wrapper around the translation logic in translate_to_dialects.py.
It randomly samples N English rows per split, translates them, and saves
side-by-side CSVs (original + translated) for human review.

Usage:
    python data/translate_to_dialects_sample.py                     # 500 samples/split
    python data/translate_to_dialects_sample.py --sample-size 100   # 100 samples/split
    python data/translate_to_dialects_sample.py --splits test       # Only test split
"""

import os
import sys
import argparse
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Reuse translation logic from the main script
from translate_to_dialects import (
    TARGET_LANGUAGES,
    ENGLISH_LANGUAGE_TAGS,
    load_nllb_model,
    translate_batch,
)


def translate_sample(
    split_name: str,
    df: pd.DataFrame,
    model,
    tokenizer,
    tgt_lang: str,
    sample_size: int = 500,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Sample English rows and translate them to the target language.
    Returns a DataFrame with original and translated text side-by-side.
    """
    lang_name = TARGET_LANGUAGES[tgt_lang]

    # Filter to English-only rows
    english_mask = df["language"].isin(ENGLISH_LANGUAGE_TAGS)
    english_df = df[english_mask].copy().reset_index(drop=True)

    # Sample N rows (or all if fewer than N)
    actual_sample = min(sample_size, len(english_df))
    sampled_df = english_df.sample(n=actual_sample, random_state=config.SEED).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Translating {split_name} -> {lang_name} ({tgt_lang})")
    print(f"  Sampled {actual_sample} from {len(english_df)} English rows")
    print(f"  Batch size: {batch_size}")

    if actual_sample == 0:
        print("  No English rows found. Skipping.")
        return pd.DataFrame()

    text_col = "text_clean" if "text_clean" in sampled_df.columns else "text"
    texts = sampled_df[text_col].fillna("").tolist()

    # Translate in batches
    translated_texts = []
    for i in tqdm(
        range(0, len(texts), batch_size),
        desc=f"  {lang_name}",
        total=(len(texts) + batch_size - 1) // batch_size,
    ):
        batch = texts[i : i + batch_size]
        non_empty_indices = [j for j, t in enumerate(batch) if t.strip()]
        non_empty_texts = [batch[j] for j in non_empty_indices]

        if non_empty_texts:
            try:
                batch_translations = translate_batch(
                    non_empty_texts,
                    model,
                    tokenizer,
                    src_lang=config.ENGLISH_CODE,
                    tgt_lang=tgt_lang,
                )
            except Exception as e:
                print(f"\n  Error translating batch at index {i}: {e}")
                batch_translations = non_empty_texts

            full_batch = [""] * len(batch)
            for j, trans_idx in enumerate(non_empty_indices):
                full_batch[trans_idx] = batch_translations[j]
        else:
            full_batch = [""] * len(batch)

        translated_texts.extend(full_batch)

    # Build side-by-side validation DataFrame
    result_df = pd.DataFrame({
        "original_text": texts,
        "translated_text": translated_texts,
        "label": sampled_df["label"].tolist(),
        "source_language": sampled_df["language"].tolist(),
        "target_language": tgt_lang,
        "target_language_name": lang_name,
        "split": split_name,
    })

    # Drop empty translations
    result_df = result_df[result_df["translated_text"].str.strip().str.len() > 0]
    print(f"  Successfully translated {len(result_df)} rows to {lang_name}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Translate a sample of English rows to dialects for human validation"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to translate (default: all)",
    )
    parser.add_argument(
        "--languages", nargs="+", default=list(TARGET_LANGUAGES.keys()),
        help="Target language codes (default: ceb_Latn ilo_Latn)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=500,
        help="Number of English rows to sample per split (default: 500)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Translation batch size (default: 16)",
    )
    args = parser.parse_args()

    config.set_seed()

    # Output directory for validation files
    validation_dir = os.path.join(config.OUTPUT_DIR, "human_validation")
    os.makedirs(validation_dir, exist_ok=True)

    # Load model once
    model, tokenizer = load_nllb_model()

    all_results = []

    for split_name in args.splits:
        split_path = os.path.join(config.PROCESSED_DIR, f"{split_name}.csv")

        if not os.path.exists(split_path):
            print(f"\nSkipping {split_name}: {split_path} not found")
            continue

        df = pd.read_csv(split_path, encoding="utf-8")
        print(f"\nLoaded {split_name}.csv: {len(df)} rows")

        for tgt_lang in args.languages:
            if tgt_lang not in TARGET_LANGUAGES:
                print(f"  Unknown language: {tgt_lang}, skipping")
                continue

            result_df = translate_sample(
                split_name=split_name,
                df=df,
                model=model,
                tokenizer=tokenizer,
                tgt_lang=tgt_lang,
                sample_size=args.sample_size,
                batch_size=args.batch_size,
            )

            if len(result_df) > 0:
                # Save per-language per-split file
                filename = f"eng_to_{tgt_lang}_{split_name}_sample.csv"
                output_path = os.path.join(validation_dir, filename)
                result_df.to_csv(output_path, index=False, encoding="utf-8")
                print(f"  Saved: {output_path}")
                all_results.append(result_df)

    # Also save a combined file for easy review
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(validation_dir, "eng_to_dialect_all_samples.csv")
        combined.to_csv(combined_path, index=False, encoding="utf-8")
        print(f"\nCombined validation file: {combined_path} ({len(combined)} rows)")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("English -> Dialect translation sampling complete!")
    print(f"Validation files are in: {validation_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
