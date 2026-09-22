"""
Translate Dialect rows (Cebuano, Ilocano, Tagalog) to English for human validation.

This is the reverse of translate_to_dialects.py. It takes the existing dialect
rows in the dataset and translates them back to English so a human validator
can compare the dialect text with its English translation.

Outputs side-by-side CSVs (original dialect text + English translation) for review.

Usage:
    python data/translate_to_english.py                     # 500 samples/split, all dialects
    python data/translate_to_english.py --sample-size 100   # 100 samples/split
    python data/translate_to_english.py --splits test       # Only test split
    python data/translate_to_english.py --languages ceb_Latn  # Only Cebuano
"""

import os
import sys
import argparse
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Dialect languages to translate FROM
DIALECT_LANGUAGES = {
    "ceb_Latn": "Cebuano",
    "ilo_Latn": "Ilocano",
    "tgl_Latn": "Tagalog",
}


def load_nllb_model():
    """Load the NLLB-200 model and tokenizer with FP16 for speed."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading NLLB-200 model: {config.BACKTRANSLATION_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BACKTRANSLATION_MODEL)

    use_fp16 = torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.BACKTRANSLATION_MODEL, torch_dtype=dtype
    )
    model.to(config.DEVICE)
    model.eval()
    print(f"Model loaded on {config.DEVICE} (dtype={dtype})")
    return model, tokenizer


def truncate_text(text: str, max_words: int = 80) -> str:
    """Truncate text to max_words to speed up translation."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def translate_batch(
    texts: list[str],
    model,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    max_length: int = 128,
) -> list[str]:
    """Translate a batch of texts from dialect to English using NLLB-200."""
    texts = [truncate_text(t) for t in texts]

    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(config.DEVICE)

    with torch.no_grad():
        translated_ids = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=1,  # Greedy decoding for speed
            do_sample=False,
        )

    decoded = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
    return decoded


def translate_dialect_to_english(
    split_name: str,
    df: pd.DataFrame,
    model,
    tokenizer,
    src_lang: str,
    sample_size: int = 500,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Sample dialect rows and translate them to English.
    Returns a DataFrame with original dialect text and English translation side-by-side.
    """
    lang_name = DIALECT_LANGUAGES[src_lang]

    # Filter to rows of this specific dialect
    dialect_mask = df["language"] == src_lang
    dialect_df = df[dialect_mask].copy().reset_index(drop=True)

    # Sample N rows (or all if fewer than N)
    actual_sample = min(sample_size, len(dialect_df))

    if actual_sample == 0:
        print(f"\n  No {lang_name} rows found in {split_name}. Skipping.")
        return pd.DataFrame()

    sampled_df = dialect_df.sample(n=actual_sample, random_state=config.SEED).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Translating {split_name}: {lang_name} ({src_lang}) -> English")
    print(f"  Sampled {actual_sample} from {len(dialect_df)} {lang_name} rows")
    print(f"  Batch size: {batch_size}")

    text_col = "text_clean" if "text_clean" in sampled_df.columns else "text"
    texts = sampled_df[text_col].fillna("").tolist()

    # Translate in batches
    translated_texts = []
    for i in tqdm(
        range(0, len(texts), batch_size),
        desc=f"  {lang_name} -> English",
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
                    src_lang=src_lang,
                    tgt_lang=config.ENGLISH_CODE,
                )
            except Exception as e:
                print(f"\n  Error translating batch at index {i}: {e}")
                batch_translations = non_empty_texts  # Fallback

            full_batch = [""] * len(batch)
            for j, trans_idx in enumerate(non_empty_indices):
                full_batch[trans_idx] = batch_translations[j]
        else:
            full_batch = [""] * len(batch)

        translated_texts.extend(full_batch)

    # Build side-by-side validation DataFrame
    result_df = pd.DataFrame({
        "original_dialect_text": texts,
        "translated_to_english": translated_texts,
        "label": sampled_df["label"].tolist(),
        "source_language": src_lang,
        "source_language_name": lang_name,
        "target_language": "eng_Latn",
        "split": split_name,
    })

    # Drop empty translations
    result_df = result_df[result_df["translated_to_english"].str.strip().str.len() > 0]
    print(f"  Successfully translated {len(result_df)} {lang_name} rows to English")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Translate dialect rows (Cebuano/Ilocano/Tagalog) to English for human validation"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to translate (default: all)",
    )
    parser.add_argument(
        "--languages", nargs="+",
        default=list(DIALECT_LANGUAGES.keys()),
        help="Source dialect language codes (default: ceb_Latn ilo_Latn tgl_Latn)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=500,
        help="Number of dialect rows to sample per language per split (default: 500)",
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

        for src_lang in args.languages:
            if src_lang not in DIALECT_LANGUAGES:
                print(f"  Unknown language: {src_lang}, skipping")
                continue

            result_df = translate_dialect_to_english(
                split_name=split_name,
                df=df,
                model=model,
                tokenizer=tokenizer,
                src_lang=src_lang,
                sample_size=args.sample_size,
                batch_size=args.batch_size,
            )

            if len(result_df) > 0:
                filename = f"{src_lang}_to_eng_{split_name}_sample.csv"
                output_path = os.path.join(validation_dir, filename)
                result_df.to_csv(output_path, index=False, encoding="utf-8")
                print(f"  Saved: {output_path}")
                all_results.append(result_df)

    # Also save a combined file for easy review
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(validation_dir, "dialect_to_eng_all_samples.csv")
        combined.to_csv(combined_path, index=False, encoding="utf-8")
        print(f"\nCombined validation file: {combined_path} ({len(combined)} rows)")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Dialect -> English translation sampling complete!")
    print(f"Validation files are in: {validation_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
