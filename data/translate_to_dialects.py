import os
import sys
import argparse
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Target languages for translation
TARGET_LANGUAGES = {
    "ceb_Latn": "Cebuano",
    "ilo_Latn": "Ilocano",
}

# Source languages to translate FROM (these are the English rows)
ENGLISH_LANGUAGE_TAGS = ["english", "extra_spam_messages", "extra_phishing_sms"]


def load_nllb_model():
    """Load the NLLB-200 model and tokenizer with FP16 for speed."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading NLLB-200 model: {config.BACKTRANSLATION_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BACKTRANSLATION_MODEL)

    # Use FP16 on CUDA for ~2x speedup
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
    """Truncate text to max_words to speed up translation of very long SMS."""
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
    """Translate a batch of texts using NLLB-200 with greedy decoding for speed."""
    # Pre-truncate very long texts to avoid slow generation
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
            num_beams=1,  # Greedy decoding (~4x faster than beam=4)
            do_sample=False,
        )

    decoded = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
    return decoded


def get_checkpoint_path(split_name: str, tgt_lang: str) -> str:
    """Get the checkpoint file path for a given split and target language."""
    checkpoint_dir = os.path.join(config.PROCESSED_DIR, ".translation_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"{split_name}_{tgt_lang}_checkpoint.csv")


def translate_split(
    split_name: str,
    df: pd.DataFrame,
    model,
    tokenizer,
    tgt_lang: str,
    batch_size: int = 16,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Translate all English rows in a split to the target language.

    Returns a new DataFrame with:
      - text: the translated text
      - text_original: the original English text
      - label, label_encoded: preserved from original
      - language: set to the target language code
    """
    lang_name = TARGET_LANGUAGES[tgt_lang]

    # Filter to English-only rows
    english_mask = df["language"].isin(ENGLISH_LANGUAGE_TAGS)
    english_df = df[english_mask].copy().reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Translating {split_name} -> {lang_name} ({tgt_lang})")
    print(f"  English rows to translate: {len(english_df)}")
    print(f"  Batch size: {batch_size}")

    if len(english_df) == 0:
        print("  No English rows found. Skipping.")
        return pd.DataFrame()

    # Use text_clean as source (already preprocessed)
    text_col = "text_clean" if "text_clean" in english_df.columns else "text"
    texts = english_df[text_col].fillna("").tolist()

    # Check for checkpoint (resume support)
    checkpoint_path = get_checkpoint_path(split_name, tgt_lang)
    start_idx = 0
    translated_texts = []

    if resume and os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_csv(checkpoint_path, encoding="utf-8")
        start_idx = len(checkpoint_df)
        translated_texts = checkpoint_df["text_translated"].tolist()
        print(f"  Resuming from checkpoint: {start_idx}/{len(texts)} already done")

    # Translate in batches
    for i in tqdm(
        range(start_idx, len(texts), batch_size),
        desc=f"  {lang_name}",
        total=(len(texts) - start_idx + batch_size - 1) // batch_size,
    ):
        batch = texts[i : i + batch_size]

        # Skip empty texts
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
                batch_translations = non_empty_texts  # Fallback to original

            # Reconstruct full batch (fill empties with "")
            full_batch_translations = [""] * len(batch)
            for j, trans_idx in enumerate(non_empty_indices):
                full_batch_translations[trans_idx] = batch_translations[j]
        else:
            full_batch_translations = [""] * len(batch)

        translated_texts.extend(full_batch_translations)

        # Save checkpoint every 500 batches
        if (i // batch_size) % 500 == 0 and i > start_idx:
            _save_checkpoint(checkpoint_path, translated_texts)

    # Final checkpoint save
    _save_checkpoint(checkpoint_path, translated_texts)

    # Build result DataFrame
    result_df = english_df.copy()
    result_df["text_original"] = result_df[text_col]
    result_df["text"] = translated_texts
    result_df["text_clean"] = translated_texts
    result_df["language"] = tgt_lang

    # Drop rows where translation is empty
    result_df = result_df[result_df["text"].str.strip().str.len() > 0]

    print(f"  Translated {len(result_df)} rows to {lang_name}")

    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return result_df


def _save_checkpoint(path: str, translated_texts: list[str]):
    """Save translation progress to a checkpoint file."""
    pd.DataFrame({"text_translated": translated_texts}).to_csv(
        path, index=False, encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Translate English dataset to Cebuano and Ilocano"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to translate (default: all)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(TARGET_LANGUAGES.keys()),
        help="Target language codes (default: ceb_Latn ilo_Latn)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Translation batch size (default: 16, increase if you have more VRAM)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    args = parser.parse_args()

    config.set_seed()

    # Load model once
    model, tokenizer = load_nllb_model()

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

            translated_df = translate_split(
                split_name=split_name,
                df=df,
                model=model,
                tokenizer=tokenizer,
                tgt_lang=tgt_lang,
                batch_size=args.batch_size,
                resume=args.resume,
            )

            if len(translated_df) > 0:
                # Save as separate file: e.g., train_ceb_Latn.csv
                output_filename = f"{split_name}_{tgt_lang}.csv"
                output_path = os.path.join(config.PROCESSED_DIR, output_filename)
                translated_df.to_csv(output_path, index=False, encoding="utf-8")

                print(f"  Saved: {output_path} ({len(translated_df)} rows)")
                print(f"  Label distribution:")
                for lbl, cnt in translated_df["label"].value_counts().items():
                    print(f"    {lbl}: {cnt}")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Translation complete!")
    print(f"Output files are in: {config.PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
