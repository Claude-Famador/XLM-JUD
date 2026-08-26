import os
import sys
import random
import pandas as pd
import torch
from tqdm import tqdm
from langdetect import detect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_nllb_model():
    """
    Load the Meta NLLB-200 model and tokenizer.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading NLLB-200 model: {config.BACKTRANSLATION_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.BACKTRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.BACKTRANSLATION_MODEL)
    model.to(config.DEVICE)
    model.eval()

    return model, tokenizer


def translate_text(
    texts: list[str],
    model,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    max_length: int = 128,
    batch_size: int = 16,
) -> list[str]:
    """
    Translate texts using NLLB-200.
    """
    tokenizer.src_lang = src_lang
    translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(config.DEVICE)
        
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        
        with torch.no_grad():
            translated_ids = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=4,
            )
            
        decoded = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


def detect_language(text: str) -> str:
    """
    Detect language of the text. Returns NLLB language code.
    Defaults to English if detection fails.
    """
    try:
        lang = detect(text)
        if lang == 'tl':
            return 'tgl_Latn'
        elif lang in ['ceb', 'ilo']:
            return f"{lang}_Latn"
        return 'eng_Latn'  # Default for everything else
    except:
        return 'eng_Latn'


def back_translate_nllb(
    texts: list[str],
    model,
    tokenizer,
    num_variants: int = 1,
) -> list[str]:
    """
    Perform Original -> Intermediate -> Original back-translation.
    """
    all_augmented = []

    for text in texts:
        src_lang = detect_language(text)
        
        for _ in range(num_variants):
            # Pick a random intermediate dialect
            intermediate = random.choice(config.DIALECT_TARGETS)
            
            # Forward: Original -> Intermediate
            translated = translate_text([text], model, tokenizer, src_lang, intermediate)
            
            # Reverse: Intermediate -> Original
            if translated and translated[0].strip():
                back_translated = translate_text(translated, model, tokenizer, intermediate, src_lang)
                if back_translated:
                    all_augmented.append(back_translated[0])
                else:
                    all_augmented.append(text)
            else:
                all_augmented.append(text)

    return all_augmented


def augment_with_back_translation(
    train_df: pd.DataFrame,
    text_column: str = "text_clean",
    label_column: str = "label",
    minority_label: str = "smishing",
    num_variants: int = None,
) -> pd.DataFrame:
    """
    Augment minority class using NLLB back-translation.
    """
    num_variants = num_variants or config.BACKTRANSLATION_NUM_VARIANTS

    minority_df = train_df[train_df[label_column] == minority_label].copy()
    minority_texts = minority_df[text_column].tolist()

    print(f"\nBack-Translation Augmentation (NLLB-200)")
    print(f"  Minority class ({minority_label}): {len(minority_texts)} samples")

    if len(minority_texts) == 0:
        return train_df

    model, tokenizer = load_nllb_model()

    augmented_texts = back_translate_nllb(
        minority_texts,
        model=model,
        tokenizer=tokenizer,
        num_variants=num_variants,
    )

    augmented_rows = []
    for i, aug_text in enumerate(augmented_texts):
        original_idx = i % len(minority_texts)
        if aug_text.strip() == minority_texts[original_idx].strip() or not aug_text.strip():
            continue

        augmented_rows.append({
            text_column: aug_text,
            label_column: minority_label,
            "label_encoded": config.LABEL_MAP[minority_label],
            "language": "augmented_bt_nllb",
            "augmentation": "back_translation",
        })

    augmented_df = pd.DataFrame(augmented_rows)

    for col in train_df.columns:
        if col not in augmented_df.columns:
            augmented_df[col] = None

    combined = pd.concat([train_df, augmented_df[train_df.columns]], ignore_index=True)

    print(f"  Generated {len(augmented_df)} new samples")
    print(f"  Total samples: {len(combined)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return combined


if __name__ == "__main__":
    config.set_seed()
    train_path = os.path.join(config.PROCESSED_DIR, "train.csv")
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path, encoding="utf-8")
        augmented_df = augment_with_back_translation(train_df)
        augmented_df.to_csv(os.path.join(config.PROCESSED_DIR, "train_augmented_bt.csv"), index=False, encoding="utf-8")
