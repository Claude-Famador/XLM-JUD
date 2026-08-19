"""
Dialect-Aware Smishing Detection Pipeline — Main Entry Point.

CLI interface to run the full experimental pipeline:
    python main.py --phase all          # Full pipeline
    python main.py --phase data         # Preprocessing only
    python main.py --phase augment      # Data augmentation
    python main.py --phase train        # Model training
    python main.py --phase evaluate     # Evaluation only

Based on the methodology described in:
"Dialect-Aware Smishing Detection using XLM-RoBERTa"
"""

import argparse
import os
import sys
import time
import json

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def phase_data():
    """Phase 1: Data Collection and Preprocessing (Section 3.1)."""
    print("\n" + "=" * 70)
    print("  PHASE 1: DATA COLLECTION AND PREPROCESSING")
    print("=" * 70)

    # Optionally download UCI and FLORES-200
    try:
        from data.download_datasets import download_uci_dataset, download_flores, download_extra_datasets, merge_datasets
        print("\nAttempting to download UCI, FLORES, and extra datasets...")
        download_uci_dataset()
        download_flores()
        download_extra_datasets()
        merge_datasets()
    except Exception as e:
        print(f"Dataset download skipped or failed: {e}")
        print("(This is optional — the pipeline works without it if sample dataset is present)")

    from data.preprocess import preprocess_dataset
    train_df, val_df, test_df = preprocess_dataset()

    return train_df, val_df, test_df


def phase_augment():
    """Phase 2: Data Augmentation (Section 3.1.2 & 3.2.2)."""
    print("\n" + "=" * 70)
    print("  PHASE 2: DATA AUGMENTATION")
    print("=" * 70)

    import pandas as pd
    from models.train import load_data, run_augmentation

    train_df, val_df, test_df = load_data()

    augmented_df, X_smote, y_smote = run_augmentation(
        train_df,
        use_back_translation=True,
        use_smote=True,
    )

    return augmented_df, X_smote, y_smote


def phase_train(
    use_back_translation: bool = False,
    use_smote: bool = True,
    tune_hyperparams: bool = False,
):
    """Phase 3: Model Development (Section 3.2)."""
    print("\n" + "=" * 70)
    print("  PHASE 3: MODEL DEVELOPMENT")
    print("=" * 70)

    from models.train import run_full_training

    results = run_full_training(
        use_back_translation=use_back_translation,
        use_smote=use_smote,
        tune_hyperparams=tune_hyperparams,
    )

    return results


def phase_evaluate():
    """Phase 4: Model Evaluation (Section 3.3)."""
    print("\n" + "=" * 70)
    print("  PHASE 4: MODEL EVALUATION")
    print("=" * 70)

    import pandas as pd
    import numpy as np
    from evaluation.evaluate import evaluate_model_comprehensive, compare_models
    from evaluation.statistical_tests import run_all_statistical_tests, paired_t_test
    from evaluation.visualize import generate_all_visualizations

    # Load test data
    test_path = os.path.join(config.PROCESSED_DIR, "test.csv")
    if not os.path.exists(test_path):
        print("Error: Test data not found. Run --phase data first.")
        return

    test_df = pd.read_csv(test_path, encoding="utf-8")

    # Load saved results
    results_path = os.path.join(config.LOG_DIR, "xlm_results.json")
    baseline_path = os.path.join(config.LOG_DIR, "baseline_results.csv")

    all_results = {}
    evaluation_results = []

    # Evaluate XLM-RoBERTa models
    from models.xlm_roberta_model import XLMRoBERTaTrainer
    import joblib

    test_texts = test_df["text_clean"].fillna("").tolist()

    if os.path.exists(results_path):
        with open(results_path) as f:
            xlm_results = json.load(f)

        for key, result in xlm_results.items():
            all_results[key] = result
            if "predictions" in result:
                y_pred = np.array(result["predictions"])
                y_probs = np.array(result["probs"]) if "probs" in result else None
                eval_result = evaluate_model_comprehensive(
                    test_df, y_pred, y_probs=y_probs,
                    model_name=result.get("model", key),
                )
                evaluation_results.append(eval_result)
            elif key in ["XLM_original", "XLM_augmented"]:
                folder_name = "xlm_original" if key == "XLM_original" else "xlm_augmented"
                ckpt_dir = os.path.join(config.MODEL_DIR, folder_name, "xlm_roberta_best")
                if os.path.exists(ckpt_dir):
                    print(f"Loading {key} checkpoint from {ckpt_dir} for evaluation...")
                    trainer = XLMRoBERTaTrainer()
                    trainer.load_best_model(ckpt_dir)
                    preds, probs = trainer.predict(test_texts)
                    all_results[key]["predictions"] = preds.tolist()
                    all_results[key]["probs"] = probs.tolist()
                    eval_result = evaluate_model_comprehensive(
                        test_df, preds, y_probs=probs,
                        model_name=result.get("model", key),
                    )
                    evaluation_results.append(eval_result)

    for model_type in ["svm_original", "svm_smote", "lr_original", "lr_smote"]:
        model_path = os.path.join(config.MODEL_DIR, f"{model_type}.joblib")
        tfidf_path = os.path.join(config.MODEL_DIR, "tfidf_vectorizer.joblib")

        if os.path.exists(model_path) and os.path.exists(tfidf_path):
            model = joblib.load(model_path)
            tfidf = joblib.load(tfidf_path)

            X_test = tfidf.transform(test_df["text_clean"].fillna("").tolist())
            y_pred = model.predict(X_test)

            eval_result = evaluate_model_comprehensive(
                test_df, y_pred,
                model_name=model_type.upper(),
            )
            evaluation_results.append(eval_result)

    # Model comparison
    if evaluation_results:
        compare_models(evaluation_results)

    # Statistical tests
    run_all_statistical_tests(all_results)

    # Generate visualizations
    generate_all_visualizations(all_results, test_df=test_df)

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print(f"  Results: {config.LOG_DIR}")
    print(f"  Figures: {config.FIGURE_DIR}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Dialect-Aware Smishing Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --phase all              # Run full pipeline
  python main.py --phase data             # Preprocessing only
  python main.py --phase augment          # Data augmentation only
  python main.py --phase train            # Training only
  python main.py --phase train --bt       # Training with back-translation
  python main.py --phase train --hpo      # Training with hyperparameter tuning
  python main.py --phase evaluate         # Evaluation only
        """,
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["data", "augment", "train", "evaluate", "all"],
        help="Pipeline phase to run (default: all)",
    )
    parser.add_argument(
        "--bt",
        action="store_true",
        help="Enable back-translation augmentation (requires NLLB-200 models)",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE augmentation",
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Enable hyperparameter optimization (slower but better results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Set seed
    config.SEED = args.seed
    config.set_seed(args.seed)

    # Print header
    print("=" * 70)
    print("  DIALECT-AWARE SMISHING DETECTION PIPELINE")
    print("  Model: XLM-RoBERTa + SVM + Logistic Regression")
    print(f"  Device: {config.DEVICE}")
    print(f"  Seed: {config.SEED}")
    print(f"  Phase: {args.phase}")
    print("=" * 70)

    start_time = time.time()

    try:
        if args.phase == "data":
            phase_data()

        elif args.phase == "augment":
            phase_augment()

        elif args.phase == "train":
            phase_train(
                use_back_translation=args.bt,
                use_smote=not args.no_smote,
                tune_hyperparams=args.hpo,
            )

        elif args.phase == "evaluate":
            phase_evaluate()

        elif args.phase == "all":
            # Full pipeline
            phase_data()
            phase_train(
                use_back_translation=args.bt,
                use_smote=not args.no_smote,
                tune_hyperparams=args.hpo,
            )
            phase_evaluate()

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\n\nPipeline error: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
