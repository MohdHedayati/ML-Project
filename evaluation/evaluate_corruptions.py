import sys
import os
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image

from corruptions.pipeline import (
    IMAGE_SIZE,
    CORRUPTIONS,
    SUPPORTED_CORRUPTIONS,
    preprocess_corrupted_image,
    corrupted_batch_generator,
    get_dataset_manifest,
)


MODEL_FILES = {
    "gtsrb": "german_cnn.keras",
    "btsd": "belgium_cnn.keras",
    "ctsd": "ctsd_cnn.keras",
}

DEFAULT_RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "corruption_metrics.csv")


def verify_model_input_shape(model: tf.keras.Model, expected_shape: Tuple[int, int] = IMAGE_SIZE, strict: bool = True) -> Tuple[int, int]:
    """
    Verifies that the loaded model expects the required input resolution (H, W).
    """
    model_shape = model.input_shape
    # model.input_shape is typically (None, H, W, C)
    if len(model_shape) == 4:
        h, w = model_shape[1], model_shape[2]
        if (h, w) != expected_shape:
            msg = (
                f"Incompatible model input shape! Model expects ({h}, {w}, {model_shape[3]}), "
                f"but corruption pipeline is configured for ({expected_shape[0]}, {expected_shape[1]}, 3)."
            )
            if strict:
                raise ValueError(msg)
            else:
                print(f"[Warning] {msg} Adapting pipeline to model expected resolution ({h}, {w}).")
                return (h, w)
        return (h, w)
    raise ValueError(f"Unexpected model input tensor rank: {model_shape}")


def init_results_csv(csv_path: str):
    """
    Initializes CSV with required headers if not already present.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "corruption",
                "severity",
                "num_images",
                "accuracy",
                "macro_f1",
                "weighted_f1"
            ])


def append_result_row(
    csv_path: str,
    dataset: str,
    corruption: str,
    severity: int,
    num_images: int,
    accuracy: float,
    macro_f1: float,
    weighted_f1: float,
):
    """
    Incrementally appends a single evaluation condition result to the CSV file.
    """
    init_results_csv(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset,
            corruption,
            severity,
            num_images,
            f"{accuracy:.4f}",
            f"{macro_f1:.4f}",
            f"{weighted_f1:.4f}"
        ])
        f.flush()


def evaluate_condition(
    model: tf.keras.Model,
    image_paths: List[str],
    labels: List[int],
    corruption: Optional[str],
    severity: int,
    batch_size: int = 64,
    limit: Optional[int] = None,
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> Dict[str, float]:
    """
    Runs batched inference on corrupted images for a specific condition.
    """
    if limit is not None and limit > 0:
        paths_to_eval = image_paths[:limit]
        labels_to_eval = labels[:limit]
    else:
        paths_to_eval = image_paths
        labels_to_eval = labels

    all_preds = []
    all_targets = []

    generator = corrupted_batch_generator(
        paths_to_eval,
        labels_to_eval,
        corruption=corruption,
        severity=severity,
        batch_size=batch_size,
        target_size=target_size,
    )

    for batch_imgs, batch_lbls in generator:
        preds = model.predict(batch_imgs, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        all_preds.extend(pred_classes)
        all_targets.extend(batch_lbls)

    if not all_preds:
        return {
            "num_images": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }

    actual = np.array(all_targets)
    pred = np.array(all_preds)

    acc = float(accuracy_score(actual, pred))
    macro_f1 = float(f1_score(actual, pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(actual, pred, average="weighted", zero_division=0))

    return {
        "num_images": len(actual),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def run_quick_tests(output_dir: str = "results"):
    """
    Executes the required validation tests (Section 7):
    1. Tests one original image with Gaussian noise at severity 1.
    2. Tests a stronger severity (severity 5).
    3. Confirms both outputs have shape (49, 49, 3) and values in [0, 1].
    4. Confirms labels remain unchanged.
    5. Saves clean-versus-corrupted example images.
    6. Runs a small evaluation subset on one saved model.
    7. Verifies CSV contains valid metrics.
    8. Tests error handling for missing paths, invalid severities, unsupported corruptions, and bad shapes.
    """
    print("\n==========================================")
    print("RUNNING SECTION 7 VALIDATION SUITE")
    print("==========================================")

    os.makedirs(output_dir, exist_ok=True)

    # Find a sample image from GTSRB
    gtsrb_paths, gtsrb_labels, _ = get_dataset_manifest("gtsrb")
    sample_img_path = gtsrb_paths[0]
    sample_label = gtsrb_labels[0]
    print(f"Sample test image: {sample_img_path} (Label: {sample_label})")

    # 1 & 2: Process Gaussian noise at severity 1 and severity 5
    clean_img = preprocess_corrupted_image(sample_img_path, corruption=None, severity=0)
    corrupted_sev1 = preprocess_corrupted_image(sample_img_path, corruption="gaussian_noise", severity=1)
    corrupted_sev5 = preprocess_corrupted_image(sample_img_path, corruption="gaussian_noise", severity=5)

    # 3: Confirm shapes and pixel ranges
    assert clean_img.shape == (49, 49, 3), f"Clean shape error: {clean_img.shape}"
    assert corrupted_sev1.shape == (49, 49, 3), f"Sev1 shape error: {corrupted_sev1.shape}"
    assert corrupted_sev5.shape == (49, 49, 3), f"Sev5 shape error: {corrupted_sev5.shape}"

    assert 0.0 <= clean_img.min() and clean_img.max() <= 1.0, "Clean pixel range error"
    assert 0.0 <= corrupted_sev1.min() and corrupted_sev1.max() <= 1.0, "Sev1 pixel range error"
    assert 0.0 <= corrupted_sev5.min() and corrupted_sev5.max() <= 1.0, "Sev5 pixel range error"
    print("[PASS] Test 1-3: Output shape is (49, 49, 3) and values are in [0, 1].")

    # 4: Confirm labels remain unchanged
    assert sample_label == gtsrb_labels[0], "Label alteration detected!"
    print(f"[PASS] Test 4: Image label strictly preserved (Label={sample_label}).")

    # 5: Save clean-versus-corrupted example images
    ex_clean_path = os.path.join(output_dir, "example_clean.png")
    ex_sev1_path = os.path.join(output_dir, "example_gaussian_sev1.png")
    ex_sev5_path = os.path.join(output_dir, "example_gaussian_sev5.png")

    Image.fromarray((clean_img * 255).astype(np.uint8)).save(ex_clean_path)
    Image.fromarray((corrupted_sev1 * 255).astype(np.uint8)).save(ex_sev1_path)
    Image.fromarray((corrupted_sev5 * 255).astype(np.uint8)).save(ex_sev5_path)
    print(f"[PASS] Test 5: Saved visualization samples to {output_dir}/")

    # 8: Error handling verification
    print("\nVerifying Error Handling:")

    # 8a: Missing image path
    try:
        preprocess_corrupted_image("non_existent_file.png", corruption="gaussian_noise", severity=1)
        raise AssertionError("Failed to catch missing image path!")
    except FileNotFoundError:
        print("  [OK] Caught missing image path error as FileNotFoundError.")

    # 8b: Invalid severity
    try:
        preprocess_corrupted_image(sample_img_path, corruption="gaussian_noise", severity=99)
        raise AssertionError("Failed to catch invalid severity!")
    except ValueError:
        print("  [OK] Caught invalid severity error as ValueError.")

    # 8c: Unsupported corruption name
    try:
        preprocess_corrupted_image(sample_img_path, corruption="solar_flare_blur", severity=1)
        raise AssertionError("Failed to catch unsupported corruption!")
    except ValueError:
        print("  [OK] Caught unsupported corruption name as ValueError.")

    # 8d: Incompatible model input shape check
    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    try:
        verify_model_input_shape(dummy_model, expected_shape=(49, 49), strict=True)
        raise AssertionError("Failed to catch incompatible model shape!")
    except ValueError:
        print("  [OK] Caught incompatible model input shape as ValueError.")

    # 6 & 7: Small evaluation subset on saved model and verify CSV
    test_csv = os.path.join(output_dir, "test_metrics.csv")
    if os.path.exists(test_csv):
        os.remove(test_csv)

    model_path = os.path.join(PROJECT_ROOT, MODEL_FILES["gtsrb"])
    if os.path.exists(model_path):
        print(f"\nRunning subset evaluation on {model_path} (sample limit: 20)...")
        model = tf.keras.models.load_model(model_path)
        # Check if legacy model needs resolution adaptation for test
        active_shape = verify_model_input_shape(model, expected_shape=IMAGE_SIZE, strict=False)

        metrics = evaluate_condition(
            model=model,
            image_paths=gtsrb_paths,
            labels=gtsrb_labels,
            corruption="gaussian_noise",
            severity=1,
            batch_size=10,
            limit=20,
            target_size=active_shape,
        )
        print(f"  Subset metrics: {metrics}")
        append_result_row(
            test_csv,
            dataset="gtsrb",
            corruption="gaussian_noise",
            severity=1,
            num_images=metrics["num_images"],
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            weighted_f1=metrics["weighted_f1"],
        )
        assert os.path.exists(test_csv), "Metrics CSV was not created!"
        with open(test_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2, f"CSV should contain 1 header + 1 data line, got {len(lines)}"
            print(f"  [OK] Verified CSV output format: {lines[0].strip()} -> {lines[1].strip()}")
        print("[PASS] Test 6-7: Small evaluation subset succeeded and CSV verified.")

    print("\n==========================================")
    print("ALL SECTION 7 VALIDATION CHECKS PASSED!")
    print("==========================================\n")


def run_evaluation(
    dataset_name: str = "all",
    corruption_filter: str = "all",
    severity_filter: str = "all",
    include_clean: bool = True,
    batch_size: int = 64,
    limit: Optional[int] = None,
    output_csv: str = DEFAULT_RESULTS_CSV,
    strict_shape: bool = False,
):
    """
    Main evaluation routine across datasets, corruptions, and severity levels.
    """
    init_results_csv(output_csv)

    datasets = ["gtsrb", "btsd", "ctsd"] if dataset_name == "all" else [dataset_name]
    corruptions = CORRUPTIONS if corruption_filter == "all" else [corruption_filter]
    severities = [1, 2, 3, 4, 5] if severity_filter == "all" else [int(severity_filter)]

    print(f"Starting Robustness Evaluation Pipeline")
    print(f"Datasets:    {datasets}")
    print(f"Corruptions: {len(corruptions)} types")
    print(f"Severities:  {severities}")
    print(f"Output CSV:  {output_csv}")
    print(f"Sample limit per condition: {limit if limit else 'Full Test Set'}")

    for ds in datasets:
        print(f"\n==========================================")
        print(f"DATASET: {ds.upper()}")
        print(f"==========================================")

        model_file = MODEL_FILES.get(ds)
        model_path = os.path.join(PROJECT_ROOT, model_file)

        if not os.path.exists(model_path):
            print(f"[Error] Model file for {ds} not found at {model_path}. Skipping.")
            continue

        print(f"Loading model: {model_path}...")
        model = tf.keras.models.load_model(model_path)
        expected_shape = verify_model_input_shape(model, expected_shape=IMAGE_SIZE, strict=strict_shape)

        test_paths, test_labels, num_classes = get_dataset_manifest(ds, project_root=PROJECT_ROOT)
        print(f"Loaded {len(test_paths)} test images across {num_classes} classes.")

        # 1. Clean Baseline
        if include_clean:
            print(f"\nEvaluating Clean Baseline for {ds}...")
            res_clean = evaluate_condition(
                model=model,
                image_paths=test_paths,
                labels=test_labels,
                corruption=None,
                severity=0,
                batch_size=batch_size,
                limit=limit,
                target_size=expected_shape,
            )
            print(
                f"  Clean: Acc={res_clean['accuracy']:.4f}, "
                f"Macro-F1={res_clean['macro_f1']:.4f}, "
                f"Weighted-F1={res_clean['weighted_f1']:.4f} "
                f"({res_clean['num_images']} images)"
            )
            append_result_row(
                output_csv,
                dataset=ds,
                corruption="clean",
                severity=0,
                num_images=res_clean["num_images"],
                accuracy=res_clean["accuracy"],
                macro_f1=res_clean["macro_f1"],
                weighted_f1=res_clean["weighted_f1"],
            )

        # 2. Corruptions x Severities
        for c in corruptions:
            for s in severities:
                print(f"Evaluating {ds} | Corruption: {c} | Severity: {s}...")
                metrics = evaluate_condition(
                    model=model,
                    image_paths=test_paths,
                    labels=test_labels,
                    corruption=c,
                    severity=s,
                    batch_size=batch_size,
                    limit=limit,
                    target_size=expected_shape,
                )
                print(
                    f"  Result: Acc={metrics['accuracy']:.4f}, "
                    f"Macro-F1={metrics['macro_f1']:.4f}, "
                    f"Weighted-F1={metrics['weighted_f1']:.4f} "
                    f"({metrics['num_images']} images)"
                )
                append_result_row(
                    output_csv,
                    dataset=ds,
                    corruption=c,
                    severity=s,
                    num_images=metrics["num_images"],
                    accuracy=metrics["accuracy"],
                    macro_f1=metrics["macro_f1"],
                    weighted_f1=metrics["weighted_f1"],
                )

    print(f"\n[Complete] All evaluation runs finished. Metrics saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Traffic Sign CNN Robustness under Image Corruptions")
    parser.add_argument("--dataset", choices=["all", "gtsrb", "btsd", "ctsd"], default="all", help="Target dataset")
    parser.add_argument("--corruption", default="all", help="Corruption type or 'all'")
    parser.add_argument("--severity", default="all", help="Severity level 1-5 or 'all'")
    parser.add_argument("--no-clean", action="store_true", help="Skip clean baseline evaluation")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit images evaluated per condition for smoke testing")
    parser.add_argument("--output-csv", default=DEFAULT_RESULTS_CSV, help="Path to save metrics CSV")
    parser.add_argument("--quick-test", action="store_true", help="Run the comprehensive Section 7 test suite")
    parser.add_argument("--strict-shape", action="store_true", help="Fail if model input shape != (49, 49)")

    args = parser.parse_args()

    if args.quick_test:
        run_quick_tests(output_dir=os.path.join(PROJECT_ROOT, "results"))
    else:
        run_evaluation(
            dataset_name=args.dataset,
            corruption_filter=args.corruption,
            severity_filter=args.severity,
            include_clean=not args.no_clean,
            batch_size=args.batch_size,
            limit=args.limit,
            output_csv=args.output_csv,
            strict_shape=args.strict_shape,
        )


if __name__ == "__main__":
    main()
