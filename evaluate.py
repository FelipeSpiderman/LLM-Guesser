#!/usr/bin/env python3
"""
Celebrity Guesser — Evaluation & Testing Script
================================================
A standalone script to evaluate the trained model, analyze performance,
and make predictions on any image.

Usage:
    python evaluate.py                    # Full evaluation
    python evaluate.py --predict image.jpg  # Single image prediction
    python evaluate.py --demo              # Run demo predictions
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = "models"
DATA_DIR = "datasets/celeb"


def load_config():
    cfg_path = os.path.join(MODEL_DIR, "model_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return None


def get_best_model_path():
    path = os.path.join(MODEL_DIR, "celebrity_best.keras")
    if os.path.exists(path):
        return path
    path = os.path.join(MODEL_DIR, "celebrity_final.keras")
    if os.path.exists(path):
        return path
    return None


# ============================================================
# LOAD MODEL
# ============================================================
print("=" * 60)
print("  CELEBRITY GUESSER — EVALUATION")
print("=" * 60)

model_path = get_best_model_path()
if model_path is None:
    print(f"ERROR: No model found in {MODEL_DIR}/")
    print("       Run the notebook first to train a model.")
    sys.exit(1)

print(f"\nLoading model from: {model_path}")
model = tf.keras.models.load_model(model_path)
print(f"Model loaded: {model.count_params():,} parameters")

# Load config
cfg = load_config()
if cfg:
    IMG_SIZE = cfg.get("IMG_SIZE", 224)
    CLASS_NAMES = cfg.get("CLASS_NAMES", [])
    NUM_CLASSES = cfg.get("NUM_CLASSES", len(CLASS_NAMES))
    best_val_acc = cfg.get("best_val_accuracy", "N/A")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Best val accuracy from training: {best_val_acc}")
else:
    IMG_SIZE = 224
    CLASS_NAMES = sorted(
        [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    )
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"Using defaults: {IMG_SIZE}x{IMG_SIZE}, {NUM_CLASSES} classes")

print()


# ============================================================
# DATA LOADING
# ============================================================
def load_image(img_path, color_mode="rgb"):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode=color_mode)
    img = img_to_array(img) / 255.0
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def load_all_data():
    all_paths = []
    all_labels = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(class_path, fname))
                all_labels.append(class_idx)
    return all_paths, all_labels


print("Loading dataset...")
all_paths, all_labels = load_all_data()
print(f"Total images: {len(all_paths)}")
print(f"Classes: {len(CLASS_NAMES)}")
for i, name in enumerate(CLASS_NAMES):
    count = sum(1 for l in all_labels if l == i)
    print(f"  {i:2d}. {name:<25s}: {count} images")
print()


# ============================================================
# FULL EVALUATION
# ============================================================
def evaluate_model():
    print("=" * 60)
    print("  FULL EVALUATION")
    print("=" * 60)

    # Load all images
    print("Loading images...")
    images = np.array([load_image(p) for p in all_paths], dtype=np.float32)
    labels = np.array(all_labels)

    # Predict
    print("Running predictions...")
    probs = model.predict(images, verbose=1)
    predictions = np.argmax(probs, axis=1)

    # Accuracy
    correct = np.sum(predictions == labels)
    total = len(labels)
    overall_acc = correct / total

    print(f"\n{'=' * 60}")
    print(f"  OVERALL ACCURACY: {overall_acc:.2%} ({correct}/{total})")
    print(f"{'=' * 60}\n")

    # Per-class accuracy
    print("Per-Class Accuracy:")
    print("-" * 60)
    class_correct = {}
    class_total = {}
    for name in CLASS_NAMES:
        class_correct[name] = 0
        class_total[name] = 0

    for true_label, pred_label in zip(labels, predictions):
        class_total[CLASS_NAMES[true_label]] += 1
        if true_label == pred_label:
            class_correct[CLASS_NAMES[true_label]] += 1

    sorted_classes = sorted(
        CLASS_NAMES,
        key=lambda n: class_correct[n] / max(1, class_total[n]),
        reverse=True,
    )
    for name in sorted_classes:
        acc = class_correct[name] / max(1, class_total[name])
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        status = "✓" if acc >= 0.8 else "◐" if acc >= 0.5 else "✗"
        print(f"  {status} {name:<25s} {acc:6.1%}  [{bar}]")

    # Confusion matrix
    print("\n" + "=" * 60)
    print("  CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title(f"Confusion Matrix — Overall Accuracy: {overall_acc:.1%}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print(f"Saved: {MODEL_DIR}/confusion_matrix.png")
    plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=CLASS_NAMES))

    return overall_acc, predictions, labels


# ============================================================
# PARTIAL IMAGE TEST
# ============================================================
def crop_image(img, crop_type):
    h, w = img.shape[:2]
    if crop_type == "eyes":
        return img[0 : h // 3, :]
    elif crop_type == "lower":
        return img[h // 4 :, :]
    elif crop_type == "left_half":
        return img[:, : w // 2]
    elif crop_type == "right_half":
        return img[:, w // 2 :]
    elif crop_type == "center":
        ch, cw = int(h * 0.6), int(w * 0.6)
        sh, sw = (h - ch) // 2, (w - cw) // 2
        return img[sh : sh + ch, sw : sw + cw]
    return img


def test_partial_images():
    print("\n" + "=" * 60)
    print("  PARTIAL IMAGE TEST")
    print("=" * 60)
    print("Testing model on cropped face images...\n")

    CROP_TYPES = ["eyes", "lower", "left_half", "right_half", "center"]
    results = {ct: {"correct": 0, "total": 0} for ct in CROP_TYPES}

    for crop_type in CROP_TYPES:
        for img_path, true_label in zip(all_paths, all_labels):
            img = load_image(img_path)
            cropped = crop_image(img, crop_type)
            resized = tf.image.resize(cropped, (IMG_SIZE, IMG_SIZE)).numpy()
            inp = np.expand_dims(resized, axis=0)
            probs = model.predict(inp, verbose=0)[0]
            pred = np.argmax(probs)
            results[crop_type]["total"] += 1
            if pred == true_label:
                results[crop_type]["correct"] += 1

    # Also test full images
    full_correct = sum(
        1
        for p, t in zip(all_paths, all_labels)
        if np.argmax(model.predict(np.expand_dims(load_image(p), 0), verbose=0)[0]) == t
    )
    full_total = len(all_paths)
    results["full"] = {"correct": full_correct, "total": full_total}
    CROP_TYPES.insert(0, "full")

    print(f"{'Type':<15} {'Accuracy':>8} {'Correct':>8} {'Total':>8}")
    print("-" * 42)
    for ct in CROP_TYPES:
        r = results[ct]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        print(f"{ct:<15} {acc:>7.1%}  {r['correct']:>7}  {r['total']:>7}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    types = CROP_TYPES
    accs = [results[ct]["correct"] / results[ct]["total"] for ct in types]
    colors = ["#27ae60"] + ["#3498db"] * (len(types) - 1)
    bars = ax.bar(types, accs, color=colors)
    ax.set_xlabel("Image Type")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy: Full vs Cropped Images")
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.1%}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "partial_test.png"), dpi=150)
    print(f"\nSaved: {MODEL_DIR}/partial_test.png")
    plt.show()


# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================
def predict_image(img_path, show_img=True, crop_types=None):
    print(f"\n{'=' * 60}")
    print(f"  PREDICTION: {img_path}")
    print("=" * 60)

    img = load_image(img_path)
    inp = np.expand_dims(img, axis=0)
    probs = model.predict(inp, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1]

    print(f"\nTop 5 Predictions:")
    print("-" * 40)
    for rank, idx in enumerate(top_indices[:5], 1):
        name = CLASS_NAMES[idx]
        prob = probs[idx]
        bar = "█" * int(prob * 30)
        print(f"  {rank}. {name:<25s} {prob:6.1%}  {bar}")

    top_name = CLASS_NAMES[top_indices[0]]
    top_conf = probs[top_indices[0]]
    print(f"\n  ➜  PREDICTION: {top_name} ({top_conf:.1%} confidence)")

    if show_img:
        if crop_types:
            n_cols = len(crop_types) + 1
            fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))
            if n_cols == 1:
                axes = [axes]

            axes[0].imshow(img)
            axes[0].set_title(f"Original\n{top_name}\n{top_conf:.1%}", fontsize=10)
            axes[0].axis("off")

            for col, crop_type in enumerate(crop_types, 1):
                cropped = crop_image(img, crop_type)
                resized = tf.image.resize(cropped, (IMG_SIZE, IMG_SIZE)).numpy()
                inp_crop = np.expand_dims(resized, axis=0)
                probs_crop = model.predict(inp_crop, verbose=0)[0]
                top_idx = np.argmax(probs_crop)
                pred_crop = CLASS_NAMES[top_idx]
                conf_crop = probs_crop[top_idx]

                axes[col].imshow(cropped)
                axes[col].set_title(
                    f"{crop_type}\n{pred_crop}\n{conf_crop:.1%}", fontsize=10
                )
                axes[col].axis("off")

            plt.suptitle(
                f"Celebrity Guesser — {os.path.basename(img_path)}", fontsize=13
            )
            plt.tight_layout()
            plt.show()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(img)
            axes[0].set_title(f"{top_name}\nConfidence: {top_conf:.1%}", fontsize=12)
            axes[0].axis("off")

            names = [CLASS_NAMES[i] for i in top_indices[:5]]
            confs = [probs[i] for i in top_indices[:5]]
            colors = ["#27ae60"] + ["#3498db"] * 4
            axes[1].barh(names[::-1], confs[::-1], color=colors[::-1])
            axes[1].set_xlabel("Confidence")
            axes[1].set_xlim(0, 1)
            axes[1].set_title("Top 5 Predictions")
            for i, c in enumerate(confs[::-1]):
                axes[1].text(c + 0.01, i, f"{c:.1%}", va="center", fontsize=9)

            plt.suptitle(
                f"Celebrity Guesser — {os.path.basename(img_path)}", fontsize=13
            )
            plt.tight_layout()
            plt.show()

    return {
        "top_predictions": [(CLASS_NAMES[i], float(probs[i])) for i in top_indices[:5]]
    }


# ============================================================
# DEMO PREDICTIONS
# ============================================================
def run_demo():
    print("\n" + "=" * 60)
    print("  DEMO PREDICTIONS")
    print("=" * 60)

    demo_count = 0
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(DATA_DIR, class_name)
        images = sorted(os.listdir(class_path))
        # Pick 3 random-ish images (indices 5, 15, 25)
        for idx in [5, 15, 25]:
            if idx < len(images):
                img_path = os.path.join(class_path, images[idx])
                print(f"\n[TRUE: {class_name}]")
                predict_image(img_path, show_img=True)
                demo_count += 1
                if demo_count >= 6:
                    break
        if demo_count >= 6:
            break


# ============================================================
# SUMMARY REPORT
# ============================================================
def print_summary(overall_acc):
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)

    cfg = load_config()
    if cfg:
        print(f"  Model type:        {cfg.get('MODEL_TYPE', 'unknown')}")
        print(f"  Image size:        {cfg.get('IMG_SIZE')}x{cfg.get('IMG_SIZE')}")
        print(f"  Total classes:     {cfg.get('NUM_CLASSES')}")
        print(f"  Total epochs:      {cfg.get('total_epochs_trained', 'N/A')}")
        print(f"  Best val accuracy: {cfg.get('best_val_accuracy', 'N/A')}")
    print()
    print(f"  Model file:         {model_path}")
    print(f"  Evaluation accuracy: {overall_acc:.2%}")
    print(f"  Saved outputs:      {MODEL_DIR}/confusion_matrix.png")
    print(f"                      {MODEL_DIR}/partial_test.png")
    print()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Celebrity Guesser Evaluation")
    parser.add_argument(
        "--predict", "-p", type=str, help="Path to image for prediction"
    )
    parser.add_argument(
        "--crop",
        "-c",
        type=str,
        help="Comma-separated crop types for prediction (eyes,lower,left_half,right_half,center)",
    )
    parser.add_argument(
        "--demo", "-d", action="store_true", help="Run demo predictions"
    )
    parser.add_argument(
        "--partial", action="store_true", help="Run partial image test only"
    )
    args = parser.parse_args()

    if args.predict:
        crops = args.crop.split(",") if args.crop else None
        predict_image(args.predict, crop_types=crops)
    elif args.partial:
        test_partial_images()
    elif args.demo:
        run_demo()
    else:
        # Full evaluation by default
        overall_acc, predictions, labels = evaluate_model()
        test_partial_images()
        print_summary(overall_acc)
        print("\nTo predict an image:")
        print("  python evaluate.py --predict path/to/image.jpg")
        print("  python evaluate.py --predict path/to/image.jpg --crop eyes,left_half")
        print("\nTo run partial image test only:")
        print("  python evaluate.py --partial")
        print("\nTo run demo predictions:")
        print("  python evaluate.py --demo")
