"""
evaluate_model.py

Usage:
    python evaluate_model.py --model-path models/warehouse_resnet_final --data-dir data/mapped_coco --batch-size 8 --device cpu

What it does:
- Loads a Hugging Face image classification model from `--model-path`
- Loads images from `--data-dir` using `datasets.load_dataset('imagefolder')` (expects train/test folders)
- Runs batched inference on the `test` split (falls back to `validation` or `train` if `test` missing)
- Computes Accuracy, Precision, Recall, F1, and Confusion Matrix
- Writes a textual report to `results/performance_report.txt` and saves `results/confusion_matrix.png`

Requirements:
- datasets
- transformers
- torch
- scikit-learn
- matplotlib

"""
import os
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="models/warehouse_resnet_final", help="Path to the HF image classification model directory")
    p.add_argument("--data-dir", type=str, default="data/mapped_coco", help="Path to the folder containing train/ and test/ image folders")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default=None, help="cpu or cuda (auto if not set)")
    p.add_argument("--output-dir", type=str, default="results")
    return p.parse_args()


class SimpleTimer:
    import time
    def __init__(self):
        self.start = self.time.time()
    def elapsed(self):
        return self.time.time() - self.start


def load_split(data_dir):
    # Try to load with datasets imagefolder
    ds = load_dataset("imagefolder", data_dir=data_dir)
    if "test" in ds:
        split = ds["test"]
    elif "validation" in ds:
        split = ds["validation"]
    elif "train" in ds:
        # If only train exists, use it (best-effort)
        split = ds["train"]
    else:
        raise RuntimeError("No suitable split found in data directory")
    return split, ds


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path} on device {device}...")
    processor = AutoImageProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    print(f"Loading dataset from {args.data_dir}...")
    split, ds_all = load_split(args.data_dir)

    # Build label mappings using the model's config.id2label when available.
    # This ensures predicted indices from the model align with label names.
    model_id2label = None
    if hasattr(model.config, "id2label") and model.config.id2label is not None:
        try:
            model_id2label = {int(k): v for k, v in model.config.id2label.items()}
        except Exception:
            # keys might already be ints
            model_id2label = {int(k): v for k, v in model.config.id2label.items()}

    dataset_label_names = None
    try:
        dataset_label_names = split.features["label"].names
    except Exception:
        dataset_label_names = None

    if model_id2label is None and dataset_label_names is None:
        raise RuntimeError("Could not determine label names from dataset or model config.")

    if model_id2label is not None:
        # Prefer model mapping for consistency with predictions
        label_names = [model_id2label[i] for i in sorted(model_id2label.keys())]
        # Build inverse map name->model_idx
        model_name_to_idx = {v: k for k, v in model_id2label.items()}
        # If dataset provides label names, build mapping from dataset idx -> model idx
        dataset_to_model_idx = None
        if dataset_label_names is not None:
            dataset_to_model_idx = {i: model_name_to_idx[dataset_label_names[i]] for i in range(len(dataset_label_names))}
        print("Using model.config.id2label for label mapping:", model_id2label)
    else:
        # Fall back to dataset ordering
        label_names = dataset_label_names
        dataset_to_model_idx = {i: i for i in range(len(label_names))}
        print("Using dataset label ordering:", label_names)

    # Collect predictions
    y_true = []
    y_pred = []

    batch_size = args.batch_size
    n = len(split)
    indices = list(range(0, n, batch_size))

    for idx, i in enumerate(indices):
        batch = split[i:i+batch_size]
        # HuggingFace datasets may return a dict of lists when slicing
        if isinstance(batch, dict):
            images = batch.get("image")
            labels = batch.get("label")
        else:
            # When batch is an iterable of examples
            images = [sample["image"] for sample in batch]
            labels = [sample["label"] for sample in batch]

        # Progress
        if (idx + 1) % 10 == 0 or (i + batch_size) >= n:
            print(f"Processing samples {i}-{min(i+batch_size, n)} / {n}")

        # Processor accepts PIL images or numpy arrays
        inputs = processor(images=images, return_tensors="pt")
        # Move tensors to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        y_true.extend(labels)
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Align dataset label indices to model label indices when needed
    if 'dataset_to_model_idx' in locals() and dataset_to_model_idx is not None:
        y_true_mapped = np.array([dataset_to_model_idx[int(i)] for i in y_true])
    else:
        # assume dataset indices already match model indices
        y_true_mapped = y_true.copy()

    # y_pred are model indices already
    y_pred_model = y_pred.copy()

    # For human-readable output, map to label names (ordered by model indices)
    y_true_names = [label_names[int(i)] for i in y_true_mapped]
    y_pred_names = [label_names[int(i)] for i in y_pred_model]

    acc = accuracy_score(y_true_mapped, y_pred_model)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_mapped, y_pred_model, average="weighted", zero_division=0)

    print(f"Samples evaluated: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Weighted F1: {f1:.4f}")

    # Classification report and confusion matrix
    report = classification_report(y_true_mapped, y_pred_model, target_names=label_names, zero_division=0)
    cm = confusion_matrix(y_true_mapped, y_pred_model)

    # Save textual report
    report_path = Path(args.output_dir) / "performance_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Samples evaluated: {len(y_true)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Weighted F1: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"Saved textual report to {report_path}")

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = Path(args.output_dir) / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    print(f"Saved confusion matrix plot to {cm_path}")
    print("Done.")


if __name__ == "__main__":
    main()
