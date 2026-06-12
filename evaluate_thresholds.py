# evaluate_thresholds.py
# Evaluate a trained ResNet18 model across multiple decision thresholds
# Goal: choose threshold that minimizes FN while keeping FP reasonable

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
NUM_WORKERS = 0

# Which lot do you want to evaluate?
# This should match the holdout lot for the model you trained.
EVAL_LOT = "lot3"

# Path to trained checkpoint.
# For Lot1, you probably want to evaluate the fallback model first.
MODEL_PATH = "models/best_screening_resnet18_holdout_lot3.pth"

# Thresholds to evaluate.
THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90
]


# ============================================================
# DATASET
# ============================================================

class SimpleMultiLotDataset(Dataset):
    """Dataset for one lot of X-ray images."""

    def __init__(self, manifest_path, image_dir, lot_name, transform=None):
        self.image_dir = Path(image_dir)
        self.lot_name = lot_name
        self.transform = transform

        manifest_path = Path(manifest_path)

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        df = pd.read_excel(manifest_path, engine="openpyxl")

        if "Xray Gross Issues" not in df.columns:
            raise ValueError(f"'Xray Gross Issues' column not found in {manifest_path}")

        def normalize_label(label):
            if pd.isna(label):
                return 0.0

            text = str(label).strip().lower()

            if text in ["", "nan", "none", "null"]:
                return 0.0

            negative_phrases = [
                "no bt",
                "not bt",
                "non-bt",
                "no battery tab",
                "functional",
                "ok",
                "pass"
            ]

            for phrase in negative_phrases:
                if phrase in text:
                    return 0.0

            positive_phrases = [
                "partial battery tab",
                "partial bt",
                "battery tab",
                "pbt",
                "bt"
            ]

            for phrase in positive_phrases:
                if phrase in text:
                    return 1.0

            return 0.0

        labels = df["Xray Gross Issues"].apply(normalize_label).tolist()

        def normalize_txid(value):
            if pd.isna(value):
                return None

            if isinstance(value, float) and value.is_integer():
                return f"{int(value)}.png"

            text = str(value).strip()

            if text.endswith(".0"):
                text = text[:-2]

            if not text.lower().endswith(".png"):
                text = f"{text}.png"

            return text

        if "TXID" in df.columns:
            filenames = [normalize_txid(x) for x in df["TXID"]]
        elif "Image Filename" in df.columns:
            filenames = [str(x).strip() for x in df["Image Filename"]]
        else:
            raise ValueError(f"No image filename column found in {manifest_path}")

        self.filenames = []
        self.labels = []

        for filename, label in zip(filenames, labels):
            if filename is None:
                continue

            full_path = self.image_dir / filename

            if full_path.exists():
                self.filenames.append(filename)
                self.labels.append(label)
            else:
                print(f"[WARNING] Missing image skipped: {full_path}")

        self.length = len(self.labels)

        pos_count = sum(int(x) for x in self.labels)
        neg_count = self.length - pos_count

        print(f"[INFO] {lot_name}: {manifest_path.name}")
        print(f"       Loaded valid records: {self.length}")
        print(f"       Positive BT/pBT:      {pos_count}")
        print(f"       Negative Functional:  {neg_count}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = self.image_dir / filename

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            "image": img,
            "label": label,
            "filename": filename,
            "lot": self.lot_name
        }


# ============================================================
# MODEL
# ============================================================

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    num_features = model.fc.in_features

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(num_features, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 1)
    )

    return model


# ============================================================
# METRICS
# ============================================================

def compute_binary_metrics_from_probs(probs, labels, threshold):
    preds = (probs >= threshold).float()
    labels = labels.float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    balanced_accuracy = (sensitivity + specificity) / 2

    flagged_count = tp + fp
    missed_count = fn

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "flagged_count": flagged_count,
        "missed_count": missed_count
    }


def collect_model_outputs(model, data_loader, device):
    model.eval()

    all_probs = []
    all_labels = []
    all_filenames = []
    all_lots = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].float()

            outputs = model(images).view(-1)
            probs = torch.sigmoid(outputs).cpu()

            all_probs.append(probs)
            all_labels.append(labels)

            all_filenames.extend(batch["filename"])
            all_lots.extend(batch["lot"])

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    return all_probs, all_labels, all_filenames, all_lots


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_thresholds():
    print("=" * 60)
    print("THRESHOLD EVALUATION")
    print("=" * 60)

    base_dir = Path(__file__).parent

    excel_files = {
        "lot1": base_dir / "Lot1_md.xlsx",
        "lot2": base_dir / "Lot2_md.xlsx",
        "lot3": base_dir / "Lot3_md.xlsx"
    }

    image_dirs = {
        "lot1": base_dir / "lot1_images",
        "lot2": base_dir / "lot2_images",
        "lot3": base_dir / "lot3_images"
    }

    if EVAL_LOT not in excel_files:
        raise ValueError(f"EVAL_LOT must be one of {list(excel_files.keys())}")

    model_path = base_dir / MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Evaluating lot:     {EVAL_LOT}")
    print(f"Model checkpoint:   {model_path}")

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_dataset = SimpleMultiLotDataset(
        manifest_path=excel_files[EVAL_LOT],
        image_dir=image_dirs[EVAL_LOT],
        lot_name=EVAL_LOT,
        transform=eval_transform
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:       {device}")

    model = build_model()
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded checkpoint dictionary with model_state_dict.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw model state_dict.")

    model.to(device)

    probs, labels, filenames, lots = collect_model_outputs(
        model=model,
        data_loader=eval_loader,
        device=device
    )

    results = []

    print("\n" + "=" * 100)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 100)
    print(
        f"{'Thresh':>7} | "
        f"{'Sens':>6} | "
        f"{'Spec':>6} | "
        f"{'Prec':>6} | "
        f"{'F1':>6} | "
        f"{'BalAcc':>6} | "
        f"{'TP':>3} | "
        f"{'TN':>3} | "
        f"{'FP':>3} | "
        f"{'FN':>3} | "
        f"{'Flagged':>7}"
    )
    print("-" * 100)

    for threshold in THRESHOLDS:
        metrics = compute_binary_metrics_from_probs(
            probs=probs,
            labels=labels,
            threshold=threshold
        )

        results.append(metrics)

        print(
            f"{threshold:7.2f} | "
            f"{metrics['sensitivity']:6.3f} | "
            f"{metrics['specificity']:6.3f} | "
            f"{metrics['precision']:6.3f} | "
            f"{metrics['f1']:6.3f} | "
            f"{metrics['balanced_accuracy']:6.3f} | "
            f"{metrics['tp']:3d} | "
            f"{metrics['tn']:3d} | "
            f"{metrics['fp']:3d} | "
            f"{metrics['fn']:3d} | "
            f"{metrics['flagged_count']:7d}"
        )

    results_df = pd.DataFrame(results)

    output_csv = base_dir / f"threshold_sweep_{EVAL_LOT}.csv"
    results_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 100)
    print("SAVED RESULTS")
    print("=" * 100)
    print(f"Threshold sweep saved to: {output_csv}")

    # Also save image-level probabilities for later failure review
    image_results = pd.DataFrame({
        "filename": filenames,
        "lot": lots,
        "true_label": labels.numpy(),
        "probability": probs.numpy()
    })

    image_results_csv = base_dir / f"image_probabilities_{EVAL_LOT}.csv"
    image_results.to_csv(image_results_csv, index=False)

    print(f"Image-level probabilities saved to: {image_results_csv}")

    print("\nSuggested next step:")
    print("Open the threshold_sweep CSV and choose the lowest FN you can tolerate")
    print("while keeping FP/review burden reasonable.")


if __name__ == "__main__":
    evaluate_thresholds()