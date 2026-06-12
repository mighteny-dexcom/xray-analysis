# train_xray_v2_screening.py
# Train ResNet18 transfer-learning model for battery tab issue detection
# Goal: minimize false negatives while keeping false positives reasonable

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # Keep 0 for Windows stability

# Set this to "lot1", "lot2", or "lot3" for leave-one-lot-out validation.
# Example: HOLDOUT_LOT = "lot2" means train on lot1 + lot3, validate on lot2.
HOLDOUT_LOT = "lot1"

# If horizontal flipping would make the image physically invalid, set this False.
USE_HORIZONTAL_FLIP = True

# Screening model-selection goal:
# Prioritize minimizing false negatives while keeping false positives reasonable.
# If this is too strict, the fallback model will still save the best balanced model.
MIN_ACCEPTABLE_SPECIFICITY = 0.8

# Decision threshold used during validation.
# Lower threshold usually increases sensitivity but also increases false positives.
DECISION_THRESHOLD = 0.5


# ============================================================
# DATASET
# ============================================================

class SimpleMultiLotDataset(Dataset):
    """Dataset for Lot1, Lot2, Lot3 X-ray images."""

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
            """
            Converts annotation into binary label:
            0 = functional / no battery tab issue
            1 = BT / pBT / Partial BT issue
            """

            if pd.isna(label):
                return 0.0

            text = str(label).strip().lower()

            if text in ["", "nan", "none", "null"]:
                return 0.0

            # Guard against accidental false positives like "No BT"
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
            """
            Handles TXID values that may be read as numeric from Excel.
            Example:
            123456789012.0 -> 123456789012.png
            """

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

    # Freeze all pretrained layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last ResNet block only
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final classifier for binary classification
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

def compute_binary_metrics(outputs, labels, threshold=0.5):
    probs = torch.sigmoid(outputs)
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

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy
    }


def evaluate_model(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device)

            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(data_loader)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    metrics = compute_binary_metrics(
        all_outputs,
        all_labels,
        threshold=DECISION_THRESHOLD
    )

    return avg_loss, metrics


# ============================================================
# TRAINING
# ============================================================

def train_all_lots_model():
    print("=" * 60)
    print("TRAIN RESNET18 TRANSFER MODEL - SCREENING MODE")
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

    # Training transform: includes augmentation
    train_transform_list = [
        transforms.Resize((224, 224)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20
            )
        ], p=0.70),
        transforms.RandomRotation(degrees=5),
    ]

    if USE_HORIZONTAL_FLIP:
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_transform = transforms.Compose(train_transform_list)

    # Validation transform: no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_datasets = []
    val_datasets = []

    for lot_name in excel_files.keys():
        if lot_name == HOLDOUT_LOT:
            dataset = SimpleMultiLotDataset(
                manifest_path=excel_files[lot_name],
                image_dir=image_dirs[lot_name],
                lot_name=lot_name,
                transform=val_transform
            )
            val_datasets.append(dataset)
        else:
            dataset = SimpleMultiLotDataset(
                manifest_path=excel_files[lot_name],
                image_dir=image_dirs[lot_name],
                lot_name=lot_name,
                transform=train_transform
            )
            train_datasets.append(dataset)

    if len(train_datasets) == 0:
        print("[ERROR] No training datasets created.")
        return

    if len(val_datasets) == 0:
        print("[ERROR] No validation datasets created.")
        return

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    total_positive = sum(
        int(sum(dataset.labels))
        for dataset in train_datasets
    )

    total_negative = len(train_dataset) - total_positive

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Holdout validation lot:       {HOLDOUT_LOT}")
    print(f"Training samples:             {len(train_dataset)}")
    print(f"Validation samples:           {len(val_dataset)}")
    print(f"Train positive BT/pBT:        {total_positive}")
    print(f"Train negative functional:    {total_negative}")
    print(f"Decision threshold:           {DECISION_THRESHOLD}")
    print(f"Min acceptable specificity:   {MIN_ACCEPTABLE_SPECIFICITY}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = build_model()
    model.to(device)

    if total_positive > 0:
        pos_weight_value = total_negative / total_positive
        pos_weight = torch.tensor([pos_weight_value], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using pos_weight: {pos_weight_value:.3f}")
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("[WARNING] No positive samples found in training set. Not using pos_weight.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Primary screening model trackers
    best_sensitivity = -1.0
    best_specificity = -1.0
    best_balanced_accuracy = -1.0
    best_fn = float("inf")
    best_fp = float("inf")

    # Fallback model trackers
    best_fallback_balanced_accuracy = -1.0
    best_fallback_sensitivity = -1.0
    best_fallback_specificity = -1.0
    best_fallback_fn = float("inf")
    best_fallback_fp = float("inf")

    found_model_meeting_specificity = False

    best_model_path = models_dir / f"best_screening_resnet18_holdout_{HOLDOUT_LOT}.pth"
    fallback_model_path = models_dir / f"best_fallback_resnet18_holdout_{HOLDOUT_LOT}.pth"
    last_model_path = models_dir / f"last_resnet18_holdout_{HOLDOUT_LOT}.pth"

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        model.train()

        train_loss_total = 0.0
        train_outputs_all = []
        train_labels_all = []

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device)

            optimizer.zero_grad()

            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

            train_outputs_all.append(outputs.detach().cpu())
            train_labels_all.append(labels.detach().cpu())

        scheduler.step()

        avg_train_loss = train_loss_total / len(train_loader)

        train_outputs_all = torch.cat(train_outputs_all)
        train_labels_all = torch.cat(train_labels_all)

        train_metrics = compute_binary_metrics(
            train_outputs_all,
            train_labels_all,
            threshold=DECISION_THRESHOLD
        )

        val_loss, val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.3f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.3f} | "
            f"Val BalAcc: {val_metrics['balanced_accuracy']:.3f} | "
            f"Val Sens: {val_metrics['sensitivity']:.3f} | "
            f"Val Spec: {val_metrics['specificity']:.3f} | "
            f"Val Prec: {val_metrics['precision']:.3f} | "
            f"Val F1: {val_metrics['f1']:.3f}"
        )

        print(
            f"             Val Confusion: "
            f"TP={val_metrics['tp']}, "
            f"TN={val_metrics['tn']}, "
            f"FP={val_metrics['fp']}, "
            f"FN={val_metrics['fn']}"
        )

        current_sensitivity = val_metrics["sensitivity"]
        current_specificity = val_metrics["specificity"]
        current_balanced_accuracy = val_metrics["balanced_accuracy"]
        current_fn = val_metrics["fn"]
        current_fp = val_metrics["fp"]

        meets_specificity_requirement = (
            current_specificity >= MIN_ACCEPTABLE_SPECIFICITY
        )

        # ====================================================
        # Fallback model:
        # Always keep the best balanced model, even if it does
        # not meet the specificity requirement.
        # This prevents saving a useless all-negative model.
        # ====================================================

        is_better_fallback_model = False

        if current_sensitivity > 0:
            if current_balanced_accuracy > best_fallback_balanced_accuracy:
                is_better_fallback_model = True

            elif (
                current_balanced_accuracy == best_fallback_balanced_accuracy
                and current_fn < best_fallback_fn
            ):
                is_better_fallback_model = True

            elif (
                current_balanced_accuracy == best_fallback_balanced_accuracy
                and current_fn == best_fallback_fn
                and current_fp < best_fallback_fp
            ):
                is_better_fallback_model = True

        if is_better_fallback_model:
            best_fallback_balanced_accuracy = current_balanced_accuracy
            best_fallback_sensitivity = current_sensitivity
            best_fallback_specificity = current_specificity
            best_fallback_fn = current_fn
            best_fallback_fp = current_fp

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "holdout_lot": HOLDOUT_LOT,
                "decision_threshold": DECISION_THRESHOLD,
                "min_acceptable_specificity": MIN_ACCEPTABLE_SPECIFICITY,
                "train_positive": total_positive,
                "train_negative": total_negative,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "model_selection": "fallback_best_balanced_accuracy"
            }, fallback_model_path)

            print(
                f"             [OK] New fallback model saved: {fallback_model_path} "
                f"(BalAcc={best_fallback_balanced_accuracy:.3f}, "
                f"Sens={best_fallback_sensitivity:.3f}, "
                f"Spec={best_fallback_specificity:.3f}, "
                f"FN={best_fallback_fn}, FP={best_fallback_fp})"
            )

        # ====================================================
        # Primary screening model:
        # Save only if it meets specificity requirement.
        # Within that constraint, prioritize sensitivity.
        # ====================================================

        is_better_screening_model = False

        if meets_specificity_requirement and current_sensitivity > 0:
            found_model_meeting_specificity = True

            # Primary goal: minimize FN by maximizing sensitivity.
            if current_sensitivity > best_sensitivity:
                is_better_screening_model = True

            # Tie-breaker 1: if sensitivity is equal, choose fewer false negatives.
            elif current_sensitivity == best_sensitivity and current_fn < best_fn:
                is_better_screening_model = True

            # Tie-breaker 2: if FN is also equal, choose fewer false positives.
            elif (
                current_sensitivity == best_sensitivity
                and current_fn == best_fn
                and current_fp < best_fp
            ):
                is_better_screening_model = True

            # Tie-breaker 3: if FN/FP are equal, choose better balanced accuracy.
            elif (
                current_sensitivity == best_sensitivity
                and current_fn == best_fn
                and current_fp == best_fp
                and current_balanced_accuracy > best_balanced_accuracy
            ):
                is_better_screening_model = True

        if is_better_screening_model:
            best_sensitivity = current_sensitivity
            best_specificity = current_specificity
            best_balanced_accuracy = current_balanced_accuracy
            best_fn = current_fn
            best_fp = current_fp

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "holdout_lot": HOLDOUT_LOT,
                "decision_threshold": DECISION_THRESHOLD,
                "min_acceptable_specificity": MIN_ACCEPTABLE_SPECIFICITY,
                "train_positive": total_positive,
                "train_negative": total_negative,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "model_selection": "screening_highest_sensitivity_with_specificity_constraint"
            }, best_model_path)

            print(
                f"             [OK] New best screening model saved: {best_model_path} "
                f"(Sens={best_sensitivity:.3f}, Spec={best_specificity:.3f}, "
                f"FN={best_fn}, FP={best_fp})"
            )

    # Save last model after all epochs
    torch.save({
        "epoch": NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "holdout_lot": HOLDOUT_LOT,
        "decision_threshold": DECISION_THRESHOLD,
        "min_acceptable_specificity": MIN_ACCEPTABLE_SPECIFICITY,
        "train_positive": total_positive,
        "train_negative": total_negative
    }, last_model_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    if found_model_meeting_specificity:
        print(f"Best screening sensitivity:        {best_sensitivity:.3f}")
        print(f"Best screening specificity:        {best_specificity:.3f}")
        print(f"Best screening balanced accuracy:  {best_balanced_accuracy:.3f}")
        print(f"Best screening FN:                 {best_fn}")
        print(f"Best screening FP:                 {best_fp}")
        print(f"Best screening model saved to:     {best_model_path}")
    else:
        print("[WARNING] No model met the minimum specificity requirement.")
        print("          Use the fallback model or lower MIN_ACCEPTABLE_SPECIFICITY.")

    print(f"Best fallback sensitivity:         {best_fallback_sensitivity:.3f}")
    print(f"Best fallback specificity:         {best_fallback_specificity:.3f}")
    print(f"Best fallback balanced accuracy:   {best_fallback_balanced_accuracy:.3f}")
    print(f"Best fallback FN:                  {best_fallback_fn}")
    print(f"Best fallback FP:                  {best_fallback_fp}")
    print(f"Fallback model saved to:           {fallback_model_path}")
    print(f"Last model saved to:               {last_model_path}")


if __name__ == "__main__":
    train_all_lots_model()