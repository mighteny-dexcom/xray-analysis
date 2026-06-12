# infer_lot4.py
# Run trained ROI model on Lot4 images (no labels)

import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = Path("lot4_images")

MODEL_PATH = Path(
    "models/best_screening_resnet18_all_lots_split_roi.pth"
)

OUTPUT_CSV = "lot4_predictions.csv"

THRESHOLD = 0.40

# Must match training
USE_ROI_CROP = True
ROI_BOX = (450, 350, 650, 500)


# ============================================================
# ROI
# ============================================================

def apply_roi_crop(img):
    if not USE_ROI_CROP:
        return img
    return img.crop(ROI_BOX)


# ============================================================
# MODEL
# ============================================================

def build_model():
    model = models.resnet18(weights=None)

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
# MAIN
# ============================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load model
    model = build_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Transform (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    results = []

    image_files = list(IMAGE_DIR.glob("*.png"))

    print(f"Found {len(image_files)} images")

    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")

        img = apply_roi_crop(img)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img).view(-1)
            prob = torch.sigmoid(output).item()

        pred = int(prob >= THRESHOLD)

        results.append({
            "filename": img_path.name,
            "probability": prob,
            "predicted_label": pred
        })

    df = pd.DataFrame(results)

    df = df.sort_values(by="probability", ascending=False)

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved predictions to: {OUTPUT_CSV}")

    print("\nTop flagged images (likely defects):")
    print(df.head(10))

    print("\nLowest probability images (likely good):")
    print(df.tail(10))


if __name__ == "__main__":
    main()