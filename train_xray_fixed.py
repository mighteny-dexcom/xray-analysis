"""
X-Ray Analysis Model Training - Fixed Version v4 (RGB Conversion)
Handles ConcatDataset properly with RGB conversion for pre-trained model
"""

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from pathlib import Path


class SimpleDataset(Dataset):
    """Simple dataset that returns labels and actual X-ray images"""
    
    def __init__(self, manifest_path, image_dir):
        df = pd.read_excel(manifest_path)
        
        # Updated categorization logic - NaN/null treated as functional (negative)
        def normalize_label(label):
            if pd.isna(label) or str(label).strip() == '':
                return 0.0
            elif 'BT' in str(label) or 'pBT' in str(label) or 'Partial BT' in str(label):
                return 1.0
            else:
                return 0.0

        # Column D is TXID (image filenames), Column E is Xray Gross Issues (labels)
        self.image_filenames = df['TXID'].tolist()
        self.labels = df['Xray Gross Issues'].apply(normalize_label).tolist()
        self.length = len(self.labels)

        # Load actual images from directory
        self.image_dir = Path(image_dir)
        self.transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get image ID from TXID column (Column D) - these are numeric IDs
        img_id = self.image_filenames[idx]

        # Make sure it's a string and handle any path issues
        if isinstance(img_id, str):
            img_filename = img_id.strip()
            # If it contains backslashes (Windows paths), extract just the filename
            if '\\' in img_filename:
                img_filename = img_filename.split('\\')[-1]
        elif not isinstance(img_id, str):
            print(f"[WARNING] TXID value is {type(img_id).__name__}, converting to string")
            img_filename = str(img_id)

        # Add .png extension if not present (images are named like 123456789.png)
        if not img_filename.lower().endswith('.png'):
            img_filename += '.png'

        # Construct full image path for loading images
        img_path = self.image_dir / img_filename
        
        try:
            # Load and transform the actual X-ray image from absolute path (Windows style preferred)
            image = transforms.functional.pil_to_tensor(
                Image.open(img_path).convert('RGB')
            )

            return {
                'image': self.transforms(image),
                'label': float(self.labels[idx]) if self.labels[idx] is not None else 0.5,
            }
        except Exception as e:
            print(f"[WARNING] Could not load image {img_filename}: {str(e)}")
            # Return dummy image if loading fails
            img_tensor = torch.zeros(3, 299, 299)
            return {'image': img_tensor, 'label': float(self.labels[idx]) if self.labels[idx] is not None else 0.5}


def train_model():
    """Train the model on all 3 lots"""
    
    print("=" * 60)
    print("X-RAY ANALYSIS MODEL TRAINING (FIXED VERSION v4)")
    print("=" * 60)
    print()
    
    # Define paths - Update with your actual image directories
    base_dir = Path(r"C:\Users\ug10271\OneDrive - Dexcom\Documents\CodeProjects\xray-analysis")
    
    lot1_manifest = base_dir / "Lot1_md.xlsx"
    lot2_manifest = base_dir / "Lot2_md.xlsx"  
    lot3_manifest = base_dir / "Lot3_md.xlsx"
    
    # Create datasets for each lot with image directories
    try:
        dataset_lot1 = SimpleDataset(lot1_manifest, base_dir / "lot1_images")
        dataset_lot2 = SimpleDataset(lot2_manifest, base_dir / "lot2_images")
        dataset_lot3 = SimpleDataset(lot3_manifest, base_dir / "lot3_images")
        
    except Exception as e:
        print(f"[ERROR] Failed to create datasets: {str(e)}")
        return
    
    # Combine all datasets
    combined_dataset = ConcatDataset([dataset_lot1, dataset_lot2, dataset_lot3])
    
    train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True)
    
    print(f"\nTotal training samples: {len(combined_dataset)}")
    total_positive = sum([sum(d.labels) for d in [dataset_lot1, dataset_lot2, dataset_lot3]])
    total_negative = len(combined_dataset) - total_positive
    print(f"  Positive (BT/pBT): {total_positive}")
    print(f"  Negative (Functional/NaN): {total_negative}")
    
    # Create model using transfer learning
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze early layers (transfer loading)
    for param in list(model.parameters())[:-5]:
        param.requires_grad = False
    
    # Replace final layer for binary classification
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_features, 1)
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model.to(device)
    
    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop - FIRST 5 EPOCHS ONLY for quick test
    num_epochs = 5
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING (5 epochs)")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract image and label from dict - this is the fix!
            images = batch['image']
            labels = torch.tensor(batch['label'])  # Convert to tensor
            
            # NO .unsqueeze(1) needed - already RGB with 3 channels
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), labels.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/xray_tab_detection_model.pth')
    print("\n[OK] Model saved successfully to models/xray_tab_detection_model.pth!")


if __name__ == "__main__":
    train_model()