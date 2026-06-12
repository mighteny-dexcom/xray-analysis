"""
X-Ray Analysis Model Training Script
Uses updated categorization logic where NaN/null values are treated as negative (functional)
"""

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class XRayDataset(Dataset):
    """Dataset class for X-ray analysis with updated categorization logic"""
    
    def __init__(self, lot_num, image_dir, manifest_path):
        self.lot_num = lot_num
        self.image_dir = Path(image_dir)
        
        # Load manifest data
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            
        df = pd.read_excel(manifest_path)
        
        # Updated categorization logic - NaN/null treated as functional (negative)
        def normalize_label(label):
            if pd.isna(label) or str(label).strip() == '':
                return 0.0  # Negative: functional/blank
            elif 'BT' in str(label) or 'pBT' in str(label) or 'Partial BT' in str(label):
                return 1.0  # Positive: battery tags (including partial BT)
            else:
                return 0.0  # Negative: everything else
        
        self.labels = df['Xray Gross Issues'].apply(normalize_label).tolist()
        
        # Get image filenames from manifest (adjust column name as needed)
        if 'Image Filename' in df.columns:
            self.filenames = [str(filename).strip() for filename in df['Image Filename']]
        elif 'Filename' in df.columns:
            self.filenames = [str(filename).strip() for filename in df['Filename']]
        else:
            # Fallback: assume filenames match manifest row indices
            print(f"[WARNING] No image filename column found in Lot {lot_num} manifest")
            self.filenames = [f"image_{i}.png" for i in range(len(df))]
        
        self.length = len(self.labels)
        print(f"[INFO] Lot {lot_num}: Loaded {self.length} records")
        print(f"[INFO] Positive samples: {sum(self.labels)} (BT/pBT)")
        print(f"[INFO] Negative samples: {len(self.labels) - sum(self.labels)} (Functional/NaN)")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Construct image path
        if not os.path.exists(filename):
            # Try common variations
            possible_paths = [
                f"{filename}",
                f"image_{idx}.png",
                f"{idx:04d}.png",
            ]
            
            for possible_path in possible_paths:
                full_path = self.image_dir / possible_path
                if full_path.exists():
                    filename = possible_path
                    break
            
        # Load and preprocess image
        try:
            img = Image.open(self.image_dir / filename).convert('RGB')
            # Resize to standard size (adjust as needed for your model)
            img = img.resize((299, 299))
            
            # Convert to tensor and normalize
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img)
            
            return {
                'image': img_tensor,
                'label': torch.tensor(self.labels[idx]),
                'filename': filename
            }
        except Exception as e:
            print(f"[ERROR] Failed to load image {filename}: {str(e)}")
            # Return dummy data for missing images (will be filtered during training)
            return {
                'image': torch.zeros(3, 299, 299),
                'label': torch.tensor(0.0),
                'filename': filename + '_missing'
            }


def train_model():
    """Train the model on all 3 lots with updated categorization"""
    
    print("=" * 60)
    print("X-RAY ANALYSIS MODEL TRAINING")
    print("=" * 60)
    print()
    
    # Define paths (adjust as needed for your setup)
    base_dir = Path.cwd()
    lot1_manifest = base_dir / "Lot1_md.xlsx"
    lot2_manifest = base_dir / "Lot2_md.xlsx"  
    lot3_manifest = base_dir / "Lot3_md.xlsx"
    
    # Image directories (adjust paths as needed)
    lot1_images_dir = base_dir / "lot1_images"
    lot2_images_dir = base_dir / "lot2_images"
    lot3_images_dir = base_dir / "lot3_images"
    
    # Create datasets for each lot
    try:
        dataset_lot1 = XRayDataset(1, str(lot1_images_dir), str(lot1_manifest))
        dataset_lot2 = XRayDataset(2, str(lot2_images_dir), str(lot2_manifest))
        dataset_lot3 = XRayDataset(3, str(lot3_images_dir), str(lot3_manifest))
        
    except Exception as e:
        print(f"[ERROR] Failed to create datasets: {str(e)}")
        return
    
    # Combine all datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([dataset_lot1, dataset_lot2, dataset_lot3])
    
    train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True)
    
    print(f"\nTotal training samples: {len(combined_dataset)}")
    total_positive = sum([sum(d.labels) for d in [dataset_lot1, dataset_lot2, dataset_lot3]])
    total_negative = len(combined_dataset) - total_positive
    print(f"  Positive (BT/pBT): {total_positive}")
    print(f"  Negative (Functional/NaN): {total_negative}")
    
    # Create model using transfer learning
    from torchvision import models
    
    # Use a pre-trained ResNet18 as base (lightweight, good for small datasets)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)    
    # Freeze early layers (transfer learning)
    for param in list(model.parameters())[:-5]:  # Keep last few layers trainable
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
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for classification
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    num_epochs = 20
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        
        for batch_idx, (images, labels, filenames) in enumerate(train_loader):
            images = images.to(device).unsqueeze(1)  # Add channel dim if needed
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), labels.float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            correct_predictions += (predictions == labels.float()).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / len(combined_dataset) * 100
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'models/xray_tab_detection_model.pth')
    print("\n[OK] Model saved successfully to models/xray_tab_detection_model.pth!")


if __name__ == "__main__":
    train_model()
