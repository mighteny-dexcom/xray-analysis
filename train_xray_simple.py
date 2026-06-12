"""
X-Ray Analysis Model Training Script (Simplified Version)
Uses updated categorization logic where NaN/null values are treated as negative (functional)
Handles cases where manifest doesn't have image filename column
"""

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
import os
import glob

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class XRayDataset(Dataset):
    """Simplified dataset that uses actual image files from folders"""
    
    def __init__(self, lot_num, manifest_path, images_dir):
        self.lot_num = lot_num
        
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
        
        # Get image filenames from manifest - try different column names
        filename_col = None
        for col in ['Image Filename', 'Filename', 'File Name', 'Name']:
            if col in df.columns:
                filename_col = col
                break
        
        if filename_col and len(df[filename_col]) > 0:
            # Use filenames from manifest
            self.filenames = [str(filename).strip() for filename in df[filename_col]]
            print(f"[INFO] Lot {lot_num}: Using filenames from '{filename_col}' column")
        else:
            # Fallback: generate sequential filenames based on image count
            images_dir_path = Path(images_dir)
            if images_dir_path.exists():
                all_images = list(glob.glob(str(images_dir_path / "*.png"))) + \
                             list(glob.glob(str(images_dir_path / "*.jpg")))
                self.filenames = [os.path.basename(img) for img in sorted(all_images)]
                print(f"[INFO] Lot {lot_num}: Found {len(self.filenames)} images in folder")
            else:
                # Create dummy dataset if no images found (for testing)
                print(f"[WARNING] Lot {lot_num}: No images directory found at '{images_dir}'")
                self.filenames = []
        
        self.length = len(self.labels)
        print(f"[INFO] Lot {lot_num}: Loaded {self.length} records with labels")
    
    def __len__(self):
        return max(len(self.labels), len(self.filenames))  # Use whichever is larger
    
    def __getitem__(self, idx):
        if idx >= self.length:
            return {'image': torch.zeros(3, 299, 299), 'label': torch.tensor(0.0)}
        
        label = self.labels[idx]
        
        # Get filename - use manifest filename or generate one
        if len(self.filenames) > idx:
            filename = self.filenames[idx]
        else:
            filename = f"image_{idx}.png"
        
        # Construct image path
        images_dir_path = Path(f"C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\lot{self.lot_num}_images")
        full_path = images_dir_path / filename
        
        # Load and preprocess image
        try:
            if not full_path.exists():
                # Try with different extensions
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = images_dir_path / (filename.replace('.png', ext))
                    if test_path.exists():
                        full_path = test_path
                        break
            
            img = Image.open(full_path).convert('RGB')
            
            # Resize to standard size
            img = img.resize((299, 299))
            
            # Convert to tensor and normalize (ImageNet normalization)
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img)
            
            return {
                'image': img_tensor,
                'label': torch.tensor(label),
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
    
    # Define paths
    base_dir = Path.cwd()
    lot1_manifest = base_dir / "Lot1_md.xlsx"
    lot2_manifest = base_dir / "Lot2_md.xlsx"  
    lot3_manifest = base_dir / "Lot3_md.xlsx"
    
    # Image directories (hardcoded for your setup)
    lot1_images_dir = "C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\lot1_images"
    lot2_images_dir = "C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\lot2_images"
    lot3_images_dir = "C:\\Users\\ug10271\\OneDrive - Dexcom\\Documents\\CodeProjects\\xray-analysis\\lot3_images"
    
    # Create datasets for each lot
    try:
        dataset_lot1 = XRayDataset(1, str(lot1_manifest), lot1_images_dir)
        dataset_lot2 = XRayDataset(2, str(lot2_manifest), lot2_images_dir)
        dataset_lot3 = XRayDataset(3, str(lot3_manifest), lot3_images_dir)
        
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