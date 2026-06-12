"""
X-Ray Analysis Model Training - Real Images Version
Loads actual X-ray images from manifest filenames
"""

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models, transforms
import pandas as pd
from pathlib import Path


class XRayDataset(Dataset):
    """Dataset that loads real X-ray images and returns labels"""
    
    def __init__(self, manifest_path, image_folder='lot1_images'):
        df = pd.read_excel(manifest_path)
        
        # Updated categorization logic - NaN/null treated as functional (negative)
        def normalize_label(label):
            if pd.isna(label) or str(label).strip() == '':
                return 0.0
            elif 'BT' in str(label) or 'pBT' in str(label) or 'Partial BT' in str(label):
                return 1.0
            else:
                return 0.0
        
        self.labels = df['Xray Gross Issues'].apply(normalize_label).tolist()
        
        # Build image paths from TXID column (remove .png extension if exists)
        self.image_paths = []
        for idx, row in df.iterrows():
            txid = str(row['TXID'])
            img_path = Path(image_folder) / f"{txid}.png"
            self.image_paths.append(img_path)
        
        # Create transform pipeline for X-ray images
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])
        
        self.length = len(self.labels)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load real image from file
        img_path = self.image_paths[idx]
        try:
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img_pil)  # Already RGB with 3 channels
        except Exception as e:
            print(f"[WARN] Could not load image {img_path}: {str(e)}")
            # Return dummy image if loading fails (for robustness during training)
            img_tensor = torch.zeros(3, 299, 299)
        
        return {
            'image': img_tensor,
            'label': self.labels[idx]
        }


def train_model():
    """Train the model on all 3 lots with real images"""
    
    print("=" * 60)
    print("X-RAY ANALYSIS MODEL TRAINING (REAL IMAGES)")
    print("=" * 60)
    print()
    
    # Define paths
    lot1_manifest = Path.cwd() / "Lot1_md.xlsx"
    lot2_manifest = Path.cwd() / "Lot2_md.xlsx"  
    lot3_manifest = Path.cwd() / "Lot3_md.xlsx"
    
    # Create datasets for each lot (specify image folder based on lot)
    try:
        dataset_lot1 = XRayDataset(lot1_manifest, image_folder='lot1_images')
        dataset_lot2 = XRayDataset(lot2_manifest, image_folder='lot2_images')
        dataset_lot3 = XRayDataset(lot3_manifest, image_folder='lot3_images')
        
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
    print("STARTING TRAINING (5 epochs with real images)")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract image and label from dict
            images = batch['image']
            labels = torch.tensor(batch['label'])
            
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
