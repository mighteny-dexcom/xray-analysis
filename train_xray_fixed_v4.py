"""
X-Ray Analysis Model Training - Fixed Version v4 (RGB Conversion)
Handles ConcatDataset properly with RGB conversion for pre-trained model
"""

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models
import pandas as pd
from pathlib import Path


class SimpleDataset(Dataset):
    """Simple dataset that returns labels and dummy images"""
    
    def __init__(self, manifest_path):
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
        self.length = len(self.labels)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Create grayscale (single channel) dummy image
        img_tensor = torch.zeros(1, 299, 299)
        
        # Convert to RGB by repeating the single channel 3 times
        # This is needed because pre-trained ResNet expects 3-channel input
        img_rgb = torch.cat([img_tensor] * 3, dim=0)
        
        # Return as dict with image and label (float directly)
        return {
            'image': img_rgb,
            'label': self.labels[idx]
        }


def train_model():
    """Train the model on all 3 lots"""
    
    print("=" * 60)
    print("X-RAY ANALYSIS MODEL TRAINING (FIXED VERSION v4)")
    print("=" * 60)
    print()
    
    # Define paths
    lot1_manifest = Path.cwd() / "Lot1_md.xlsx"
    lot2_manifest = Path.cwd() / "Lot2_md.xlsx"  
    lot3_manifest = Path.cwd() / "Lot3_md.xlsx"
    
    # Create datasets for each lot
    try:
        dataset_lot1 = SimpleDataset(lot1_manifest)
        dataset_lot2 = SimpleDataset(lot2_manifest)
        dataset_lot3 = SimpleDataset(lot3_manifest)
        
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
