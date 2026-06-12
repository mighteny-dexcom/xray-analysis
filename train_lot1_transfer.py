# train_lot1_transfer.py - Train ResNet18 model on Lot1 images only

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from pathlib import Path


class SimpleLot1Dataset(Dataset):
    """Simple dataset for Lot1 X-ray images"""
    
    def __init__(self, manifest_path, image_dir):
        self.image_dir = Path(image_dir)
        
        # Load manifest data
        if not Path(manifest_path).exists():
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
        
        # Get image filenames from TXID column
        if 'TXID' in df.columns:
            self.filenames = [f"{str(filename).strip()}.png" for filename in df['TXID']]
        elif 'Image Filename' in df.columns:
            self.filenames = [str(filename).strip() for filename in df['Image Filename']]
        else:
            print(f"[WARNING] No image filename column found, using row indices")
            self.filenames = [f"image_{i}.png" for i in range(len(df))]
        
        self.length = len(self.labels)
        print(f"[INFO] Lot1: Loaded {self.length} records")
        print(f"[INFO] Positive samples (BT/pBT): {sum(self.labels)}")
        print(f"[INFO] Negative samples (Functional/NaN): {len(self.labels) - sum(self.labels)}")

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Construct image path
        if not Path(filename).exists():
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


def train_lot1_model():
    """Train ResNet18 model on Lot1 images only"""
    
    print("=" * 60)
    print("TRAIN LOT1 MODEL (ResNet18 Transfer Learning)")
    print("=" * 60)
    print()
    
    # Define paths
    base_dir = Path(__file__).parent
    
    lot1_manifest = base_dir / "Lot1_md.xlsx"
    lot1_images_dir = base_dir / "lot1_images"
    
    # Create dataset for Lot1 only
    try:
        dataset_lot1 = SimpleLot1Dataset(lot1_manifest, str(lot1_images_dir))
        
    except Exception as e:
        print(f"[ERROR] Failed to create dataset: {str(e)}")
        return
    
    # Combine datasets (only Lot1 in this case)
    combined_dataset = ConcatDataset([dataset_lot1])
    
    train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True)
    
    print(f"\nTotal training samples: {len(combined_dataset)}")
    total_positive = sum(dataset_lot1.labels)
    total_negative = len(combined_dataset) - total_positive
    print(f"  Positive (BT/pBT): {total_positive}")
    print(f"  Negative (Functional/NaN): {total_negative}")
    
    # Create model using transfer learning with ResNet18
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
    
    # Training loop - FIRST 5 EPOCHS ONLY for quick test
    num_epochs = 20
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING (5 epochs)")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract image and label from dict
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
    torch.save(model.state_dict(), 'models/lot1_xray_tab_detection_model.pth')
    print("\n[OK] Model saved successfully to models/lot1_xray_tab_detection_model.pth!")


if __name__ == "__main__":
    train_lot1_model()