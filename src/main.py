import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from models.tab_detection import TabDetectionModel
from utils.data_loader import create_dataloader


def train_model():
    """Train the model on all 3 lots"""
    
    # Create datasets for each lot
    print("Loading data from Lot 1...")
    dataloader_lot1 = create_dataloader(1, batch_size=8)
    
    print("Loading data from Lot 2...")
    dataloader_lot2 = create_dataloader(2, batch_size=8)
    
    print("Loading data from Lot 3...")
    dataloader_lot3 = create_dataloader(3, batch_size=8)
    
    # Combine all datasets
    combined_dataset = ConcatDataset([dataloader_lot1.dataset, 
                                      dataloader_lot2.dataset, 
                                      dataloader_lot3.dataset])
    
    train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TabDetectionModel(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for images, labels, filenames in train_loader:
            images = images.to(device).unsqueeze(1).expand(-1, 3, -1, -1)  # Add channel dim
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/tab_detection_model.pth')
    print("Model saved successfully!")


def validate_model():
    """Validate and generate predictions"""
    from PIL import Image
    
    model = TabDetectionModel(pretrained=True)
    model.load_state_dict(torch.load('models/tab_detection_model.pth'))
    model.eval()
    
    # Test on a single image (example)
    test_image_path = 'data/images/lot1/test_sample.jpg'  # Replace with actual path
    
    if os.path.exists(test_image_path):
        img = Image.open(test_image_path).convert('RGB')
        img_tensor = torch.tensor(img.resize((299, 299)).resize(299, 299))
        
        with torch.no_grad():
            pred_prob = model(img_tensor.unsqueeze(0).unsqueeze(0)).item()
        
        status = "INVALID" if pred_prob > 0.5 else "VALID"
        print(f"\nPrediction for {test_image_path}:")
        print(f"Probability of bad tab: {pred_prob:.4f}")
        print(f"Status: {status}")


if __name__ == "__main__":
    train_model()
