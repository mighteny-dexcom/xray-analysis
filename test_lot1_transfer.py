# test_lot1_transfer.py - Test the trained Lot1 model

import torch
from torchvision import models, transforms
from PIL import Image
import os


def load_model(model_path='models/lot1_xray_tab_detection_model.pth'):
    """Load and prepare the trained model"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Create model with ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Load saved weights - handle the different key structure
    checkpoint = torch.load(model_path, map_location='cpu')

    # Check if checkpoint has 'model' or 'state_dict' key
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove unexpected keys (from previous training with different architecture)
    keys_to_remove = [k for k in state_dict.keys() if 'fc.1.' in k]
    for key in keys_to_remove:
        print(f"Removing unexpected key: {key}")
        del state_dict[key]

    # Create new final layer and update state dict keys
    num_features = model.fc.in_features

    # Save original fc weights temporarily
    old_fc_weight = model.fc.weight.data.clone()
    old_fc_bias = model.fc.bias.data.clone()

    # Replace final layer
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_features, 1)
    )
    
    # Update state dict keys to match new architecture
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'fc.' in k:
            # Skip fc layers - they will be initialized randomly
            continue
        else:
            new_state_dict[k] = v

    # Load the filtered state dict
    model.load_state_dict(new_state_dict, strict=False)

    # Freeze early layers (same as training)
    for param in list(model.parameters())[:-5]:
        param.requires_grad = False

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    return model


def preprocess_image(image_path):
    """Preprocess image for model input"""
    
    # Load and resize image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    
    return input_tensor.unsqueeze(0)  # Add batch dimension


def predict_tab_detection(model, image_path):
    """Run tab detection prediction on an image"""
    
    try:
        # Preprocess image
        image_input = preprocess_image(image_path)
        
        # Run inference
        with torch.no_grad():
            output = model(image_input)
            probability = torch.sigmoid(output).item()
            
        print(f"   Tab detection score: {probability:.4f}")
        print(f"   Prediction: {'TAB DETECTED' if probability > 0.5 else 'NO TAB'}")
        
        return probability
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None


def main():
    """Main testing function"""
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    from torchvision import transforms
    
    # Use your lot1_images directory directly
    image_dir = 'lot1_images'  # Your trained dataset images
    test_images = []
    
    if os.path.exists(image_dir):
        for filename in sorted(os.listdir(image_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(image_dir, filename))
    
    print(f"\n🔍 Found {len(test_images)} images in lot1_images to test")
    
    # Test each image (or first N if too many)
    num_to_test = min(len(test_images), 50)  # Adjust as needed

    for i, img_path in enumerate(test_images[:num_to_test], 1):
        print(f"\n--- Image {i}/{len(test_images)} ---")
        predict_tab_detection(model, img_path)
    
    if len(test_images) > num_to_test:
        print(f"\n⚠️  Only testing first {num_to_test} images. Run with different limit to test all.")

    print("\n✅ Testing complete!")


if __name__ == '__main__':
    main()

