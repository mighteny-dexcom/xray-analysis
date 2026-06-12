# evaluate_lot1_model.py - Evaluate model performance on lot1_images using lot1_md.xlsx

import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd


def load_model(model_path='models/lot1_xray_tab_detection_model.pth'):
    """Load and prepare the trained model"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Create model with ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint structures
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove unexpected keys
    keys_to_remove = [k for k in state_dict.keys() if 'fc.1.' in k]
    for key in keys_to_remove:
        del state_dict[key]
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_features, 1)
    )
    
    # Update state dict (skip fc layers)
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'fc.' not in k:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Freeze early layers
    for param in list(model.parameters())[:-5]:
        param.requires_grad = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    return model


def preprocess_image(image_path):
    """Preprocess image for model input"""
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    
    return input_tensor.unsqueeze(0)


def predict(model, image_path):
    """Run prediction and get probability"""
    
    with torch.no_grad():
        image_input = preprocess_image(image_path)
        output = model(image_input)
        probability = torch.sigmoid(output).item()
        
    return probability


def load_ground_truth(excel_file='lot2_md.xlsx'):
    """Load ground truth labels from Excel file"""
    
    # Check if Excel file exists
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return {}
    
    try:
        # Load Excel file
        df = pd.read_excel(excel_file)
        
        print(f"✅ Loaded annotations from {excel_file}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Total rows: {len(df)}\n")
        
        ground_truth = {}
        
        # For Lot1, use TXID column and look for tab detection in 'Xray Gross Issues' or similar
        image_col = None
        label_col = None
        
        # Look for TXID column (image identifier)
        if 'TXID' in df.columns:
            image_col = 'TXID'
        
        # Look for label-related columns  
        label_cols = ['label', 'tab', 'has_tab', 'detected', 'value', 'y_true']
        for col in label_cols:
            if col.lower() in [c.lower() for c in df.columns]:
                label_col = col
                break
        
        # If no standard label column found, try common columns that might contain tab info
        if label_col is None:
            print("⚠️  No standard label column found. Checking alternative columns...")
            
            # Check 'Xray Gross Issues' or similar columns for tab detection results
            alt_label_cols = ['Xray Gross Issues', 'issues', 'status', 'result', 'tab_detected']
            for col in alt_label_cols:
                if col.lower() in [c.lower() for c in df.columns]:
                    label_col = col
                    print(f"   Using alternative column: '{col}'")
                    break
        
        # If still no label found, assume all are positive (1) or check first available column
        if image_col is None or label_col is None:
            print("⚠️  Could not find proper columns. Assuming all images have tabs.")
            
            if 'TXID' in df.columns:
                image_col = 'TXID'
            
            # Assume positive (1) for now - you can adjust this logic
            label_col = None  # Will set default label to 1
            
        if image_col and label_col:
            # Process each row
            for idx, row in df.iterrows():
                txid = str(row[image_col]).strip()
                
                # Add .png extension as required
                img_name = f"{txid}.png"
                
                # Get label value - if no label column, default to 1 (has tab)
                if label_col is None:
                    label = 1  # Assume all have tabs for now
                else:
                    label_raw = row[label_col]
                    
                    # Handle different data types
                    if pd.isna(label_raw):
                        label = 0  # No tab
                    else:
                        label_str = str(label_raw).strip().lower()
                        
                        # Convert various formats to binary
                        if any(x in label_str for x in ['1', 'true', 'yes', 'detected', 'tab', 'positive']):
                            label = 1
                        elif any(x in label_str for x in ['0', 'false', 'no', 'not', 'negative']):
                            label = 0
                        else:
                            # Try to convert to number
                            try:
                                label = int(float(label_raw))
                            except:
                                label = 1  # Default to has tab if can't parse
                
                ground_truth[img_name] = label
            
            print(f"✅ Loaded annotations for {len(ground_truth)} images")
            
        return ground_truth
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        import traceback
        traceback.print_exc()
        return {}
    

def evaluate_model(model, image_dir='lot1_images', threshold=0.4):
    """Evaluate model performance"""
    
    # Get all images
    test_images = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(filename)
    
    print(f"\n🔍 Found {len(test_images)} images to evaluate")
    
    # Load ground truth from Excel file
    ground_truth = load_ground_truth()
    print(f"   Loaded annotations for {len(ground_truth)} images\n")
    
    if len(ground_truth) == 0:
        print("⚠️  No ground truth labels found!")
        print("   The model will still predict, but we can't evaluate accuracy.")
        return None
    
    # Evaluate each image
    results = []
    
    for i, img_name in enumerate(test_images[:100], 1):  # Test first 50 images (test_images[:50], 1)
        img_path = os.path.join(image_dir, img_name)
        
        try:
            probability = predict(model, img_path)
            predicted_label = 1 if probability >= threshold else 0
            
            # Get ground truth (if available)
            true_label = ground_truth.get(img_name, None)
            
            results.append({
                'image': img_name,
                'probability': probability,
                'predicted': predicted_label,
                'true_label': true_label,
                'correct': true_label is not None and predicted_label == true_label
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Calculate metrics
    if len(df) > 0:
        correct_predictions = (df['predicted'] == df['true_label']).sum()
        total_with_labels = df['true_label'].notna().sum()
        
        accuracy = correct_predictions / total_with_labels if total_with_labels > 0 else 0
        
        print(f"\n{'='*60}")
        print("📊 MODEL EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total images tested: {len(df)}")
        print(f"Images with ground truth: {total_with_labels}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
        
        # Show some examples
        correct_examples = df[df['correct'] == True].head(3)
        incorrect_examples = df[(df['true_label'].notna()) & (df['correct'] == False)].head(3)
        
        if len(correct_examples) > 0:
            print(f"\n✅ Correct predictions:")
            for _, row in correct_examples.iterrows():
                status = "✓" if row['predicted'] == row['true_label'] else "✗"
                print(f"   {status} {row['image']}: pred={row['predicted']} (prob={row['probability']:.3f}), true={row['true_label']}")
        
        if len(incorrect_examples) > 0:
            print(f"\n❌ Incorrect predictions:")
            for _, row in incorrect_examples.iterrows():
                status = "✓" if row['predicted'] == row['true_label'] else "✗"
                print(f"   {status} {row['image']}: pred={row['predicted']} (prob={row['probability']:.3f}), true={row['true_label']}")
        
        return df
    
    return None


def main():
    """Main evaluation function"""
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Evaluate
    results_df = evaluate_model(
        model=model,
        image_dir='lot2_images',
        threshold=0.5
    )
    
    if results_df is not None and len(results_df) > 0:
        # Save results to CSV
        output_file = 'lot1_evaluation_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to {output_file}")


if __name__ == '__main__':
    main()