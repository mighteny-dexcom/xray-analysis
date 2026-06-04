"""
Validate new images using trained model
"""

import cv2
from pathlib import Path
from model_trainer import ModelTrainer


def validate_single_image(image_path: str, model_path: str = 'models/xray_validator.pkl'):
    """
    Validate a single image using trained model
    
    Args:
        image_path: Path to X-ray image
        model_path: Path to trained model
        
    Returns:
        Prediction (0=invalid, 1=valid) and confidence
    """
    model_file = Path(__file__).parent / model_path
    
    if not model_file.exists():
        print(f"❌ Model not found: {model_file}")
        print("   Train a model first: python train_model.py")
        return None, 0
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return None, 0
    
    # Load model
    trainer = ModelTrainer()
    trainer.load_model(str(model_file))
    
    # Predict
    prediction, confidence = trainer.predict(img)
    
    label = 'VALID' if prediction == 1 else 'INVALID'
    
    return label, confidence


def validate_directory(image_dir: str, model_path: str = 'models/xray_validator.pkl'):
    """
    Validate all images in a directory
    
    Args:
        image_dir: Directory containing images
        model_path: Path to trained model
    """
    model_file = Path(__file__).parent / model_path
    
    if not model_file.exists():
        print(f"❌ Model not found: {model_file}")
        return
    
    # Load model
    trainer = ModelTrainer()
    trainer.load_model(str(model_file))
    
    # Find images
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"🔍 Validating {len(image_files)} images...")
    print("="*70)
    
    valid_count = 0
    invalid_count = 0
    
    for img_file in image_files:
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            prediction, confidence = trainer.predict(img)
            label = 'VALID' if prediction == 1 else 'INVALID'
            
            if prediction == 1:
                valid_count += 1
            else:
                invalid_count += 1
            
            # Print with formatting
            status_icon = '✓' if prediction == 1 else '✗'
            print(f"{status_icon} {img_file.name:30s} | {label:7s} | Confidence: {confidence:.4f}")
        
        except Exception as e:
            print(f"⚠️  Error processing {img_file.name}: {e}")
    
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    print(f"  Total: {valid_count + invalid_count}")
    print(f"  Valid ratio: {valid_count/(valid_count+invalid_count)*100:.1f}%")


def main():
    import sys
    
    print("="*60)
    print("Validate Images with Trained Model")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\n📋 Usage:")
        print("  python validate_images.py <image_or_directory_path>")
        print("\nExamples:")
        print("  python validate_images.py image.png")
        print("  python validate_images.py training_images/")
        return
    
    target_path = sys.argv[1]
    target = Path(target_path)
    
    if target.is_file():
        # Validate single image
        label, confidence = validate_single_image(target_path)
        if label:
            print(f"\n{'='*60}")
            print(f"Result: {label} (Confidence: {confidence:.4f})")
            print(f"{'='*60}")
    
    elif target.is_dir():
        # Validate directory
        validate_directory(target_path)
    
    else:
        print(f"❌ Path not found: {target_path}")


if __name__ == '__main__':
    main()
