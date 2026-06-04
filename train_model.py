"""
Train model on real X-ray images with labels
"""

import cv2
import json
from pathlib import Path
from data_loader import DataLoader
from model_trainer import ModelTrainer


def train_from_manifest(manifest_path: str = 'training_manifest.json'):
    """
    Train model from labeled manifest file
    
    Expected manifest format:
    {
        "samples": [
            {"image_path": "path/to/image.png", "label": "valid"},
            {"image_path": "path/to/image.png", "label": "invalid"}
        ]
    }
    """
    manifest_file = Path(__file__).parent / manifest_path
    
    if not manifest_file.exists():
        print(f"❌ Manifest file not found: {manifest_file}")
        print(f"\n📋 Steps:")
        print(f"  1. Run: python prepare_training_data.py")
        print(f"  2. Edit training_manifest_template.json and label images")
        print(f"  3. Save as {manifest_path}")
        return
    
    # Load manifest
    with open(manifest_file, 'r') as f:
        manifest_data = json.load(f)
    
    samples = manifest_data.get('samples', [])
    
    # Filter out unlabeled samples
    labeled_samples = [
        s for s in samples 
        if s.get('label') in ['valid', 'invalid']
    ]
    
    if len(labeled_samples) == 0:
        print("❌ No labeled samples found in manifest!")
        return
    
    print(f"✓ Found {len(labeled_samples)} labeled samples")
    
    # Count labels
    valid_count = sum(1 for s in labeled_samples if s['label'] == 'valid')
    invalid_count = sum(1 for s in labeled_samples if s['label'] == 'invalid')
    
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    
    if valid_count == 0 or invalid_count == 0:
        print("⚠️  Warning: Unbalanced dataset!")
        print("   Recommended: Have both valid and invalid samples")
    
    # Load images
    print(f"\n📥 Loading images...")
    training_images = []
    
    for i, sample in enumerate(labeled_samples):
        try:
            img_path = sample.get('image_path')
            label = sample.get('label')
            
            img = cv2.imread(img_path)
            if img is not None:
                training_images.append((img, label))
            else:
                print(f"  ⚠️  Failed to load: {img_path}")
        except Exception as e:
            print(f"  ⚠️  Error loading sample {i}: {e}")
    
    print(f"✓ Loaded {len(training_images)} images")
    
    if len(training_images) < 10:
        print("❌ Need at least 10 labeled samples to train")
        return
    
    # Train model
    print(f"\n🤖 Training Random Forest model...")
    trainer = ModelTrainer(model_name='random_forest')
    
    # Extract features
    print("  Extracting features...")
    X, y = trainer.prepare_training_data(training_images)
    
    # Train
    print("  Training...")
    history = trainer.train(X, y, test_size=0.2, n_estimators=100)
    
    print(f"\n📊 Training Results:")
    print(f"  Train accuracy: {history['train_accuracy']:.4f}")
    print(f"  Test accuracy: {history['test_accuracy']:.4f}")
    
    # Save model
    model_dir = Path(__file__).parent / 'models'
    model_dir.mkdir(exist_ok=True)
    model_path = str(model_dir / 'xray_validator.pkl')
    
    trainer.save_model(model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"\nNext step: python validate_images.py")


def main():
    print("="*60)
    print("Train X-Ray Validation Model")
    print("="*60)
    
    # Try to train
    train_from_manifest('training_manifest.json')


if __name__ == '__main__':
    main()
