"""
Script to prepare real X-ray images for training
Load images from training_images folder and create an annotation interface
"""

import cv2
import numpy as np
from pathlib import Path
from data_loader import DataLoader, ImageSample
import json


def create_annotation_manifest():
    """
    Create a JSON manifest for manual annotation
    This creates a template where you can mark images as valid/invalid
    """
    training_dir = Path(__file__).parent / 'training_images'
    
    if not training_dir.exists():
        print(f"Training directory not found: {training_dir}")
        return
    
    # Get all PNG images
    image_files = sorted(training_dir.glob('*.png'))
    
    samples = []
    for img_file in image_files:
        sample = {
            'image_path': str(img_file),
            'label': '',  # To be filled in by user
            'region_of_interest': None,
            'metadata': {'source': 'real_xray', 'filename': img_file.name}
        }
        samples.append(sample)
    
    # Save manifest template
    manifest = {
        'metadata': {
            'description': 'Real X-ray images from Lot 2 (5 trays)',
            'total_samples': len(samples),
            'created': '2026-06-03',
            'instructions': 'Fill in "label" field with "valid" or "invalid" based on your validation criteria'
        },
        'samples': samples
    }
    
    manifest_path = training_dir.parent / 'training_manifest_template.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Created annotation manifest: {manifest_path}")
    print(f"  {len(samples)} images ready for labeling")
    print(f"\n📋 Instructions:")
    print(f"  1. Open training_manifest_template.json")
    print(f"  2. For each sample, set 'label' to either 'valid' or 'invalid'")
    print(f"  3. Save as training_manifest.json")
    print(f"  4. Run: python load_and_analyze.py")
    
    return manifest_path


def analyze_image_properties():
    """
    Analyze properties of images to understand what we're working with
    """
    training_dir = Path(__file__).parent / 'training_images'
    image_files = list(training_dir.glob('*.png'))[:10]  # Analyze first 10
    
    sizes = []
    pixel_stats = []
    
    print("\n📊 Analyzing image properties...")
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            sizes.append(img.shape)
            pixel_stats.append({
                'mean': np.mean(img),
                'std': np.std(img),
                'min': np.min(img),
                'max': np.max(img)
            })
    
    if sizes:
        # Get most common size
        from collections import Counter
        size_counter = Counter(sizes)
        most_common_size = size_counter.most_common(1)[0][0]
        
        print(f"  Common image size: {most_common_size}")
        print(f"  Pixel value ranges (from random samples):")
        for i, stats in enumerate(pixel_stats[:3]):
            print(f"    Image {i}: mean={stats['mean']:.1f}, std={stats['std']:.1f}, min={stats['min']}, max={stats['max']}")


def simple_validation_check():
    """
    Run a simple check on downloaded images
    Shows image statistics without requiring labels
    """
    training_dir = Path(__file__).parent / 'training_images'
    image_files = sorted(training_dir.glob('*.png'))
    
    print(f"\n✓ Found {len(image_files)} training images")
    
    if len(image_files) > 0:
        # Load and analyze first few images
        print(f"\nAnalyzing first 5 images...")
        for img_file in image_files[:5]:
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w, c = img.shape
                print(f"  {img_file.name}: {w}x{h} pixels, {c} channels")
        
        analyze_image_properties()
        
        return True
    
    return False


def main():
    """Main workflow"""
    print("="*60)
    print("X-Ray Training Data Preparation")
    print("="*60)
    
    # Step 1: Verify images are present
    if not simple_validation_check():
        print("❌ No training images found!")
        return
    
    # Step 2: Create annotation manifest
    create_annotation_manifest()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("  1. Label the images in training_manifest_template.json")
    print("  2. Rename to training_manifest.json") 
    print("  3. Run: python train_model.py")
    print("="*60)


if __name__ == '__main__':
    main()
