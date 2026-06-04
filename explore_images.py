"""
Image browser and analyzer for training data exploration
"""

import cv2
import numpy as np
from pathlib import Path
from image_processor import FeatureExtractor


def show_image_info(image_path: str):
    """Display image and extract features"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return False
    
    # Basic info
    h, w, c = img.shape
    print(f"\n📷 Image: {Path(image_path).name}")
    print(f"   Size: {w}x{h} pixels")
    print(f"   Channels: {c}")
    print(f"   Format: {'Color' if c == 3 else 'Grayscale'}")
    
    # Pixel stats
    print(f"\n📊 Pixel Statistics:")
    print(f"   Mean: {np.mean(img):.1f}")
    print(f"   Std Dev: {np.std(img):.1f}")
    print(f"   Min: {np.min(img)}")
    print(f"   Max: {np.max(img)}")
    
    # Features
    if c == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    edge_density = FeatureExtractor.extract_edge_density(gray)
    contour_info = FeatureExtractor.extract_contour_features(gray)
    
    print(f"\n🔍 Image Features:")
    print(f"   Edge density: {edge_density:.4f}")
    print(f"   Detected contours: {contour_info['num_contours']}")
    if contour_info['contour_areas']:
        avg_area = np.mean(contour_info['contour_areas'])
        print(f"   Avg contour area: {avg_area:.1f}")
    
    return True


def batch_analyze(start_idx: int = 0, count: int = 10):
    """Analyze multiple images"""
    training_dir = Path(__file__).parent / 'training_images'
    image_files = sorted(training_dir.glob('*.png'))
    
    if not image_files:
        print("No images found in training_images/")
        return
    
    print(f"\n{'='*70}")
    print(f"Analyzing {len(image_files)} images")
    print(f"{'='*70}")
    
    # Collect stats
    sizes = []
    edge_densities = []
    contour_counts = []
    
    for img_file in image_files[:count]:
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            h, w, c = img.shape
            sizes.append((w, h))
            
            if c == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            edge_density = FeatureExtractor.extract_edge_density(gray)
            contour_info = FeatureExtractor.extract_contour_features(gray)
            
            edge_densities.append(edge_density)
            contour_counts.append(contour_info['num_contours'])
        
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
    
    # Summary
    print(f"\n📊 Dataset Summary (first {min(count, len(image_files))} images):")
    
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"\n  Image Dimensions:")
        print(f"    Width: {np.mean(widths):.0f}x{np.min(widths)}-{np.max(widths)} pixels")
        print(f"    Height: {np.mean(heights):.0f}x{np.min(heights)}-{np.max(heights)} pixels")
    
    if edge_densities:
        print(f"\n  Edge Density:")
        print(f"    Mean: {np.mean(edge_densities):.4f}")
        print(f"    Range: {np.min(edge_densities):.4f} - {np.max(edge_densities):.4f}")
    
    if contour_counts:
        print(f"\n  Detected Contours:")
        print(f"    Mean: {np.mean(contour_counts):.1f}")
        print(f"    Range: {np.min(contour_counts)} - {np.max(contour_counts)}")
    
    print(f"\n{'='*70}\n")


def list_training_images(preview: bool = True):
    """List all training images with optional basic info"""
    training_dir = Path(__file__).parent / 'training_images'
    image_files = sorted(training_dir.glob('*.png'))
    
    if not image_files:
        print("No images in training_images/")
        return
    
    print(f"\n📁 Training Images ({len(image_files)} total):\n")
    
    for i, img_file in enumerate(image_files, 1):
        if preview:
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w, c = img.shape
                print(f"  {i:3d}. {img_file.name:25s} | {w}x{h} pixels")
            else:
                print(f"  {i:3d}. {img_file.name:25s} | [failed to load]")
        else:
            print(f"  {i:3d}. {img_file.name}")
        
        if (i) % 20 == 0:
            print()


def main():
    import sys
    
    print("="*70)
    print("X-Ray Training Images Browser")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\n📋 Usage:")
        print("  python explore_images.py list              # List all images")
        print("  python explore_images.py stats             # Show dataset statistics")
        print("  python explore_images.py info <image.png>  # Show image details")
        print("  python explore_images.py batch <count>     # Analyze multiple images")
        
        print("\n📊 Quick Analysis:")
        batch_analyze(count=20)
        
        return
    
    command = sys.argv[1]
    
    if command == 'list':
        list_training_images(preview=True)
    
    elif command == 'stats':
        batch_analyze(count=len(list(Path(__file__).parent.glob('training_images/*.png'))))
    
    elif command == 'info' and len(sys.argv) > 2:
        image_path = sys.argv[2]
        show_image_info(image_path)
    
    elif command == 'batch' and len(sys.argv) > 2:
        count = int(sys.argv[2])
        batch_analyze(count=count)
    
    else:
        print(f"Unknown command: {command}")


if __name__ == '__main__':
    main()
