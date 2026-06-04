"""
Example usage script demonstrating the xray-analysis framework
"""

import cv2
import sys
from pathlib import Path

from image_processor import ImageProcessor, FeatureExtractor
from image_validator import TemplateMatchingValidator, EdgeBasedValidator, ColorBasedValidator
from data_loader import DataLoader, create_test_dataset
from model_trainer import ModelTrainer


def example_1_basic_image_processing():
    """Example 1: Basic image processing"""
    print("\n" + "="*60)
    print("Example 1: Basic Image Processing")
    print("="*60)
    
    processor = ImageProcessor()
    
    # Create a synthetic test image
    import numpy as np
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(test_image, (150, 150), 80, (0, 255, 0), -1)
    cv2.rectangle(test_image, (50, 50), (250, 250), (255, 0, 0), 3)
    
    # Preprocess
    preprocessed = processor.preprocess(test_image)
    print(f"Image shape: {test_image.shape}")
    print(f"Preprocessed shape: {preprocessed.shape}")
    
    # Detect edges
    edges = processor.detect_edges(test_image)
    print(f"Detected edges")
    
    # Extract ROI
    roi = processor.extract_roi(test_image, 50, 50, 200, 200)
    print(f"ROI shape: {roi.shape}")


def example_2_template_matching_validation():
    """Example 2: Template matching based validation"""
    print("\n" + "="*60)
    print("Example 2: Template Matching Validation")
    print("="*60)
    
    import numpy as np
    
    # Create template (small circle)
    template = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(template, (50, 50), 40, (0, 255, 0), -1)
    
    # Create test image with the template
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    test_image[100:200, 100:200] = template
    
    # Validate using template matching
    validator = TemplateMatchingValidator(template)
    result = validator.validate(test_image, threshold=0.6)
    
    print(f"Is valid: {result.is_valid}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Detected regions: {len(result.detected_regions)}")
    if result.detected_regions:
        print(f"  Location: {result.detected_regions[0]['location']}")


def example_3_edge_based_validation():
    """Example 3: Edge-based validation"""
    print("\n" + "="*60)
    print("Example 3: Edge-Based Validation")
    print("="*60)
    
    import numpy as np
    
    # Create test image with edges/shapes
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(test_image, (150, 150), 80, (0, 255, 0), 2)
    cv2.rectangle(test_image, (50, 50), (250, 250), (255, 0, 0), 2)
    
    # Validate using edge detection
    validator = EdgeBasedValidator()
    result = validator.validate(test_image, min_contours=1)
    
    print(f"Is valid: {result.is_valid}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Features: {result.features}")
    print(f"Detected regions: {len(result.detected_regions)}")


def example_4_dataset_management():
    """Example 4: Dataset management"""
    print("\n" + "="*60)
    print("Example 4: Dataset Management")
    print("="*60)
    
    # Create test dataset
    dataset_dir = './test_dataset'
    create_test_dataset(dataset_dir, num_valid=10, num_invalid=10)
    
    # Load dataset
    loader = DataLoader(dataset_dir)
    num_loaded = loader.load_from_directory_structure()
    
    print(f"Loaded {num_loaded} samples")
    stats = loader.get_statistics()
    print(f"Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Split dataset
    train_samples, val_samples = loader.split_dataset(train_ratio=0.8)
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")


def example_5_model_training():
    """Example 5: Model training"""
    print("\n" + "="*60)
    print("Example 5: Model Training (Simplified Demo)")
    print("="*60)
    
    import numpy as np
    
    # Create simple synthetic training data
    print("Creating synthetic training data...")
    
    processor = ImageProcessor()
    trainer = ModelTrainer(model_name='random_forest')
    
    # Generate synthetic images
    training_images = []
    
    # Valid samples
    for i in range(10):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(img, (128, 128), 50, (0, 255, 0), -1)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), 2)
        training_images.append((img, 'valid'))
    
    # Invalid samples
    for i in range(10):
        img = np.random.randint(50, 100, (256, 256, 3), dtype=np.uint8)
        training_images.append((img, 'invalid'))
    
    # Prepare training data
    print("Extracting features...")
    X, y = trainer.prepare_training_data(training_images)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Train model
    print("Training model...")
    trainer.build_model(n_estimators=50)
    history = trainer.train(X, y, test_size=0.3)
    
    print(f"Training completed!")
    print(f"  Train accuracy: {history['train_accuracy']:.4f}")
    print(f"  Test accuracy: {history['test_accuracy']:.4f}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("XRay Analysis Framework Examples")
    print("="*60)
    
    try:
        example_1_basic_image_processing()
        example_2_template_matching_validation()
        example_3_edge_based_validation()
        example_4_dataset_management()
        example_5_model_training()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
