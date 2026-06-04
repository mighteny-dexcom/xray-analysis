# XRay Analysis - Image Validation Framework

A machine learning and computer vision framework for validating images and detecting specific regions/features in X-ray and medical images.

## Project Structure

```
xray-analysis/
├── image_processor.py      # Image preprocessing and feature extraction
├── image_validator.py      # Multiple validation approaches (template matching, edge detection, color-based)
├── data_loader.py          # Dataset management and loading utilities
├── model_trainer.py        # ML model training (Random Forest, expandable)
├── examples.py             # Usage examples and demos
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Features

### Image Processing (`image_processor.py`)
- **ImageProcessor**: Core image processing utilities
  - Image loading and resizing
  - Preprocessing (grayscale conversion, Gaussian blur)
  - Edge detection (Canny)
  - ROI extraction
  - Image normalization

- **FeatureExtractor**: Extract features from images
  - Histogram features
  - Edge density
  - Contour-based features
  - HOG (Histogram of Oriented Gradients)

### Image Validation Approaches (`image_validator.py`)

#### 1. Template Matching Validator
Match a known template pattern in images. Best for detecting specific shapes/objects.
```python
validator = TemplateMatchingValidator(template_image)
result = validator.validate(test_image, threshold=0.7)
```

#### 2. Edge-Based Validator
Detect edges and contours to find structured regions. Good for detecting boundaries and structures.
```python
validator = EdgeBasedValidator()
result = validator.validate(test_image, min_contours=1)
```

#### 3. Color-Based Validator
Detect specific colored regions using HSV color space.
```python
validator = ColorBasedValidator(color_lower, color_upper)
result = validator.validate(test_image, min_area_ratio=0.01)
```

### Dataset Management (`data_loader.py`)
- Load images from directory structure (valid/invalid folders)
- Load from JSON manifest files
- Split datasets for training/validation
- Per-sample metadata and ROI tracking

### Model Training (`model_trainer.py`)
- Random Forest classifier for learning validation patterns
- Automatic feature extraction from images
- Model serialization (save/load)
- Training history and evaluation metrics

## Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start with Real X-Ray Images

### Workflow for Training on Your Data

1. **Prepare training data** - Generates annotation template from your 199 real X-ray images:
   ```bash
   python prepare_training_data.py
   ```
   This creates `training_manifest_template.json` with all images listed.

2. **Label your images** - Edit `training_manifest_template.json`:
   - Open the file and for each image sample, set `"label"` to either `"valid"` or `"invalid"`
   - Save it as `training_manifest.json` when done

3. **Train the model** - Train on your labeled data:
   ```bash
   python train_model.py
   ```
   This extracts features from images and trains a Random Forest classifier.
   Output shows train/test accuracy.

4. **Validate new images** - Test your model:
   ```bash
   python validate_images.py training_images/        # Validate all images
   python validate_images.py single_image.png        # Validate single image
   ```

## Quick Start

### 1. Basic Image Processing

```python
from image_processor import ImageProcessor

processor = ImageProcessor()
image = processor.load_image('path/to/image.jpg')
edges = processor.detect_edges(image)
roi = processor.extract_roi(image, x=10, y=20, width=100, height=100)
```

### 2. Validate Images (Multiple Approaches)

```python
from image_validator import TemplateMatchingValidator, EdgeBasedValidator

# Template matching
validator = TemplateMatchingValidator(template)
result = validator.validate(test_image)
print(f"Valid: {result.is_valid}, Confidence: {result.confidence}")

# Edge-based validation
edge_validator = EdgeBasedValidator()
result = edge_validator.validate(test_image, min_contours=2)
```

### 3. Load and Manage Datasets

```python
from data_loader import DataLoader

loader = DataLoader('path/to/dataset')
loader.load_from_directory_structure(valid_dir='valid', invalid_dir='invalid')

# Get statistics
stats = loader.get_statistics()
print(stats)

# Split for training
train_samples, val_samples = loader.split_dataset(train_ratio=0.8)
```

### 4. Train ML Model

```python
from model_trainer import ModelTrainer
from data_loader import DataLoader

# Load dataset
loader = DataLoader('path/to/dataset')
images = loader.load_images()

# Train
trainer = ModelTrainer(model_name='random_forest')
X, y = trainer.prepare_training_data(images)
trainer.train(X, y)

# Predict
prediction, confidence = trainer.predict(test_image)
print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")

# Save model
trainer.save_model('models/validation_model.pkl')
```

## Run Examples

```bash
python examples.py
```

This will run several examples demonstrating:
1. Basic image processing
2. Template matching validation
3. Edge-based validation
4. Dataset management
5. Model training

## Dataset Format

### Directory Structure
```
dataset/
├── valid/
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── ...
└── invalid/
    ├── sample_001.jpg
    ├── sample_002.jpg
    └── ...
```

### JSON Manifest Format
```json
{
  "metadata": {
    "description": "Training dataset for X-ray validation",
    "created": "2024-01-01"
  },
  "samples": [
    {
      "image_path": "path/to/image.jpg",
      "label": "valid",
      "region_of_interest": {
        "x": 10,
        "y": 20,
        "width": 100,
        "height": 100
      },
      "metadata": {}
    }
  ]
}
```

## Validation Result Format

All validators return a `ValidationResult` object with:

```python
@dataclass
class ValidationResult:
    is_valid: bool              # Whether image passes validation
    confidence: float           # Confidence score (0-1)
    detected_regions: List[Dict]  # Bounding boxes of detected regions
    features: Dict              # Extracted features
    metadata: Dict              # Method-specific metadata
```

## Recommended Workflow

1. **Start with rule-based validators** (template matching, edge detection) for quick prototyping
2. **Collect labeled dataset** with valid/invalid examples
3. **Train ML model** once you have ~100+ labeled samples
4. **Combine approaches** - use rule-based for preprocessing + ML for final validation

## Extending the Framework

### Add New Validator Type
```python
from image_validator import ImageValidator, ValidationResult

class MyCustomValidator(ImageValidator):
    def validate(self, image):
        # Your validation logic here
        return ValidationResult(
            is_valid=...,
            confidence=...,
            detected_regions=[...],
            features={...},
            metadata={...}
        )
```

### Add New Feature Extractor
```python
from image_processor import FeatureExtractor

class MyFeatureExtractor(FeatureExtractor):
    @staticmethod
    def extract_my_features(image):
        # Your feature extraction logic
        return feature_vector
```

## Dependencies

- **opencv-python**: Image processing
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning
- **tensorflow**: Deep learning (optional, for future CNN models)
- **PyQt6**: GUI utilities (optional)
- **pillow**: Image utilities
- **matplotlib**: Visualization

## Notes

- Images are loaded and processed in BGR format (OpenCV default)
- Feature extraction is modular - easy to add new feature types
- Models are saved with their scalers for consistent preprocessing
- The framework is designed to be extended with deep learning models (CNN, etc.)

## Future Enhancements

- CNN-based validators using TensorFlow/PyTorch
- Real-time validation with webcam/video streams  
- GUI application for annotation and testing
- Batch processing utilities
- Model ensemble methods
- Active learning for efficient labeling
