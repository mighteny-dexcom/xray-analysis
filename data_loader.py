"""
Data loading utilities for training and validation datasets
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class ImageSample:
    """Represents a single training/validation sample"""
    image_path: str
    label: str  # 'valid' or 'invalid'
    region_of_interest: Optional[Dict] = None  # Optional ROI specification
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DataLoader:
    """Load and manage image datasets for validation tasks"""
    
    def __init__(self, dataset_dir: str):
        """
        Initialize data loader
        
        Args:
            dataset_dir: Root directory containing training/validation images
        """
        self.dataset_dir = Path(dataset_dir)
        self.samples: List[ImageSample] = []
        self.metadata = {}
    
    def load_from_directory_structure(self, 
                                     valid_dir: str = 'valid', 
                                     invalid_dir: str = 'invalid') -> int:
        """
        Load images from directory structure:
        dataset_root/
            valid/
                *.jpg, *.png
            invalid/
                *.jpg, *.png
        
        Args:
            valid_dir: Subdirectory name for valid images
            invalid_dir: Subdirectory name for invalid images
            
        Returns:
            Number of samples loaded
        """
        # Load valid images
        valid_path = self.dataset_dir / valid_dir
        if valid_path.exists():
            for img_file in valid_path.glob('*.jpg') + list(valid_path.glob('*.png')):
                self.samples.append(ImageSample(
                    image_path=str(img_file),
                    label='valid'
                ))
        
        # Load invalid images
        invalid_path = self.dataset_dir / invalid_dir
        if invalid_path.exists():
            for img_file in invalid_path.glob('*.jpg') + list(invalid_path.glob('*.png')):
                self.samples.append(ImageSample(
                    image_path=str(img_file),
                    label='invalid'
                ))
        
        return len(self.samples)
    
    def load_from_manifest(self, manifest_file: str):
        """
        Load from JSON manifest file
        
        Expected format:
        {
            "samples": [
                {
                    "image_path": "path/to/image.jpg",
                    "label": "valid",
                    "region_of_interest": {"x": 10, "y": 20, "width": 100, "height": 100},
                    "metadata": {}
                }
            ]
        }
        
        Args:
            manifest_file: Path to JSON manifest
        """
        with open(manifest_file, 'r') as f:
            data = json.load(f)
        
        for sample_data in data.get('samples', []):
            sample = ImageSample(**sample_data)
            self.samples.append(sample)
        
        self.metadata = data.get('metadata', {})
        return len(self.samples)
    
    def save_manifest(self, output_file: str):
        """Save dataset manifest to JSON"""
        data = {
            'metadata': self.metadata,
            'samples': [sample.to_dict() for sample in self.samples],
            'total_samples': len(self.samples)
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_samples_by_label(self, label: str) -> List[ImageSample]:
        """Get all samples with a specific label"""
        return [s for s in self.samples if s.label == label]
    
    def load_images(self, max_samples: Optional[int] = None) -> List[Tuple[np.ndarray, str]]:
        """
        Load actual image arrays
        
        Args:
            max_samples: Maximum number of samples to load
            
        Returns:
            List of (image_array, label) tuples
        """
        images = []
        samples = self.samples[:max_samples] if max_samples else self.samples
        
        for sample in samples:
            try:
                img = cv2.imread(sample.image_path)
                if img is not None:
                    images.append((img, sample.label))
                else:
                    print(f"Failed to load {sample.image_path}")
            except Exception as e:
                print(f"Error loading {sample.image_path}: {e}")
        
        return images
    
    def load_image_with_roi(self, sample: ImageSample) -> Optional[np.ndarray]:
        """Load image and optionally extract ROI"""
        img = cv2.imread(sample.image_path)
        if img is None:
            return None
        
        if sample.region_of_interest:
            roi = sample.region_of_interest
            x, y = roi['x'], roi['y']
            w, h = roi['width'], roi['height']
            return img[y:y+h, x:x+w]
        
        return img
    
    def split_dataset(self, train_ratio: float = 0.8) -> Tuple[List[ImageSample], List[ImageSample]]:
        """
        Split dataset into training and validation sets
        
        Args:
            train_ratio: Ratio of samples for training (e.g., 0.8 for 80/20 split)
            
        Returns:
            Tuple of (train_samples, val_samples)
        """
        n_samples = len(self.samples)
        n_train = int(n_samples * train_ratio)
        
        # Shuffle while maintaining label balance
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_samples = [self.samples[i] for i in train_indices]
        val_samples = [self.samples[i] for i in val_indices]
        
        return train_samples, val_samples
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        labels = [s.label for s in self.samples]
        unique_labels = set(labels)
        
        stats = {
            'total_samples': len(self.samples),
            'unique_labels': list(unique_labels),
            'label_distribution': {label: labels.count(label) for label in unique_labels}
        }
        
        return stats


def create_test_dataset(output_dir: str, num_valid: int = 10, num_invalid: int = 10):
    """
    Create a test dataset with synthetic images
    
    Args:
        output_dir: Directory to save test dataset
        num_valid: Number of valid samples to create
        num_invalid: Number of invalid samples to create
    """
    output_path = Path(output_dir)
    valid_dir = output_path / 'valid'
    invalid_dir = output_path / 'invalid'
    
    valid_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)
    
    # Create valid samples (images with clear patterns)
    for i in range(num_valid):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Draw some patterns
        cv2.circle(img, (128, 128), 50, (0, 255, 0), -1)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), 2)
        cv2.imwrite(str(valid_dir / f'valid_{i:03d}.png'), img)
    
    # Create invalid samples (images with minimal patterns)
    for i in range(num_invalid):
        img = np.random.randint(50, 100, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(invalid_dir / f'invalid_{i:03d}.png'), img)
    
    print(f"Created test dataset: {num_valid} valid and {num_invalid} invalid samples in {output_dir}")
