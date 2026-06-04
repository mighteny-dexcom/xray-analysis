"""
Training module for image validation models
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from image_processor import ImageProcessor, FeatureExtractor
from data_loader import ImageSample


class ModelTrainer:
    """Train machine learning models for image validation"""
    
    def __init__(self, model_name: str = 'random_forest'):
        """
        Initialize trainer
        
        Args:
            model_name: Type of model to train ('random_forest', etc.)
        """
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.image_processor = ImageProcessor()
        self.training_history = {}
    
    def build_model(self, **kwargs):
        """Build model with specified parameters"""
        if self.model_name == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 20),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def extract_features_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from image for ML model
        
        Args:
            image: Input BGR image
            
        Returns:
            Feature vector
        """
        # Preprocess image
        preprocessed = self.image_processor.preprocess(image)
        
        # Extract multiple feature types
        hist_features = self.feature_extractor.extract_histogram_features(preprocessed)
        edge_density = self.feature_extractor.extract_edge_density(preprocessed)
        contour_info = self.feature_extractor.extract_contour_features(preprocessed)
        
        # Combine features
        features = np.concatenate([
            hist_features,
            [edge_density],
            [contour_info['num_contours']],
            [np.mean(contour_info['contour_areas']) if contour_info['contour_areas'] else 0]
        ])
        
        return features
    
    def prepare_training_data(self, samples: List[Tuple[np.ndarray, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from image samples
        
        Args:
            samples: List of (image_array, label) tuples
            
        Returns:
            Tuple of (features, labels)
        """
        features_list = []
        labels_list = []
        
        for i, (image, label) in enumerate(samples):
            try:
                features = self.extract_features_from_image(image)
                features_list.append(features)
                labels_list.append(1 if label == 'valid' else 0)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(samples)} samples")
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2, **model_kwargs):
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Labels (0 or 1)
            test_size: Ratio of test set
            **model_kwargs: Additional arguments for model
        """
        # Build model
        self.build_model(**model_kwargs)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Train
        print(f"Training {self.model_name} on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        self.training_history = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        return self.training_history
    
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Predict if image is valid
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        features = self.extract_features_from_image(image)
        features_scaled = self.scaler.transform([features])
        
        prediction = self.model.predict(features_scaled)[0]
        confidence = max(self.model.predict_proba(features_scaled)[0])
        
        return prediction, confidence
    
    def save_model(self, model_path: str):
        """Save trained model and scaler"""
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, str(model_dir / 'scaler.pkl'))
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load pre-trained model and scaler"""
        model_dir = Path(model_path).parent
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(str(model_dir / 'scaler.pkl'))
        
        print(f"Model loaded from {model_path}")
