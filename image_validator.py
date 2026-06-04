"""
Image validation module for detecting specific regions/features in xray images
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of image validation"""
    is_valid: bool
    confidence: float
    detected_regions: List[Dict]
    features: Dict
    metadata: Dict
    

class ImageValidator:
    """Base class for image validation using various detection methods"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize validator
        
        Args:
            model_path: Path to pre-trained model (if using ML-based validation)
        """
        self.model_path = model_path
        self.model = None
        self.config = {}
        
    def validate(self, image: np.ndarray) -> ValidationResult:
        """
        Validate image for presence of specific features/regions
        
        Args:
            image: Input BGR image
            
        Returns:
            ValidationResult with detection results
        """
        raise NotImplementedError("Subclasses must implement validate()")
    
    def save_config(self, config_path: str):
        """Save validator configuration"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_config(self, config_path: str):
        """Load validator configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)


class TemplateMatchingValidator(ImageValidator):
    """
    Template matching based validator
    Detects if a template pattern appears in the image
    """
    
    def __init__(self, template_image: Optional[np.ndarray] = None):
        super().__init__()
        self.template = template_image
        self.threshold = 0.7
        self.config = {
            'method': 'template_matching',
            'threshold': self.threshold,
            'template_size': None
        }
    
    def set_template(self, template: np.ndarray):
        """Set the template to search for"""
        self.template = template
        self.config['template_size'] = template.shape
    
    def validate(self, image: np.ndarray, threshold: Optional[float] = None) -> ValidationResult:
        """
        Find template in image
        
        Args:
            image: Input image to validate
            threshold: Confidence threshold (0-1)
            
        Returns:
            ValidationResult with detection results
        """
        if self.template is None:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                detected_regions=[],
                features={},
                metadata={'error': 'No template set'}
            )
        
        thresh = threshold or self.threshold
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image
            template_gray = self.template
        
        # Template matching
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find matches above threshold
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        detected_regions = []
        if max_val >= thresh:
            h, w = template_gray.shape
            detected_regions.append({
                'location': {'x': max_loc[0], 'y': max_loc[1]},
                'size': {'width': w, 'height': h},
                'confidence': float(max_val)
            })
        
        is_valid = len(detected_regions) > 0
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=float(max_val),
            detected_regions=detected_regions,
            features={'max_match_score': float(max_val)},
            metadata={'method': 'template_matching', 'threshold_used': thresh}
        )


class EdgeBasedValidator(ImageValidator):
    """
    Edge-based validation
    Detects specific edge patterns or structures in images
    """
    
    def __init__(self):
        super().__init__()
        self.config = {
            'method': 'edge_based',
            'canny_low': 50,
            'canny_high': 150,
            'min_contour_area': 100
        }
    
    def validate(self, image: np.ndarray, min_contours: int = 1) -> ValidationResult:
        """
        Detect edges and contours in image
        
        Args:
            image: Input image
            min_contours: Minimum number of significant contours to consider valid
            
        Returns:
            ValidationResult with detection results
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, self.config['canny_low'], self.config['canny_high'])
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        significant_contours = [
            c for c in contours 
            if cv2.contourArea(c) >= self.config['min_contour_area']
        ]
        
        detected_regions = []
        for contour in significant_contours:
            x, y, w, h = cv2.boundingRect(contour)
            detected_regions.append({
                'location': {'x': x, 'y': y},
                'size': {'width': w, 'height': h},
                'area': float(cv2.contourArea(contour))
            })
        
        is_valid = len(significant_contours) >= min_contours
        confidence = min(len(significant_contours) / max(min_contours, 1), 1.0)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=float(confidence),
            detected_regions=detected_regions,
            features={
                'total_contours': len(contours),
                'significant_contours': len(significant_contours),
                'edge_pixels': int(np.sum(edges > 0))
            },
            metadata={'method': 'edge_based', 'threshold_contours': min_contours}
        )


class ColorBasedValidator(ImageValidator):
    """
    Color-based validation using HSV color space
    Detects specific colored regions in images
    """
    
    def __init__(self, color_lower: Optional[np.ndarray] = None, 
                 color_upper: Optional[np.ndarray] = None):
        super().__init__()
        self.color_lower = color_lower or np.array([0, 0, 0])
        self.color_upper = color_upper or np.array([180, 255, 255])
        self.config = {
            'method': 'color_based',
            'color_lower': self.color_lower.tolist(),
            'color_upper': self.color_upper.tolist()
        }
    
    def validate(self, image: np.ndarray, min_area_ratio: float = 0.01) -> ValidationResult:
        """
        Detect colored regions in image
        
        Args:
            image: Input BGR image
            min_area_ratio: Minimum ratio of colored pixels to total image area
            
        Returns:
            ValidationResult with detection results
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for color range
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_regions = []
        total_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:
                total_area += area
                x, y, w, h = cv2.boundingRect(contour)
                detected_regions.append({
                    'location': {'x': x, 'y': y},
                    'size': {'width': w, 'height': h},
                    'area': float(area)
                })
        
        image_area = image.shape[0] * image.shape[1]
        area_ratio = total_area / image_area
        is_valid = area_ratio >= min_area_ratio
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=float(area_ratio),
            detected_regions=detected_regions,
            features={
                'total_colored_pixels': int(total_area),
                'area_ratio': float(area_ratio),
                'num_regions': len(detected_regions)
            },
            metadata={'method': 'color_based', 'min_area_ratio': min_area_ratio}
        )
