"""
Image processing utilities for xray image analysis
Reusable components for edge detection, feature extraction, and preprocessing
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class ImageProcessor:
    """Core image processing functionality"""
    
    def __init__(self, canny_low: int = 50, canny_high: int = 150):
        """
        Initialize image processor with Canny edge detection parameters
        
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def preprocess(self, image: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
        """
        Preprocess image: convert to grayscale and apply Gaussian blur
        
        Args:
            image: Input BGR image
            blur_kernel: Size of Gaussian blur kernel
            
        Returns:
            Preprocessed grayscale image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 1)
        return blurred
    
    def detect_edges(self, image: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
        """
        Detect edges using Canny edge detection
        
        Args:
            image: Input BGR image
            blur_kernel: Size of Gaussian blur kernel
            
        Returns:
            Binary edge map
        """
        preprocessed = self.preprocess(image, blur_kernel)
        edges = cv2.Canny(preprocessed, self.canny_low, self.canny_high)
        return edges
    
    def extract_roi(self, image: np.ndarray, x: int, y: int, 
                    width: int, height: int) -> np.ndarray:
        """
        Extract region of interest from image
        
        Args:
            image: Input image
            x, y: Top-left corner coordinates
            width, height: Size of ROI
            
        Returns:
            Cropped region
        """
        return image[y:y+height, x:x+width]
    
    def resize_image(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize image to specified dimensions"""
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range"""
        return image.astype(np.float32) / 255.0


class FeatureExtractor:
    """Extract features from images for validation"""
    
    @staticmethod
    def extract_histogram_features(image: np.ndarray, bins: int = 256) -> np.ndarray:
        """Extract histogram features from image"""
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        return hist.flatten() / (image.shape[0] * image.shape[1])
    
    @staticmethod
    def extract_edge_density(image: np.ndarray) -> float:
        """Calculate edge density (ratio of edge pixels to total pixels)"""
        edges = cv2.Canny(image, 50, 150)
        return np.sum(edges > 0) / edges.size
    
    @staticmethod
    def extract_contour_features(image: np.ndarray) -> dict:
        """Extract contour-based features"""
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        return {
            'num_contours': len(contours),
            'contour_areas': [cv2.contourArea(c) for c in contours] if contours else [],
        }
    
    @staticmethod
    def extract_hog_features(image: np.ndarray, cell_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients (HOG) features
        
        Args:
            image: Input image
            cell_size: Size of cells for HOG computation
            
        Returns:
            Flattened HOG features
        """
        # Ensure image is square for HOG
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute HOG
        hog = cv2.HOGDescriptor()
        features = hog.compute(image, (cell_size[0], cell_size[1]))
        return features.flatten() if features is not None else np.array([])


def draw_rect(image: np.ndarray, x: int, y: int, width: int, height: int, 
              color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw rectangle on image"""
    result = image.copy()
    cv2.rectangle(result, (x, y), (x + width, y + height), color, thickness)
    return result
