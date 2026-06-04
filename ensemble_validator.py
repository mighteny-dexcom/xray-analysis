"""
Integration example: Combining multiple validation approaches
"""

import cv2
import numpy as np
from image_validator import TemplateMatchingValidator, EdgeBasedValidator, ColorBasedValidator
from image_processor import ImageProcessor


class EnsembleValidator:
    """Combine multiple validators for robust validation"""
    
    def __init__(self):
        self.validators = []
        self.weights = []
    
    def add_validator(self, validator, weight: float = 1.0):
        """Add validator to ensemble"""
        self.validators.append(validator)
        self.weights.append(weight)
    
    def validate(self, image: np.ndarray) -> dict:
        """
        Run all validators and combine results
        
        Returns:
            Dictionary with individual and ensemble results
        """
        results = []
        confidences = []
        
        for validator, weight in zip(self.validators, self.weights):
            result = validator.validate(image)
            results.append(result)
            confidences.append(result.confidence * weight)
        
        # Weighted ensemble decision
        weighted_confidence = np.mean(confidences) if confidences else 0.0
        ensemble_is_valid = weighted_confidence > 0.5
        
        return {
            'individual_results': results,
            'ensemble_confidence': float(weighted_confidence),
            'ensemble_is_valid': ensemble_is_valid,
            'method': 'ensemble'
        }


def visualize_validation_results(image: np.ndarray, result: dict, output_path: str = None):
    """
    Visualize validation results on image
    
    Args:
        image: Original image
        result: Validation result from ensemble or single validator
        output_path: Optional path to save annotated image
    """
    vis_image = image.copy()
    
    # Handle both single and ensemble results
    if 'detected_regions' in result:
        regions = result['detected_regions']
        confidence = result['confidence']
    else:
        regions = result['individual_results'][0].detected_regions
        confidence = result['ensemble_confidence']
    
    # Draw detected regions
    for region in regions:
        loc = region['location']
        size = region['size']
        x, y = loc['x'], loc['y']
        w, h = size['width'], size['height']
        
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add text
    status = "VALID" if result.get('is_valid', result.get('ensemble_is_valid')) else "INVALID"
    text = f"{status} (conf: {confidence:.2f})"
    cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0) if status == "VALID" else (0, 0, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


def main():
    """Example: Using ensemble validator"""
    print("Ensemble Validator Example")
    print("="*50)
    
    # Create test image
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(test_image, (150, 150), 80, (0, 255, 0), -1)
    cv2.rectangle(test_image, (50, 50), (250, 250), (255, 0, 0), 2)
    
    # Create validators
    template = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2.circle(template, (80, 80), 80, (0, 255, 0), -1)
    
    ensemble = EnsembleValidator()
    ensemble.add_validator(TemplateMatchingValidator(template), weight=0.4)
    ensemble.add_validator(EdgeBasedValidator(), weight=0.6)
    
    # Validate
    result = ensemble.validate(test_image)
    
    print(f"Ensemble confidence: {result['ensemble_confidence']:.4f}")
    print(f"Ensemble decision: {'VALID' if result['ensemble_is_valid'] else 'INVALID'}")
    print(f"Individual results: {len(result['individual_results'])} validators")
    
    # Visualize
    vis = visualize_validation_results(test_image, result, 'validation_result.png')
    print("Saved visualization to 'validation_result.png'")


if __name__ == '__main__':
    main()
