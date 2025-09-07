"""
Smoke tests for OCR preprocessing utilities.

Tests that preprocessing functions work correctly and return
images in the expected format.
"""

import pytest
import numpy as np

from gaze_lab.ocr.preprocess import (
    prep_for_ocr,
    enhance_contrast,
    denoise_image,
    binarize_image,
)


class TestPreprocessSmoke:
    """Smoke tests for preprocessing functions."""
    
    @pytest.fixture
    def test_image_color(self):
        """Create a test color image."""
        # Create a simple image with some structure
        image = np.ones((100, 200, 3), dtype=np.uint8) * 128
        # Add some variation
        image[20:40, 50:150] = [200, 200, 200]  # Light rectangle
        image[60:80, 75:125] = [50, 50, 50]     # Dark rectangle
        return image
    
    @pytest.fixture
    def test_image_gray(self):
        """Create a test grayscale image."""
        image = np.ones((100, 200), dtype=np.uint8) * 128
        # Add some variation
        image[20:40, 50:150] = 200  # Light rectangle
        image[60:80, 75:125] = 50   # Dark rectangle
        return image
    
    def test_prep_for_ocr_modes(self, test_image_color):
        """Test prep_for_ocr with different modes."""
        modes = ["auto", "doc", "ui"]
        
        for mode in modes:
            result = prep_for_ocr(test_image_color, mode=mode)
            
            # Should return grayscale image
            assert len(result.shape) == 2
            assert result.dtype == np.uint8
            
            # Should have same height/width as input
            assert result.shape[0] == test_image_color.shape[0]
            assert result.shape[1] == test_image_color.shape[1]
    
    def test_prep_for_ocr_grayscale_input(self, test_image_gray):
        """Test prep_for_ocr with grayscale input."""
        result = prep_for_ocr(test_image_gray, mode="auto")
        
        # Should return grayscale image
        assert len(result.shape) == 2
        assert result.dtype == np.uint8
        assert result.shape == test_image_gray.shape
    
    def test_prep_for_ocr_with_deskew(self, test_image_color):
        """Test prep_for_ocr with deskewing enabled."""
        result = prep_for_ocr(test_image_color, mode="auto", deskew=True)
        
        # Should return grayscale image
        assert len(result.shape) == 2
        assert result.dtype == np.uint8
        assert result.shape[0] == test_image_color.shape[0]
        assert result.shape[1] == test_image_color.shape[1]
    
    def test_prep_for_ocr_invalid_mode(self, test_image_color):
        """Test prep_for_ocr with invalid mode."""
        with pytest.raises(ValueError) as exc_info:
            prep_for_ocr(test_image_color, mode="invalid")
        
        assert "Unknown preprocessing mode" in str(exc_info.value)
    
    def test_prep_for_ocr_empty_image(self):
        """Test prep_for_ocr with empty image."""
        empty_image = np.array([])
        
        with pytest.raises(ValueError) as exc_info:
            prep_for_ocr(empty_image)
        
        assert "Input image is empty" in str(exc_info.value)
    
    def test_enhance_contrast_methods(self, test_image_gray):
        """Test contrast enhancement methods."""
        methods = ["clahe", "histogram", "gamma"]
        
        for method in methods:
            result = enhance_contrast(test_image_gray, method=method)
            
            assert result.shape == test_image_gray.shape
            assert result.dtype == np.uint8
    
    def test_enhance_contrast_invalid_method(self, test_image_gray):
        """Test enhance_contrast with invalid method."""
        with pytest.raises(ValueError) as exc_info:
            enhance_contrast(test_image_gray, method="invalid")
        
        assert "Unknown contrast enhancement method" in str(exc_info.value)
    
    def test_denoise_image_methods(self, test_image_gray):
        """Test denoising methods."""
        methods = ["median", "gaussian", "bilateral"]
        
        for method in methods:
            result = denoise_image(test_image_gray, method=method)
            
            assert result.shape == test_image_gray.shape
            assert result.dtype == np.uint8
    
    def test_denoise_image_invalid_method(self, test_image_gray):
        """Test denoise_image with invalid method."""
        with pytest.raises(ValueError) as exc_info:
            denoise_image(test_image_gray, method="invalid")
        
        assert "Unknown denoising method" in str(exc_info.value)
    
    def test_binarize_image_methods(self, test_image_gray):
        """Test binarization methods."""
        methods = ["adaptive", "otsu", "threshold"]
        
        for method in methods:
            result = binarize_image(test_image_gray, method=method)
            
            assert result.shape == test_image_gray.shape
            assert result.dtype == np.uint8
            
            # Binary image should only have 0 and 255 values
            unique_values = np.unique(result)
            assert len(unique_values) <= 2
            if len(unique_values) == 2:
                assert 0 in unique_values
                assert 255 in unique_values
    
    def test_binarize_image_invalid_method(self, test_image_gray):
        """Test binarize_image with invalid method."""
        with pytest.raises(ValueError) as exc_info:
            binarize_image(test_image_gray, method="invalid")
        
        assert "Unknown binarization method" in str(exc_info.value)
