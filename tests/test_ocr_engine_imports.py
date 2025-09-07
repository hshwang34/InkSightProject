"""
Test OCR engine import handling and fallback behavior.

Tests that OCR engines handle missing dependencies gracefully
and provide helpful error messages.
"""

import pytest
import numpy as np

from gaze_lab.ocr.engine import (
    recognize_text,
    get_available_engines,
    is_ocr_available,
    TESSERACT_AVAILABLE,
    EASYOCR_AVAILABLE,
)


class TestOCREngineImports:
    """Test OCR engine availability and fallback behavior."""
    
    def test_engine_availability_flags(self):
        """Test that availability flags are boolean."""
        assert isinstance(TESSERACT_AVAILABLE, bool)
        assert isinstance(EASYOCR_AVAILABLE, bool)
    
    def test_get_available_engines(self):
        """Test getting list of available engines."""
        engines = get_available_engines()
        assert isinstance(engines, list)
        
        # Should only contain valid engine names
        valid_engines = {"tesseract", "easyocr"}
        for engine in engines:
            assert engine in valid_engines
    
    def test_is_ocr_available(self):
        """Test OCR availability check."""
        available = is_ocr_available()
        assert isinstance(available, bool)
        
        # Should match individual engine availability
        expected = TESSERACT_AVAILABLE or EASYOCR_AVAILABLE
        assert available == expected
    
    def test_recognize_text_with_no_engines(self, monkeypatch):
        """Test that recognize_text fails gracefully when no engines available."""
        # Mock both engines as unavailable
        monkeypatch.setattr("gaze_lab.ocr.engine.TESSERACT_AVAILABLE", False)
        monkeypatch.setattr("gaze_lab.ocr.engine.EASYOCR_AVAILABLE", False)
        
        # Create a simple test image
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        with pytest.raises(RuntimeError) as exc_info:
            recognize_text(test_image)
        
        error_msg = str(exc_info.value)
        assert "No OCR engines available" in error_msg
        assert "pip install pytesseract" in error_msg
        assert "pip install easyocr" in error_msg
    
    @pytest.mark.skipif(not is_ocr_available(), reason="No OCR engines available")
    def test_recognize_text_with_available_engines(self):
        """Test OCR with available engines (if any)."""
        # Create a simple test image with high contrast text
        test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255
        
        # This should not raise an exception
        try:
            result = recognize_text(test_image)
            assert isinstance(result, str)
        except RuntimeError as e:
            # OCR might fail on blank image, but should not be import error
            assert "not available" not in str(e).lower()
    
    def test_recognize_text_invalid_engine_preference(self):
        """Test handling of invalid engine preferences."""
        test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255
        
        if not is_ocr_available():
            with pytest.raises(RuntimeError) as exc_info:
                recognize_text(test_image, engine_preference=("invalid_engine",))
            assert "No OCR engines available" in str(exc_info.value)
        else:
            # With available engines, invalid preference should try available ones
            try:
                result = recognize_text(test_image, engine_preference=("invalid_engine",))
                assert isinstance(result, str)
            except RuntimeError:
                # May fail due to OCR processing, not engine availability
                pass
