"""
Smoke tests for snapshot extraction and AOI cropping.

Tests that frame extraction and cropping functions work correctly
and return images in the expected format.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from gaze_lab.snapshots.extractor import (
    extract_frame_at_ts,
    crop_aoi,
    get_video_info,
    validate_timestamp,
)


class TestSnapshotCropSmoke:
    """Smoke tests for snapshot extraction functions."""
    
    @pytest.fixture
    def test_frame(self):
        """Create a test frame image."""
        # Create a 640x480 color frame with some structure
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        
        # Add some colored regions
        frame[100:200, 100:300] = [255, 0, 0]    # Red rectangle
        frame[300:400, 200:500] = [0, 255, 0]    # Green rectangle
        frame[50:150, 400:600] = [0, 0, 255]     # Blue rectangle
        
        return frame
    
    @pytest.fixture
    def rectangle_aoi(self):
        """Create a rectangular AOI."""
        return {
            "name": "test_rect",
            "type": "rectangle",
            "coordinates": [[100, 100], [300, 100], [300, 200], [100, 200]],
            "metadata": {}
        }
    
    @pytest.fixture
    def circle_aoi(self):
        """Create a circular AOI."""
        return {
            "name": "test_circle",
            "type": "circle",
            "coordinates": [[320, 240, 50]],  # center (320, 240), radius 50
            "metadata": {}
        }
    
    @pytest.fixture
    def polygon_aoi(self):
        """Create a polygon AOI."""
        return {
            "name": "test_polygon",
            "type": "polygon",
            "coordinates": [[200, 300], [500, 300], [400, 400], [300, 400]],
            "metadata": {}
        }
    
    def test_crop_aoi_rectangle(self, test_frame, rectangle_aoi):
        """Test cropping rectangular AOI."""
        result = crop_aoi(test_frame, rectangle_aoi)
        
        # Should return cropped region
        assert len(result.shape) == 3  # Color image
        assert result.dtype == np.uint8
        
        # Check dimensions (should be 200x100 based on coordinates)
        expected_height = 200 - 100  # y2 - y1
        expected_width = 300 - 100   # x2 - x1
        assert result.shape[0] == expected_height
        assert result.shape[1] == expected_width
        assert result.shape[2] == 3  # Color channels
    
    def test_crop_aoi_circle(self, test_frame, circle_aoi):
        """Test cropping circular AOI (as bounding box)."""
        result = crop_aoi(test_frame, circle_aoi)
        
        # Should return cropped region
        assert len(result.shape) == 3
        assert result.dtype == np.uint8
        
        # Should be roughly 100x100 (2*radius)
        assert result.shape[0] == 100  # 2 * radius
        assert result.shape[1] == 100  # 2 * radius
        assert result.shape[2] == 3
    
    def test_crop_aoi_polygon(self, test_frame, polygon_aoi):
        """Test cropping polygon AOI."""
        result = crop_aoi(test_frame, polygon_aoi)
        
        # Should return cropped region
        assert len(result.shape) == 3
        assert result.dtype == np.uint8
        
        # Should have reasonable dimensions
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        assert result.shape[2] == 3
    
    def test_crop_aoi_empty_frame(self, rectangle_aoi):
        """Test cropping with empty frame."""
        empty_frame = np.array([])
        
        with pytest.raises(ValueError) as exc_info:
            crop_aoi(empty_frame, rectangle_aoi)
        
        assert "Input frame is empty" in str(exc_info.value)
    
    def test_crop_aoi_invalid_type(self, test_frame):
        """Test cropping with invalid AOI type."""
        invalid_aoi = {
            "name": "invalid",
            "type": "invalid_type",
            "coordinates": [[100, 100], [200, 200]],
            "metadata": {}
        }
        
        with pytest.raises(ValueError) as exc_info:
            crop_aoi(test_frame, invalid_aoi)
        
        assert "Unsupported AOI type" in str(exc_info.value)
    
    def test_crop_aoi_empty_coordinates(self, test_frame):
        """Test cropping with empty coordinates."""
        empty_coords_aoi = {
            "name": "empty",
            "type": "rectangle",
            "coordinates": [],
            "metadata": {}
        }
        
        with pytest.raises(ValueError) as exc_info:
            crop_aoi(test_frame, empty_coords_aoi)
        
        assert "AOI coordinates are empty" in str(exc_info.value)
    
    def test_crop_aoi_out_of_bounds(self, test_frame):
        """Test cropping with coordinates outside frame bounds."""
        # AOI extends beyond frame boundaries
        out_of_bounds_aoi = {
            "name": "oob",
            "type": "rectangle",
            "coordinates": [[500, 400], [800, 400], [800, 600], [500, 600]],
            "metadata": {}
        }
        
        # Should not crash, but clamp to frame bounds
        result = crop_aoi(test_frame, out_of_bounds_aoi)
        
        assert len(result.shape) == 3
        assert result.dtype == np.uint8
        assert result.shape[0] > 0  # Should have some height
        assert result.shape[1] > 0  # Should have some width
    
    def test_crop_aoi_grayscale_frame(self, rectangle_aoi):
        """Test cropping with grayscale frame."""
        gray_frame = np.ones((480, 640), dtype=np.uint8) * 128
        gray_frame[100:200, 100:300] = 255  # White rectangle
        
        result = crop_aoi(gray_frame, rectangle_aoi)
        
        # Should return grayscale crop
        assert len(result.shape) == 2  # Grayscale
        assert result.dtype == np.uint8
        assert result.shape[0] == 100  # Height
        assert result.shape[1] == 200  # Width
    
    def test_get_video_info_nonexistent_file(self):
        """Test getting video info for non-existent file."""
        info = get_video_info("nonexistent_video.mp4")
        
        # Should return empty dict for non-existent file
        assert info == {}
    
    def test_validate_timestamp_nonexistent_file(self):
        """Test timestamp validation for non-existent file."""
        result = validate_timestamp(5_000_000_000, "nonexistent_video.mp4")
        
        # Should return False for non-existent file
        assert result is False
    
    def test_extract_frame_nonexistent_file(self):
        """Test frame extraction from non-existent file."""
        with pytest.raises(ValueError) as exc_info:
            extract_frame_at_ts("nonexistent_video.mp4", 5_000_000_000)
        
        assert "Video file not found" in str(exc_info.value)
    
    # Note: Testing actual video file operations would require creating
    # temporary video files, which is complex and slow. The above tests
    # cover the main error conditions and input validation.
