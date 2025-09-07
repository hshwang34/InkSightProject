"""
Test peak dwell timestamp selection functionality.

Tests the peak_dwell_timestamp function with synthetic gaze data
to ensure it correctly identifies periods of maximum attention.
"""

import pytest
import numpy as np
import pandas as pd

from gaze_lab.processing.aoi import peak_dwell_timestamp


class TestPeakDwellSelection:
    """Test peak dwell timestamp selection."""
    
    @pytest.fixture
    def simple_aoi(self):
        """Create a simple rectangular AOI."""
        return {
            "name": "test_aoi",
            "type": "rectangle",
            "coordinates": [[100, 100], [200, 100], [200, 200], [100, 200]],
            "metadata": {}
        }
    
    @pytest.fixture
    def circular_aoi(self):
        """Create a circular AOI."""
        return {
            "name": "circle_aoi",
            "type": "circle",
            "coordinates": [[150, 150, 50]],  # center (150, 150), radius 50
            "metadata": {}
        }
    
    def create_gaze_data(self, timestamps_ns, gx_coords, gy_coords):
        """Helper to create gaze DataFrame."""
        return pd.DataFrame({
            "t_ns": timestamps_ns,
            "gx_px": gx_coords,
            "gy_px": gy_coords,
            "frame_w": [1280] * len(timestamps_ns),
            "frame_h": [720] * len(timestamps_ns),
        })
    
    def test_peak_dwell_single_cluster(self, simple_aoi):
        """Test peak dwell with single cluster of gaze points."""
        # Create gaze data with one concentrated period inside AOI
        timestamps = np.linspace(0, 10_000_000_000, 100)  # 10 seconds
        
        # Most points outside AOI
        gx_coords = [50] * 40 + [150] * 20 + [50] * 40  # Inside AOI for middle 20 points
        gy_coords = [50] * 40 + [150] * 20 + [50] * 40
        
        gaze_df = self.create_gaze_data(timestamps, gx_coords, gy_coords)
        
        result_ts = peak_dwell_timestamp(gaze_df, simple_aoi, window_ms=2000)
        
        # Should return timestamp near the middle of the sequence (where AOI points are)
        expected_center = timestamps[50]  # Middle of the sequence
        assert abs(result_ts - expected_center) < 2_000_000_000  # Within 2 seconds
    
    def test_peak_dwell_multiple_clusters(self, simple_aoi):
        """Test peak dwell with multiple clusters, should pick the densest."""
        timestamps = np.linspace(0, 20_000_000_000, 200)  # 20 seconds
        
        # Two clusters: first smaller, second larger
        gx_coords = (
            [50] * 50 +          # Outside AOI
            [150] * 30 +         # First cluster (smaller)
            [50] * 40 +          # Outside AOI
            [150] * 60 +         # Second cluster (larger) - this should win
            [50] * 20            # Outside AOI
        )
        gy_coords = (
            [50] * 50 +
            [150] * 30 +
            [50] * 40 +
            [150] * 60 +
            [50] * 20
        )
        
        gaze_df = self.create_gaze_data(timestamps, gx_coords, gy_coords)
        
        result_ts = peak_dwell_timestamp(gaze_df, simple_aoi, window_ms=5000)
        
        # Should be closer to the larger cluster (around index 150)
        larger_cluster_center = timestamps[150]
        assert abs(result_ts - larger_cluster_center) < 5_000_000_000  # Within 5 seconds
    
    def test_peak_dwell_circular_aoi(self, circular_aoi):
        """Test peak dwell with circular AOI."""
        timestamps = np.linspace(0, 10_000_000_000, 100)
        
        # Points inside circle (center 150,150, radius 50)
        gx_coords = [150] * 100  # All at center
        gy_coords = [150] * 100
        
        gaze_df = self.create_gaze_data(timestamps, gx_coords, gy_coords)
        
        result_ts = peak_dwell_timestamp(gaze_df, circular_aoi, window_ms=2000)
        
        # Should return a valid timestamp
        assert 0 <= result_ts <= 10_000_000_000
    
    def test_peak_dwell_no_aoi_points(self, simple_aoi):
        """Test peak dwell when no points are in AOI."""
        timestamps = np.linspace(0, 10_000_000_000, 100)
        
        # All points outside AOI
        gx_coords = [50] * 100  # Outside rectangle
        gy_coords = [50] * 100
        
        gaze_df = self.create_gaze_data(timestamps, gx_coords, gy_coords)
        
        with pytest.raises(ValueError) as exc_info:
            peak_dwell_timestamp(gaze_df, simple_aoi, window_ms=2000)
        
        assert "No gaze data found within AOI" in str(exc_info.value)
    
    def test_peak_dwell_empty_data(self, simple_aoi):
        """Test peak dwell with empty gaze data."""
        empty_df = pd.DataFrame(columns=["t_ns", "gx_px", "gy_px", "frame_w", "frame_h"])
        
        with pytest.raises(ValueError) as exc_info:
            peak_dwell_timestamp(empty_df, simple_aoi, window_ms=2000)
        
        assert "Gaze data is empty" in str(exc_info.value)
    
    def test_peak_dwell_invalid_window(self, simple_aoi):
        """Test peak dwell with invalid window size."""
        timestamps = np.linspace(0, 10_000_000_000, 100)
        gx_coords = [150] * 100
        gy_coords = [150] * 100
        gaze_df = self.create_gaze_data(timestamps, gx_coords, gy_coords)
        
        with pytest.raises(ValueError) as exc_info:
            peak_dwell_timestamp(gaze_df, simple_aoi, window_ms=0)
        
        assert "Window size must be positive" in str(exc_info.value)
    
    def test_peak_dwell_short_data_span(self, simple_aoi):
        """Test peak dwell when data span is shorter than window."""
        # Very short time span (1 second) with large window (5 seconds)
        timestamps = np.linspace(0, 1_000_000_000, 10)
        gx_coords = [150] * 10  # Inside AOI
        gy_coords = [150] * 10
        
        gaze_df = self.create_gaze_data(timestamps, gx_coords, gy_coords)
        
        result_ts = peak_dwell_timestamp(gaze_df, simple_aoi, window_ms=5000)
        
        # Should return middle timestamp
        expected_middle = 500_000_000  # 0.5 seconds
        assert abs(result_ts - expected_middle) < 100_000_000  # Within 0.1 seconds
    
    def test_peak_dwell_with_nan_values(self, simple_aoi):
        """Test peak dwell handling of NaN values (blinks)."""
        timestamps = np.linspace(0, 10_000_000_000, 100)
        
        # Mix of valid points and NaN (blinks)
        gx_coords = [150] * 50 + [np.nan] * 20 + [150] * 30
        gy_coords = [150] * 50 + [np.nan] * 20 + [150] * 30
        
        gaze_df = self.create_gaze_data(timestamps, gx_coords, gy_coords)
        
        result_ts = peak_dwell_timestamp(gaze_df, simple_aoi, window_ms=2000)
        
        # Should work despite NaN values
        assert isinstance(result_ts, (int, np.integer))
        assert 0 <= result_ts <= 10_000_000_000
    
    def test_peak_dwell_polygon_aoi(self):
        """Test peak dwell with polygon AOI."""
        # Triangle AOI
        triangle_aoi = {
            "name": "triangle",
            "type": "polygon",
            "coordinates": [[100, 100], [200, 100], [150, 200]],
            "metadata": {}
        }
        
        timestamps = np.linspace(0, 10_000_000_000, 100)
        
        # Points inside triangle
        gx_coords = [150] * 100  # Center of triangle
        gy_coords = [120] * 100
        
        gaze_df = self.create_gaze_data(timestamps, gx_coords, gy_coords)
        
        result_ts = peak_dwell_timestamp(gaze_df, triangle_aoi, window_ms=2000)
        
        assert isinstance(result_ts, (int, np.integer))
        assert 0 <= result_ts <= 10_000_000_000
