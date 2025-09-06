"""
Smoke tests for heatmap functionality.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaze_lab.viz.heatmap import create_heatmap, get_heatmap_statistics


class TestHeatmapSmoke:
    """Smoke tests for heatmap functionality."""
    
    def test_create_heatmap_basic(self):
        """Test basic heatmap creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test gaze data
            gaze_data = []
            for i in range(200):
                gaze_data.append({
                    't_ns': i * 10_000_000,
                    'gx_px': np.random.normal(640, 100),  # Center around 640
                    'gy_px': np.random.normal(360, 100),  # Center around 360
                    'frame_w': 1280,
                    'frame_h': 720,
                })
            
            gaze_df = pd.DataFrame(gaze_data)
            
            # Create heatmap
            output_path = temp_path / "heatmap.png"
            create_heatmap(
                gaze_df=gaze_df,
                output_path=output_path,
                background_mode="none",
                bandwidth=20.0,
                grid_size=100,
            )
            
            # Check that output file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_create_heatmap_with_background(self):
        """Test heatmap creation with background."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test gaze data
            gaze_data = []
            for i in range(100):
                gaze_data.append({
                    't_ns': i * 10_000_000,
                    'gx_px': np.random.normal(640, 50),
                    'gy_px': np.random.normal(360, 50),
                    'frame_w': 1280,
                    'frame_h': 720,
                })
            
            gaze_df = pd.DataFrame(gaze_data)
            
            # Create heatmap with world mode
            output_path = temp_path / "heatmap_world.png"
            create_heatmap(
                gaze_df=gaze_df,
                output_path=output_path,
                background_mode="world",
                world_video_path="data/world.mp4",  # This will fail but we're testing the function
                bandwidth=15.0,
                grid_size=50,
            )
            
            # The function should handle missing video gracefully
            # We're just testing that it doesn't crash
    
    def test_heatmap_statistics(self):
        """Test heatmap statistics calculation."""
        # Create test gaze data
        gaze_data = []
        for i in range(100):
            gaze_data.append({
                't_ns': i * 10_000_000,
                'gx_px': np.random.normal(640, 100),
                'gy_px': np.random.normal(360, 100),
                'frame_w': 1280,
                'frame_h': 720,
            })
        
        gaze_df = pd.DataFrame(gaze_data)
        
        # Get statistics
        stats = get_heatmap_statistics(gaze_df)
        
        # Check that statistics were calculated
        assert stats['gaze_points'] == 100
        assert 'x_range' in stats
        assert 'y_range' in stats
        assert 'x_mean' in stats
        assert 'y_mean' in stats
        assert 'mean_distance_from_center' in stats
    
    def test_heatmap_with_empty_data(self):
        """Test heatmap creation with empty gaze data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create empty gaze data
            gaze_df = pd.DataFrame(columns=['t_ns', 'gx_px', 'gy_px', 'frame_w', 'frame_h'])
            
            # This should raise an error
            output_path = temp_path / "heatmap.png"
            with pytest.raises(ValueError, match="No gaze data provided"):
                create_heatmap(
                    gaze_df=gaze_df,
                    output_path=output_path,
                    background_mode="none",
                )
    
    def test_heatmap_custom_parameters(self):
        """Test heatmap creation with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test gaze data
            gaze_data = []
            for i in range(50):
                gaze_data.append({
                    't_ns': i * 10_000_000,
                    'gx_px': np.random.normal(640, 50),
                    'gy_px': np.random.normal(360, 50),
                    'frame_w': 1280,
                    'frame_h': 720,
                })
            
            gaze_df = pd.DataFrame(gaze_data)
            
            # Create heatmap with custom parameters
            output_path = temp_path / "heatmap_custom.png"
            create_heatmap(
                gaze_df=gaze_df,
                output_path=output_path,
                background_mode="none",
                bandwidth=30.0,
                grid_size=200,
                colormap="viridis",
                alpha=0.8,
                dpi=150,
            )
            
            # Check that output file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
