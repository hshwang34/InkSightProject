"""
Smoke tests for gaze overlay functionality.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from gaze_lab.viz.overlay import create_gaze_overlay, create_static_overlay


class TestOverlaySmoke:
    """Smoke tests for overlay functionality."""
    
    def test_create_static_overlay(self):
        """Test static overlay creation."""
        # Create test frame
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 240
        
        # Create test gaze data
        gaze_data = []
        for i in range(10):
            gaze_data.append({
                't_ns': i * 10_000_000,
                'gx_px': 100 + i * 10,
                'gy_px': 100 + i * 5,
                'frame_w': 1280,
                'frame_h': 720,
            })
        
        gaze_df = pd.DataFrame(gaze_data)
        
        # Create static overlay
        overlay_frame = create_static_overlay(frame, gaze_df)
        
        # Check that overlay was created
        assert overlay_frame is not None
        assert overlay_frame.shape == frame.shape
        assert not np.array_equal(overlay_frame, frame)  # Should be different from original
    
    def test_create_gaze_overlay_video(self):
        """Test gaze overlay video creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test video
            video_path = temp_path / "test_video.mp4"
            self._create_test_video(video_path)
            
            # Create test gaze data
            gaze_data = []
            for i in range(15):  # 0.5 seconds at 30 FPS
                gaze_data.append({
                    't_ns': i * 33_333_333,  # ~30 FPS
                    'gx_px': 100 + i * 10,
                    'gy_px': 100 + i * 5,
                    'frame_w': 1280,
                    'frame_h': 720,
                })
            
            gaze_df = pd.DataFrame(gaze_data)
            
            # Create overlay video
            output_path = temp_path / "overlay.mp4"
            create_gaze_overlay(
                world_video_path=video_path,
                gaze_df=gaze_df,
                output_path=output_path,
                dot_radius=5,
                trail_length=5,
            )
            
            # Check that output file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            # Verify video properties
            cap = cv2.VideoCapture(str(output_path))
            assert cap.isOpened()
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            assert frame_count > 0
            assert width == 1280
            assert height == 720
            
            cap.release()
    
    def _create_test_video(self, output_path: Path, frames: int = 15) -> None:
        """Create a simple test video."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (1280, 720))
        
        for i in range(frames):
            # Create frame with moving square
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 240
            
            # Draw moving square
            x = int(100 + i * 20)
            y = int(100 + i * 10)
            cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 100, 200), -1)
            
            out.write(frame)
        
        out.release()
    
    def test_overlay_with_empty_gaze_data(self):
        """Test overlay creation with empty gaze data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test video
            video_path = temp_path / "test_video.mp4"
            self._create_test_video(video_path)
            
            # Create empty gaze data
            gaze_df = pd.DataFrame(columns=['t_ns', 'gx_px', 'gy_px', 'frame_w', 'frame_h'])
            
            # Create overlay video
            output_path = temp_path / "overlay.mp4"
            create_gaze_overlay(
                world_video_path=video_path,
                gaze_df=gaze_df,
                output_path=output_path,
            )
            
            # Check that output file was created (should be same as input)
            assert output_path.exists()
            assert output_path.stat().st_size > 0