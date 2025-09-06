"""
Gaze overlay visualization for video rendering.

Provides comprehensive gaze overlay capabilities for rendering gaze data
onto video frames with configurable appearance and behavior.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from ..logging_setup import get_logger

logger = get_logger(__name__)


def create_gaze_overlay(
    world_video_path: Union[str, Path],
    gaze_df: pd.DataFrame,
    output_path: Union[str, Path],
    fixation_df: Optional[pd.DataFrame] = None,
    dot_radius: int = 8,
    dot_alpha: float = 0.8,
    trail_length: int = 10,
    show_fixations: bool = False,
    fixation_radius: int = 20,
    fixation_alpha: float = 0.6,
    video_fps: Optional[float] = None,
    video_codec: str = "mp4v",
) -> None:
    """Create gaze overlay video."""
    world_video_path = Path(world_video_path)
    output_path = Path(output_path)
    
    if not world_video_path.exists():
        raise FileNotFoundError(f"World video not found: {world_video_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open input video
    cap = cv2.VideoCapture(str(world_video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open world video: {world_video_path}")
    
    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use input FPS if not specified
    if video_fps is None:
        video_fps = input_fps
    
    logger.info(f"Creating gaze overlay: {width}x{height} @ {video_fps} FPS")
    
    # Set up output video
    fourcc = cv2.VideoWriter_fourcc(*video_codec)
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Failed to create output video: {output_path}")
    
    # Prepare gaze data
    gaze_timestamps = gaze_df['t_ns'].values
    gaze_x = gaze_df['gx_px'].values
    gaze_y = gaze_df['gy_px'].values
    
    # Process frames
    frame_count = 0
    gaze_trail = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate frame timestamp
        frame_timestamp_ns = int((frame_count / input_fps) * 1_000_000_000)
        
        # Find gaze points for this frame
        frame_gaze_points = _find_gaze_points_for_frame(
            frame_timestamp_ns, gaze_timestamps, gaze_x, gaze_y, trail_length
        )
        
        # Update gaze trail
        if frame_gaze_points:
            gaze_trail.extend(frame_gaze_points)
            # Keep only recent points
            if len(gaze_trail) > trail_length * 10:
                gaze_trail = gaze_trail[-trail_length * 10:]
        
        # Draw overlay
        overlay_frame = _draw_gaze_overlay(
            frame, gaze_trail, dot_radius, dot_alpha
        )
        
        # Write frame
        out.write(overlay_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{total_frames} frames")
    
    # Clean up
    cap.release()
    out.release()
    
    logger.info(f"Created gaze overlay video: {output_path}")


def _find_gaze_points_for_frame(
    frame_timestamp_ns: int,
    gaze_timestamps: np.ndarray,
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    trail_length: int,
) -> List[Tuple[float, float]]:
    """Find gaze points for a specific frame timestamp."""
    # Find gaze points within a small time window around the frame
    time_window_ns = 16_666_667  # ~16.67ms (60 FPS)
    
    mask = np.abs(gaze_timestamps - frame_timestamp_ns) <= time_window_ns
    if not np.any(mask):
        return []
    
    # Get gaze points
    frame_gaze_x = gaze_x[mask]
    frame_gaze_y = gaze_y[mask]
    
    # Return as list of tuples
    return list(zip(frame_gaze_x, frame_gaze_y))


def _draw_gaze_overlay(
    frame: np.ndarray,
    gaze_trail: List[Tuple[float, float]],
    dot_radius: int,
    dot_alpha: float,
) -> np.ndarray:
    """Draw gaze overlay on frame."""
    overlay_frame = frame.copy()
    
    # Draw gaze trail
    if gaze_trail:
        for i, (x, y) in enumerate(gaze_trail):
            # Fade trail based on age
            trail_alpha = dot_alpha * (1.0 - i / len(gaze_trail))
            
            if trail_alpha > 0:
                # Draw gaze dot
                cv2.circle(overlay_frame, (int(x), int(y)), dot_radius, (0, 255, 0), -1)
    
    return overlay_frame
