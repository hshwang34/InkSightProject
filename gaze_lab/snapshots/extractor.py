"""
Frame extraction and AOI cropping utilities.

Provides functions to extract video frames at specific timestamps
and crop areas of interest for OCR processing.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon, Point

from ..logging_setup import get_logger

logger = get_logger(__name__)


def extract_frame_at_ts(world_path: str, t_ns: int) -> np.ndarray:
    """
    Extract the video frame nearest to the specified timestamp.
    
    Args:
        world_path: Path to world camera video file
        t_ns: Target timestamp in nanoseconds
        
    Returns:
        Frame as numpy array (BGR format)
        
    Raises:
        ValueError: If video cannot be opened or timestamp is invalid
        RuntimeError: If frame extraction fails
    """
    world_path = Path(world_path)
    if not world_path.exists():
        raise ValueError(f"Video file not found: {world_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(world_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {world_path}")
    
    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0
        
        logger.debug(f"Video: {fps:.2f} FPS, {frame_count} frames, {duration_sec:.2f}s duration")
        
        # Convert timestamp to seconds
        t_sec = t_ns / 1_000_000_000
        
        if t_sec < 0 or t_sec > duration_sec:
            logger.warning(f"Timestamp {t_sec:.3f}s outside video duration (0-{duration_sec:.3f}s)")
        
        # Calculate target frame index
        target_frame = int(t_sec * fps)
        target_frame = max(0, min(target_frame, frame_count - 1))
        
        logger.debug(f"Extracting frame {target_frame} for timestamp {t_sec:.3f}s")
        
        # Seek to target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # Read frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame {target_frame}")
        
        logger.debug(f"Extracted frame shape: {frame.shape}")
        return frame
        
    finally:
        cap.release()


def crop_aoi(frame: np.ndarray, aoi: Dict) -> np.ndarray:
    """
    Crop area of interest from frame.
    
    Args:
        frame: Input frame as numpy array
        aoi: AOI dictionary with 'type' and 'coordinates' keys
        
    Returns:
        Cropped region as numpy array
        
    Raises:
        ValueError: If AOI format is invalid or coordinates are out of bounds
    """
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty")
    
    aoi_type = aoi.get("type", "").lower()
    coordinates = aoi.get("coordinates", [])
    
    if not coordinates:
        raise ValueError("AOI coordinates are empty")
    
    h, w = frame.shape[:2]
    
    if aoi_type == "rectangle":
        return _crop_rectangle(frame, coordinates, w, h)
    elif aoi_type == "circle":
        return _crop_circle(frame, coordinates, w, h)
    elif aoi_type == "polygon":
        return _crop_polygon(frame, coordinates, w, h)
    else:
        raise ValueError(f"Unsupported AOI type: {aoi_type}")


def _crop_rectangle(frame: np.ndarray, coords: list, w: int, h: int) -> np.ndarray:
    """Crop rectangular AOI."""
    if len(coords) != 4:
        raise ValueError(f"Rectangle AOI requires 4 points, got {len(coords)}")
    
    # Convert to x1, y1, x2, y2
    points = np.array(coords)
    x1, y1 = np.min(points, axis=0).astype(int)
    x2, y2 = np.max(points, axis=0).astype(int)
    
    # Clamp to frame bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    
    logger.debug(f"Cropping rectangle: ({x1}, {y1}) to ({x2}, {y2})")
    
    return frame[y1:y2, x1:x2]


def _crop_circle(frame: np.ndarray, coords: list, w: int, h: int) -> np.ndarray:
    """Crop circular AOI as bounding box."""
    if len(coords) != 1 or len(coords[0]) != 3:
        raise ValueError("Circle AOI requires format [[x, y, radius]]")
    
    cx, cy, radius = coords[0]
    
    # Calculate bounding box
    x1 = max(0, int(cx - radius))
    y1 = max(0, int(cy - radius))
    x2 = min(w, int(cx + radius))
    y2 = min(h, int(cy + radius))
    
    logger.debug(f"Cropping circle center ({cx}, {cy}) radius {radius} as bbox ({x1}, {y1}) to ({x2}, {y2})")
    
    return frame[y1:y2, x1:x2]


def _crop_polygon(frame: np.ndarray, coords: list, w: int, h: int) -> np.ndarray:
    """Crop polygonal AOI using mask."""
    if len(coords) < 3:
        raise ValueError("Polygon AOI requires at least 3 points")
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    points = np.array(coords, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    
    # Find bounding box of the polygon
    x, y, bbox_w, bbox_h = cv2.boundingRect(points)
    
    # Clamp to frame bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bbox_w)
    y2 = min(h, y + bbox_h)
    
    logger.debug(f"Cropping polygon with {len(coords)} points, bbox ({x1}, {y1}) to ({x2}, {y2})")
    
    # Crop frame and mask
    cropped_frame = frame[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    # Apply mask (set background to white for better OCR)
    if len(cropped_frame.shape) == 3:
        cropped_frame[cropped_mask == 0] = [255, 255, 255]
    else:
        cropped_frame[cropped_mask == 0] = 255
    
    return cropped_frame


def warp_to_reference(
    frame: np.ndarray,
    aoi: Dict,
    reference_image_path: str,
    homography: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Warp AOI region to reference image coordinates using homography.
    
    Args:
        frame: Input frame
        aoi: AOI dictionary
        reference_image_path: Path to reference image
        homography: Precomputed homography matrix (if None, will attempt to compute)
        
    Returns:
        Warped AOI region aligned to reference image
        
    Raises:
        ValueError: If reference image cannot be loaded or homography computation fails
        NotImplementedError: If homography computation is not available
    """
    ref_path = Path(reference_image_path)
    if not ref_path.exists():
        raise ValueError(f"Reference image not found: {ref_path}")
    
    # Load reference image
    ref_image = cv2.imread(str(ref_path))
    if ref_image is None:
        raise ValueError(f"Cannot load reference image: {ref_path}")
    
    # If no homography provided, we would need to compute it
    # This requires feature matching between frame and reference
    if homography is None:
        logger.warning("Homography computation not implemented in this version")
        raise NotImplementedError(
            "Homography computation requires feature matching implementation. "
            "Please provide a precomputed homography matrix."
        )
    
    # Get AOI coordinates
    aoi_type = aoi.get("type", "").lower()
    coordinates = aoi.get("coordinates", [])
    
    if aoi_type == "rectangle" and len(coordinates) == 4:
        # Convert rectangle to 4 corner points
        points = np.array(coordinates, dtype=np.float32)
    elif aoi_type == "polygon":
        points = np.array(coordinates, dtype=np.float32)
    else:
        raise ValueError(f"Warping not supported for AOI type: {aoi_type}")
    
    # Transform AOI points using homography
    transformed_points = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    
    # Get bounding box of transformed points
    x_min, y_min = np.min(transformed_points, axis=0).astype(int)
    x_max, y_max = np.max(transformed_points, axis=0).astype(int)
    
    # Clamp to reference image bounds
    ref_h, ref_w = ref_image.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(ref_w, x_max)
    y_max = min(ref_h, y_max)
    
    # Warp the entire frame
    warped_frame = cv2.warpPerspective(
        frame, homography, (ref_w, ref_h)
    )
    
    # Extract the AOI region from warped frame
    warped_aoi = warped_frame[y_min:y_max, x_min:x_max]
    
    logger.debug(f"Warped AOI region: {warped_aoi.shape}")
    
    return warped_aoi


def get_video_info(video_path: str) -> Dict:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration,
        }
    finally:
        cap.release()


def validate_timestamp(t_ns: int, video_path: str) -> bool:
    """
    Validate that timestamp is within video duration.
    
    Args:
        t_ns: Timestamp in nanoseconds
        video_path: Path to video file
        
    Returns:
        True if timestamp is valid
    """
    info = get_video_info(video_path)
    if not info:
        return False
    
    t_sec = t_ns / 1_000_000_000
    return 0 <= t_sec <= info["duration_sec"]
