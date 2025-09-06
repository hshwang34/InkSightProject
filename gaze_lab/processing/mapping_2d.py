"""
2D coordinate mapping and homography computation.

Provides utilities for mapping gaze coordinates between different coordinate systems
and computing homographies for reference image mapping.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from ..logging_setup import get_logger

logger = get_logger(__name__)


def compute_homography(
    world_frame: np.ndarray,
    reference_image: np.ndarray,
    method: str = "orb",
    max_features: int = 1000,
    match_ratio: float = 0.75,
) -> Optional[np.ndarray]:
    """
    Compute homography matrix between world frame and reference image.
    
    Args:
        world_frame: World camera frame
        reference_image: Reference image
        method: Feature detection method ("orb", "sift")
        max_features: Maximum number of features to detect
        match_ratio: Ratio for feature matching
        
    Returns:
        Homography matrix (3x3) or None if computation fails
    """
    logger.info(f"Computing homography using {method} method")
    
    # Convert to grayscale if needed
    if len(world_frame.shape) == 3:
        world_gray = cv2.cvtColor(world_frame, cv2.COLOR_BGR2GRAY)
    else:
        world_gray = world_frame
    
    if len(reference_image.shape) == 3:
        ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = reference_image
    
    # Detect features and compute descriptors
    if method == "orb":
        detector = cv2.ORB_create(nfeatures=max_features)
    elif method == "sift":
        try:
            detector = cv2.SIFT_create(nfeatures=max_features)
        except AttributeError:
            logger.warning("SIFT not available, falling back to ORB")
            detector = cv2.ORB_create(nfeatures=max_features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Detect keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(world_gray, None)
    kp2, des2 = detector.detectAndCompute(ref_gray, None)
    
    if des1 is None or des2 is None:
        logger.error("Failed to detect features")
        return None
    
    logger.info(f"Detected {len(kp1)} features in world frame, {len(kp2)} in reference image")
    
    # Match features
    if method == "orb":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Filter matches by distance
    good_matches = matches[:int(len(matches) * match_ratio)]
    
    if len(good_matches) < 4:
        logger.error(f"Not enough good matches: {len(good_matches)}")
        return None
    
    logger.info(f"Found {len(good_matches)} good matches")
    
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if homography is None:
        logger.error("Failed to compute homography")
        return None
    
    # Count inliers
    inliers = np.sum(mask)
    logger.info(f"Homography computed with {inliers} inliers")
    
    return homography


def map_gaze_to_reference(
    gaze_df: pd.DataFrame,
    homography: np.ndarray,
    reference_size: Tuple[int, int],
) -> pd.DataFrame:
    """
    Map gaze coordinates to reference image using homography.
    
    Args:
        gaze_df: DataFrame with gaze data
        homography: Homography matrix (3x3)
        reference_size: Reference image size (width, height)
        
    Returns:
        DataFrame with mapped gaze coordinates
    """
    if len(gaze_df) == 0:
        return pd.DataFrame()
    
    logger.info(f"Mapping {len(gaze_df)} gaze points to reference image")
    
    # Extract gaze coordinates
    gaze_points = gaze_df[['gx_px', 'gy_px']].values.astype(np.float32)
    
    # Apply homography transformation
    gaze_points_homogeneous = np.column_stack([gaze_points, np.ones(len(gaze_points))])
    mapped_points_homogeneous = homography @ gaze_points_homogeneous.T
    mapped_points = mapped_points_homogeneous[:2] / mapped_points_homogeneous[2]
    mapped_points = mapped_points.T
    
    # Create mapped DataFrame
    mapped_df = gaze_df.copy()
    mapped_df['rx_px'] = mapped_points[:, 0]
    mapped_df['ry_px'] = mapped_points[:, 1]
    mapped_df['ref_w'] = reference_size[0]
    mapped_df['ref_h'] = reference_size[1]
    
    # Filter out points outside reference image bounds
    valid_mask = (
        (mapped_df['rx_px'] >= 0) & 
        (mapped_df['rx_px'] < reference_size[0]) &
        (mapped_df['ry_px'] >= 0) & 
        (mapped_df['ry_px'] < reference_size[1])
    )
    
    mapped_df = mapped_df[valid_mask].reset_index(drop=True)
    
    logger.info(f"Mapped {len(mapped_df)} valid gaze points to reference image")
    
    return mapped_df


def map_gaze_from_video_to_reference(
    gaze_df: pd.DataFrame,
    world_video_path: Union[str, Path],
    reference_image_path: Union[str, Path],
    frame_index: int = 0,
    method: str = "orb",
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Map gaze coordinates from video to reference image.
    
    Args:
        gaze_df: DataFrame with gaze data
        world_video_path: Path to world video
        reference_image_path: Path to reference image
        frame_index: Frame index to use for homography computation
        method: Feature detection method
        
    Returns:
        Tuple of (mapped_gaze_df, homography_matrix)
    """
    world_video_path = Path(world_video_path)
    reference_image_path = Path(reference_image_path)
    
    if not world_video_path.exists():
        raise FileNotFoundError(f"World video not found: {world_video_path}")
    
    if not reference_image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
    
    # Load reference image
    reference_image = cv2.imread(str(reference_image_path))
    if reference_image is None:
        raise ValueError(f"Failed to load reference image: {reference_image_path}")
    
    # Load world frame
    cap = cv2.VideoCapture(str(world_video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open world video: {world_video_path}")
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, world_frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read frame {frame_index} from video")
    
    # Compute homography
    homography = compute_homography(world_frame, reference_image, method=method)
    
    if homography is None:
        logger.error("Failed to compute homography")
        return pd.DataFrame(), None
    
    # Map gaze coordinates
    reference_size = (reference_image.shape[1], reference_image.shape[0])
    mapped_df = map_gaze_to_reference(gaze_df, homography, reference_size)
    
    return mapped_df, homography
