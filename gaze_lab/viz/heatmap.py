"""
Gaze heatmap visualization.

Provides comprehensive heatmap generation capabilities for gaze data
with configurable appearance and statistical analysis.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import gaussian_kde

from ..logging_setup import get_logger

logger = get_logger(__name__)


def create_heatmap(
    gaze_df: pd.DataFrame,
    output_path: Union[str, Path],
    background_image: Optional[np.ndarray] = None,
    background_mode: str = "world",
    background_frame_index: int = 0,
    world_video_path: Optional[Union[str, Path]] = None,
    bandwidth: float = 20.0,
    grid_size: int = 100,
    colormap: str = "hot",
    alpha: float = 0.7,
    dpi: int = 300,
) -> None:
    """
    Create gaze heatmap visualization.
    
    Args:
        gaze_df: DataFrame with gaze data
        output_path: Output image path
        background_image: Optional background image
        background_mode: Background mode ("world", "reference", "none")
        background_frame_index: Frame index for world video background
        world_video_path: Path to world video for background
        bandwidth: KDE bandwidth parameter
        grid_size: Grid resolution for heatmap
        colormap: Matplotlib colormap name
        alpha: Transparency of heatmap overlay
        dpi: Output image DPI
    """
    if len(gaze_df) == 0:
        raise ValueError("No gaze data provided")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating heatmap with {len(gaze_df)} gaze points")
    
    # Determine image dimensions
    if background_mode == "world" and world_video_path:
        width, height = _get_video_dimensions(world_video_path)
    elif background_image is not None:
        height, width = background_image.shape[:2]
    else:
        # Use frame dimensions from gaze data
        width = int(gaze_df['frame_w'].iloc[0])
        height = int(gaze_df['frame_h'].iloc[0])
    
    # Create heatmap
    heatmap = _compute_heatmap(gaze_df, width, height, bandwidth, grid_size)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    
    # Set background
    if background_mode == "world" and world_video_path:
        background = _load_video_frame(world_video_path, background_frame_index)
        ax.imshow(background)
    elif background_image is not None:
        ax.imshow(background_image)
    else:
        # Create white background
        background = np.ones((height, width, 3), dtype=np.uint8) * 255
        ax.imshow(background)
    
    # Overlay heatmap
    im = ax.imshow(heatmap, cmap=colormap, alpha=alpha, origin='upper')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Gaze Density', rotation=270, labelpad=20)
    
    # Configure plot
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()
    
    logger.info(f"Created heatmap: {output_path}")


def _compute_heatmap(
    gaze_df: pd.DataFrame,
    width: int,
    height: int,
    bandwidth: float,
    grid_size: int,
) -> np.ndarray:
    """Compute heatmap from gaze data."""
    # Extract gaze coordinates
    x_coords = gaze_df['gx_px'].values
    y_coords = gaze_df['gy_px'].values
    
    # Filter valid coordinates
    valid_mask = (
        (x_coords >= 0) & (x_coords < width) &
        (y_coords >= 0) & (y_coords < height)
    )
    
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    
    if len(x_coords) == 0:
        return np.zeros((height, width))
    
    # Create grid
    x_grid = np.linspace(0, width, grid_size)
    y_grid = np.linspace(0, height, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute KDE
    try:
        kde = gaussian_kde(np.vstack([x_coords, y_coords]), bw_method=bandwidth/100.0)
        positions = np.vstack([X.ravel(), Y.ravel()])
        density = kde(positions).reshape(X.shape)
    except Exception as e:
        logger.warning(f"KDE computation failed: {e}, using histogram method")
        density = _compute_histogram_heatmap(x_coords, y_coords, width, height, grid_size)
    
    # Normalize density
    if density.max() > 0:
        density = density / density.max()
    
    # Resize to target dimensions
    if grid_size != width or grid_size != height:
        density = cv2.resize(density, (width, height), interpolation=cv2.INTER_CUBIC)
    
    return density


def _compute_histogram_heatmap(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    width: int,
    height: int,
    grid_size: int,
) -> np.ndarray:
    """Compute heatmap using histogram method."""
    # Create histogram
    hist, x_edges, y_edges = np.histogram2d(
        x_coords, y_coords,
        bins=[grid_size, grid_size],
        range=[[0, width], [0, height]]
    )
    
    # Apply Gaussian smoothing
    hist = ndimage.gaussian_filter(hist, sigma=2.0)
    
    return hist.T


def _get_video_dimensions(video_path: Union[str, Path]) -> Tuple[int, int]:
    """Get video dimensions."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return width, height


def _load_video_frame(video_path: Union[str, Path], frame_index: int) -> np.ndarray:
    """Load frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read frame {frame_index} from video")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame


def create_heatmap_from_mapped_gaze(
    mapped_gaze_df: pd.DataFrame,
    reference_image_path: Union[str, Path],
    output_path: Union[str, Path],
    bandwidth: float = 20.0,
    grid_size: int = 100,
    colormap: str = "hot",
    alpha: float = 0.7,
    dpi: int = 300,
) -> None:
    """
    Create heatmap from mapped gaze data on reference image.
    
    Args:
        mapped_gaze_df: DataFrame with mapped gaze data
        reference_image_path: Path to reference image
        output_path: Output image path
        bandwidth: KDE bandwidth parameter
        grid_size: Grid resolution for heatmap
        colormap: Matplotlib colormap name
        alpha: Transparency of heatmap overlay
        dpi: Output image DPI
    """
    # Load reference image
    reference_image = cv2.imread(str(reference_image_path))
    if reference_image is None:
        raise ValueError(f"Failed to load reference image: {reference_image_path}")
    
    # Convert BGR to RGB
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    
    # Create heatmap
    create_heatmap(
        gaze_df=mapped_gaze_df,
        output_path=output_path,
        background_image=reference_image,
        background_mode="reference",
        bandwidth=bandwidth,
        grid_size=grid_size,
        colormap=colormap,
        alpha=alpha,
        dpi=dpi,
    )


def get_heatmap_statistics(gaze_df: pd.DataFrame) -> dict:
    """Get statistics about gaze data for heatmap."""
    if len(gaze_df) == 0:
        return {"gaze_points": 0}
    
    stats = {
        "gaze_points": len(gaze_df),
        "x_range": (gaze_df['gx_px'].min(), gaze_df['gx_px'].max()),
        "y_range": (gaze_df['gy_px'].min(), gaze_df['gy_px'].max()),
        "x_mean": gaze_df['gx_px'].mean(),
        "y_mean": gaze_df['gy_px'].mean(),
        "x_std": gaze_df['gx_px'].std(),
        "y_std": gaze_df['gy_px'].std(),
    }
    
    # Calculate gaze spread
    x_center = stats["x_mean"]
    y_center = stats["y_mean"]
    distances = np.sqrt((gaze_df['gx_px'] - x_center)**2 + (gaze_df['gy_px'] - y_center)**2)
    
    stats.update({
        "mean_distance_from_center": distances.mean(),
        "max_distance_from_center": distances.max(),
        "std_distance_from_center": distances.std(),
    })
    
    return stats