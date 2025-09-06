"""
Gaze data filtering and preprocessing.

Provides functions for basic denoising, blink removal, and outlier detection
in gaze data.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

from ..logging_setup import get_logger

logger = get_logger(__name__)


def filter_gaze_data(
    df: pd.DataFrame,
    confidence_threshold: float = 0.5,
    velocity_threshold_px_s: float = 100.0,
    median_filter_window: int = 5,
    remove_outliers: bool = True,
    outlier_std_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Apply comprehensive filtering to gaze data.
    
    Args:
        df: DataFrame with gaze data (columns: t_ns, gx_px, gy_px, frame_w, frame_h)
        confidence_threshold: Minimum confidence threshold
        velocity_threshold_px_s: Maximum velocity threshold in pixels per second
        median_filter_window: Window size for median filtering
        remove_outliers: Whether to remove statistical outliers
        outlier_std_threshold: Standard deviation threshold for outlier detection
        
    Returns:
        Filtered DataFrame
    """
    if len(df) == 0:
        return df
    
    original_count = len(df)
    filtered_df = df.copy()
    
    logger.info(f"Starting filtering with {original_count} samples")
    
    # 1. Confidence filtering
    if 'confidence' in filtered_df.columns:
        confidence_mask = filtered_df['confidence'] >= confidence_threshold
        filtered_df = filtered_df[confidence_mask]
        logger.info(f"Confidence filtering: {len(filtered_df)} samples remaining")
    
    # 2. Velocity filtering
    if len(filtered_df) > 1:
        filtered_df = _filter_by_velocity(filtered_df, velocity_threshold_px_s)
        logger.info(f"Velocity filtering: {len(filtered_df)} samples remaining")
    
    # 3. Median filtering
    if len(filtered_df) > median_filter_window:
        filtered_df = _apply_median_filter(filtered_df, median_filter_window)
        logger.info(f"Median filtering: {len(filtered_df)} samples remaining")
    
    # 4. Outlier removal
    if remove_outliers and len(filtered_df) > 10:
        filtered_df = _remove_outliers(filtered_df, outlier_std_threshold)
        logger.info(f"Outlier removal: {len(filtered_df)} samples remaining")
    
    # 5. Final validation
    filtered_df = _validate_coordinates(filtered_df)
    
    final_count = len(filtered_df)
    removed_count = original_count - final_count
    
    logger.info(f"Filtering complete: {removed_count} samples removed ({removed_count/original_count*100:.1f}%)")
    
    return filtered_df


def _filter_by_velocity(df: pd.DataFrame, velocity_threshold_px_s: float) -> pd.DataFrame:
    """Filter samples based on velocity threshold."""
    if len(df) < 2:
        return df
    
    # Calculate velocities
    dt = np.diff(df['t_ns'].values) / 1_000_000_000  # Convert to seconds
    dx = np.diff(df['gx_px'].values)
    dy = np.diff(df['gy_px'].values)
    
    # Calculate velocity in pixels per second
    velocities = np.sqrt(dx**2 + dy**2) / dt
    
    # Create mask for valid velocities (keep first sample and valid subsequent samples)
    valid_velocity = np.concatenate([[True], velocities <= velocity_threshold_px_s])
    
    return df[valid_velocity].reset_index(drop=True)


def _apply_median_filter(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Apply median filtering to gaze coordinates."""
    if len(df) < window_size:
        return df
    
    filtered_df = df.copy()
    
    # Apply median filter to x and y coordinates
    filtered_df['gx_px'] = ndimage.median_filter(df['gx_px'].values, size=window_size)
    filtered_df['gy_px'] = ndimage.median_filter(df['gy_px'].values, size=window_size)
    
    return filtered_df


def _remove_outliers(df: pd.DataFrame, std_threshold: float) -> pd.DataFrame:
    """Remove statistical outliers from gaze data."""
    if len(df) < 10:
        return df
    
    # Calculate z-scores for x and y coordinates
    x_mean = df['gx_px'].mean()
    x_std = df['gx_px'].std()
    y_mean = df['gy_px'].mean()
    y_std = df['gy_px'].std()
    
    if x_std == 0 or y_std == 0:
        return df
    
    x_z_scores = np.abs((df['gx_px'] - x_mean) / x_std)
    y_z_scores = np.abs((df['gy_px'] - y_mean) / y_std)
    
    # Keep samples within threshold
    outlier_mask = (x_z_scores <= std_threshold) & (y_z_scores <= std_threshold)
    
    return df[outlier_mask].reset_index(drop=True)


def _validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Validate coordinate bounds and remove invalid samples."""
    if len(df) == 0:
        return df
    
    # Check coordinate bounds
    valid_x = (df['gx_px'] >= 0) & (df['gx_px'] <= df['frame_w'])
    valid_y = (df['gy_px'] >= 0) & (df['gy_px'] <= df['frame_h'])
    
    # Check for NaN values
    valid_coords = ~(df['gx_px'].isna() | df['gy_px'].isna())
    
    # Combine all validity checks
    valid_mask = valid_x & valid_y & valid_coords
    
    return df[valid_mask].reset_index(drop=True)


def remove_blinks(
    df: pd.DataFrame,
    blink_duration_ms: float = 100.0,
    gap_threshold_ms: float = 50.0,
) -> pd.DataFrame:
    """
    Remove blink periods from gaze data.
    
    Args:
        df: DataFrame with gaze data
        blink_duration_ms: Maximum duration for a blink
        gap_threshold_ms: Minimum gap between samples to consider a blink
        
    Returns:
        DataFrame with blink periods removed
    """
    if len(df) < 2:
        return df
    
    # Calculate time gaps between consecutive samples
    dt = np.diff(df['t_ns'].values) / 1_000_000  # Convert to milliseconds
    
    # Find gaps larger than threshold
    large_gaps = dt > gap_threshold_ms
    
    # Find start and end indices of gaps
    gap_starts = np.where(large_gaps)[0]
    gap_ends = gap_starts + 1
    
    # Filter out gaps that are too long (likely blinks)
    valid_indices = []
    current_idx = 0
    
    for gap_start, gap_end in zip(gap_starts, gap_ends):
        gap_duration = dt[gap_start]
        
        if gap_duration <= blink_duration_ms:
            # Keep samples up to gap start
            valid_indices.extend(range(current_idx, gap_start + 1))
            current_idx = gap_end
    
    # Add remaining samples
    valid_indices.extend(range(current_idx, len(df)))
    
    return df.iloc[valid_indices].reset_index(drop=True)


def smooth_gaze_trajectory(
    df: pd.DataFrame,
    smoothing_factor: float = 0.1,
) -> pd.DataFrame:
    """
    Apply smoothing to gaze trajectory.
    
    Args:
        df: DataFrame with gaze data
        smoothing_factor: Smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing)
        
    Returns:
        DataFrame with smoothed coordinates
    """
    if len(df) < 3:
        return df
    
    smoothed_df = df.copy()
    
    # Apply exponential smoothing
    alpha = smoothing_factor
    
    # Smooth x coordinates
    x_values = df['gx_px'].values
    x_smoothed = np.zeros_like(x_values)
    x_smoothed[0] = x_values[0]
    
    for i in range(1, len(x_values)):
        x_smoothed[i] = alpha * x_values[i] + (1 - alpha) * x_smoothed[i - 1]
    
    # Smooth y coordinates
    y_values = df['gy_px'].values
    y_smoothed = np.zeros_like(y_values)
    y_smoothed[0] = y_values[0]
    
    for i in range(1, len(y_values)):
        y_smoothed[i] = alpha * y_values[i] + (1 - alpha) * y_smoothed[i - 1]
    
    smoothed_df['gx_px'] = x_smoothed
    smoothed_df['gy_px'] = y_smoothed
    
    return smoothed_df


def interpolate_missing_samples(
    df: pd.DataFrame,
    max_gap_ms: float = 100.0,
) -> pd.DataFrame:
    """
    Interpolate missing samples in gaze data.
    
    Args:
        df: DataFrame with gaze data
        max_gap_ms: Maximum gap to interpolate
        
    Returns:
        DataFrame with interpolated samples
    """
    if len(df) < 2:
        return df
    
    # Calculate time gaps
    dt = np.diff(df['t_ns'].values) / 1_000_000  # Convert to milliseconds
    
    # Find gaps to interpolate
    gaps_to_interpolate = []
    for i, gap_duration in enumerate(dt):
        if gap_duration > 16.67 and gap_duration <= max_gap_ms:  # More than one frame but within threshold
            gaps_to_interpolate.append((i, gap_duration))
    
    if not gaps_to_interpolate:
        return df
    
    # Create interpolated data
    interpolated_rows = []
    
    for i, (gap_idx, gap_duration) in enumerate(gaps_to_interpolate):
        start_idx = gap_idx
        end_idx = gap_idx + 1
        
        start_time = df.iloc[start_idx]['t_ns']
        end_time = df.iloc[end_idx]['t_ns']
        
        start_x = df.iloc[start_idx]['gx_px']
        start_y = df.iloc[start_idx]['gy_px']
        end_x = df.iloc[end_idx]['gx_px']
        end_y = df.iloc[end_idx]['gy_px']
        
        # Calculate number of samples to interpolate
        n_samples = int(gap_duration / 16.67)  # Assume 60 FPS
        
        if n_samples > 1:
            # Create time points
            time_points = np.linspace(start_time, end_time, n_samples + 1)[1:-1]
            
            # Interpolate coordinates
            x_points = np.linspace(start_x, end_x, n_samples + 1)[1:-1]
            y_points = np.linspace(start_y, end_y, n_samples + 1)[1:-1]
            
            # Create interpolated rows
            for t, x, y in zip(time_points, x_points, y_points):
                interpolated_row = df.iloc[start_idx].copy()
                interpolated_row['t_ns'] = int(t)
                interpolated_row['gx_px'] = x
                interpolated_row['gy_px'] = y
                interpolated_rows.append(interpolated_row)
    
    # Combine original and interpolated data
    if interpolated_rows:
        interpolated_df = pd.DataFrame(interpolated_rows)
        combined_df = pd.concat([df, interpolated_df], ignore_index=True)
        combined_df = combined_df.sort_values('t_ns').reset_index(drop=True)
        
        logger.info(f"Interpolated {len(interpolated_rows)} missing samples")
        return combined_df
    
    return df


def get_filtering_statistics(
    original_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
) -> dict:
    """
    Get statistics about the filtering process.
    
    Args:
        original_df: Original DataFrame
        filtered_df: Filtered DataFrame
        
    Returns:
        Dictionary with filtering statistics
    """
    original_count = len(original_df)
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    
    stats = {
        "original_samples": original_count,
        "filtered_samples": filtered_count,
        "removed_samples": removed_count,
        "removal_percentage": (removed_count / original_count * 100) if original_count > 0 else 0,
    }
    
    if original_count > 0 and filtered_count > 0:
        # Time range statistics
        original_duration = (original_df['t_ns'].max() - original_df['t_ns'].min()) / 1_000_000_000
        filtered_duration = (filtered_df['t_ns'].max() - filtered_df['t_ns'].min()) / 1_000_000_000
        
        stats.update({
            "original_duration_s": original_duration,
            "filtered_duration_s": filtered_duration,
            "duration_reduction_s": original_duration - filtered_duration,
        })
        
        # Sample rate statistics
        original_sample_rate = original_count / original_duration if original_duration > 0 else 0
        filtered_sample_rate = filtered_count / filtered_duration if filtered_duration > 0 else 0
        
        stats.update({
            "original_sample_rate_hz": original_sample_rate,
            "filtered_sample_rate_hz": filtered_sample_rate,
        })
    
    return stats