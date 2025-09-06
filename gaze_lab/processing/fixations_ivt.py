"""
I-VT (Velocity-Threshold) fixation detection algorithm.

Implements the standard I-VT algorithm for detecting fixations from gaze data
with configurable parameters and comprehensive output.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

logger = get_logger(__name__)


def detect_fixations_ivt(
    df: pd.DataFrame,
    velocity_threshold_deg_s: float = 30.0,
    min_duration_ms: float = 50.0,
    max_duration_ms: float = 2000.0,
    px_per_degree: float = 30.0,
    min_samples: int = 3,
) -> pd.DataFrame:
    """
    Detect fixations using the I-VT (Velocity-Threshold) algorithm.
    
    Args:
        df: DataFrame with gaze data (columns: t_ns, gx_px, gy_px)
        velocity_threshold_deg_s: Velocity threshold in degrees per second
        min_duration_ms: Minimum fixation duration in milliseconds
        max_duration_ms: Maximum fixation duration in milliseconds
        px_per_degree: Pixels per degree of visual angle
        min_samples: Minimum number of samples for a fixation
        
    Returns:
        DataFrame with fixation data (columns: start_ns, end_ns, dur_ms, cx_px, cy_px, n_samples)
    """
    if len(df) < 2:
        return pd.DataFrame(columns=['start_ns', 'end_ns', 'dur_ms', 'cx_px', 'cy_px', 'n_samples'])
    
    logger.info(f"Starting I-VT fixation detection on {len(df)} samples")
    
    # Calculate velocities
    velocities = _calculate_velocities(df, px_per_degree)
    
    # Classify samples as fixations or saccades
    fixation_mask = velocities <= velocity_threshold_deg_s
    
    # Group consecutive fixation samples
    fixation_groups = _group_consecutive_samples(fixation_mask)
    
    # Convert groups to fixations
    fixations = []
    for group in fixation_groups:
        fixation = _create_fixation_from_group(
            df, group, min_duration_ms, max_duration_ms, min_samples
        )
        if fixation is not None:
            fixations.append(fixation)
    
    # Create DataFrame
    if fixations:
        fixation_df = pd.DataFrame(fixations)
        logger.info(f"Detected {len(fixation_df)} fixations")
    else:
        fixation_df = pd.DataFrame(columns=['start_ns', 'end_ns', 'dur_ms', 'cx_px', 'cy_px', 'n_samples'])
        logger.info("No fixations detected")
    
    return fixation_df


def _calculate_velocities(df: pd.DataFrame, px_per_degree: float) -> np.ndarray:
    """Calculate angular velocities between consecutive gaze points."""
    if len(df) < 2:
        return np.array([])
    
    # Calculate time differences in seconds
    dt = np.diff(df['t_ns'].values) / 1_000_000_000
    
    # Calculate spatial differences in pixels
    dx = np.diff(df['gx_px'].values)
    dy = np.diff(df['gy_px'].values)
    
    # Calculate angular distances
    angular_distances = np.sqrt(dx**2 + dy**2) / px_per_degree
    
    # Calculate velocities in degrees per second
    velocities = angular_distances / dt
    
    # Handle division by zero
    velocities = np.where(dt > 0, velocities, 0)
    
    # Add zero velocity for first sample
    velocities = np.concatenate([[0], velocities])
    
    return velocities


def _group_consecutive_samples(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Group consecutive True values in mask into ranges."""
    if len(mask) == 0:
        return []
    
    groups = []
    start_idx = None
    
    for i, is_fixation in enumerate(mask):
        if is_fixation and start_idx is None:
            start_idx = i
        elif not is_fixation and start_idx is not None:
            groups.append((start_idx, i - 1))
            start_idx = None
    
    # Handle case where fixation continues to end
    if start_idx is not None:
        groups.append((start_idx, len(mask) - 1))
    
    return groups


def _create_fixation_from_group(
    df: pd.DataFrame,
    group: Tuple[int, int],
    min_duration_ms: float,
    max_duration_ms: float,
    min_samples: int,
) -> Optional[dict]:
    """Create fixation data from a group of consecutive samples."""
    start_idx, end_idx = group
    n_samples = end_idx - start_idx + 1
    
    # Check minimum sample count
    if n_samples < min_samples:
        return None
    
    # Calculate duration
    start_time = df.iloc[start_idx]['t_ns']
    end_time = df.iloc[end_idx]['t_ns']
    duration_ms = (end_time - start_time) / 1_000_000
    
    # Check duration constraints
    if duration_ms < min_duration_ms or duration_ms > max_duration_ms:
        return None
    
    # Calculate fixation center
    fixation_samples = df.iloc[start_idx:end_idx + 1]
    center_x = fixation_samples['gx_px'].mean()
    center_y = fixation_samples['gy_px'].mean()
    
    return {
        'start_ns': start_time,
        'end_ns': end_time,
        'dur_ms': duration_ms,
        'cx_px': center_x,
        'cy_px': center_y,
        'n_samples': n_samples,
    }


def detect_saccades(
    df: pd.DataFrame,
    fixation_df: pd.DataFrame,
    min_amplitude_deg: float = 0.5,
    min_duration_ms: float = 10.0,
    px_per_degree: float = 30.0,
) -> pd.DataFrame:
    """
    Detect saccades from gaze data and fixations.
    
    Args:
        df: DataFrame with gaze data
        fixation_df: DataFrame with fixation data
        min_amplitude_deg: Minimum saccade amplitude in degrees
        min_duration_ms: Minimum saccade duration in milliseconds
        px_per_degree: Pixels per degree of visual angle
        
    Returns:
        DataFrame with saccade data
    """
    if len(fixation_df) < 2:
        return pd.DataFrame(columns=['start_ns', 'end_ns', 'dur_ms', 'amplitude_deg', 'velocity_deg_s'])
    
    saccades = []
    
    for i in range(len(fixation_df) - 1):
        current_fixation = fixation_df.iloc[i]
        next_fixation = fixation_df.iloc[i + 1]
        
        # Calculate saccade parameters
        start_time = current_fixation['end_ns']
        end_time = next_fixation['start_ns']
        duration_ms = (end_time - start_time) / 1_000_000
        
        # Calculate amplitude
        dx = next_fixation['cx_px'] - current_fixation['cx_px']
        dy = next_fixation['cy_px'] - current_fixation['cy_px']
        amplitude_px = np.sqrt(dx**2 + dy**2)
        amplitude_deg = amplitude_px / px_per_degree
        
        # Calculate velocity
        velocity_deg_s = amplitude_deg / (duration_ms / 1000) if duration_ms > 0 else 0
        
        # Check constraints
        if amplitude_deg >= min_amplitude_deg and duration_ms >= min_duration_ms:
            saccades.append({
                'start_ns': start_time,
                'end_ns': end_time,
                'dur_ms': duration_ms,
                'amplitude_deg': amplitude_deg,
                'velocity_deg_s': velocity_deg_s,
            })
    
    if saccades:
        saccade_df = pd.DataFrame(saccades)
        logger.info(f"Detected {len(saccade_df)} saccades")
    else:
        saccade_df = pd.DataFrame(columns=['start_ns', 'end_ns', 'dur_ms', 'amplitude_deg', 'velocity_deg_s'])
        logger.info("No saccades detected")
    
    return saccade_df


def get_fixation_statistics(fixation_df: pd.DataFrame) -> dict:
    """
    Calculate statistics for fixations.
    
    Args:
        fixation_df: DataFrame with fixation data
        
    Returns:
        Dictionary with fixation statistics
    """
    if len(fixation_df) == 0:
        return {
            "count": 0,
            "total_duration_ms": 0.0,
            "mean_duration_ms": 0.0,
            "std_duration_ms": 0.0,
            "median_duration_ms": 0.0,
            "min_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "mean_samples": 0.0,
            "std_samples": 0.0,
        }
    
    durations = fixation_df['dur_ms'].values
    sample_counts = fixation_df['n_samples'].values
    
    return {
        "count": len(fixation_df),
        "total_duration_ms": durations.sum(),
        "mean_duration_ms": durations.mean(),
        "std_duration_ms": durations.std(),
        "median_duration_ms": np.median(durations),
        "min_duration_ms": durations.min(),
        "max_duration_ms": durations.max(),
        "mean_samples": sample_counts.mean(),
        "std_samples": sample_counts.std(),
    }


def get_saccade_statistics(saccade_df: pd.DataFrame) -> dict:
    """
    Calculate statistics for saccades.
    
    Args:
        saccade_df: DataFrame with saccade data
        
    Returns:
        Dictionary with saccade statistics
    """
    if len(saccade_df) == 0:
        return {
            "count": 0,
            "mean_amplitude_deg": 0.0,
            "std_amplitude_deg": 0.0,
            "mean_velocity_deg_s": 0.0,
            "std_velocity_deg_s": 0.0,
            "mean_duration_ms": 0.0,
            "std_duration_ms": 0.0,
        }
    
    amplitudes = saccade_df['amplitude_deg'].values
    velocities = saccade_df['velocity_deg_s'].values
    durations = saccade_df['dur_ms'].values
    
    return {
        "count": len(saccade_df),
        "mean_amplitude_deg": amplitudes.mean(),
        "std_amplitude_deg": amplitudes.std(),
        "mean_velocity_deg_s": velocities.mean(),
        "std_velocity_deg_s": velocities.std(),
        "mean_duration_ms": durations.mean(),
        "std_duration_ms": durations.std(),
    }


def filter_fixations(
    fixation_df: pd.DataFrame,
    min_duration_ms: Optional[float] = None,
    max_duration_ms: Optional[float] = None,
    min_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filter fixations based on criteria.
    
    Args:
        fixation_df: DataFrame with fixation data
        min_duration_ms: Minimum duration threshold
        max_duration_ms: Maximum duration threshold
        min_samples: Minimum sample count threshold
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = fixation_df.copy()
    
    if min_duration_ms is not None:
        filtered_df = filtered_df[filtered_df['dur_ms'] >= min_duration_ms]
    
    if max_duration_ms is not None:
        filtered_df = filtered_df[filtered_df['dur_ms'] <= max_duration_ms]
    
    if min_samples is not None:
        filtered_df = filtered_df[filtered_df['n_samples'] >= min_samples]
    
    return filtered_df.reset_index(drop=True)


def export_fixations_to_csv(
    fixation_df: pd.DataFrame,
    output_path: str,
    include_statistics: bool = True,
) -> None:
    """
    Export fixations to CSV file.
    
    Args:
        fixation_df: DataFrame with fixation data
        output_path: Output file path
        include_statistics: Whether to include statistics in the file
    """
    fixation_df.to_csv(output_path, index=False)
    
    if include_statistics:
        stats = get_fixation_statistics(fixation_df)
        
        # Append statistics to file
        with open(output_path, 'a') as f:
            f.write('\n# Statistics\n')
            for key, value in stats.items():
                f.write(f'# {key}: {value}\n')
    
    logger.info(f"Exported {len(fixation_df)} fixations to {output_path}")


def validate_fixation_data(fixation_df: pd.DataFrame) -> List[str]:
    """
    Validate fixation data and return list of issues.
    
    Args:
        fixation_df: DataFrame with fixation data
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if len(fixation_df) == 0:
        return issues
    
    # Check required columns
    required_cols = ['start_ns', 'end_ns', 'dur_ms', 'cx_px', 'cy_px', 'n_samples']
    missing_cols = [col for col in required_cols if col not in fixation_df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for negative durations
    if 'dur_ms' in fixation_df.columns:
        negative_durations = fixation_df['dur_ms'] < 0
        if negative_durations.any():
            issues.append(f"{negative_durations.sum()} fixations have negative duration")
    
    # Check for zero durations
    if 'dur_ms' in fixation_df.columns:
        zero_durations = fixation_df['dur_ms'] == 0
        if zero_durations.any():
            issues.append(f"{zero_durations.sum()} fixations have zero duration")
    
    # Check for negative sample counts
    if 'n_samples' in fixation_df.columns:
        negative_samples = fixation_df['n_samples'] < 0
        if negative_samples.any():
            issues.append(f"{negative_samples.sum()} fixations have negative sample count")
    
    # Check for zero sample counts
    if 'n_samples' in fixation_df.columns:
        zero_samples = fixation_df['n_samples'] == 0
        if zero_samples.any():
            issues.append(f"{zero_samples.sum()} fixations have zero sample count")
    
    # Check timestamp consistency
    if 'start_ns' in fixation_df.columns and 'end_ns' in fixation_df.columns:
        inconsistent_timestamps = fixation_df['start_ns'] >= fixation_df['end_ns']
        if inconsistent_timestamps.any():
            issues.append(f"{inconsistent_timestamps.sum()} fixations have inconsistent timestamps")
    
    return issues