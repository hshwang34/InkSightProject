"""
Cloud loader for Pupil Labs export data.

Handles loading and normalization of CSV exports from Pupil Cloud and Player
"Raw Data Exporter" with canonical schema mapping.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

logger = get_logger(__name__)

# Canonical column names
CANONICAL_COLUMNS = {
    "t_ns": "timestamp in nanoseconds",
    "gx_px": "gaze x coordinate in pixels",
    "gy_px": "gaze y coordinate in pixels", 
    "frame_w": "frame width in pixels",
    "frame_h": "frame height in pixels",
}

# Common field name mappings from various Pupil export formats
FIELD_MAPPINGS = {
    # Timestamp variations
    "t_ns": [
        "timestamp [ns]", "timestamp_ns", "timestamp", "t_ns", "time_ns",
        "gaze_timestamp", "gaze_timestamp_ns", "t"
    ],
    # Gaze X variations
    "gx_px": [
        "gaze x [px]", "gaze_x", "gaze_x_px", "gx_px", "gx", "x",
        "norm_pos_x", "gaze_point_3d_x", "gaze_point_3d.x"
    ],
    # Gaze Y variations
    "gy_px": [
        "gaze y [px]", "gaze_y", "gaze_y_px", "gy_px", "gy", "y", 
        "norm_pos_y", "gaze_point_3d_y", "gaze_point_3d.y"
    ],
    # Frame width variations
    "frame_w": [
        "frame_width", "frame_w", "width", "w", "world_frame_width",
        "world_camera_frame_width", "frame_size_x"
    ],
    # Frame height variations
    "frame_h": [
        "frame_height", "frame_h", "height", "h", "world_frame_height",
        "world_camera_frame_height", "frame_size_y"
    ],
}


def detect_field_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect field mappings from DataFrame columns to canonical names.
    
    Args:
        df: Input DataFrame with gaze data
        
    Returns:
        Dictionary mapping canonical names to actual column names
    """
    mapping = {}
    
    for canonical_name, possible_names in FIELD_MAPPINGS.items():
        # Try exact matches first
        for possible_name in possible_names:
            if possible_name in df.columns:
                mapping[canonical_name] = possible_name
                break
        
        # Try case-insensitive matches
        if canonical_name not in mapping:
            df_cols_lower = {col.lower(): col for col in df.columns}
            for possible_name in possible_names:
                if possible_name.lower() in df_cols_lower:
                    mapping[canonical_name] = df_cols_lower[possible_name.lower()]
                    break
        
        # Try partial matches (for columns with additional text)
        if canonical_name not in mapping:
            for col in df.columns:
                col_lower = col.lower()
                for possible_name in possible_names:
                    if possible_name.lower() in col_lower:
                        mapping[canonical_name] = col
                        break
                if canonical_name in mapping:
                    break
    
    return mapping


def normalize_timestamps(timestamps: pd.Series) -> pd.Series:
    """
    Normalize timestamps to nanoseconds.
    
    Args:
        timestamps: Series of timestamps
        
    Returns:
        Series of timestamps in nanoseconds
    """
    # Detect timestamp format and scale
    sample_ts = timestamps.dropna().iloc[0] if len(timestamps.dropna()) > 0 else 0
    
    # If already in nanoseconds (large values)
    if sample_ts > 1e15:  # Nanoseconds since epoch
        return timestamps
    
    # If in microseconds
    elif sample_ts > 1e12:  # Microseconds since epoch
        return timestamps * 1000
    
    # If in milliseconds
    elif sample_ts > 1e9:  # Milliseconds since epoch
        return timestamps * 1_000_000
    
    # If in seconds
    elif sample_ts > 1e6:  # Seconds since epoch
        return timestamps * 1_000_000_000
    
    # If relative timestamps (small values)
    else:
        # Assume seconds and convert to nanoseconds
        return timestamps * 1_000_000_000


def normalize_coordinates(coords: pd.Series, frame_size: Optional[int] = None) -> pd.Series:
    """
    Normalize coordinates to pixel values.
    
    Args:
        coords: Series of coordinates
        frame_size: Frame size for normalization (width or height)
        
    Returns:
        Series of coordinates in pixels
    """
    if coords.isna().all():
        return coords
    
    # Check if coordinates are already in pixels (large values)
    sample_coord = coords.dropna().iloc[0]
    
    # If values are between 0 and 1, they're normalized coordinates
    if 0 <= sample_coord <= 1:
        if frame_size is not None:
            return coords * frame_size
        else:
            # Default to 1920x1080 if frame size unknown
            return coords * 1920
    
    # If values are already in pixels, return as-is
    return coords


def load_gaze(
    path: Union[str, Path],
    format_type: str = "auto",
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load gaze data from CSV file and normalize to canonical schema.
    
    Args:
        path: Path to CSV file or directory containing gaze data
        format_type: Export format type ("auto", "pupil_cloud", "player", "core")
        frame_width: Override frame width if not detected
        frame_height: Override frame height if not detected
        
    Returns:
        DataFrame with canonical columns: t_ns, gx_px, gy_px, frame_w, frame_h
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns cannot be found or data is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Load CSV file
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV file {path}: {e}")
    
    if len(df) == 0:
        raise ValueError(f"CSV file is empty: {path}")
    
    # Detect field mappings
    mapping = detect_field_mapping(df)
    logger.info(f"Detected field mappings: {mapping}")
    
    # Check for required columns
    required_cols = ["t_ns", "gx_px", "gy_px"]
    missing_cols = [col for col in required_cols if col not in mapping]
    
    if missing_cols:
        available_cols = list(df.columns)
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {available_cols}. "
            f"Detected mappings: {mapping}"
        )
    
    # Create normalized DataFrame
    normalized_df = pd.DataFrame()
    
    # Copy and normalize timestamp
    if "t_ns" in mapping:
        normalized_df["t_ns"] = normalize_timestamps(df[mapping["t_ns"]])
    
    # Copy and normalize gaze coordinates
    if "gx_px" in mapping:
        normalized_df["gx_px"] = normalize_coordinates(
            df[mapping["gx_px"]], frame_width
        )
    
    if "gy_px" in mapping:
        normalized_df["gy_px"] = normalize_coordinates(
            df[mapping["gy_px"]], frame_height
        )
    
    # Handle frame dimensions
    if "frame_w" in mapping:
        normalized_df["frame_w"] = df[mapping["frame_w"]]
    elif frame_width is not None:
        normalized_df["frame_w"] = frame_width
    else:
        # Try to infer from data
        if "gx_px" in normalized_df.columns:
            max_x = normalized_df["gx_px"].max()
            if max_x > 0 and max_x <= 1:
                # Normalized coordinates, assume 1920
                normalized_df["frame_w"] = 1920
            else:
                # Pixel coordinates, use max value
                normalized_df["frame_w"] = int(max_x) + 100  # Add some padding
        else:
            normalized_df["frame_w"] = 1920  # Default
    
    if "frame_h" in mapping:
        normalized_df["frame_h"] = df[mapping["frame_h"]]
    elif frame_height is not None:
        normalized_df["frame_h"] = frame_height
    else:
        # Try to infer from data
        if "gy_px" in normalized_df.columns:
            max_y = normalized_df["gy_px"].max()
            if max_y > 0 and max_y <= 1:
                # Normalized coordinates, assume 1080
                normalized_df["frame_h"] = 1080
            else:
                # Pixel coordinates, use max value
                normalized_df["frame_h"] = int(max_y) + 100  # Add some padding
        else:
            normalized_df["frame_h"] = 1080  # Default
    
    # Handle missing/NaN values
    original_count = len(normalized_df)
    
    # Remove rows with invalid timestamps
    normalized_df = normalized_df.dropna(subset=["t_ns"])
    
    # Remove rows with invalid gaze coordinates
    normalized_df = normalized_df.dropna(subset=["gx_px", "gy_px"])
    
    # Remove rows with coordinates outside frame bounds
    if "frame_w" in normalized_df.columns and "frame_h" in normalized_df.columns:
        mask = (
            (normalized_df["gx_px"] >= 0) & 
            (normalized_df["gx_px"] <= normalized_df["frame_w"]) &
            (normalized_df["gy_px"] >= 0) & 
            (normalized_df["gy_px"] <= normalized_df["frame_h"])
        )
        normalized_df = normalized_df[mask]
    
    filtered_count = len(normalized_df)
    if filtered_count < original_count:
        logger.info(f"Filtered {original_count - filtered_count} invalid rows")
    
    # Ensure monotonic timestamps
    normalized_df = normalized_df.sort_values("t_ns").reset_index(drop=True)
    
    # Validate final data
    if len(normalized_df) == 0:
        raise ValueError("No valid gaze data found after filtering")
    
    # Check for monotonic timestamps
    if not normalized_df["t_ns"].is_monotonic_increasing:
        logger.warning("Timestamps are not monotonically increasing")
    
    logger.info(f"Normalized gaze data: {len(normalized_df)} rows")
    logger.info(f"Time range: {normalized_df['t_ns'].min()} - {normalized_df['t_ns'].max()} ns")
    logger.info(f"Frame size: {normalized_df['frame_w'].iloc[0]}x{normalized_df['frame_h'].iloc[0]}")
    
    return normalized_df


def load_multiple_sessions(
    directory: Union[str, Path],
    pattern: str = "*.csv"
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple gaze data files from a directory.
    
    Args:
        directory: Directory containing CSV files
        pattern: File pattern to match
        
    Returns:
        Dictionary mapping filenames to DataFrames
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    csv_files = list(directory.glob(pattern))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory} matching pattern {pattern}")
    
    sessions = {}
    
    for csv_file in csv_files:
        try:
            df = load_gaze(csv_file)
            sessions[csv_file.name] = df
            logger.info(f"Loaded session: {csv_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")
    
    return sessions


def validate_gaze_data(df: pd.DataFrame) -> List[str]:
    """
    Validate gaze data and return list of issues.
    
    Args:
        df: DataFrame with gaze data
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Check required columns
    required_cols = ["t_ns", "gx_px", "gy_px", "frame_w", "frame_h"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check data types
    if "t_ns" in df.columns and not pd.api.types.is_numeric_dtype(df["t_ns"]):
        issues.append("Timestamp column 't_ns' is not numeric")
    
    if "gx_px" in df.columns and not pd.api.types.is_numeric_dtype(df["gx_px"]):
        issues.append("Gaze X column 'gx_px' is not numeric")
    
    if "gy_px" in df.columns and not pd.api.types.is_numeric_dtype(df["gy_px"]):
        issues.append("Gaze Y column 'gy_px' is not numeric")
    
    # Check for empty data
    if len(df) == 0:
        issues.append("DataFrame is empty")
    
    # Check for all NaN values
    if "t_ns" in df.columns and df["t_ns"].isna().all():
        issues.append("All timestamp values are NaN")
    
    if "gx_px" in df.columns and df["gx_px"].isna().all():
        issues.append("All gaze X values are NaN")
    
    if "gy_px" in df.columns and df["gy_px"].isna().all():
        issues.append("All gaze Y values are NaN")
    
    # Check coordinate bounds
    if "gx_px" in df.columns and "frame_w" in df.columns:
        invalid_x = (df["gx_px"] < 0) | (df["gx_px"] > df["frame_w"])
        if invalid_x.any():
            issues.append(f"{invalid_x.sum()} gaze X coordinates outside frame bounds")
    
    if "gy_px" in df.columns and "frame_h" in df.columns:
        invalid_y = (df["gy_px"] < 0) | (df["gy_px"] > df["frame_h"])
        if invalid_y.any():
            issues.append(f"{invalid_y.sum()} gaze Y coordinates outside frame bounds")
    
    return issues