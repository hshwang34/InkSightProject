"""
Snapshot extraction module for GazeLab.

This module provides functionality to extract frames from video files
and crop regions of interest for further analysis.
"""

from .extractor import extract_frame_at_ts, crop_aoi, warp_to_reference

__all__ = ["extract_frame_at_ts", "crop_aoi", "warp_to_reference"]
