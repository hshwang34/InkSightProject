"""
GazeLab: Production-ready toolkit for Pupil Labs eye-tracking data analysis.

This package provides comprehensive functionality for:
- Loading and processing Pupil Labs export data
- Real-time data acquisition (with optional hardware support)
- Gaze overlay generation on video
- AOI (Area of Interest) analysis and reporting
- Heatmap generation
- 2D coordinate mapping and homography
"""

__version__ = "1.0.0"
__author__ = "GazeLab Team"

# Core imports
from .config import Config
from .logging_setup import setup_logging

# I/O modules
from .io.cloud_loader import load_gaze
from .io.player_layout import discover_session_files

# Processing modules
from .processing.filters import filter_gaze_data
from .processing.fixations_ivt import detect_fixations_ivt
from .processing.aoi import AOIAnalyzer
from .processing.mapping_2d import compute_homography, map_gaze_to_reference

# Visualization modules
from .viz.overlay import create_gaze_overlay
from .viz.heatmap import create_heatmap

__all__ = [
    # Core
    "Config",
    "setup_logging",
    # I/O
    "load_gaze",
    "discover_session_files",
    # Processing
    "filter_gaze_data",
    "detect_fixations_ivt",
    "AOIAnalyzer",
    "compute_homography",
    "map_gaze_to_reference",
    # Visualization
    "create_gaze_overlay",
    "create_heatmap",
]