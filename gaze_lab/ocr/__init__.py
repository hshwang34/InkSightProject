"""
OCR module for extracting text from gaze snapshots.

This module provides optical character recognition capabilities for GazeLab,
allowing extraction of text from video frames at specific timestamps or
areas of interest.
"""

from .engine import recognize_text
from .preprocess import prep_for_ocr

__all__ = ["recognize_text", "prep_for_ocr"]
