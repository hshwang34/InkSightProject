"""
Tests for I-VT fixation detection algorithm.
"""

import numpy as np
import pandas as pd
import pytest

from gaze_lab.processing.fixations_ivt import (
    detect_fixations_ivt,
    get_fixation_statistics,
    validate_fixation_data,
)


class TestFixationDetection:
    """Test fixation detection functionality."""
    
    def test_detect_fixations_simple(self):
        """Test basic fixation detection."""
        # Create synthetic gaze data with two fixations
        gaze_data = []
        
        # First fixation: 100ms at (100, 100)
        for i in range(10):
            gaze_data.append({
                't_ns': i * 10_000_000,  # 10ms intervals
                'gx_px': 100 + np.random.normal(0, 2),
                'gy_px': 100 + np.random.normal(0, 2),
                'frame_w': 1280,
                'frame_h': 720,
            })
        
        # Saccade: 50ms
        for i in range(5):
            gaze_data.append({
                't_ns': (10 + i) * 10_000_000,
                'gx_px': 100 + (i * 20),  # Moving right
                'gy_px': 100 + (i * 10),  # Moving down
                'frame_w': 1280,
                'frame_h': 720,
            })
        
        # Second fixation: 150ms at (200, 200)
        for i in range(15):
            gaze_data.append({
                't_ns': (15 + i) * 10_000_000,
                'gx_px': 200 + np.random.normal(0, 2),
                'gy_px': 200 + np.random.normal(0, 2),
                'frame_w': 1280,
                'frame_h': 720,
            })
        
        df = pd.DataFrame(gaze_data)
        
        # Detect fixations
        fixations = detect_fixations_ivt(
            df,
            velocity_threshold_deg_s=30.0,
            min_duration_ms=50.0,
            px_per_degree=30.0,
        )
        
        # Check results
        assert len(fixations) == 2
        
        # Check first fixation
        first_fixation = fixations.iloc[0]
        assert 90 <= first_fixation['dur_ms'] <= 110  # Allow some tolerance
        assert 95 <= first_fixation['cx_px'] <= 105
        assert 95 <= first_fixation['cy_px'] <= 105
        
        # Check second fixation
        second_fixation = fixations.iloc[1]
        assert 140 <= second_fixation['dur_ms'] <= 160
        assert 195 <= second_fixation['cx_px'] <= 205
        assert 195 <= second_fixation['cy_px'] <= 205
    
    def test_detect_fixations_empty_data(self):
        """Test fixation detection with empty data."""
        df = pd.DataFrame(columns=['t_ns', 'gx_px', 'gy_px'])
        fixations = detect_fixations_ivt(df)
        assert len(fixations) == 0
    
    def test_detect_fixations_single_sample(self):
        """Test fixation detection with single sample."""
        df = pd.DataFrame([{
            't_ns': 0,
            'gx_px': 100,
            'gy_px': 100,
            'frame_w': 1280,
            'frame_h': 720,
        }])
        fixations = detect_fixations_ivt(df)
        assert len(fixations) == 0  # Need at least 3 samples
    
    def test_fixation_statistics(self):
        """Test fixation statistics calculation."""
        fixations = pd.DataFrame([
            {'start_ns': 0, 'end_ns': 100_000_000, 'dur_ms': 100, 'cx_px': 100, 'cy_px': 100, 'n_samples': 10},
            {'start_ns': 200_000_000, 'end_ns': 350_000_000, 'dur_ms': 150, 'cx_px': 200, 'cy_px': 200, 'n_samples': 15},
        ])
        
        stats = get_fixation_statistics(fixations)
        
        assert stats['count'] == 2
        assert stats['total_duration_ms'] == 250
        assert stats['mean_duration_ms'] == 125
        assert stats['min_duration_ms'] == 100
        assert stats['max_duration_ms'] == 150
        assert stats['mean_samples'] == 12.5
    
    def test_validate_fixation_data(self):
        """Test fixation data validation."""
        # Valid fixations
        valid_fixations = pd.DataFrame([
            {'start_ns': 0, 'end_ns': 100_000_000, 'dur_ms': 100, 'cx_px': 100, 'cy_px': 100, 'n_samples': 10},
        ])
        issues = validate_fixation_data(valid_fixations)
        assert len(issues) == 0
        
        # Invalid fixations
        invalid_fixations = pd.DataFrame([
            {'start_ns': 100_000_000, 'end_ns': 0, 'dur_ms': -100, 'cx_px': 100, 'cy_px': 100, 'n_samples': 0},
        ])
        issues = validate_fixation_data(invalid_fixations)
        assert len(issues) > 0
        assert any("negative duration" in issue for issue in issues)
        assert any("zero sample count" in issue for issue in issues)
        assert any("inconsistent timestamps" in issue for issue in issues)
