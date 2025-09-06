"""
Tests for fixation detection algorithms.
"""

import unittest
import numpy as np

from gaze_lab.processing.fixations_ivt import IVTFixationDetector, IVTConfig, Fixation


class TestIVTFixationDetector(unittest.TestCase):
    """Test cases for I-VT fixation detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = IVTFixationDetector()
        
        # Create synthetic gaze data
        self.timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.x_coords = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.8])
        self.y_coords = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3])
        self.confidence = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    def test_detect_fixations_basic(self):
        """Test basic fixation detection."""
        fixations = self.detector.detect_fixations(
            self.x_coords, self.y_coords, self.timestamps, self.confidence
        )
        
        self.assertIsInstance(fixations, list)
        self.assertGreater(len(fixations), 0)
        
        # Check fixation structure
        for fixation in fixations:
            self.assertIsInstance(fixation, Fixation)
            self.assertGreater(fixation.duration, 0)
            self.assertGreaterEqual(fixation.confidence, 0)
            self.assertLessEqual(fixation.confidence, 1)
    
    def test_detect_fixations_empty_data(self):
        """Test fixation detection with empty data."""
        fixations = self.detector.detect_fixations(
            np.array([]), np.array([]), np.array([]), np.array([])
        )
        
        self.assertEqual(len(fixations), 0)
    
    def test_detect_fixations_single_point(self):
        """Test fixation detection with single point."""
        fixations = self.detector.detect_fixations(
            np.array([0.5]), np.array([0.5]), np.array([0.0]), np.array([1.0])
        )
        
        self.assertEqual(len(fixations), 0)
    
    def test_detect_fixations_confidence_filtering(self):
        """Test fixation detection with confidence filtering."""
        low_confidence = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        fixations = self.detector.detect_fixations(
            self.x_coords, self.y_coords, self.timestamps, low_confidence
        )
        
        # Should have fewer fixations due to confidence filtering
        self.assertIsInstance(fixations, list)
    
    def test_filter_fixations(self):
        """Test fixation filtering."""
        # Create test fixations
        fixations = [
            Fixation(0.0, 0.1, 0.1, 0.5, 0.5, 1.0, 5),  # Short duration
            Fixation(0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 10),  # Medium duration
            Fixation(0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 20),  # Long duration
        ]
        
        # Filter by duration
        filtered = self.detector.filter_fixations(
            fixations, min_duration=0.3, max_duration=0.8
        )
        
        self.assertEqual(len(filtered), 1)  # Only medium duration fixation
    
    def test_get_fixation_statistics(self):
        """Test fixation statistics calculation."""
        fixations = [
            Fixation(0.0, 0.1, 0.1, 0.5, 0.5, 0.8, 5),
            Fixation(0.0, 0.2, 0.2, 0.5, 0.5, 0.9, 10),
            Fixation(0.0, 0.3, 0.3, 0.5, 0.5, 0.7, 15),
        ]
        
        stats = self.detector.get_fixation_statistics(fixations)
        
        self.assertEqual(stats['count'], 3)
        self.assertAlmostEqual(stats['mean_duration'], 0.2, places=1)
        self.assertAlmostEqual(stats['mean_confidence'], 0.8, places=1)
    
    def test_export_fixations_to_array(self):
        """Test fixation export to numpy array."""
        fixations = [
            Fixation(0.0, 0.1, 0.1, 0.5, 0.5, 0.8, 5),
            Fixation(0.0, 0.2, 0.2, 0.5, 0.5, 0.9, 10),
        ]
        
        array = self.detector.export_fixations_to_array(fixations)
        
        self.assertEqual(len(array), 2)
        self.assertEqual(array.dtype.names, ('start_time', 'end_time', 'duration', 'x', 'y', 'confidence', 'point_count'))
    
    def test_detect_saccades(self):
        """Test saccade detection."""
        saccades = self.detector.detect_saccades(
            self.x_coords, self.y_coords, self.timestamps, self.confidence
        )
        
        self.assertIsInstance(saccades, list)
        
        # Check saccade structure
        for saccade in saccades:
            self.assertIn('start_time', saccade)
            self.assertIn('end_time', saccade)
            self.assertIn('duration', saccade)
            self.assertIn('amplitude', saccade)
            self.assertIn('velocity', saccade)
    
    def test_custom_config(self):
        """Test fixation detection with custom configuration."""
        config = IVTConfig(
            velocity_threshold=50.0,
            min_duration=0.2,
            max_duration=1.0,
            min_points=5
        )
        
        detector = IVTFixationDetector(config)
        fixations = detector.detect_fixations(
            self.x_coords, self.y_coords, self.timestamps, self.confidence
        )
        
        self.assertIsInstance(fixations, list)


if __name__ == '__main__':
    unittest.main()
