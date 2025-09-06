"""
Tests for AOI analysis functionality.
"""

import numpy as np
import pandas as pd
import pytest

from gaze_lab.processing.aoi import AOI, AOIAnalyzer


class TestAOI:
    """Test AOI functionality."""
    
    def test_rectangle_aoi(self):
        """Test rectangle AOI creation and point containment."""
        aoi = AOI(
            name="test_rect",
            aoi_type="rectangle",
            coordinates=[(100, 100), (200, 100), (200, 200), (100, 200)]
        )
        
        # Test point containment
        assert aoi.contains_point(150, 150)  # Inside
        assert not aoi.contains_point(50, 150)  # Outside left
        assert not aoi.contains_point(250, 150)  # Outside right
        assert not aoi.contains_point(150, 50)  # Outside top
        assert not aoi.contains_point(150, 250)  # Outside bottom
        
        # Test bounds
        bounds = aoi.get_bounds()
        assert bounds == (100, 100, 200, 200)
        
        # Test area
        area = aoi.get_area()
        assert area == 10000  # 100x100
    
    def test_circle_aoi(self):
        """Test circle AOI creation and point containment."""
        aoi = AOI(
            name="test_circle",
            aoi_type="circle",
            coordinates=[(100, 100, 50)]  # center_x, center_y, radius
        )
        
        # Test point containment
        assert aoi.contains_point(100, 100)  # Center
        assert aoi.contains_point(120, 100)  # Inside
        assert not aoi.contains_point(160, 100)  # Outside
        assert not aoi.contains_point(100, 160)  # Outside
    
    def test_polygon_aoi(self):
        """Test polygon AOI creation and point containment."""
        aoi = AOI(
            name="test_polygon",
            aoi_type="polygon",
            coordinates=[(100, 100), (200, 100), (150, 200)]
        )
        
        # Test point containment
        assert aoi.contains_point(150, 120)  # Inside
        assert not aoi.contains_point(50, 50)  # Outside
        assert not aoi.contains_point(250, 250)  # Outside


class TestAOIAnalyzer:
    """Test AOI analyzer functionality."""
    
    def test_aoi_analyzer_basic(self):
        """Test basic AOI analyzer functionality."""
        analyzer = AOIAnalyzer()
        
        # Add AOI
        aoi = AOI(
            name="test_aoi",
            aoi_type="rectangle",
            coordinates=[(100, 100), (200, 100), (200, 200), (100, 200)]
        )
        analyzer.add_aoi(aoi)
        
        assert len(analyzer.aois) == 1
        assert "test_aoi" in analyzer.aois
    
    def test_aoi_analysis(self):
        """Test AOI analysis with synthetic gaze data."""
        analyzer = AOIAnalyzer()
        
        # Add AOI
        aoi = AOI(
            name="test_aoi",
            aoi_type="rectangle",
            coordinates=[(100, 100), (200, 100), (200, 200), (100, 200)]
        )
        analyzer.add_aoi(aoi)
        
        # Create gaze data
        gaze_data = []
        
        # Gaze points inside AOI
        for i in range(10):
            gaze_data.append({
                't_ns': i * 10_000_000,
                'gx_px': 150 + np.random.normal(0, 10),
                'gy_px': 150 + np.random.normal(0, 10),
                'frame_w': 1280,
                'frame_h': 720,
            })
        
        # Gaze points outside AOI
        for i in range(5):
            gaze_data.append({
                't_ns': (10 + i) * 10_000_000,
                'gx_px': 300 + np.random.normal(0, 10),
                'gy_px': 300 + np.random.normal(0, 10),
                'frame_w': 1280,
                'frame_h': 720,
            })
        
        df = pd.DataFrame(gaze_data)
        
        # Analyze gaze data
        aoi_hits = analyzer.analyze_gaze_data(df)
        
        assert "test_aoi" in aoi_hits
        assert len(aoi_hits["test_aoi"]) == 10  # Only points inside AOI
    
    def test_dwell_calculation(self):
        """Test dwell time calculation."""
        analyzer = AOIAnalyzer()
        
        # Add AOI
        aoi = AOI(
            name="test_aoi",
            aoi_type="rectangle",
            coordinates=[(100, 100), (200, 100), (200, 200), (100, 200)]
        )
        analyzer.add_aoi(aoi)
        
        # Create hits data with dwell periods
        hits_data = []
        
        # First dwell: 100ms
        for i in range(10):
            hits_data.append({
                't_ns': i * 10_000_000,
                'gx_px': 150,
                'gy_px': 150,
                'aoi_name': 'test_aoi',
            })
        
        # Gap: 200ms
        # (no hits)
        
        # Second dwell: 150ms
        for i in range(15):
            hits_data.append({
                't_ns': (30 + i) * 10_000_000,
                'gx_px': 150,
                'gy_px': 150,
                'aoi_name': 'test_aoi',
            })
        
        hits_df = pd.DataFrame(hits_data)
        aoi_hits = {"test_aoi": hits_df}
        
        # Calculate dwells
        aoi_dwells = analyzer.calculate_dwell_times(aoi_hits, min_dwell_ms=50.0)
        
        assert "test_aoi" in aoi_dwells
        dwells = aoi_dwells["test_aoi"]
        assert len(dwells) == 2
        
        # Check first dwell
        first_dwell = dwells.iloc[0]
        assert 90 <= first_dwell['dur_ms'] <= 110
        
        # Check second dwell
        second_dwell = dwells.iloc[1]
        assert 140 <= second_dwell['dur_ms'] <= 160
    
    def test_aoi_report_generation(self):
        """Test AOI report generation."""
        analyzer = AOIAnalyzer()
        
        # Add AOI
        aoi = AOI(
            name="test_aoi",
            aoi_type="rectangle",
            coordinates=[(100, 100), (200, 100), (200, 200), (100, 200)]
        )
        analyzer.add_aoi(aoi)
        
        # Create mock hits and dwells
        hits_data = [{'t_ns': i * 10_000_000, 'gx_px': 150, 'gy_px': 150, 'aoi_name': 'test_aoi'} for i in range(10)]
        hits_df = pd.DataFrame(hits_data)
        aoi_hits = {"test_aoi": hits_df}
        
        dwells_data = [{'aoi_name': 'test_aoi', 'start_ns': 0, 'end_ns': 100_000_000, 'dur_ms': 100, 'n_hits': 10}]
        dwells_df = pd.DataFrame(dwells_data)
        aoi_dwells = {"test_aoi": dwells_df}
        
        # Generate report
        report = analyzer.generate_aoi_report(aoi_hits, aoi_dwells)
        
        assert len(report) == 1
        assert report.iloc[0]['aoi_name'] == 'test_aoi'
        assert report.iloc[0]['total_hits'] == 10
        assert report.iloc[0]['total_dwells'] == 1
        assert report.iloc[0]['total_dwell_time_ms'] == 100