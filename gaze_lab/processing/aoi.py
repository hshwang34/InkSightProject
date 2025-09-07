"""
Area of Interest (AOI) analysis for gaze data.

Provides comprehensive AOI analysis including dwell time calculation,
fixation counting, and timeline generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.prepared import prep

from ..logging_setup import get_logger

logger = get_logger(__name__)


class AOI:
    """Area of Interest definition."""
    
    def __init__(
        self,
        name: str,
        aoi_type: str,
        coordinates: List[Tuple[float, float]],
        metadata: Optional[Dict] = None,
    ):
        """
        Initialize AOI.
        
        Args:
            name: AOI name
            aoi_type: AOI type ("rectangle", "circle", "polygon")
            coordinates: List of coordinate tuples
            metadata: Optional metadata dictionary
        """
        self.name = name
        self.type = aoi_type
        self.coordinates = coordinates
        self.metadata = metadata or {}
        
        # Create Shapely geometry
        self._geometry = self._create_geometry()
        self._prepared_geometry = prep(self._geometry)
    
    def _create_geometry(self):
        """Create Shapely geometry from coordinates."""
        if self.type == "rectangle":
            if len(self.coordinates) != 4:
                raise ValueError("Rectangle AOI must have 4 coordinates")
            
            # Get bounding box
            x_coords = [coord[0] for coord in self.coordinates]
            y_coords = [coord[1] for coord in self.coordinates]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            return box(min_x, min_y, max_x, max_y)
        
        elif self.type == "circle":
            if len(self.coordinates) != 1:
                raise ValueError("Circle AOI must have 1 coordinate (center, radius)")
            
            center_x, center_y, radius = self.coordinates[0]
            
            # Create circle as polygon approximation
            angles = np.linspace(0, 2 * np.pi, 32)
            x_coords = center_x + radius * np.cos(angles)
            y_coords = center_y + radius * np.sin(angles)
            
            return Polygon(list(zip(x_coords, y_coords)))
        
        elif self.type == "polygon":
            if len(self.coordinates) < 3:
                raise ValueError("Polygon AOI must have at least 3 coordinates")
            
            return Polygon(self.coordinates)
        
        else:
            raise ValueError(f"Unknown AOI type: {self.type}")
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside AOI."""
        point = Point(x, y)
        return self._prepared_geometry.contains(point)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get AOI bounds (min_x, min_y, max_x, max_y)."""
        return self._geometry.bounds
    
    def get_area(self) -> float:
        """Get AOI area."""
        return self._geometry.area
    
    def to_dict(self) -> Dict:
        """Convert AOI to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "coordinates": self.coordinates,
            "metadata": self.metadata,
        }


class AOIAnalyzer:
    """AOI analyzer for gaze data."""
    
    def __init__(self):
        """Initialize AOI analyzer."""
        self.aois: Dict[str, AOI] = {}
    
    def add_aoi(self, aoi: AOI) -> None:
        """Add AOI to analyzer."""
        self.aois[aoi.name] = aoi
        logger.info(f"Added AOI: {aoi.name} ({aoi.type})")
    
    def load_aois_from_file(self, file_path: Union[str, Path]) -> None:
        """Load AOIs from JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"AOI file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for aoi_data in data.get("aois", []):
            aoi = AOI(
                name=aoi_data["name"],
                aoi_type=aoi_data["type"],
                coordinates=aoi_data["coordinates"],
                metadata=aoi_data.get("metadata", {}),
            )
            self.add_aoi(aoi)
        
        logger.info(f"Loaded {len(self.aois)} AOIs from {file_path}")
    
    def save_aois_to_file(self, file_path: Union[str, Path]) -> None:
        """Save AOIs to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "aois": [aoi.to_dict() for aoi in self.aois.values()],
            "metadata": {
                "total_aois": len(self.aois),
                "created_by": "GazeLab",
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.aois)} AOIs to {file_path}")
    
    def analyze_gaze_data(
        self,
        df: pd.DataFrame,
        min_dwell_ms: float = 100.0,
        gap_threshold_ms: float = 500.0,
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze gaze data against AOIs.
        
        Args:
            df: DataFrame with gaze data
            min_dwell_ms: Minimum dwell time in milliseconds
            gap_threshold_ms: Maximum gap between samples to consider continuous
            
        Returns:
            Dictionary mapping AOI names to hit DataFrames
        """
        if len(df) == 0:
            return {aoi_name: pd.DataFrame() for aoi_name in self.aois.keys()}
        
        aoi_hits = {}
        
        for aoi_name, aoi in self.aois.items():
            hits = self._find_aoi_hits(df, aoi)
            aoi_hits[aoi_name] = hits
        
        logger.info(f"Analyzed gaze data against {len(self.aois)} AOIs")
        
        return aoi_hits
    
    def _find_aoi_hits(self, df: pd.DataFrame, aoi: AOI) -> pd.DataFrame:
        """Find hits for a specific AOI."""
        hits = []
        
        for idx, row in df.iterrows():
            if aoi.contains_point(row['gx_px'], row['gy_px']):
                hits.append({
                    't_ns': row['t_ns'],
                    'gx_px': row['gx_px'],
                    'gy_px': row['gy_px'],
                    'aoi_name': aoi.name,
                })
        
        return pd.DataFrame(hits)
    
    def calculate_dwell_times(
        self,
        aoi_hits: Dict[str, pd.DataFrame],
        min_dwell_ms: float = 100.0,
        gap_threshold_ms: float = 500.0,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate dwell times from AOI hits.
        
        Args:
            aoi_hits: Dictionary mapping AOI names to hit DataFrames
            min_dwell_ms: Minimum dwell time in milliseconds
            gap_threshold_ms: Maximum gap between samples to consider continuous
            
        Returns:
            Dictionary mapping AOI names to dwell DataFrames
        """
        aoi_dwells = {}
        
        for aoi_name, hits in aoi_hits.items():
            if len(hits) == 0:
                aoi_dwells[aoi_name] = pd.DataFrame()
                continue
            
            dwells = self._calculate_aoi_dwells(hits, min_dwell_ms, gap_threshold_ms)
            aoi_dwells[aoi_name] = dwells
        
        logger.info(f"Calculated dwell times for {len(self.aois)} AOIs")
        
        return aoi_dwells
    
    def _calculate_aoi_dwells(
        self,
        hits: pd.DataFrame,
        min_dwell_ms: float,
        gap_threshold_ms: float,
    ) -> pd.DataFrame:
        """Calculate dwells for a specific AOI."""
        if len(hits) == 0:
            return pd.DataFrame()
        
        # Sort by timestamp
        hits = hits.sort_values('t_ns').reset_index(drop=True)
        
        dwells = []
        current_dwell_start = None
        current_dwell_hits = []
        
        for idx, hit in hits.iterrows():
            if current_dwell_start is None:
                # Start new dwell
                current_dwell_start = hit['t_ns']
                current_dwell_hits = [hit]
            else:
                # Check if this hit continues the current dwell
                time_gap = hit['t_ns'] - current_dwell_hits[-1]['t_ns']
                gap_ms = time_gap / 1_000_000
                
                if gap_ms <= gap_threshold_ms:
                    # Continue current dwell
                    current_dwell_hits.append(hit)
                else:
                    # End current dwell and start new one
                    dwell_duration_ms = (current_dwell_hits[-1]['t_ns'] - current_dwell_start) / 1_000_000
                    
                    if dwell_duration_ms >= min_dwell_ms:
                        dwells.append({
                            'aoi_name': current_dwell_hits[0]['aoi_name'],
                            'start_ns': current_dwell_start,
                            'end_ns': current_dwell_hits[-1]['t_ns'],
                            'dur_ms': dwell_duration_ms,
                            'n_hits': len(current_dwell_hits),
                        })
                    
                    # Start new dwell
                    current_dwell_start = hit['t_ns']
                    current_dwell_hits = [hit]
        
        # Handle final dwell
        if current_dwell_hits:
            dwell_duration_ms = (current_dwell_hits[-1]['t_ns'] - current_dwell_start) / 1_000_000
            
            if dwell_duration_ms >= min_dwell_ms:
                dwells.append({
                    'aoi_name': current_dwell_hits[0]['aoi_name'],
                    'start_ns': current_dwell_start,
                    'end_ns': current_dwell_hits[-1]['t_ns'],
                    'dur_ms': dwell_duration_ms,
                    'n_hits': len(current_dwell_hits),
                })
        
        return pd.DataFrame(dwells)
    
    def generate_aoi_report(
        self,
        aoi_hits: Dict[str, pd.DataFrame],
        aoi_dwells: Dict[str, pd.DataFrame],
        fixation_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate comprehensive AOI report.
        
        Args:
            aoi_hits: Dictionary mapping AOI names to hit DataFrames
            aoi_dwells: Dictionary mapping AOI names to dwell DataFrames
            fixation_df: Optional DataFrame with fixation data
            
        Returns:
            DataFrame with AOI report
        """
        report_data = []
        
        for aoi_name in self.aois.keys():
            hits = aoi_hits.get(aoi_name, pd.DataFrame())
            dwells = aoi_dwells.get(aoi_name, pd.DataFrame())
            
            # Calculate basic statistics
            total_hits = len(hits)
            total_dwells = len(dwells)
            total_dwell_time_ms = dwells['dur_ms'].sum() if len(dwells) > 0 else 0.0
            
            # Calculate fixation statistics
            fixation_count = 0
            if fixation_df is not None and len(fixation_df) > 0:
                fixation_count = self._count_fixations_in_aoi(fixation_df, aoi_name)
            
            # Calculate first and last hit times
            first_hit_time = hits['t_ns'].min() if len(hits) > 0 else None
            last_hit_time = hits['t_ns'].max() if len(hits) > 0 else None
            
            # Calculate mean dwell duration
            mean_dwell_duration_ms = dwells['dur_ms'].mean() if len(dwells) > 0 else 0.0
            
            report_data.append({
                'aoi_name': aoi_name,
                'total_hits': total_hits,
                'total_dwells': total_dwells,
                'total_dwell_time_ms': total_dwell_time_ms,
                'fixation_count': fixation_count,
                'first_hit_time_ns': first_hit_time,
                'last_hit_time_ns': last_hit_time,
                'mean_dwell_duration_ms': mean_dwell_duration_ms,
            })
        
        return pd.DataFrame(report_data)
    
    def _count_fixations_in_aoi(self, fixation_df: pd.DataFrame, aoi_name: str) -> int:
        """Count fixations that overlap with AOI."""
        if len(fixation_df) == 0:
            return 0
        
        aoi = self.aois[aoi_name]
        count = 0
        
        for _, fixation in fixation_df.iterrows():
            if aoi.contains_point(fixation['cx_px'], fixation['cy_px']):
                count += 1
        
        return count
    
    def generate_aoi_timeline(
        self,
        aoi_dwells: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Generate AOI timeline with enter/exit events.
        
        Args:
            aoi_dwells: Dictionary mapping AOI names to dwell DataFrames
            
        Returns:
            DataFrame with timeline events
        """
        timeline_data = []
        
        for aoi_name, dwells in aoi_dwells.items():
            for _, dwell in dwells.iterrows():
                timeline_data.append({
                    'aoi_name': aoi_name,
                    'enter_ns': dwell['start_ns'],
                    'exit_ns': dwell['end_ns'],
                    'dwell_ms': dwell['dur_ms'],
                })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        # Sort by enter time
        if len(timeline_df) > 0:
            timeline_df = timeline_df.sort_values('enter_ns').reset_index(drop=True)
        
        return timeline_df
    
    def get_aoi_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all AOIs."""
        stats = {}
        
        for aoi_name, aoi in self.aois.items():
            bounds = aoi.get_bounds()
            area = aoi.get_area()
            
            stats[aoi_name] = {
                'type': aoi.type,
                'bounds': bounds,
                'area': area,
                'coordinates': aoi.coordinates,
                'metadata': aoi.metadata,
            }
        
        return stats
    
    def validate_aois(self) -> List[str]:
        """Validate AOI definitions and return list of issues."""
        issues = []
        
        for aoi_name, aoi in self.aois.items():
            # Check for empty coordinates
            if not aoi.coordinates:
                issues.append(f"AOI '{aoi_name}' has no coordinates")
            
            # Check for invalid geometry
            try:
                if not aoi._geometry.is_valid:
                    issues.append(f"AOI '{aoi_name}' has invalid geometry")
            except Exception as e:
                issues.append(f"AOI '{aoi_name}' geometry error: {e}")
            
            # Check for zero area
            if aoi.get_area() == 0:
                issues.append(f"AOI '{aoi_name}' has zero area")
        
        return issues


def peak_dwell_timestamp(
    gaze_df: pd.DataFrame,
    aoi: Dict,
    window_ms: int = 5000
) -> int:
    """
    Find timestamp with peak dwell time within an AOI using sliding window.
    
    Args:
        gaze_df: Gaze data with canonical schema (t_ns, gx_px, gy_px, frame_w, frame_h)
        aoi: AOI dictionary with 'type' and 'coordinates' keys
        window_ms: Sliding window size in milliseconds
        
    Returns:
        Timestamp (t_ns) at center of window with maximum dwell time
        
    Raises:
        ValueError: If no gaze data found within AOI or invalid parameters
    """
    if gaze_df.empty:
        raise ValueError("Gaze data is empty")
    
    if window_ms <= 0:
        raise ValueError("Window size must be positive")
    
    # Create AOI object for hit testing
    aoi_obj = AOI(
        name="temp",
        aoi_type=aoi["type"],
        coordinates=aoi["coordinates"],
        metadata=aoi.get("metadata", {})
    )
    
    # Filter gaze data to points within AOI
    in_aoi_mask = []
    for _, row in gaze_df.iterrows():
        if pd.isna(row["gx_px"]) or pd.isna(row["gy_px"]):
            in_aoi_mask.append(False)
        else:
            point = Point(row["gx_px"], row["gy_px"])
            in_aoi_mask.append(aoi_obj.contains_point(point))
    
    aoi_gaze = gaze_df[in_aoi_mask].copy()
    
    if aoi_gaze.empty:
        raise ValueError(f"No gaze data found within AOI '{aoi.get('name', 'unnamed')}'")
    
    logger.debug(f"Found {len(aoi_gaze)} gaze points within AOI out of {len(gaze_df)} total")
    
    # Convert window size to nanoseconds
    window_ns = window_ms * 1_000_000
    
    # Get time range
    min_time = aoi_gaze["t_ns"].min()
    max_time = aoi_gaze["t_ns"].max()
    
    if max_time - min_time < window_ns:
        # Data span is shorter than window, return middle timestamp
        middle_time = int((min_time + max_time) / 2)
        logger.debug(f"Data span ({(max_time - min_time) / 1e9:.2f}s) shorter than window ({window_ms/1000:.2f}s), returning middle timestamp")
        return middle_time
    
    # Sliding window analysis
    best_dwell = 0
    best_timestamp = int((min_time + max_time) / 2)  # Default to middle
    
    # Step size (smaller = more precise, larger = faster)
    step_ns = window_ns // 10  # 10 steps per window
    
    current_time = min_time
    while current_time + window_ns <= max_time:
        window_start = current_time
        window_end = current_time + window_ns
        
        # Count gaze points in this window
        window_mask = (aoi_gaze["t_ns"] >= window_start) & (aoi_gaze["t_ns"] < window_end)
        window_gaze = aoi_gaze[window_mask]
        
        if len(window_gaze) > 0:
            # Calculate dwell time (sum of inter-sample intervals)
            window_gaze_sorted = window_gaze.sort_values("t_ns")
            time_diffs = window_gaze_sorted["t_ns"].diff().fillna(0)
            
            # Assume average sampling rate for first sample
            if len(time_diffs) > 1:
                avg_interval = time_diffs[1:].mean()
                time_diffs.iloc[0] = avg_interval
            
            total_dwell = time_diffs.sum()
            
            if total_dwell > best_dwell:
                best_dwell = total_dwell
                best_timestamp = int(window_start + window_ns // 2)  # Center of window
        
        current_time += step_ns
    
    logger.debug(f"Peak dwell: {best_dwell / 1e9:.3f}s at timestamp {best_timestamp} ({best_timestamp / 1e9:.3f}s)")
    
    return best_timestamp