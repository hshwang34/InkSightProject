"""
Compatibility layer for legacy Pupil Core/Player data formats.

Provides backward compatibility for older Pupil Labs data exports,
ensuring seamless integration with existing analysis workflows.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class CoreSessionInfo:
    """Legacy Core session information."""
    session_name: str
    recording_date: datetime
    duration: float
    sample_rate: float
    world_camera_fps: float
    eye_camera_fps: float
    pupil_version: str


@dataclass
class CoreGazeData:
    """Legacy Core gaze data structure."""
    timestamps: np.ndarray
    norm_pos_x: np.ndarray
    norm_pos_y: np.ndarray
    confidence: np.ndarray
    pupil_diameter: Optional[np.ndarray] = None
    eye_id: Optional[np.ndarray] = None
    session_info: CoreSessionInfo = None


class PlayerCompat:
    """
    Compatibility layer for legacy Pupil Core/Player data.
    
    Handles loading and conversion of older Pupil Labs data formats,
    providing standardized interfaces for modern analysis tools.
    """
    
    def __init__(self):
        """Initialize the compatibility layer."""
        self._legacy_field_mappings = self._initialize_legacy_mappings()
    
    def _initialize_legacy_mappings(self) -> Dict[str, str]:
        """Initialize field mappings for legacy Core data."""
        return {
            "timestamp": "timestamp",
            "norm_pos_x": "norm_pos_x", 
            "norm_pos_y": "norm_pos_y",
            "confidence": "confidence",
            "diameter": "diameter",
            "eye_id": "eye_id",
            "base_data": "base_data",
            "topic": "topic"
        }
    
    def load_core_session(self, session_path: Union[str, Path]) -> CoreGazeData:
        """
        Load legacy Pupil Core session data.
        
        Args:
            session_path: Path to Core session directory
            
        Returns:
            CoreGazeData object with standardized data
            
        Raises:
            FileNotFoundError: If session directory doesn't exist
            ValueError: If session data is invalid or corrupted
        """
        session_path = Path(session_path)
        if not session_path.exists():
            raise FileNotFoundError(f"Core session not found: {session_path}")
        
        try:
            # Load session info
            session_info = self._load_session_info(session_path)
            
            # Load gaze data
            gaze_data = self._load_core_gaze_data(session_path)
            
            return CoreGazeData(
                timestamps=gaze_data["timestamps"],
                norm_pos_x=gaze_data["norm_pos_x"],
                norm_pos_y=gaze_data["norm_pos_y"],
                confidence=gaze_data["confidence"],
                pupil_diameter=gaze_data.get("pupil_diameter"),
                eye_id=gaze_data.get("eye_id"),
                session_info=session_info
            )
            
        except Exception as e:
            logger.error(f"Failed to load Core session: {e}")
            raise ValueError(f"Invalid Core session: {e}")
    
    def _load_session_info(self, session_path: Path) -> CoreSessionInfo:
        """Load session information from Core session."""
        info_file = session_path / "info.csv"
        if not info_file.exists():
            # Try alternative info file locations
            for alt_name in ["info.json", "session_info.csv", "recording_info.csv"]:
                alt_file = session_path / alt_name
                if alt_file.exists():
                    info_file = alt_file
                    break
        
        if not info_file.exists():
            # Create minimal session info from directory name
            session_name = session_path.name
            recording_date = datetime.fromtimestamp(session_path.stat().st_mtime)
            return CoreSessionInfo(
                session_name=session_name,
                recording_date=recording_date,
                duration=0.0,
                sample_rate=0.0,
                world_camera_fps=0.0,
                eye_camera_fps=0.0,
                pupil_version="unknown"
            )
        
        try:
            if info_file.suffix == ".json":
                return self._parse_json_info(info_file)
            else:
                return self._parse_csv_info(info_file)
        except Exception as e:
            logger.warning(f"Failed to parse session info: {e}")
            # Return minimal info
            return CoreSessionInfo(
                session_name=session_path.name,
                recording_date=datetime.fromtimestamp(session_path.stat().st_mtime),
                duration=0.0,
                sample_rate=0.0,
                world_camera_fps=0.0,
                eye_camera_fps=0.0,
                pupil_version="unknown"
            )
    
    def _parse_json_info(self, info_file: Path) -> CoreSessionInfo:
        """Parse JSON session info file."""
        with open(info_file, 'r') as f:
            data = json.load(f)
        
        return CoreSessionInfo(
            session_name=data.get("session_name", info_file.parent.name),
            recording_date=datetime.fromisoformat(data.get("recording_date", "1970-01-01")),
            duration=float(data.get("duration", 0)),
            sample_rate=float(data.get("sample_rate", 0)),
            world_camera_fps=float(data.get("world_camera_fps", 0)),
            eye_camera_fps=float(data.get("eye_camera_fps", 0)),
            pupil_version=data.get("pupil_version", "unknown")
        )
    
    def _parse_csv_info(self, info_file: Path) -> CoreSessionInfo:
        """Parse CSV session info file."""
        df = pd.read_csv(info_file)
        
        # Extract info from CSV (format may vary)
        session_name = df.get("session_name", [info_file.parent.name]).iloc[0]
        recording_date = datetime.fromisoformat(
            df.get("recording_date", ["1970-01-01"]).iloc[0]
        )
        duration = float(df.get("duration", [0]).iloc[0])
        sample_rate = float(df.get("sample_rate", [0]).iloc[0])
        world_camera_fps = float(df.get("world_camera_fps", [0]).iloc[0])
        eye_camera_fps = float(df.get("eye_camera_fps", [0]).iloc[0])
        pupil_version = df.get("pupil_version", ["unknown"]).iloc[0]
        
        return CoreSessionInfo(
            session_name=session_name,
            recording_date=recording_date,
            duration=duration,
            sample_rate=sample_rate,
            world_camera_fps=world_camera_fps,
            eye_camera_fps=eye_camera_fps,
            pupil_version=pupil_version
        )
    
    def _load_core_gaze_data(self, session_path: Path) -> Dict[str, np.ndarray]:
        """Load gaze data from Core session."""
        # Look for gaze data files
        gaze_files = []
        for pattern in ["gaze.csv", "gaze_data.csv", "pupil_positions.csv"]:
            gaze_file = session_path / pattern
            if gaze_file.exists():
                gaze_files.append(gaze_file)
        
        if not gaze_files:
            raise FileNotFoundError("No gaze data files found in Core session")
        
        # Use the first available gaze file
        gaze_file = gaze_files[0]
        df = pd.read_csv(gaze_file)
        
        # Extract gaze data with legacy field names
        gaze_data = {}
        
        # Required fields
        gaze_data["timestamps"] = df["timestamp"].values.astype(float)
        gaze_data["norm_pos_x"] = df["norm_pos_x"].values.astype(float)
        gaze_data["norm_pos_y"] = df["norm_pos_y"].values.astype(float)
        gaze_data["confidence"] = df["confidence"].values.astype(float)
        
        # Optional fields
        if "diameter" in df.columns:
            gaze_data["pupil_diameter"] = df["diameter"].values.astype(float)
        
        if "eye_id" in df.columns:
            gaze_data["eye_id"] = df["eye_id"].values
        
        return gaze_data
    
    def convert_to_modern_format(self, core_data: CoreGazeData) -> Dict[str, np.ndarray]:
        """
        Convert legacy Core data to modern format.
        
        Args:
            core_data: Legacy Core gaze data
            
        Returns:
            Dictionary with modern field names and data
        """
        return {
            "timestamps": core_data.timestamps,
            "gaze_x": core_data.norm_pos_x,
            "gaze_y": core_data.norm_pos_y,
            "confidence": core_data.confidence,
            "pupil_diameter": core_data.pupil_diameter,
            "eye_id": core_data.eye_id
        }
    
    def load_core_fixations(self, session_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load fixation data from Core session.
        
        Args:
            session_path: Path to Core session directory
            
        Returns:
            Dictionary with fixation data
            
        Raises:
            FileNotFoundError: If fixation data not found
        """
        session_path = Path(session_path)
        
        # Look for fixation files
        fixation_files = []
        for pattern in ["fixations.csv", "fixation_data.csv", "fixations_ivt.csv"]:
            fixation_file = session_path / pattern
            if fixation_file.exists():
                fixation_files.append(fixation_file)
        
        if not fixation_files:
            raise FileNotFoundError("No fixation data files found in Core session")
        
        # Load fixation data
        fixation_file = fixation_files[0]
        df = pd.read_csv(fixation_file)
        
        return {
            "timestamps": df["timestamp"].values.astype(float),
            "fixation_x": df["norm_pos_x"].values.astype(float),
            "fixation_y": df["norm_pos_y"].values.astype(float),
            "duration": df["duration"].values.astype(float),
            "confidence": df.get("confidence", pd.Series([1.0] * len(df))).values.astype(float)
        }
    
    def validate_core_session(self, session_path: Union[str, Path]) -> Dict[str, bool]:
        """
        Validate Core session integrity.
        
        Args:
            session_path: Path to Core session directory
            
        Returns:
            Dictionary with validation results
        """
        session_path = Path(session_path)
        validation = {
            "session_exists": session_path.exists(),
            "has_gaze_data": False,
            "has_fixation_data": False,
            "has_session_info": False,
            "data_consistency": False
        }
        
        if not validation["session_exists"]:
            return validation
        
        # Check for gaze data
        for pattern in ["gaze.csv", "gaze_data.csv", "pupil_positions.csv"]:
            if (session_path / pattern).exists():
                validation["has_gaze_data"] = True
                break
        
        # Check for fixation data
        for pattern in ["fixations.csv", "fixation_data.csv", "fixations_ivt.csv"]:
            if (session_path / pattern).exists():
                validation["has_fixation_data"] = True
                break
        
        # Check for session info
        for pattern in ["info.csv", "info.json", "session_info.csv"]:
            if (session_path / pattern).exists():
                validation["has_session_info"] = True
                break
        
        # Validate data consistency if gaze data exists
        if validation["has_gaze_data"]:
            try:
                gaze_data = self._load_core_gaze_data(session_path)
                if (len(gaze_data["timestamps"]) == len(gaze_data["norm_pos_x"]) and
                    len(gaze_data["timestamps"]) == len(gaze_data["norm_pos_y"])):
                    validation["data_consistency"] = True
            except Exception:
                validation["data_consistency"] = False
        
        return validation
    
    def list_core_sessions(self, directory: Union[str, Path]) -> List[Path]:
        """
        List Core sessions in a directory.
        
        Args:
            directory: Directory to search for Core sessions
            
        Returns:
            List of Path objects for found Core sessions
        """
        directory = Path(directory)
        if not directory.exists():
            return []
        
        sessions = []
        for item in directory.iterdir():
            if item.is_dir():
                # Check if it looks like a Core session
                validation = self.validate_core_session(item)
                if validation["has_gaze_data"]:
                    sessions.append(item)
        
        return sorted(sessions)
    
    def get_core_session_summary(self, session_path: Union[str, Path]) -> Dict:
        """
        Get summary information about a Core session.
        
        Args:
            session_path: Path to Core session directory
            
        Returns:
            Dictionary with session summary
        """
        session_path = Path(session_path)
        validation = self.validate_core_session(session_path)
        
        summary = {
            "session_path": str(session_path),
            "session_name": session_path.name,
            "validation": validation,
            "file_count": 0,
            "total_size": 0
        }
        
        if validation["session_exists"]:
            # Count files and calculate total size
            for file_path in session_path.rglob("*"):
                if file_path.is_file():
                    summary["file_count"] += 1
                    summary["total_size"] += file_path.stat().st_size
            
            # Try to get session info
            try:
                session_info = self._load_session_info(session_path)
                summary["session_info"] = {
                    "recording_date": session_info.recording_date.isoformat(),
                    "duration": session_info.duration,
                    "sample_rate": session_info.sample_rate,
                    "pupil_version": session_info.pupil_version
                }
            except Exception:
                summary["session_info"] = None
        
        return summary
