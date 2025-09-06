"""
Real-time interfaces for gaze data acquisition.

Defines protocols and data structures for real-time gaze data streaming
from Pupil Labs devices and mock implementations.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Protocol
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


class GazeSample(BaseModel):
    """Gaze data sample from real-time stream."""
    
    t_ns: int = Field(description="Timestamp in nanoseconds")
    gx_px: float = Field(description="Gaze X coordinate in pixels")
    gy_px: float = Field(description="Gaze Y coordinate in pixels")
    frame_w: int = Field(description="Frame width in pixels")
    frame_h: int = Field(description="Frame height in pixels")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence value")
    pupil_diameter: Optional[float] = Field(default=None, description="Pupil diameter")
    eye_id: Optional[str] = Field(default=None, description="Eye identifier")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()
    
    def is_valid(self) -> bool:
        """Check if sample is valid."""
        return (
            self.confidence > 0 and
            0 <= self.gx_px <= self.frame_w and
            0 <= self.gy_px <= self.frame_h
        )


class Annotation(BaseModel):
    """Annotation data from real-time stream."""
    
    t_ns: int = Field(description="Timestamp in nanoseconds")
    label: str = Field(description="Annotation label")
    data: dict = Field(default_factory=dict, description="Additional annotation data")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()


class RealtimeClient(ABC):
    """Abstract base class for real-time gaze data clients."""
    
    @abstractmethod
    def subscribe_gaze(self, callback: Callable[[GazeSample], None]) -> None:
        """
        Subscribe to gaze data stream.
        
        Args:
            callback: Function to call with each gaze sample
        """
        pass
    
    @abstractmethod
    def subscribe_annotations(self, callback: Callable[[Annotation], None]) -> None:
        """
        Subscribe to annotation stream.
        
        Args:
            callback: Function to call with each annotation
        """
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the real-time client."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the real-time client."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if client is running."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> dict:
        """Get device information."""
        pass


class GazeCallback(Protocol):
    """Protocol for gaze data callbacks."""
    
    def __call__(self, sample: GazeSample) -> None:
        """Handle gaze sample."""
        ...


class AnnotationCallback(Protocol):
    """Protocol for annotation callbacks."""
    
    def __call__(self, annotation: Annotation) -> None:
        """Handle annotation."""
        ...


class RealtimeClientFactory:
    """Factory for creating real-time clients."""
    
    @staticmethod
    def create_mock_client(
        gaze_file: Optional[str] = None,
        world_video: Optional[str] = None,
        playback_speed: float = 1.0
    ) -> "MockRealtimeClient":
        """
        Create a mock real-time client.
        
        Args:
            gaze_file: Path to gaze CSV file
            world_video: Path to world video file
            playback_speed: Playback speed multiplier
            
        Returns:
            MockRealtimeClient instance
        """
        from .mock_client import MockRealtimeClient
        return MockRealtimeClient(gaze_file, world_video, playback_speed)
    
    @staticmethod
    def create_pupil_client(
        host: str = "127.0.0.1",
        port: int = 8080
    ) -> "PupilRealtimeClient":
        """
        Create a Pupil Labs real-time client.
        
        Args:
            host: Device host address
            port: Device port
            
        Returns:
            PupilRealtimeClient instance
            
        Raises:
            ImportError: If pupil client dependencies are not available
        """
        try:
            from .pupil_client import PupilRealtimeClient
            return PupilRealtimeClient(host, port)
        except ImportError as e:
            raise ImportError(
                "Pupil Labs real-time client dependencies not available. "
                "Install with: pip install gaze-lab[realtime]"
            ) from e
    
    @staticmethod
    def create_client(
        mode: str = "mock",
        **kwargs
    ) -> RealtimeClient:
        """
        Create a real-time client based on mode.
        
        Args:
            mode: Client mode ("mock" or "pupil")
            **kwargs: Additional arguments for client creation
            
        Returns:
            RealtimeClient instance
        """
        if mode == "mock":
            return RealtimeClientFactory.create_mock_client(**kwargs)
        elif mode == "pupil":
            return RealtimeClientFactory.create_pupil_client(**kwargs)
        else:
            raise ValueError(f"Unknown client mode: {mode}")


class RealtimeRecorder:
    """Utility class for recording real-time data."""
    
    def __init__(self, client: RealtimeClient):
        self.client = client
        self.gaze_samples: list[GazeSample] = []
        self.annotations: list[Annotation] = []
        self._gaze_callback: Optional[Callable[[GazeSample], None]] = None
        self._annotation_callback: Optional[Callable[[Annotation], None]] = None
    
    def start_recording(self) -> None:
        """Start recording data."""
        self.gaze_samples.clear()
        self.annotations.clear()
        
        # Set up callbacks
        self._gaze_callback = self._on_gaze_sample
        self._annotation_callback = self._on_annotation
        
        # Subscribe to streams
        self.client.subscribe_gaze(self._gaze_callback)
        self.client.subscribe_annotations(self._annotation_callback)
        
        # Start client
        self.client.start()
    
    def stop_recording(self) -> None:
        """Stop recording data."""
        self.client.stop()
    
    def _on_gaze_sample(self, sample: GazeSample) -> None:
        """Handle gaze sample."""
        self.gaze_samples.append(sample)
    
    def _on_annotation(self, annotation: Annotation) -> None:
        """Handle annotation."""
        self.annotations.append(annotation)
    
    def get_gaze_data(self) -> list[GazeSample]:
        """Get recorded gaze data."""
        return self.gaze_samples.copy()
    
    def get_annotations(self) -> list[Annotation]:
        """Get recorded annotations."""
        return self.annotations.copy()
    
    def save_gaze_data(self, filepath: str) -> None:
        """Save gaze data to CSV file."""
        import pandas as pd
        
        if not self.gaze_samples:
            raise ValueError("No gaze data to save")
        
        # Convert to DataFrame
        data = [sample.to_dict() for sample in self.gaze_samples]
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
    
    def save_annotations(self, filepath: str) -> None:
        """Save annotations to JSON file."""
        import json
        
        if not self.annotations:
            raise ValueError("No annotations to save")
        
        # Convert to list of dicts
        data = [annotation.to_dict() for annotation in self.annotations]
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
