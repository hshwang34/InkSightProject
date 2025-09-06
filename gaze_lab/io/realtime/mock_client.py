"""
Mock real-time client for testing and demonstration.

Replays gaze data from CSV files and video frames in real-time,
useful for demos without hardware.
"""

import asyncio
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import pandas as pd

from .interfaces import GazeSample, Annotation, RealtimeClient
from ...logging_setup import get_logger

logger = get_logger(__name__)


class MockRealtimeClient(RealtimeClient):
    """Mock real-time client that replays recorded data."""
    
    def __init__(
        self,
        gaze_file: Optional[str] = None,
        world_video: Optional[str] = None,
        playback_speed: float = 1.0
    ):
        """
        Initialize mock client.
        
        Args:
            gaze_file: Path to gaze CSV file
            world_video: Path to world video file
            playback_speed: Playback speed multiplier
        """
        self.gaze_file = Path(gaze_file) if gaze_file else None
        self.world_video = Path(world_video) if world_video else None
        self.playback_speed = playback_speed
        
        self._running = False
        self._gaze_callback: Optional[Callable[[GazeSample], None]] = None
        self._annotation_callback: Optional[Callable[[Annotation], None]] = None
        
        self._gaze_data: Optional[pd.DataFrame] = None
        self._video_cap: Optional[cv2.VideoCapture] = None
        self._start_time: Optional[float] = None
        self._gaze_index = 0
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load gaze data and video."""
        if self.gaze_file and self.gaze_file.exists():
            try:
                from ...io.cloud_loader import load_gaze
                self._gaze_data = load_gaze(self.gaze_file)
                logger.info(f"Loaded {len(self._gaze_data)} gaze samples from {self.gaze_file}")
            except Exception as e:
                logger.error(f"Failed to load gaze data: {e}")
                self._gaze_data = None
        else:
            logger.warning("No gaze file provided or file not found")
        
        if self.world_video and self.world_video.exists():
            try:
                self._video_cap = cv2.VideoCapture(str(self.world_video))
                if not self._video_cap.isOpened():
                    logger.error(f"Failed to open video: {self.world_video}")
                    self._video_cap = None
                else:
                    fps = self._video_cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    logger.info(f"Loaded video: {frame_count} frames at {fps} FPS")
            except Exception as e:
                logger.error(f"Failed to load video: {e}")
                self._video_cap = None
        else:
            logger.warning("No video file provided or file not found")
    
    def subscribe_gaze(self, callback: Callable[[GazeSample], None]) -> None:
        """Subscribe to gaze data stream."""
        self._gaze_callback = callback
        logger.info("Subscribed to gaze stream")
    
    def subscribe_annotations(self, callback: Callable[[Annotation], None]) -> None:
        """Subscribe to annotation stream."""
        self._annotation_callback = callback
        logger.info("Subscribed to annotation stream")
    
    def start(self) -> None:
        """Start the mock client."""
        if self._running:
            logger.warning("Client is already running")
            return
        
        if not self._gaze_data is not None:
            raise ValueError("No gaze data available to replay")
        
        self._running = True
        self._start_time = time.time()
        self._gaze_index = 0
        
        logger.info("Started mock real-time client")
        
        # Start the replay loop in a separate thread
        import threading
        self._replay_thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._replay_thread.start()
    
    def stop(self) -> None:
        """Stop the mock client."""
        if not self._running:
            logger.warning("Client is not running")
            return
        
        self._running = False
        
        if hasattr(self, '_replay_thread'):
            self._replay_thread.join(timeout=1.0)
        
        if self._video_cap:
            self._video_cap.release()
        
        logger.info("Stopped mock real-time client")
    
    def is_running(self) -> bool:
        """Check if client is running."""
        return self._running
    
    def get_device_info(self) -> dict:
        """Get mock device information."""
        info = {
            "device_type": "mock",
            "device_name": "Mock Pupil Labs Device",
            "version": "1.0.0",
            "gaze_samples": len(self._gaze_data) if self._gaze_data is not None else 0,
            "has_video": self._video_cap is not None,
        }
        
        if self._video_cap:
            info.update({
                "video_fps": self._video_cap.get(cv2.CAP_PROP_FPS),
                "video_frames": int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "video_width": int(self._video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "video_height": int(self._video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            })
        
        return info
    
    def _replay_loop(self) -> None:
        """Main replay loop."""
        if not self._gaze_data is not None:
            return
        
        # Get first timestamp as reference
        first_timestamp = self._gaze_data.iloc[0]['t_ns']
        
        while self._running and self._gaze_index < len(self._gaze_data):
            current_time = time.time()
            elapsed_time = (current_time - self._start_time) * self.playback_speed
            
            # Convert elapsed time to nanoseconds
            elapsed_ns = int(elapsed_time * 1_000_000_000)
            
            # Find gaze samples to send
            while (self._gaze_index < len(self._gaze_data) and
                   self._gaze_data.iloc[self._gaze_index]['t_ns'] <= first_timestamp + elapsed_ns):
                
                row = self._gaze_data.iloc[self._gaze_index]
                
                # Create gaze sample
                sample = GazeSample(
                    t_ns=int(row['t_ns']),
                    gx_px=float(row['gx_px']),
                    gy_px=float(row['gy_px']),
                    frame_w=int(row['frame_w']),
                    frame_h=int(row['frame_h']),
                    confidence=1.0,  # Mock data has full confidence
                )
                
                # Send sample
                if self._gaze_callback:
                    try:
                        self._gaze_callback(sample)
                    except Exception as e:
                        logger.error(f"Error in gaze callback: {e}")
                
                self._gaze_index += 1
            
            # Small sleep to prevent busy waiting
            time.sleep(0.001)
        
        # Replay complete
        logger.info("Gaze data replay complete")
        self._running = False
    
    def get_video_frame(self, timestamp_ns: int) -> Optional[tuple]:
        """
        Get video frame for given timestamp.
        
        Args:
            timestamp_ns: Timestamp in nanoseconds
            
        Returns:
            Tuple of (frame_data, frame_timestamp) or None
        """
        if not self._video_cap:
            return None
        
        # Calculate frame index based on timestamp
        if not self._gaze_data is not None or len(self._gaze_data) == 0:
            return None
        
        # Get first and last timestamps
        first_timestamp = self._gaze_data.iloc[0]['t_ns']
        last_timestamp = self._gaze_data.iloc[-1]['t_ns']
        
        # Calculate relative timestamp
        relative_timestamp = timestamp_ns - first_timestamp
        total_duration = last_timestamp - first_timestamp
        
        if relative_timestamp < 0 or relative_timestamp > total_duration:
            return None
        
        # Calculate frame index
        fps = self._video_cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_index = int((relative_timestamp / total_duration) * frame_count)
        frame_index = max(0, min(frame_index, frame_count - 1))
        
        # Seek to frame
        self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._video_cap.read()
        
        if ret:
            return frame, timestamp_ns
        
        return None
    
    def reset(self) -> None:
        """Reset the client to beginning."""
        self._gaze_index = 0
        if self._video_cap:
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logger.info("Reset mock client to beginning")
    
    def set_playback_speed(self, speed: float) -> None:
        """Set playback speed."""
        self.playback_speed = max(0.1, min(speed, 10.0))  # Clamp between 0.1x and 10x
        logger.info(f"Set playback speed to {self.playback_speed}x")
    
    def get_progress(self) -> float:
        """Get replay progress (0.0 to 1.0)."""
        if not self._gaze_data is not None or len(self._gaze_data) == 0:
            return 0.0
        
        return self._gaze_index / len(self._gaze_data)
    
    def get_remaining_time(self) -> float:
        """Get estimated remaining time in seconds."""
        if not self._running or not self._gaze_data is not None:
            return 0.0
        
        remaining_samples = len(self._gaze_data) - self._gaze_index
        if remaining_samples == 0:
            return 0.0
        
        # Estimate based on current playback speed
        # This is a rough estimate
        return remaining_samples / (1000 * self.playback_speed)  # Assume ~1000 samples per second
