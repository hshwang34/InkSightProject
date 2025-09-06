"""
Real-time client for Pupil Labs Neon streaming.

Implements the official Neon Real-Time API for live gaze data acquisition
with support for gaze, fixation, and scene camera streams.
"""

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Available stream types from Neon."""
    GAZE = "gaze"
    FIXATION = "fixation"
    SCENE_CAMERA = "scene_camera"


@dataclass
class GazeData:
    """Standardized gaze data structure."""
    timestamp: float
    x: float
    y: float
    confidence: float
    pupil_diameter: Optional[float] = None
    eye_id: Optional[str] = None


@dataclass
class FixationData:
    """Standardized fixation data structure."""
    timestamp: float
    x: float
    y: float
    duration: float
    confidence: float


@dataclass
class SceneFrame:
    """Scene camera frame data."""
    timestamp: float
    frame_id: int
    width: int
    height: int
    data: bytes


class RealtimeClient:
    """
    Real-time client for Pupil Labs Neon devices.
    
    Provides streaming access to gaze, fixation, and scene camera data
    following the official Neon Real-Time API specification.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        """
        Initialize the real-time client.
        
        Args:
            host: Neon device IP address or hostname
            port: Neon real-time API port (default: 8080)
        """
        self.host = host
        self.port = port
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.subscriptions: Dict[str, bool] = {}
        self._running = False
        
    async def connect(self) -> bool:
        """
        Establish connection to Neon device.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            uri = f"ws://{self.host}:{self.port}"
            self.websocket = await websockets.connect(uri)
            logger.info(f"Connected to Neon device at {uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neon device: {e}")
            return False
    
    async def disconnect(self):
        """Close connection to Neon device."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from Neon device")
    
    async def subscribe(self, stream_type: StreamType) -> bool:
        """
        Subscribe to a specific stream type.
        
        Args:
            stream_type: Type of stream to subscribe to
            
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.websocket:
            logger.error("Not connected to Neon device")
            return False
            
        try:
            message = {
                "type": "subscribe",
                "stream": stream_type.value
            }
            await self.websocket.send(json.dumps(message))
            self.subscriptions[stream_type.value] = True
            logger.info(f"Subscribed to {stream_type.value} stream")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to {stream_type.value}: {e}")
            return False
    
    async def unsubscribe(self, stream_type: StreamType) -> bool:
        """
        Unsubscribe from a specific stream type.
        
        Args:
            stream_type: Type of stream to unsubscribe from
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        if not self.websocket:
            return False
            
        try:
            message = {
                "type": "unsubscribe", 
                "stream": stream_type.value
            }
            await self.websocket.send(json.dumps(message))
            self.subscriptions[stream_type.value] = False
            logger.info(f"Unsubscribed from {stream_type.value} stream")
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {stream_type.value}: {e}")
            return False
    
    async def stream_gaze(self) -> AsyncIterator[GazeData]:
        """
        Stream gaze data from Neon device.
        
        Yields:
            GazeData objects with normalized coordinates and metadata
        """
        if not await self.subscribe(StreamType.GAZE):
            raise RuntimeError("Failed to subscribe to gaze stream")
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data.get("type") == "gaze":
                    gaze_data = self._parse_gaze_message(data)
                    if gaze_data:
                        yield gaze_data
        except ConnectionClosed:
            logger.warning("Gaze stream connection closed")
        except Exception as e:
            logger.error(f"Error in gaze stream: {e}")
        finally:
            await self.unsubscribe(StreamType.GAZE)
    
    async def stream_fixations(self) -> AsyncIterator[FixationData]:
        """
        Stream fixation data from Neon device.
        
        Yields:
            FixationData objects with fixation coordinates and duration
        """
        if not await self.subscribe(StreamType.FIXATION):
            raise RuntimeError("Failed to subscribe to fixation stream")
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data.get("type") == "fixation":
                    fixation_data = self._parse_fixation_message(data)
                    if fixation_data:
                        yield fixation_data
        except ConnectionClosed:
            logger.warning("Fixation stream connection closed")
        except Exception as e:
            logger.error(f"Error in fixation stream: {e}")
        finally:
            await self.unsubscribe(StreamType.FIXATION)
    
    async def stream_scene_camera(self) -> AsyncIterator[SceneFrame]:
        """
        Stream scene camera frames from Neon device.
        
        Yields:
            SceneFrame objects with frame data and metadata
        """
        if not await self.subscribe(StreamType.SCENE_CAMERA):
            raise RuntimeError("Failed to subscribe to scene camera stream")
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data.get("type") == "scene_camera":
                    frame_data = self._parse_scene_camera_message(data)
                    if frame_data:
                        yield frame_data
        except ConnectionClosed:
            logger.warning("Scene camera stream connection closed")
        except Exception as e:
            logger.error(f"Error in scene camera stream: {e}")
        finally:
            await self.unsubscribe(StreamType.SCENE_CAMERA)
    
    def _parse_gaze_message(self, data: Dict) -> Optional[GazeData]:
        """Parse gaze message from Neon API."""
        try:
            gaze = data.get("gaze", {})
            if not gaze:
                return None
                
            return GazeData(
                timestamp=data.get("timestamp", time.time()),
                x=gaze.get("x", 0.0),
                y=gaze.get("y", 0.0),
                confidence=gaze.get("confidence", 0.0),
                pupil_diameter=gaze.get("pupil_diameter"),
                eye_id=gaze.get("eye_id")
            )
        except Exception as e:
            logger.error(f"Failed to parse gaze message: {e}")
            return None
    
    def _parse_fixation_message(self, data: Dict) -> Optional[FixationData]:
        """Parse fixation message from Neon API."""
        try:
            fixation = data.get("fixation", {})
            if not fixation:
                return None
                
            return FixationData(
                timestamp=data.get("timestamp", time.time()),
                x=fixation.get("x", 0.0),
                y=fixation.get("y", 0.0),
                duration=fixation.get("duration", 0.0),
                confidence=fixation.get("confidence", 0.0)
            )
        except Exception as e:
            logger.error(f"Failed to parse fixation message: {e}")
            return None
    
    def _parse_scene_camera_message(self, data: Dict) -> Optional[SceneFrame]:
        """Parse scene camera message from Neon API."""
        try:
            frame = data.get("frame", {})
            if not frame:
                return None
                
            return SceneFrame(
                timestamp=data.get("timestamp", time.time()),
                frame_id=frame.get("frame_id", 0),
                width=frame.get("width", 0),
                height=frame.get("height", 0),
                data=frame.get("data", b"")
            )
        except Exception as e:
            logger.error(f"Failed to parse scene camera message: {e}")
            return None
    
    async def get_device_info(self) -> Optional[Dict]:
        """
        Get device information from Neon.
        
        Returns:
            Dictionary with device information or None if failed
        """
        if not self.websocket:
            return None
            
        try:
            message = {"type": "get_device_info"}
            await self.websocket.send(json.dumps(message))
            
            response = await self.websocket.recv()
            data = json.loads(response)
            return data.get("device_info")
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return None
    
    async def start_recording(self, session_name: str = None) -> bool:
        """
        Start recording session on Neon device.
        
        Args:
            session_name: Optional name for the recording session
            
        Returns:
            True if recording started successfully, False otherwise
        """
        if not self.websocket:
            return False
            
        try:
            message = {
                "type": "start_recording",
                "session_name": session_name or f"session_{int(time.time())}"
            }
            await self.websocket.send(json.dumps(message))
            logger.info(f"Started recording: {message['session_name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    async def stop_recording(self) -> bool:
        """
        Stop recording session on Neon device.
        
        Returns:
            True if recording stopped successfully, False otherwise
        """
        if not self.websocket:
            return False
            
        try:
            message = {"type": "stop_recording"}
            await self.websocket.send(json.dumps(message))
            logger.info("Stopped recording")
            return True
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return False
