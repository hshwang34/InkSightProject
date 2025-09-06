"""
Pupil Labs real-time client adapter.

Optional adapter for the official Pupil Labs real-time Python client.
Import is guarded with try/except; if missing, raises clear ImportError.
"""

from typing import Callable, Optional

from .interfaces import GazeSample, Annotation, RealtimeClient
from ...logging_setup import get_logger

logger = get_logger(__name__)

# Try to import pupil client dependencies
try:
    import pupil_labs.realtime_api as pupil_api
    from pupil_labs.realtime_api import Device, StatusUpdateNotifier
    PUPIL_AVAILABLE = True
except ImportError:
    PUPIL_AVAILABLE = False
    pupil_api = None
    Device = None
    StatusUpdateNotifier = None


class PupilRealtimeClient(RealtimeClient):
    """Real-time client for Pupil Labs devices."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        """
        Initialize Pupil Labs real-time client.
        
        Args:
            host: Device host address
            port: Device port
            
        Raises:
            ImportError: If pupil client dependencies are not available
        """
        if not PUPIL_AVAILABLE:
            raise ImportError(
                "Pupil Labs real-time client dependencies not available. "
                "Install with: pip install pupil-labs-realtime-api"
            )
        
        self.host = host
        self.port = port
        self.device: Optional[Device] = None
        self.status_notifier: Optional[StatusUpdateNotifier] = None
        
        self._running = False
        self._gaze_callback: Optional[Callable[[GazeSample], None]] = None
        self._annotation_callback: Optional[Callable[[Annotation], None]] = None
        
        logger.info(f"Initialized Pupil client for {host}:{port}")
    
    def subscribe_gaze(self, callback: Callable[[GazeSample], None]) -> None:
        """Subscribe to gaze data stream."""
        self._gaze_callback = callback
        logger.info("Subscribed to gaze stream")
    
    def subscribe_annotations(self, callback: Callable[[Annotation], None]) -> None:
        """Subscribe to annotation stream."""
        self._annotation_callback = callback
        logger.info("Subscribed to annotation stream")
    
    def start(self) -> None:
        """Start the Pupil client."""
        if self._running:
            logger.warning("Client is already running")
            return
        
        try:
            # Create device connection
            self.device = Device(address=f"{self.host}:{self.port}")
            
            # Set up status notifier
            self.status_notifier = StatusUpdateNotifier(self.device)
            self.status_notifier.add_observer(self._on_status_update)
            
            # Start the device
            self.device.start()
            
            # Set up gaze data subscription
            if self._gaze_callback:
                self.device.gaze.subscribe(self._on_gaze_data)
            
            # Set up annotation subscription
            if self._annotation_callback:
                self.device.annotation.subscribe(self._on_annotation)
            
            self._running = True
            logger.info("Started Pupil real-time client")
            
        except Exception as e:
            logger.error(f"Failed to start Pupil client: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the Pupil client."""
        if not self._running:
            logger.warning("Client is not running")
            return
        
        try:
            if self.device:
                self.device.stop()
            
            if self.status_notifier:
                self.status_notifier.remove_observer(self._on_status_update)
            
            self._running = False
            logger.info("Stopped Pupil real-time client")
            
        except Exception as e:
            logger.error(f"Error stopping Pupil client: {e}")
    
    def is_running(self) -> bool:
        """Check if client is running."""
        return self._running and self.device is not None and self.device.is_connected
    
    def get_device_info(self) -> dict:
        """Get device information."""
        if not self.device:
            return {"error": "Device not connected"}
        
        try:
            info = {
                "device_type": "pupil_labs",
                "host": self.host,
                "port": self.port,
                "connected": self.device.is_connected,
            }
            
            # Get device status if available
            if hasattr(self.device, 'status'):
                status = self.device.status
                if status:
                    info.update({
                        "device_name": getattr(status, 'device_name', 'Unknown'),
                        "version": getattr(status, 'version', 'Unknown'),
                        "recording": getattr(status, 'recording', False),
                    })
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return {"error": str(e)}
    
    def _on_gaze_data(self, gaze_data) -> None:
        """Handle incoming gaze data."""
        if not self._gaze_callback:
            return
        
        try:
            # Convert pupil gaze data to our format
            sample = GazeSample(
                t_ns=int(gaze_data.timestamp * 1_000_000_000),  # Convert to nanoseconds
                gx_px=float(gaze_data.gaze_point_3d.x),
                gy_px=float(gaze_data.gaze_point_3d.y),
                frame_w=int(gaze_data.world_camera_frame_size.width),
                frame_h=int(gaze_data.world_camera_frame_size.height),
                confidence=float(gaze_data.confidence),
                pupil_diameter=getattr(gaze_data, 'pupil_diameter', None),
                eye_id=getattr(gaze_data, 'eye_id', None),
            )
            
            self._gaze_callback(sample)
            
        except Exception as e:
            logger.error(f"Error processing gaze data: {e}")
    
    def _on_annotation(self, annotation_data) -> None:
        """Handle incoming annotation data."""
        if not self._annotation_callback:
            return
        
        try:
            # Convert pupil annotation to our format
            annotation = Annotation(
                t_ns=int(annotation_data.timestamp * 1_000_000_000),  # Convert to nanoseconds
                label=str(annotation_data.label),
                data=getattr(annotation_data, 'data', {}),
            )
            
            self._annotation_callback(annotation)
            
        except Exception as e:
            logger.error(f"Error processing annotation: {e}")
    
    def _on_status_update(self, status) -> None:
        """Handle device status updates."""
        logger.info(f"Device status update: {status}")
    
    def start_recording(self, session_name: Optional[str] = None) -> bool:
        """
        Start recording on the device.
        
        Args:
            session_name: Optional session name
            
        Returns:
            True if recording started successfully
        """
        if not self.device:
            logger.error("Device not connected")
            return False
        
        try:
            if session_name:
                self.device.recording.start(session_name)
            else:
                self.device.recording.start()
            
            logger.info(f"Started recording: {session_name or 'default'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self) -> bool:
        """
        Stop recording on the device.
        
        Returns:
            True if recording stopped successfully
        """
        if not self.device:
            logger.error("Device not connected")
            return False
        
        try:
            self.device.recording.stop()
            logger.info("Stopped recording")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return False
    
    def send_annotation(self, label: str, data: Optional[dict] = None) -> bool:
        """
        Send annotation to the device.
        
        Args:
            label: Annotation label
            data: Optional annotation data
            
        Returns:
            True if annotation sent successfully
        """
        if not self.device:
            logger.error("Device not connected")
            return False
        
        try:
            if data:
                self.device.annotation.send(label, data)
            else:
                self.device.annotation.send(label)
            
            logger.info(f"Sent annotation: {label}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send annotation: {e}")
            return False


def check_pupil_availability() -> bool:
    """
    Check if Pupil Labs real-time client is available.
    
    Returns:
        True if available, False otherwise
    """
    return PUPIL_AVAILABLE


def get_pupil_version() -> Optional[str]:
    """
    Get Pupil Labs client version if available.
    
    Returns:
        Version string or None if not available
    """
    if not PUPIL_AVAILABLE:
        return None
    
    try:
        return pupil_api.__version__
    except AttributeError:
        return "unknown"
