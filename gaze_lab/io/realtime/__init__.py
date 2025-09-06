"""Real-time interfaces for gaze data acquisition."""

from .interfaces import GazeSample, Annotation, RealtimeClient
from .mock_client import MockRealtimeClient

# Try to import pupil client, but don't fail if not available
try:
    from .pupil_client import PupilRealtimeClient
    __all__ = ["GazeSample", "Annotation", "RealtimeClient", "MockRealtimeClient", "PupilRealtimeClient"]
except ImportError:
    __all__ = ["GazeSample", "Annotation", "RealtimeClient", "MockRealtimeClient"]
