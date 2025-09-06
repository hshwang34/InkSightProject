"""
Configuration management for GazeLab.

Provides utilities for reading configuration overrides from files and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class ProcessingConfig(BaseModel):
    """Configuration for data processing parameters."""
    
    # Filtering
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    velocity_threshold_px_s: float = Field(default=100.0, gt=0.0)
    median_filter_window: int = Field(default=5, ge=1, le=50)
    
    # Fixation detection
    fixation_velocity_threshold_deg_s: float = Field(default=30.0, gt=0.0)
    fixation_min_duration_ms: float = Field(default=50.0, gt=0.0)
    fixation_max_duration_ms: float = Field(default=2000.0, gt=0.0)
    px_per_degree: float = Field(default=30.0, gt=0.0)  # Approximate for typical viewing distance
    
    # AOI analysis
    aoi_min_dwell_ms: float = Field(default=100.0, gt=0.0)
    aoi_gap_threshold_ms: float = Field(default=500.0, gt=0.0)


class VisualizationConfig(BaseModel):
    """Configuration for visualization parameters."""
    
    # Overlay settings
    dot_radius: int = Field(default=8, ge=1, le=50)
    dot_alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    trail_length: int = Field(default=10, ge=0, le=100)
    fixation_radius: int = Field(default=20, ge=1, le=100)
    fixation_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    
    # Heatmap settings
    heatmap_bandwidth: float = Field(default=20.0, gt=0.0)
    heatmap_grid_size: int = Field(default=100, ge=10, le=1000)
    heatmap_colormap: str = Field(default="hot")
    heatmap_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Video settings
    video_fps: float = Field(default=30.0, gt=0.0)
    video_codec: str = Field(default="mp4v")


class Config(BaseModel):
    """Main configuration class for GazeLab."""
    
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    
    # General settings
    log_level: str = Field(default="INFO")
    output_dir: str = Field(default="outputs")
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from a JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config_data = {}
        
        # Processing config
        processing_data = {}
        if os.getenv("GAZE_CONFIDENCE_THRESHOLD"):
            processing_data["confidence_threshold"] = float(os.getenv("GAZE_CONFIDENCE_THRESHOLD"))
        if os.getenv("GAZE_VELOCITY_THRESHOLD"):
            processing_data["velocity_threshold_px_s"] = float(os.getenv("GAZE_VELOCITY_THRESHOLD"))
        if os.getenv("GAZE_FIXATION_VEL_THRESHOLD"):
            processing_data["fixation_velocity_threshold_deg_s"] = float(os.getenv("GAZE_FIXATION_VEL_THRESHOLD"))
        
        if processing_data:
            config_data["processing"] = processing_data
        
        # Visualization config
        viz_data = {}
        if os.getenv("GAZE_DOT_RADIUS"):
            viz_data["dot_radius"] = int(os.getenv("GAZE_DOT_RADIUS"))
        if os.getenv("GAZE_TRAIL_LENGTH"):
            viz_data["trail_length"] = int(os.getenv("GAZE_TRAIL_LENGTH"))
        if os.getenv("GAZE_HEATMAP_BANDWIDTH"):
            viz_data["heatmap_bandwidth"] = float(os.getenv("GAZE_HEATMAP_BANDWIDTH"))
        
        if viz_data:
            config_data["visualization"] = viz_data
        
        # General settings
        if os.getenv("GAZE_LOG_LEVEL"):
            config_data["log_level"] = os.getenv("GAZE_LOG_LEVEL")
        if os.getenv("GAZE_OUTPUT_DIR"):
            config_data["output_dir"] = os.getenv("GAZE_OUTPUT_DIR")
        
        return cls(**config_data)
    
    def to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    def update(self, **kwargs: Any) -> "Config":
        """Create a new config with updated values."""
        current_data = self.model_dump()
        
        # Deep merge for nested configs
        for key, value in kwargs.items():
            if key in current_data and isinstance(current_data[key], dict) and isinstance(value, dict):
                current_data[key].update(value)
            else:
                current_data[key] = value
        
        return Config(**current_data)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file or environment."""
    if config_path:
        config = Config.from_file(config_path)
    else:
        config = Config.from_env()
    
    set_config(config)
    return config
