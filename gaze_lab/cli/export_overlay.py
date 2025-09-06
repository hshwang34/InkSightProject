"""
Gaze overlay export CLI tool.

Provides command-line interface for creating gaze overlay videos
from recorded gaze data and world video.
"""

import argparse
import sys
from pathlib import Path

from ..io.cloud_loader import load_gaze
from ..processing.fixations_ivt import detect_fixations_ivt
from ..viz.overlay import create_gaze_overlay
from ..logging_setup import setup_logging


def main() -> None:
    """Main entry point for gaze-overlay command."""
    parser = argparse.ArgumentParser(
        description="Create gaze overlay video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic overlay
  gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4
  
  # Overlay with fixations
  gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4 \\
               --show-fixations --fixation-radius 25
  
  # Custom appearance
  gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4 \\
               --dot-radius 12 --trail 20 --dot-alpha 0.9
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--world",
        type=str,
        required=True,
        help="Path to world video file"
    )
    parser.add_argument(
        "--gaze",
        type=str,
        required=True,
        help="Path to gaze CSV file"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output video path"
    )
    
    # Gaze appearance
    parser.add_argument(
        "--dot-radius",
        type=int,
        default=8,
        help="Radius of gaze dots (default: 8)"
    )
    parser.add_argument(
        "--dot-alpha",
        type=float,
        default=0.8,
        help="Transparency of gaze dots (default: 0.8)"
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=10,
        help="Length of gaze trail (default: 10)"
    )
    
    # Fixation options
    parser.add_argument(
        "--show-fixations",
        action="store_true",
        help="Show fixation markers"
    )
    parser.add_argument(
        "--fixation-radius",
        type=int,
        default=20,
        help="Radius of fixation circles (default: 20)"
    )
    parser.add_argument(
        "--fixation-alpha",
        type=float,
        default=0.6,
        help="Transparency of fixation circles (default: 0.6)"
    )
    
    # Video options
    parser.add_argument(
        "--fps",
        type=float,
        help="Output video FPS (uses input FPS if not specified)"
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="Video codec (default: mp4v)"
    )
    
    # Processing options
    parser.add_argument(
        "--detect-fixations",
        action="store_true",
        help="Detect fixations from gaze data"
    )
    parser.add_argument(
        "--fixation-velocity-threshold",
        type=float,
        default=30.0,
        help="Fixation velocity threshold in deg/s (default: 30.0)"
    )
    parser.add_argument(
        "--fixation-min-duration",
        type=float,
        default=50.0,
        help="Minimum fixation duration in ms (default: 50.0)"
    )
    
    # General options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level)
    
    try:
        # Validate input files
        world_video_path = Path(args.world)
        gaze_file_path = Path(args.gaze)
        
        if not world_video_path.exists():
            print(f"Error: World video not found: {world_video_path}")
            sys.exit(1)
        
        if not gaze_file_path.exists():
            print(f"Error: Gaze file not found: {gaze_file_path}")
            sys.exit(1)
        
        # Load gaze data
        print(f"Loading gaze data from {gaze_file_path}")
        gaze_df = load_gaze(gaze_file_path)
        print(f"Loaded {len(gaze_df)} gaze samples")
        
        # Detect fixations if requested
        fixation_df = None
        if args.detect_fixations or args.show_fixations:
            print("Detecting fixations...")
            fixation_df = detect_fixations_ivt(
                gaze_df,
                velocity_threshold_deg_s=args.fixation_velocity_threshold,
                min_duration_ms=args.fixation_min_duration,
            )
            print(f"Detected {len(fixation_df)} fixations")
        
        # Create overlay
        print(f"Creating gaze overlay video...")
        create_gaze_overlay(
            world_video_path=world_video_path,
            gaze_df=gaze_df,
            output_path=args.out,
            fixation_df=fixation_df,
            dot_radius=args.dot_radius,
            dot_alpha=args.dot_alpha,
            trail_length=args.trail,
            show_fixations=args.show_fixations,
            fixation_radius=args.fixation_radius,
            fixation_alpha=args.fixation_alpha,
            video_fps=args.fps,
            video_codec=args.codec,
        )
        
        print(f"Created gaze overlay video: {args.out}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()