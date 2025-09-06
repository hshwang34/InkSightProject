"""
Synthetic demo data generator.

Generates realistic synthetic gaze data and world video for testing
and demonstration purposes.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaze_lab.logging_setup import setup_logging


def generate_synthetic_world_video(
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    duration: float = 5.0,
    fps: float = 30.0,
) -> None:
    """Generate synthetic world video with moving elements."""
    print(f"Generating synthetic world video: {output_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for frame_idx in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add grid pattern
        for i in range(0, width, 100):
            cv2.line(frame, (i, 0), (i, height), (200, 200, 200), 1)
        for i in range(0, height, 100):
            cv2.line(frame, (0, i), (width, i), (200, 200, 200), 1)
        
        # Add moving square
        t = frame_idx / total_frames
        square_size = 80
        square_x = int(width * 0.2 + (width * 0.6) * t)
        square_y = int(height * 0.3 + (height * 0.4) * np.sin(t * 4 * np.pi))
        
        # Draw square
        cv2.rectangle(
            frame,
            (square_x - square_size//2, square_y - square_size//2),
            (square_x + square_size//2, square_y + square_size//2),
            (0, 100, 200),
            -1
        )
        
        # Add text
        cv2.putText(
            frame,
            f"Frame {frame_idx + 1}/{total_frames}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        
        # Add timestamp
        timestamp = frame_idx / fps
        cv2.putText(
            frame,
            f"Time: {timestamp:.2f}s",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )
        
        out.write(frame)
    
    out.release()
    print(f"Generated {total_frames} frames at {fps} FPS")


def generate_synthetic_gaze_data(
    output_path: Path,
    world_video_path: Path,
    width: int = 1280,
    height: int = 720,
    duration: float = 5.0,
    fps: float = 30.0,
    sample_rate: float = 60.0,
) -> None:
    """Generate synthetic gaze data that follows the moving square."""
    print(f"Generating synthetic gaze data: {output_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate parameters
    total_samples = int(duration * sample_rate)
    sample_interval_ns = int(1_000_000_000 / sample_rate)
    
    # Generate gaze data
    gaze_data = []
    
    for sample_idx in range(total_samples):
        t = sample_idx / total_samples
        
        # Calculate target position (following the moving square)
        target_x = width * 0.2 + (width * 0.6) * t
        target_y = height * 0.3 + (height * 0.4) * np.sin(t * 4 * np.pi)
        
        # Add realistic gaze behavior
        # 1. Saccades to target
        if sample_idx % 60 == 0:  # Saccade every second
            # Add saccade with some overshoot
            saccade_overshoot = 20
            gaze_x = target_x + np.random.normal(0, saccade_overshoot)
            gaze_y = target_y + np.random.normal(0, saccade_overshoot)
        else:
            # Fixation with micro-movements
            gaze_x = target_x + np.random.normal(0, 5)
            gaze_y = target_y + np.random.normal(0, 5)
        
        # Add some noise and blinks
        if np.random.random() < 0.05:  # 5% blink rate
            gaze_x = np.nan
            gaze_y = np.nan
        
        # Ensure coordinates are within bounds
        if not np.isnan(gaze_x):
            gaze_x = max(0, min(width, gaze_x))
        if not np.isnan(gaze_y):
            gaze_y = max(0, min(height, gaze_y))
        
        # Calculate timestamp
        timestamp_ns = sample_idx * sample_interval_ns
        
        gaze_data.append({
            't_ns': timestamp_ns,
            'gx_px': gaze_x,
            'gy_px': gaze_y,
            'frame_w': width,
            'frame_h': height,
        })
    
    # Create DataFrame
    gaze_df = pd.DataFrame(gaze_data)
    
    # Remove NaN values (blinks)
    gaze_df = gaze_df.dropna()
    
    # Save to CSV
    gaze_df.to_csv(output_path, index=False)
    print(f"Generated {len(gaze_df)} gaze samples")


def generate_reference_image(
    output_path: Path,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Generate reference image for mapping."""
    print(f"Generating reference image: {output_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create image
    image = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Add grid pattern
    for i in range(0, width, 100):
        cv2.line(image, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, 100):
        cv2.line(image, (0, i), (width, i), (200, 200, 200), 1)
    
    # Add reference elements
    cv2.rectangle(image, (100, 100), (300, 200), (0, 100, 200), -1)
    cv2.circle(image, (width//2, height//2), 50, (200, 100, 0), -1)
    cv2.rectangle(image, (width-200, height-200), (width-100, height-100), (100, 200, 0), -1)
    
    # Add text
    cv2.putText(image, "Reference Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Save image
    cv2.imwrite(str(output_path), image)
    print(f"Generated reference image: {output_path}")


def generate_aoi_example(
    output_path: Path,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Generate example AOI configuration."""
    print(f"Generating AOI example: {output_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define AOIs
    aois = [
        {
            "name": "top_left_square",
            "type": "rectangle",
            "coordinates": [(100, 100), (300, 100), (300, 200), (100, 200)],
            "metadata": {"description": "Top-left reference square"}
        },
        {
            "name": "center_circle",
            "type": "circle",
            "coordinates": [(width//2, height//2, 50)],
            "metadata": {"description": "Center reference circle"}
        },
        {
            "name": "bottom_right_square",
            "type": "rectangle",
            "coordinates": [(width-200, height-200), (width-100, height-200), (width-100, height-100), (width-200, height-100)],
            "metadata": {"description": "Bottom-right reference square"}
        }
    ]
    
    # Create JSON structure
    aoi_config = {
        "aois": aois,
        "metadata": {
            "created_by": "GazeLab",
            "description": "Example AOI configuration for synthetic data",
            "frame_size": {"width": width, "height": height}
        }
    }
    
    # Save to JSON
    import json
    with open(output_path, 'w') as f:
        json.dump(aoi_config, f, indent=2)
    
    print(f"Generated AOI example: {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo data for GazeLab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all demo data
  python -m scripts.make_synthetic_demo
  
  # Generate specific components
  python -m scripts.make_synthetic_demo --video-only
  python -m scripts.make_synthetic_demo --gaze-only
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)"
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="examples",
        help="Examples directory (default: examples)"
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Generate only video data"
    )
    parser.add_argument(
        "--gaze-only",
        action="store_true",
        help="Generate only gaze data"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Video duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS (default: 30.0)"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=60.0,
        help="Gaze sample rate in Hz (default: 60.0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height (default: 720)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level="INFO")
    
    # Create directories
    data_dir = Path(args.output_dir)
    examples_dir = Path(args.examples_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate world video
        if not args.gaze_only:
            world_video_path = data_dir / "world.mp4"
            generate_synthetic_world_video(
                world_video_path,
                width=args.width,
                height=args.height,
                duration=args.duration,
                fps=args.fps,
            )
        
        # Generate gaze data
        if not args.video_only:
            gaze_csv_path = data_dir / "gaze.csv"
            world_video_path = data_dir / "world.mp4"
            generate_synthetic_gaze_data(
                gaze_csv_path,
                world_video_path,
                width=args.width,
                height=args.height,
                duration=args.duration,
                fps=args.fps,
                sample_rate=args.sample_rate,
            )
        
        # Generate reference image
        if not args.video_only and not args.gaze_only:
            reference_image_path = examples_dir / "reference_example.png"
            generate_reference_image(
                reference_image_path,
                width=args.width,
                height=args.height,
            )
        
        # Generate AOI example
        if not args.video_only and not args.gaze_only:
            aoi_example_path = examples_dir / "aoi_example.json"
            generate_aoi_example(
                aoi_example_path,
                width=args.width,
                height=args.height,
            )
        
        print("\nSynthetic demo data generation complete!")
        print(f"Data directory: {data_dir}")
        print(f"Examples directory: {examples_dir}")
        
        if not args.gaze_only:
            print(f"World video: {data_dir / 'world.mp4'}")
        
        if not args.video_only:
            print(f"Gaze data: {data_dir / 'gaze.csv'}")
            print(f"Reference image: {examples_dir / 'reference_example.png'}")
            print(f"AOI example: {examples_dir / 'aoi_example.json'}")
        
        print("\nYou can now run the CLI tools:")
        if not args.gaze_only:
            print(f"gaze-overlay --world {data_dir / 'world.mp4'} --gaze {data_dir / 'gaze.csv'} --out outputs/overlay.mp4")
        if not args.video_only:
            print(f"gaze-aoi --gaze {data_dir / 'gaze.csv'} --aoi {examples_dir / 'aoi_example.json'} --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv")
            print(f"gaze-heatmap --gaze {data_dir / 'gaze.csv'} --out outputs/heatmap.png --mode world --world {data_dir / 'world.mp4'}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
