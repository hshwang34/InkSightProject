"""
Reference mapping CLI tool.

Provides command-line interface for mapping gaze coordinates from world video
to reference image using homography computation.
"""

import argparse
import sys
from pathlib import Path

from ..io.cloud_loader import load_gaze
from ..processing.mapping_2d import map_gaze_from_video_to_reference
from ..logging_setup import setup_logging


def main() -> None:
    """Main entry point for gaze-map command."""
    parser = argparse.ArgumentParser(
        description="Map gaze coordinates to reference image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic mapping
  gaze-map --world data/world.mp4 --gaze data/gaze.csv \\
           --reference examples/reference_example.png --out outputs/mapped_gaze.csv
  
  # Mapping with specific frame
  gaze-map --world data/world.mp4 --gaze data/gaze.csv \\
           --reference examples/reference_example.png --out outputs/mapped_gaze.csv \\
           --frame-index 100
  
  # Mapping with SIFT features
  gaze-map --world data/world.mp4 --gaze data/gaze.csv \\
           --reference examples/reference_example.png --out outputs/mapped_gaze.csv \\
           --method sift
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
        "--reference",
        type=str,
        required=True,
        help="Path to reference image file"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for mapped gaze CSV"
    )
    
    # Mapping options
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to use for homography computation (default: 0)"
    )
    parser.add_argument(
        "--method",
        choices=["orb", "sift"],
        default="orb",
        help="Feature detection method (default: orb)"
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
        reference_image_path = Path(args.reference)
        
        if not world_video_path.exists():
            print(f"Error: World video not found: {world_video_path}")
            sys.exit(1)
        
        if not gaze_file_path.exists():
            print(f"Error: Gaze file not found: {gaze_file_path}")
            sys.exit(1)
        
        if not reference_image_path.exists():
            print(f"Error: Reference image not found: {reference_image_path}")
            sys.exit(1)
        
        # Load gaze data
        print(f"Loading gaze data from {gaze_file_path}")
        gaze_df = load_gaze(gaze_file_path)
        print(f"Loaded {len(gaze_df)} gaze samples")
        
        # Map gaze coordinates
        print(f"Mapping gaze coordinates to reference image...")
        print(f"Using frame {args.frame_index} for homography computation")
        print(f"Feature detection method: {args.method}")
        
        mapped_df, homography = map_gaze_from_video_to_reference(
            gaze_df=gaze_df,
            world_video_path=world_video_path,
            reference_image_path=reference_image_path,
            frame_index=args.frame_index,
            method=args.method,
        )
        
        if homography is None:
            print("Error: Failed to compute homography")
            sys.exit(1)
        
        # Save mapped gaze data
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mapped_df.to_csv(output_path, index=False)
        
        print(f"Saved mapped gaze data to {output_path}")
        print(f"Mapped {len(mapped_df)} gaze points")
        
        # Print statistics
        original_count = len(gaze_df)
        mapped_count = len(mapped_df)
        success_rate = (mapped_count / original_count * 100) if original_count > 0 else 0
        
        print(f"\nMapping Statistics:")
        print(f"Original gaze points: {original_count}")
        print(f"Mapped gaze points: {mapped_count}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if len(mapped_df) > 0:
            print(f"Reference image size: {mapped_df['ref_w'].iloc[0]}x{mapped_df['ref_h'].iloc[0]}")
            print(f"Mapped X range: {mapped_df['rx_px'].min():.1f} - {mapped_df['rx_px'].max():.1f}")
            print(f"Mapped Y range: {mapped_df['ry_px'].min():.1f} - {mapped_df['ry_px'].max():.1f}")
        
        print("Mapping complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
