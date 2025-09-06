"""
Heatmap export CLI tool.

Provides command-line interface for generating gaze heatmaps
from gaze data with various visualization options.
"""

import argparse
import sys
from pathlib import Path

from ..io.cloud_loader import load_gaze
from ..viz.heatmap import create_heatmap, create_heatmap_from_mapped_gaze
from ..logging_setup import setup_logging


def main() -> None:
    """Main entry point for gaze-heatmap command."""
    parser = argparse.ArgumentParser(
        description="Generate gaze heatmap visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic heatmap
  gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png
  
  # Heatmap with world video background
  gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png \\
               --mode world --world data/world.mp4 --background-frame 0
  
  # Heatmap on reference image
  gaze-heatmap --gaze outputs/mapped_gaze.csv --out outputs/ref_heatmap.png \\
               --mode reference --reference examples/reference_example.png
  
  # Custom heatmap appearance
  gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png \\
               --bandwidth 30 --grid-size 200 --colormap viridis --alpha 0.8
        """
    )
    
    # Required arguments
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
        help="Output image path"
    )
    
    # Background mode
    parser.add_argument(
        "--mode",
        choices=["world", "reference", "none"],
        default="none",
        help="Background mode (default: none)"
    )
    parser.add_argument(
        "--world",
        type=str,
        help="Path to world video file (for world mode)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Path to reference image file (for reference mode)"
    )
    parser.add_argument(
        "--background-frame",
        type=int,
        default=0,
        help="Frame index for world video background (default: 0)"
    )
    
    # Heatmap appearance
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=20.0,
        help="KDE bandwidth parameter (default: 20.0)"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Grid resolution for heatmap (default: 100)"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="hot",
        help="Matplotlib colormap name (default: hot)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Transparency of heatmap overlay (default: 0.7)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI (default: 300)"
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
        gaze_file_path = Path(args.gaze)
        
        if not gaze_file_path.exists():
            print(f"Error: Gaze file not found: {gaze_file_path}")
            sys.exit(1)
        
        # Load gaze data
        print(f"Loading gaze data from {gaze_file_path}")
        gaze_df = load_gaze(gaze_file_path)
        print(f"Loaded {len(gaze_df)} gaze samples")
        
        # Validate mode-specific arguments
        if args.mode == "world" and not args.world:
            print("Error: World mode requires --world argument")
            sys.exit(1)
        
        if args.mode == "reference" and not args.reference:
            print("Error: Reference mode requires --reference argument")
            sys.exit(1)
        
        # Create heatmap
        print(f"Creating heatmap in {args.mode} mode...")
        
        if args.mode == "reference":
            # Use mapped gaze data on reference image
            reference_path = Path(args.reference)
            if not reference_path.exists():
                print(f"Error: Reference image not found: {reference_path}")
                sys.exit(1)
            
            create_heatmap_from_mapped_gaze(
                mapped_gaze_df=gaze_df,
                reference_image_path=reference_path,
                output_path=args.out,
                bandwidth=args.bandwidth,
                grid_size=args.grid_size,
                colormap=args.colormap,
                alpha=args.alpha,
                dpi=args.dpi,
            )
        else:
            # Use regular heatmap creation
            world_video_path = None
            if args.mode == "world":
                world_video_path = Path(args.world)
                if not world_video_path.exists():
                    print(f"Error: World video not found: {world_video_path}")
                    sys.exit(1)
            
            create_heatmap(
                gaze_df=gaze_df,
                output_path=args.out,
                background_mode=args.mode,
                background_frame_index=args.background_frame,
                world_video_path=world_video_path,
                bandwidth=args.bandwidth,
                grid_size=args.grid_size,
                colormap=args.colormap,
                alpha=args.alpha,
                dpi=args.dpi,
            )
        
        print(f"Created heatmap: {args.out}")
        
        # Print statistics
        from ..viz.heatmap import get_heatmap_statistics
        stats = get_heatmap_statistics(gaze_df)
        print(f"\nHeatmap Statistics:")
        print(f"Gaze points: {stats['gaze_points']}")
        print(f"X range: {stats['x_range'][0]:.1f} - {stats['x_range'][1]:.1f}")
        print(f"Y range: {stats['y_range'][0]:.1f} - {stats['y_range'][1]:.1f}")
        print(f"Mean distance from center: {stats['mean_distance_from_center']:.1f} px")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()