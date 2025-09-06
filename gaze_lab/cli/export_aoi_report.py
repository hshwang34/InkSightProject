"""
AOI report export CLI tool.

Provides command-line interface for generating AOI analysis reports
from gaze data and AOI definitions.
"""

import argparse
import sys
from pathlib import Path

from ..io.cloud_loader import load_gaze
from ..processing.aoi import AOIAnalyzer
from ..processing.fixations_ivt import detect_fixations_ivt
from ..logging_setup import setup_logging


def main() -> None:
    """Main entry point for gaze-aoi command."""
    parser = argparse.ArgumentParser(
        description="Generate AOI analysis report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic AOI analysis
  gaze-aoi --gaze data/gaze.csv --aoi examples/aoi_example.json \\
           --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv
  
  # AOI analysis with fixations
  gaze-aoi --gaze data/gaze.csv --aoi examples/aoi_example.json \\
           --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv \\
           --detect-fixations --fixation-file outputs/fixations.csv
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
        "--aoi",
        type=str,
        required=True,
        help="Path to AOI JSON file"
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Output path for AOI report CSV"
    )
    parser.add_argument(
        "--timeline",
        type=str,
        required=True,
        help="Output path for AOI timeline CSV"
    )
    
    # AOI analysis options
    parser.add_argument(
        "--min-dwell-ms",
        type=float,
        default=100.0,
        help="Minimum dwell time in milliseconds (default: 100.0)"
    )
    parser.add_argument(
        "--gap-threshold-ms",
        type=float,
        default=500.0,
        help="Maximum gap between samples to consider continuous (default: 500.0)"
    )
    
    # Fixation options
    parser.add_argument(
        "--detect-fixations",
        action="store_true",
        help="Detect fixations from gaze data"
    )
    parser.add_argument(
        "--fixation-file",
        type=str,
        help="Path to existing fixation CSV file"
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
        gaze_file_path = Path(args.gaze)
        aoi_file_path = Path(args.aoi)
        
        if not gaze_file_path.exists():
            print(f"Error: Gaze file not found: {gaze_file_path}")
            sys.exit(1)
        
        if not aoi_file_path.exists():
            print(f"Error: AOI file not found: {aoi_file_path}")
            sys.exit(1)
        
        # Load gaze data
        print(f"Loading gaze data from {gaze_file_path}")
        gaze_df = load_gaze(gaze_file_path)
        print(f"Loaded {len(gaze_df)} gaze samples")
        
        # Load AOIs
        print(f"Loading AOIs from {aoi_file_path}")
        aoi_analyzer = AOIAnalyzer()
        aoi_analyzer.load_aois_from_file(aoi_file_path)
        print(f"Loaded {len(aoi_analyzer.aois)} AOIs")
        
        # Load or detect fixations
        fixation_df = None
        if args.fixation_file:
            print(f"Loading fixations from {args.fixation_file}")
            fixation_df = load_gaze(args.fixation_file)
            print(f"Loaded {len(fixation_df)} fixations")
        elif args.detect_fixations:
            print("Detecting fixations...")
            fixation_df = detect_fixations_ivt(
                gaze_df,
                velocity_threshold_deg_s=args.fixation_velocity_threshold,
                min_duration_ms=args.fixation_min_duration,
            )
            print(f"Detected {len(fixation_df)} fixations")
        
        # Analyze gaze data against AOIs
        print("Analyzing gaze data against AOIs...")
        aoi_hits = aoi_analyzer.analyze_gaze_data(
            gaze_df,
            min_dwell_ms=args.min_dwell_ms,
            gap_threshold_ms=args.gap_threshold_ms,
        )
        
        # Calculate dwell times
        print("Calculating dwell times...")
        aoi_dwells = aoi_analyzer.calculate_dwell_times(
            aoi_hits,
            min_dwell_ms=args.min_dwell_ms,
            gap_threshold_ms=args.gap_threshold_ms,
        )
        
        # Generate reports
        print("Generating AOI report...")
        aoi_report = aoi_analyzer.generate_aoi_report(
            aoi_hits, aoi_dwells, fixation_df
        )
        
        print("Generating AOI timeline...")
        aoi_timeline = aoi_analyzer.generate_aoi_timeline(aoi_dwells)
        
        # Save reports
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        aoi_report.to_csv(report_path, index=False)
        print(f"Saved AOI report to {report_path}")
        
        timeline_path = Path(args.timeline)
        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        aoi_timeline.to_csv(timeline_path, index=False)
        print(f"Saved AOI timeline to {timeline_path}")
        
        # Print summary
        print("\nAOI Analysis Summary:")
        print(f"Total AOIs: {len(aoi_analyzer.aois)}")
        print(f"Total dwells: {len(aoi_timeline)}")
        print(f"Total dwell time: {aoi_timeline['dwell_ms'].sum():.1f} ms")
        
        if len(aoi_report) > 0:
            print(f"Most visited AOI: {aoi_report.loc[aoi_report['total_dwell_time_ms'].idxmax(), 'aoi_name']}")
            print(f"Longest dwell: {aoi_timeline['dwell_ms'].max():.1f} ms")
        
        print("AOI analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()