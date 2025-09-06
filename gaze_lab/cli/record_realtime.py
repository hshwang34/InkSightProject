"""
Real-time recording CLI tool.

Provides command-line interface for recording gaze data from real-time sources
including mock clients and optional Pupil Labs hardware.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..io.realtime import RealtimeClientFactory, RealtimeRecorder
from ..logging_setup import setup_logging


def main() -> None:
    """Main entry point for gaze-record command."""
    parser = argparse.ArgumentParser(
        description="Record gaze data from real-time sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record from mock client
  gaze-record --mode mock --gaze data/gaze.csv --world data/world.mp4 --out data/recording/
  
  # Record from Pupil Labs device (if available)
  gaze-record --mode pupil --host 127.0.0.1 --port 8080 --out data/recording/
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["mock", "pupil"],
        default="mock",
        help="Recording mode (default: mock)"
    )
    
    # Mock client options
    parser.add_argument(
        "--gaze",
        type=str,
        help="Path to gaze CSV file for mock client"
    )
    parser.add_argument(
        "--world",
        type=str,
        help="Path to world video file for mock client"
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed for mock client (default: 1.0)"
    )
    
    # Pupil client options
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Device host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Device port (default: 8080)"
    )
    
    # Output options
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for recorded data"
    )
    parser.add_argument(
        "--session-name",
        type=str,
        help="Session name for recording"
    )
    
    # General options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Recording duration in seconds (default: unlimited)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level)
    
    try:
        # Create output directory
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create real-time client
        if args.mode == "mock":
            if not args.gaze or not args.world:
                print("Error: Mock mode requires --gaze and --world arguments")
                sys.exit(1)
            
            client = RealtimeClientFactory.create_mock_client(
                gaze_file=args.gaze,
                world_video=args.world,
                playback_speed=args.playback_speed
            )
        elif args.mode == "pupil":
            try:
                client = RealtimeClientFactory.create_pupil_client(
                    host=args.host,
                    port=args.port
                )
            except ImportError as e:
                print(f"Error: {e}")
                print("Install Pupil Labs real-time client dependencies:")
                print("pip install pupil-labs-realtime-api")
                sys.exit(1)
        else:
            print(f"Error: Unknown mode: {args.mode}")
            sys.exit(1)
        
        # Create recorder
        recorder = RealtimeRecorder(client)
        
        # Get device info
        device_info = client.get_device_info()
        print(f"Device: {device_info.get('device_name', 'Unknown')}")
        print(f"Mode: {args.mode}")
        
        # Start recording
        print("Starting recording...")
        recorder.start_recording()
        
        # Record for specified duration or until interrupted
        if args.duration:
            import time
            time.sleep(args.duration)
            recorder.stop_recording()
        else:
            print("Recording... Press Ctrl+C to stop")
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping recording...")
                recorder.stop_recording()
        
        # Save recorded data
        gaze_data = recorder.get_gaze_data()
        annotations = recorder.get_annotations()
        
        if gaze_data:
            gaze_file = output_dir / "gaze.csv"
            recorder.save_gaze_data(str(gaze_file))
            print(f"Saved {len(gaze_data)} gaze samples to {gaze_file}")
        
        if annotations:
            annotation_file = output_dir / "annotations.json"
            recorder.save_annotations(str(annotation_file))
            print(f"Saved {len(annotations)} annotations to {annotation_file}")
        
        print("Recording complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()