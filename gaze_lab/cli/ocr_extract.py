"""
CLI tool for extracting question text from gaze data using OCR.

Provides functionality to extract video frames at specific timestamps or
peak dwell times within AOIs, then perform OCR to extract text content.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from ..io.cloud_loader import load_gaze_data
from ..logging_setup import setup_logging, get_logger
from ..ocr.engine import recognize_text, is_ocr_available, get_available_engines
from ..ocr.preprocess import prep_for_ocr
from ..processing.aoi import load_aoi_config, peak_dwell_timestamp
from ..snapshots.extractor import extract_frame_at_ts, crop_aoi, warp_to_reference, validate_timestamp

logger = get_logger(__name__)


def main() -> None:
    """Main entry point for gaze-ocr command."""
    parser = argparse.ArgumentParser(
        description="Extract question text from gaze data using OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract text at specific timestamp
  gaze-ocr --world data/world.mp4 --gaze data/gaze.csv \\
           --aoi examples/aoi_example.json --aoi-name "Question" \\
           --t-ns 12500000000 --out-dir outputs/ocr

  # Find peak dwell time automatically
  gaze-ocr --world data/world.mp4 --gaze data/gaze.csv \\
           --aoi examples/aoi_example.json --aoi-name "Question" \\
           --select peak-dwell --window-ms 5000 --out-dir outputs/ocr

  # Use reference image warping
  gaze-ocr --world data/world.mp4 --gaze data/gaze.csv \\
           --aoi examples/aoi_example.json --aoi-name "Question" \\
           --reference examples/reference_example.png \\
           --select peak-dwell --out-dir outputs/ocr
        """
    )
    
    # Input files
    parser.add_argument(
        "--world",
        type=str,
        required=True,
        help="Path to world camera video file"
    )
    parser.add_argument(
        "--gaze",
        type=str,
        required=True,
        help="Path to gaze data CSV file"
    )
    parser.add_argument(
        "--aoi",
        type=str,
        required=True,
        help="Path to AOI configuration JSON file"
    )
    parser.add_argument(
        "--aoi-name",
        type=str,
        required=True,
        help="Name of AOI to extract from"
    )
    
    # Timestamp selection
    timestamp_group = parser.add_mutually_exclusive_group(required=True)
    timestamp_group.add_argument(
        "--t-ns",
        type=int,
        help="Explicit timestamp in nanoseconds"
    )
    timestamp_group.add_argument(
        "--select",
        choices=["peak-dwell"],
        help="Automatic timestamp selection method"
    )
    
    # Optional reference image
    parser.add_argument(
        "--reference",
        type=str,
        help="Path to reference image for homography warping"
    )
    
    # OCR parameters
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help="OCR language code(s), comma-separated (default: eng)"
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "tesseract", "easyocr"],
        default="auto",
        help="OCR engine preference (default: auto)"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "doc", "ui"],
        default="auto",
        help="Image preprocessing mode (default: auto)"
    )
    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Attempt to deskew rotated text"
    )
    
    # Peak dwell parameters
    parser.add_argument(
        "--window-ms",
        type=int,
        default=5000,
        help="Window size for peak dwell selection in milliseconds (default: 5000)"
    )
    
    # Output
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    # General options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(level=log_level)
    
    try:
        # Check OCR availability
        if not is_ocr_available():
            print("❌ No OCR engines available!")
            print("\nInstall OCR dependencies:")
            print("  pip install 'gaze-lab[ocr]'")
            print("\nFor Tesseract (recommended):")
            print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("  macOS: brew install tesseract")
            print("  Linux: sudo apt-get install tesseract-ocr")
            sys.exit(1)
        
        available_engines = get_available_engines()
        logger.info(f"Available OCR engines: {available_engines}")
        
        # Validate input files
        world_path = Path(args.world)
        if not world_path.exists():
            print(f"❌ World video file not found: {world_path}")
            sys.exit(1)
        
        gaze_path = Path(args.gaze)
        if not gaze_path.exists():
            print(f"❌ Gaze data file not found: {gaze_path}")
            sys.exit(1)
        
        aoi_path = Path(args.aoi)
        if not aoi_path.exists():
            print(f"❌ AOI configuration file not found: {aoi_path}")
            sys.exit(1)
        
        # Load data
        logger.info("Loading gaze data...")
        gaze_df = load_gaze_data(str(gaze_path))
        logger.info(f"Loaded {len(gaze_df)} gaze samples")
        
        logger.info("Loading AOI configuration...")
        aoi_config = load_aoi_config(str(aoi_path))
        
        # Find target AOI
        target_aoi = None
        for aoi in aoi_config.get("aois", []):
            if aoi.get("name") == args.aoi_name:
                target_aoi = aoi
                break
        
        if target_aoi is None:
            print(f"❌ AOI '{args.aoi_name}' not found in configuration")
            available_aois = [aoi.get("name", "unnamed") for aoi in aoi_config.get("aois", [])]
            print(f"Available AOIs: {available_aois}")
            sys.exit(1)
        
        logger.info(f"Found target AOI: {args.aoi_name} ({target_aoi['type']})")
        
        # Determine timestamp
        if args.t_ns is not None:
            timestamp_ns = args.t_ns
            logger.info(f"Using explicit timestamp: {timestamp_ns} ({timestamp_ns / 1e9:.3f}s)")
            
            # Validate timestamp
            if not validate_timestamp(timestamp_ns, str(world_path)):
                logger.warning("Timestamp may be outside video duration")
        
        elif args.select == "peak-dwell":
            logger.info(f"Finding peak dwell timestamp with {args.window_ms}ms window...")
            try:
                timestamp_ns = peak_dwell_timestamp(
                    gaze_df, target_aoi, window_ms=args.window_ms
                )
                logger.info(f"Peak dwell timestamp: {timestamp_ns} ({timestamp_ns / 1e9:.3f}s)")
            except ValueError as e:
                print(f"❌ Peak dwell selection failed: {e}")
                sys.exit(1)
        
        # Create output directory
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {out_dir}")
        
        # Extract frame
        logger.info("Extracting video frame...")
        try:
            frame = extract_frame_at_ts(str(world_path), timestamp_ns)
            logger.info(f"Extracted frame: {frame.shape}")
        except Exception as e:
            print(f"❌ Frame extraction failed: {e}")
            sys.exit(1)
        
        # Crop AOI
        logger.info("Cropping AOI region...")
        try:
            aoi_crop = crop_aoi(frame, target_aoi)
            logger.info(f"AOI crop: {aoi_crop.shape}")
        except Exception as e:
            print(f"❌ AOI cropping failed: {e}")
            sys.exit(1)
        
        # Save raw crop
        crop_path = out_dir / "question_snapshot.png"
        cv2.imwrite(str(crop_path), aoi_crop)
        logger.info(f"Saved raw crop: {crop_path}")
        
        # Optional reference warping
        warped_crop = None
        if args.reference:
            ref_path = Path(args.reference)
            if not ref_path.exists():
                print(f"❌ Reference image not found: {ref_path}")
                sys.exit(1)
            
            logger.info("Attempting reference image warping...")
            try:
                # Note: This would require homography computation or pre-computed homography
                logger.warning("Reference warping requires homography - using raw crop for OCR")
                warped_crop = aoi_crop  # Fallback to raw crop
            except Exception as e:
                logger.warning(f"Reference warping failed: {e}, using raw crop")
                warped_crop = aoi_crop
        
        # Choose image for OCR
        ocr_image = warped_crop if warped_crop is not None else aoi_crop
        
        # Preprocess for OCR
        logger.info(f"Preprocessing image with mode '{args.mode}'...")
        try:
            processed_image = prep_for_ocr(
                ocr_image, 
                mode=args.mode, 
                deskew=args.deskew
            )
            logger.info(f"Preprocessed image: {processed_image.shape}")
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}, using original image")
            processed_image = ocr_image
        
        # Run OCR
        logger.info(f"Running OCR with engine preference: {args.engine}")
        try:
            engine_preference = (
                ("tesseract", "easyocr") if args.engine == "auto"
                else (args.engine,)
            )
            
            extracted_text = recognize_text(
                processed_image,
                lang=args.lang,
                engine_preference=engine_preference
            )
            
            logger.info(f"OCR completed, extracted {len(extracted_text)} characters")
            
        except Exception as e:
            print(f"❌ OCR failed: {e}")
            sys.exit(1)
        
        # Save results
        # Text file
        text_path = out_dir / "question_text.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        logger.info(f"Saved extracted text: {text_path}")
        
        # Metadata
        metadata = {
            "timestamp_ns": timestamp_ns,
            "timestamp_sec": timestamp_ns / 1e9,
            "aoi_name": args.aoi_name,
            "aoi_type": target_aoi["type"],
            "aoi_coordinates": target_aoi["coordinates"],
            "world_video": str(world_path),
            "gaze_data": str(gaze_path),
            "aoi_config": str(aoi_path),
            "ocr_engine": args.engine,
            "ocr_language": args.lang,
            "preprocess_mode": args.mode,
            "deskew_enabled": args.deskew,
            "text_length": len(extracted_text),
            "frame_shape": list(frame.shape),
            "crop_shape": list(aoi_crop.shape),
        }
        
        if args.select == "peak-dwell":
            metadata["selection_method"] = "peak-dwell"
            metadata["window_ms"] = args.window_ms
        else:
            metadata["selection_method"] = "explicit"
        
        if args.reference:
            metadata["reference_image"] = str(args.reference)
        
        meta_path = out_dir / "question_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {meta_path}")
        
        # Save warped crop if available
        if warped_crop is not None and args.reference:
            warped_path = out_dir / "question_snapshot_warped.png"
            cv2.imwrite(str(warped_path), warped_crop)
            logger.info(f"Saved warped crop: {warped_path}")
        
        # Print summary
        print("\n✅ OCR extraction completed successfully!")
        print(f"📁 Output directory: {out_dir}")
        print(f"📄 Text file: {text_path.name}")
        print(f"🖼️  Snapshot: {crop_path.name}")
        print(f"📊 Metadata: {meta_path.name}")
        
        if extracted_text.strip():
            print(f"\n📝 Extracted text ({len(extracted_text)} chars):")
            print("─" * 50)
            print(extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""))
            print("─" * 50)
        else:
            print("\n⚠️  No text was extracted (empty result)")
            print("Try adjusting --mode, --lang, or --engine parameters")
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
