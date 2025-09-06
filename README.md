# GazeLab: Production-Ready Pupil Labs Eye-Tracking Analysis Toolkit

[![Status: Works with sample exports; realtime optional](https://img.shields.io/badge/Status-Works%20with%20sample%20exports%3B%20realtime%20optional-green)](https://github.com/gazelab/gaze-lab)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GazeLab is a comprehensive, production-ready toolkit for analyzing Pupil Labs eye-tracking data. It provides end-to-end functionality for processing both real-time streams and post-hoc exports, generating gaze overlays, AOI dwell reports, and heatmaps.

## Features

- **📊 Comprehensive Data Processing**: Load and normalize data from Pupil Cloud, Player exports, and legacy Core formats
- **🎥 Gaze Overlay Generation**: Create high-quality gaze overlay videos with configurable appearance
- **🎯 AOI Analysis**: Define Areas of Interest and generate detailed dwell reports and timelines
- **🔥 Heatmap Visualization**: Generate static and dynamic gaze heatmaps with KDE analysis
- **🗺️ 2D Coordinate Mapping**: Map gaze coordinates between world video and reference images using homography
- **⚡ Real-time Support**: Optional real-time data acquisition with mock and hardware clients
- **🛠️ CLI Tools**: Complete command-line interface for all analysis tasks
- **🧪 Comprehensive Testing**: Full test suite with synthetic data generation

## Quick Start

### Installation

```bash
# Install from source
git clone https://github.com/gazelab/gaze-lab.git
cd gaze-lab
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Generate Demo Data

```bash
# Create synthetic demo data
python -m scripts.make_synthetic_demo

# This creates:
# - data/world.mp4 (synthetic world video)
# - data/gaze.csv (synthetic gaze data)
# - examples/reference_example.png (reference image)
# - examples/aoi_example.json (AOI configuration)
```

### Run Analysis

```bash
# Create gaze overlay video
gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4

# Generate AOI analysis report
gaze-aoi --gaze data/gaze.csv --aoi examples/aoi_example.json \
         --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv

# Create heatmap visualization
gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png \
             --mode world --world data/world.mp4 --background-frame 0

# Map gaze to reference image
gaze-map --world data/world.mp4 --gaze data/gaze.csv \
         --reference examples/reference_example.png --out outputs/mapped_gaze.csv
```

## API Usage

### Loading Gaze Data

```python
from gaze_lab.io.cloud_loader import load_gaze
import pandas as pd

# Load and normalize gaze data
gaze_df = load_gaze("path/to/gaze.csv")
print(f"Loaded {len(gaze_df)} gaze samples")

# Data is automatically normalized to canonical schema:
# - t_ns: timestamp in nanoseconds
# - gx_px, gy_px: gaze coordinates in pixels
# - frame_w, frame_h: frame dimensions
```

### AOI Analysis

```python
from gaze_lab.processing.aoi import AOIAnalyzer

# Create AOI analyzer
analyzer = AOIAnalyzer()

# Load AOI definitions
analyzer.load_aois_from_file("examples/aoi_example.json")

# Analyze gaze data
aoi_hits = analyzer.analyze_gaze_data(gaze_df)
aoi_dwells = analyzer.calculate_dwell_times(aoi_hits)

# Generate reports
report = analyzer.generate_aoi_report(aoi_hits, aoi_dwells)
timeline = analyzer.generate_aoi_timeline(aoi_dwells)
```

### Fixation Detection

```python
from gaze_lab.processing.fixations_ivt import detect_fixations_ivt

# Detect fixations using I-VT algorithm
fixations = detect_fixations_ivt(
    gaze_df,
    velocity_threshold_deg_s=30.0,
    min_duration_ms=50.0,
    px_per_degree=30.0
)

print(f"Detected {len(fixations)} fixations")
```

### Visualization

```python
from gaze_lab.viz.overlay import create_gaze_overlay
from gaze_lab.viz.heatmap import create_heatmap

# Create gaze overlay video
create_gaze_overlay(
    world_video_path="data/world.mp4",
    gaze_df=gaze_df,
    output_path="outputs/overlay.mp4",
    dot_radius=10,
    trail_length=12,
    show_fixations=True
)

# Create heatmap
create_heatmap(
    gaze_df=gaze_df,
    output_path="outputs/heatmap.png",
    background_mode="world",
    world_video_path="data/world.mp4",
    bandwidth=20.0,
    colormap="hot"
)
```

## CLI Commands

### gaze-overlay
Create gaze overlay videos with configurable appearance.

```bash
gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4 \
             --dot-radius 10 --trail 12 --show-fixations
```

### gaze-aoi
Generate comprehensive AOI analysis reports.

```bash
gaze-aoi --gaze data/gaze.csv --aoi examples/aoi_example.json \
         --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv
```

### gaze-heatmap
Create gaze heatmap visualizations.

```bash
# World video background
gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png \
             --mode world --world data/world.mp4 --background-frame 0

# Reference image background
gaze-heatmap --gaze outputs/mapped_gaze.csv --out outputs/ref_heatmap.png \
             --mode reference --reference examples/reference_example.png
```

### gaze-map
Map gaze coordinates to reference images.

```bash
gaze-map --world data/world.mp4 --gaze data/gaze.csv \
         --reference examples/reference_example.png --out outputs/mapped_gaze.csv
```

### gaze-record
Record real-time gaze data (mock or hardware).

```bash
# Mock recording
gaze-record --mode mock --gaze data/gaze.csv --world data/world.mp4 --out data/recording/

# Hardware recording (if available)
gaze-record --mode pupil --host 127.0.0.1 --out data/recording/
```

## AOI Definition Format

AOIs are defined in JSON format with support for rectangles, circles, and polygons:

```json
{
  "aois": [
    {
      "name": "button_1",
      "type": "rectangle",
      "coordinates": [[100, 100], [200, 100], [200, 150], [100, 150]],
      "metadata": {"description": "Submit button"}
    },
    {
      "name": "logo",
      "type": "circle",
      "coordinates": [[640, 360, 50]],
      "metadata": {"description": "Company logo"}
    },
    {
      "name": "menu_area",
      "type": "polygon",
      "coordinates": [[50, 50], [300, 50], [300, 200], [100, 200], [50, 150]],
      "metadata": {"description": "Navigation menu"}
    }
  ],
  "metadata": {
    "created_by": "GazeLab",
    "frame_size": {"width": 1280, "height": 720}
  }
}
```

## Data Schema

### Canonical Gaze Data Format

All gaze data is normalized to this canonical schema:

| Column | Type | Description |
|--------|------|-------------|
| `t_ns` | int | Timestamp in nanoseconds |
| `gx_px` | float | Gaze X coordinate in pixels (origin: top-left) |
| `gy_px` | float | Gaze Y coordinate in pixels (origin: top-left) |
| `frame_w` | int | Frame width in pixels |
| `frame_h` | int | Frame height in pixels |

### Output Files

- **overlay.mp4**: MP4 video with gaze dots and optional fixation markers
- **aoi_report.csv**: Per-AOI aggregates (dwell time, entries, fixation count)
- **aoi_timeline.csv**: AOI enter/exit timeline with nanosecond timestamps
- **heatmap.png**: KDE heatmap visualization
- **mapped_gaze.csv**: Gaze points in reference image coordinates

## Coordinate Systems

- **Pixel Coordinates**: Origin at top-left, units in pixels
- **Timestamps**: Nanoseconds since epoch (monotonically increasing)
- **NaN Handling**: Missing gaze data (blinks, lost tracking) handled gracefully
- **Bounds Validation**: Coordinates automatically validated against frame dimensions

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_fixations_ivt.py
pytest tests/test_aoi.py
pytest tests/test_overlay_smoke.py
pytest tests/test_heatmap_smoke.py

# Run with coverage
pytest --cov=gaze_lab --cov-report=html
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black gaze_lab tests scripts
isort gaze_lab tests scripts

# Run linting
flake8 gaze_lab tests
mypy gaze_lab

# Run tests
pytest
```

## Ethical Considerations

- **Consent**: Ensure proper informed consent for eye-tracking data collection
- **Privacy**: Handle gaze data as personally identifiable information (PII)
- **Data Protection**: Follow applicable data protection regulations (GDPR, CCPA, etc.)
- **Research Ethics**: Comply with institutional review board (IRB) requirements

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for compatibility with [Pupil Labs](https://pupil-labs.com/) eye-tracking systems
- Follows official Pupil Labs documentation and best practices
- Inspired by the eye-tracking research community

## Support

- **Documentation**: [https://gaze-lab.readthedocs.io](https://gaze-lab.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/gazelab/gaze-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/gazelab/gaze-lab/discussions)