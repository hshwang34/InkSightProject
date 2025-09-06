# GazeLab Quick Start Guide

This guide will walk you through setting up GazeLab and running your first analysis.

## Prerequisites

- Python 3.10 or higher
- Git (for cloning the repository)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gazelab/gaze-lab.git
   cd gaze-lab
   ```

2. **Install GazeLab:**
   ```bash
   pip install -e .
   ```

3. **Verify installation:**
   ```bash
   gaze-overlay --help
   ```

## Generate Demo Data

GazeLab includes a synthetic data generator for testing and demonstration:

```bash
python -m scripts.make_synthetic_demo
```

This creates:
- `data/world.mp4` - Synthetic world video (5 seconds, 1280x720)
- `data/gaze.csv` - Synthetic gaze data following a moving square
- `examples/reference_example.png` - Reference image for mapping
- `examples/aoi_example.json` - Example AOI configuration

## Your First Analysis

### 1. Create a Gaze Overlay

```bash
gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4
```

This creates a video with gaze dots overlaid on the world video.

### 2. Generate AOI Analysis

```bash
gaze-aoi --gaze data/gaze.csv --aoi examples/aoi_example.json \
         --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv
```

This analyzes gaze data against the defined Areas of Interest.

### 3. Create a Heatmap

```bash
gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png \
             --mode world --world data/world.mp4 --background-frame 0
```

This generates a heatmap showing gaze density over the world video.

### 4. Map to Reference Image

```bash
gaze-map --world data/world.mp4 --gaze data/gaze.csv \
         --reference examples/reference_example.png --out outputs/mapped_gaze.csv
```

This maps gaze coordinates to the reference image using homography.

## Understanding the Output

### Gaze Overlay Video
- Green dots show gaze positions
- Dots fade over time to show gaze trail
- Video maintains original frame rate and quality

### AOI Report
The `aoi_report.csv` contains:
- `aoi_name`: Name of the AOI
- `total_hits`: Number of gaze points in the AOI
- `total_dwells`: Number of dwell periods
- `total_dwell_time_ms`: Total time spent in the AOI
- `fixation_count`: Number of fixations overlapping the AOI

### AOI Timeline
The `aoi_timeline.csv` contains:
- `aoi_name`: Name of the AOI
- `enter_ns`: Timestamp when gaze entered the AOI
- `exit_ns`: Timestamp when gaze exited the AOI
- `dwell_ms`: Duration of the dwell period

### Heatmap
- Color intensity represents gaze density
- Hot colors (red/yellow) indicate high gaze density
- Cool colors (blue) indicate low gaze density

## Working with Your Own Data

### 1. Prepare Your Data

GazeLab can load data from various Pupil Labs export formats:

- **Pupil Cloud exports**: CSV files with gaze data
- **Player exports**: Raw data exporter CSV files
- **Legacy Core exports**: Older format CSV files

Place your files in a directory structure like:
```
your_data/
├── world.mp4          # World camera video
├── gaze.csv           # Gaze data
└── info.csv           # Session metadata (optional)
```

### 2. Load and Analyze

```bash
# Create overlay
gaze-overlay --world your_data/world.mp4 --gaze your_data/gaze.csv --out outputs/overlay.mp4

# Generate heatmap
gaze-heatmap --gaze your_data/gaze.csv --out outputs/heatmap.png \
             --mode world --world your_data/world.mp4
```

### 3. Define AOIs

Create an AOI configuration file:

```json
{
  "aois": [
    {
      "name": "button_1",
      "type": "rectangle",
      "coordinates": [[100, 100], [200, 100], [200, 150], [100, 150]],
      "metadata": {"description": "Submit button"}
    }
  ],
  "metadata": {
    "frame_size": {"width": 1280, "height": 720}
  }
}
```

Then run AOI analysis:

```bash
gaze-aoi --gaze your_data/gaze.csv --aoi your_aois.json \
         --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv
```

## Advanced Usage

### Custom Visualization Parameters

```bash
# Custom overlay appearance
gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4 \
             --dot-radius 12 --trail 20 --dot-alpha 0.9

# Custom heatmap appearance
gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png \
             --bandwidth 30 --grid-size 200 --colormap viridis --alpha 0.8
```

### Fixation Detection

```bash
# Detect fixations and show in overlay
gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4 \
             --detect-fixations --show-fixations --fixation-radius 25
```

### Real-time Recording (Mock)

```bash
# Record from mock client
gaze-record --mode mock --gaze data/gaze.csv --world data/world.mp4 --out data/recording/
```

## Troubleshooting

### Common Issues

1. **"No gaze data found"**
   - Check that your CSV file has the correct column names
   - GazeLab automatically detects common column name variations

2. **"Failed to open video"**
   - Ensure the video file exists and is readable
   - Check that OpenCV can decode the video format

3. **"No valid gaze data after filtering"**
   - Check for NaN values in your gaze data
   - Verify coordinate bounds are within frame dimensions

### Getting Help

- Check the [README.md](../README.md) for detailed documentation
- Review the [API documentation](api.md) for programmatic usage
- Open an [issue](https://github.com/gazelab/gaze-lab/issues) for bugs or feature requests

## Next Steps

- Explore the [API documentation](api.md) for programmatic usage
- Learn about [custom AOI definitions](aoi-guide.md)
- See [advanced visualization options](visualization-guide.md)
- Check out [real-time data collection](realtime-guide.md)
