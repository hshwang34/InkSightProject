# Data Directory

This directory contains gaze data and world video files for analysis.

## File Organization

Place your Pupil Labs export data in this directory:

```
data/
├── world.mp4          # World camera video
├── gaze.csv           # Gaze data (normalized to canonical schema)
├── info.csv           # Session metadata (optional)
└── fixations.csv      # Fixation data (optional)
```

## Supported Formats

GazeLab automatically detects and normalizes data from:

- **Pupil Cloud exports**: CSV files with gaze data
- **Player exports**: Raw data exporter CSV files  
- **Legacy Core exports**: Older format CSV files

## Data Requirements

### World Video
- Format: MP4, AVI, MOV
- Resolution: Any (will be detected automatically)
- Frame rate: Any (will be detected automatically)

### Gaze Data
- Format: CSV
- Required columns: timestamp, gaze coordinates
- Optional columns: confidence, pupil diameter, eye ID

## Generated Data

When you run `python -m scripts.make_synthetic_demo`, this directory will contain:

- `world.mp4` - Synthetic world video (5 seconds, 1280x720)
- `gaze.csv` - Synthetic gaze data following a moving square

## Usage

```bash
# Load gaze data
gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4

# Generate heatmap
gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png --mode world --world data/world.mp4
```

## Data Privacy

- Gaze data is personally identifiable information (PII)
- Ensure proper consent and data protection measures
- Follow applicable regulations (GDPR, CCPA, etc.)
- Consider data anonymization for research purposes
