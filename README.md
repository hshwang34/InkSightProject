# 🎯 GazeLab – Eye-Tracking Analysis Pipeline for Research & Actionable Insights

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)

**GazeLab** is a Python toolkit for turning raw eye-tracking recordings into **actionable insights** for research and study settings. Beyond traditional gaze analysis, GazeLab includes **advanced OCR capabilities** to automatically extract and analyze text content from areas where participants focused their attention.

It is designed for **Pupil Labs devices**, but also works with simulated/sample data — making it ideal for pilot testing, classroom studies, and usability research.

> ⚠️ **Beta Software**: GazeLab is currently in active development. While functional, expect bugs, incomplete features, and breaking changes. We welcome feedback and contributions to help improve the toolkit.

---

## ✨ Why GazeLab?

In research and applied settings, **raw gaze coordinates alone are not useful**. What matters are the **insights**:

- Which elements drew attention?  
- How long did participants dwell on critical areas?  
- **What specific text or questions were participants reading?**
- Did participants scan efficiently or get "lost"?  
- How did gaze patterns differ between groups or conditions?  

GazeLab provides a **complete pipeline** from raw gaze data → metrics → visualizations → **automatic text extraction** → interpretable study outputs.

---

## 🔬 The Analysis Pipeline

1. **Capture / Load**  
   - Import CSVs from **Pupil Cloud** or **Pupil Player**  
   - Mock client for replaying synthetic data (useful for demos/pilots)

2. **Pre-Processing**  
   - Data cleaning (remove NaNs, blink gaps, noise filtering)  
   - Normalization into a canonical schema (`t_ns, gx_px, gy_px, frame_w, frame_h`)

3. **Gaze Event Detection**  
   - **I-VT Fixation Detection**: identify fixations and saccades  
   - Optional dispersion-based methods (planned)

4. **AOI Analysis**  
   - Define Areas of Interest (AOIs) via JSON  
   - Compute dwell times, entry counts, exit patterns, fixation density  
   - Export per-AOI summary CSVs + detailed timelines

5. **Visualization & Insight Generation**  
   - **Overlay Videos**: show exactly where participants looked  
   - **Heatmaps**: aggregate attention hotspots  
   - **Reference Mapping**: align gaze to a study stimulus (e.g., webpage, textbook, dashboard)

6. **📝 Text Extraction & OCR Analysis**
   - **Automatic Snapshots**: capture frames at peak attention moments
   - **Smart Cropping**: extract specific AOIs (questions, passages, UI elements)
   - **OCR Processing**: convert visual text to machine-readable content
   - **Content Analysis**: identify exactly what participants were reading

7. **Export for Interpretation**  
   - CSV summaries (for stats in R/SPSS/Python)  
   - Visuals (MP4 overlays, PNG heatmaps) to include in study reports
   - **Text files** with extracted content from areas of interest  

---

## 📊 Example Use Cases

- 🧑‍🏫 **Education Research**  
  - Study how students allocate attention across reading passages, diagrams, or problem sets  
  - **Automatically extract question text** that students spent the most time reading
  - Compare gaze patterns between novices vs experts  
  - **Identify difficult questions** by analyzing dwell patterns and extracting content

- 🧑‍🔬 **Behavioral Science**  
  - Track decision-making processes in ambiguous or multi-option tasks  
  - **Extract text from choice options** participants focused on most
  - Measure attentional biases in clinical studies  
  - **Analyze content** of materials that influenced decision-making

- 🖥️ **UX & Usability**  
  - Quantify whether critical buttons/labels were noticed  
  - **Extract text from UI elements** users interacted with
  - Diagnose confusion in navigation flows  
  - **Identify problematic content** through attention + text analysis  

---

## 🛠️ Current Features (Beta)

### 📝 **OCR & Text Extraction**
- **Intelligent Snapshots**: Automatically capture frames at peak attention moments
- **Smart AOI Cropping**: Extract rectangular, circular, and polygonal regions
- **Dual OCR Engines**: Tesseract and EasyOCR with automatic fallback
- **Preprocessing Pipeline**: Document/UI-optimized image enhancement
- **Peak Dwell Detection**: Find moments of maximum attention within AOIs
- **Multi-language Support**: Extract text in 50+ languages

### 🔍 **Gaze Analysis**
- I-VT fixation detection algorithm
- Area of Interest (AOI) analysis with dwell times and entry patterns
- 2D homography mapping for reference image alignment
- Automatic data normalization for Pupil Labs exports

### 🎥 **Visualizations**
- Gaze overlay videos with customizable trails
- Heatmap generation with multiple colormaps
- Fixation visualization overlays
- Reference image mapping capabilities

### ⚡ **Data Collection**
- Integration with Pupil Labs real-time API (when available)
- Mock replay client for development and testing
- Session recording and management
- Synthetic data generation for demos

### 🛠️ **Development Tools**
- Command-line interface for all operations
- Python API for programmatic access
- Test suite with synthetic data
- Type hints throughout codebase

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **FFmpeg** (for video processing)
- **Pupil Labs Device** (optional, for real-time data collection)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/gaze-lab.git
cd gaze-lab

# Create virtual environment
   python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install GazeLab with OCR capabilities (recommended)
pip install -e ".[ocr]"

# OR install basic version without OCR
   pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### OCR Engine Setup (Required for Text Extraction)

**For best results, install Tesseract OCR:**

```bash
# Windows: Download installer from
# https://github.com/UB-Mannheim/tesseract/wiki

# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Verify installation
tesseract --version
```

**EasyOCR** (alternative, no additional setup needed):
- Automatically installed with `pip install -e ".[ocr]"`
- Larger download but works out-of-the-box
- Good fallback if Tesseract setup issues occur

### Generate Demo Data

```bash
# Create synthetic demo data for testing
   python -m scripts.make_synthetic_demo
```

This generates:
- `data/world.mp4` - Synthetic world camera video
- `data/gaze.csv` - Realistic gaze data following moving objects
- `examples/aoi_example.json` - Sample AOI configuration
- `examples/reference_example.png` - Reference image for mapping

### Your First Analysis

```bash
# Create outputs directory
mkdir -p outputs

# 1. Extract text from areas of peak attention (NEW!)
gaze-ocr --world data/world.mp4 --gaze data/gaze.csv \
         --aoi examples/aoi_example.json --aoi-name "top_left_square" \
         --select peak-dwell --out-dir outputs/ocr

# 2. Generate gaze overlay video
   gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4

# 3. Analyze Areas of Interest
gaze-aoi --gaze data/gaze.csv --aoi examples/aoi_example.json \
         --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv

# 4. Create gaze heatmap
gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png \
             --mode world --world data/world.mp4 --background-frame 0

# 5. Map gaze to reference image
gaze-map --world data/world.mp4 --gaze data/gaze.csv \
         --reference examples/reference_example.png --out outputs/mapped_gaze.csv
```

## 📝 OCR & Text Extraction - Core Feature

**One of GazeLab's most powerful capabilities** is automatically extracting text content from areas where participants focused their attention. This bridges the gap between **where** people looked and **what** they were actually reading.

### Key OCR Capabilities

- 🎯 **Peak Attention Detection**: Automatically find moments of maximum focus within text regions
- 📸 **Smart Snapshots**: Extract video frames at optimal timestamps for text recognition  
- ✂️ **Intelligent Cropping**: Support for rectangular, circular, and polygonal AOIs
- 🧠 **Dual OCR Engines**: Tesseract (fast, accurate) with EasyOCR fallback (robust)
- 🌍 **Multi-language**: Extract text in 50+ languages including English, Spanish, French, German, Chinese, etc.
- 🔧 **Smart Preprocessing**: Automatic image enhancement optimized for documents vs. UI text

### Quick OCR Example

```bash
# Find peak attention moment and extract question text
gaze-ocr --world study_video.mp4 --gaze participant_001.csv \
         --aoi questions.json --aoi-name "Question_3" \
         --select peak-dwell --mode doc --lang eng \
         --out-dir results/question_analysis
```

**Output:**
- `question_text.txt` - Extracted text content
- `question_snapshot.png` - Image of the text region  
- `question_meta.json` - Timestamp, confidence, and analysis metadata

---

## 📊 CLI Tools

GazeLab provides powerful command-line tools for all major operations:

| Tool | Description | Example |
|------|-------------|---------|
| `gaze-ocr` | **Extract text from AOIs using OCR** | `gaze-ocr --world video.mp4 --gaze data.csv --aoi aois.json --aoi-name Question --select peak-dwell` |
| `gaze-record` | Record real-time data from Pupil Labs devices | `gaze-record --mode pupil --out recordings/` |
| `gaze-overlay` | Create gaze overlay videos | `gaze-overlay --world video.mp4 --gaze data.csv --out overlay.mp4` |
| `gaze-aoi` | Analyze Areas of Interest | `gaze-aoi --gaze data.csv --aoi aois.json --report report.csv` |
| `gaze-heatmap` | Generate gaze heatmaps | `gaze-heatmap --gaze data.csv --out heatmap.png` |
| `gaze-map` | Map gaze to reference images | `gaze-map --world video.mp4 --gaze data.csv --reference ref.png` |

## 🔧 Pupil Labs Integration

### Supported Data Sources

- **Pupil Cloud Exports**: Direct CSV import with automatic schema detection
- **Pupil Player Exports**: Raw data exporter compatibility
- **Legacy Core Exports**: Support for older Pupil Labs formats
- **Real-Time API**: Live data collection from Pupil Labs devices

### Real-Time Data Collection

```bash
# Record from Pupil Labs device
gaze-record --mode pupil --host 127.0.0.1 --port 8080 --out recordings/session_1/

# Mock replay for development
gaze-record --mode mock --gaze data/gaze.csv --world data/world.mp4 --out recordings/mock/
```

### Optional Dependencies

For real-time Pupil Labs integration, install the optional dependencies:

```bash
pip install pupil-labs-realtime-api
```

## 📈 Advanced Usage

### Custom Visualization Parameters

```bash
# Custom overlay appearance
gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4 \
             --dot-radius 12 --trail 20 --dot-alpha 0.9 --show-fixations

# Custom heatmap appearance
gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png \
             --bandwidth 30 --grid-size 200 --colormap viridis --alpha 0.8
```

### AOI Configuration

Create custom Areas of Interest with JSON configuration:

```json
{
  "aois": [
    {
      "name": "button_submit",
      "type": "rectangle",
      "coordinates": [[100, 100], [200, 100], [200, 150], [100, 150]],
      "metadata": {"description": "Submit button"}
    },
    {
      "name": "logo_area",
      "type": "circle",
      "coordinates": [[640, 360, 50]],
      "metadata": {"description": "Company logo"}
    }
  ],
  "metadata": {
    "frame_size": {"width": 1280, "height": 720}
  }
}
```

### Python API Usage

```python
import gaze_lab
from gaze_lab.io.cloud_loader import load_gaze_data
from gaze_lab.viz.overlay import create_gaze_overlay
from gaze_lab.processing.aoi import analyze_aois

# Load gaze data
gaze_data = load_gaze_data("data/gaze.csv")

# Create overlay
create_gaze_overlay(
    world_video="data/world.mp4",
    gaze_data=gaze_data,
    output_path="outputs/overlay.mp4",
    dot_radius=10,
    show_fixations=True
)

# Analyze AOIs
aoi_results = analyze_aois(
    gaze_data=gaze_data,
    aoi_config="examples/aoi_example.json"
)
```

## 🧪 Testing & Development

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gaze_lab --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
```

### Code Quality

```bash
# Format code
black gaze_lab tests scripts
isort gaze_lab tests scripts

# Lint code
flake8 gaze_lab tests
mypy gaze_lab

# Run all quality checks
make lint
make format
```

### Development Setup

```bash
# Install development dependencies
make install-dev

# Generate demo data
make demo

# Run full demo pipeline
make demo-full

# Clean build artifacts
make clean
```

## 📁 Project Structure

```
gaze_lab/
├── cli/                    # Command-line interface tools
│   ├── export_aoi_report.py
│   ├── export_heatmap.py
│   ├── export_overlay.py
│   ├── map_to_reference.py
│   └── record_realtime.py
├── io/                     # Data input/output modules
│   ├── cloud_loader.py     # Pupil Labs data loading
│   ├── player_compat.py    # Player compatibility
│   └── realtime/           # Real-time data collection
├── processing/             # Core analysis algorithms
│   ├── aoi.py             # Area of Interest analysis
│   ├── fixations_ivt.py   # I-VT fixation detection
│   ├── filters.py         # Data filtering
│   └── mapping_2d.py      # 2D homography mapping
├── viz/                   # Visualization modules
│   ├── heatmap.py         # Heatmap generation
│   └── overlay.py         # Gaze overlay creation
└── config.py              # Configuration management
```

## 🔬 Data Schema

GazeLab uses a canonical data schema for consistent processing:

| Column | Type | Description |
|--------|------|-------------|
| `t_ns` | int64 | Timestamp in nanoseconds (UTC) |
| `gx_px` | float64 | Gaze X coordinate in pixels |
| `gy_px` | float64 | Gaze Y coordinate in pixels |
| `frame_w` | int32 | World frame width |
| `frame_h` | int32 | World frame height |
| `confidence` | float64 | Gaze confidence (0-1) |
| `pupil_diameter` | float64 | Pupil diameter (optional) |

## 🛡️ Privacy & Ethics

**Important**: Eye-tracking data is highly sensitive personal information.

- ✅ **Always obtain informed consent** before recording participants
- ✅ **Use synthetic demo data** for development and testing
- ✅ **Anonymize data** before sharing or publication
- ❌ **Never share** videos or gaze data containing PII
- ❌ **Never record** without explicit participant consent


## 📚 Documentation

- [Quick Start Guide](docs/quickstart.md) - Get up and running quickly
- [API Reference](docs/api.md) - Complete API documentation
- [AOI Guide](docs/aoi-guide.md) - Creating and managing Areas of Interest
- [Visualization Guide](docs/visualization-guide.md) - Advanced visualization options
- [Real-Time Guide](docs/realtime-guide.md) - Live data collection setup

## 🐛 Troubleshooting

### Common Issues

**"No gaze data found"**
- Check CSV column names match expected schema
- Verify data contains valid coordinates within frame bounds

**"Failed to open video"**
- Ensure video file exists and is readable
- Check OpenCV can decode the video format

**"Pupil Labs connection failed"**
- Verify device is connected and running
- Check host/port settings match device configuration
- Install real-time dependencies: `pip install pupil-labs-realtime-api`

## 📄 License

This project is licensed under the MIT License 

## 🙏 Acknowledgments

- **Pupil Labs** for their excellent eye-tracking hardware and software
- **OpenCV** community for computer vision capabilities
- **NumPy/SciPy** for numerical computing foundations

## 📊 Citation

If you use GazeLab in your research, please cite:

```bibtex
@software{gazelab2024,
  title={GazeLab: Professional Eye-Tracking Analysis Toolkit},
  author={GazeLab Team},
  year={2024},
  url={https://github.com/your-username/gaze-lab},
  license={MIT}
}
```

## 📝 Advanced OCR & Text Extraction

### Real-World Research Applications

**Education Studies:**
```bash
# Extract text from math problems students struggled with
gaze-ocr --world classroom_study.mp4 --gaze student_data.csv \
         --aoi math_problems.json --aoi-name "Problem_5" \
         --select peak-dwell --mode doc --out-dir analysis/difficult_problems
```

**Reading Comprehension:**
```bash
# Find which passage segments got most attention
gaze-ocr --world reading_task.mp4 --gaze participant.csv \
         --aoi reading_passages.json --aoi-name "Paragraph_3" \
         --select peak-dwell --window-ms 8000 --lang eng
```

**Interface Usability:**
```bash
# Extract UI text users focused on during errors
gaze-ocr --world usability_test.mp4 --gaze user_session.csv \
         --aoi interface_elements.json --aoi-name "Error_Message" \
         --select peak-dwell --mode ui --out-dir error_analysis
```

### Complete OCR Workflow

#### 1. Automatic Peak Detection (Recommended)
```bash
# Let GazeLab find the moment of peak attention
gaze-ocr --world study_video.mp4 --gaze data.csv \
         --aoi questions.json --aoi-name "Question_1" \
         --select peak-dwell \
         --window-ms 5000 \
         --mode doc \
         --out-dir results/q1_analysis
```

#### 2. Specific Timestamp Extraction  
```bash
# Extract at known timestamp (12.5 seconds)
gaze-ocr --world study_video.mp4 --gaze data.csv \
         --aoi questions.json --aoi-name "Question_1" \
         --t-ns 12500000000 \
         --mode doc \
         --out-dir results/q1_timestamp
```

#### 3. Multi-language Support
```bash
# Extract Spanish and English text
gaze-ocr --world bilingual_study.mp4 --gaze data.csv \
         --aoi content.json --aoi-name "Instructions" \
         --select peak-dwell \
         --lang "eng+spa" \
         --mode doc
```

### Understanding OCR Output

**Generated Files:**
- `question_text.txt` - Clean, extracted text ready for analysis
- `question_snapshot.png` - Visual crop showing exactly what was extracted
- `question_meta.json` - Timestamps, confidence scores, processing details

**Sample Output Structure:**
```
outputs/ocr/
├── question_text.txt          # "What is the capital of France?"
├── question_snapshot.png      # 200x100 pixel crop of the question
├── question_meta.json         # {"timestamp_ns": 12500000000, "confidence": 0.94, ...}
└── question_snapshot_warped.png  # (if using reference alignment)
```

### OCR Processing Modes

| Mode | Best For | Image Processing | Use Case |
|------|----------|------------------|----------|
| `doc` | Printed text, PDFs, books | Aggressive contrast, binarization | Reading studies, textbook analysis |
| `ui` | Screen text, interfaces | Gentle enhancement, preserve details | Software usability, web studies |
| `auto` | Mixed content | Adaptive based on image characteristics | General purpose, unknown content |

### Advanced Parameters

```bash
gaze-ocr \
  --world video.mp4 --gaze data.csv --aoi config.json --aoi-name "Target" \
  --select peak-dwell \
  --window-ms 3000 \        # Shorter window for quick reading
  --mode doc \              # Document-optimized processing  
  --lang "eng+fra" \        # Multi-language recognition
  --engine tesseract \      # Force specific OCR engine
  --deskew \                # Fix rotated text
  --verbose                 # Detailed logging
```

### Research Integration Tips

**Statistical Analysis:**
- Use `question_meta.json` timestamps for correlation analysis
- Combine with AOI dwell reports for comprehensive insights
- Export text content for content analysis in R/Python

**Quality Control:**
- Review `question_snapshot.png` images to verify extraction accuracy
- Check confidence scores in metadata for reliability assessment
- Use multiple preprocessing modes for difficult content

**Batch Processing:**
```bash
# Process multiple AOIs automatically
for aoi in Question_1 Question_2 Question_3; do
    gaze-ocr --world study.mp4 --gaze data.csv \
             --aoi questions.json --aoi-name "$aoi" \
             --select peak-dwell --out-dir "results/$aoi"
done
```

---

*GazeLab - Transforming gaze data into actionable insights*
