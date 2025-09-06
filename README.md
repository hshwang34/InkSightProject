gaze_lab: Pupil Labs–Compatible Gaze Analysis Toolkit
gaze_lab is a Python toolkit designed to analyze eye-tracking data exported from Pupil Labs Cloud/Player (Raw Data Exporter). It provides utilities to generate gaze overlay videos, heatmaps, fixation metrics, and AOI dwell reports. The toolkit also includes a mock realtime client and an optional adapter for the official Pupil Labs Realtime API.
Features
•	Load Pupil Cloud/Player gaze exports (CSV + world.mp4) and normalize to canonical schema
•	Overlay gaze points and fixations on world video (MP4 output)
•	Generate heatmaps (PNG) over world frames or reference images
•	Detect fixations using I-VT algorithm
•	Compute AOI dwell times and entry/exit timelines (CSV outputs)
•	Map gaze to a static reference image using 2D homography
•	Mock realtime client to replay data without hardware
•	Optional adapter stub for Pupil Labs Realtime Python client (import guarded)
Installation
Requirements: Python 3.10–3.12, ffmpeg, pip.
1. Clone the repo:
   git clone https://github.com/<your-username>/gaze_lab.git
2. Enter directory and set up virtual environment:
   cd gaze_lab
   python -m venv .venv
   .\.venv\Scripts\activate (Windows)
   source .venv/bin/activate (macOS/Linux)
3. Install package:
   pip install -e .
Quickstart (Synthetic Demo)
1. Generate demo data:
   python -m scripts.make_synthetic_demo
This will create data/world.mp4 and data/gaze.csv.
2. Run overlay:
   gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4
3. Run AOI report:
   gaze-aoi --gaze data/gaze.csv --aoi examples/aoi_example.json --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv
4. Generate heatmap:
   gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.pnggaze_lab: Pupil Labs–Compatible Gaze Analysis Toolkit
gaze_lab is a Python toolkit designed to analyze eye-tracking data exported from Pupil Labs Cloud/Player (Raw Data Exporter). It provides utilities to generate gaze overlay videos, heatmaps, fixation metrics, and AOI dwell reports. The toolkit also includes a mock realtime client and an optional adapter for the official Pupil Labs Realtime API.
Features
•	Load Pupil Cloud/Player gaze exports (CSV + world.mp4) and normalize to canonical schema
•	Overlay gaze points and fixations on world video (MP4 output)
•	Generate heatmaps (PNG) over world frames or reference images
•	Detect fixations using I-VT algorithm
•	Compute AOI dwell times and entry/exit timelines (CSV outputs)
•	Map gaze to a static reference image using 2D homography
•	Mock realtime client to replay data without hardware
•	Optional adapter stub for Pupil Labs Realtime Python client (import guarded)
Installation
Requirements: Python 3.10–3.12, ffmpeg, pip.
1. Clone the repo:
   git clone https://github.com/<your-username>/gaze_lab.git
2. Enter directory and set up virtual environment:
   cd gaze_lab
   python -m venv .venv
   .\.venv\Scripts\activate (Windows)
   source .venv/bin/activate (macOS/Linux)
3. Install package:
   pip install -e .
Quickstart (Synthetic Demo)
1. Generate demo data:
   python -m scripts.make_synthetic_demo
This will create data/world.mp4 and data/gaze.csv.
2. Run overlay:
   gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4
3. Run AOI report:
   gaze-aoi --gaze data/gaze.csv --aoi examples/aoi_example.json --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv
4. Generate heatmap:
   gaze-heatmap --gaze data/gaze.csv --out outputs/hgaze_lab: Pupil Labs–Compatible Gaze Analysis Toolkit
gaze_lab is a Python toolkit designed to analyze eye-tracking data exported from Pupil Labs Cloud/Player (Raw Data Exporter). It provides utilities to generate gaze overlay videos, heatmaps, fixation metrics, and AOI dwell reports. The toolkit also includes a mock realtime client and an optional adapter for the official Pupil Labs Realtime API.
Features
•	Load Pupil Cloud/Player gaze exports (CSV + world.mp4) and normalize to canonical schema
•	Overlay gaze points and fixations on world video (MP4 output)
•	Generate heatmaps (PNG) over world frames or reference images
•	Detect fixations using I-VT algorithm
•	Compute AOI dwell times and entry/exit timelines (CSV outputs)
•	Map gaze to a static reference image using 2D homography
•	Mock realtime client to replay data without hardware
•	Optional adapter stub for Pupil Labs Realtime Python client (import guarded)
Installation
Requirements: Python 3.10–3.12, ffmpeg, pip.
1. Clone the repo:
   git clone https://github.com/<your-username>/gaze_lab.git
2. Enter directory and set up virtual environment:
   cd gaze_lab
   python -m venv .venv
   .\.venv\Scripts\activate (Windows)
   source .venv/bin/activate (macOS/Linux)
3. Install package:
   pip install -e .
Quickstart (Synthetic Demo)
1. Generate demo data:
   python -m scripts.make_synthetic_demo
This will create data/world.mp4 and data/gaze.csv.
2. Run overlay:
   gaze-overlay --world data/world.mp4 --gaze data/gaze.csv --out outputs/overlay.mp4
3. Run AOI report:
   gaze-aoi --gaze data/gaze.csv --aoi examples/aoi_example.json --report outputs/aoi_report.csv --timeline outputs/aoi_timeline.csv
4. Generate heatmap:
   gaze-heatmap --gaze data/gaze.csv --out outputs/heatmap.png
CLI Tools
• gaze-record — record realtime or replay via mock client
• gaze-overlay — export gaze overlay video
• gaze-aoi — compute AOI reports and timelines
• gaze-heatmap — generate heatmap from gaze data
• gaze-map — map gaze to reference image (homography)
Data Schema
Canonical columns in gaze.csv:
•	t_ns — timestamp in nanoseconds (UTC)
•	gx_px, gy_px — gaze x/y in pixels (origin top-left)
•	frame_w, frame_h — associated world frame resolution
Testing
Run tests with:
   pytest -q
Ethics & Privacy
Eye-tracking data is sensitive. Only record participants with informed consent. Do not share videos or gaze CSVs containing PII. Synthetic demo data is provided for experimentation without privacy concerns.
atmap.png
CLI Tools
• gaze-record — record realtime or replay via mock client
• gaze-overlay — export gaze overlay video
• gaze-aoi — compute AOI reports and timelines
• gaze-heatmap — generate heatmap from gaze data
• gaze-map — map gaze to reference image (homography)
Data Schema
Canonical columns in gaze.csv:
•	t_ns — timestamp in nanoseconds (UTC)
•	gx_px, gy_px — gaze x/y in pixels (origin top-left)
•	frame_w, frame_h — associated world frame resolution
Testing
Run tests with:
   pytest -q
Ethics & Privacy
Eye-tracking data is sensitive. Only record participants with informed consent. Do not share videos or gaze CSVs containing PII. Synthetic demo data is provided for experimentation without privacy concerns.

CLI Tools
• gaze-record — record realtime or replay via mock client
• gaze-overlay — export gaze overlay video
• gaze-aoi — compute AOI reports and timelines
• gaze-heatmap — generate heatmap from gaze data
• gaze-map — map gaze to reference image (homography)
Data Schema
Canonical columns in gaze.csv:
•	t_ns — timestamp in nanoseconds (UTC)
•	gx_px, gy_px — gaze x/y in pixels (origin top-left)
•	frame_w, frame_h — associated world frame resolution
Testing
Run tests with:
   pytest -q
Ethics & Privacy
Eye-tracking data is sensitive. Only record participants with informed consent. Do not share videos or gaze CSVs containing PII. Synthetic demo data is provided for experimentation without privacy concerns.
