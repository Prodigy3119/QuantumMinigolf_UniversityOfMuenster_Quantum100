# Quantum Mini Golf

Quantum Mini Golf is an interactive physics sandbox that blends a classical mini golf ball simulation with a fully fledged quantum wave packet solver. Switch between classical, quantum, or mixed views, explore iconic obstacle layouts, and even drive the simulation from a dual-LED putter tracked by OpenCV.

---

## Highlights
- **Hybrid physics** – propagate the classical ball alongside a quantum wave amplitude in real time.
- **Multiple maps & guided courses** – cycle between wall/slit layouts or launch curated demo sequences.
- **Configurable control panel** – adjust shot parameters, rendering cadence, and physics tunings at runtime.
- **GPU acceleration ready** – automatically uses CuPy when available, with fallbacks to NumPy or pyFFTW.
- **LED putter tracking** – optional OpenCV tracker translates real-world swings into in-game shots.
- **Video capture** – render scripted shots to MP4 through the bundled `RecordVideo.py` utility.

---

## Requirements

| Component | Notes |
|-----------|-------|
| Python 3.10+ | Tested on Windows; other platforms supported when `QtAgg` backend is available. |
| NumPy, SciPy | Core numerics (SciPy optional but recommended for faster blurs). |
| Matplotlib + Qt bindings | UI built around `QtAgg`; install `PyQt5` or `PySide6`. |
| OpenCV (`opencv-python`) | Needed for LED tracker and debug windows. |
| CuPy (optional) | Enables GPU-accelerated wave stepping when CUDA is present. |
| pyFFTW (optional) | Faster FFTs on CPU when GPU is unavailable. |
| FFmpeg (optional) | Required for `RecordVideo.py` MP4 exports. |

Install dependencies manually, for example:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell; use source .venv/bin/activate on Unix
pip install numpy scipy matplotlib PyQt5 opencv-python
pip install cupy-cuda12x pyfftw  # optional accelerators
```

---

## Quick Start
1. **Clone** this repository and install the requirements above.
2. **Launch the sandbox** with:
   ```bash
   python run_game.py
   ```
3. **Interact** with the figure window – drag the shot indicator with the mouse, press hotkeys to change modes, and watch the ball and wave evolve.

Use `python run_game.py --help` for available command-line overrides, including map selection, playback mode, movement speed, and course presets.

---

## Keyboard Reference

| Key | Action |
|-----|--------|
| `Q` | Quit the application. |
| `R` | Reset the current map (or restore from playback). |
| `Tab` | Cycle through available course layouts. |
| `C` | Toggle display mode (`classical`, `quantum`, `mixed`). |
| `M` | Force a measurement when in a quantum-enabled mode. |
| `I` | Toggle quantum info overlay (expectation ellipse, measurement data). |
| `L` | Show or hide the interference side panel (quantum or mixed mode only). |
| `B` | Toggle edge boundary rule (reflect vs absorb). |
| `W` | Switch initial wave profile (`packet` vs `front`). |
| `T` | Toggle shot termination mode (`time` vs `friction`). |
| `G` | Enable/disable mouse swing control (useful when tracker is active). |
| `U` | Show or hide the in-game configuration panel (sliders). |
| `D` | Play back the most recent recording (if available). |
| `H` | Print the full hotkey help message to the console. |

Mouse dragging inside the course aims the shot indicator; releasing triggers the swing. When the tracker is enabled, putter hits trigger automatically based on the LED swing speed and alignment.

---

## LED Tracker Integration
- Enable tracker support via `GameConfig.use_tracker = True` (default) or the CLI `--mouse-swing` flag to switch back to mouse-only control.
- Configure the tracker in `quantum_minigolf/tracker.py` or the LED-specific settings under `quantum_minigolf/tracker_led/cfg_tracker.py`.
- Performance tip: constrain the camera processing area with the new crop fields:
  ```python
  cfg.tracker_crop_x1 = 320  # left
  cfg.tracker_crop_x2 = 960  # right (exclusive)
  cfg.tracker_crop_y1 = 180  # top
  cfg.tracker_crop_y2 = 540  # bottom (exclusive)
  ```
  The tracker thread crops frames before thresholding, reducing CPU cost while keeping LED coordinates in full-frame space. A dashed rectangle in the debug window visualizes the active ROI.
- To align a skewed camera/projector setup, run the manual helper and click the four corners in order:
  ```bash
  python calibrate_course_boundaries.py
  ```
  The tool saves both JSON and pickle outputs (`calibration/course_calibration.json/.pkl`). The game automatically reuses the pickle from the previous run, so you only need to recalibrate after moving the camera or course. You can still override the path explicitly via `GameConfig.tracker_calibration_path`.
- Prefer to auto-detect four LEDs? Use the dedicated script:
  ```bash
  python calibrate_course_boundaries_LED.py --output calibration/course_calibration.json
  ```
  It threshold-detects the illuminated markers, previews the warp, and writes the same calibration files for the main game.
- The tracker expects a dual-LED putter; adjust thresholds, association radius, and impact detection in `cfg_tracker.py` as needed.

---

## Recording the Demo
Use `RecordVideo.py` to export a showcase shot without opening GUI windows:
```bash
python RecordVideo.py
```
The script prepares a mixed-mode double-slit shot, captures frames with Matplotlib’s Agg backend, and writes `DoubleSlitDemo.mp4` (FFmpeg must be installed and on PATH).

---

## Repository Layout

```
quantum_minigolf/
├─ __init__.py             # Package exports (QuantumMiniGolfGame, GameConfig)
├─ game.py                 # Main game loop, UI, modes, and hotkeys
├─ course.py               # Course geometry, barriers, and hole logic
├─ physics.py              # Wave propagation, measurement, and density prep
├─ visuals.py              # Matplotlib rendering helpers and overlays
├─ backends.py             # CPU/GPU backend selection and FFT helpers
├─ tracker.py              # Threaded LED tracker bridge used by the game
├─ tracker_led/            # Standalone tracker app & configuration utilities
└─ ...
run_game.py                # Entry point for interactive play
RecordVideo.py             # Headless demo capture utility
```

---

## Troubleshooting
- **Missing Qt backend**: install `PyQt5` or `PySide6` to satisfy `QtAgg` requirements.
- **Tracker cannot open camera**: verify the camera index in `cfg_tracker.py` and ensure no other app is using the device.
- **GPU not detected**: the app falls back to CPU automatically; check console logs for the CuPy import error reason.
- **FFmpeg errors**: confirm `ffmpeg` is installed and discoverable for video export.

---

## Contributing & License
Contributions are welcome via pull request. Please include a summary of changes and, if applicable, updated instructions for new features.

No explicit license is supplied with this repository. Contact the project maintainers before redistributing or using the code in commercial settings.
