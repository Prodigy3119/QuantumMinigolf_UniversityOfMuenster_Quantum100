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

Refer to the command-line flags listed below to tailor launch mode, tracker behaviour, automation helpers, and diagnostics straight from the CLI.

## Command-Line Flags
Flags are grouped by theme; defaults are shown in parentheses.

### Core Launch
- `--map <double_slit|single_slit|single_wall|no_obstacle>` (default `double_slit`): choose the opening course layout.
- `--mode <classical|quantum|mixed>` (default `mixed`): set the initial visualisation mode.
- `--course <quantum_demo|advanced_showcase>` (default none): jump directly into a guided multi-stage course.
- `--wave-profile <packet|front>` (default `front`): pick the initial wave packet shape.
- `--stop-mode <time|friction>` (default `time`): decide how shots terminate.
- `--wall-thickness <float>` (default `1.0`): scale the central barrier thickness.
- `--movement-speed <float>` (default `1.0`): multiply ball and wave travel speed.
- `--shot-time <float>` (default `50.0`, `<=0` means infinite): cap simulation time per shot.
- `--res-scale <float>` (default `1.0`): render at a higher or lower resolution.
- `--draw-every <int>` (default `3`): draw every Nth simulation frame.
- `--target-fps <float>` (default `30`): set the desired redraw cadence.
- `--mouse-swing` (default enabled): ensure mouse swings stay active even if tracker hardware is connected.
- `--config-panel` / `--no-control-panel` (default show): force the separate control panel to open or remain hidden.

### Tracker & Calibration
- `--vr` / `--no-vr` (default hybrid: mouse swing enabled, tracker armed): toggle between tracker-driven swings and mouse-only play.
- `--display-tracker` / `--no-display-tracker` (default show): show or hide the OpenCV tracker debug window.
- `--calibration-path <file>` (default auto-discovery): point to an explicit calibration file.
- `--skip-calibration-preview` (default show preview): suppress the OpenCV snapshot after loading calibration data.
- `--calibrate-course`: launch the manual corner picker before the game starts.
- `--calibrate-course-led`: run the LED auto-calibration helper before launching.

### Gameplay & Physics
- `--sink-threshold <float>` (default `0.25`): adjust the probability threshold for hole detection.
- `--max-steps-per-shot <int>` (default `2048`): hard-limit simulation steps per attempt.
- `--quantum-measure` / `--no-quantum-measure` (default enabled): control automatic quantum measurements.
- `--measurement-gamma <float>` (default `1.0`): tune the measurement sampling sharpness.
- `--sink-rule <prob_threshold|measurement>` (default `prob_threshold`): pick the sink resolution rule.
- `--edge-boundary <reflect|absorb>` (default `reflect`): decide how the arena boundary behaves.
- `--boost-hole` / `--no-boost-hole` (default enabled): toggle probability boosting toward the hole.
- `--boost-factor <float>` (default `0.10`): set the baseline boost factor when boosting is enabled.
- `--boost-increment <float>` (default `0.08`): increment applied after each measurement when boosting is active.

### Performance & Rendering
- `--perf-profile <quality|balanced|fast>` (default `quality`): apply a preset bundle of performance flags.
- `--blit` / `--no-blit` (default off): force matplotlib blitting on or off.
- `--gpu-viz` / `--no-gpu-viz` (default off): control GPU-backed visualisation if a CUDA device is present.

### Recording & Automation
- `--record-video [PATH]` (default disabled): render the scripted double-slit demo instead of launching the UI. Provide a path to override the destination.
- `--record-output <PATH>` (default `QuantumMinigolfDemo.mp4`): specify the output file when recording (takes precedence over `--record-video`).
- `--headless` (default off): run with the Agg backend and skip the Qt window.

### Diagnostics & Backend
- `--dump-config` (default off): print the resolved `GameConfig` object before launching.
- `--backend <auto|cpu|gpu>` (default `auto`): override FFT backend selection (`QUANTUM_MINIGOLF_BACKEND`). 

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
| `K` | Arm or cancel automatic recording for the next shot (replays it after completion). |
| `V` | Re-simulate and record the previous shot. |
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
Run `python run_game.py --record-video` (optionally with `--record-output PATH`) to invoke the same capture pipeline before exiting the launcher.

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
