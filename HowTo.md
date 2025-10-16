

# 🎮 Quantum Minigolf — Quick Play Guide



## 🧭 Table of Contents
- [🚀 Launch Essentials](#-launch-essentials)
- [📐 Calibration Flow](#-calibration-flow)
- [🔁 Core Gameplay Loop](#-core-gameplay-loop)
- [⌨️ Must-Know Hotkeys](#️-mustknow-hotkeys)
- [🎛️ Control Panel Sliders](#️-control-panel-sliders)
- [🛰️ Tracker Expectations & Tips](#️-tracker-expectations--tips)
- [🗺️ Course & Mode Notes](#️-course--mode-notes)
- [🧰 Troubleshooting at a Glance](#-troubleshooting-at-a-glance)

---

## 🚀 Launch Essentials

```bash
# Quick start (mixed: quantum + classical; tracker enabled if calibration exists)
python run_game.py

# Print resolved settings before window appears (great while tuning flags)
python run_game.py --dump-config
```

### 🏳️ Flags (mix & match)
```bash
# Visualization / content
--mode <classical|quantum|mixed>     # Pick your starting visualization mode
--map <double_slit|single_slit|single_wall|no_obstacle>   # Start course layout

# Input mode
--mouse-swing                         # Force mouse-only swings
--no-vr                               # Alias: disable tracker-driven swings
--vr                                  # Insist on tracker swings (mouse indicator off unless re-enabled in-game)

# Tracker & calibration
--calibration-path <FILE>             # Load specific tracker calibration JSON/PKL
--skip-calibration-preview            # Suppress snapshot viewer during startup

# Rendering cadence
--target-fps <N>                      # Hint desired FPS
--draw-every <N>                      # Render every Nth simulation step for smoother playback

# One-shot showcase: render and exit
--record-video                        # Renders scripted demo to MP4 and quits
```

### 🎥 Scripted Demo
```bash
python run_game.py --record-video
```

---

## 📐 Calibration Flow
1. Capture board homography:
   ```bash
   # Manual corner picking
   python run_game.py --calibrate-course

   # LED detection assisted
   python run_game.py --calibrate-course-led
   ```
   Both save to `calibration/course_calibration.{json,pkl}`.
2. On game start, calibration auto-loads (or from `--calibration-path`). A preview figure appears—close it to continue.
3. The tracker projects LED detections into course coordinates so the virtual club aligns with the table.
   - The **magenta debug circle** is the collision target; if it drifts, recalibrate.

---

## 🔁 Core Gameplay Loop

**1) Swing Source**
```text
TRACKER: Real putter with twin LEDs. A hit registers when the detected rectangle crosses the ball center.
         Area limit & speed thresholds prevent noisy detections.
MOUSE:   Click–drag inside the arena when mouse mode is enabled (G). The purple indicator is your swing arc; release to shoot.
```

**2) Shot Execution**  
The shot runs in both the **classical** and **quantum** simulations (per current mode). Watch ball path, wave evolution, and measurement feedback.

**3) End-of-Shot Overlays**  
Measurement point, classical marker, etc., highlight results. Press **R** to reset or wait for auto-reset after game over.

---

## ⌨️ Must-Know Hotkeys

```text
Q     Quit session
R     Reset current course (double-tap once to restore normal state if a special mode is active)
Tab   Cycle courses
C     Cycle display modes (classical → quantum → mixed)
M     Trigger immediate quantum measurement (when allowed)
I     Toggle info overlays (covariance ellipses, measurement text)
B     Switch edge boundary behavior (reflect ↔ absorb)
W     Toggle wave initial profile (packet ↔ front)
T     Toggle shot stop mode (time-limited ↔ friction)
G     Toggle mouse swing (ON forces tracker OFF: no virtual putter or tracker hits)
O     Toggle on-screen tracker overlay (hit detection unchanged while tracker is active)
U     Show/hide control panel window
L     Toggle interference pattern plot
D     Play the scripted recording once (if available)
#/-   Cycle pre-made multi-stage demos
H     Print hotkey summary in terminal
```

---

## 🎛️ Control Panel Sliders

| Slider | What it does |
|---|---|
| **Boost Increment** | Bias toward the hole applied after measurements. |
| **Move Speed** | Multiplies ball/wave travel after each shot. |
| **Shot Time** | Caps simulation time (set to `0` or drag to “inf” for no limit). |
| **Wall Thickness** | Scales central barrier width/height for the current map. |
| **Tracker Threshold** | Binary threshold for LED detection—tune for lighting changes. |
| **Tracker Speed** | Multiplier converting tracker-measured speed → in-game swing strength. |
| **Tracker Max Area** | Upper bound on virtual club area in course space; exceeding hides overlay & ignores tracker hits (`0` ⇒ disabled). |

Open the panel with **U** and tune during runtime.

---

## 🛰️ Tracker Expectations & Tips

- Club overlay updates at ~**30 FPS**—keep sweeps smooth. Gentle forward motion still registers via closing-speed detection, but ensure the swing path intersects the **magenta circle**.  
- If the virtual club looks oversized, lower **Tracker Max Area** or adjust LED spacing/rotation so detected span matches the real head.  
- For debugging, enable the OpenCV window (default) to inspect LED contours, contact point (yellow dot), and magenta impact circle.  
- Press **O** to hide the overlay while keeping tracker hits active (handy for clean recordings).

---

## 🗺️ Course & Mode Notes

> Courses are variations of the **double-slit minigolf** table; barriers act as quantum potentials and classical obstacles.

- **Classical mode**: integrates a simple friction model.  
- **Quantum mode**: evolves the wave packet.  
- **Mixed mode**: draws both, sharing shot parameters.  
- The hole sits near the right edge; **probability boosting** speeds up sinks but can mask weak shots if set too high.  
- **Absorbing boundaries** clear stray probability; switch to **reflect** if you want the wave to bounce longer.

---

## 🧰 Troubleshooting at a Glance

```text
NO TRACKER HITS
• Verify calibration.
• Lower “Tracker Max Area”.
• Check magenta circle alignment in debug window.
• Ensure G is OFF (tracker enabled).

TRACKER HITTING OFF-SCREEN
• Reduce “Tracker Max Area”.
• Adjust “Tracker Threshold” to avoid background blobs.

MOUSE SWING INACTIVE
• Click inside the axes.
• Ensure G is ON (mouse mode).
• Confirm no shot is currently running (shot_in_progress).
```
