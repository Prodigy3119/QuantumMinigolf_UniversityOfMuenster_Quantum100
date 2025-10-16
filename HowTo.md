Quantum Minigolf – Quick Play Guide
===================================

Launch Essentials
-----------------
- From the project root run `python run_game.py`. The default starts the mixed (quantum + classical) demo with tracker support enabled if a calibration file is available.
- Use `python run_game.py --dump-config` to print the resolved settings before the window appears. Helpful when adjusting flags.
- Common launch modifiers:
  - `--mode <classical|quantum|mixed>` – pick your starting visualization mode.
  - `--map <double_slit|single_slit|single_wall|no_obstacle>` – choose the initial course layout.
  - `--mouse-swing` or `--no-vr` – force mouse-only swings even if the tracker is configured.
  - `--vr` – insist on tracker-driven swings (mouse indicator is disabled unless re-enabled in-game).
  - `--calibration-path FILE` – load a specific tracker calibration JSON/PKL.
  - `--skip-calibration-preview` – suppress the snapshot viewer during startup.
  - `--target-fps N` or `--draw-every N` – influence rendering cadence if you need smoother playback.
- Scripted demo: `python run_game.py --record-video` renders the showcase sequence to MP4 and exits.

Calibration Flow
----------------
1. Use `python run_game.py --calibrate-course` (manual corner picking) or `--calibrate-course-led` (LED detection) to capture a board homography. Both tools save `calibration/course_calibration.{json,pkl}`.
2. On game start the calibration is auto-loaded (or from `--calibration-path`). A preview figure lets you confirm the mapping; close it to continue.
3. Once calibrated, the tracker projects detected LEDs into course coordinates so the virtual club aligns with the table. The magenta debug circle represents the collision target; if it drifts, recalibrate.

Core Gameplay Loop
------------------
1. Swing Source:
   - **Tracker**: real putter with twin LEDs; hits register when the detected rectangle crosses the ball center. Area limit and speed thresholds keep noisy detections from triggering.
   - **Mouse**: click-drag inside the arena when mouse mode is enabled (`G`). The purple indicator defines your swing arc; release to shoot.
2. Shot happens in both classical and quantum simulations (depending on mode). Watch the ball path, wave evolution, and measurement feedback.
3. End-of-shot overlays (measurement point, classical marker, etc.) highlight results. Press `R` to reset or wait for automatic reset after game over.

Must-Know Hotkeys
-----------------
- `Q` – Quit the session.
- `R` – Reset the current course (double-tap once to restore normal state if a special mode is active).
- `Tab` – Cycle courses.
- `C` – Cycle display modes (classical, quantum, mixed).
- `M` – Trigger an immediate quantum measurement when allowed.
- `I` – Toggle info overlays (covariance ellipses, measurement text).
- `B` – Switch edge boundary behavior (reflect vs absorb).
- `W` – Toggle wave initial profile (packet/front).
- `T` – Toggle shot stop mode (time limited vs friction).
- `G` – Toggle mouse swing. When ON the tracker is forced off (no virtual putter or tracker hits).
- `O` – Toggle the on-screen tracker overlay (does not change hit detection while tracker is active).
- `U` – Show/hide the control panel window.
- `L` – Toggle interference pattern plot.
- `D` – Play the scripted recording once (if available).
- `#` / `-` – Cycle pre-made multi-stage demos.
- `H` – Print hotkey summary in the terminal.

Control Panel Sliders
---------------------
Open the panel with `U` and tune during runtime:
- **Boost Increment** – Adjusts the bias toward the hole applied after measurements.
- **Move Speed** – Multiplies ball/wave travel after each shot.
- **Shot Time** – Caps simulation time (set to 0 or drag to “inf” for no limit).
- **Wall Thickness** – Scales central barrier width/height for the current map.
- **Tracker Threshold** – Binary threshold for LED detection; tweak if lighting changes.
- **Tracker Speed** – Multiplier converting tracker-measured speed to in-game swing strength.
- **Tracker Max Area** – New limit on the virtual club’s area in course space; when exceeded the overlay hides and tracker hits are ignored (0 ⇒ disabled).

Tracker Expectations & Tips
---------------------------
- The club overlay updates at ~30 FPS; keep sweeps smooth. Gentle forward motion still registers thanks to closing-speed detection, but ensure the swing path actually intersects the magenta circle.
- If the virtual club looks too big, adjust `Tracker Max Area` or stretch/rotate the LEDs in hardware so the detected span matches the real head.
- For debugging, enable the OpenCV window (default) to monitor LED contours, contact point (yellow dot), and the magenta impact circle.
- Press `O` if you want to hide the overlay while keeping tracker hits active (useful when recording footage).

Course & Mode Notes
-------------------
- Courses are variations of the double-slit minigolf table; barriers are quantum potentials that also act as classical obstacles.
- Classical mode integrates a simple friction model; quantum mode evolves the wave packet. Mixed mode draws both, sharing the shot parameters.
- The hole sits near the right edge; probability boosting can speed up sinks but may mask weak shots if set too high.
- Absorbing boundaries help remove stray probability; switch to reflect if you want the wave to bounce around for longer.

Troubleshooting at a Glance
---------------------------
- **No tracker hits**: verify calibration, lower `Tracker Max Area`, check the debug window for the magenta circle alignment, ensure `G` is off.
- **Tracker hitting while off-screen**: Reduce `Tracker Max Area` or adjust `Tracker Threshold` so background blobs aren’t picked up.
- **Mouse swing inactive**: confirm you’re inside the axes, ensure `G` is on (mouse mode), and that no shot is currently running (`shot_in_progress`).

Enjoy experimenting—mix quantum wave antics with precise putting, tweak parameters live, and capture demos once you have a favorite setup!
