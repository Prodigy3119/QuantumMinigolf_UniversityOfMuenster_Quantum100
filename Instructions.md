# Quantum Minigolf – Quick Operator Guide

Follow these steps to start, calibrate, and run the show within one short session.

## Launch Checklist
1. Open the **Anaconda Prompt** shortcut that ships with Anaconda/Miniconda.
2. Activate the project environment: `conda activate QMENV`.
3. Change into the project: `cd "C:\Users\jhlar\OneDrive\Studium\WWU\Physik\AGW\00 Quantum Minigolf\PythonApproach\00 Quantum Minigolf"`.
4. Connect the camera/projector rig, power on the dual-LED putter, and clear stray reflective objects from the course.
5. Start with a fresh calibration run:\
   `python run_game.py --vr --calibrate-course --display-tracker`\
   (omit `--calibrate-course` on subsequent runs if the course has not moved).
6. When the calibration UI appears, click the four course corners in order (top‑left → top‑right → bottom‑right → bottom‑left), review the preview frame, and confirm to save `calibration/course_calibration.json|.pkl`.
7. The game window plus control panel opens automatically; keep the tracker debug feed visible while test-putting a few swings.

### Reusing or replacing calibration
- To reuse the latest file: `python run_game.py --vr --calibration-path calibration/course_calibration.pkl`.
- To rerun the automatic LED finder instead of manual clicks: `python run_game.py --vr --calibrate-course-led`.
- Skip the preview when you are confident in the alignment: add `--skip-calibration-preview`.

## Live Control Panel Sliders
| Slider | Purpose | Slide Right | Slide Left |
| --- | --- | --- | --- |
| **Boost Increment** | How quickly probability boost ramps after misses. | Aggressive auto-assist, easier sinks. | Gentle assist, more authentic play. |
| **Move Speed** | Scales ball & wave travel distance per shot. | Faster, dramatic swings. | Slower, precision show-and-tell. |
| **Shot Time** | Maximum simulated time per attempt. | Longer rallies before reset. | Snappier turnover between shots. |
| **Wall Thickness** | Central barrier scaling. | Narrows slits/obstacles. | Opens channels for clearer paths. |
| **Tracker Threshold** | LED brightness cut-off. | Ignores dim noise; need brighter LEDs. | Accepts weaker LEDs; may add speckles. |
| **Tracker Speed** | Maps swing velocity to in-game power. | Stronger ball launches. | Softer contact for calibration shots. |

## Essential Hotkeys (main window)
- `q` quit | `f` log display info | `r` reset wave & ball | `tab` cycle obstacle map.
- `c` cycle view (classical ↔ quantum ↔ mixed) | `m` force quantum measurement | `i` toggle info overlay / post-shot measurement.
- `#` cycle quantum demo course | `-` cycle advanced showcase course | `b` toggle reflect/absorb boundary | `w` swap wave packet/front.
- `t` switch shot stop (time/friction) | `g` toggle mouse swing | `o` toggle tracker overlay | `u` show/hide control panel | `p` cycle background art.
- `l` toggle interference profile (after a quantum shot) | `k` arm/cancel automatic recording for the next shot | `v` record the previous shot via replay | `d` play the latest recording | `h` print this hotkey list in the console.

## Rapid Troubleshooting
- **Tracker feed frozen?** Check the console for camera index errors, then relaunch with `--display-tracker` and verify LEDs are within the dashed ROI.
- **Putts feel weak/strong?** Adjust `Tracker Speed` first, then fine-tune `Move Speed` or the CLI flags `--movement-speed` / `--shot-time`.
- **Wave overlay missing?** Press `i` once after a shot; the info overlay only updates when the quantum mode is active.
- **Performance dips?** Lower `Move Speed`, raise `Shot Time` slightly, or launch with `--draw-every 4` to skip extra frames.

Finish by closing the windows (`q`) and deactivating the environment if desired: `conda deactivate`.
