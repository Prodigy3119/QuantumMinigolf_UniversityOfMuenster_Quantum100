# Quantum Minigolf – Quick Operator Guide

## Launch

1. **Launch the “Anaconda Prompt”**

2. **Activate the project environment**

   ```bat
   conda activate QMENV
   ```

3. **Change into the project directory**

   ```bat
   cd "C:\...\00 Quantum Minigolf"
   ```

4. **Start game in VR with calibration**

   ```bat
   python run_game.py --vr --calibrate-course --choose-camera --display-tracker
   ```

   * `--calibrate-course` walks you through mapping camera pixels → table coordinates → projector coordinates.
   * On restart if the hardware was not moved you can drop `--calibrate-course`.

5. **Calibration**

   * Select the four physical course corners in order:

     1. top-left
     2. top-right
     3. bottom-right
     4. bottom-left
   * Review the preview frame.
   * Confirm to save (this writes out something like `calibration/course_calibration.json` / `.pkl`, which the tracker then reloads automatically on startup to know geometry and frame size). 

6. **Game start**

   * After calibration, the main Quantum Minigolf game window opens (this is the projected course + live ball/wave simulation).
   * A second window **“Quantum Mini-Golf – Control Panel”** opens with live sliders for tuning physics and tracking. This panel is created from inside the game using `matplotlib.widgets.Slider` and is tracked by `self._config_panel_active`. You can close/reopen it at any time with the `u` hotkey. 
   * If tracking is enabled, the code also spawns a “Tracker Debug” feed that shows what the camera sees, the two LED positions, and the inferred putter rectangle; this uses OpenCV windows named things like `"Tracker Debug"` / `"LEDs"`. 

## Live Control Panel Sliders

The Control Panel is a live tuning surface. Each slider writes straight into the active `GameConfig` (e.g. `movement_speed_scale`, `shot_time_limit`, `single_wall_thickness_factor`, `tracker_threshold`, etc.), and the game immediately re-applies those values to physics, rendering, and tracking — no restart required. The panel is built at runtime with `matplotlib.widgets.Slider` and wired so that moving a slider calls a handler like `_on_wall_thickness_change`, `_on_tracker_threshold_change`, and so on.  

| Slider                        | Functionality                                                                                                                                                                                                                                                                                                                       | Increase value →                                                                                                                                 | Decrease value →                                                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| **Move Speed** (2–25×)        | Global speed multiplier for post-shot motion (`movement_speed_scale`). The game uses this to scale how fast the ball & quantum wave travel after impact, essentially a fast-forward / slow-mo timescale. Internally it also renormalizes swing tuning so difficulty doesn’t secretly change when you crank it.                      | Faster ball/wave propagation and generally snappier demos for outreach mode (high-energy “wow” runs).                                            | Slower motion for teaching/analysis; makes diffraction, tunneling, and interference easier to point at in real time.       |
| **Shot Time** (sec)           | How long one shot is allowed to run before the game auto-stops / resets, in simulation seconds (`shot_time_limit`). If `shot_stop_mode` is set to `"time"`, this is the main cutoff.                                                                                                                                                | Shots keep evolving longer before auto-reset, so you can watch the wave spread and interact with obstacles for more time.                        | Shots terminate sooner, good for rapid-fire demos or if you don’t want the wave drifting forever.                          |
| **Wall Thickness** (scale)    | Scales the obstacle wall thickness/height for “single wall / slit / logo” style maps (`single_wall_thickness_factor`). Thicker walls become more classically solid and reduce tunneling; thinner walls become more penetrable/quantum-leaky. The game immediately rebuilds the course geometry and barrier exponents and redraws.   | Beefier barrier → harder for the quantum wave to tunnel through, more “solid wall” behavior.                                                     | Skinny / weak barrier → easier tunneling and more visible interference past the slit or wall.                              |
| **LED Recognition Threshold** | Brightness cutoff for LED detection in the camera feed (`tracker_threshold`). The tracking loop thresholds the grayscale frame and looks for two bright blobs (the LEDs on the putter) to infer swing angle, speed, and impact.                                                                                                     | Requires brighter pixels to count as “LED,” which filters noise / false reflections and stabilizes tracking in messy lighting.                   | More sensitive detection in darker rooms or with dimmer LEDs — but also more risk of picking up shiny junk as “the club.”  |
| **Putter Speed Increase**     | Multiplier that maps measured putter swing speed (from the tracker) into shot power (`tracker_speed_scale`, relative to a base `tracker_speed_base`).                                                                                                                                                                               | Even a gentle tap launches the ball/wave with more kinetic energy — dramatic long shots with small wrist flicks.                                 | You have to really swing to get distance; good for tighter demos in small projection areas.                                |
| **Putter Size (scale)**       | Visual calibration for the on-screen putter overlay. This rescales the projected club rectangle (length & thickness) so the holographic “ghost putter” aligns with the real physical putter (`tracker_length_scale`, `tracker_thickness_scale`). The change is pushed both into the renderer and the live tracker config.           | Overlay club looks longer/thicker -> easier for the audience to see, and helpful if the projection looks too short compared to the real putter.  | Overlay club shrinks -> use if the drawn putter is oversized versus the real stick, so alignment feels “true to size.”     |

## Hotkeys

These work in the main game window (the projected course / matplotlib figure). Press `h` at any time to print an on-console cheat sheet with current states (mode, map, overlay status, etc.).

* `q` – <span style="color:#9cdcfe;font-weight:bold;">Quit the game</span> (closes the main window).
* `f` – <span style="color:#9cdcfe;font-weight:bold;">Dump display + tracker diagnostics</span> to the console and refresh the debug info overlay. This logs things like display sizes, tracker camera frame size, etc., then schedules a redraw.  
* `r` – <span style="color:#9cdcfe;font-weight:bold;">Reset / abort current shot</span>: stop the current shot, restore normal course+mode settings if you were in a scripted “demo stage,” and respawn ball and wave.  
* `Tab` – <span style="color:#9cdcfe;font-weight:bold;">Cycle obstacle map</span> (double slit ↔ single slit ↔ wall ↔ logo ↔ no obstacle). Updates the course geometry and re-draws.  
* `c` – <span style="color:#9cdcfe;font-weight:bold;">Cycle display mode</span> (classical ↔ quantum ↔ mixed). Lets you flip between “only ball,” “only wave,” or both at once. 
* `m` – <span style="color:#9cdcfe;font-weight:bold;">Force a quantum measurement</span>. If the mode allows measurement, collapse/measure the wave and pick where the “ball” actually is.  
* `i` – <span style="color:#9cdcfe;font-weight:bold;">Toggle post-shot info overlay</span>. Shows stats like covariance ellipse of the wave packet, measurement point, and “did it sink?” messaging. (If you press it mid-shot, it arms the overlay to pop up right after the shot finishes.)  
* `#` – <span style="color:#9cdcfe;font-weight:bold;">Advance quantum_demo stage</span> (pre-baked teaching scenes). Press again to advance stages.  
* `-` – <span style="color:#9cdcfe;font-weight:bold;">Advance advanced_showcase stage</span> (flashier / high-intensity presets). Same cycling behavior as `#`.  
* `b` – <span style="color:#9cdcfe;font-weight:bold;">Toggle edge boundary mode</span> between reflective rails and absorbing edges (`edge_boundary`: `"reflect"` ↔ `"absorb"`). This immediately rebuilds the course and restarts the ball.   
* `w` – <span style="color:#9cdcfe;font-weight:bold;">Toggle wave launch profile</span> (`wave_initial_profile`): localized “packet” vs broad “front.” This changes how the quantum wave launches.  
* `t` – <span style="color:#9cdcfe;font-weight:bold;">Toggle shot stop mode</span> (`shot_stop_mode`): `"time"` (hard stop after Shot Time limit) ↔ `"friction"` (wave/ball decays out via frictional damping).   
* `g` – <span style="color:#9cdcfe;font-weight:bold;">Toggle mouse swing control</span>. When ON, you can “swing” with the mouse instead of the physical putter; tracking is force-disabled so you don’t get conflicting inputs. When OFF, control goes back to the real tracker.  
* `o` – <span style="color:#9cdcfe;font-weight:bold;">Toggle tracker overlay visibility</span>. Shows or hides the projected ghost putter / impact zone overlay in the main game window (helpful for alignment).  
* `u` – <span style="color:#9cdcfe;font-weight:bold;">Show / hide Control Panel sliders</span>. If you accidentally closed it, hit `u` to bring it back.  
* `p` – <span style="color:#9cdcfe;font-weight:bold;">Cycle background image</span> behind the simulation (black background ↔ themed backgrounds / logo textures). The game keeps a list of candidate images from `BackgroundImages/` and your config, and steps through them.  
* `l` – <span style="color:#9cdcfe;font-weight:bold;">Toggle interference profile panel</span>. After a shot in quantum/mixed mode, this shows the measured intensity vs position along the exit wall, live-updated in a side axis.   
* `v` – <span style="color:#9cdcfe;font-weight:bold;">Record last shot to video</span> (or frame sequence). This replays the saved shot data in an off-screen replay instance and writes it out.  
* `d` – <span style="color:#9cdcfe;font-weight:bold;">Replay most recent recording</span> inside the main window. Press `d` again to stop playback / close the preview.  
* `h` – <span style="color:#9cdcfe;font-weight:bold;">Print full hotkey help</span> (including current mode, which course preset is active, whether overlay is visible, etc.) to the console.
