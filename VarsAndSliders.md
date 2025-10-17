# Vars & Controls Reference

This reference summarises every configurable variable (`GameConfig` and `PerformanceFlags`), every CLI flag, and each control-panel slider. For each item you’ll find a short description plus what happens when you push the value up or down (or enable/disable it).

## GameConfig Variables

### Grid & Timing

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `Nx` | Horizontal simulation cells; sets course resolution. | Finer spatial detail but heavier CPU/GPU load. | Coarser geometry with faster simulation but chunkier visuals. |
| `Ny` | Vertical simulation cells; pairs with `Nx`. | More vertical detail at the cost of performance. | Less detail and smoother performance. |
| `dx` | Physical width per grid cell. | Enlarges the simulated course and slows propagation. | Shrinks the virtual course and speeds up motion. |
| `dy` | Physical height per grid cell. | Taller effective arena; slower vertical dynamics. | Compresses vertical scale; faster vertical motion. |
| `dt` | Base physics timestep. | Larger steps increase speed but risk instability. | Smaller steps stabilise the solver but need more iterations. |
| `steps_per_shot` | Nominal simulation steps per swing. | Longer ball lifetime before auto-reset. | Shots end sooner; less time to observe behaviour. |
| `max_steps_per_shot` | Hard ceiling on steps. | Allows extended rallies at higher cost. | Forces resets earlier, guarding performance. |
| `draw_every` | Frames skipped between renders. | Fewer redraws → faster but choppier animation. | More redraws → smoother visuals but slower. |

### Potentials & Obstacles

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `V_edge` | Potential wall at arena boundary. | Stronger reflection/containment; harder to leak out. | Softer edges; waves creep outward more. |
| `V_wall` | Potential for central barriers. | Tougher barrier, tighter quantum tunnelling. | Easier passage through walls/slots. |
| `single_wall_width` | Width (cells) for single barrier map. | Broader wall, increasing obstruction. | Slimmer wall that’s easier to bypass. |
| `single_wall_thickness_factor` | Multiplier for wall width. | Thickens obstacles proportionally. | Thinner walls, larger openings. |
| `slit_height` | Aperture height for slit maps. | Wider slit allowing more vertical spread. | Narrow slit causing stronger diffraction. |
| `slit_sep` | Distance between slits. | Greater spacing, widening interference pattern. | Closer slits, tighter interference fringes. |
| `center_wall_width` | Thickness of the central plate. | Heavier obstruction between slits. | Lighter obstruction, easier passage. |

### Hole & Course Layout

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `hole_r` | Radius of the hole capture zone. | Easier to sink shots; larger target. | Harder to score; smaller capture area. |
| `ball_start_x_frac` | Initial ball X as fraction of width. | Starts closer to the hole/right edge. | Starts nearer the left tee. |

### Boundaries & Absorption

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `absorb_width` | Absorbing strip width at edges. | Wider damping zone killing reflections. | Narrow strip; more boundary bounce-back. |
| `absorb_strength` | Strength of absorbing border. | Faster decay of outbound waves. | Weaker absorption, more ringing. |
| `edge_boundary` | Boundary condition (reflect/absorb). | `reflect` keeps energy in bounds. | `absorb` drains energy at the walls. |
| `edge_reflect_cells` | Thickness of reflective layer. | Increased buffer preventing leakage. | Thinner layer; easier to escape. |
| `edge_reflect_height` | Potential height for reflecting edge. | Strong push-back at border. | Softer reflection. |

### Measurement & Sinking

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `quantum_measure` | Auto-collapse of the wave when triggers fire. | Enables measurement events for hybrid modes. | Disables automatic measurements. |
| `measure_gamma` | Sharpness of measurement sampling. | Collapses tightly around peaks. | Produces broader, fuzzier measurements. |
| `sink_rule` | Method used to detect a hole. | `prob_threshold` checks probability mass. | `measurement` waits for explicit collapse hits. |
| `sink_prob_threshold` | Probability needed to count as holed. | Requires more certainty to score. | Easier to register makes. |
| `measurement_sink_min_prob` | Minimum probability considered for measurement sink. | Filters out tiny probabilities. | Accepts even faint measurement hits. |

### Controls & Logging

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `enable_mouse_swing` | Allows mouse-based swings alongside tracker. | Enables desktop control. | Tracker-only input. |
| `multiple_shots` | Keeps the ball where it stopped until holed. | Sequential shot challenges. | Ball resets after each attempt. |
| `log_data` | Records tracker telemetry. | Writes periodic data to `vr_debug_log.txt`. | Suppresses logging overhead. |

### Tunnelling & Swing Mapping

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `kmin_frac` | Minimum k-vector fraction for slow swings. | Raises baseline energy, boosting soft taps. | Allows gentler swings to travel less. |
| `kmax_frac` | Maximum k-vector fraction. | Higher cap allows harder hits before aliasing. | Caps extreme swings sooner. |
| `tunneling_speed_weight` | Influence of swing speed on tunnelling. | Speed has stronger impact on barrier penetration. | Thickness dominates tunnelling behaviour. |
| `barrier_thickness_power` | Exponent scaling wall strength. | Thick barriers become dramatically harder. | Wall strength grows more gently with thickness. |
| `tunneling_thickness_weight` | Weight of thickness in tunnelling mix. | Wall geometry heavily influences outcome. | Speed dominates tunnelling probability. |

### Wave Initialisation

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `wave_initial_profile` | Starting wave shape (`packet` or `front`). | Choose `front` for planar launches. | Choose `packet` for localized starter waves. |
| `wavefront_transition_len` | Smoothing length when using `front`. | Softer transition from front to rest. | Sharper front edge, more ringing. |
| `wavefront_sigma_y` | Vertical spread of initial wave. | Broader vertical coverage. | Narrow beam at launch. |
| `wavefront_sigma_forward` | Forward decay for wavefront profile. | Longer wake before decay. | Shorter tail, more localized start. |

### Shot Termination & Friction

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `shot_stop_mode` | Decide between time-based or friction-based cutoff. | `friction` stops when speed drops sufficiently. | `time` stops after the step quota. |
| `shot_friction_linear` | Linear drag term. | Faster slowing even at gentle speeds. | Lets the ball roll longer. |
| `shot_friction_quadratic` | Quadratic drag term. | Hits fast swings harder. | Preserves high-speed motion. |
| `shot_friction_cubic` | High-order drag term. | Strong braking at very high velocity. | Less braking on powerful hits. |
| `shot_friction_min_scale` | Lower bound for combined friction multiplier. | Prevents friction from getting too low. | Allows shots to coast more when slowed. |

### Probability Boosting

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `boost_hole_probability` | Toggles probability-boost mechanic. | Hole probability increases after misses. | No artificial bias toward sinking. |
| `boost_hole_probability_factor` | Base boost amount. | Large constant nudge toward the cup. | Minimal assist from the booster. |
| `boost_hole_probability_increment` | Per-measurement boost. | Booster ramps up quickly on repeated tries. | Slower ramp that feels more natural. |
| `boost_hole_probability_autoincrement_on_measure` | Auto-bump boost when measurements happen. | Guarantees gradual assistance. | Keeps boost fixed unless manually adjusted. |

### Visibility & Rendering

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `PlotBall` | Draw the classical ball. | Ball remains visible. | Hides the ball marker. |
| `PlotWavePackage` | Draw the quantum wave heatmap. | Shows quantum field. | Removes wave overlay (classic-only look). |
| `smooth_passes` | Extra blur passes on the wave. | Softer visuals, slight cost. | Sharper but noisier rendering. |
| `vis_interpolation` | Matplotlib interpolation mode. | `bilinear`/`bicubic` smooth the texture. | `nearest` keeps pixelated look (faster). |
| `display_downsample_factor` | Downsample factor for display pipeline. | Smaller render buffer, faster but softer. | Higher fidelity at cost of speed. |
| `low_dpi_value` | DPI for low-DPI mode. | Larger numbers make UI crisper. | Smaller numbers lighten load but blur UI. |
| `target_fps` | Desired redraw rate. | Aims for smoother animation, more CPU. | Frees CPU, may look choppier. |
| `debounce_ms` | Mouse event debounce. | Prevents double-firing at cost of responsiveness. | Snappier controls but risk of double hits. |
| `path_decimation_stride` | Sampling stride for overlay paths. | Coarser path that renders faster. | Denser path, cleaner trails but slower. |
| `overlay_every` | Frames between overlay updates. | Less frequent overlay refresh, saving time. | Overlay updates every frame for maximum fidelity. |

### Motion & Swing Scaling

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `movement_speed_scale` | Multiplier applied after a shot. | Ball and wave travel faster across the grid. | Slower movement for precision play. |
| `swing_power_scale` | Converts tracker speed to shot energy. | Smaller wrist motion produces big shots. | Requires stronger swings to launch. |
| `impact_min_speed` | Minimum tracker speed to trigger a hit. | Filters weak swings to reduce false hits. | Allows gentle taps to register. |

### Tracker Integration

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `use_tracker` | Toggles LED putter tracking. | Tracker drives swings. | Mouse-only play. |
| `tracker_speed_scale` | Converts tracker velocity to in-game power. | Faster ball for the same real swing. | Dampened in-game response. |
| `tracker_threshold` | Brightness threshold for LED detection. | Ignores dim background noise. | More sensitive but can mis-detect lights. |
| `tracker_length_scale` | Scales overlay putter length. | Longer virtual club for the same LED spacing. | Shorter overlay to match smaller rigs. |
| `tracker_thickness_scale` | Scales overlay putter width. | Thicker virtual paddle. | Slimmer overlay. |
| `tracker_min_span_px` | Minimum LED spacing to consider valid. | Rejects cramped LED detections. | Accepts shorter tracked spans. |
| `tracker_overlay_thickness_px` | Baseline overlay thickness in pixels. | Visually thicker putter. | Thinner overlay. |
| 	racker_coord_margin | Extra padding when clamping tracker coordinates. | Lets overlay drift slightly beyond borders to avoid lag. | Strict clamp that snaps the putter inside the board. |
| `tracker_area_limit` | Maximum allowed overlay area before suppression. | Larger puts allowed before hits suppressed. | Tight limit used to avoid covering the hole. |
| `tracker_debug_window` | Show OpenCV debug window. | Visual feedback while tracking. | Keeps desktop clutter-free. |
| `tracker_crop_x1/track_crop_x2/...` | Manual ROI for the camera frame. | Smaller ROI speeds processing, must include LEDs. | Leave `None` for full-frame capture. |
| `tracker_calibration_path` | Path to course calibration file. | Loads specific homography. | Auto-search default locations. |
| `tracker_auto_scale` | Learns scale/offset from observed motion. | Auto-calibrates when no homography exists. | Relies solely on provided calibration. |
| `tracker_max_span_px` | Max LED span before rejecting frame. | Allows very wide stance clubs to register. | Rejects frames when LEDs separate too far. |
| `tracker_coord_margin` | Extra margin when clamping tracker coordinates. | Lets overlay drift slightly past borders, reducing drift. | Tighter clamp that snaps overlay inside the board. |

### Timing & Playback

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `shot_time_limit` | Time quota per shot (seconds). | Allows drawn-out rallies. | Ends a shot sooner. |
| `video_playback_speed` | Playback speed multiplier for recordings. | Faster review of recorded shots. | Slower, more detailed review. |

### Map & Resolution

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `map_kind` | Starting course layout. | Switch to obstacle-heavy courses. | Choose simplified layouts. |
| `res_scale` | Multiplier for simulation grid size. | High-res simulation with extra detail. | Lower resolution for speed. |

### Swing & Visual Constants

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `ball_r` | Ball radius for drawing/collision. | Larger ball gives visual emphasis. | Smaller ball for delicate visuals. |
| `indicator_r` | Aim indicator radius. | Easier to track pointer. | Minimal indicator for precision. |
| `sigma0` | Default wave packet spread. | Broader initial wave. | Tighter wave focus. |
| `perf_sigma0` | Reduced spread used in performance modes. | Slightly bigger fallback packet. | Tighter fallback for speed runs. |
| `perf_steps_factor` | Step reduction ratio in performance mode. | Keeps more physics steps while still accelerating. | Heavier trimming for FPS recovery. |

### Background & Overlay

| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `background_image_path` | Path to background texture. | Loads the specified image. | Default black backdrop. |
| `background_image_alpha` | Opacity of the background image. | More visible texture. | More of the default dark background. |
| `wave_overlay_alpha` | Opacity of wave heatmap. | Stronger overlay, hides background. | Fainter overlay, lets background shine through. |

### PerformanceFlags

| Flag | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `blitting` | Reuses background buffers in Matplotlib. | Faster redraws, especially for static backgrounds. | Safer when artist count changes often. |
| `display_downsample` | Downsample images before display. | Reduces bandwidth to UI; blurrier picture. | Highest fidelity but heavier load. |
| `gpu_viz` | Uses GPU pipeline for visuals. | Offloads drawing to GPU when available. | CPU-only rendering. |
| `low_dpi` | Use reduced DPI for the window. | Lighter on older GPUs/monitors. | Full native DPI. |
| `inplace_step` | Reuse arrays in physics solver. | Less memory churn, faster. | Safer when debugging modifications. |
| `adaptive_draw` | Skip frames adaptively when busy. | Keeps UI responsive under load. | Renders every frame regardless of cost. |
| `path_decimation` | Simplify stroke paths before drawing. | Less detail but faster overlay. | Full detail for precise review. |
| `event_debounce` | Debounce Matplotlib events. | Prevents double-firing but adds delay. | Maximum responsiveness. |
| `fast_blur` | Use faster (approximate) blur kernels. | Saves CPU with slight visual trade-off. | Highest-quality blur at extra cost. |
| `pixel_upscale` | Fake pixel-art scaling for wave texture. | Crisp retro look and less interpolation. | Smooth interpolation between cells. |

## Command-Line Flags

| Flag | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| `--map` | Picks the starting obstacle layout. | Choose complex maps for extra challenge. | Choose simpler layouts for quick demos. |
| `--mode` | Sets initial visualization mode. | `quantum`/`mixed` show wave effects. | `classical` keeps things familiar. |
| `--course` | Launches a scripted multi-stage course. | Jump straight into guided scenarios. | Free-play mode. |
| `--wave-profile` | Chooses starting wavefront shape. | `front` for wall of energy. | `packet` for localized launch. |
| `--stop-mode` | Select shot termination rule. | `friction` ends when motion stops. | `time` uses fixed duration. |
| `--wall-thickness` | Scales central wall thickness. | Narrow the channel; harder shots. | Wider gaps; easier tunnelling. |
| `--movement-speed` | Overrides `movement_speed_scale`. | Speeds traversal between collisions. | Slows the overall gameplay pace. |
| `--shot-time` | Overrides `shot_time_limit`. | Longer shot life; marathon rallies. | Shorter attempts; quick resets. |
| `--sink-threshold` | Sets sink probability threshold. | Requires higher certainty before sinking. | Easier to score with lower bar. |
| `--boost-increment` | Overrides per-measurement boost. | Booster ramps up faster. | Subtle assistance. |
| `--boost-factor` | Overrides baseline boost. | Higher constant bias toward the hole. | Minimal or no bias. |
| `--mouse-swing` | Forces mouse control on. | Desktop swings always available. | Tracker takes priority when present. |
| `--multiple-shots` | Keeps ball position between turns. | Campaign-style chained attempts. | Classic reset after each stroke. |
| `--log-data` | Enables tracker telemetry logging. | Generates detailed vr-debug logs. | No logging overhead. |
| `--background-path` | Loads a specific background image. | Supplies a custom texture. | Falls back to black backdrop. |
| --performance-increase | Enables the experimental performance toolkit. | Activates GPU preference, reduced resolution, cached operators. | Keeps the default feature set. |
| --fast-mode-paraxial | Optional arcade/paraxial propagation (perf-mode only). | Trades accuracy for additional speed-ups. | Ignores the flag and uses standard propagation. |
| `--no-tracker-auto-scale` | Disables automatic tracker scaling. | Honors provided calibration entirely. | Lets tracker self-calibrate. |
| `--tracker-max-span` | Sets max LED separation before rejecting. | Accepts wide stances. | Enforces compact club movement. |
| `--config-panel` | Opens slider control panel on start. | Immediate access to runtime tuning. | Panel stays hidden unless toggled. |
| `--no-control-panel` | Keeps control panel hidden. | Prevents accidental panel popups. | Panel can still be opened manually. |
| `--quantum-measure` | Turns automatic quantum measurements on. | Hybrids collapse automatically. | Requires manual triggering. |
| `--no-quantum-measure` | Forces measurements off. | Purely classical/visual play. | Automatic collapse stays available. |
| `--measurement-gamma` | Adjusts measurement sharpness. | Tighter collapses around peaks. | Broader sampling. |
| `--sink-rule` | Chooses scoring rule. | `measurement` waits for collapse hits. | `prob_threshold` uses probability mass. |
| `--edge-boundary` | Chooses boundary type. | `reflect` keeps energy inside. | `absorb` drains it away. |
| `--max-steps-per-shot` | Overrides `max_steps_per_shot`. | Longer computations per stroke. | Early resets to guard frame rate. |
| `--perf-profile` | Applies performance presets. | `fast` trades fidelity for speed. | `quality` keeps visuals pristine. |
| `--blit` / `--no-blit` | Force-enable/disable blitting. | Speeds up rendering when stable. | Safer when artists change frequently. |
| `--gpu-viz` / `--no-gpu-viz` | Toggle GPU-backed visuals. | Attempt GPU acceleration. | Stay on CPU for compatibility. |
| `--target-fps` | Override target FPS. | Smoother animation target. | Lower demand for weak hardware. |
| `--draw-every` | Override draw skip factor. | Skip more frames to boost speed. | Draw every frame for smoothness. |
| `--res-scale` | Rescale grid resolution. | Higher detail, slower sim. | Lower detail, faster sim. |
| `--vr` / `--no-vr` | Toggle tracker swing input globally. | Fully VR putter control. | Mouse-only control. |
| `--display-tracker` / `--no-display-tracker` | Show/hide tracker debug window. | Visual feedback while tuning. | Cleaner desktop. |
| `--calibrate-course` | Run manual corner picker. | Captures fresh homography. | Skips manual calibration. |
| `--calibrate-course-led` | Run LED auto-calibration. | Auto-detect LEDs to build homography. | Relies on existing calibration. |
| `--calibration-path` | Specify calibration file. | Explicit file loaded each run. | Auto-discover fallback paths. |
| `--skip-calibration-preview` | Skip post-calibration preview window. | Faster startup; no visual check. | Preview ensures alignment. |
| `--record-video` | Render scripted demo to MP4. | Generates showcase video (optional path). | Launch interactive app. |
| `--record-output` | Path for `--record-video` output. | Directs MP4 to chosen file. | Uses default demo filename. |
| `--headless` | Use non-interactive backend. | Run without opening a GUI (useful for CI). | GUI launches as normal. |
| `--boost-hole` / `--no-boost-hole` | Toggle probability boosting. | Enables auto-assist after misses. | Keeps strict physics. |
| `--dump-config` | Print resolved `GameConfig`. | Useful for logging/debugging. | Keeps console quieter. |
| `--backend` | Force physics backend (`auto`, `cpu`, `gpu`). | Choose GPU for acceleration. | Choose CPU for compatibility. |

## Control Panel Sliders

| Slider | Purpose | Increase | Decrease |
| --- | --- | --- | --- |
| `Boost Increment` | Adjusts `boost_hole_probability_increment`. | Booster ramps up faster after each measurement. | Booster grows slowly, keeping play fair. |
| `Move Speed` | Tweak `movement_speed_scale`. | Ball and wave travel farther per shot. | Shots advance gently for precision. |
| `Shot Time` | Modifies `shot_time_limit`. | Shots stay alive longer before reset. | Shots end sooner to keep pace brisk. |
| `Wall Thickness` | Adjusts `single_wall_thickness_factor`. | Walls bulk up, narrowing channels. | Wider openings through the barrier. |
| `Tracker Threshold` | Sets `tracker_threshold`. | Ignores dim speckles; needs brighter LEDs. | Accepts dim LEDs but risks noise. |
| `Tracker Speed` | Controls `tracker_speed_scale`. | Amplifies swing-to-shot power mapping. | Softens swing impact. |
| `Tracker Max Area` | Sets `tracker_area_limit`. | Allows large overlay area before suppressing hits. | Prevents swings when the club covers too much of the board. |

---

Use this sheet as a quick cheat sheet whenever you fine-tune the simulator, script launches, or explain the UI to newcomers. Feel free to extend it with your own project-specific defaults or presets.***

### Performance Mode Controls
| Setting | Purpose | Higher / Enabled | Lower / Disabled |
| --- | --- | --- | --- |
| performance_increase | Master performance toggle. | Enables the full optimisation suite. | Uses the original high-fidelity pipeline. |
| 
ast_mode_paraxial | Optional paraxial propagator. | Switches to an arcade-style approximation. | Keeps the standard split-step solver. |
| performance_theta | Phase-budget safety factor (radians). | Permits larger stable timesteps. | Keeps timesteps conservative. |
| performance_dt_tolerance | Tolerance before rebuilding exponent operators. | Reuses cached operators longer. | Rebuilds more often for accuracy. |
| performance_friction_bins | LUT bins for friction-speed scaling. | Smoother interpolation of exponent pairs. | Coarser quantisation, less memory. |
| performance_res_scale | Default resolution scale when perf mode engages. | Stronger downscale for speed. | Retains higher fidelity. |
| performance_display_downsample | Minimum display downsample factor. | Smaller buffers, quicker rendering. | Full-resolution frames. |
| performance_smooth_passes | Blur passes during perf mode. | Adds polish when needed. | Zero for maximum performance. |
| performance_draw_every | Baseline draw stride under perf mode. | Skips more frames to stabilise FPS. | Renders more often for richer detail. |
| performance_drift_threshold | Density threshold to allow burst drift. | Burst drift triggers less often. | Aggressive acceleration far from obstacles. |
| performance_max_drift_steps | Cap on burst-drift length. | Longer bursts per FFT pair. | Short bursts, closer to baseline feel. |
| performance_enable_window | Moving-window optimiser switch. | Attempts cropped evolution when supported. | Sticks to full-grid evolution. |
| performance_window_margin | Safety padding for the moving window. | Wider margin reduces clipping risk. | Tighter margin for maximum speed-up. |