
import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import numpy as np

from quantum_minigolf.calibration import CalibrationData
from quantum_minigolf.config import GameConfig, PerformanceFlags


plt = None  # Will be configured at runtime via _configure_matplotlib.


def _configure_matplotlib(headless: bool) -> None:
    """
    Select an appropriate matplotlib backend before pyplot is imported.
    """
    backend = "Agg" if headless else "QtAgg"
    matplotlib.use(backend)
    global plt  # noqa: PLW0603 - pyplot must be bound once the backend is set
    import matplotlib.pyplot as plt_module  # type: ignore

    plt = plt_module


def _find_course_calibration(explicit: Optional[str] = None) -> Tuple[Optional[CalibrationData], Optional[Path]]:
    """
    Probe a few standard locations for the course calibration file.
    """
    candidates: list[Path] = []
    seen: set[Path] = set()

    if explicit:
        explicit_path = Path(explicit)
        if not explicit_path.is_absolute():
            explicit_path = Path.cwd() / explicit_path
        candidates.append(explicit_path)

    script_dir = Path(__file__).resolve().parent
    search_roots = {
        Path.cwd(),
        script_dir,
        Path.cwd() / "calibration",
        script_dir / "calibration",
    }
    default_names = ("course_calibration.pkl", "course_calibration.json")
    for root in search_roots:
        for name in default_names:
            candidates.append(root / name)

    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            continue
        try:
            calibration = CalibrationData.load(resolved)
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[warn] Failed to load course calibration from {resolved}: {exc}")
            continue
        return calibration, resolved
    return None, None


def _show_calibration_snapshot(
    calibration: CalibrationData,
    *,
    camera_index: int = 0,
    warmup_frames: int = 5,
) -> bool:
    """
    Capture a snapshot from the tracker camera and display both the raw view with the
    detected outline and the warped board preview.
    """
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[warn] Skipping calibration snapshot (OpenCV unavailable: {exc})")
        return False

    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[warn] Could not open camera index {camera_index} for calibration snapshot.")
        return False

    try:
        if calibration.frame_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(calibration.frame_width))
        if calibration.frame_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(calibration.frame_height))

        frame = None
        for _ in range(max(1, warmup_frames)):
            ok, candidate = cap.read()
            if not ok or candidate is None:
                frame = None
                break
            frame = candidate
        if frame is None:
            print("[warn] Unable to capture frame for calibration snapshot.")
            return False

        overlay = frame.copy()
        outline = calibration.outline_in_camera().astype(np.int32)
        if outline.size >= 8:
            pts = outline.reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
            labels = ("TL", "TR", "BR", "BL")
            for idx, (x, y) in enumerate(outline):
                cv2.putText(
                    overlay,
                    labels[idx],
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        H = np.array(calibration.homography, dtype=np.float32).reshape(3, 3)
        board_w = max(1, int(round(calibration.board_width)))
        board_h = max(1, int(round(calibration.board_height)))
        warped = cv2.warpPerspective(frame, H, (board_w, board_h))
    finally:
        cap.release()

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, num="Calibration Snapshot", figsize=(12, 5))
    axes[0].imshow(overlay_rgb)
    axes[0].set_title("Camera View with Outline")
    axes[0].axis("off")
    axes[1].imshow(warped_rgb)
    axes[1].set_title("Warped Course Preview")
    axes[1].axis("off")
    fig.tight_layout()
    fig.canvas.draw_idle()
    return True


def _run_calibration_helper(script_name: str) -> int:
    """
    Launch a standalone calibration helper script using the current interpreter.
    """
    script_path = Path(__file__).resolve().parent / "quantum_minigolf" / script_name
    if not script_path.exists():
        print(f"[error] Calibration helper {script_name} not found at {script_path}")
        return 1
    print(f"[info] Launching {script_name} ...")
    completed = subprocess.run([sys.executable, str(script_path)], check=False)
    if completed.returncode != 0:
        print(f"[error] {script_name} exited with code {completed.returncode}")
    return int(completed.returncode or 0)

MAP_CHOICES = [
    'double_slit', 'single_slit', 'single_wall', 'no_obstacle'
]
MODE_CHOICES = ['classical', 'quantum', 'mixed']
COURSE_CHOICES = ['quantum_demo', 'advanced_showcase']
STOP_CHOICES = ['time', 'friction']
WAVE_CHOICES = ['packet', 'front']


def build_config(args):
    def _flags_for_profile(profile: Optional[str]) -> PerformanceFlags:
        if profile == "fast":
            return PerformanceFlags(
                blitting=True,
                display_downsample=True,
                gpu_viz=False,
                low_dpi=True,
                inplace_step=True,
                adaptive_draw=True,
                path_decimation=True,
                event_debounce=True,
                fast_blur=True,
                pixel_upscale=True,
            )
        if profile == "balanced":
            return PerformanceFlags(
                blitting=True,
                display_downsample=True,
                gpu_viz=False,
                low_dpi=False,
                inplace_step=True,
                adaptive_draw=True,
                path_decimation=True,
                event_debounce=True,
                fast_blur=False,
                pixel_upscale=False,
            )
        return PerformanceFlags(
            blitting=False,
            display_downsample=False,
            gpu_viz=False,
            low_dpi=False,
            inplace_step=False,
            adaptive_draw=False,
            path_decimation=False,
            event_debounce=False,
            fast_blur=False,
            pixel_upscale=False,
        )

    flags = _flags_for_profile(getattr(args, "perf_profile", None))
    if getattr(args, "blitting", None) is not None:
        flags.blitting = bool(args.blitting)
    if getattr(args, "gpu_viz", None) is not None:
        flags.gpu_viz = bool(args.gpu_viz)

    cfg = GameConfig(
        Nx=288, Ny=144, dx=1.0, dy=1.0, dt=0.35,
        steps_per_shot=240, draw_every=3,
        V_edge=200.0, V_wall=80.0,
        single_wall_width=3, slit_height=12, slit_sep=28, center_wall_width=6,
        hole_r=9, ball_start_x_frac=0.25,
        absorb_width=16, absorb_strength=0.04,
        quantum_measure=True, measure_gamma=1.0, sink_rule="prob_threshold",
        boost_hole_probability=True,
        boost_hole_probability_factor=0.10,
        boost_hole_probability_increment=0.08,
        boost_hole_probability_autoincrement_on_measure=True,
        PlotBall=True, PlotWavePackage=True,
        smooth_passes=1, vis_interpolation='bilinear',
        display_downsample_factor=3, low_dpi_value=48, target_fps=30,
        debounce_ms=14, path_decimation_stride=3, overlay_every=3,
        movement_speed_scale=1,
        shot_time_limit=50,
        map_kind='double_slit',
        res_scale=1.0,
        flags=flags,
    )

    cfg.performance_increase = bool(getattr(args, "performance_increase", False))
    cfg.fast_mode_paraxial = bool(getattr(args, "fast_mode_paraxial", False) and cfg.performance_increase)

    if cfg.performance_increase:
        os.environ.setdefault("QUANTUM_MINIGOLF_PERF", "1")
        if "QUANTUM_MINIGOLF_BACKEND" not in os.environ:
            os.environ["QUANTUM_MINIGOLF_BACKEND"] = "gpu"

    if args.map:
        cfg.map_kind = args.map
    if args.wave_profile:
        cfg.wave_initial_profile = args.wave_profile
    if args.stop_mode:
        cfg.shot_stop_mode = args.stop_mode
    if args.sink_threshold is not None:
        cfg.sink_prob_threshold = float(args.sink_threshold)
    if args.wall_thickness is not None:
        cfg.single_wall_thickness_factor = float(args.wall_thickness)
    if args.movement_speed is not None:
        cfg.movement_speed_scale = float(args.movement_speed)
    if args.shot_time is not None:
        cfg.shot_time_limit = None if args.shot_time <= 0 else float(args.shot_time)
    if args.boost_increment is not None:
        cfg.boost_hole_probability_increment = float(args.boost_increment)
        if cfg.boost_hole_probability_increment > 0.0:
            cfg.boost_hole_probability = True
    if args.target_fps is not None:
        cfg.target_fps = float(args.target_fps)
    if args.draw_every is not None:
        cfg.draw_every = max(1, int(args.draw_every))
    if args.res_scale is not None:
        cfg.res_scale = max(0.1, float(args.res_scale))
    if args.quantum_measure is not None:
        cfg.quantum_measure = bool(args.quantum_measure)
    if args.measurement_gamma is not None:
        cfg.measure_gamma = float(args.measurement_gamma)
    if args.sink_rule:
        cfg.sink_rule = args.sink_rule
    if args.edge_boundary:
        cfg.edge_boundary = args.edge_boundary
    if args.max_steps_per_shot is not None:
        cfg.max_steps_per_shot = int(args.max_steps_per_shot)
    if args.boost_factor is not None:
        cfg.boost_hole_probability_factor = float(args.boost_factor)
    if args.boost_hole is not None:
        cfg.boost_hole_probability = bool(args.boost_hole)
    if args.display_tracker is not None:
        cfg.tracker_debug_window = bool(args.display_tracker)
    if args.calibration_path:
        cfg.tracker_calibration_path = str(Path(args.calibration_path).expanduser())
    if args.mouse_swing:
        cfg.enable_mouse_swing = True
    if getattr(args, "background_path", None):
        cfg.background_image_path = str(Path(args.background_path).expanduser())
    if args.no_control_panel:
        cfg.show_control_panel = False
    elif args.config_panel:
        cfg.show_control_panel = True

    if getattr(args, "multiple_shots", False):
        cfg.multiple_shots = True
    if getattr(args, "log_data", False):
        cfg.log_data = True
    if getattr(args, "no_tracker_auto_scale", False):
        cfg.tracker_auto_scale = False
    if getattr(args, "tracker_max_span", None) is not None:
        cfg.tracker_max_span_px = max(1.0, float(args.tracker_max_span))

    if cfg.performance_increase:
        if args.res_scale is None:
            cfg.res_scale = min(cfg.res_scale, float(getattr(cfg, 'performance_res_scale', 0.5)))
        cfg.flags.gpu_viz = True
        cfg.flags.adaptive_draw = True
        cfg.flags.display_downsample = True
        if args.draw_every is None:
            cfg.draw_every = max(cfg.draw_every, int(getattr(cfg, 'performance_draw_every', 3)))
        cfg.display_downsample_factor = max(
            cfg.display_downsample_factor,
            int(getattr(cfg, 'performance_display_downsample', 2)),
        )
        if getattr(cfg, 'performance_smooth_passes', 0) is not None:
            cfg.smooth_passes = int(getattr(cfg, 'performance_smooth_passes', 0))
        cfg.performance_enable_window = True

    if args.vr is not None:
        if args.vr:
            cfg.enable_mouse_swing = False
            cfg.use_tracker = True
        else:
            cfg.enable_mouse_swing = True
            cfg.use_tracker = False
    elif args.mouse_swing:
        cfg.use_tracker = False

    return cfg


def apply_runtime_overrides(game, args):
    if args.mode:
        game._set_mode(args.mode)
    if args.course:
        game._cycle_course(args.course)
    if args.no_control_panel:
        game.cfg.show_control_panel = False
        game._deactivate_config_panel()
    elif args.config_panel:
        game.cfg.show_control_panel = True
        game._activate_config_panel()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Quantum Mini-Golf with quick configuration overrides.")

    # Core gameplay setup
    parser.add_argument('--map', choices=MAP_CHOICES, help='Initial obstacle map')
    parser.add_argument('--mode', choices=MODE_CHOICES, help='Initial display mode')
    parser.add_argument('--course', choices=COURSE_CHOICES, help='Start in a guided course')
    parser.add_argument('--wave-profile', choices=WAVE_CHOICES, help='Initial wave profile')
    parser.add_argument('--stop-mode', choices=STOP_CHOICES, help='Shot stop criterion')
    parser.add_argument('--wall-thickness', type=float, help='Single-wall thickness factor (0.05 - 5.0)')
    parser.add_argument('--movement-speed', type=float, help='Movement speed scale')
    parser.add_argument('--shot-time', type=float, help='Shot time limit (<=0 for infinity)')
    parser.add_argument('--sink-threshold', type=float, help='Sink probability threshold (0-1)')
    parser.add_argument('--boost-increment', type=float, help='Boost increment per measurement')
    parser.add_argument('--boost-factor', type=float, help='Base probability boost applied when enabled')
    parser.add_argument('--mouse-swing', action='store_true', help='Enable mouse swing control')
    parser.add_argument('--multiple-shots', action='store_true', help='Allow consecutive shots and track attempts')
    parser.add_argument('--log-data', action='store_true', help='Record VR debug telemetry to vr_debug_log.txt')
    parser.add_argument('--background-path', type=str, help='Load a custom course background image')
    parser.add_argument('--performance-increase', action='store_true', help='Enable experimental performance optimisations')
    parser.add_argument('--fast-mode-paraxial', action='store_true', help='Approximate paraxial fast mode (requires --performance-increase)')
    parser.add_argument('--no-tracker-auto-scale', action='store_true', help='Disable automatic tracker scaling correction')
    parser.add_argument('--tracker-max-span', type=float, help='Maximum LED span in pixels before rejecting tracker frames')
    parser.add_argument('--config-panel', action='store_true', help='Force the control panel window to open on start')
    parser.add_argument('--no-control-panel', action='store_true', help='Disable the separate control panel window')

    # Simulation tuning
    parser.add_argument('--quantum-measure', dest='quantum_measure', action='store_true', help='Enable automatic quantum measurements')
    parser.add_argument('--no-quantum-measure', dest='quantum_measure', action='store_false', help='Disable automatic quantum measurements')
    parser.add_argument('--measurement-gamma', type=float, help='Override the quantum measurement gamma value')
    parser.add_argument('--sink-rule', choices=['prob_threshold', 'measurement'], help='Select the sink resolution rule')
    parser.add_argument('--edge-boundary', choices=['reflect', 'absorb'], help='Select the edge boundary behaviour')
    parser.add_argument('--max-steps-per-shot', type=int, help='Maximum simulation steps per shot before forcing a reset')

    # Performance & visuals
    parser.add_argument('--perf-profile', choices=['quality', 'balanced', 'fast'], help='Preset performance flag bundle')
    parser.add_argument('--blit', dest='blitting', action='store_true', help='Force matplotlib blitting on')
    parser.add_argument('--no-blit', dest='blitting', action='store_false', help='Force matplotlib blitting off')
    parser.add_argument('--gpu-viz', dest='gpu_viz', action='store_true', help='Enable GPU visualisation pipeline when available')
    parser.add_argument('--no-gpu-viz', dest='gpu_viz', action='store_false', help='Disable GPU visualisation pipeline')
    parser.add_argument('--target-fps', type=float, help='Target rendering rate (frames per second)')
    parser.add_argument('--draw-every', type=int, help='Render every Nth simulation frame')
    parser.add_argument('--res-scale', type=float, help='Scale factor applied to the simulation resolution')

    # Tracker & VR
    parser.add_argument('--vr', dest='vr', action='store_true', help='Enable tracker-driven VR swing control')
    parser.add_argument('--no-vr', dest='vr', action='store_false', help='Disable tracker input and rely on mouse swings')
    parser.add_argument('--display-tracker', dest='display_tracker', action='store_true', help='Show the tracker debug window')
    parser.add_argument('--no-display-tracker', dest='display_tracker', action='store_false', help='Hide the tracker debug window')

    # Calibration helpers
    parser.add_argument('--calibrate-course', action='store_true', help='Run manual boundary calibration before launching')
    parser.add_argument('--calibrate-course-led', action='store_true', help='Run LED auto-calibration before launching')
    parser.add_argument('--calibration-path', type=str, help='Explicit course calibration file to load')
    parser.add_argument('--skip-calibration-preview', action='store_true', help='Skip the calibration snapshot preview')

    # Recording & automation
    parser.add_argument('--record-video', nargs='?', const='', metavar='PATH', help='Record the scripted demo instead of launching the UI (optional PATH overrides the output)')
    parser.add_argument('--record-output', metavar='PATH', help='Output path to use when recording the scripted demo')
    parser.add_argument('--headless', action='store_true', help='Use a non-interactive backend and skip launching the Qt window')

    # Probability boosting tweaks
    parser.add_argument('--boost-hole', dest='boost_hole', action='store_true', help='Enable hole probability boosting')
    parser.add_argument('--no-boost-hole', dest='boost_hole', action='store_false', help='Disable hole probability boosting')

    # Diagnostics & backend overrides
    parser.add_argument('--dump-config', action='store_true', help='Print the resolved configuration before launching')
    parser.add_argument('--backend', choices=['auto', 'cpu', 'gpu'], help='Force backend selection for FFT/physics compute')

    parser.set_defaults(
        vr=None,
        display_tracker=None,
        blitting=None,
        gpu_viz=None,
        quantum_measure=None,
        boost_hole=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)

    if args.backend:
        os.environ["QUANTUM_MINIGOLF_BACKEND"] = args.backend

    if args.dump_config:
        snapshot = json.dumps(asdict(cfg), indent=2, default=str)
        print(snapshot)

    if args.calibrate_course:
        exit_code = _run_calibration_helper("calibrate_course_boundaries.py")
        if exit_code != 0:
            sys.exit(exit_code)
    if args.calibrate_course_led:
        exit_code = _run_calibration_helper("calibrate_course_boundaries_LED.py")
        if exit_code != 0:
            sys.exit(exit_code)

    record_requested = args.record_video is not None
    if args.record_output and not record_requested:
        print("[warn] --record-output has no effect without --record-video.")
    if record_requested:
        from quantum_minigolf.RecordVideo import OUTPUT_PATH as DEFAULT_OUTPUT_PATH, record_demo

        if args.record_output:
            destination = Path(args.record_output)
        elif args.record_video and args.record_video.strip():
            destination = Path(args.record_video)
        else:
            destination = DEFAULT_OUTPUT_PATH
        destination = destination if isinstance(destination, Path) else Path(destination)
        destination = destination.expanduser()
        print(f"[info] Recording scripted demo to {destination.resolve()}")
        record_demo(destination)
        return

    headless_requested = bool(args.headless)
    _configure_matplotlib(headless_requested)

    calibration, calibration_path = _find_course_calibration(getattr(cfg, "tracker_calibration_path", None))
    if calibration is None or calibration_path is None:
        print("[warn] No course calibration file found. Tracker will use uncalibrated coordinates.")
    else:
        cfg.tracker_calibration_path = str(calibration_path)
        cfg.tracker_calibration_data = calibration
        print(f"[info] Course calibration loaded from {calibration_path}")
        if not (args.skip_calibration_preview or headless_requested):
            if _show_calibration_snapshot(calibration):
                print("[info] Displayed calibration snapshot; close the figure window if adjustments are needed.")
            else:
                print("[warn] Unable to display calibration snapshot.")
        else:
            print("[info] Calibration preview skipped.")
    from quantum_minigolf.game import QuantumMiniGolfGame

    game = QuantumMiniGolfGame(cfg)
    apply_runtime_overrides(game, args)
    if headless_requested:
        print("[info] Headless mode active; skipping interactive matplotlib window.")
        return
    plt.show()


if __name__ == "__main__":
    main()
