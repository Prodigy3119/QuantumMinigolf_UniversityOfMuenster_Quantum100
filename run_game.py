
import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

from quantum_minigolf import QuantumMiniGolfGame, GameConfig, PerformanceFlags
from quantum_minigolf.calibration import CalibrationData


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

MAP_CHOICES = [
    'double_slit', 'single_slit', 'single_wall', 'no_obstacle'
]
MODE_CHOICES = ['classical', 'quantum', 'mixed']
COURSE_CHOICES = ['quantum_demo', 'advanced_showcase']
STOP_CHOICES = ['time', 'friction']
WAVE_CHOICES = ['packet', 'front']


def build_config(args):
    flags = PerformanceFlags(
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
    if args.mouse_swing:
        cfg.enable_mouse_swing = True
    if args.no_control_panel:
        cfg.show_control_panel = False
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
    parser.add_argument('--mouse-swing', action='store_true', help='Enable mouse swing control')
    parser.add_argument('--config-panel', action='store_true', help='Force the control panel window to open on start')
    parser.add_argument('--no-control-panel', action='store_true', help='Disable the separate control panel window')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)
    calibration, calibration_path = _find_course_calibration(getattr(cfg, "tracker_calibration_path", None))
    if calibration is None or calibration_path is None:
        print("[warn] No course calibration file found. Tracker will use uncalibrated coordinates.")
    else:
        cfg.tracker_calibration_path = str(calibration_path)
        cfg.tracker_calibration_data = calibration
        print(f"[info] Course calibration loaded from {calibration_path}")
        if _show_calibration_snapshot(calibration):
            print("[info] Displayed calibration snapshot; close the figure window if adjustments are needed.")
        else:
            print("[warn] Unable to display calibration snapshot.")
    game = QuantumMiniGolfGame(cfg)
    apply_runtime_overrides(game, args)
    plt.show()


if __name__ == "__main__":
    main()
