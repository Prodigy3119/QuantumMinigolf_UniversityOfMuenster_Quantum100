"""Record a mixed-mode double-slit demonstration to DoubleSlitDemo.mp4.

Running this module will load the Quantum Mini Golf game, configure it to
the double-slit course in mixed (quantum + classical) mode, fire a straight
shot towards the goal, and capture the entire sequence as an MP4 file. The
resulting video is saved alongside the script as ``DoubleSlitDemo.mp4``.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib

# Use a non-interactive backend so the script can run headlessly.
matplotlib.use("Agg")  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from quantum_minigolf import GameConfig, PerformanceFlags, QuantumMiniGolfGame  # noqa: E402


OUTPUT_PATH = Path("QuantumMinigolfDemo.mp4")


def _build_config() -> GameConfig:
    flags = PerformanceFlags(
        blitting=False,
        display_downsample=False,
        gpu_viz=False,
        low_dpi=False,
        inplace_step=True,
        adaptive_draw=False,
        path_decimation=False,
        event_debounce=False,
        fast_blur=False,
        pixel_upscale=False,
    )

    cfg = GameConfig(
        flags=flags,
        map_kind="double_slit",
        movement_speed_scale=3.0,
        draw_every=2,
        steps_per_shot=420,
        shot_time_limit=100.0,
        wave_initial_profile="front",  # packet?
        show_control_panel=False,
    )

    cfg.enable_mouse_swing = False
    cfg.use_tracker = False
    cfg.quantum_measure = True
    cfg.shot_stop_mode = "time"
    cfg.edge_boundary = "reflect"
    cfg.boost_hole_probability = False
    cfg.display_downsample_factor = 1
    return cfg


def _compute_straight_kvec(game: QuantumMiniGolfGame) -> np.ndarray:
    start = game.ball_pos.astype(np.float32)
    goal = np.asarray(game.hole_center, dtype=np.float32)
    direction = goal - start
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    direction /= norm

    swing_speed = float(game.cfg.impact_min_speed + 40.0)
    proxy_velocity = direction * swing_speed
    kvec = game._compute_kvec_from_swing(proxy_velocity, swing_speed)  # type: ignore[attr-defined]
    if kvec is None:
        kmax = float(game.be.to_cpu(game.k_max))
        frac = 0.5 * (game.cfg.kmin_frac + game.cfg.kmax_frac)
        kvec = direction * (kmax * frac)
    return np.asarray(kvec, dtype=np.float32)


def record_demo(output_path: Path = OUTPUT_PATH) -> Path:
    cfg = _build_config()
    game = QuantumMiniGolfGame(cfg)
    game._set_mode("mixed")  # mixed = classical + quantum views
    game._reset()

    # Avoid GUI event-loop pauses while using the Agg backend.
    plt.pause = lambda *args, **kwargs: None  # type: ignore[assignment]
    plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]

    kvec = _compute_straight_kvec(game)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "title": "Quantum Mini Golf Double-Slit Demo",
        "artist": "RecordVideo.py",
        "comment": "Mixed-state shot through the double slit map",
    }
    writer = FFMpegWriter(fps=30, metadata=metadata)
    if not writer.isAvailable():  # pragma: no cover - depends on local ffmpeg installation
        raise RuntimeError("ffmpeg executable not found. Please install ffmpeg to enable video export.")

    fig = game.viz.fig

    with writer.saving(fig, str(output_path), dpi=120):
        def capture_frame(_game: QuantumMiniGolfGame) -> None:
            fig.canvas.draw()
            writer.grab_frame()

        game.add_frame_listener(capture_frame)
        try:
            capture_frame(game)
            game._shoot(kvec)
            capture_frame(game)
        finally:
            game.remove_frame_listener(capture_frame)

    plt.close(fig)
    print(f"Recorded double-slit demo saved to {output_path}")
    return output_path


if __name__ == "__main__":
    record_demo()
