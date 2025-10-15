from __future__ import annotations

import json
import pickle
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, List, Any

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


Point = Tuple[float, float]
PointArray = np.ndarray  # shape (N, 2)

_COURSE_PREVIEW = None


def ensure_course_preview(map_kind: str = "double_slit") -> None:
    """
    Launch a lightweight course preview window so calibration can be performed with
    the current board layout in view. Intended for manual calibration scripts.
    """
    global _COURSE_PREVIEW
    if _COURSE_PREVIEW is not None:
        return
    try:
        import matplotlib  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Course preview unavailable (matplotlib import failed: {exc})")
        return

    # Prefer the QtAgg backend so the preview matches the main game.
    backend = ""
    try:
        backend = str(matplotlib.get_backend()).lower()
    except Exception:
        backend = ""
    if backend != "qtagg":
        try:  # pragma: no branch - best effort
            matplotlib.use("QtAgg")  # type: ignore[attr-defined]
        except Exception:
            pass

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Course preview unavailable (pyplot import failed: {exc})")
        return

    try:
        # Deferred import avoids circular references while quantum_minigolf.game loads.
        from . import GameConfig, PerformanceFlags, QuantumMiniGolfGame
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Course preview unavailable (game import failed: {exc})")
        return

    try:
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
        cfg = GameConfig(flags=flags)
        cfg.map_kind = map_kind
        cfg.use_tracker = False
        cfg.show_control_panel = False
        cfg.enable_mouse_swing = False
        cfg.PlotBall = False
        cfg.PlotWavePackage = False
        cfg.quantum_measure = False
        cfg.shot_time_limit = None
        preview = QuantumMiniGolfGame(cfg)
        try:
            manager = preview.viz.fig.canvas.manager  # type: ignore[attr-defined]
        except Exception:
            manager = None
        if manager is not None:
            try:
                manager.set_window_title("Quantum Mini-Golf Course Preview")  # type: ignore[attr-defined]
            except Exception:
                pass
        preview.viz.fig.canvas.draw_idle()
        try:
            plt.show(block=False)
            plt.pause(0.05)
        except Exception:
            pass
        _COURSE_PREVIEW = preview
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Course preview failed to launch: {exc}")
        try:
            plt.close("all")
        except Exception:
            pass


def _ensure_cv2():
    if cv2 is None:  # pragma: no cover - only hit on misconfigured environments
        raise RuntimeError(
            "OpenCV (cv2) is required for calibration routines"
            + (f": {_CV2_IMPORT_ERROR}" if _CV2_IMPORT_ERROR else "")
        )


def order_points_clockwise(points: Sequence[Point]) -> np.ndarray:
    """
    Order a list of four points as top-left, top-right, bottom-right, bottom-left.
    """
    if len(points) != 4:
        raise ValueError(f"Expected exactly four points, got {len(points)}")
    pts = np.asarray(points, dtype=np.float32)
    # Sum / difference heuristic for rectangle corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def compute_homography(
    camera_points: Sequence[Point],
    board_size: Tuple[float, float],
    *,
    enforce_clockwise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a homography that maps camera pixel coordinates to board-space pixels.

    Parameters
    ----------
    camera_points:
        Iterable with four corner points observed by the camera.
    board_size:
        Physical/course size expressed in board pixel coordinates (width, height).
    enforce_clockwise:
        If True (default) the input points are re-ordered to TL, TR, BR, BL.

    Returns
    -------
    H:
        3x3 float32 homography matrix mapping [x, y, 1]^T in camera space to board space.
    ordered_points:
        The ordered version of the supplied camera points.
    """
    _ensure_cv2()
    if len(camera_points) != 4:
        raise ValueError("Need exactly four camera points to compute a homography.")

    ordered_cam = order_points_clockwise(camera_points) if enforce_clockwise else np.asarray(
        camera_points, dtype=np.float32
    )
    board_w, board_h = map(float, board_size)
    dest_board = np.array(
        [
            [0.0, 0.0],
            [board_w, 0.0],
            [board_w, board_h],
            [0.0, board_h],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(ordered_cam, dest_board)
    return H.astype(np.float32), ordered_cam


def perspective_error(
    H: np.ndarray,
    src_points: Sequence[Point],
    dst_points: Sequence[Point],
) -> float:
    """
    Compute RMS reprojection error for homography H between the provided point sets.
    """
    _ensure_cv2()
    src = np.asarray(src_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H)
    diff = projected - dst
    err = np.linalg.norm(diff.reshape(-1, 2), axis=1)
    return float(math.sqrt(np.mean(err ** 2)))


@dataclass(frozen=True)
class CalibrationData:
    """
    Holds the mapping between camera pixels and course coordinates.

    Board coordinates follow image-style orientation: origin at the top-left corner,
    x increases to the right, y increases downward. Consumers that prefer a different
    origin (e.g. bottom-left) should adjust when using the helpers.
    """

    frame_width: int
    frame_height: int
    board_width: float
    board_height: float
    camera_points: tuple[Point, Point, Point, Point]
    board_points: tuple[Point, Point, Point, Point]
    homography: tuple[float, ...]  # row-major 3x3 flattened
    rms_error: float
    timestamp_utc: float
    metadata: Mapping[str, str] | None = None

    def camera_to_board(self, point: Point) -> Point:
        """
        Map a single camera pixel coordinate into board-space.
        """
        H = np.asarray(self.homography, dtype=np.float32).reshape(3, 3)
        xy1 = np.array([point[0], point[1], 1.0], dtype=np.float32)
        mapped = H @ xy1
        if abs(mapped[2]) < 1e-6:
            raise ZeroDivisionError("Degenerate homography mapping; denominator is near zero.")
        mapped /= mapped[2]
        return float(mapped[0]), float(mapped[1])

    def board_to_camera(self, point: Point) -> Point:
        """
        Map a board pixel coordinate back into camera pixel space.
        """
        H = np.asarray(self.homography, dtype=np.float32).reshape(3, 3)
        H_inv = np.linalg.inv(H)
        xy1 = np.array([point[0], point[1], 1.0], dtype=np.float32)
        mapped = H_inv @ xy1
        if abs(mapped[2]) < 1e-6:
            raise ZeroDivisionError("Degenerate inverse homography mapping; denominator is near zero.")
        mapped /= mapped[2]
        return float(mapped[0]), float(mapped[1])

    def as_dict(self) -> MutableMapping[str, object]:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "board_width": self.board_width,
            "board_height": self.board_height,
            "camera_points": [list(pt) for pt in self.camera_points],
            "board_points": [list(pt) for pt in self.board_points],
            "homography": list(self.homography),
            "rms_error": float(self.rms_error),
            "timestamp_utc": float(self.timestamp_utc),
            "metadata": dict(self.metadata) if self.metadata else None,
        }

    def save(self, path: Path | str) -> None:
        """
        Persist the calibration to JSON. Legacy alias for ``save_json``.
        """
        self.save_json(path)

    def save_json(self, path: Path | str) -> None:
        """
        Persist the calibration to JSON.
        """
        payload = self.as_dict()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def save_pickle(self, path: Path | str) -> None:
        """
        Persist the calibration to a pickle file. Stores the underlying dictionary so the format
        remains stable even if the class definition changes.
        """
        payload = self.as_dict()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("wb") as fh:
            pickle.dump(payload, fh)

    def outline_in_camera(self) -> np.ndarray:
        """
        Return the four board corners expressed in camera coordinates.
        """
        cam_pts = np.asarray(self.camera_points, dtype=np.float32)
        return cam_pts.copy()

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationData":
        """
        Load calibration data from JSON or pickle (based on file suffix).
        """
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        if suffix in (".pkl", ".pickle"):
            return cls.load_pickle(path_obj)
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def load_pickle(cls, path: Path | str) -> "CalibrationData":
        """
        Load calibration data from a pickle file.
        """
        path_obj = Path(path)
        with path_obj.open("rb") as fh:
            payload: Any = pickle.load(fh)
        if isinstance(payload, CalibrationData):
            return payload
        if isinstance(payload, Mapping):
            return cls.from_dict(payload)
        raise TypeError(f"Unsupported pickle payload type: {type(payload)!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CalibrationData":
        camera_points = tuple(tuple(map(float, pt)) for pt in data["camera_points"])  # type: ignore[index]
        board_points = tuple(tuple(map(float, pt)) for pt in data["board_points"])  # type: ignore[index]
        homography = tuple(float(x) for x in data["homography"])  # type: ignore[index]
        return cls(
            frame_width=int(data["frame_width"]),  # type: ignore[index]
            frame_height=int(data["frame_height"]),  # type: ignore[index]
            board_width=float(data["board_width"]),  # type: ignore[index]
            board_height=float(data["board_height"]),  # type: ignore[index]
            camera_points=camera_points,  # type: ignore[arg-type]
            board_points=board_points,  # type: ignore[arg-type]
            homography=homography,
            rms_error=float(data.get("rms_error", 0.0)),
            timestamp_utc=float(data.get("timestamp_utc", time.time())),
            metadata=data.get("metadata"),
        )

    @classmethod
    def build(
        cls,
        camera_points: Sequence[Point],
        frame_size: Tuple[int, int],
        board_size: Tuple[float, float],
        metadata: Optional[Mapping[str, str]] = None,
    ) -> "CalibrationData":
        """
        Construct a calibration entry from observed camera points.
        """
        H, ordered_cam = compute_homography(camera_points, board_size)
        board_w, board_h = map(float, board_size)
        board_pts = np.array(
            [
                [0.0, 0.0],
                [board_w, 0.0],
                [board_w, board_h],
                [0.0, board_h],
            ],
            dtype=np.float32,
        )
        rms = perspective_error(H, ordered_cam, board_pts)
        homography_tuple = tuple(float(v) for v in H.reshape(-1))
        return cls(
            frame_width=int(frame_size[0]),
            frame_height=int(frame_size[1]),
            board_width=board_w,
            board_height=board_h,
            camera_points=tuple((float(p[0]), float(p[1])) for p in ordered_cam),  # type: ignore[arg-type]
            board_points=tuple((float(p[0]), float(p[1])) for p in board_pts),  # type: ignore[arg-type]
            homography=homography_tuple,
            rms_error=float(rms),
            timestamp_utc=time.time(),
            metadata=dict(metadata) if metadata else None,
        )


def apply_homography_points(H: np.ndarray, points: Iterable[Point]) -> np.ndarray:
    """
    Apply a homography to a list of points. Returns an (N, 2) float32 array.
    """
    _ensure_cv2()
    pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H.astype(np.float32))
    return mapped.reshape(-1, 2)
