from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[error] OpenCV is required for calibration: {exc}", file=sys.stderr)
    raise

from quantum_minigolf.calibration import CalibrationData, ensure_course_preview, order_points_clockwise

Point = Tuple[float, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the course boundaries manually by clicking the four corners."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument(
        "--frame-width",
        type=int,
        default=None,
        help="Optional frame width to request from the camera.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=None,
        help="Optional frame height to request from the camera.",
    )
    parser.add_argument(
        "--board-width",
        type=float,
        default=288.0,
        help="Board width in simulation pixels (default: 288).",
    )
    parser.add_argument(
        "--board-height",
        type=float,
        default=144.0,
        help="Board height in simulation pixels (default: 144).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("calibration/course_calibration.json"),
        help="Where to store the resulting calibration JSON (default: calibration/course_calibration.json).",
    )
    parser.add_argument(
        "--output-pickle",
        type=Path,
        default=Path("calibration/course_calibration.pkl"),
        help="Where to store the calibration pickle (default: calibration/course_calibration.pkl).",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Optional note to include in the calibration metadata.",
    )
    return parser.parse_args()


def _load_previous_calibration(paths: Iterable[Path]) -> Optional[CalibrationData]:
    for candidate in paths:
        try:
            if candidate and candidate.exists():
                return CalibrationData.load(candidate)
        except Exception as exc:
            print(f"[warn] Could not load calibration from {candidate}: {exc}")
    return None


def _nearest_index(points: Sequence[Point], x: float, y: float) -> int:
    dists = [(float((px - x) ** 2 + (py - y) ** 2), idx) for idx, (px, py) in enumerate(points)]
    dists.sort(key=lambda item: item[0])
    return dists[0][1]


def _collect_points(
    image: np.ndarray,
    previous_points: Optional[Sequence[Point]] = None,
) -> Optional[List[Point]]:
    window = "Manual Corner Selection"
    points: List[Point] = []
    ghost_points = list(previous_points) if previous_points else []

    print(
        "Click the four corners in order (TL, TR, BR, BL). "
        "Left-click adds or adjusts a point, right-click removes the closest point.\n"
        "Hotkeys: [ENTER] accept, [BACKSPACE] undo last, [R] reset, "
        "[P] prefill previous, [ESC] cancel."
    )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(image.shape[1], 1280), min(image.shape[0], 720))

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((float(x), float(y)))
            else:
                idx = _nearest_index(points, float(x), float(y))
                points[idx] = (float(x), float(y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            idx = _nearest_index(points, float(x), float(y))
            points.pop(idx)

    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            canvas = image.copy()
            if ghost_points:
                for gx, gy in ghost_points:
                    cv2.circle(canvas, (int(round(gx)), int(round(gy))), 8, (160, 160, 160), 1)
            for idx, (px, py) in enumerate(points):
                cv2.circle(canvas, (int(round(px)), int(round(py))), 10, (0, 255, 0), -1)
                cv2.putText(
                    canvas,
                    str(idx + 1),
                    (int(round(px)) + 12, int(round(py)) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                canvas,
                "ENTER=accept  P=prefill last  R=reset  ESC=cancel",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):  # enter
                if len(points) == 4:
                    return points
                print("Need exactly four points; keep clicking.")
            elif key in (8, 127):  # backspace/delete
                if points:
                    points.pop()
            elif key in (ord("r"), ord("R")):
                points.clear()
            elif key in (ord("p"), ord("P")) and ghost_points:
                points[:] = list(ghost_points)
            elif key in (27, ord("q")):
                return None
    finally:
        cv2.destroyWindow(window)


def _capture_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    print("Press SPACE to capture a calibration frame, or Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[error] Failed to read frame from camera.")
            return None
        preview = frame.copy()
        cv2.putText(
            preview,
            "SPACE=capture  Q=quit",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Calibration Preview", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # space
            cv2.destroyWindow("Calibration Preview")
            return frame
        if key in (ord("q"), 27):
            cv2.destroyWindow("Calibration Preview")
            return None


def _draw_ordered_overlay(image: np.ndarray, points: Sequence[Point]) -> np.ndarray:
    overlay = image.copy()
    ordered = order_points_clockwise(points)
    labels = ("TL", "TR", "BR", "BL")
    colors = [(0, 255, 0), (0, 200, 255), (0, 140, 255), (255, 64, 64)]
    for pt, label, color in zip(ordered, labels, colors):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(overlay, (x, y), 10, color, 2)
        cv2.putText(
            overlay,
            label,
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.polylines(
        overlay,
        [np.asarray(ordered, dtype=np.int32)],
        True,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return overlay


def main() -> int:
    args = _parse_args()

    ensure_course_preview()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[error] Unable to open camera index {args.camera}", file=sys.stderr)
        return 1

    if args.frame_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.frame_width))
    if args.frame_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.frame_height))

    previous = _load_previous_calibration(
        [
            path
            for path in (
                args.output_pickle if args.output_pickle else None,
                args.output_json if args.output_json else None,
            )
            if path is not None
        ]
    )

    try:
        while True:
            frame = _capture_frame(cap)
            if frame is None:
                print("Calibration cancelled.")
                return 0

            points = _collect_points(frame, previous_points=previous.camera_points if previous else None)
            if points is None:
                print("Point selection cancelled; capture again.")
                continue

            ordered = order_points_clockwise(points)
            overlay = _draw_ordered_overlay(frame, ordered)
            cv2.imshow("Corner Check", overlay)
            frame_h, frame_w = frame.shape[:2]
            preview_calibration = CalibrationData.build(
                ordered,
                frame_size=(frame_w, frame_h),
                board_size=(args.board_width, args.board_height),
            )
            warped = cv2.warpPerspective(
                frame,
                np.asarray(preview_calibration.homography, dtype=np.float32).reshape(3, 3),
                (int(round(args.board_width)), int(round(args.board_height))),
            )
            cv2.imshow("Warped Preview", warped)
            print("Press ENTER to accept, BACKSPACE to re-select points, or ESC to capture a new frame.")
            decision = None
            while decision is None:
                key = cv2.waitKey(0) & 0xFF
                if key in (13, 10):
                    decision = "accept"
                elif key in (8, 127):
                    decision = "redo_points"
                elif key in (27, ord("q")):
                    decision = "retry_frame"
            cv2.destroyWindow("Corner Check")
            cv2.destroyWindow("Warped Preview")

            if decision == "redo_points":
                print("Re-select the corners on this frame.")
                previous = preview_calibration
                continue
            if decision == "retry_frame":
                print("Capture a new frame.")
                continue

            metadata = {"script": "calibrate_course_boundaries_manual"}
            if args.note:
                metadata["note"] = args.note

            calibration = CalibrationData.build(
                ordered,
                frame_size=(frame_w, frame_h),
                board_size=(args.board_width, args.board_height),
                metadata=metadata,
            )
            if args.output_json:
                calibration.save_json(args.output_json)
                print(f"Calibration JSON written to {args.output_json.resolve()}")
            if args.output_pickle:
                calibration.save_pickle(args.output_pickle)
                print(f"Calibration pickle written to {args.output_pickle.resolve()}")
            print(f"RMS reprojection error: {calibration.rms_error:.4f} pixels")
            print("Calibration complete.")
            return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
