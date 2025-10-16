from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[error] OpenCV is required for calibration: {exc}", file=sys.stderr)
    raise

try:
    from quantum_minigolf.calibration import (
        CalibrationData,
        compute_homography,
        ensure_course_preview,
        order_points_clockwise,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback when run as a script
    package_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_root))
    from quantum_minigolf.calibration import (
        CalibrationData,
        compute_homography,
        ensure_course_preview,
        order_points_clockwise,
    )


Point = Tuple[float, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect four calibration LEDs and compute the course homography."
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
        "--frames",
        type=int,
        default=40,
        help="Number of frames to average when sampling the LEDs (default: 40).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration/course_calibration.json"),
        help="Where to store the resulting calibration JSON.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Optional note to embed into the calibration metadata.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip the preview window and capture immediately.",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Skip automatic LED detection and collect the four corners manually.",
    )
    return parser.parse_args()


def _acquire_stack(cap: cv2.VideoCapture, frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Capture ``frames`` frames and return the most recent color frame and the median grayscale image.
    """
    gray_stack: List[np.ndarray] = []
    color_frame: Optional[np.ndarray] = None
    for _ in range(max(1, frames)):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera during capture.")
        color_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_stack.append(gray.astype(np.float32))
        cv2.waitKey(1)  # let UI update
        time.sleep(0.01)
    if color_frame is None:
        raise RuntimeError("No frames captured from camera.")
    median_gray = np.median(np.stack(gray_stack, axis=0), axis=0).astype(np.uint8)
    return color_frame, median_gray


def _connected_component_centroids(
    mask: np.ndarray,
    reference: np.ndarray,
    *,
    min_area: float,
    max_area: float,
    top_k: int,
) -> List[Point]:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return []
    candidates: List[Tuple[float, Point]] = []
    for label_idx in range(1, num):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[label_idx]
        x0 = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        patch = reference[y0 : y0 + h, x0 : x0 + w]
        score = float(area * (patch.mean() + patch.max()))
        candidates.append((score, (float(cx), float(cy))))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [pt for _, pt in candidates[:top_k]]


def _detect_led_points(gray: np.ndarray, *, top_k: int = 4) -> List[Point]:
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
    h, w = blurred.shape[:2]
    frame_area = float(h * w)
    min_area = 1.0
    max_area = max(frame_area * 0.05, 16.0)

    percentiles = (99.9, 99.7, 99.4, 99.0, 98.5, 98.0, 97.0, 96.0)
    kernel = np.ones((3, 3), np.uint8)
    best: List[Point] = []
    for p in percentiles:
        thresh_val = float(np.percentile(blurred, p))
        if thresh_val <= 2.0:
            continue
        cutoff = max(2.0, thresh_val * 0.9)
        _, mask = cv2.threshold(blurred, cutoff, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, kernel, iterations=2)
        pts = _connected_component_centroids(
            mask,
            blurred,
            min_area=min_area,
            max_area=max_area,
            top_k=top_k,
        )
        if len(pts) >= top_k:
            return pts[:top_k]
        if len(pts) > len(best):
            best = pts

    if len(best) < top_k:
        flat = blurred.reshape(-1)
        idx = np.argpartition(flat, -top_k)[-top_k:]
        coords = [(float(i % w), float(i // w)) for i in idx]
        coords.sort(key=lambda pt: -blurred[int(pt[1]), int(pt[0])])
        seen = {(round(px, 1), round(py, 1)) for px, py in best}
        for pt in coords:
            key = (round(pt[0], 1), round(pt[1], 1))
            if key in seen:
                continue
            best.append(pt)
            seen.add(key)
            if len(best) >= top_k:
                break
    return best[:top_k]


def _collect_manual_points(image: np.ndarray) -> List[Point]:
    window = "Manual Corner Selection"
    display = image.copy()
    points: List[Point] = []

    instructions = textwrap.dedent(
        """
        Click the four course corners in order: top-left, top-right, bottom-right, bottom-left.
        Press BACKSPACE to undo the last point, ENTER to accept, or ESC to cancel.
        """
    ).strip()
    print(instructions)

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = display.copy()
        for idx, (px, py) in enumerate(points):
            cv2.circle(canvas, (int(px), int(py)), 6, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                str(idx + 1),
                (int(px) + 8, int(py) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(window, canvas)
        key = cv2.waitKey(25) & 0xFF
        if key in (27, ord("q")):
            points.clear()
            break
        if key in (8, 127) and points:
            points.pop()
        if key in (13, 10) and len(points) == 4:
            break
    cv2.destroyWindow(window)
    return points


def _draw_points(image: np.ndarray, points: Sequence[Point]) -> np.ndarray:
    overlay = image.copy()
    ordered = order_points_clockwise(points)
    colors = [(0, 255, 0), (0, 200, 255), (0, 140, 255), (255, 64, 64)]
    labels = ("TL", "TR", "BR", "BL")
    for idx, (pt, color, label) in enumerate(zip(ordered, colors, labels)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(overlay, (x, y), 8, color, 2)
        cv2.putText(
            overlay,
            f"{label}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.polylines(
        overlay,
        [ordered.astype(np.int32)],
        isClosed=True,
        color=(255, 255, 255),
        thickness=2,
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

    try:
        if args.no_preview:
            color_frame, gray = _acquire_stack(cap, args.frames)
        else:
            print(
                "Press SPACE to capture the four LEDs once they are stable, "
                "M for manual selection, or Q to quit."
            )
            color_frame = None
            gray = None
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[error] Failed to read frame from camera.", file=sys.stderr)
                    return 2
                preview = frame.copy()
                cv2.putText(
                    preview,
                    "SPACE: capture   M: manual  Q: quit",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Calibration Preview", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Cancelled.")
                    return 0
                if key in (ord("m"), ord("M")):
                    color_frame = frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    points = _collect_manual_points(color_frame)
                    if len(points) != 4:
                        print("[warn] Manual selection aborted.")
                        continue
                    break
                if key == 32:  # space -> capture
                    color_frame, gray = _acquire_stack(cap, args.frames)
                    points = _detect_led_points(gray, top_k=4)
                    if len(points) < 4 and not args.manual_only:
                        print(
                            f"[warn] Only detected {len(points)} bright spots. "
                            "Press 'm' to pick corners manually."
                        )
                        continue
                    break
            cv2.destroyWindow("Calibration Preview")

        if color_frame is None or gray is None:
            raise RuntimeError("No frame captured for calibration.")

        if args.manual_only:
            points = _collect_manual_points(color_frame)
            if len(points) != 4:
                print("[error] Manual selection cancelled.")
                return 3
        elif "points" not in locals() or len(points) != 4:
            points = _detect_led_points(gray, top_k=4)
            if len(points) != 4:
                print(
                    f"[error] Only detected {len(points)} bright spots. "
                    "Rerun with --manual-only to click the corners."
                )
                return 4

        ordered = order_points_clockwise(points)
        overlay = _draw_points(color_frame, ordered)
        cv2.imshow("Detected Corners", overlay)
        print("Detected (TL, TR, BR, BL):")
        for label, pt in zip(("TL", "TR", "BR", "BL"), ordered):
            print(f"  {label}: ({pt[0]:7.2f}, {pt[1]:7.2f})")

        H, _ = compute_homography(ordered, (args.board_width, args.board_height))
        warped = cv2.warpPerspective(
            color_frame,
            H,
            (int(round(args.board_width)), int(round(args.board_height))),
        )
        cv2.imshow("Warped Board Preview", warped)
        print(
            f"Press ENTER to accept calibration, BACKSPACE to retry detection, "
            f"or ESC to abort."
        )

        decision = None
        while decision is None:
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10):
                decision = "accept"
            elif key in (8, 127):
                decision = "retry"
            elif key in (27, ord("q")):
                decision = "abort"
        cv2.destroyWindow("Detected Corners")
        cv2.destroyWindow("Warped Board Preview")

        if decision == "retry":
            print("Retry requested. Please run the script again.")
            return 5
        if decision == "abort":
            print("Cancelled.")
            return 0

        frame_h, frame_w = gray.shape[:2]
        metadata = {}
        if args.note:
            metadata["note"] = args.note
        metadata["script"] = "calibrate_course_boundaries.py"

        calibration = CalibrationData.build(
            ordered,
            frame_size=(frame_w, frame_h),
            board_size=(args.board_width, args.board_height),
            metadata=metadata,
        )
        calibration.save(args.output)
        print(f"Calibration saved to {args.output.resolve()}")
        print(f"RMS reprojection error: {calibration.rms_error:.4f} pixels")
        print("You can now remove the LED markers.")
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
