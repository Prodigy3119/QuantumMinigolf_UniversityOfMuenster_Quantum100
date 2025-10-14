from __future__ import annotations
import threading
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Deque
from collections import deque

import numpy as np

from .calibration import CalibrationData

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    cv2 = None
    _cv2_import_error = exc
else:
    _cv2_import_error = None

try:
    from .tracker_led import cfg_tracker as _tracker_led_cfg  # type: ignore
except Exception:
    _tracker_led_cfg = None


def _tracker_windows_enabled(default: bool) -> bool:
    if not default:
        return False
    cfg = _tracker_led_cfg
    if cfg is None:
        return default
    try:
        show_led = bool(getattr(cfg, 'SHOW_LED_WINDOW', True))
        show_overlay = bool(getattr(cfg, 'SHOW_OVERLAY_WINDOW', True))
        enable_overlay = bool(getattr(cfg, 'ENABLE_OVERLAY_WINDOW', True))
    except Exception:
        return default
    return show_led or (show_overlay and enable_overlay)

@dataclass
class TrackerConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    crop_x1: Optional[int] = None
    crop_x2: Optional[int] = None
    crop_y1: Optional[int] = None
    crop_y2: Optional[int] = None
    target_fps: int = 60
    threshold: int = 200
    min_area: int = 5
    assoc_max_px: float = 120.0
    putter_length_px: float = 380.0
    putter_thickness_px: float = 90.0
    vel_ema_alpha: float = 0.25
    dir_ema_alpha: float = 0.30
    impact_radius_px: float = 20.0
    min_speed_impact: float = 100.0
    impact_cooldown_sec: float = 0.50
    show_debug_window: bool = True
    debug_window_name: str = "Tracker Debug"
    sweep_margin_factor: float = 0.5
    calibration: Optional[CalibrationData] = None


@dataclass
class TrackerHit:
    timestamp: float
    center_px: Tuple[float, float]
    direction_px: Tuple[float, float]
    speed_px_s: float
    contact_px: Tuple[float, float]


@dataclass
class TrackerState:
    timestamp: float
    center_px: Optional[Tuple[float, float]] = None
    direction_px: Optional[Tuple[float, float]] = None
    span_px: float = 0.0
    visible: bool = False


def _check_cv2_available():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for tracker integration" +
            (f": {_cv2_import_error}" if _cv2_import_error else "")
        )


def safe_find_contours(binary_img):
    cnts = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts[0] if len(cnts) == 2 else cnts[1]


def find_top_two_blobs(contours, min_area=5):
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < max(1, min_area):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        blobs.append((float(x), float(y), float(r), float(area)))
    blobs.sort(key=lambda b: b[3], reverse=True)
    return blobs[:2]


def associate_leds(blobs, prev_led, assoc_max_px=120.0):
    led = [None, None]
    radii = [0.0, 0.0]
    if len(blobs) == 1:
        bx, by, br, _ = blobs[0]
        if prev_led[0] is None and prev_led[1] is None:
            slot = 0
        else:
            d0 = np.hypot(bx - prev_led[0][0], by - prev_led[0][1]) if prev_led[0] is not None else 1e9
            d1 = np.hypot(bx - prev_led[1][0], by - prev_led[1][1]) if prev_led[1] is not None else 1e9
            slot = 0 if d0 <= d1 else 1
        if prev_led[slot] is None or np.hypot(bx - prev_led[slot][0], by - prev_led[slot][1]) <= assoc_max_px:
            led[slot] = np.array([bx, by], dtype=np.float32)
            radii[slot] = br
        else:
            led = [np.array([bx, by], dtype=np.float32), None]
            radii = [br, 0.0]
    elif len(blobs) == 2:
        (x1, y1, r1, _), (x2, y2, r2, _) = blobs
        swap = False
        if prev_led[0] is not None and prev_led[1] is not None:
            cost_a = np.hypot(x1-prev_led[0][0], y1-prev_led[0][1]) + np.hypot(x2-prev_led[1][0], y2-prev_led[1][1])
            cost_b = np.hypot(x2-prev_led[0][0], y2-prev_led[0][1]) + np.hypot(x1-prev_led[1][0], y1-prev_led[1][1])
            swap = cost_b < cost_a
        if not swap:
            led = [np.array([x1, y1], dtype=np.float32), np.array([x2, y2], dtype=np.float32)]
            radii = [r1, r2]
        else:
            led = [np.array([x2, y2], dtype=np.float32), np.array([x1, y1], dtype=np.float32)]
            radii = [r2, r1]
    return led, radii


def compute_center(led):
    both = (led[0] is not None and led[1] is not None)
    if both:
        return (led[0] + led[1]) * 0.5
    elif led[0] is not None:
        return led[0].copy()
    elif led[1] is not None:
        return led[1].copy()
    return None


def update_velocity_ema(center, prev_led, vel_center, dt, alpha):
    if center is not None and dt and dt > 0:
        prev_center = (
            (prev_led[0] + prev_led[1]) * 0.5 if (prev_led[0] is not None and prev_led[1] is not None)
            else prev_led[0] if prev_led[0] is not None
            else prev_led[1] if prev_led[1] is not None
            else center
        )
        inst_v = (center - prev_center) / dt
        vel_center = (1.0 - alpha) * vel_center + alpha * inst_v
    else:
        vel_center *= 0.95
    return vel_center


def compute_face_direction(led, dir_ema, prev_face_angle, dt, alpha):
    face_angle_deg = float('nan')
    ang_vel_deg_s = float('nan')
    led_span = float('nan')
    if led[0] is not None and led[1] is not None:
        vec = led[1] - led[0]
        led_span = float(np.linalg.norm(vec))
        if led_span > 1e-6:
            u = vec / led_span
            if dir_ema is None:
                dir_ema = u
            else:
                if np.dot(dir_ema, u) < 0:
                    u = -u
                dir_ema = (1.0 - alpha) * dir_ema + alpha * u
                dir_ema /= max(np.linalg.norm(dir_ema), 1e-9)
            face_angle_rad = math.atan2(dir_ema[1], dir_ema[0])
            face_angle_deg = math.degrees(face_angle_rad)
            if prev_face_angle is not None and dt and dt > 0:
                d = math.atan2(math.sin(face_angle_rad - prev_face_angle), math.cos(face_angle_rad - prev_face_angle))
                ang_vel_deg_s = math.degrees(d / dt)
            prev_face_angle = face_angle_rad
    return dir_ema, prev_face_angle, face_angle_deg, ang_vel_deg_s, led_span


def circle_vs_rotated_rect(rect_center_xy, u_dir, length_px, thick_px, circle_xy, radius_px):
    rc = np.array(rect_center_xy, dtype=np.float32)
    q = np.array(circle_xy, dtype=np.float32)
    u = np.array(u_dir, dtype=np.float32)
    norm = np.linalg.norm(u)
    if norm < 1e-9:
        return False, rc, 0.0
    u /= norm
    v = np.array([-u[1], u[0]], dtype=np.float32)
    L = 0.5 * float(length_px)
    T = 0.5 * float(thick_px)
    r = q - rc
    lx = float(np.dot(r, u))
    ly = float(np.dot(r, v))
    cx = min(max(lx, -L), L)
    cy = min(max(ly, -T), T)
    dx = lx - cx
    dy = ly - cy
    dist_sq = dx * dx + dy * dy
    hit = dist_sq <= (radius_px * radius_px)
    closest_world = rc + u * cx + v * cy
    return hit, closest_world, dist_sq


@dataclass
class TrackerResult:
    state: TrackerState
    hits: Deque[TrackerHit] = field(default_factory=deque)


class TrackerManager:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self._show_debug_window = _tracker_windows_enabled(bool(self.cfg.show_debug_window))
        _check_cv2_available()
        self._state = TrackerState(timestamp=0.0, visible=False)
        self._state_lock = threading.Lock()
        self._hit_queue: Deque[TrackerHit] = deque()
        self._hit_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._calibration = self.cfg.calibration
        if self._calibration is not None:
            self.cfg.frame_width = self._calibration.frame_width
            self.cfg.frame_height = self._calibration.frame_height
            center_board = (
                self._calibration.board_width * 0.5,
                self._calibration.board_height * 0.5,
            )
            ref_cam = self._calibration.board_to_camera(center_board)
            ref_x = float(np.clip(ref_cam[0], 0.0, max(1.0, self.cfg.frame_width - 1)))
            ref_y = float(np.clip(ref_cam[1], 0.0, max(1.0, self.cfg.frame_height - 1)))
            ref_xy = np.array([ref_x, ref_y], dtype=np.float32)
        else:
            ref_xy = np.array([self.cfg.frame_width / 2, self.cfg.frame_height / 2], dtype=np.float32)
        self._reference_px = ref_xy
        self._last_reference_update = 0.0
        self._crop_rect = self._normalize_crop_rect()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="TrackerThread", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if cv2 is not None and self._show_debug_window:
            try:
                cv2.destroyWindow(self.cfg.debug_window_name)
            except Exception:
                pass

    def update_reference_point(self, game_xy: Tuple[float, float], game_extent: Tuple[int, int]):
        gx, gy = game_xy
        nx, ny = game_extent
        if self._calibration is not None:
            scale_x = self._calibration.board_width / max(nx, 1e-6)
            scale_y = self._calibration.board_height / max(ny, 1e-6)
            board_x = float(np.clip(gx * scale_x, 0.0, self._calibration.board_width))
            board_y = float(np.clip((ny - gy) * scale_y, 0.0, self._calibration.board_height))
            cam_x, cam_y = self._calibration.board_to_camera((board_x, board_y))
            x_px = float(np.clip(cam_x, 0.0, max(1.0, self.cfg.frame_width - 1)))
            y_px = float(np.clip(cam_y, 0.0, max(1.0, self.cfg.frame_height - 1)))
        else:
            x_px = float(np.clip(gx / max(nx, 1e-6) * self.cfg.frame_width, 0, self.cfg.frame_width))
            y_px = float(np.clip((1.0 - gy / max(ny, 1e-6)) * self.cfg.frame_height, 0, self.cfg.frame_height))
        self._reference_px = np.array([x_px, y_px], dtype=np.float32)
        self._last_reference_update = time.perf_counter()

    def get_state(self) -> TrackerState:
        with self._state_lock:
            return TrackerState(**vars(self._state))

    def pop_hits(self) -> list[TrackerHit]:
        with self._hit_lock:
            hits = list(self._hit_queue)
            self._hit_queue.clear()
        return hits

    # Internal -----------------------------------------------------------------

    def _register_state(self, state: TrackerState):
        with self._state_lock:
            self._state = state

    def _register_hit(self, hit: TrackerHit):
        with self._hit_lock:
            self._hit_queue.append(hit)

    def _normalize_crop_rect(self) -> Optional[tuple[int, int, int, int]]:
        cfg = self.cfg
        vals = (cfg.crop_x1, cfg.crop_x2, cfg.crop_y1, cfg.crop_y2)
        if not all(v is not None for v in vals):
            return None
        try:
            x1, x2, y1, y2 = [int(round(float(v))) for v in vals]
        except Exception:
            return None
        width = max(1, int(cfg.frame_width))
        height = max(1, int(cfg.frame_height))
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, x2, y1, y2)

    def _run(self):
        cfg = self.cfg
        cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        cap.set(cv2.CAP_PROP_FPS, cfg.target_fps)
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        except Exception:
            pass
        if not cap.isOpened():
            raise RuntimeError("Tracker could not open camera")

        if self._show_debug_window:
            cv2.namedWindow(cfg.debug_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cfg.debug_window_name, 900, 600)

        prev_led = [None, None]
        dir_ema = None
        prev_face_angle = None
        vel_center = np.zeros(2, dtype=np.float32)
        prev_time = None
        overlap_prev = False
        last_contact = None
        last_impact_time = -1e9

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame_h, frame_w = frame.shape[:2]
                roi = frame
                proc_rect = None
                offset_xy = (0.0, 0.0)
                crop_rect = self._crop_rect
                if crop_rect is not None:
                    x1, x2, y1, y2 = crop_rect
                    x1 = max(0, min(int(x1), frame_w - 1))
                    x2 = max(x1 + 1, min(int(x2), frame_w))
                    y1 = max(0, min(int(y1), frame_h - 1))
                    y2 = max(y1 + 1, min(int(y2), frame_h))
                    roi = frame[y1:y2, x1:x2]
                    offset_xy = (float(x1), float(y1))
                    proc_rect = (x1, x2, y1, y2)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, cfg.threshold, 255, cv2.THRESH_BINARY)
                blur_kernel = 5
                min_dim = min(roi.shape[0], roi.shape[1])
                if min_dim < 5:
                    if min_dim >= 3:
                        blur_kernel = 3
                    else:
                        blur_kernel = None
                if blur_kernel is not None:
                    mask = cv2.medianBlur(mask, blur_kernel)

                cnts = safe_find_contours(mask)
                blobs = find_top_two_blobs(cnts, cfg.min_area)
                if offset_xy != (0.0, 0.0) and blobs:
                    ox, oy = offset_xy
                    blobs = [(bx + ox, by + oy, br, area) for (bx, by, br, area) in blobs]

                now = time.perf_counter()
                dt = None if prev_time is None else max(1e-4, now - prev_time)

                led, radii = associate_leds(blobs, prev_led, cfg.assoc_max_px)
                center = compute_center(led)
                both = (led[0] is not None and led[1] is not None)

                vel_center = update_velocity_ema(center, prev_led, vel_center, dt, cfg.vel_ema_alpha)
                dir_ema, prev_face_angle, face_angle_deg, ang_vel_deg_s, led_span = compute_face_direction(
                    led, dir_ema, prev_face_angle, dt, cfg.dir_ema_alpha)

                debug_img = frame.copy()
                putter_dir = None
                if both and dir_ema is not None:
                    p0, p1 = led[0], led[1]
                    cv2.line(debug_img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (180, 180, 255), 2)
                    mid = (p0 + p1) * 0.5
                    putter_dir = dir_ema.copy()
                    _draw_putter(debug_img, mid, dir_ema, cfg.putter_length_px, cfg.putter_thickness_px)
                elif center is not None:
                    cv2.circle(debug_img, (int(center[0]), int(center[1])), 12, (180, 180, 255), 2)

                impact_hit = False
                contact_pt = None
                if both and dir_ema is not None:
                    mid = (led[0] + led[1]) * 0.5
                    sweep_margin = 0.0
                    if dt:
                        speed = float(np.linalg.norm(vel_center))
                        sweep_margin = cfg.sweep_margin_factor * speed * dt
                    eff_radius = cfg.impact_radius_px + sweep_margin
                    hit, contact, _ = circle_vs_rotated_rect(
                        mid, dir_ema, cfg.putter_length_px, cfg.putter_thickness_px,
                        self._reference_px, eff_radius)
                    if hit:
                        contact_pt = contact
                        cv2.circle(debug_img, (int(contact[0]), int(contact[1])), 6, (0, 255, 255), -1)
                    if hit and not overlap_prev:
                        speed = float(np.linalg.norm(vel_center))
                        if speed >= cfg.min_speed_impact and (now - last_impact_time) > cfg.impact_cooldown_sec:
                            last_impact_time = now
                            impact_hit = True
                    overlap_prev = hit
                else:
                    overlap_prev = False

                if self._show_debug_window:
                    if proc_rect is not None:
                        x1, x2, y1, y2 = proc_rect
                        cv2.rectangle(debug_img, (x1, y1), (x2 - 1, y2 - 1), (0, 140, 255), 1)
                    cv2.circle(debug_img, (int(self._reference_px[0]), int(self._reference_px[1])), int(cfg.impact_radius_px), (255, 0, 255), 2)
                    cv2.imshow(cfg.debug_window_name, debug_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        self._stop.set()
                        break

                state = TrackerState(
                    timestamp=now,
                    center_px=None if center is None else (float(center[0]), float(center[1])),
                    direction_px=None if putter_dir is None else (float(putter_dir[0]), float(putter_dir[1])),
                    span_px=float(led_span if not math.isnan(led_span) else 0.0),
                    visible=bool(both)
                )
                self._register_state(state)

                if impact_hit and center is not None:
                    vel_mag = float(np.linalg.norm(vel_center))
                    if vel_mag > 1e-6:
                        dir_vel = vel_center / vel_mag
                        hit = TrackerHit(
                            timestamp=now,
                            center_px=(float(center[0]), float(center[1])),
                            direction_px=(float(dir_vel[0]), float(dir_vel[1])),
                            speed_px_s=vel_mag,
                            contact_px=(float(contact_pt[0]) if contact_pt is not None else float(center[0]),
                                        float(contact_pt[1]) if contact_pt is not None else float(center[1]))
                        )
                        self._register_hit(hit)

                prev_led = led
                prev_time = now
        finally:
            cap.release()
            if self._show_debug_window and cv2 is not None:
                try:
                    cv2.destroyWindow(cfg.debug_window_name)
                except Exception:
                    pass


def _draw_putter(img, center, dir_vec, length_px, thick_px):
    u = np.array(dir_vec, dtype=np.float32)
    norm = np.linalg.norm(u)
    if norm < 1e-9:
        return
    u /= norm
    v = np.array([-u[1], u[0]], dtype=np.float32)
    cx, cy = float(center[0]), float(center[1])
    L = 0.5 * float(length_px)
    T = 0.5 * float(thick_px)
    c1 = (cx + u[0]*L + v[0]*T, cy + u[1]*L + v[1]*T)
    c2 = (cx - u[0]*L + v[0]*T, cy - u[1]*L + v[1]*T)
    c3 = (cx - u[0]*L - v[0]*T, cy - u[1]*L - v[1]*T)
    c4 = (cx + u[0]*L - v[0]*T, cy + u[1]*L - v[1]*T)
    pts = np.array([[c1, c2, c3, c4]], dtype=np.int32)
    cv2.polylines(img, pts, True, (200, 220, 255), 2)
