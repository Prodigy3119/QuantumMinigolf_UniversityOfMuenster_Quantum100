"""
Two-LED IR Putter Tracker - dual-window views (LED-only + overlay-only),
pose/kinematics, and robust rectangle-vs-ball impact prints - FAST PATH.
- CCL (connected components) for blob finding (no contours).
- CPU threshold/median (avoid CuPy hops at this resolution).
- Persistent buffers for drawing.
- Overlay throttling to cut HUD cost.
"""

import cv2
import numpy as np
import time
import math
import sys

# --- config ---
from cfg_tracker import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    THRESHOLD, MIN_AREA, ASSOC_MAX_PX,
    PUTTER_LENGTH_PX, PUTTER_THICKNESS_PX,
    VEL_EMA_ALPHA, DIR_EMA_ALPHA,
    REF_POINT, IMPACT_RADIUS_PX, MIN_SPEED_IMPACT, IMPACT_COOLDOWN_SEC,
    DEBUG_PRINT_EVERY_N_FRAMES,
    SHOW_LED_WINDOW, SHOW_OVERLAY_WINDOW, ENABLE_OVERLAY_WINDOW, LED_WINDOW_NAME, OVERLAY_WINDOW_NAME,
    LED_VIEW_MIN_RADIUS, LED_VIEW_GLOW, LED_VIEW_GLOW_KERNEL,
    SHOW_OVERLAY_HUD,
    USE_RECT_IMPACT,
    ROI_SEARCH_MARGIN, ROI_MIN_SIZE, ROI_MAX_MISSES,
    SMALL_LED_THRESHOLD_DELTA, SMALL_LED_MIN_AREA, SMALL_LED_DILATE_ITER,
    AUTO_CALIBRATION, USE_GPU_FILTERING,   # USE_GPU_FILTERING is ignored here (CPU is faster at 720p)
    SHOW_GHOST_PUTTER, GHOST_PUTTER_COLOR, GHOST_PUTTER_ALPHA,
    GHOST_ORIENTATION_DEG, GHOST_CENTER_OFFSET_PX,
)
# Optional config: OVERLAY_UPDATE_EVERY (default 2)
try:
    import cfg_tracker as CFG  # noqa: F401
    OVERLAY_UPDATE_EVERY = int(getattr(CFG, "OVERLAY_UPDATE_EVERY", 2))
except Exception:
    OVERLAY_UPDATE_EVERY = 2

from geom_tracker import (
    angle_deg_from_vec, angle_between_deg, wrap_angle_rad,
    draw_rotated_rect, circle_vs_rotated_rect, segment_hits_circle
)
from vision_tracker import (
    associate_leds, compute_center,
    update_velocity_ema, compute_face_direction
)

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(cv2.getNumberOfCPUs())
except Exception:
    pass


# =========================
# Helpers / Fast Paths
# =========================

def _ensure_u8(img):
    return img if img.dtype == np.uint8 else img.astype(np.uint8, copy=False)


def find_top_two_blobs_ccl(mask, min_area=5):
    """
    FAST: binary mask -> top-2 blobs via connected components.
    Returns list [(cx, cy, r, area), ...] sorted by area desc (max 2).
    """
    mask = _ensure_u8(mask)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return []

    areas = stats[1:, cv2.CC_STAT_AREA]
    centers = centroids[1:]
    order = np.argsort(-areas)[:2]

    out = []
    for idx in order:
        area = float(areas[idx])
        if area < max(1, min_area):
            continue
        cx, cy = float(centers[idx][0]), float(centers[idx][1])
        r = float(math.sqrt(area / math.pi))  # equivalent circle radius
        out.append((cx, cy, r, area))
    return out


def build_led_view(led, radii):
    """LED-only window image (grayscale), reusing a persistent buffer."""
    buf = getattr(build_led_view, "_buf", None)
    if buf is None or buf.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        buf = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        build_led_view._buf = buf
    else:
        buf.fill(0)

    for i in (0, 1):
        if led[i] is None:
            continue
        cx, cy = int(round(led[i][0])), int(round(led[i][1]))
        cr = max(LED_VIEW_MIN_RADIUS, int(round(radii[i])))
        cv2.circle(buf, (cx, cy), cr, 255, -1)

    if LED_VIEW_GLOW:
        k = max(3, LED_VIEW_GLOW_KERNEL | 1)  # ensure odd >= 3
        cv2.GaussianBlur(buf, (k, k), 0, dst=buf)
    return buf


def build_overlay(led, dir_ema, both, debug_contact_pt,
                  speed, path_angle_deg, face_angle_deg,
                  face_to_path_deg, ang_vel_deg_s, led_span,
                  stats=None):
    """
    Overlay-only window image (black background + geometry/HUD),
    reusing a persistent buffer.
    """
    overlay = getattr(build_overlay, "_buf", None)
    if overlay is None or overlay.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        overlay = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        build_overlay._buf = overlay
    else:
        overlay[...] = 0

    # ball zone
    cv2.circle(overlay, (int(REF_POINT[0]), int(REF_POINT[1])), IMPACT_RADIUS_PX, (255, 0, 255), 2)

    # ghost putter (visual only)
    if SHOW_GHOST_PUTTER:
        ghost_center = np.array(REF_POINT, dtype=np.float32) + np.array(GHOST_CENTER_OFFSET_PX, dtype=np.float32)
        theta = math.radians(GHOST_ORIENTATION_DEG)
        u_dir = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        ghost_color = (int(GHOST_PUTTER_COLOR[2]), int(GHOST_PUTTER_COLOR[1]), int(GHOST_PUTTER_COLOR[0]))

        draw_rotated_rect(overlay, ghost_center, u_dir, PUTTER_LENGTH_PX, PUTTER_THICKNESS_PX,
                          color=ghost_color, thickness=2)
        if GHOST_PUTTER_ALPHA > 0.0:
            fill = np.zeros_like(overlay)
            u = u_dir / max(np.linalg.norm(u_dir), 1e-6)
            v = np.array([-u[1], u[0]], dtype=np.float32)
            L = PUTTER_LENGTH_PX * 0.5
            T = PUTTER_THICKNESS_PX * 0.5
            corners = np.array([
                ghost_center + u * L + v * T,
                ghost_center - u * L + v * T,
                ghost_center - u * L - v * T,
                ghost_center + u * L - v * T,
            ], dtype=np.int32)
            cv2.fillConvexPoly(fill, corners, ghost_color)
            overlay[:] = cv2.addWeighted(overlay, 1.0, fill, float(GHOST_PUTTER_ALPHA), 0.0)

    # LEDs
    colors = [(50, 200, 255), (255, 180, 50)]
    for i in (0, 1):
        if led[i] is not None:
            p = (int(round(led[i][0])), int(round(led[i][1])))
            cv2.circle(overlay, p, 5, colors[i], -1, lineType=cv2.LINE_AA)

    # dir crosshair if available
    if dir_ema is not None and (led[0] is not None or led[1] is not None):
        cx, cy = compute_center(led)
        u = dir_ema / max(np.linalg.norm(dir_ema), 1e-6)
        v = np.array([-u[1], u[0]], dtype=np.float32)
        scale = 60
        a = (int(cx - u[0] * scale), int(cy - u[1] * scale))
        b = (int(cx + u[0] * scale), int(cy + u[1] * scale))
        c = (int(cx - v[0] * scale), int(cy - v[1] * scale))
        d = (int(cx + v[0] * scale), int(cy + v[1] * scale))
        cv2.line(overlay, a, b, (120, 200, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, c, d, (120, 200, 255), 1, lineType=cv2.LINE_AA)

    # HUD text
    if SHOW_OVERLAY_HUD:
        def put(s, y):
            cv2.putText(overlay, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
        put("Two-LED Putter Tracker", 22)
        put(f"speed={speed:6.1f}px/s   path={path_angle_deg:6.1f}deg   face={face_angle_deg:6.1f}deg   Delta={face_to_path_deg:5.1f}deg   omega={ang_vel_deg_s:6.1f}deg/s   span={led_span:5.1f}px", 42)
        if stats is not None and "frame_ms" in stats:
            put(f"frame={stats['frame_ms']:5.1f} ms   proc={stats.get('proc_ms', 0.0):5.1f} ms   center=({stats.get('center_x', 0):.1f},{stats.get('center_y', 0):.1f})", 62)

    # debug impact closest point
    if debug_contact_pt is not None:
        q = (int(debug_contact_pt[0]), int(debug_contact_pt[1]))
        cv2.circle(overlay, q, 3, (255, 0, 255), -1, lineType=cv2.LINE_AA)

    return overlay


# =========================
# Main App
# =========================

def main():
    import cv2, math, time
    import numpy as np

    # ---- Imports aus deinem Projekt ----
    from cfg_tracker import (
        CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS, THRESHOLD, MIN_AREA,
        ASSOC_MAX_PX, PUTTER_LENGTH_PX, PUTTER_THICKNESS_PX,
        REF_POINT, IMPACT_RADIUS_PX, MIN_SPEED_IMPACT, IMPACT_COOLDOWN_SEC,
        SHOW_LED_WINDOW, SHOW_OVERLAY_WINDOW, ENABLE_OVERLAY_WINDOW,
        LED_WINDOW_NAME, OVERLAY_WINDOW_NAME,
        LED_VIEW_MIN_RADIUS, LED_VIEW_GLOW, LED_VIEW_GLOW_KERNEL,
        SHOW_OVERLAY_HUD, SHOW_GHOST_PUTTER,
        GHOST_PUTTER_COLOR, GHOST_PUTTER_ALPHA,
        GHOST_ORIENTATION_DEG, GHOST_CENTER_OFFSET_PX,
        USE_RECT_IMPACT, VEL_EMA_ALPHA, DIR_EMA_ALPHA,
        ROI_SEARCH_MARGIN, ROI_MIN_SIZE, ROI_MAX_MISSES,
        SMALL_LED_THRESHOLD_DELTA, SMALL_LED_MIN_AREA, SMALL_LED_DILATE_ITER,
        AUTO_CALIBRATION, USE_GPU_FILTERING, OVERLAY_UPDATE_EVERY,
        DEBUG_PRINT_EVERY_N_FRAMES
    )
    from vision_tracker import (
        find_top_two_blobs_ccl, associate_leds, compute_center,
        update_velocity_ema, compute_face_direction
    )
    from geom_tracker import (
        angle_deg_from_vec, circle_vs_rotated_rect, segment_hits_circle
    )

    # ---- Local overrides (für 60 fps, ohne cfg zu ändern) ----
    local_TARGET_FPS       = 60
    local_ASSOC_MAX_PX     = max(ASSOC_MAX_PX, 280.0)           # robustere LED-Zuordnung
    local_MIN_SPEED_IMPACT = max(60.0, MIN_SPEED_IMPACT * 0.8)  # etwas toleranter

    # ---- kleine Helpers ----
    def now_s():
        return time.perf_counter()

    # ---- Kamera Setup ----
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS,          int(local_TARGET_FPS))

    # ---- Fenster ----
    if SHOW_LED_WINDOW:
        cv2.namedWindow(LED_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if SHOW_OVERLAY_WINDOW or ENABLE_OVERLAY_WINDOW:
        cv2.namedWindow(OVERLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)

    # ---- States ----
    t_prev = now_s()
    t0     = t_prev
    frame_idx = 0

    prev_led = [None, None]
    prev_center_raw = None
    vel_center = np.zeros(2, dtype=np.float32)
    dir_ema = None
    prev_face_angle = None

    overlap_prev = False
    last_impact_t = -1e9

    # ---- Main Loop ----
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Warn: Kein Frame von der Kamera.")
            break

        # Timing
        t_now = now_s()
        dt = t_now - t_prev
        if dt <= 0:
            dt = 1.0 / max(1, local_TARGET_FPS)
        t_prev = t_now
        frame_idx += 1

        # Preproc
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        # Threshold + Blob-Find (Fallback-Pfad wenn nix gefunden)
        _, mask = cv2.threshold(gray, int(THRESHOLD), 255, cv2.THRESH_BINARY)
        blobs = find_top_two_blobs_ccl(mask, min_area=int(MIN_AREA))
        if len(blobs) == 0:
            thr2 = max(0, int(THRESHOLD) - int(SMALL_LED_THRESHOLD_DELTA))
            _, mask2 = cv2.threshold(gray, thr2, 255, cv2.THRESH_BINARY)
            if SMALL_LED_DILATE_ITER > 0:
                kernel = np.ones((3,3), np.uint8)
                mask2 = cv2.dilate(mask2, kernel, iterations=int(SMALL_LED_DILATE_ITER))
            blobs = find_top_two_blobs_ccl(mask2, min_area=int(SMALL_LED_MIN_AREA))
            mask = mask2

        # LED-Association (größerer Radius bei schneller Bewegung)
        led, radii = associate_leds(blobs, prev_led, assoc_max_px=float(local_ASSOC_MAX_PX))

        # Center + Geschwindigkeit (EMA)
        center = compute_center(led)
        vel_center = update_velocity_ema(center, prev_led, vel_center, dt=float(dt), alpha=float(VEL_EMA_ALPHA))
        speed = float(np.linalg.norm(vel_center))

        # Pfadrichtung
        path_angle_deg = angle_deg_from_vec(vel_center[0], vel_center[1]) if speed > 1e-3 else float("nan")

        # Schlagfläche + Winkelgeschwindigkeit
        dir_ema, prev_face_angle, face_angle_deg, ang_vel_deg_s, led_span = \
            compute_face_direction(led, dir_ema, prev_face_angle, dt=float(dt), alpha=float(DIR_EMA_ALPHA))

        # --------- Impact Detection (Continuous + Rechteck) ---------
        seg_hit = False
        seg_t   = None
        seg_pt  = None
        if prev_center_raw is not None and center is not None:
            seg_hit, seg_t, seg_pt = segment_hits_circle(prev_center_raw, center, REF_POINT, float(IMPACT_RADIUS_PX))

        hit_rect = False
        debug_contact_pt = None

        if USE_RECT_IMPACT and center is not None:
            theta = math.radians(float(GHOST_ORIENTATION_DEG))
            u_dir = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
            ghost_center = np.array(REF_POINT, dtype=np.float32) + np.array(GHOST_CENTER_OFFSET_PX, dtype=np.float32)

            hit_rect, closest_world, _ = circle_vs_rotated_rect(
                ghost_center, u_dir,
                float(PUTTER_LENGTH_PX), float(PUTTER_THICKNESS_PX),
                np.array(center, dtype=np.float32),
                float(IMPACT_RADIUS_PX)
            )
            if hit_rect:
                debug_contact_pt = closest_world

        # finaler Hit: Continuous ODER Rechteck UND Mindestgeschwindigkeit
        hit_now = (seg_hit or hit_rect) and (speed >= float(local_MIN_SPEED_IMPACT))

        # Entprellung (edge→inside + Cooldown)
        if hit_now and not overlap_prev and (t_now - last_impact_t) > float(IMPACT_COOLDOWN_SEC):
            last_impact_t = t_now
            tsec = t_now - t0

            if seg_pt is not None:
                debug_contact_pt = seg_pt  # hübscherer Kontaktpunkt fürs HUD

            print(f"[IMPACT {tsec:7.3f}s] center=({center[0]:.1f},{center[1]:.1f})  "
                  f"|v|={speed:.0f}px/s  path={path_angle_deg:.1f}deg  face={face_angle_deg:.1f}deg  "
                  f"Delta={(face_angle_deg - path_angle_deg) if (not math.isnan(face_angle_deg) and not math.isnan(path_angle_deg)) else float('nan'):.1f}deg  "
                  f"omega={ang_vel_deg_s:.0f}deg/s  LEDspan={led_span:.1f}px")

        overlap_prev = hit_now

        # --------- LED-View ---------
        if SHOW_LED_WINDOW:
            ledview = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            colors = [(0,255,255), (255,0,255)]
            for i, p in enumerate(led):
                if p is not None:
                    r = max(LED_VIEW_MIN_RADIUS, int(radii[i]))
                    cv2.circle(ledview, (int(p[0]), int(p[1])), r, colors[i], -1, lineType=cv2.LINE_AA)
            if seg_pt is not None:
                cv2.circle(ledview, (int(seg_pt[0]), int(seg_pt[1])), 6, (0, 220, 0), -1, lineType=cv2.LINE_AA)
            if LED_VIEW_GLOW:
                k = max(3, int(LED_VIEW_GLOW_KERNEL) | 1)
                ledview = cv2.GaussianBlur(ledview, (k,k), 0)
            cv2.imshow(LED_WINDOW_NAME, ledview)

        # --------- Overlay-View (throttled) ---------
        if SHOW_OVERLAY_WINDOW or ENABLE_OVERLAY_WINDOW:
            do_draw = (OVERLAY_UPDATE_EVERY <= 1) or (frame_idx % int(OVERLAY_UPDATE_EVERY) == 0)
            if do_draw:
                overlay = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

                # Ball/Impact-Zone
                cv2.circle(overlay, (int(REF_POINT[0]), int(REF_POINT[1])), int(IMPACT_RADIUS_PX), (60, 180, 60), 2, cv2.LINE_AA)

                # Ghost-Putter (optional)
                if SHOW_GHOST_PUTTER:
                    theta = math.radians(float(GHOST_ORIENTATION_DEG))
                    u_dir = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
                    ghost_center = np.array(REF_POINT, dtype=np.float32) + np.array(GHOST_CENTER_OFFSET_PX, dtype=np.float32)

                    L = float(PUTTER_LENGTH_PX) * 0.5
                    T = float(PUTTER_THICKNESS_PX) * 0.5
                    v_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float32)
                    corners = [
                        ghost_center + u_dir*(-L) + v_dir*(-T),
                        ghost_center + u_dir*(+L) + v_dir*(-T),
                        ghost_center + u_dir*(+L) + v_dir*(+T),
                        ghost_center + u_dir*(-L) + v_dir*(+T),
                    ]
                    pts = np.array(corners, dtype=np.int32)
                    cv2.fillConvexPoly(overlay, pts, GHOST_PUTTER_COLOR)
                    cv2.addWeighted(overlay, GHOST_PUTTER_ALPHA, overlay, 1.0 - GHOST_PUTTER_ALPHA, 0, dst=overlay)

                # LEDs & center
                if center is not None:
                    cv2.circle(overlay, (int(center[0]), int(center[1])), 4, (255,255,255), -1, cv2.LINE_AA)
                for p in led:
                    if p is not None:
                        cv2.circle(overlay, (int(p[0]), int(p[1])), 5, (200,200,0), -1, cv2.LINE_AA)

                # Kontaktpunkt
                if debug_contact_pt is not None:
                    cv2.circle(overlay, (int(debug_contact_pt[0]), int(debug_contact_pt[1])), 6, (0,255,0), -1, cv2.LINE_AA)

                # HUD
                if SHOW_OVERLAY_HUD:
                    def put(txt, y):
                        cv2.putText(overlay, txt, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240,240,240), 1, cv2.LINE_AA)
                    put(f"fps: {1.0/dt:5.1f}", 24)
                    put(f"speed: {speed:6.0f} px/s", 44)
                    put(f"path:  {path_angle_deg:6.1f} deg", 64)
                    put(f"face:  {face_angle_deg:6.1f} deg", 84)
                    if not (math.isnan(face_angle_deg) or math.isnan(path_angle_deg)):
                        put(f"delta: {(face_angle_deg - path_angle_deg):6.1f} deg", 104)
                    put(f"omega: {ang_vel_deg_s:6.1f} deg/s", 124)
                    put(f"LED span: {led_span:6.1f} px", 144)

                cv2.imshow(OVERLAY_WINDOW_NAME, overlay)

        # Debug-Prints alle N Frames
        if DEBUG_PRINT_EVERY_N_FRAMES and (frame_idx % int(DEBUG_PRINT_EVERY_N_FRAMES) == 0):
            print(f"[{frame_idx}] |v|={speed:.0f}px/s path={path_angle_deg:.1f} face={face_angle_deg:.1f} span={led_span:.1f}")

        # Update prevs
        prev_led = [None if led[0] is None else led[0].copy(),
                    None if led[1] is None else led[1].copy()]
        if center is not None:
            prev_center_raw = center.copy()

        # Exit?
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# =========================
# Calibration (optional)
# =========================

def auto_calibrate(cap):
    """
    Quick interactive calibration:
      1) Dark frames -> noise baseline (mean,std)
      2) Bright frames with LEDs -> brightness stats
      3) Estimate threshold, MIN_AREA, ROI_SEARCH_MARGIN
    """
    global THRESHOLD, MIN_AREA, ROI_SEARCH_MARGIN

    print("\n[calibration] Auto calibration begins.")
    print("Step 1: Cover the camera/keep LEDs off. Press Enter when ready...")
    try:
        input()
    except EOFError:
        pass
    dark = _collect_samples(cap, sample_count=60)
    dark_mean = float(np.mean(dark))
    dark_std = float(np.std(dark))
    print(f"[calibration] Dark frames -> mean={dark_mean:.2f}, std={dark_std:.2f}")

    print("Step 2: Uncover camera / point LEDs at lens. Press Enter when ready...")
    try:
        input()
    except EOFError:
        pass
    bright = _collect_samples(cap, sample_count=60)
    bright_mean = float(np.mean(bright))
    bright_max  = float(np.max(bright))
    bright_std  = float(np.std(bright))
    print(f"[calibration] Bright frames -> mean={bright_mean:.2f}, max={bright_max:.2f}, std={bright_std:.2f}")

    # Heuristic: pick threshold between dark tail and bright bulk
    # clamp to [50, 250]
    new_thresh = int(np.clip(dark_mean + 4.0 * dark_std, 50, 250))
    THRESHOLD = max(THRESHOLD, new_thresh)
    print(f"[calibration] Updated threshold -> {THRESHOLD}")

    # Small pass to estimate MIN_AREA / ROI range
    test = _collect_samples(cap, sample_count=20).astype(np.uint8)
    test_mask = cv2.medianBlur(cv2.threshold(test.max(axis=0), THRESHOLD, 255, cv2.THRESH_BINARY)[1], 3)
    blobs = find_top_two_blobs_ccl(test_mask, min_area=1)
    if blobs:
        areas = [b[3] for b in blobs]
        MIN_AREA = max(1, int(np.mean(areas) * 0.25))
        xs = [b[0] for b in blobs]
        ys = [b[1] for b in blobs]
        if xs and ys:
            spread = max(np.std(xs), np.std(ys))
            ROI_SEARCH_MARGIN = int(np.clip(spread * 4, 80, 320))
        print(f"[calibration] Estimated MIN_AREA={MIN_AREA}, ROI_MARGIN={ROI_SEARCH_MARGIN}")

    print("[calibration] Complete.\n")


def _collect_samples(cap, sample_count=60):
    frames = []
    for _ in range(sample_count):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
        time.sleep(max(1.0 / TARGET_FPS * 0.5, 0.005))
    if not frames:
        return np.zeros((1,), dtype=np.float32)
    return np.stack(frames)


if __name__ == "__main__":
    main()
