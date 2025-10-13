import cv2
import numpy as np
import math

from geom_tracker import wrap_angle_rad


def safe_find_contours(binary_img):
    """cv2.findContours compatibility wrapper."""
    cnts = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts[0] if len(cnts) == 2 else cnts[1]


def find_top_two_blobs(contours, min_area=5):
    """
    From contours, compute min-enclosing-circle blobs, filter by area,
    and return the two largest by area.
    Returns list of tuples: (x, y, r, area)
    """
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < max(1, min_area):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        blobs.append((float(x), float(y), float(r), float(area)))
    blobs.sort(key=lambda b: b[3], reverse=True)
    return blobs[:2]


# FAST blob detection via connected components (binary mask -> top 2 blobs)
def find_top_two_blobs_ccl(mask, min_area=5):
    """
    mask: uint8 binary image 0/255
    returns list [(cx, cy, r, area), ...] for the two largest blobs
    """
    # ensure binary uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8, copy=False)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return []

    # drop background row 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    centers = centroids[1:]

    order = np.argsort(-areas)[:2]  # top-2 by area
    out = []
    for idx in order:
        area = float(areas[idx])
        if area < max(1, min_area):
            continue
        cx, cy = float(centers[idx][0]), float(centers[idx][1])
        # area of a circle -> equivalent radius (good enough for drawing/assoc)
        r = float(math.sqrt(area / math.pi))
        out.append((cx, cy, r, area))
    return out



def associate_leds(blobs, prev_led, assoc_max_px=120.0):
    """
    Associate detected blobs to LED0/LED1 using nearest-neighbor logic,
    replicating the original behavior.
    Returns (led, radii) where led is [p0, p1] with np.ndarray or None.
    """
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
            led[slot] = np.array([bx, by], dtype=np.float32); radii[slot] = br
        else:
            led = [np.array([bx, by], dtype=np.float32), None]; radii = [br, 0.0]

    elif len(blobs) == 2:
        (x1, y1, r1, _), (x2, y2, r2, _) = blobs
        if prev_led[0] is not None and prev_led[1] is not None:
            cost_a = np.hypot(x1-prev_led[0][0], y1-prev_led[0][1]) + np.hypot(x2-prev_led[1][0], y2-prev_led[1][1])
            cost_b = np.hypot(x2-prev_led[0][0], y2-prev_led[0][1]) + np.hypot(x1-prev_led[1][0], y1-prev_led[1][1])
            swap = cost_b < cost_a
        else:
            swap = False
        if not swap:
            led = [np.array([x1, y1], dtype=np.float32), np.array([x2, y2], dtype=np.float32)]
            radii = [r1, r2]
        else:
            led = [np.array([x2, y2], dtype=np.float32), np.array([x1, y1], dtype=np.float32)]
            radii = [r2, r1]

    return led, radii


def compute_center(led):
    """Compute center from LED list [p0, p1]."""
    both = (led[0] is not None and led[1] is not None)
    if both:
        return (led[0] + led[1]) * 0.5
    elif led[0] is not None:
        return led[0].copy()
    elif led[1] is not None:
        return led[1].copy()
    return None


def update_velocity_ema(center, prev_led, vel_center, dt, alpha):
    """
    Update EMA of center velocity, matching original logic.
    Returns updated vel_center (np.ndarray).
    """
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
    """
    Compute face direction (smoothed), angular velocity, and LED span.
    Returns (dir_ema_new, prev_face_angle_new, face_angle_deg, ang_vel_deg_s, led_span)
    """
    face_angle_deg = float("nan")
    ang_vel_deg_s = float("nan")
    led_span = float("nan")

    if led[0] is not None and led[1] is not None:
        vec = led[1] - led[0]        # LED0 -> LED1
        led_span = float(np.linalg.norm(vec))
        if led_span > 1e-6:
            u = vec / led_span
            if dir_ema is None:
                dir_ema = u
            else:
                if np.dot(dir_ema, u) < 0:  # avoid 180deg flips
                    u = -u
                dir_ema = (1.0 - alpha) * dir_ema + alpha * u
                dir_ema /= max(np.linalg.norm(dir_ema), 1e-9)

            face_angle_rad = math.atan2(dir_ema[1], dir_ema[0])
            face_angle_deg = math.degrees(face_angle_rad)

            if prev_face_angle is not None and dt and dt > 0:
                d = wrap_angle_rad(face_angle_rad - prev_face_angle)
                ang_vel_deg_s = math.degrees(d / dt)
            prev_face_angle = face_angle_rad

    return dir_ema, prev_face_angle, face_angle_deg, ang_vel_deg_s, led_span
