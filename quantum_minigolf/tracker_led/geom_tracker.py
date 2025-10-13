import cv2
import numpy as np
import math


def angle_deg_from_vec(vx, vy):
    """0deg along +x, CCW positive."""
    return math.degrees(math.atan2(vy, vx))


def angle_between_deg(v1, v2):
    """Angle in degrees between 2D vectors."""
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(c))


def wrap_angle_rad(a):
    """Wrap radian angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def draw_rotated_rect(img, center_xy, u_dir, length_px, thick_px, color=(200, 220, 255), thickness=2):
    """Draw an oriented rectangle centered at center_xy, axis u_dir."""
    u = np.array(u_dir, dtype=np.float32)
    n = np.linalg.norm(u)
    if n < 1e-9:
        return
    u /= n
    v = np.array([-u[1], u[0]], dtype=np.float32)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    L, T = length_px * 0.5, thick_px * 0.5
    c1 = (cx +  u[0]*L + v[0]*T, cy +  u[1]*L + v[1]*T)
    c2 = (cx -  u[0]*L + v[0]*T, cy -  u[1]*L + v[1]*T)
    c3 = (cx -  u[0]*L - v[0]*T, cy -  u[1]*L - v[1]*T)
    c4 = (cx +  u[0]*L - v[0]*T, cy +  u[1]*L - v[1]*T)
    pts = np.array([[c1, c2, c3, c4]], dtype=np.int32)
    cv2.polylines(img, pts, True, color, thickness)


# ===== Rectangle-vs-circle impact =====

def _unit(v):
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-9 else np.array([1.0, 0.0], dtype=np.float32)


def circle_vs_rotated_rect(rect_center_xy, u_dir, length_px, thick_px, circle_xy, radius_px):
    """
    Returns (hit: bool, closest_pt_world: np.ndarray, dist_sq: float)
    Tests if a circle (the ball) intersects an oriented rectangle (the putter),
    and gives the closest point on the rectangle to the circle center.
    """
    rc = np.array(rect_center_xy, dtype=np.float32)
    q  = np.array(circle_xy,      dtype=np.float32)
    u  = _unit(u_dir)                          # putter axis (LED0->LED1 direction, smoothed)
    v  = np.array([-u[1], u[0]], np.float32)   # perpendicular
    L  = 0.5 * float(length_px)
    T  = 0.5 * float(thick_px)

    # circle center in rect local coordinates
    r   = q - rc
    lx  = float(np.dot(r, u))
    ly  = float(np.dot(r, v))

    # clamp to rectangle to get closest point
    cx  = min(max(lx, -L), L)
    cy  = min(max(ly, -T), T)

    # distance from circle center to that closest point
    dx  = lx - cx
    dy  = ly - cy
    dist_sq = dx*dx + dy*dy
    hit = dist_sq <= (radius_px * radius_px)

    # back to world coords
    closest_world = rc + u*cx + v*cy
    return hit, closest_world, dist_sq


def segment_hits_circle(P0, P1, C, R):
    """
    Prüft, ob das Segment P0->P1 den Kreis (C,R) schneidet.
    Gibt (hit: bool, t_star: float|None, hit_point: np.ndarray|None) zurück.
    """
    P0 = np.array(P0, dtype=np.float32)
    P1 = np.array(P1, dtype=np.float32)
    C  = np.array(C,  dtype=np.float32)
    d  = P1 - P0
    f  = P0 - C

    a = float(np.dot(d, d))
    if a < 1e-12:
        # Degenerates Segment -> Punkt-in-Kreis
        if float(np.dot(f, f)) <= R*R:
            return True, 0.0, P0.copy()
        return False, None, None

    b = 2.0 * float(np.dot(f, d))
    c = float(np.dot(f, f)) - R*R
    disc = b*b - 4*a*c
    if disc < 0.0:
        return False, None, None

    sqrt_disc = float(np.sqrt(disc))
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)

    for t in (t1, t2):
        if 0.0 <= t <= 1.0:
            hit_pt = P0 + t * d
            return True, float(t), hit_pt
    return False, None, None
