from __future__ import annotations
import numpy as np
from matplotlib.patches import Rectangle
from typing import List, Tuple
from .backends import Backend
from .config import GameConfig

class Course:
    """
    Builds the static course state: potential V, absorber W, hole mask,
    visible patches, and solid rectangles for classical collisions.
    Also maintains expV_half/expK for the current dt.
    """
    def __init__(self, cfg: GameConfig, be: Backend):
        self.cfg = cfg
        self.be = be

        # (Optionally) scale resolution
        if cfg.res_scale != 1.0:
            self.cfg.Nx = max(64, int(round(cfg.Nx * cfg.res_scale)))
            self.cfg.Ny = max(48, int(round(cfg.Ny * cfg.res_scale)))
            self.cfg.hole_r = max(4, int(round(cfg.hole_r * cfg.res_scale)))

        self.Nx, self.Ny = self.cfg.Nx, self.cfg.Ny
        self.hole_center = np.array([self.Nx - self.Nx * .1, self.Ny / 2], dtype=float)

        # CPU masks useful for sampling/bias
        self.hole_mask_cpu = None  # type: np.ndarray | None

        # Boundary handling
        self.edge_boundary = getattr(self.cfg, 'edge_boundary', 'reflect')
        self.edge_reflect_cells = int(max(0, getattr(self.cfg, 'edge_reflect_cells', 0)))
        self.edge_reflect_height = float(getattr(self.cfg, 'edge_reflect_height', self.cfg.V_edge))
        self._active_reflect_cells = 0

        self.V = None
        self.W_absorb = None
        self.expV_half = None
        self.expK = None

        # displayed walls and solid hitboxes
        self.course_patches: List[Rectangle] = []
        self.solid_rects: List[Tuple[int,int,int,int]] = []

        # initial build
        self.set_map(self.cfg.map_kind)

    def _build_common_border(self, V: np.ndarray):
        Nx, Ny = self.Nx, self.Ny
        max_width = max(1, min(Nx // 2, Ny // 2))
        if self.edge_boundary == 'reflect':
            width = int(max(1, self.edge_reflect_cells or 0))
            width = min(width, max_width)
            if width <= 0:
                width = 1
            high = max(self.cfg.V_edge, self.edge_reflect_height)
            ramp = np.linspace(high, self.cfg.V_edge, width, dtype=np.float32)
            self._active_reflect_cells = width
        else:
            width = min(3, max_width)
            ramp = np.full(width, self.cfg.V_edge, dtype=np.float32)
            self._active_reflect_cells = 0

        for i, val in enumerate(ramp):
            V[i, :] = np.maximum(V[i, :], val)
            V[-(i + 1), :] = np.maximum(V[-(i + 1), :], val)
            V[:, i] = np.maximum(V[:, i], val)
            V[:, -(i + 1)] = np.maximum(V[:, -(i + 1)], val)

    def _build_absorber(self) -> np.ndarray:
        Nx, Ny = self.Nx, self.Ny
        if self.edge_boundary == 'reflect':
            return np.zeros((Ny, Nx), dtype=np.float32)
        aw = int(self.cfg.absorb_width)
        W = np.zeros((Ny, Nx), dtype=np.float32)
        if aw > 0 and self.cfg.absorb_strength > 0:
            rx = np.minimum(np.arange(Nx), np.arange(Nx)[::-1]).astype(np.float32)
            ry = np.minimum(np.arange(Ny), np.arange(Ny)[::-1]).astype(np.float32)
            RX, RY = np.meshgrid(rx, ry, indexing='xy')
            d = np.minimum(RX, RY)
            m = np.clip((aw - d) / max(1, aw), 0.0, 1.0)
            W = (self.cfg.absorb_strength * (m ** 2)).astype(np.float32)
        return W

    def _hole_mask(self) -> np.ndarray:
        Xc, Yc = np.meshgrid(np.arange(self.Nx, dtype=np.float32),
                             np.arange(self.Ny, dtype=np.float32), indexing='xy')
        dxm = Xc - self.hole_center[0]
        dym = Yc - self.hole_center[1]
        return (dxm * dxm + dym * dym) <= (self.cfg.hole_r ** 2)

    def set_map(self, kind: str):
        self.cfg.map_kind = kind
        Nx, Ny = self.Nx, self.Ny
        V = np.zeros((Ny, Nx), dtype=np.float32)
        patches: list[Rectangle] = []
        solids: list[tuple[int,int,int,int]] = []

        self._build_common_border(V)
        cx = Nx // 2
        wall_w = max(2, self.cfg.center_wall_width)

        if kind == "single_wall":
            x1, y1, x2, y2 = self._apply_single_wall(V, cx, Ny)
            patches.append(Rectangle((x1, y1), x2 - x1, y2 - y1))
            solids.append((x1, y1, x2, y2))

        elif kind == "double_slit":
            slit_h = max(4, self.cfg.slit_height)
            slit_sep = max(6, self.cfg.slit_sep)
            margin = int(Ny * 0.08)
            x1, x2 = cx - wall_w // 2, cx + wall_w // 2
            y1, y2 = margin, Ny - margin
            V[y1:y2, x1:x2] = self.cfg.V_wall
            cy = Ny // 2
            s1y1 = max(y1, cy - slit_sep // 2 - slit_h // 2)
            s1y2 = min(y2, cy - slit_sep // 2 + slit_h // 2)
            s2y1 = max(y1, cy + slit_sep // 2 - slit_h // 2)
            s2y2 = min(y2, cy + slit_sep // 2 + slit_h // 2)
            V[s1y1:s1y2, x1:x2] = 0.0
            V[s2y1:s2y2, x1:x2] = 0.0

            if s1y1 > y1:   patches.append(Rectangle((x1, y1), x2 - x1, s1y1 - y1));   solids.append((x1, y1, x2, s1y1))
            if s2y1 > s1y2: patches.append(Rectangle((x1, s1y2), x2 - x1, s2y1 - s1y2)); solids.append((x1, s1y2, x2, s2y1))
            if y2 > s2y2:   patches.append(Rectangle((x1, s2y2), x2 - x1, y2 - s2y2));   solids.append((x1, s2y2, x2, y2))

        elif kind == "single_slit":
            slit_h = max(4, self.cfg.slit_height)
            margin = int(Ny * 0.08)
            x1, x2 = cx - wall_w // 2, cx + wall_w // 2
            y1, y2 = margin, Ny - margin
            V[y1:y2, x1:x2] = self.cfg.V_wall
            cy = Ny // 2
            sy1 = max(y1, cy - slit_h // 2)
            sy2 = min(y2, cy + slit_h // 2)
            V[sy1:sy2, x1:x2] = 0.0

            if sy1 > y1: patches.append(Rectangle((x1, y1), x2 - x1, sy1 - y1)); solids.append((x1, y1, x2, sy1))
            if y2 > sy2: patches.append(Rectangle((x1, sy2), x2 - x1, y2 - sy2)); solids.append((x1, sy2, x2, y2))

        elif kind == "no_obstacle":
            pass
        else:
            raise ValueError(f"Unknown map kind: {kind}")

        self.V = self.be.to_xp(V, np.float32)
        self.W_absorb = self.be.to_xp(self._build_absorber(), np.float32)
        self.course_patches = patches
        self.solid_rects = solids

        # Hole mask (CPU + XP)
        self.hole_mask_cpu = self._hole_mask()
        self.hole_mask = self.be.to_xp(self.hole_mask_cpu, bool)

        # Combined operator used for dynamic dt scaling
        self.V_operator = ((-1j * self.V) - self.W_absorb).astype(np.complex64)

    def update_exponents(self, dt: float, k2, dtype):
        # expV_half = exp(((-iV) - W) * dt/2), expK = exp(-i k^2 dt / 2)
        xp = self.be.xp
        self.expV_half = xp.exp(((-1j * self.V) - self.W_absorb) * (dt * 0.5)).astype(dtype)
        self.expK = xp.exp((-1j * k2 * dt / 2.0).astype(dtype)).astype(dtype)

    def _apply_single_wall(self, V: np.ndarray, cx: int, Ny: int):
        thickness = float(np.clip(getattr(self.cfg, 'single_wall_thickness_factor', 1.0), 0.05, 5.0))
        base_width = max(1, self.cfg.single_wall_width)
        width_scale = float(np.clip(thickness ** 0.6, 0.2, 3.0))
        w = max(1, int(round(base_width * width_scale)))
        x1 = max(3, cx - w)
        x2 = min(self.Nx - 3, cx + w)
        if x2 <= x1:
            x1 = max(3, cx - 1)
            x2 = min(self.Nx - 3, cx + 1)
        y1 = int(Ny * 0.02)
        y2 = int(Ny * 0.98)

        weight = float(np.clip(getattr(self.cfg, 'tunneling_thickness_weight', 1.0), 0.0, 1.0))
        power = float(self.cfg.barrier_thickness_power * (1.0 + weight))
        height_scale = float(np.clip(thickness ** power, 1e-3, 1e6))
        V[y1:y2, x1:x2] = self.cfg.V_wall * height_scale
        return x1, y1, x2, y2

    def apply_edge_boundary(self, psi):
        if psi is None:
            return psi
        if self.edge_boundary != 'reflect':
            return psi
        b = int(max(0, self._active_reflect_cells))
        if b <= 0:
            return psi
        Ny, Nx = psi.shape
        b = min(b, Ny // 2, Nx // 2)
        if b <= 0:
            return psi
        xp = self.be.xp

        top = xp.flip(psi[b:2 * b, :], axis=0).copy()
        bottom = xp.flip(psi[-2 * b:-b, :], axis=0).copy()
        left = xp.flip(psi[:, b:2 * b], axis=1).copy()
        right = xp.flip(psi[:, -2 * b:-b], axis=1).copy()

        psi[:b, :] = top
        psi[-b:, :] = bottom
        psi[:, :b] = left
        psi[:, -b:] = right
        return psi
