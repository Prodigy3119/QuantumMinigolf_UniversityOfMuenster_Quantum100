from __future__ import annotations
import math
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

    def _thickness_scales(self) -> tuple[float, float]:
        thickness = float(np.clip(getattr(self.cfg, 'single_wall_thickness_factor', 1.0), 0.05, 2.5))
        width_scale = float(np.clip(thickness ** 0.6, 0.2, 3.0))
        weight = float(np.clip(getattr(self.cfg, 'tunneling_thickness_weight', 1.0), 0.0, 1.0))
        power = float(self.cfg.barrier_thickness_power * (1.0 + weight))
        height_scale = float(np.clip(thickness ** power, 1e-3, 1e6))
        return width_scale, height_scale

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
        width_scale, height_scale = self._thickness_scales()
        wall_w_base = max(2, self.cfg.center_wall_width)
        wall_w = max(2, int(round(wall_w_base * width_scale)))

        if kind == "single_wall":
            x1, y1, x2, y2 = self._apply_single_wall(V, cx, Ny)
            patches.append(Rectangle((x1, y1), x2 - x1, y2 - y1))
            solids.append((x1, y1, x2, y2))

        elif kind == "double_slit":
            slit_h = max(4, self.cfg.slit_height)
            slit_sep = max(6, self.cfg.slit_sep)
            margin = int(Ny * 0.08)
            half = wall_w // 2
            x1 = max(3, cx - half)
            x2 = min(self.Nx - 3, cx + half + (wall_w % 2))
            if x2 <= x1:
                x1 = max(3, cx - 1)
                x2 = min(self.Nx - 3, cx + 1)
            y1, y2 = margin, Ny - margin
            V[y1:y2, x1:x2] = self.cfg.V_wall * height_scale
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
            half = wall_w // 2
            x1 = max(3, cx - half)
            x2 = min(self.Nx - 3, cx + half + (wall_w % 2))
            if x2 <= x1:
                x1 = max(3, cx - 1)
                x2 = min(self.Nx - 3, cx + 1)
            y1, y2 = margin, Ny - margin
            V[y1:y2, x1:x2] = self.cfg.V_wall * height_scale
            cy = Ny // 2
            sy1 = max(y1, cy - slit_h // 2)
            sy2 = min(y2, cy + slit_h // 2)
            V[sy1:sy2, x1:x2] = 0.0

            if sy1 > y1: patches.append(Rectangle((x1, y1), x2 - x1, sy1 - y1)); solids.append((x1, y1, x2, sy1))
            if y2 > sy2: patches.append(Rectangle((x1, sy2), x2 - x1, y2 - sy2)); solids.append((x1, sy2, x2, y2))

        elif kind == "Uni_Logo":
            def add_rect(
                xf0: float,
                xf1: float,
                yf0: float,
                yf1: float,
                *,
                scale_x: bool = True,
                scale_y: bool = False,
                anchor_right: bool = False,
                anchor_left: bool = False,
                anchor_top: bool = False,
                anchor_bottom: bool = False,
            ):
                xf0 = float(np.clip(xf0, 0.0, 1.0))
                xf1 = float(np.clip(xf1, 0.0, 1.0))
                yf0 = float(np.clip(yf0, 0.0, 1.0))
                yf1 = float(np.clip(yf1, 0.0, 1.0))
                if xf1 <= xf0 or yf1 <= yf0:
                    return

                width = max(1e-6, xf1 - xf0)
                height = max(1e-6, yf1 - yf0)

                if scale_x:
                    if anchor_right and not anchor_left:
                        x2f = xf1
                        new_width = width * width_scale
                        x1f = x2f - new_width
                    elif anchor_left and not anchor_right:
                        x1f = xf0
                        new_width = width * width_scale
                        x2f = x1f + new_width
                    else:
                        cx = 0.5 * (xf0 + xf1)
                        half_w = 0.5 * width * width_scale
                        x1f = cx - half_w
                        x2f = cx + half_w
                else:
                    x1f, x2f = xf0, xf1

                if scale_y:
                    if anchor_top and not anchor_bottom:
                        y2f = yf1
                        new_height = height * width_scale
                        y1f = y2f - new_height
                    elif anchor_bottom and not anchor_top:
                        y1f = yf0
                        new_height = height * width_scale
                        y2f = y1f + new_height
                    else:
                        cy = 0.5 * (yf0 + yf1)
                        half_h = 0.5 * height * width_scale
                        y1f = cy - half_h
                        y2f = cy + half_h
                else:
                    y1f, y2f = yf0, yf1

                x1f = float(np.clip(x1f, 0.0, 1.0))
                x2f = float(np.clip(x2f, 0.0, 1.0))
                y1f = float(np.clip(y1f, 0.0, 1.0))
                y2f = float(np.clip(y2f, 0.0, 1.0))
                if x2f <= x1f or y2f <= y1f:
                    return

                x1 = int(math.floor(x1f * Nx))
                x2 = int(math.ceil(x2f * Nx))
                y1 = int(math.floor(y1f * Ny))
                y2 = int(math.ceil(y2f * Ny))
                x1 = max(1, min(x1, Nx - 1))
                x2 = max(x1 + 1, min(x2, Nx))
                y1 = max(1, min(y1, Ny - 1))
                y2 = max(y1 + 1, min(y2, Ny))
                V[y1:y2, x1:x2] = self.cfg.V_wall * height_scale
                patches.append(Rectangle((x1, y1), x2 - x1, y2 - y1))
                solids.append((x1, y1, x2, y2))

            add_rect(0.3259, 0.3708, 0.4914, 0.5057)  # shifted top spine
            add_rect(0.3883, 0.4098, 0.4671, 0.5300)  # shifted mid spine connector
            add_rect(0.4799, 0.5248, 0.4457, 0.5514)  # shifted base spine hub
            add_rect(0.5196, 0.5625, 0.0343, 0.3943)  # shifted lower block
            add_rect(0.5196, 0.5625, 0.6029, 0.9629)  # shifted upper block
            add_rect(0.6044, 0.6258, 0.0343, 0.9643)  # inner vertical bar
            add_rect(0.7096, 0.7544, 0.0343, 0.9643, anchor_right=True)  # fixed outer right wall

        elif kind == "no_obstacle":
            pass
        else:
            raise ValueError(f"Unknown map kind: {kind}")

        self.V_cpu = V.astype(np.float32, copy=True)
        absorber = self._build_absorber()
        self.W_absorb_cpu = absorber.astype(np.float32, copy=True)
        self.V = self.be.to_xp(self.V_cpu, np.float32)
        self.W_absorb = self.be.to_xp(self.W_absorb_cpu, np.float32)
        self.course_patches = patches
        self.solid_rects = solids

        # Hole mask (CPU + XP)
        self.hole_mask_cpu = self._hole_mask()
        self.hole_mask = self.be.to_xp(self.hole_mask_cpu, bool)

        # Combined operator used for dynamic dt scaling
        self.V_operator = ((-1j * self.V) - self.W_absorb).astype(np.complex64)
        self.V_operator_cpu = ((-1j * self.V_cpu) - self.W_absorb_cpu).astype(np.complex64)

        obstacle_mask = (self.V_cpu > (0.5 * max(1e-6, float(self.cfg.V_wall)))).astype(np.float32)
        self.obstacle_mask_cpu = obstacle_mask
        self.obstacle_mask = self.be.to_xp(obstacle_mask, np.float32)
        self.max_potential = float(np.max(np.abs(self.V_cpu))) if self.V_cpu.size else 0.0
        self.max_absorber = float(np.max(self.W_absorb_cpu)) if self.W_absorb_cpu.size else 0.0
        self._last_dt: float | None = None

    def update_exponents(self, dt: float, k2, dtype):
        if self._last_dt is not None and abs(self._last_dt - dt) <= 1e-12:
            return
        # expV_half = exp(((-iV) - W) * dt/2), expK = exp(-i k^2 dt / 2)
        xp = self.be.xp
        self.expV_half = xp.exp(((-1j * self.V) - self.W_absorb) * (dt * 0.5)).astype(dtype)
        self.expK = xp.exp((-1j * k2 * dt / 2.0).astype(dtype)).astype(dtype)
        self._last_dt = dt

    def _apply_single_wall(self, V: np.ndarray, cx: int, Ny: int):
        width_scale, height_scale = self._thickness_scales()
        base_width = max(1, self.cfg.single_wall_width)
        w = max(1, int(round(base_width * width_scale)))
        x1 = max(3, cx - w)
        x2 = min(self.Nx - 3, cx + w)
        if x2 <= x1:
            x1 = max(3, cx - 1)
            x2 = min(self.Nx - 3, cx + 1)
        y1 = int(Ny * 0.02)
        y2 = int(Ny * 0.98)

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
